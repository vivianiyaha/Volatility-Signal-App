"""
==============================================================================
 DERIV VOLATILITY INDEX — REAL-TIME SMC + EXTREME-OSCILLATOR SIGNAL DASHBOARD
==============================================================================

Strategy summary
-----------------
1. Trend Filter      : EMA(200) on Close.
2. Trigger Indicator : RSI(1) — an unsmoothed, single-bar oscillator that
                        snaps toward 0/100 on every strong impulse candle.
                        Extreme zones: <= 8 (oversold) / >= 92 (overbought),
                        baseline reference at 50.
3. SMC Filter        : Swing-based market structure (BOS / CHoCH) plus the
                        most recent bullish/bearish Order Block, used to
                        confirm (or veto) raw oscillator extremes.
4. Confidence Engine : Weighted score across the three confluences above.
                        A "perfect" stack (extreme touch + fresh BOS/CHoCH in
                        the trade direction + trend alignment + OB tag)
                        prints a 90%+ High-Confidence badge.
5. Risk Engine       : SL beyond the triggering swing/OB boundary (ATR
                        buffered), TP at a minimum 1:2 R:R or the next
                        liquidity pool, whichever is further.

Data
----
Live candles are pulled from Deriv's public WebSocket API (no auth/app
registration required for `ticks_history`). If the socket can't be reached
(sandboxed environment, firewall, offline demo, etc.) the app transparently
falls back to a synthetic-but-statistically-similar volatility-index feed so
the dashboard is always fully demoable.

Run
---
    pip install -r requirements.txt
    streamlit run app.py
==============================================================================
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from websocket import create_connection  # websocket-client package
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


# ==============================================================================
# 1. CONFIG & CONSTANTS
# ==============================================================================

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"

SYMBOLS = {
    "Volatility 10 Index": "R_10",
    "Volatility 25 Index": "R_25",
    "Volatility 50 Index": "R_50",
    "Volatility 75 Index": "R_75",
    "Volatility 100 Index": "R_100",
    "Volatility 25 (1s) Index": "1HZ25V",
    "Volatility 50 (1s) Index": "1HZ50V",
    "Volatility 75 (1s) Index": "1HZ75V",
    "Volatility 100 (1s) Index": "1HZ100V",
}

TIMEFRAMES = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

EMA_PERIOD = 200
RSI_PERIOD = 1
RSI_OVERSOLD = 8
RSI_OVERBOUGHT = 92
RSI_BASELINE = 50
SWING_LOOKBACK = 5          # bars each side for fractal swing detection
MIN_RR = 2.0                # minimum reward:risk
ATR_PERIOD = 14
ATR_SL_BUFFER = 0.25        # extra ATR fraction added beyond swing/OB for SL
HIGH_CONFIDENCE_THRESHOLD = 90
CANDLE_COUNT = 500

DARK_BG = "#0e1117"
PANEL_BG = "#161b22"
GOLD = "#d4af37"
GREEN = "#26a69a"
RED = "#ef5350"
GREY = "#8b949e"


# ==============================================================================
# 2. PAGE CONFIG & THEME
# ==============================================================================

st.set_page_config(
    page_title="Deriv SMC Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
    div[data-testid="stMetric"] {{
        background-color: {PANEL_BG};
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 12px 14px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {GREY}; }}
    .badge-high {{
        background-color: {GOLD}; color: #111; font-weight: 700;
        padding: 3px 10px; border-radius: 12px; font-size: 0.8rem;
    }}
    .badge-med {{
        background-color: #3d4451; color: {GREY}; font-weight: 600;
        padding: 3px 10px; border-radius: 12px; font-size: 0.8rem;
    }}
    .stDataFrame {{ border: 1px solid #2a2f3a; }}
    h1, h2, h3 {{ color: #e6e6e6; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 3. DATA LAYER
# ==============================================================================

def fetch_deriv_candles(symbol: str, granularity: int, count: int = CANDLE_COUNT) -> Optional[pd.DataFrame]:
    """Pull OHLC candles from Deriv's public WebSocket API.

    Returns None on any failure so the caller can fall back to synthetic data
    instead of crashing the dashboard.
    """
    if not WEBSOCKET_AVAILABLE:
        return None
    try:
        ws = create_connection(DERIV_WS_URL, timeout=6)
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
        }
        ws.send(json.dumps(request))
        response = json.loads(ws.recv())
        ws.close()

        if "error" in response:
            return None

        candles = response.get("candles", [])
        if not candles:
            return None

        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["epoch"], unit="s")
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
        df = df[["time", "open", "high", "low", "close"]].astype(
            {"open": float, "high": float, "low": float, "close": float}
        )
        return df.reset_index(drop=True)
    except Exception:
        return None


def generate_synthetic_data(symbol: str, granularity: int, count: int = CANDLE_COUNT) -> pd.DataFrame:
    """Statistically-similar fallback feed.

    Deriv volatility indices are synthetic instruments themselves (fixed
    annualised volatility, no real-world trend bias), so a seeded Geometric
    Brownian Motion walk is a reasonable stand-in when the live socket is
    unreachable (e.g. sandboxed/offline environments).
    """
    vol_map = {"R_10": 0.10, "R_25": 0.25, "R_50": 0.50, "R_75": 0.75, "R_100": 1.00,
               "1HZ25V": 0.25, "1HZ50V": 0.50, "1HZ75V": 0.75, "1HZ100V": 1.00}
    sigma = vol_map.get(symbol, 0.5)
    seed = abs(hash((symbol, granularity))) % (2 ** 32)
    rng = np.random.default_rng(seed)

    base_price = 5000.0
    dt = granularity / 86400  # fraction of a day per bar
    returns = rng.normal(loc=0.0, scale=sigma * np.sqrt(dt) * 0.02, size=count)
    close = base_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.0015, count)))
    low = close * (1 - np.abs(rng.normal(0, 0.0015, count)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    now = pd.Timestamp.utcnow().tz_localize(None)
    times = pd.date_range(end=now, periods=count, freq=pd.Timedelta(seconds=granularity))

    df = pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close})
    return df


@st.cache_data(ttl=15, show_spinner=False)
def get_market_data(symbol: str, granularity: int) -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_live) with a short TTL cache to respect rate limits."""
    df = fetch_deriv_candles(symbol, granularity)
    if df is not None and len(df) > EMA_PERIOD:
        return df, True
    return generate_synthetic_data(symbol, granularity), False


# ==============================================================================
# 4. INDICATOR LAYER
# ==============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA(200), RSI(1), and ATR(14) to the OHLC dataframe."""
    df = df.copy()
    df["ema200"] = ta.ema(df["close"], length=EMA_PERIOD)
    df["rsi1"] = ta.rsi(df["close"], length=RSI_PERIOD)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["trend"] = np.where(df["close"] > df["ema200"], "bullish", "bearish")
    return df


# ==============================================================================
# 5. SMART MONEY CONCEPTS (SMC) LAYER
# ==============================================================================

def detect_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """Fractal-style swing high/low detection: a bar is a swing high/low if
    its high/low is the extreme within a symmetric window around it."""
    df = df.copy()
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            is_swing_high[i] = True
        if lows[i] == window_l.min():
            is_swing_low[i] = True

    df["swing_high"] = is_swing_high
    df["swing_low"] = is_swing_low
    return df


def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH).

    Logic:
      - Track the most recent confirmed swing high / swing low.
      - A close beyond the last swing high while the prevailing internal
        trend is bullish (or was bearish) is a bullish BOS (or CHoCH).
      - Symmetric logic for bearish structure breaks.
    """
    df = df.copy()
    n = len(df)
    event = [None] * n           # 'BOS_BULL' / 'CHOCH_BULL' / 'BOS_BEAR' / 'CHOCH_BEAR'
    last_swing_high = np.nan
    last_swing_low = np.nan
    internal_trend = None        # 'bullish' | 'bearish'

    for i in range(n):
        close = df["close"].iat[i]

        # Bullish break
        if not np.isnan(last_swing_high) and close > last_swing_high:
            if internal_trend == "bearish":
                event[i] = "CHOCH_BULL"
            else:
                event[i] = "BOS_BULL"
            internal_trend = "bullish"
            last_swing_high = np.nan  # consumed; wait for next swing

        # Bearish break
        elif not np.isnan(last_swing_low) and close < last_swing_low:
            if internal_trend == "bullish":
                event[i] = "CHOCH_BEAR"
            else:
                event[i] = "BOS_BEAR"
            internal_trend = "bearish"
            last_swing_low = np.nan

        if df["swing_high"].iat[i]:
            last_swing_high = df["high"].iat[i]
        if df["swing_low"].iat[i]:
            last_swing_low = df["low"].iat[i]

    df["structure_event"] = event
    return df


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Tag the last opposite-colour candle preceding a structure break as an
    Order Block (a classic SMC definition):
      - Bullish OB: the last bearish (red) candle before a BOS_BULL/CHOCH_BULL.
      - Bearish OB: the last bullish (green) candle before a BOS_BEAR/CHOCH_BEAR.
    """
    df = df.copy()
    n = len(df)
    ob_type = [None] * n
    ob_top = [np.nan] * n
    ob_bottom = [np.nan] * n

    for i in range(n):
        ev = df["structure_event"].iat[i]
        if ev in ("BOS_BULL", "CHOCH_BULL"):
            for j in range(i - 1, max(i - 30, -1), -1):
                if df["close"].iat[j] < df["open"].iat[j]:  # bearish candle
                    ob_type[i] = "bullish_ob"
                    ob_top[i] = df["open"].iat[j]
                    ob_bottom[i] = df["low"].iat[j]
                    break
        elif ev in ("BOS_BEAR", "CHOCH_BEAR"):
            for j in range(i - 1, max(i - 30, -1), -1):
                if df["close"].iat[j] > df["open"].iat[j]:  # bullish candle
                    ob_type[i] = "bearish_ob"
                    ob_top[i] = df["high"].iat[j]
                    ob_bottom[i] = df["open"].iat[j]
                    break

    df["ob_type"] = ob_type
    df["ob_top"] = ob_top
    df["ob_bottom"] = ob_bottom
    return df


def run_smc_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = detect_swings(df)
    df = detect_structure(df)
    df = detect_order_blocks(df)
    return df


# ==============================================================================
# 6. SIGNAL & CONFIDENCE ENGINE
# ==============================================================================

@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    direction: str          # BUY / SELL
    entry: float
    stop_loss: float
    take_profit: float
    confidence: int
    reasons: list = field(default_factory=list)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= HIGH_CONFIDENCE_THRESHOLD

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return round(reward / risk, 2) if risk else 0.0


def _recent_structure_event(df: pd.DataFrame, direction: str, lookback: int = 6) -> Optional[dict]:
    """Look back a few bars for a structure break aligned with `direction`."""
    tail = df.iloc[-lookback:]
    wanted = {"BUY": ("BOS_BULL", "CHOCH_BULL"), "SELL": ("BOS_BEAR", "CHOCH_BEAR")}[direction]
    hits = tail[tail["structure_event"].isin(wanted)]
    if hits.empty:
        return None
    row = hits.iloc[-1]
    return {
        "event": row["structure_event"],
        "ob_top": row["ob_top"],
        "ob_bottom": row["ob_bottom"],
        "ob_type": row["ob_type"],
    }


def compute_confidence(rsi_extreme: bool, structure_hit: Optional[dict], trend_aligned: bool,
                        ob_present: bool) -> tuple[int, list]:
    """Weighted confluence score, 0-100.

    Weighting rationale:
      40 pts — the core trigger (oscillator extreme touch)
      30 pts — SMC structure break in the same direction
      20 pts — 200 EMA trend alignment
      10 pts — a tagged order block backing the structure break
    A perfect stack (all four) = 100; the spec's "perfect" 90% case is met
    once the extreme touch + structure break + trend align (40+30+20=90),
    with the OB confluence as the final 10-point cherry on top.
    """
    score = 0
    reasons = []
    if rsi_extreme:
        score += 40
        reasons.append("Oscillator extreme touch (RSI-1)")
    if structure_hit:
        score += 30
        reasons.append(f"SMC structure break: {structure_hit['event']}")
    if trend_aligned:
        score += 20
        reasons.append("200 EMA trend alignment")
    if ob_present:
        score += 10
        reasons.append("Order Block confluence")
    return score, reasons


def calculate_sl_tp(direction: str, entry: float, atr: float, structure_hit: Optional[dict],
                     recent_swing_low: float, recent_swing_high: float) -> tuple[float, float]:
    """Risk engine: SL beyond the OB/swing boundary (+ATR buffer), TP at the
    better of a 1:2 R:R or the next liquidity pool (opposite recent swing)."""
    buffer = atr * ATR_SL_BUFFER if not np.isnan(atr) else entry * 0.001

    if direction == "BUY":
        if structure_hit and not np.isnan(structure_hit.get("ob_bottom", np.nan)):
            sl = structure_hit["ob_bottom"] - buffer
        else:
            sl = recent_swing_low - buffer
        risk = entry - sl
        tp_min_rr = entry + MIN_RR * risk
        liquidity_target = recent_swing_high if recent_swing_high > entry else tp_min_rr
        tp = max(tp_min_rr, liquidity_target)
    else:  # SELL
        if structure_hit and not np.isnan(structure_hit.get("ob_top", np.nan)):
            sl = structure_hit["ob_top"] + buffer
        else:
            sl = recent_swing_high + buffer
        risk = sl - entry
        tp_min_rr = entry - MIN_RR * risk
        liquidity_target = recent_swing_low if recent_swing_low < entry else tp_min_rr
        tp = min(tp_min_rr, liquidity_target)

    return round(sl, 4), round(tp, 4)


def generate_signal(df: pd.DataFrame, symbol_label: str) -> Optional[Signal]:
    """Evaluate the latest confirmed bar for a BUY/SELL setup."""
    if len(df) < EMA_PERIOD + SWING_LOOKBACK + 5:
        return None

    latest = df.iloc[-1]
    rsi = latest["rsi1"]
    trend = latest["trend"]

    recent_swings_high = df[df["swing_high"]]["high"]
    recent_swings_low = df[df["swing_low"]]["low"]
    recent_swing_high = recent_swings_high.iloc[-1] if not recent_swings_high.empty else latest["high"]
    recent_swing_low = recent_swings_low.iloc[-1] if not recent_swings_low.empty else latest["low"]

    direction = None
    if rsi is not None and rsi <= RSI_OVERSOLD:
        direction = "BUY"
    elif rsi is not None and rsi >= RSI_OVERBOUGHT:
        direction = "SELL"
    if direction is None:
        return None

    trend_aligned = (direction == "BUY" and trend == "bullish") or (direction == "SELL" and trend == "bearish")
    structure_hit = _recent_structure_event(df, direction)
    ob_present = bool(structure_hit and structure_hit.get("ob_type"))

    # Strict rule: BUY/SELL only fires when the SMC structure aligns (per spec).
    if structure_hit is None:
        return None

    confidence, reasons = compute_confidence(
        rsi_extreme=True, structure_hit=structure_hit, trend_aligned=trend_aligned, ob_present=ob_present
    )

    entry = float(latest["close"])
    sl, tp = calculate_sl_tp(direction, entry, latest["atr"], structure_hit, recent_swing_low, recent_swing_high)

    return Signal(
        timestamp=latest["time"],
        symbol=symbol_label,
        direction=direction,
        entry=round(entry, 4),
        stop_loss=sl,
        take_profit=tp,
        confidence=confidence,
        reasons=reasons,
    )


# ==============================================================================
# 7. CHARTING
# ==============================================================================

def build_chart(df: pd.DataFrame, signal: Optional[Signal]) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.04, subplot_titles=("Price Action / SMC / EMA(200)", "RSI(1) Oscillator"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color=GREEN, decreasing_line_color=RED, name="Price",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["ema200"], line=dict(color=GOLD, width=1.6), name="EMA 200"),
        row=1, col=1,
    )

    # Order block shading (most recent few)
    ob_rows = df[df["ob_type"].notna()].tail(6)
    for _, r in ob_rows.iterrows():
        color = "rgba(38,166,154,0.18)" if r["ob_type"] == "bullish_ob" else "rgba(239,83,80,0.18)"
        fig.add_shape(
            type="rect", x0=r["time"], x1=df["time"].iloc[-1],
            y0=r["ob_bottom"], y1=r["ob_top"], fillcolor=color, line_width=0, row=1, col=1,
        )

    # BOS / CHoCH markers
    struct_rows = df[df["structure_event"].notna()].tail(20)
    for _, r in struct_rows.iterrows():
        is_bull = "BULL" in r["structure_event"]
        fig.add_annotation(
            x=r["time"], y=r["high"] if is_bull else r["low"],
            text=r["structure_event"].replace("_", " "),
            showarrow=True, arrowhead=1, arrowsize=0.6,
            font=dict(size=9, color=GREEN if is_bull else RED),
            arrowcolor=GREEN if is_bull else RED,
            ax=0, ay=-25 if is_bull else 25, row=1, col=1,
        )

    # Signal marker
    if signal is not None:
        marker_color = GREEN if signal.direction == "BUY" else RED
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp], y=[signal.entry], mode="markers",
                marker=dict(size=14, color=marker_color, symbol="triangle-up" if signal.direction == "BUY" else "triangle-down"),
                name=f"{signal.direction} Signal",
            ),
            row=1, col=1,
        )
        for label, level, dash in (("SL", signal.stop_loss, "dot"), ("TP", signal.take_profit, "dash")):
            fig.add_hline(y=level, line_dash=dash, line_color=RED if label == "SL" else GREEN,
                          annotation_text=label, row=1, col=1)

    # Oscillator subplot
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi1"], line=dict(color="#7c8cf8", width=1.4), name="RSI(1)"), row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_color=RED, line_dash="dash", row=2, col=1,
                  annotation_text=f"Overbought {RSI_OVERBOUGHT}")
    fig.add_hline(y=RSI_OVERSOLD, line_color=GREEN, line_dash="dash", row=2, col=1,
                  annotation_text=f"Oversold {RSI_OVERSOLD}")
    fig.add_hline(y=RSI_BASELINE, line_color=GREY, line_dash="dot", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        height=720, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ==============================================================================
# 8. SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.title("⚙️ Dashboard Controls")

symbol_label = st.sidebar.selectbox("Volatility Index", list(SYMBOLS.keys()), index=3)
symbol_code = SYMBOLS[symbol_label]

timeframe_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=4)  # default H1
granularity = TIMEFRAMES[timeframe_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Parameters")
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=10.0, value=1000.0, step=50.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25)
min_confidence_filter = st.sidebar.slider("Minimum Confidence to Log (%)", 50, 100, 70, 5)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("Auto-refresh (every 15s)", value=False)
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: Deriv WebSocket API (`ticks_history`) when reachable, "
    "otherwise a seeded synthetic fallback feed for uninterrupted demoing."
)


# ==============================================================================
# 9. MAIN APP LOGIC
# ==============================================================================

if "signal_log" not in st.session_state:
    st.session_state.signal_log = []  # list[Signal]

st.title("📊 Deriv Volatility Index — SMC Signal Dashboard")
st.caption("EMA(200) trend filter · RSI(1) extreme trigger (8 / 92) · Smart Money structure confirmation")

raw_df, is_live = get_market_data(symbol_code, granularity)
df = add_indicators(raw_df)
df = run_smc_pipeline(df)

signal = generate_signal(df, symbol_label)

# Only append a *new* signal to the log (avoid duplicate spam on rerun)
if signal and signal.confidence >= min_confidence_filter:
    already_logged = any(
        s.timestamp == signal.timestamp and s.symbol == signal.symbol for s in st.session_state.signal_log
    )
    if not already_logged:
        st.session_state.signal_log.insert(0, signal)
        st.session_state.signal_log = st.session_state.signal_log[:100]

latest = df.iloc[-1]

# ---- Status / metric row ----------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Live Price", f"{latest['close']:.4f}",
          delta=f"{(latest['close'] - df['close'].iloc[-2]):.4f}")
c2.metric("Trend (EMA 200)", "Bullish 📈" if latest["trend"] == "bullish" else "Bearish 📉")
c3.metric("RSI(1)", f"{latest['rsi1']:.1f}" if pd.notna(latest["rsi1"]) else "—")
c4.metric("Data Feed", "🟢 Live (Deriv)" if is_live else "🟡 Synthetic (fallback)")
if signal:
    badge = "HIGH CONFIDENCE" if signal.is_high_confidence else "MODERATE"
    c5.metric("Active Signal", f"{signal.direction}", delta=f"{signal.confidence}% · {badge}")
else:
    c5.metric("Active Signal", "None", delta="Waiting for confluence")

st.markdown("---")

# ---- Signal detail card ------------------------------------------------------
if signal:
    badge_class = "badge-high" if signal.is_high_confidence else "badge-med"
    dir_color = GREEN if signal.direction == "BUY" else RED
    risk_amount = account_balance * (risk_pct / 100)
    st.markdown(
        f"""
        <div style="background-color:{PANEL_BG}; border:1px solid #2a2f3a; border-radius:12px; padding:16px 20px;">
          <span style="font-size:1.3rem; font-weight:700; color:{dir_color};">{signal.direction} SIGNAL</span>
          &nbsp;&nbsp;<span class="{badge_class}">{signal.confidence}% confidence</span>
          <div style="color:{GREY}; margin-top:6px;">{symbol_label} · {timeframe_label} · {signal.timestamp}</div>
          <div style="margin-top:10px; display:flex; gap:32px;">
            <div><b>Entry</b><br>{signal.entry}</div>
            <div><b>Stop Loss</b><br>{signal.stop_loss}</div>
            <div><b>Take Profit</b><br>{signal.take_profit}</div>
            <div><b>R : R</b><br>1 : {signal.risk_reward}</div>
            <div><b>Suggested Risk</b><br>${risk_amount:.2f}</div>
          </div>
          <div style="margin-top:10px; color:{GREY}; font-size:0.85rem;">
            Confluence: {" · ".join(signal.reasons)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No qualifying BUY/SELL confluence on the latest confirmed bar. The dashboard is monitoring "
            "for an RSI(1) extreme touch aligned with a fresh SMC structure break.")

st.markdown("###")

# ---- Chart --------------------------------------------------------------------
st.plotly_chart(build_chart(df.tail(200), signal), use_container_width=True)

# ---- Signal log table -----------------------------------------------------
st.subheader("📋 Signal Log")
if st.session_state.signal_log:
    log_df = pd.DataFrame([
        {
            "Timestamp": s.timestamp,
            "Asset": s.symbol,
            "Signal": s.direction,
            "Entry": s.entry,
            "SL": s.stop_loss,
            "TP": s.take_profit,
            "R:R": f"1:{s.risk_reward}",
            "Confidence": f"{s.confidence}%" + (" 🏆" if s.is_high_confidence else ""),
        }
        for s in st.session_state.signal_log
    ])

    def _highlight_signal(row):
        color = "color: #26a69a" if row["Signal"] == "BUY" else "color: #ef5350"
        return [color] * len(row)

    st.dataframe(
        log_df.style.apply(_highlight_signal, axis=1),
        use_container_width=True, hide_index=True,
    )
    if st.button("Clear Log"):
        st.session_state.signal_log = []
        st.rerun()
else:
    st.caption("No signals logged yet this session.")

# ---- Auto-refresh loop ------------------------------------------------------
if auto_refresh:
    time.sleep(15)
    st.rerun()
