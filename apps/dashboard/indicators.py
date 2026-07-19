"""Technical-analysis indicators (Jarvis markets suite).

Pure-Python, dependency-free. Every function takes a list of closing prices
(oldest → newest) and returns a series aligned to the input length, with ``None``
during the warm-up period so it lines up 1:1 with the price array for chart
overlays. ``read_signals`` turns the latest values into short, transparent,
NON-advisory read-outs (the UI labels them informational only).
"""

from __future__ import annotations


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average, aligned to input length."""
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= period:
            run -= values[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average (seeded with the SMA of the first period)."""
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    """Wilder's Relative Strength Index. 100 = only gains, 0 = only losses."""
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line (EMAfast − EMAslow), its signal EMA, and the histogram."""
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    line: list[float | None] = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(ema_fast, ema_slow)
    ]
    # signal EMA runs over the defined portion of the MACD line
    defined = [v for v in line if v is not None]
    sig_defined = ema(defined, signal)
    sig: list[float | None] = [None] * len(values)
    hist: list[float | None] = [None] * len(values)
    j = 0
    for i, v in enumerate(line):
        if v is None:
            continue
        s = sig_defined[j]
        sig[i] = s
        if s is not None:
            hist[i] = v - s
        j += 1
    return {"macd": line, "signal": sig, "hist": hist}


def bollinger(values: list[float], period: int = 20, mult: float = 2.0) -> dict:
    """Bollinger Bands: SMA mid with ±mult·σ envelope (population σ)."""
    mid = sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        if mid[i] is None:
            continue
        window = values[i - period + 1: i + 1]
        m = mid[i]
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        upper[i] = m + mult * sd
        lower[i] = m - mult * sd
    return {"mid": mid, "upper": upper, "lower": lower}


def _last(series: list[float | None]) -> float | None:
    for v in reversed(series):
        if v is not None:
            return v
    return None


def read_signals(values: list[float]) -> list[dict]:
    """Latest-value read-outs for the detail panel. Informational, not advice.

    Each entry is {label, value, tone} where tone ∈ up|down|neutral so the UI can
    color it. Kept deliberately transparent — plain thresholds, no black boxes.
    """
    signals: list[dict] = []
    if len(values) < 15:
        return signals
    price = values[-1]

    r = _last(rsi(values))
    if r is not None:
        tone = "down" if r >= 70 else "up" if r <= 30 else "neutral"
        note = "overbought" if r >= 70 else "oversold" if r <= 30 else "neutral"
        signals.append({"label": "RSI(14)", "value": f"{r:.0f} — {note}", "tone": tone})

    s50 = _last(sma(values, 50))
    if s50 is not None:
        above = price >= s50
        signals.append({"label": "vs SMA(50)", "tone": "up" if above else "down",
                        "value": f"{'above' if above else 'below'} ({s50:,.2f})"})

    m = macd(values)
    ml, sg = _last(m["macd"]), _last(m["signal"])
    if ml is not None and sg is not None:
        up = ml >= sg
        signals.append({"label": "MACD", "tone": "up" if up else "down",
                        "value": f"{'bullish' if up else 'bearish'} ({ml:.2f} vs {sg:.2f})"})

    hi = max(values)
    lo = min(values)
    signals.append({"label": "range", "tone": "neutral",
                    "value": f"{lo:,.2f} – {hi:,.2f}"})
    return signals
