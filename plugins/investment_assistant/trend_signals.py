"""Technical signal extraction for portfolio monitoring."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .monitoring_models import MarketPriceData, PriceBar, TrendSignal


def compute_trend_signals(
    market_data: dict[str, MarketPriceData] | Iterable[MarketPriceData],
) -> dict[str, TrendSignal]:
    """Compute compact deterministic trend artifacts from price data."""

    if isinstance(market_data, dict):
        items = market_data.values()
    else:
        items = market_data
    signals: dict[str, TrendSignal] = {}
    for item in items:
        signal = compute_trend_signal(item)
        signals[signal.symbol] = signal
    return signals


def compute_trend_signal(data: MarketPriceData) -> TrendSignal:
    closes = [bar.close for bar in data.kline if bar.close is not None and bar.close > 0]
    highs = [bar.high for bar in data.kline if bar.high is not None and bar.high > 0]
    lows = [bar.low for bar in data.kline if bar.low is not None and bar.low > 0]
    price = data.last_price or (closes[-1] if closes else None)
    warnings = list(data.warnings)
    if price is None or price <= 0:
        warnings.append("Missing last price; trend state is unknown.")
        return TrendSignal(symbol=data.symbol.upper(), warnings=warnings)
    if len(closes) < 20:
        warnings.append("Fewer than 20 daily bars; trend state is unknown.")
        return TrendSignal(
            symbol=data.symbol.upper(),
            price=round(price, 4),
            trend_state="unknown",
            data_asof=data.update_time,
            warnings=warnings,
        )

    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)
    ma200 = _sma(closes, 200)
    rsi14 = _rsi(closes, 14)
    atr14 = _atr(data.kline, 14)
    return_20d = _return_over(closes, 20)
    return_60d = _return_over(closes, 60)
    support_levels, resistance_levels = _support_resistance(data.kline, price)
    trend_state = _trend_state(price, ma20, ma50, ma200, rsi14, atr14)
    return TrendSignal(
        symbol=data.symbol.upper(),
        price=round(price, 4),
        trend_state=trend_state,
        ma20=_round_optional(ma20),
        ma50=_round_optional(ma50),
        ma200=_round_optional(ma200),
        rsi14=_round_optional(rsi14),
        atr14=_round_optional(atr14),
        return_20d=_round_optional(return_20d),
        return_60d=_round_optional(return_60d),
        support_levels=[round(value, 4) for value in support_levels],
        resistance_levels=[round(value, 4) for value in resistance_levels],
        data_asof=data.update_time,
        warnings=warnings,
    )


def parse_market_data(payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, MarketPriceData]:
    """Parse CLI/test market-data JSON into MarketPriceData objects."""

    rows: Iterable[dict[str, Any]]
    if isinstance(payload, dict) and "symbols" in payload:
        rows = payload.get("symbols") or []
    elif isinstance(payload, dict):
        rows = [
            {"symbol": symbol, **value}
            for symbol, value in payload.items()
            if isinstance(value, dict)
        ]
    else:
        rows = payload
    result: dict[str, MarketPriceData] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("symbol"):
            continue
        symbol = str(row["symbol"]).upper()
        kline = [_parse_bar(item) for item in row.get("kline", []) if isinstance(item, dict)]
        result[symbol] = MarketPriceData(
            symbol=symbol,
            last_price=_safe_float(row.get("last_price") or row.get("price")),
            update_time=str(row.get("update_time") or row.get("data_asof") or ""),
            kline=kline,
            warnings=[str(value) for value in row.get("warnings", [])],
        )
    return result


def _parse_bar(row: dict[str, Any]) -> PriceBar:
    return PriceBar(
        date=str(row.get("date") or row.get("time_key") or row.get("time") or ""),
        open=_safe_float(row.get("open") or row.get("open_price")),
        high=_safe_float(row.get("high") or row.get("high_price")),
        low=_safe_float(row.get("low") or row.get("low_price")),
        close=_safe_float(row.get("close") or row.get("close_price") or row.get("last_price")),
        volume=_safe_float(row.get("volume")),
    )


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _return_over(values: list[float], period: int) -> float | None:
    if len(values) <= period or values[-period - 1] <= 0:
        return None
    return values[-1] / values[-period - 1] - 1


def _rsi(values: list[float], period: int) -> float | None:
    if len(values) <= period:
        return None
    gains = []
    losses = []
    for prev, curr in zip(values[-period - 1 : -1], values[-period:]):
        delta = curr - prev
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(bars: list[PriceBar], period: int) -> float | None:
    if len(bars) <= period:
        return None
    ranges: list[float] = []
    previous_close = bars[-period - 1].close
    for bar in bars[-period:]:
        if bar.high is None or bar.low is None:
            continue
        values = [bar.high - bar.low]
        if previous_close is not None:
            values.append(abs(bar.high - previous_close))
            values.append(abs(bar.low - previous_close))
        ranges.append(max(values))
        previous_close = bar.close
    if len(ranges) < period:
        return None
    return sum(ranges) / period


def _support_resistance(bars: list[PriceBar], price: float) -> tuple[list[float], list[float]]:
    lows = [bar.low for bar in bars[-80:] if bar.low is not None and bar.low > 0]
    highs = [bar.high for bar in bars[-80:] if bar.high is not None and bar.high > 0]
    supports = sorted({value for value in lows if value < price}, reverse=True)[:3]
    resistances = sorted({value for value in highs if value > price})[:3]
    return supports, resistances


def _trend_state(
    price: float,
    ma20: float | None,
    ma50: float | None,
    ma200: float | None,
    rsi14: float | None,
    atr14: float | None,
) -> str:
    if ma20 is None:
        return "unknown"
    atr_ratio = atr14 / price if atr14 and price > 0 else 0
    if atr_ratio > 0.08:
        return "high_volatility"
    if rsi14 is not None and rsi14 >= 75 and price > ma20:
        return "extended_uptrend"
    if ma50 is not None and price > ma20 > ma50 and (ma200 is None or ma50 > ma200):
        return "uptrend"
    if ma50 is not None and price < ma20 < ma50:
        return "downtrend"
    if ma50 is not None and price < ma20 and ma20 >= ma50:
        return "weakening"
    return "neutral"


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
