"""NanoFenix-inspired size multiplier — reduces size only, never increases."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from hermes_trader.market_state import MarketState


class _IntentLike(Protocol):
    confidence: float
    pool_liquidity_usd: Optional[float]


def compute_size_multiplier(
    intent: _IntentLike,
    *,
    market_state: Optional[MarketState] = None,
    ohlcv_features: Optional[dict[str, Any]] = None,
) -> float:
    """Return multiplier in [0.5, 1.0] based on confidence and liquidity context."""
    multiplier = 1.0

    confidence = float(intent.confidence or 0.0)
    if confidence < 0.65:
        multiplier -= 0.15
    elif confidence < 0.75:
        multiplier -= 0.05

    liquidity = intent.pool_liquidity_usd
    if liquidity is not None:
        if liquidity < 100_000:
            multiplier -= 0.2
        elif liquidity < 200_000:
            multiplier -= 0.1

    if ohlcv_features:
        volatility = float(ohlcv_features.get("volatility_24h_pct") or 0.0)
        if volatility > 40:
            multiplier -= 0.15
        elif volatility > 25:
            multiplier -= 0.05

    if market_state and len(market_state.new_pools) > 5:
        multiplier -= 0.05

    return max(0.5, min(1.0, multiplier))


def apply_size_multiplier(size_usd: float, multiplier: float) -> float:
    return max(0.0, size_usd * max(0.5, min(1.0, multiplier)))