from __future__ import annotations

import math

from .models import MarketMetadata, OrderBookQuote, TradeDecision


def round_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    decimals = max(0, int(round(-math.log10(tick_size)))) if tick_size < 1 else 0
    return round(round(price / tick_size) * tick_size, decimals)


def expected_fee_per_share(price: float, fee_rate_bps: int, fee_exponent: int) -> float:
    fee_base = min(price, 1.0 - price) ** fee_exponent
    return fee_base * (fee_rate_bps / 10_000.0)


def evaluate_buy(
    strategy: str,
    market: MarketMetadata,
    quote: OrderBookQuote,
    *,
    model_probability: float,
    collateral_size: float,
    min_edge: float,
) -> TradeDecision:
    if quote.ask is None:
        return TradeDecision(strategy, "SKIP", market.token_id, "BUY", None, 0.0, None, "no executable ask")
    price = round_to_tick(float(quote.ask), market.tick_size)
    fee_per_share = expected_fee_per_share(price, market.fee_rate_bps, market.fee_exponent)
    edge_after_fees = model_probability - price - fee_per_share
    if edge_after_fees < min_edge:
        return TradeDecision(
            strategy=strategy,
            action="SKIP",
            token_id=market.token_id,
            side="BUY",
            price=price,
            collateral_size=0.0,
            edge_after_fees=edge_after_fees,
            reason=f"edge after fee {edge_after_fees:.4f} below MIN_EDGE {min_edge:.4f}",
            metadata={"fee_per_share": fee_per_share, "model_probability": model_probability},
        )
    return TradeDecision(
        strategy=strategy,
        action="BUY",
        token_id=market.token_id,
        side="BUY",
        price=price,
        collateral_size=collateral_size,
        edge_after_fees=edge_after_fees,
        reason=f"model fair {model_probability:.4f} clears price {price:.4f} and fee {fee_per_share:.4f}",
        metadata={"fee_per_share": fee_per_share, "model_probability": model_probability},
    )
