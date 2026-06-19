"""Deterministic construction-plan engine for selected portfolio maps."""

from __future__ import annotations

import math

from .adapters import holding_index
from .schemas import (
    CandidatePool,
    ConstructionPlan,
    CurrentPortfolio,
    PortfolioMap,
    SimulatedOrder,
    StockTranche,
)
from .storage import new_id, utc_now


TRANCHE_WEIGHTS = (0.30, 0.30, 0.40)


def build_construction_plan(
    selected_map: PortfolioMap,
    candidate_pool: CandidatePool,
    portfolio: CurrentPortfolio,
) -> ConstructionPlan:
    """Build deterministic buy/sell tranches and simulated order parameters."""
    candidate_by_symbol = {candidate.symbol: candidate for candidate in candidate_pool.candidates}
    holdings_by_symbol = holding_index(portfolio)
    stock_tranches: list[StockTranche] = []
    simulated_orders: list[SimulatedOrder] = []
    cash_required = 0.0
    cash_released = 0.0
    warnings = list(portfolio.warnings)

    for target in selected_map.holdings:
        candidate = candidate_by_symbol.get(target.symbol)
        if candidate is None:
            warnings.append(f"Missing market data for {target.symbol}; skipped.")
            continue
        last_price = candidate.futu_data.quote.last_price if candidate.futu_data.quote else None
        if not last_price or last_price <= 0:
            warnings.append(f"Missing Futu last price for {target.symbol}; skipped.")
            continue
        holding = holdings_by_symbol.get(target.symbol)
        current_value = holding.market_value if holding else 0.0
        target_value = portfolio.total_assets * target.target_weight
        delta_value = target_value - current_value
        if abs(delta_value) < last_price:
            continue
        if delta_value > 0:
            tranches = _buy_tranches(
                symbol=target.symbol,
                price=last_price,
                desired_value=delta_value,
            )
            cash_required += sum(item.estimated_value for item in tranches)
        else:
            can_sell_qty = holding.can_sell_qty if holding else 0
            tranches = _sell_tranches(
                symbol=target.symbol,
                price=last_price,
                desired_value=abs(delta_value),
                can_sell_qty=can_sell_qty,
            )
            cash_released += sum(item.estimated_value for item in tranches)
        stock_tranches.extend(tranches)
        simulated_orders.extend(
            SimulatedOrder(
                code=tranche.symbol,
                side=tranche.side,
                quantity=tranche.quantity,
                price=tranche.limit_price,
            )
            for tranche in tranches
            if tranche.quantity > 0
        )

    post_trade_cash = portfolio.cash - cash_required + cash_released
    if post_trade_cash < portfolio.total_assets * selected_map.cash_weight:
        warnings.append(
            "Plan may breach the requested cash reserve; reduce selected map size or add budget."
        )

    return ConstructionPlan(
        plan_id=new_id("iap"),
        selected_map_id=selected_map.map_id,
        generated_at=utc_now(),
        cash_required=round(cash_required, 2),
        cash_released=round(cash_released, 2),
        post_trade_cash=round(post_trade_cash, 2),
        target_theme_weight=selected_map.sleeve_weight,
        stock_tranches=stock_tranches,
        simulated_orders=simulated_orders,
        invalidation=[
            "Rebuild the plan if Futu quote data, holdings, or user risk policy changes.",
            "Rebuild before placing any live order; this MVP only emits SIMULATE parameters.",
        ],
        warnings=warnings,
    )


def _buy_tranches(symbol: str, price: float, desired_value: float) -> list[StockTranche]:
    limit_prices = [price, price * 0.97, price * 0.94]
    triggers = [
        "near current price",
        "pullback toward first support placeholder",
        "pullback toward second support placeholder",
    ]
    return _split_tranches(
        symbol=symbol,
        side="BUY",
        desired_value=desired_value,
        limit_prices=limit_prices,
        triggers=triggers,
        invalidation="Stop buying if thesis or market data is stale.",
    )


def _sell_tranches(
    symbol: str,
    price: float,
    desired_value: float,
    can_sell_qty: int,
) -> list[StockTranche]:
    target_qty = min(can_sell_qty, math.floor(desired_value / price))
    if target_qty <= 0:
        return []
    limit_prices = [price, price * 1.03, price * 1.06]
    triggers = [
        "trim immediately",
        "trim into first resistance placeholder",
        "trim into second resistance placeholder",
    ]
    desired = target_qty * price
    return _split_tranches(
        symbol=symbol,
        side="SELL",
        desired_value=desired,
        limit_prices=limit_prices,
        triggers=triggers,
        invalidation="Cancel trim if target weight is revised upward.",
    )


def _split_tranches(
    symbol: str,
    side: str,
    desired_value: float,
    limit_prices: list[float],
    triggers: list[str],
    invalidation: str,
) -> list[StockTranche]:
    tranches: list[StockTranche] = []
    for weight, limit_price, trigger in zip(TRANCHE_WEIGHTS, limit_prices, triggers):
        quantity = math.floor((desired_value * weight) / limit_price)
        if quantity <= 0:
            continue
        value = quantity * limit_price
        tranches.append(
            StockTranche(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=round(limit_price, 2),
                trigger=trigger,
                invalidation=invalidation,
                estimated_value=round(value, 2),
            )
        )
    return tranches
