"""Build deterministic rebalance actions from drift and trend artifacts."""

from __future__ import annotations

import math

from .monitoring_models import (
    DriftPosition,
    MonitoringPreferences,
    RebalanceAction,
    RebalancePlan,
    RebalanceTranche,
    TrendSignal,
)
from .schemas import SimulatedOrder


TRANCHE_WEIGHTS = (0.30, 0.30, 0.40)


def build_rebalance_plan(
    *,
    drift_report,
    trend_signals: dict[str, TrendSignal],
    preferences: MonitoringPreferences | None = None,
) -> RebalancePlan:
    """Create add/trim/hold/watch actions from drift and trend signals."""

    prefs = preferences or MonitoringPreferences()
    warnings = list(drift_report.warnings)
    actions: list[RebalanceAction] = []
    simulated_orders: list[SimulatedOrder] = []
    cash_required = 0.0
    cash_released = 0.0

    for drift in drift_report.positions:
        if drift.status == "overweight" or (drift.status == "extra" and prefs.trim_extra_positions):
            action = _build_trim_action(drift, trend_signals.get(drift.symbol), prefs, drift_report.total_assets)
            actions.append(action)
            if action.simulated_order:
                simulated_orders.append(action.simulated_order)
                cash_released += action.target_trade_value
        elif drift.status == "extra":
            actions.append(_build_watch_action(drift, trend_signals.get(drift.symbol), "extra_position"))

    reserve_cash = drift_report.total_assets * drift_report.target_cash_weight
    available_cash = max(0.0, drift_report.cash + cash_released - reserve_cash)
    available_cash *= prefs.max_add_cash_fraction

    for drift in drift_report.positions:
        if drift.status == "underweight":
            action, consumed_cash = _build_add_action(
                drift,
                trend_signals.get(drift.symbol),
                prefs,
                available_cash,
            )
            available_cash -= consumed_cash
            cash_required += consumed_cash
            actions.append(action)
            if action.simulated_order:
                simulated_orders.append(action.simulated_order)
        elif drift.status == "within_band":
            actions.append(_build_hold_or_watch_action(drift, trend_signals.get(drift.symbol), prefs))
        elif drift.status == "missing_price":
            actions.append(_build_watch_action(drift, trend_signals.get(drift.symbol), "missing_price"))

    post_trade_cash = drift_report.cash - cash_required + cash_released
    if post_trade_cash < reserve_cash:
        warnings.append("Rebalance plan would leave cash below target reserve; reduce add actions.")
    return RebalancePlan(
        selected_map_id=drift_report.selected_map_id,
        target_cash_weight=drift_report.target_cash_weight,
        current_cash_weight=drift_report.cash_weight,
        cash_required=round(cash_required, 2),
        cash_released=round(cash_released, 2),
        post_trade_cash=round(post_trade_cash, 2),
        actions=actions,
        simulated_orders=simulated_orders,
        warnings=warnings,
    )


def _build_add_action(
    drift: DriftPosition,
    trend: TrendSignal | None,
    prefs: MonitoringPreferences,
    available_cash: float,
) -> tuple[RebalanceAction, float]:
    price = _price_for(drift, trend)
    trend_state = trend.trend_state if trend else "unknown"
    if not price:
        return _build_watch_action(drift, trend, "missing_price"), 0.0
    desired_value = abs(drift.drift_value)
    if desired_value < prefs.min_trade_value:
        return _build_watch_action(drift, trend, "underweight_below_min_trade"), 0.0
    trade_value = min(desired_value, max(0.0, available_cash))
    if trade_value < prefs.min_trade_value:
        return _build_watch_action(drift, trend, "insufficient_cash_after_reserve"), 0.0
    if prefs.trend_overlay_enabled and trend_state == "extended_uptrend":
        trade_value *= 0.5
    quantity = _floor_to_lot(trade_value / price, prefs.lot_size)
    if quantity <= 0:
        return _build_watch_action(drift, trend, "quantity_below_lot_size"), 0.0
    trade_value = quantity * price
    tranches = _buy_tranches(drift.symbol, quantity, price, trend, prefs)
    order = SimulatedOrder(
        code=drift.symbol,
        side="BUY",
        quantity=quantity,
        price=round(tranches[0].limit_price if tranches else price, 2),
        market=prefs.market,
        trd_env=prefs.trd_env,
    )
    reason_code = "underweight"
    trigger = "Add toward target because current weight is below the lower band."
    if prefs.trend_overlay_enabled and trend_state == "extended_uptrend":
        reason_code = "underweight_wait_pullback"
        trigger = "Underweight, but price is extended; use smaller staged buy limits."
    return (
        RebalanceAction(
            symbol=drift.symbol,
            action="add",
            reason_code=reason_code,
            target_trade_value=round(trade_value, 2),
            quantity=quantity,
            cash_impact=round(-trade_value, 2),
            current_weight=drift.current_weight,
            target_weight=drift.target_weight,
            band=(drift.lower_bound, drift.upper_bound),
            trend_state=trend_state,
            trigger=trigger,
            invalidation="Rebuild if target map, cash, quote, or trend artifact changes.",
            tranches=tranches,
            simulated_order=order,
            evidence_refs=drift.evidence_refs,
            warnings=list(drift.warnings),
        ),
        trade_value,
    )


def _build_trim_action(
    drift: DriftPosition,
    trend: TrendSignal | None,
    prefs: MonitoringPreferences,
    total_assets: float,
) -> RebalanceAction:
    price = _price_for(drift, trend)
    trend_state = trend.trend_state if trend else "unknown"
    if not price:
        return _build_watch_action(drift, trend, "missing_price")
    if drift.can_sell_qty <= 0:
        return _build_watch_action(drift, trend, "no_sellable_quantity")

    target_trim_value = max(0.0, drift.current_value - drift.target_value)
    if prefs.trend_overlay_enabled and trend_state in {"uptrend", "extended_uptrend"}:
        target_trim_value = max(0.0, drift.current_value - drift.upper_bound * total_assets)
    if drift.status == "extra":
        target_trim_value = drift.current_value
    if target_trim_value < prefs.min_trade_value:
        return _build_watch_action(drift, trend, "overweight_below_min_trade")

    quantity = min(drift.can_sell_qty, _floor_to_lot(target_trim_value / price, prefs.lot_size))
    if quantity <= 0:
        return _build_watch_action(drift, trend, "quantity_below_lot_size")
    trade_value = quantity * price
    tranches = _sell_tranches(drift.symbol, quantity, price, trend, prefs)
    order = SimulatedOrder(
        code=drift.symbol,
        side="SELL",
        quantity=quantity,
        price=round(tranches[0].limit_price if tranches else price, 2),
        market=prefs.market,
        trd_env=prefs.trd_env,
    )
    reason_code = "overweight"
    trigger = "Trim toward target because current weight is above the upper band."
    if drift.status == "extra":
        reason_code = "extra_position"
        trigger = "Position is outside the selected target map."
    elif prefs.trend_overlay_enabled and trend_state in {"uptrend", "extended_uptrend"}:
        reason_code = "overweight_take_profit_to_band"
        trigger = "Trim only back toward the upper band while trend remains strong."
    elif prefs.trend_overlay_enabled and trend_state in {"downtrend", "weakening"}:
        reason_code = "overweight_defensive_trim"
        trigger = "Trim faster because the position is overweight and trend is weakening."
    return RebalanceAction(
        symbol=drift.symbol,
        action="trim",
        reason_code=reason_code,
        target_trade_value=round(trade_value, 2),
        quantity=quantity,
        cash_impact=round(trade_value, 2),
        current_weight=drift.current_weight,
        target_weight=drift.target_weight,
        band=(drift.lower_bound, drift.upper_bound),
        trend_state=trend_state,
        trigger=trigger,
        invalidation="Cancel trim if target map is revised upward or sellable quantity changes.",
        tranches=tranches,
        simulated_order=order,
        evidence_refs=drift.evidence_refs,
        warnings=list(drift.warnings),
    )


def _build_hold_or_watch_action(
    drift: DriftPosition,
    trend: TrendSignal | None,
    prefs: MonitoringPreferences,
) -> RebalanceAction:
    trend_state = trend.trend_state if trend else "unknown"
    if not prefs.trend_overlay_enabled:
        return _build_hold_action(drift, trend, "within_band")
    if trend_state == "extended_uptrend" and drift.current_weight >= drift.target_weight:
        return _build_watch_action(drift, trend, "within_band_extended_watch")
    if trend_state in {"downtrend", "weakening"} and drift.current_weight <= drift.target_weight:
        return _build_watch_action(drift, trend, "within_band_weakening_watch")
    return _build_hold_action(drift, trend, "within_band")


def _build_hold_action(drift: DriftPosition, trend: TrendSignal | None, reason_code: str) -> RebalanceAction:
    trend_state = trend.trend_state if trend else "unknown"
    return RebalanceAction(
        symbol=drift.symbol,
        action="hold",
        reason_code=reason_code,
        target_trade_value=0,
        quantity=0,
        cash_impact=0,
        current_weight=drift.current_weight,
        target_weight=drift.target_weight,
        band=(drift.lower_bound, drift.upper_bound),
        trend_state=trend_state,
        trigger="Current weight is within the target band.",
        invalidation="Rebuild if target map, holdings, or market data changes.",
        evidence_refs=drift.evidence_refs,
        warnings=list(drift.warnings),
    )


def _build_watch_action(drift: DriftPosition, trend: TrendSignal | None, reason_code: str) -> RebalanceAction:
    trend_state = trend.trend_state if trend else "unknown"
    warnings = list(drift.warnings)
    if trend:
        warnings.extend(trend.warnings)
    return RebalanceAction(
        symbol=drift.symbol,
        action="watch",
        reason_code=reason_code,
        target_trade_value=0,
        quantity=0,
        cash_impact=0,
        current_weight=drift.current_weight,
        target_weight=drift.target_weight,
        band=(drift.lower_bound, drift.upper_bound),
        trend_state=trend_state,
        trigger="No simulated order generated for this monitoring run.",
        invalidation="Rebuild if target map, holdings, or market data changes.",
        evidence_refs=drift.evidence_refs,
        warnings=warnings,
    )


def _buy_tranches(
    symbol: str,
    total_quantity: int,
    price: float,
    trend: TrendSignal | None,
    prefs: MonitoringPreferences,
) -> list[RebalanceTranche]:
    atr = trend.atr14 if trend and trend.atr14 else price * 0.03
    supports = trend.support_levels if trend else []
    if trend and trend.trend_state == "extended_uptrend":
        limit_prices = [
            supports[0] if len(supports) >= 1 else max(0.01, price - atr),
            supports[1] if len(supports) >= 2 else max(0.01, price - 2 * atr),
            supports[2] if len(supports) >= 3 else max(0.01, price - 3 * atr),
        ]
        triggers = [
            "first pullback support",
            "second pullback support",
            "deeper pullback support",
        ]
    else:
        limit_prices = [
            price,
            supports[0] if len(supports) >= 1 else max(0.01, price - atr),
            supports[1] if len(supports) >= 2 else max(0.01, price - 2 * atr),
        ]
        triggers = [
            "near current price",
            "pullback toward first support",
            "pullback toward second support",
        ]
    return _split_quantity_tranches(
        symbol=symbol,
        side="BUY",
        total_quantity=total_quantity,
        limit_prices=limit_prices,
        triggers=triggers,
        invalidation="Stop adding if price trend or target map changes.",
        lot_size=prefs.lot_size,
    )


def _sell_tranches(
    symbol: str,
    total_quantity: int,
    price: float,
    trend: TrendSignal | None,
    prefs: MonitoringPreferences,
) -> list[RebalanceTranche]:
    resistances = trend.resistance_levels if trend else []
    if trend and trend.trend_state in {"downtrend", "weakening"}:
        limit_prices = [price, max(0.01, price * 0.98), max(0.01, price * 0.96)]
        triggers = [
            "defensive trim near current price",
            "trim if weakness continues",
            "trim on deeper support break",
        ]
    else:
        limit_prices = [
            price,
            resistances[0] if len(resistances) >= 1 else price * 1.03,
            resistances[1] if len(resistances) >= 2 else price * 1.06,
        ]
        triggers = [
            "trim near current price",
            "trim into first resistance",
            "trim into second resistance",
        ]
    return _split_quantity_tranches(
        symbol=symbol,
        side="SELL",
        total_quantity=total_quantity,
        limit_prices=limit_prices,
        triggers=triggers,
        invalidation="Cancel trim if target weight is revised upward.",
        lot_size=prefs.lot_size,
    )


def _split_quantity_tranches(
    *,
    symbol: str,
    side: str,
    total_quantity: int,
    limit_prices: list[float],
    triggers: list[str],
    invalidation: str,
    lot_size: int,
) -> list[RebalanceTranche]:
    tranches: list[RebalanceTranche] = []
    remaining = total_quantity
    for index, (weight, limit_price, trigger) in enumerate(zip(TRANCHE_WEIGHTS, limit_prices, triggers)):
        if index == len(TRANCHE_WEIGHTS) - 1:
            quantity = remaining
        else:
            quantity = _floor_to_lot(total_quantity * weight, lot_size)
            remaining -= quantity
        if quantity <= 0:
            continue
        tranches.append(
            RebalanceTranche(
                symbol=symbol,
                side=side,
                weight=weight,
                quantity=quantity,
                limit_price=round(limit_price, 2),
                trigger=trigger,
                invalidation=invalidation,
                estimated_value=round(quantity * limit_price, 2),
            )
        )
    return tranches


def _price_for(drift: DriftPosition, trend: TrendSignal | None) -> float | None:
    if trend and trend.price and trend.price > 0:
        return trend.price
    if drift.quantity > 0 and drift.current_value > 0:
        return drift.current_value / drift.quantity
    return None


def _floor_to_lot(value: float, lot_size: int) -> int:
    quantity = math.floor(value)
    return quantity - (quantity % lot_size)
