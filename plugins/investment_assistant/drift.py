"""Portfolio drift calculation against a selected AI-authored map."""

from __future__ import annotations

from typing import Any

from .adapters import holding_index
from .monitoring_models import (
    DriftPosition,
    MonitoringPreferences,
    PortfolioDriftReport,
    PortfolioMonitorSnapshot,
    TargetPosition,
)
from .schemas import CurrentHolding, CurrentPortfolio


def extract_target_positions(portfolio_map: Any) -> list[TargetPosition]:
    """Extract target holdings from PortfolioMap or DirectPortfolioMap shaped data."""

    if isinstance(portfolio_map, dict):
        holdings = portfolio_map.get("holdings") or []
    else:
        holdings = getattr(portfolio_map, "holdings", [])
    targets: list[TargetPosition] = []
    for holding in holdings:
        item = _as_mapping(holding)
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        targets.append(
            TargetPosition(
                symbol=symbol,
                target_weight=float(item.get("target_weight") or 0),
                sleeve_key=str(item.get("sleeve_key") or item.get("sleeve") or ""),
                role=str(item.get("role") or ""),
                rationale=str(item.get("rationale") or ""),
                suggested_weight_band=_normalize_band(item.get("suggested_weight_band")),
                evidence_refs=[str(value) for value in item.get("evidence_refs", [])],
            )
        )
    return targets


def selected_map_id(portfolio_map: Any) -> str:
    if isinstance(portfolio_map, dict):
        return str(portfolio_map.get("map_id") or portfolio_map.get("portfolio_map_id") or "")
    return str(getattr(portfolio_map, "map_id", "") or getattr(portfolio_map, "portfolio_map_id", ""))


def selected_map_cash_weight(portfolio_map: Any) -> float:
    if isinstance(portfolio_map, dict):
        return float(portfolio_map.get("cash_weight") or 0)
    return float(getattr(portfolio_map, "cash_weight", 0) or 0)


def build_monitor_snapshot(
    *,
    portfolio_map: Any,
    portfolio: CurrentPortfolio,
    data_asof: dict[str, str] | None = None,
) -> PortfolioMonitorSnapshot:
    cash_weight = portfolio.cash / portfolio.total_assets if portfolio.total_assets else 0.0
    return PortfolioMonitorSnapshot(
        selected_map_id=selected_map_id(portfolio_map),
        data_asof=data_asof or {"portfolio": portfolio.data_asof},
        total_assets=round(portfolio.total_assets, 2),
        cash=round(portfolio.cash, 2),
        cash_weight=round(cash_weight, 6),
        positions=portfolio.holdings,
        warnings=list(portfolio.warnings),
    )


def compute_portfolio_drift(
    *,
    portfolio_map: Any,
    portfolio: CurrentPortfolio,
    preferences: MonitoringPreferences | None = None,
    include_extra_positions: bool = True,
) -> PortfolioDriftReport:
    """Compare current holdings to target weights with tolerance bands."""

    prefs = preferences or MonitoringPreferences()
    targets = extract_target_positions(portfolio_map)
    holdings_by_symbol = holding_index(portfolio)
    target_by_symbol = {target.symbol: target for target in targets}
    positions: list[DriftPosition] = []
    warnings = list(portfolio.warnings)

    for target in targets:
        holding = holdings_by_symbol.get(target.symbol)
        current_value = holding.market_value if holding else 0.0
        quantity = holding.quantity if holding else 0
        can_sell_qty = holding.can_sell_qty if holding else 0
        target_value = portfolio.total_assets * target.target_weight
        tolerance = _tolerance_for(target, prefs)
        lower, upper = _bounds(target.target_weight, tolerance, prefs.min_absolute_band_weight)
        current_weight = current_value / portfolio.total_assets
        status = _status_for(current_weight, lower, upper)
        positions.append(
            DriftPosition(
                symbol=target.symbol,
                target_weight=round(target.target_weight, 6),
                current_weight=round(current_weight, 6),
                lower_bound=round(lower, 6),
                upper_bound=round(upper, 6),
                tolerance=round(tolerance, 6),
                target_value=round(target_value, 2),
                current_value=round(current_value, 2),
                drift_weight=round(current_weight - target.target_weight, 6),
                drift_value=round(current_value - target_value, 2),
                quantity=quantity,
                can_sell_qty=can_sell_qty,
                status=status,
                sleeve_key=target.sleeve_key,
                role=target.role,
                rationale=target.rationale,
                evidence_refs=target.evidence_refs,
            )
        )

    if include_extra_positions:
        for holding in portfolio.holdings:
            symbol = holding.symbol.upper()
            if symbol in target_by_symbol:
                continue
            current_weight = holding.market_value / portfolio.total_assets
            status = "extra"
            positions.append(
                DriftPosition(
                    symbol=symbol,
                    target_weight=0,
                    current_weight=round(current_weight, 6),
                    lower_bound=0,
                    upper_bound=0,
                    tolerance=0,
                    target_value=0,
                    current_value=round(holding.market_value, 2),
                    drift_weight=round(current_weight, 6),
                    drift_value=round(holding.market_value, 2),
                    quantity=holding.quantity,
                    can_sell_qty=holding.can_sell_qty,
                    status=status,
                    warnings=["Holding is not in the selected target map."],
                )
            )

    return PortfolioDriftReport(
        selected_map_id=selected_map_id(portfolio_map),
        total_assets=round(portfolio.total_assets, 2),
        cash=round(portfolio.cash, 2),
        cash_weight=round(portfolio.cash / portfolio.total_assets, 6),
        target_cash_weight=round(selected_map_cash_weight(portfolio_map), 6),
        positions=positions,
        warnings=warnings,
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _normalize_band(value: Any) -> tuple[float, float] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
        if 0 <= low <= high <= 1:
            return (low, high)
    return None


def _tolerance_for(target: TargetPosition, prefs: MonitoringPreferences) -> float:
    if target.suggested_weight_band:
        low, high = target.suggested_weight_band
        if target.target_weight > 0:
            return max(target.target_weight - low, high - target.target_weight) / target.target_weight
    role_text = f"{target.sleeve_key} {target.role}".lower()
    if "core" in role_text or target.target_weight >= 0.10:
        return prefs.core_relative_band
    if "satellite" in role_text or target.target_weight <= 0.03:
        return prefs.satellite_relative_band
    if "conviction" in role_text or target.target_weight >= 0.05:
        return prefs.high_conviction_relative_band
    return prefs.default_relative_band


def _bounds(target_weight: float, tolerance: float, min_abs_band: float) -> tuple[float, float]:
    low = target_weight * (1 - tolerance)
    high = target_weight * (1 + tolerance)
    low = max(0.0, min(low, target_weight - min_abs_band))
    high = min(1.0, max(high, target_weight + min_abs_band))
    return low, high


def _status_for(current_weight: float, lower: float, upper: float) -> str:
    if current_weight < lower:
        return "underweight"
    if current_weight > upper:
        return "overweight"
    return "within_band"
