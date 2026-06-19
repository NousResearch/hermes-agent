"""Exposure-aware drift calculation against a selected target map."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .drift import extract_target_positions, selected_map_cash_weight, selected_map_id
from .exposure_ledger import NormalizedExposureLedger, NormalizedExposurePosition, OptionStrategyGroup
from .monitoring_models import MonitoringPreferences, TargetPosition
from .storage import new_id, utc_now


ExposureDriftStatus = Literal["underweight", "overweight", "within_band", "extra"]


class ExposureContribution(BaseModel):
    raw_code: str
    name: str = ""
    instrument_type: str
    underlying: str
    theme_sleeve: str = ""
    portfolio_role: str = ""
    direction: str = ""
    market_value: float = 0
    leverage_factor: float = 1
    effective_long_exposure: float = 0
    effective_short_exposure: float = 0
    net_effective_exposure: float = 0
    warnings: list[str] = Field(default_factory=list)


class ExposureOverlaySummary(BaseModel):
    strategy_id: str
    underlying: str
    strategy_type: str
    expiry: str
    intent_guess: str = ""
    coverage_status: str = "not_applicable"
    market_value: float = 0
    max_profit: float | None = None
    max_loss: float | None = None
    short_assignment_notional: float = 0
    counted_in_target_exposure: bool = False
    target_exposure_reason: str = ""
    warnings: list[str] = Field(default_factory=list)


class ExposureDriftPosition(BaseModel):
    symbol: str
    status: ExposureDriftStatus
    sleeve_key: str = ""
    role: str = ""
    target_weight: float = Field(ge=0, le=1)
    current_effective_weight: float
    lower_bound: float = Field(ge=0)
    upper_bound: float = Field(ge=0)
    target_value: float = Field(ge=0)
    current_effective_exposure: float = 0
    direct_market_value: float = 0
    leveraged_effective_exposure: float = 0
    option_market_value: float = 0
    option_max_loss: float = 0
    option_short_assignment_notional: float = 0
    drift_weight: float
    drift_value: float
    contributions: list[ExposureContribution] = Field(default_factory=list)
    overlays: list[ExposureOverlaySummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExposureSleeveDrift(BaseModel):
    sleeve_key: str
    target_weight: float = Field(ge=0)
    current_effective_weight: float
    target_value: float = Field(ge=0)
    current_effective_exposure: float = 0
    drift_weight: float
    drift_value: float
    symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExposureDriftReport(BaseModel):
    artifact_type: str = "exposure_drift_report"
    report_id: str = Field(default_factory=lambda: new_id("edr"))
    selected_map_id: str = ""
    generated_at: str = Field(default_factory=utc_now)
    source_ledger_id: str = ""
    total_assets: float = Field(gt=0)
    cash: float = 0
    cash_weight: float = 0
    target_cash_weight: float = Field(ge=0, le=1)
    positions: list[ExposureDriftPosition] = Field(default_factory=list)
    sleeve_drifts: list[ExposureSleeveDrift] = Field(default_factory=list)
    overlays: list[ExposureOverlaySummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def compute_exposure_drift(
    *,
    portfolio_map: Any,
    ledger: NormalizedExposureLedger,
    preferences: MonitoringPreferences | None = None,
) -> ExposureDriftReport:
    """Compare normalized effective exposure to target weights.

    Stock, ETF, and user-confirmed strategic leveraged ETF exposure are counted
    through `effective_long_exposure - effective_short_exposure`. Option legs
    remain overlays in V1 unless a later Greeks-aware module explicitly maps
    them into delta exposure.
    """

    if ledger.total_assets <= 0:
        raise ValueError("normalized_exposure_ledger.total_assets must be > 0.")
    prefs = preferences or MonitoringPreferences()
    targets = extract_target_positions(portfolio_map)
    target_by_symbol = {target.symbol: target for target in targets}
    contributions_by_underlying = _contributions_by_underlying(ledger.positions)
    overlays_by_underlying = _overlays_by_underlying(ledger.option_strategies)

    positions: list[ExposureDriftPosition] = []
    warnings = _dedupe([*ledger.warnings])
    if ledger.cash < 0:
        warnings.append("Cash is negative; additions should be constrained until margin risk is reviewed.")

    for target in targets:
        contributions = contributions_by_underlying.get(target.symbol, [])
        overlays = overlays_by_underlying.get(target.symbol, [])
        positions.append(
            _build_target_position(
                target=target,
                total_assets=ledger.total_assets,
                contributions=contributions,
                overlays=overlays,
                preferences=prefs,
            )
        )

    target_symbols = set(target_by_symbol)
    extra_underlyings = sorted(
        symbol
        for symbol, contributions in contributions_by_underlying.items()
        if symbol not in target_symbols and _net_effective_exposure(contributions) != 0
    )
    for symbol in extra_underlyings:
        contributions = contributions_by_underlying[symbol]
        overlays = overlays_by_underlying.get(symbol, [])
        positions.append(
            _build_extra_position(
                symbol=symbol,
                total_assets=ledger.total_assets,
                contributions=contributions,
                overlays=overlays,
            )
        )

    sleeve_drifts = _build_sleeve_drifts(
        targets=targets,
        positions=positions,
        total_assets=ledger.total_assets,
    )
    all_overlays = [
        _overlay_summary(strategy)
        for strategy in ledger.option_strategies
    ]
    return ExposureDriftReport(
        selected_map_id=selected_map_id(portfolio_map) or ledger.selected_map_id,
        source_ledger_id=ledger.ledger_id,
        total_assets=round(ledger.total_assets, 2),
        cash=round(ledger.cash, 2),
        cash_weight=round(ledger.cash_weight, 6),
        target_cash_weight=round(selected_map_cash_weight(portfolio_map), 6),
        positions=positions,
        sleeve_drifts=sleeve_drifts,
        overlays=all_overlays,
        warnings=_dedupe(warnings),
    )


def build_exposure_drift_from_files(
    *,
    portfolio_map_path: str | Path,
    ledger_path: str | Path,
    output_dir: str | Path | None = None,
) -> ExposureDriftReport:
    portfolio_map = _read_json(Path(portfolio_map_path))
    ledger = NormalizedExposureLedger.model_validate(_read_json(Path(ledger_path)))
    report = compute_exposure_drift(portfolio_map=portfolio_map, ledger=ledger)
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "exposure_drift_report.json", report.model_dump(mode="json"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_exposure_drift_from_files(
            portfolio_map_path=args.portfolio_map_path,
            ledger_path=args.ledger_path,
            output_dir=args.output_dir,
        )
        payload = report.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"report_id: {report.report_id}")
            print(f"positions: {len(report.positions)}")
            print(f"overlays: {len(report.overlays)}")
            if args.output_dir:
                print(f"output_dir: {args.output_dir}")
            if report.warnings:
                print("warnings:")
                for warning in report.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-exposure-drift",
        description="Compare a normalized exposure ledger to a target portfolio map.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build one exposure drift report.")
    build.add_argument("--portfolio-map-path", required=True, help="Selected portfolio map JSON path.")
    build.add_argument("--ledger-path", required=True, help="normalized_exposure_ledger.json path.")
    build.add_argument("--output-dir", help="Directory for exposure_drift_report.json.")
    build.add_argument("--json", action="store_true")
    return parser


def _build_target_position(
    *,
    target: TargetPosition,
    total_assets: float,
    contributions: list[ExposureContribution],
    overlays: list[OptionStrategyGroup],
    preferences: MonitoringPreferences,
) -> ExposureDriftPosition:
    current_exposure = _net_effective_exposure(contributions)
    current_weight = current_exposure / total_assets
    target_value = total_assets * target.target_weight
    tolerance = _tolerance_for(target, preferences)
    lower, upper = _bounds(target.target_weight, tolerance, preferences.min_absolute_band_weight)
    status = _status_for(current_weight, lower, upper)
    overlay_summaries = [_overlay_summary(strategy) for strategy in overlays]
    warnings = _target_position_warnings(contributions, overlay_summaries)
    return ExposureDriftPosition(
        symbol=target.symbol,
        status=status,
        sleeve_key=target.sleeve_key,
        role=target.role,
        target_weight=round(target.target_weight, 6),
        current_effective_weight=round(current_weight, 6),
        lower_bound=round(lower, 6),
        upper_bound=round(upper, 6),
        target_value=round(target_value, 2),
        current_effective_exposure=round(current_exposure, 2),
        direct_market_value=round(_direct_market_value(contributions, target.symbol), 2),
        leveraged_effective_exposure=round(_leveraged_effective_exposure(contributions), 2),
        option_market_value=round(sum(item.market_value for item in overlay_summaries), 2),
        option_max_loss=round(sum(item.max_loss or 0 for item in overlay_summaries), 2),
        option_short_assignment_notional=round(sum(item.short_assignment_notional for item in overlay_summaries), 2),
        drift_weight=round(current_weight - target.target_weight, 6),
        drift_value=round(current_exposure - target_value, 2),
        contributions=contributions,
        overlays=overlay_summaries,
        warnings=warnings,
    )


def _build_extra_position(
    *,
    symbol: str,
    total_assets: float,
    contributions: list[ExposureContribution],
    overlays: list[OptionStrategyGroup],
) -> ExposureDriftPosition:
    current_exposure = _net_effective_exposure(contributions)
    overlay_summaries = [_overlay_summary(strategy) for strategy in overlays]
    warnings = _dedupe(
        [
            "Underlying effective exposure is not in the selected target map.",
            *_target_position_warnings(contributions, overlay_summaries),
        ]
    )
    return ExposureDriftPosition(
        symbol=symbol,
        status="extra",
        sleeve_key=_first_non_empty(item.theme_sleeve for item in contributions) or "extra",
        target_weight=0,
        current_effective_weight=round(current_exposure / total_assets, 6),
        lower_bound=0,
        upper_bound=0,
        target_value=0,
        current_effective_exposure=round(current_exposure, 2),
        direct_market_value=round(_direct_market_value(contributions, symbol), 2),
        leveraged_effective_exposure=round(_leveraged_effective_exposure(contributions), 2),
        option_market_value=round(sum(item.market_value for item in overlay_summaries), 2),
        option_max_loss=round(sum(item.max_loss or 0 for item in overlay_summaries), 2),
        option_short_assignment_notional=round(sum(item.short_assignment_notional for item in overlay_summaries), 2),
        drift_weight=round(current_exposure / total_assets, 6),
        drift_value=round(current_exposure, 2),
        contributions=contributions,
        overlays=overlay_summaries,
        warnings=warnings,
    )


def _contributions_by_underlying(
    positions: list[NormalizedExposurePosition],
) -> dict[str, list[ExposureContribution]]:
    grouped: dict[str, list[ExposureContribution]] = defaultdict(list)
    for position in positions:
        if position.portfolio_role == "ignored_market":
            continue
        if (
            position.instrument_type == "option_leg"
            and position.effective_long_exposure == 0
            and position.effective_short_exposure == 0
        ):
            continue
        underlying = position.underlying.upper()
        if not underlying:
            continue
        contribution = ExposureContribution(
            raw_code=position.raw_code,
            name=position.name,
            instrument_type=position.instrument_type,
            underlying=underlying,
            theme_sleeve=position.theme_sleeve,
            portfolio_role=position.portfolio_role,
            direction=position.direction,
            market_value=round(position.market_value, 2),
            leverage_factor=position.leverage_factor,
            effective_long_exposure=round(position.effective_long_exposure, 2),
            effective_short_exposure=round(position.effective_short_exposure, 2),
            net_effective_exposure=round(position.effective_long_exposure - position.effective_short_exposure, 2),
            warnings=list(position.warnings),
        )
        grouped[underlying].append(contribution)
    return {symbol: sorted(items, key=lambda item: (item.instrument_type, item.raw_code)) for symbol, items in grouped.items()}


def _overlays_by_underlying(strategies: list[OptionStrategyGroup]) -> dict[str, list[OptionStrategyGroup]]:
    grouped: dict[str, list[OptionStrategyGroup]] = defaultdict(list)
    for strategy in strategies:
        if strategy.underlying:
            grouped[strategy.underlying.upper()].append(strategy)
    return dict(grouped)


def _overlay_summary(strategy: OptionStrategyGroup) -> ExposureOverlaySummary:
    counted_in_target = (
        strategy.intent_guess == "long_term_capital_efficiency"
        and (strategy.effective_long_exposure != 0 or strategy.effective_short_exposure != 0)
    )
    return ExposureOverlaySummary(
        strategy_id=strategy.strategy_id,
        underlying=strategy.underlying,
        strategy_type=strategy.strategy_type,
        expiry=strategy.expiry,
        intent_guess=strategy.intent_guess,
        coverage_status=strategy.coverage_status,
        market_value=round(strategy.market_value, 2),
        max_profit=strategy.max_profit,
        max_loss=strategy.max_loss,
        short_assignment_notional=round(strategy.short_assignment_notional, 2),
        counted_in_target_exposure=counted_in_target,
        target_exposure_reason=_overlay_target_exposure_reason(strategy),
        warnings=list(strategy.warnings),
    )


def _overlay_target_exposure_reason(strategy: OptionStrategyGroup) -> str:
    if strategy.strategy_type == "covered_call":
        return "Covered call modifies income/upside profile; underlying shares remain the counted target exposure."
    if strategy.strategy_type == "short_put" and strategy.intent_guess == "premium_income_willing_assignment":
        return "Short put is a contingent assignment plan, not current stock exposure."
    if strategy.intent_guess == "long_term_capital_efficiency":
        if strategy.effective_long_exposure or strategy.effective_short_exposure:
            return "Long call is mapped as delta-adjusted effective exposure using option Greeks."
        return "Long call needs delta/Greeks before V1 can map it into effective exposure."
    if strategy.intent_guess == "independent_high_iv_premium_income":
        return "Independent premium overlay; defined-risk max loss is tracked separately from target exposure."
    return "Option overlay is not mapped to target exposure in V1."


def _build_sleeve_drifts(
    *,
    targets: list[TargetPosition],
    positions: list[ExposureDriftPosition],
    total_assets: float,
) -> list[ExposureSleeveDrift]:
    target_weight_by_sleeve: dict[str, float] = defaultdict(float)
    symbols_by_sleeve: dict[str, set[str]] = defaultdict(set)
    for target in targets:
        sleeve = target.sleeve_key or "unknown"
        target_weight_by_sleeve[sleeve] += target.target_weight
        symbols_by_sleeve[sleeve].add(target.symbol)

    current_by_sleeve: dict[str, float] = defaultdict(float)
    warnings_by_sleeve: dict[str, list[str]] = defaultdict(list)
    for position in positions:
        sleeve = position.sleeve_key or ("extra" if position.status == "extra" else "unknown")
        current_by_sleeve[sleeve] += position.current_effective_exposure
        symbols_by_sleeve[sleeve].add(position.symbol)
        warnings_by_sleeve[sleeve].extend(position.warnings)

    sleeve_keys = sorted(set(target_weight_by_sleeve) | set(current_by_sleeve))
    return [
        ExposureSleeveDrift(
            sleeve_key=sleeve,
            target_weight=round(target_weight_by_sleeve[sleeve], 6),
            current_effective_weight=round(current_by_sleeve[sleeve] / total_assets, 6),
            target_value=round(target_weight_by_sleeve[sleeve] * total_assets, 2),
            current_effective_exposure=round(current_by_sleeve[sleeve], 2),
            drift_weight=round((current_by_sleeve[sleeve] / total_assets) - target_weight_by_sleeve[sleeve], 6),
            drift_value=round(current_by_sleeve[sleeve] - target_weight_by_sleeve[sleeve] * total_assets, 2),
            symbols=sorted(symbols_by_sleeve[sleeve]),
            warnings=_dedupe(warnings_by_sleeve[sleeve]),
        )
        for sleeve in sleeve_keys
    ]


def _target_position_warnings(
    contributions: list[ExposureContribution],
    overlays: list[ExposureOverlaySummary],
) -> list[str]:
    warnings: list[str] = []
    for contribution in contributions:
        warnings.extend(contribution.warnings)
        if contribution.instrument_type == "leveraged_etf":
            warnings.append("Effective exposure includes leveraged ETF approximation; rebalance should respect limited-rebalance intent.")
    for overlay in overlays:
        if overlay.max_loss:
            warnings.append(f"Option overlay max loss tracked separately: {overlay.strategy_type} max_loss={overlay.max_loss:g}.")
        if overlay.short_assignment_notional:
            warnings.append(
                f"Option overlay has contingent assignment notional: {overlay.strategy_type} {overlay.short_assignment_notional:g}."
            )
        warnings.extend(overlay.warnings)
    return _dedupe(warnings)


def _net_effective_exposure(contributions: list[ExposureContribution]) -> float:
    return round(sum(item.net_effective_exposure for item in contributions), 2)


def _direct_market_value(contributions: list[ExposureContribution], symbol: str) -> float:
    return sum(
        item.market_value
        for item in contributions
        if item.raw_code == symbol and item.instrument_type in {"stock", "etf", "unknown"}
    )


def _leveraged_effective_exposure(contributions: list[ExposureContribution]) -> float:
    return sum(item.net_effective_exposure for item in contributions if item.instrument_type == "leveraged_etf")


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


def _status_for(current_weight: float, lower: float, upper: float) -> ExposureDriftStatus:
    if current_weight < lower:
        return "underweight"
    if current_weight > upper:
        return "overweight"
    return "within_band"


def _first_non_empty(values) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
