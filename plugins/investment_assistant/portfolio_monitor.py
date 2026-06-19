"""Run portfolio monitoring against a selected target map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .drift import build_monitor_snapshot, compute_portfolio_drift, extract_target_positions, selected_map_id
from .monitoring_models import MarketPriceData, MonitoringPreferences, PortfolioMonitorResult, PriceBar, TrendSignal
from .rebalance_planner import build_rebalance_plan
from .schemas import CurrentHolding, CurrentPortfolio
from .trend_signals import compute_trend_signals, parse_market_data

PORTFOLIO_MONITOR_RESULT_FILENAME = "portfolio_monitor_result.json"
PORTFOLIO_DRIFT_REPORT_FILENAME = "portfolio_drift_report.json"
PORTFOLIO_TREND_SIGNALS_FILENAME = "portfolio_trend_signals.json"
PORTFOLIO_REBALANCE_PLAN_FILENAME = "portfolio_rebalance_plan.json"


def monitor_portfolio(
    *,
    portfolio_map: Any,
    portfolio: CurrentPortfolio,
    market_data: dict[str, MarketPriceData] | None = None,
    preferences: MonitoringPreferences | None = None,
) -> PortfolioMonitorResult:
    """Build one monitoring artifact set for an already selected target map."""

    prefs = preferences or MonitoringPreferences()
    market = market_data or _market_data_from_current_holdings(portfolio)
    target_symbols = {target.symbol for target in extract_target_positions(portfolio_map)}
    for symbol in target_symbols:
        market.setdefault(symbol, MarketPriceData(symbol=symbol, warnings=["Missing market data for target symbol."]))
    snapshot = build_monitor_snapshot(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        data_asof={
            "portfolio": portfolio.data_asof,
            "market": _latest_market_asof(market),
        },
    )
    drift_report = compute_portfolio_drift(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        preferences=prefs,
    )
    trend_by_symbol = compute_trend_signals(market)
    rebalance_plan = build_rebalance_plan(
        drift_report=drift_report,
        trend_signals=trend_by_symbol,
        preferences=prefs,
    )
    warnings = _dedupe(
        [
            *snapshot.warnings,
            *drift_report.warnings,
            *rebalance_plan.warnings,
            *[warning for signal in trend_by_symbol.values() for warning in signal.warnings],
        ]
    )
    return PortfolioMonitorResult(
        selected_map_id=selected_map_id(portfolio_map),
        snapshot=snapshot,
        drift_report=drift_report,
        trend_signals=list(trend_by_symbol.values()),
        rebalance_plan=rebalance_plan,
        warnings=warnings,
    )


def monitor_portfolio_from_files(
    *,
    portfolio_map_path: str | Path,
    current_portfolio_path: str | Path,
    market_data_path: str | Path | None = None,
    preferences: MonitoringPreferences | None = None,
    output_dir: str | Path | None = None,
) -> PortfolioMonitorResult:
    portfolio_map = _read_json(Path(portfolio_map_path))
    portfolio = _read_current_portfolio(Path(current_portfolio_path))
    market_data = parse_market_data(_read_json(Path(market_data_path))) if market_data_path else None
    result = monitor_portfolio(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        market_data=market_data,
        preferences=preferences,
    )
    if output_dir:
        _write_monitor_result(Path(output_dir), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        preferences = MonitoringPreferences(
            core_relative_band=args.core_relative_band,
            high_conviction_relative_band=args.high_conviction_relative_band,
            satellite_relative_band=args.satellite_relative_band,
            default_relative_band=args.default_relative_band,
            min_absolute_band_weight=args.min_absolute_band_weight,
            min_trade_value=args.min_trade_value,
            max_add_cash_fraction=args.max_add_cash_fraction,
            trend_overlay_enabled=not args.disable_trend_overlay,
            trim_extra_positions=args.trim_extra_positions,
            lot_size=args.lot_size,
            market=args.market,
            trd_env=args.trd_env,
        )
        result = monitor_portfolio_from_files(
            portfolio_map_path=args.portfolio_map_path,
            current_portfolio_path=args.current_portfolio_path,
            market_data_path=args.market_data_path,
            preferences=preferences,
            output_dir=args.output_dir,
        )
        payload = result.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"monitor_id: {result.monitor_id}")
            print(f"selected_map_id: {result.selected_map_id}")
            print(f"actions: {len(result.rebalance_plan.actions)}")
            print(f"simulated_orders: {len(result.rebalance_plan.simulated_orders)}")
            print(f"cash_required: {result.rebalance_plan.cash_required}")
            print(f"cash_released: {result.rebalance_plan.cash_released}")
            if args.output_dir:
                print(f"output_dir: {args.output_dir}")
            if result.warnings:
                print("warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-portfolio-monitor",
        description="Monitor a current portfolio against an already selected target map.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one portfolio monitoring pass.")
    run.add_argument("--portfolio-map-path", required=True, help="Selected portfolio map JSON.")
    run.add_argument("--current-portfolio-path", required=True, help="CurrentPortfolio JSON.")
    run.add_argument("--market-data-path", help="Optional market data JSON with quotes and daily kline.")
    run.add_argument("--output-dir", help="Directory for monitoring artifacts.")
    run.add_argument("--core-relative-band", type=float, default=0.20)
    run.add_argument("--high-conviction-relative-band", type=float, default=0.25)
    run.add_argument("--satellite-relative-band", type=float, default=0.35)
    run.add_argument("--default-relative-band", type=float, default=0.25)
    run.add_argument("--min-absolute-band-weight", type=float, default=0.005)
    run.add_argument("--min-trade-value", type=float, default=500.0)
    run.add_argument("--max-add-cash-fraction", type=float, default=1.0)
    run.add_argument("--disable-trend-overlay", action="store_true")
    run.add_argument("--trim-extra-positions", action="store_true")
    run.add_argument("--lot-size", type=int, default=1)
    run.add_argument("--market", default="US")
    run.add_argument("--trd-env", default="SIMULATE")
    run.add_argument("--json", action="store_true")
    return parser


def _read_current_portfolio(path: Path) -> CurrentPortfolio:
    payload = _read_json(path)
    portfolio = CurrentPortfolio.model_validate(payload)
    return CurrentPortfolio(
        total_assets=portfolio.total_assets,
        cash=portfolio.cash,
        holdings=[
            CurrentHolding(
                symbol=holding.symbol.upper(),
                quantity=holding.quantity,
                market_value=holding.market_value,
                cost_basis=holding.cost_basis,
                can_sell_qty=holding.can_sell_qty,
            )
            for holding in portfolio.holdings
        ],
        data_asof=portfolio.data_asof,
        source=portfolio.source,
        warnings=portfolio.warnings,
    )


def _market_data_from_current_holdings(portfolio: CurrentPortfolio) -> dict[str, MarketPriceData]:
    result: dict[str, MarketPriceData] = {}
    for holding in portfolio.holdings:
        price = holding.market_value / holding.quantity if holding.quantity > 0 else None
        result[holding.symbol.upper()] = MarketPriceData(
            symbol=holding.symbol.upper(),
            last_price=price,
            update_time=portfolio.data_asof,
            kline=[
                PriceBar(close=price, high=price, low=price)
                for _ in range(20)
                if price is not None
            ],
            warnings=["Market data inferred from current holding value; fetch live Futu data before acting."],
        )
    return result


def _latest_market_asof(market: dict[str, MarketPriceData]) -> str:
    values = sorted({item.update_time for item in market.values() if item.update_time})
    return values[-1] if values else ""


def _write_monitor_result(output_dir: Path, result: PortfolioMonitorResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / PORTFOLIO_MONITOR_RESULT_FILENAME, result.model_dump(mode="json"))
    _write_json(output_dir / PORTFOLIO_DRIFT_REPORT_FILENAME, result.drift_report.model_dump(mode="json"))
    _write_json(
        output_dir / PORTFOLIO_TREND_SIGNALS_FILENAME,
        [signal.model_dump(mode="json") for signal in result.trend_signals],
    )
    _write_json(output_dir / PORTFOLIO_REBALANCE_PLAN_FILENAME, result.rebalance_plan.model_dump(mode="json"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
