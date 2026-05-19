from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from institutional_btc_vol.backtester import EVIDENCE_STATUS, BacktestResult, MarketSnapshot, run_vol_spread_backtest


def _parse_as_of_cst(value: str) -> datetime:
    text = value.replace(" CDT", "").replace(" CST", "")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def build_spread_snapshots_from_manifest_rows(rows: Iterable[dict[str, Any]], *, tenor: str = "7d") -> list[MarketSnapshot]:
    key = f"spread_{tenor}_vol_pts"
    usable = [row for row in rows if row.get(key) is not None and row.get("as_of_cst")]
    snapshots: list[MarketSnapshot] = []
    for current in usable:
        timestamp = _parse_as_of_cst(str(current["as_of_cst"]))
        snapshots.append(
            MarketSnapshot(
                timestamp=timestamp,
                data={
                    "run_id": current.get("run_id"),
                    key: float(current[key]),
                    "evidence_status": EVIDENCE_STATUS,
                },
            )
        )
    return snapshots


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _money(value: float) -> str:
    return f"${value:,.2f}"


def render_backtest_markdown(*, run_id: str, result: BacktestResult, tenor: str, snapshot_count: int, sample_gate: str | None = None) -> str:
    lines = [
        f"# BTC Vol Spread Backtest — {run_id}",
        "",
        f"**Evidence status:** `{result.evidence_status}`",
        "",
        "This is research evidence, not executable economics. The engine uses historical monitor snapshots, synthetic fills, and fixed cost assumptions; it does not prove that any spread was tradable.",
        "",
        "## Controls",
        "",
        "- No future leakage: the strategy receives only prior snapshots at each decision point.",
        "- Chronology is enforced; out-of-order snapshots fail instead of being silently sorted.",
        "- Synthetic fills are explicitly labeled and are not quotes, RFQs, or executions.",
        "",
        "## Summary",
        "",
        f"- Tenor: `{tenor}`",
        f"- Snapshot count: {snapshot_count}",
        f"- Trade count: {result.trade_count}",
        f"- Gross PnL: {_money(result.gross_pnl)}",
        f"- Max drawdown: {_money(result.max_drawdown)}",
        f"- Win rate: {result.win_rate:.1%}",
        f"- Minimum evidence gate: {sample_gate or ('pass' if snapshot_count >= 30 and result.trade_count >= 20 else 'insufficient-history')}",
        "",
        "## Assumptions",
        "",
    ]
    for key, value in result.assumptions.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Trades", ""])
    if not result.trades:
        lines.append("No threshold-qualified synthetic trades.")
    else:
        lines.extend(["| # | Side | Entry | Exit | Net PnL | Signal |", "|---:|---|---:|---:|---:|---|"])
        for idx, trade in enumerate(result.trades, 1):
            signal = trade.metadata.get("signal", "n/a")
            lines.append(f"| {idx} | {trade.side} | {trade.entry_price:.2f} | {trade.exit_price:.2f} | {_money(trade.net_pnl)} | {signal} |")
    lines.append("")
    return "\n".join(lines)


def _scenario_summary(
    *,
    tenor: str,
    result: BacktestResult,
    snapshot_count: int,
    threshold_vol_pts: float,
    notional_vega: float,
    cost_per_trade: float,
    slippage_vol_pts: float,
) -> dict[str, Any]:
    sample_gate = "pass" if snapshot_count >= 30 and result.trade_count >= 20 else "insufficient-history"
    return {
        "tenor": tenor,
        "snapshot_count": snapshot_count,
        "trade_count": result.trade_count,
        "gross_pnl": result.gross_pnl,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "threshold_vol_pts": threshold_vol_pts,
        "notional_vega": notional_vega,
        "cost_per_trade": cost_per_trade,
        "slippage_vol_pts": slippage_vol_pts,
        "sample_gate": sample_gate,
        "evidence_status": result.evidence_status,
    }


def _robustness_metrics(
    scenarios: list[dict[str, Any]],
    *,
    cost_per_trade: float,
    slippage_vol_pts: float,
    notional_vega: float,
    min_snapshot_gate: int = 30,
    min_trade_gate: int = 20,
) -> dict[str, Any]:
    effective_cost = float(cost_per_trade) + abs(float(slippage_vol_pts)) * float(notional_vega)
    if not scenarios:
        return {
            "scenario_count": 0,
            "sample_gate_pass_count": 0,
            "sample_gate_ready": False,
            "min_snapshot_gate": min_snapshot_gate,
            "min_trade_gate": min_trade_gate,
            "max_snapshot_count": 0,
            "max_trade_count": 0,
            "tenors_with_trades": [],
            "best_gross_pnl_tenor": None,
            "worst_gross_pnl_tenor": None,
            "cost_sensitivity": {
                "cost_per_trade": cost_per_trade,
                "slippage_vol_pts": slippage_vol_pts,
                "effective_cost_per_trade": effective_cost,
                "control_note": "Synthetic fills only; cost sensitivity is illustrative and not executable economics.",
            },
        }
    pass_count = sum(1 for row in scenarios if row.get("sample_gate") == "pass")
    best = max(scenarios, key=lambda row: float(row.get("gross_pnl") or 0.0))
    worst = min(scenarios, key=lambda row: float(row.get("gross_pnl") or 0.0))
    return {
        "scenario_count": len(scenarios),
        "sample_gate_pass_count": pass_count,
        "sample_gate_ready": pass_count == len(scenarios),
        "min_snapshot_gate": min_snapshot_gate,
        "min_trade_gate": min_trade_gate,
        "max_snapshot_count": max(int(row.get("snapshot_count") or 0) for row in scenarios),
        "max_trade_count": max(int(row.get("trade_count") or 0) for row in scenarios),
        "tenors_with_trades": [str(row.get("tenor")) for row in scenarios if int(row.get("trade_count") or 0) > 0],
        "best_gross_pnl_tenor": best.get("tenor"),
        "worst_gross_pnl_tenor": worst.get("tenor"),
        "cost_sensitivity": {
            "cost_per_trade": cost_per_trade,
            "slippage_vol_pts": slippage_vol_pts,
            "effective_cost_per_trade": effective_cost,
            "control_note": "Synthetic fills only; cost sensitivity is illustrative and not executable economics.",
        },
    }


def _render_robustness_section(metrics: dict[str, Any]) -> str:
    cost = metrics.get("cost_sensitivity") or {}
    return "\n".join(
        [
            "## Robustness Metrics",
            "",
            f"- Sample gate pass count: {metrics.get('sample_gate_pass_count', 0)} / {metrics.get('scenario_count', 0)}",
            f"- Sample gate ready: {metrics.get('sample_gate_ready')}",
            f"- Minimum gates: {metrics.get('min_snapshot_gate')} snapshots and {metrics.get('min_trade_gate')} synthetic trades per tenor",
            f"- Max observed sample: {metrics.get('max_snapshot_count')} snapshots / {metrics.get('max_trade_count')} synthetic trades",
            f"- Tenors with synthetic trades: {', '.join(metrics.get('tenors_with_trades') or []) or 'none'}",
            f"- Best gross PnL tenor: {metrics.get('best_gross_pnl_tenor') or 'n/a'}",
            f"- Worst gross PnL tenor: {metrics.get('worst_gross_pnl_tenor') or 'n/a'}",
            f"- Cost sensitivity: ${float(cost.get('cost_per_trade') or 0):,.2f} explicit cost + {float(cost.get('slippage_vol_pts') or 0):.2f} vol pts slippage = ${float(cost.get('effective_cost_per_trade') or 0):,.2f} effective cost/trade",
            f"- Control: {cost.get('control_note') or 'Synthetic fills only; not executable economics.'}",
            "",
        ]
    )


def run_multi_tenor_backtests(
    manifest_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    *,
    tenors: tuple[str, ...] = ("1d", "7d", "30d"),
    threshold_vol_pts: float = 5.0,
    notional_vega: float = 10_000.0,
    cost_per_trade: float = 250.0,
    slippage_vol_pts: float = 0.0,
) -> dict[str, Any]:
    rows = load_manifest_rows(manifest_path)
    scenarios: list[dict[str, Any]] = []
    sections: list[str] = [
        "# BTC Vol Spread Backtest — Multi-Tenor Research Pack",
        "",
        f"**Evidence status:** `{EVIDENCE_STATUS}`",
        "",
        "Research evidence only. All modeled results use screen-only historical marks and synthetic fills; not executable economics.",
        "",
    ]
    effective_cost = float(cost_per_trade) + abs(float(slippage_vol_pts)) * float(notional_vega)
    for tenor in tenors:
        snapshots = build_spread_snapshots_from_manifest_rows(rows, tenor=tenor)
        result = run_vol_spread_backtest(
            snapshots,
            threshold_vol_pts=threshold_vol_pts,
            notional_vega=notional_vega,
            cost_per_trade=effective_cost,
            spread_field=f"spread_{tenor}_vol_pts",
        )
        scenario = _scenario_summary(
            tenor=tenor,
            result=result,
            snapshot_count=len(snapshots),
            threshold_vol_pts=threshold_vol_pts,
            notional_vega=notional_vega,
            cost_per_trade=cost_per_trade,
            slippage_vol_pts=slippage_vol_pts,
        )
        scenarios.append(scenario)
        sections.append(render_backtest_markdown(run_id=f"{tenor}-spread-threshold-{threshold_vol_pts:g}", result=result, tenor=tenor, snapshot_count=len(snapshots), sample_gate=scenario["sample_gate"]))
    metrics = _robustness_metrics(
        scenarios,
        cost_per_trade=cost_per_trade,
        slippage_vol_pts=slippage_vol_pts,
        notional_vega=notional_vega,
    )
    sections.append(_render_robustness_section(metrics))
    payload = {
        "ok": True,
        "evidence_status": EVIDENCE_STATUS,
        "controls": [
            "strict chronology enforced",
            "strategy sees prior history only",
            "synthetic fills; not executable economics",
        ],
        "scenarios": scenarios,
        "robustness_metrics": metrics,
    }
    out = Path(output_path)
    summary = Path(summary_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {**payload, "output_path": str(out), "summary_path": str(summary)}


def run_manifest_backtest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    tenor: str = "7d",
    threshold_vol_pts: float = 5.0,
    notional_vega: float = 10_000.0,
    cost_per_trade: float = 250.0,
) -> dict[str, Any]:
    rows = load_manifest_rows(manifest_path)
    snapshots = build_spread_snapshots_from_manifest_rows(rows, tenor=tenor)
    result = run_vol_spread_backtest(
        snapshots,
        threshold_vol_pts=threshold_vol_pts,
        notional_vega=notional_vega,
        cost_per_trade=cost_per_trade,
        spread_field=f"spread_{tenor}_vol_pts",
    )
    run_id = f"{tenor}-spread-threshold-{threshold_vol_pts:g}"
    markdown = render_backtest_markdown(run_id=run_id, result=result, tenor=tenor, snapshot_count=len(snapshots))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return {
        "ok": True,
        "run_id": run_id,
        "output_path": str(out),
        "snapshot_count": len(snapshots),
        "trade_count": result.trade_count,
        "gross_pnl": result.gross_pnl,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "evidence_status": result.evidence_status,
    }
