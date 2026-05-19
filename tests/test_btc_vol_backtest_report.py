from institutional_btc_vol.backtest_report import (
    build_spread_snapshots_from_manifest_rows,
    render_backtest_markdown,
    run_multi_tenor_backtests,
)
from institutional_btc_vol.backtester import run_vol_spread_backtest


def test_build_spread_snapshots_preserves_only_decision_time_observables():
    rows = [
        {"run_id": "r1", "as_of_cst": "2026-05-17 10:00:00 CDT", "spread_7d_vol_pts": 6.0},
        {"run_id": "r2", "as_of_cst": "2026-05-17 11:00:00 CDT", "spread_7d_vol_pts": 4.0},
        {"run_id": "r3", "as_of_cst": "2026-05-17 12:00:00 CDT", "spread_7d_vol_pts": -5.5},
    ]

    snapshots = build_spread_snapshots_from_manifest_rows(rows, tenor="7d")

    assert len(snapshots) == 3
    assert snapshots[0].data["run_id"] == "r1"
    assert snapshots[0].data["spread_7d_vol_pts"] == 6.0
    assert "next_spread_7d_vol_pts" not in snapshots[0].data
    assert "next_run_id" not in snapshots[0].data
    assert snapshots[2].data["run_id"] == "r3"


def test_render_backtest_markdown_pins_screen_only_research_language():
    snapshots = [
        *build_spread_snapshots_from_manifest_rows(
            [
                {"run_id": "r1", "as_of_cst": "2026-05-17 10:00:00 CDT", "spread_7d_vol_pts": 6.0},
                {"run_id": "r2", "as_of_cst": "2026-05-17 11:00:00 CDT", "spread_7d_vol_pts": 4.0},
                {"run_id": "r3", "as_of_cst": "2026-05-17 12:00:00 CDT", "spread_7d_vol_pts": -5.5},
                {"run_id": "r4", "as_of_cst": "2026-05-17 13:00:00 CDT", "spread_7d_vol_pts": -2.5},
            ],
            tenor="7d",
        )
    ]
    result = run_vol_spread_backtest(snapshots, threshold_vol_pts=5.0, notional_vega=10_000, cost_per_trade=250)

    markdown = render_backtest_markdown(
        run_id="backtest-demo",
        result=result,
        tenor="7d",
        snapshot_count=len(snapshots),
    )

    assert "# BTC Vol Spread Backtest — backtest-demo" in markdown
    assert "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE" in markdown
    assert "No future leakage" in markdown
    assert "Synthetic fills" in markdown
    assert "Trade count: 2" in markdown
    assert "Gross PnL: $49,500.00" in markdown
    assert "This is research evidence, not executable economics" in markdown


def test_run_multi_tenor_backtests_writes_summary_json_with_sample_gates(tmp_path):
    manifest = tmp_path / "run_manifest.jsonl"
    rows = [
        {
            "run_id": f"r{idx}",
            "as_of_cst": f"2026-05-17 {10 + idx:02d}:00:00 CDT",
            "spread_1d_vol_pts": 6.0 if idx % 2 == 0 else 3.0,
            "spread_7d_vol_pts": -6.0 if idx % 2 == 0 else -3.0,
            "spread_30d_vol_pts": 2.0,
        }
        for idx in range(4)
    ]
    import json

    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    markdown = tmp_path / "backtests" / "multi.md"
    summary = tmp_path / "backtests" / "multi.json"

    result = run_multi_tenor_backtests(
        manifest,
        markdown,
        summary,
        tenors=("1d", "7d", "30d"),
        threshold_vol_pts=5.0,
        notional_vega=10_000,
        cost_per_trade=250,
        slippage_vol_pts=0.25,
    )

    assert result["ok"] is True
    assert result["summary_path"] == str(summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [scenario["tenor"] for scenario in payload["scenarios"]] == ["1d", "7d", "30d"]
    assert payload["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"
    assert payload["scenarios"][0]["slippage_vol_pts"] == 0.25
    assert payload["scenarios"][0]["trade_count"] == 2
    assert payload["scenarios"][1]["trade_count"] == 2
    assert payload["scenarios"][2]["trade_count"] == 0
    assert payload["scenarios"][0]["sample_gate"] == "insufficient-history"
    assert payload["robustness_metrics"] == {
        "scenario_count": 3,
        "sample_gate_pass_count": 0,
        "sample_gate_ready": False,
        "min_snapshot_gate": 30,
        "min_trade_gate": 20,
        "max_snapshot_count": 4,
        "max_trade_count": 2,
        "tenors_with_trades": ["1d", "7d"],
        "best_gross_pnl_tenor": "1d",
        "worst_gross_pnl_tenor": "30d",
        "cost_sensitivity": {
            "cost_per_trade": 250,
            "slippage_vol_pts": 0.25,
            "effective_cost_per_trade": 2750.0,
            "control_note": "Synthetic fills only; cost sensitivity is illustrative and not executable economics.",
        },
    }
    assert "## Robustness Metrics" in markdown.read_text(encoding="utf-8")
    assert "Sample gate pass count: 0 / 3" in markdown.read_text(encoding="utf-8")
    assert "synthetic fills" in markdown.read_text(encoding="utf-8")
