import hashlib
import json
from pathlib import Path

from institutional_btc_vol.site_data import build_site_data, write_site_data


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_site_data_uses_latest_run_and_preserves_evidence_controls(tmp_path):
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "latest.md"
    bundle = data_dir / "normalized" / "run2" / "bundle.zip"
    manifest = data_dir / "normalized" / "run2" / "evidence_manifest.json"
    ledger = data_dir / "normalized" / "run2" / "candidate_triage.jsonl"
    quote_ledger = data_dir / "normalized" / "run2" / "quote_evidence.jsonl"
    quote_record = {
        "rfq_id": "rfq-site-001",
        "candidate_id": "candidate-site-001",
        "structure": "IBIT 7D ATM vs Deribit 7D ATM",
        "as_of_cst": "2026-05-15 12:20:00 CDT",
        "counterparty": "Dealer A",
        "venue": "manual-rfq",
        "instrument": "IBIT / Deribit comparison",
        "side": "two-way",
        "notional_btc": 20,
        "bid_iv": 0.4,
        "ask_iv": 0.44,
        "mid_iv": 0.42,
        "execution_confidence": "manual-indicative",
        "source_confidence": "manual-indicative-rfq",
        "status": "indicative",
        "evidence_ref": "internal://rfq/site-001",
        "notes": "Indicative only.",
    }
    for path, content in (
        (report, "# Report"),
        (bundle, "zipbytes"),
        (manifest, json.dumps({"artifacts": []})),
        (ledger, json.dumps({"rank": 1, "candidate": "IBIT 7D ATM vs Deribit 7D ATM", "priority": "high", "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}) + "\n"),
        (quote_ledger, json.dumps(quote_record) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_jsonl(
        data_dir / "run_manifest.jsonl",
        [
            {"run_id": "run1", "as_of_cst": "old", "btc_spot": 79000},
            {
                "run_id": "run2",
                "as_of_cst": "2026-05-15 12:17:52 CDT",
                "btc_spot": 79682.66,
                "btc_per_share": 0.0005679118586151414,
                "quality_score": 100,
                "quality_grade": "green",
                "freshness_grade": "green",
                "cme_rows": 1605,
                "dislocations": 7,
                "quote_review_candidates": 2,
                "evidence_bundle_sha256": "abc123",
                "report_path": str(report),
                "evidence_bundle_path": str(bundle),
                "evidence_manifest_path": str(manifest),
                "candidate_ledger_path": str(ledger),
                "quote_evidence_ledger_path": str(quote_ledger),
            },
        ],
    )

    site_data = build_site_data(data_dir)

    assert site_data["positioning"] == "BTC Treasury & Miner Hedging Desk"
    assert site_data["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert site_data["latest_run"]["run_id"] == "run2"
    assert site_data["latest_run"]["quality_grade"] == "green"
    assert site_data["latest_run"]["freshness_grade"] == "green"
    assert site_data["latest_run"]["evidence_bundle_sha256"] == "abc123"
    assert site_data["top_candidates"][0]["candidate"] == "IBIT 7D ATM vs Deribit 7D ATM"
    assert site_data["top_candidates"][0]["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert site_data["case_studies"]["treasury"]["hedged_btc"] == 350
    assert site_data["case_studies"]["treasury"]["quote_control"] == "Premium and executable levels require two-counterparty quote verification."
    assert site_data["case_studies"]["miner"]["hedged_monthly_btc"] == 60
    assert site_data["case_studies"]["miner"]["quote_control"] == "Indicative economics require quote verification before investor/client use."
    assert site_data["quote_verification_board"]["title"] == "Quote Verification Demo Board"
    assert site_data["quote_verification_board"]["control"] == "Manual demo workflow only; no RFQ is sent and no executable quote is implied."
    assert site_data["quote_verification_board"]["rows"][0]["stage"] == "screen_only"
    assert site_data["quote_verification_board"]["summary"]["screen_only"] == 1
    assert site_data["latest_run"]["quote_evidence_ledger_path"] == str(quote_ledger)
    assert site_data["quote_evidence_ledger"]["valid_count"] == 1
    assert site_data["quote_evidence_ledger"]["summary"]["quote_verified_records"] == 0
    assert site_data["quote_evidence_ledger"]["summary"]["manual_indicative_records"] == 1
    assert site_data["quote_evidence_ledger"]["summary"]["candidates"][0]["stage"] == "one-demo-indicative-quote"
    assert site_data["quote_evidence_ledger"]["summary"]["candidates"][0]["publishability"] == "not-investor-publishable"
    gate = site_data["institutional_readiness_gate"]
    assert gate["status"] == "not-ready"
    assert gate["summary"] == "0/4 readiness gates passed"
    assert "licensed historical source coverage incomplete" in gate["blockers"]
    assert "backtest sample gate insufficient" in gate["blockers"]
    assert "two-counterparty quote evidence missing" in gate["blockers"]
    assert "counsel-approved wrapper missing" in gate["blockers"]
    assert gate["next_actions"][0] == "Replace IBIT/Deribit fixtures and missing groups with licensed/replay-ready source manifests."
    assert gate["next_actions"][1] == "Accumulate enough point-in-time replay observations for sample-gate pass thresholds."
    assert gate["next_actions"][2] == "Capture two distinct external indicative quote records for at least one candidate."
    assert gate["next_actions"][3] == "Obtain counsel-approved business/legal wrapper before external investor/client use."
    intake = site_data["licensed_source_intake_contract"]
    assert intake["status"] == "not-ready"
    assert intake["evidence_status"] == "SCREEN-ONLY · INTAKE CONTRACT · NOT EXECUTABLE"
    assert len(intake["required_sources"]) == 6
    assert intake["required_sources"][0]["source_group"] == "IBIT options history"
    assert intake["required_sources"][0]["accepted_formats"] == ["csv", "jsonl", "parquet"]
    assert intake["required_sources"][0]["required_fields"] == ["available_ts", "expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest", "source_ref"]
    assert intake["validation_gates"] == [
        "No future available_ts relative to decision_ts",
        "Raw file SHA-256 captured before normalization",
        "Provider/license label present for every source group",
        "Fixture/manual_fixture sources cannot satisfy readiness",
        "Rows with crossed/wide/missing markets remain diagnostic, not executable",
    ]
    validation = site_data["source_intake_validation"]
    assert validation["ready"] is False
    assert validation["manifest_path"] == "missing"
    assert validation["covered_source_groups"] == 0
    assert "source intake manifest missing" in validation["blockers"]
    assert "execute trade" not in json.dumps(site_data).lower()
    assert site_data["latest_run"]["coverage_completeness"] == "Current screen/vendor captures available: Deribit + IBIT + CME Databento"
    assert site_data["latest_run"]["coverage_completeness_label"] == "Current screen-source availability"
    assert site_data["latest_run"]["cme_rows"] == 1605


def test_build_site_data_keeps_cme_unavailable_when_no_databento_rows(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(
        data_dir / "run_manifest.jsonl",
        [{"run_id": "run-no-cme", "as_of_cst": "now", "cme_rows": 0}],
    )

    site_data = build_site_data(data_dir)

    assert site_data["latest_run"]["coverage_completeness"] == "Current screen/vendor captures partial: CME unavailable"


def test_build_site_data_splits_current_availability_from_licensed_readiness(tmp_path):
    data_dir = tmp_path / "data"
    historical_dir = tmp_path / "historical"
    run_id = "btcvol-test-current"
    raw_dir = data_dir / "raw" / run_id
    raw_dir.mkdir(parents=True)
    raw_files = {
        "nasdaq_ibit_option_chain.json": '{"data": []}',
        "deribit_book_summary_btc_options.json": '{"result": []}',
        "databento_cme_btc_options_bbo_1m.csv": "instrument_id,bid_px_00,ask_px_00\n1,1,2\n",
        "ishares_ibit_holdings_fallback_cached.csv": "Ticker,Name,Shares\nBTC,Bitcoin,1\n",
    }
    for name, content in raw_files.items():
        (raw_dir / name).write_text(content, encoding="utf-8")
    fred = historical_dir / "source_audit" / "fred_rates_sofr_dgs1_dgs3mo_current.csv"
    fred.parent.mkdir(parents=True)
    fred.write_text("observation_date,SOFR,DGS1,DGS3MO\n2026-05-18,3.55,3.80,3.70\n", encoding="utf-8")
    _write_jsonl(
        data_dir / "run_manifest.jsonl",
        [
            {
                "run_id": run_id,
                "as_of_cst": "2026-05-18 12:30:00 CDT",
                "btc_spot": 77080.70,
                "btc_per_share": 0.0005679,
                "cme_rows": 1135,
                "deribit_atm_rows": 11,
                "ibit_atm_rows": 6,
            }
        ],
    )

    site_data = build_site_data(data_dir, historical_dir=historical_dir)

    availability = site_data["current_source_availability"]
    assert availability["summary"] == "6/6 current source groups available"
    assert availability["ready_for_internal_diligence"] is True
    assert availability["ready_for_investment_or_client_use"] is False
    assert {row["label"]: row["status"] for row in availability["groups"]} == {
        "IBIT options": "available",
        "Deribit options": "available",
        "CME Bitcoin options": "available",
        "BTC reference": "available",
        "IBIT holdings": "available",
        "Rates/fee curves": "available",
    }
    assert availability["groups"][2]["provider"] == "Databento CME"
    assert availability["groups"][2]["row_count"] == 1135
    assert availability["groups"][2]["sha256"] == hashlib.sha256(raw_files["databento_cme_btc_options_bbo_1m.csv"].encode()).hexdigest()
    assert availability["control_note"] == "Current captured sources can support internal diligence, but they do not clear licensed historical readiness, quote verification, legal approval, or executable-economics gates."
    assert site_data["institutional_readiness_gate"]["status"] == "not-ready"


def test_write_site_data_outputs_json_file(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run1", "as_of_cst": "now"}])

    output = write_site_data(data_dir, tmp_path / "site" / "site-data.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["latest_run"]["run_id"] == "run1"
    assert payload["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"


def test_build_site_data_computes_bundle_hash_when_manifest_row_omits_it(tmp_path):
    data_dir = tmp_path / "data"
    bundle = data_dir / "normalized" / "run3" / "bundle.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("bundle contents", encoding="utf-8")
    _write_jsonl(
        data_dir / "run_manifest.jsonl",
        [{"run_id": "run3", "as_of_cst": "now", "evidence_bundle_path": str(bundle)}],
    )

    site_data = build_site_data(data_dir)

    assert site_data["latest_run"]["evidence_bundle_sha256"] == hashlib.sha256(b"bundle contents").hexdigest()


def test_build_site_data_exposes_latest_backtest_research_evidence(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-backtest", "as_of_cst": "now"}])
    older = tmp_path / "backtests" / "vol-spread-backtest-7d-old.md"
    latest = tmp_path / "backtests" / "vol-spread-backtest-7d-latest.md"
    older.parent.mkdir(parents=True, exist_ok=True)
    older.write_text("# Old\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\nsynthetic fills\n", encoding="utf-8")
    latest.write_text(
        "# BTC Vol Spread Backtest — 7d-spread-threshold-5\n"
        "**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`\n\n"
        "synthetic fills; not executable economics\n\n"
        "- Tenor: `7d`\n"
        "- Snapshot count: 14\n"
        "- Trade count: 2\n"
        "- Gross PnL: $54,000.00\n"
        "- Max drawdown: $0.00\n"
        "- Win rate: 100.0%\n",
        encoding="utf-8",
    )

    site_data = build_site_data(data_dir, backtests_dir=latest.parent)

    backtest = site_data["backtest_research"]
    assert backtest["available"] is True
    assert backtest["path"] == str(latest)
    assert backtest["sha256"] == hashlib.sha256(latest.read_bytes()).hexdigest()
    assert backtest["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"
    assert backtest["tenor"] == "7d"
    assert backtest["snapshot_count"] == 14
    assert backtest["trade_count"] == 2
    assert backtest["gross_pnl"] == 54000.0
    assert backtest["sample_gate"] == "insufficient-history"
    assert "synthetic fills" in backtest["control_note"]


def test_build_site_data_exposes_multi_tenor_backtest_summary_json(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-backtest", "as_of_cst": "now"}])
    backtests_dir = tmp_path / "backtests"
    report = backtests_dir / "vol-spread-backtest-multitenor-v1.md"
    summary = backtests_dir / "vol-spread-backtest-multitenor-v1.json"
    backtests_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Multi tenor backtest\n"
        "**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`\n\n"
        "synthetic fills; not executable economics\n\n"
        "- Tenor: `1d`\n"
        "- Snapshot count: 14\n"
        "- Trade count: 0\n"
        "- Gross PnL: $0.00\n"
        "- Max drawdown: $0.00\n"
        "- Win rate: 0.0%\n",
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "ok": True,
                "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
                "controls": [
                    "strict chronology enforced",
                    "strategy sees prior history only",
                    "synthetic fills; not executable economics",
                ],
                "scenarios": [
                    {"tenor": "1d", "snapshot_count": 14, "trade_count": 0, "gross_pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "sample_gate": "insufficient-history", "cost_per_trade": 250, "slippage_vol_pts": 0.25, "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"},
                    {"tenor": "7d", "snapshot_count": 14, "trade_count": 2, "gross_pnl": 49000.0, "max_drawdown": 2050.0, "win_rate": 0.5, "sample_gate": "insufficient-history", "cost_per_trade": 250, "slippage_vol_pts": 0.25, "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"},
                    {"tenor": "30d", "snapshot_count": 14, "trade_count": 0, "gross_pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "sample_gate": "insufficient-history", "cost_per_trade": 250, "slippage_vol_pts": 0.25, "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"},
                ],
                "robustness_metrics": {
                    "scenario_count": 3,
                    "sample_gate_pass_count": 0,
                    "sample_gate_ready": False,
                    "min_snapshot_gate": 30,
                    "min_trade_gate": 20,
                    "max_snapshot_count": 14,
                    "max_trade_count": 2,
                    "tenors_with_trades": ["7d"],
                    "best_gross_pnl_tenor": "7d",
                    "worst_gross_pnl_tenor": "1d",
                    "cost_sensitivity": {
                        "cost_per_trade": 250,
                        "slippage_vol_pts": 0.25,
                        "effective_cost_per_trade": 2750.0,
                        "control_note": "Synthetic fills only; cost sensitivity is illustrative and not executable economics.",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    site_data = build_site_data(data_dir, backtests_dir=backtests_dir)

    backtest = site_data["backtest_research"]
    assert backtest["summary_json_path"] == str(summary)
    assert backtest["summary_json_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert [row["tenor"] for row in backtest["scenarios"]] == ["1d", "7d", "30d"]
    assert backtest["scenarios"][1]["gross_pnl"] == 49000.0
    assert backtest["scenarios"][1]["trade_count"] == 2
    assert backtest["scenarios"][1]["sample_gate"] == "insufficient-history"
    assert backtest["controls"] == [
        "strict chronology enforced",
        "strategy sees prior history only",
        "synthetic fills; not executable economics",
    ]
    assert backtest["robustness_metrics"]["sample_gate_pass_count"] == 0
    assert backtest["robustness_metrics"]["sample_gate_ready"] is False
    assert backtest["robustness_metrics"]["tenors_with_trades"] == ["7d"]
    assert backtest["robustness_metrics"]["cost_sensitivity"]["effective_cost_per_trade"] == 2750.0


def test_build_site_data_exposes_historical_source_diagnostics(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-source-diag", "as_of_cst": "now"}])
    raw = tmp_path / "historical" / "raw" / "ibit.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("timestamp,symbol,bid,ask\n2025-01-02T15:30:00Z,IBIT,1,2\n", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = tmp_path / "historical" / "manifests" / "historical-source-manifest-v1.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "ok": True,
                "manifest_type": "btc_vol_historical_source_manifest",
                "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "ibit_opra_2025_01_02",
                        "source_name": "IBIT OPRA historical options",
                        "provider": "OPRA vendor export",
                        "venue": "OPRA",
                        "instrument_scope": "IBIT options",
                        "license_label": "licensed_vendor_api_databento",
                        "redistribution": "internal_only",
                        "execution_confidence": "screen_only_not_executable",
                        "backtest_status": "backtest_only_not_executable",
                        "status": "available",
                        "coverage_start": "2025-01-02T15:30:00Z",
                        "coverage_end": "2025-01-02T15:30:00Z",
                        "raw_path": str(raw),
                        "bytes": raw.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    site_data = build_site_data(data_dir, historical_dir=tmp_path / "historical")

    diagnostics = site_data["historical_source_diagnostics"]
    assert diagnostics["available"] is True
    assert diagnostics["manifest_path"] == str(manifest)
    assert diagnostics["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert diagnostics["source_count"] == 1
    assert diagnostics["required_source_labels"] == ["IBIT", "Deribit", "CME", "BTC reference", "IBIT holdings", "Rates/fees"]
    assert diagnostics["coverage_summary"] == "1/6 licensed source groups covered · 0 fixture-only · 5 missing"
    assert diagnostics["coverage_ready"] is False
    assert diagnostics["missing_required_source_labels"] == ["Deribit", "CME", "BTC reference", "IBIT holdings", "Rates/fees"]
    assert diagnostics["sources"][0]["source_name"] == "IBIT OPRA historical options"
    assert diagnostics["sources"][0]["raw_path"].endswith("ibit.csv")
    assert diagnostics["sources"][0]["sha256"] == digest
    assert diagnostics["sources"][0]["coverage_labels"] == ["IBIT"]
    assert diagnostics["sources"][0]["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"
    assert diagnostics["coverage_matrix"] == [
        {"label": "IBIT", "status": "covered", "source_count": 1},
        {"label": "Deribit", "status": "missing", "source_count": 0},
        {"label": "CME", "status": "missing", "source_count": 0},
        {"label": "BTC reference", "status": "missing", "source_count": 0},
        {"label": "IBIT holdings", "status": "missing", "source_count": 0},
        {"label": "Rates/fees", "status": "missing", "source_count": 0},
    ]
    assert diagnostics["source_tracker_summary"] == "1 covered · 5 missing · 0 fixture-only"
    tracker = diagnostics["source_coverage_tracker"]
    assert tracker[0] == {
        "label": "IBIT",
        "status": "covered",
        "source_count": 1,
        "provider": "OPRA vendor export",
        "coverage_start": "2025-01-02T15:30:00Z",
        "coverage_end": "2025-01-02T15:30:00Z",
        "row_count": None,
        "sha256": digest,
        "license_label": "licensed_vendor_api_databento",
        "redistribution": "internal_only",
        "execution_confidence": "screen_only_not_executable",
        "blocker": "No blocker captured; verify license scope and expand date coverage before investment use.",
        "next_action": "Expand IBIT historical coverage window and retain manifest hashes for replay.",
    }
    assert tracker[1]["label"] == "Deribit"
    assert tracker[1]["status"] == "missing"
    assert tracker[1]["blocker"] == "No Deribit source captured in latest historical manifest."
    assert tracker[1]["next_action"] == "Load Deribit historical options replay source with raw-file hash and coverage dates."
    assert tracker[2]["label"] == "CME"
    assert tracker[2]["next_action"] == "Configure licensed CME/Databento historical source; keep screen-only until quote/trade evidence exists."
    assert diagnostics["control_note"] == "Historical backtests use hashed point-in-time sources; synthetic fills only; not executable economics."


def test_build_site_data_validates_source_intake_manifest(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-source-intake", "as_of_cst": "2026-05-03 00:00:00 UTC"}])
    historical_dir = tmp_path / "historical"
    manifest = historical_dir / "source_intake_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "decision_ts": "2026-05-03T00:00:00+00:00",
                "sources": [
                    {
                        "source_group": "BTC reference history",
                        "provenance": "manual_fixture",
                        "license_label": "fixture",
                        "format": "csv",
                        "raw_sha256": "d" * 64,
                        "fields": ["available_ts", "btc_usd", "venue_or_index", "source_ref"],
                        "row_count": 10,
                        "available_end": "2026-05-02T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    site_data = build_site_data(data_dir, historical_dir=historical_dir)

    validation = site_data["source_intake_validation"]
    assert validation["ready"] is False
    assert validation["manifest_path"] == str(manifest)
    assert validation["covered_source_groups"] == 0
    assert validation["required_source_groups"] == 6
    assert "BTC reference history: fixture/manual_fixture sources cannot satisfy readiness" in validation["blockers"]
    assert "IBIT options history: source group missing" in validation["blockers"]
    assert validation["evidence_status"] == "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE"


def test_source_coverage_matrix_counts_fixture_only_separately(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-source-fixture", "as_of_cst": "now"}])
    hist = tmp_path / "historical"
    manifest = hist / "manifests" / "historical-source-manifest-fixtures.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
                "source_count": 2,
                "sources": [
                    {
                        "source_id": "ibit_manual_fixture",
                        "source_name": "IBIT fixture historical options",
                        "provider": "manual_fixture",
                        "venue": "OPRA",
                        "instrument_scope": "IBIT options",
                        "license_label": "fixture",
                        "coverage_start": "2026-05-01T00:00:00Z",
                        "coverage_end": "2026-05-02T00:00:00Z",
                        "sha256": "a" * 64,
                    },
                    {
                        "source_id": "deribit_demo_fixture",
                        "source_name": "Deribit fixture historical options",
                        "provider": "demo_fixture",
                        "venue": "Deribit",
                        "instrument_scope": "Deribit BTC options",
                        "license_label": "manual_fixture",
                        "coverage_start": "2026-05-01T00:00:00Z",
                        "coverage_end": "2026-05-02T00:00:00Z",
                        "sha256": "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    site_data = build_site_data(data_dir, historical_dir=hist)

    diagnostics = site_data["historical_source_diagnostics"]
    assert diagnostics["coverage_summary"] == "0/6 licensed source groups covered · 2 fixture-only · 4 missing"
    assert diagnostics["coverage_ready"] is False
    assert diagnostics["missing_required_source_labels"] == ["IBIT", "Deribit", "CME", "BTC reference", "IBIT holdings", "Rates/fees"]
    assert diagnostics["coverage_matrix"][0] == {"label": "IBIT", "status": "fixture-only", "source_count": 1}
    assert diagnostics["coverage_matrix"][1] == {"label": "Deribit", "status": "fixture-only", "source_count": 1}
    assert site_data["institutional_readiness_gate"]["status"] == "not-ready"


def test_build_site_data_suppresses_case_studies_when_btc_spot_missing(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "run_manifest.jsonl", [{"run_id": "run-no-spot", "as_of_cst": "now"}])

    site_data = build_site_data(data_dir)

    assert site_data["latest_run"]["btc_spot"] is None
    assert site_data["case_studies"]["available"] is False
    assert "spot unavailable" in site_data["case_studies"]["control_note"]
    assert "treasury" not in site_data["case_studies"]
    assert "miner" not in site_data["case_studies"]
