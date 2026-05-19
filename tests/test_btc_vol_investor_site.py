from institutional_btc_vol.investor_site import render_investor_site_html, write_investor_site


def _site_data():
    return {
        "positioning": "BTC Treasury & Miner Hedging Desk",
        "full_positioning": "BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "run_count": 8,
        "institutional_readiness_gate": {
            "status": "not-ready",
            "summary": "0/4 readiness gates passed",
            "label": "NOT READY FOR INVESTMENT/CLIENT USE",
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            "control_note": "Readiness gates are diligence controls, not approval to trade, advise, market, or raise capital.",
            "blockers": [
                "licensed historical source coverage incomplete",
                "backtest sample gate insufficient",
                "two-counterparty quote evidence missing",
                "counsel-approved wrapper missing",
            ],
            "next_actions": [
                "Replace IBIT/Deribit fixtures and missing groups with licensed/replay-ready source manifests.",
                "Accumulate enough point-in-time replay observations for sample-gate pass thresholds.",
                "Capture two distinct external indicative quote records for at least one candidate.",
                "Obtain counsel-approved business/legal wrapper before external investor/client use.",
            ],
            "gates": [
                {"label": "Licensed historical source coverage", "passed": False, "blocker": "licensed historical source coverage incomplete", "required": "All required source groups covered by non-fixture, replay-ready sources"},
                {"label": "Backtest sample gate", "passed": False, "blocker": "backtest sample gate insufficient", "required": "At least one scenario passes minimum snapshot and synthetic-trade thresholds"},
                {"label": "Two-counterparty quote evidence", "passed": False, "blocker": "two-counterparty quote evidence missing", "required": "At least one candidate has quote-verified evidence from two distinct counterparties"},
                {"label": "Counsel-approved legal wrapper", "passed": False, "blocker": "counsel-approved wrapper missing", "required": "Approved legal/business wrapper before external investor/client use"},
            ],
        },
        "licensed_source_intake_contract": {
            "status": "not-ready",
            "evidence_status": "SCREEN-ONLY · INTAKE CONTRACT · NOT EXECUTABLE",
            "control_note": "Defines the licensed/replay-ready source package required before historical coverage can pass readiness gates. It does not create live data access or executable economics.",
            "validation_result": {
                "ready": False,
                "covered_source_groups": 1,
                "required_source_groups": 6,
                "blockers": ["BTC reference history: fixture/manual_fixture sources cannot satisfy readiness"],
                "evidence_status": "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE",
            },
            "required_sources": [
                {
                    "source_group": "IBIT options history",
                    "accepted_formats": ["csv", "jsonl", "parquet"],
                    "required_fields": ["available_ts", "expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest", "source_ref"],
                    "provider_examples": ["OPRA/vendor export", "broker historical option chain"],
                }
            ],
            "validation_gates": [
                "No future available_ts relative to decision_ts",
                "Fixture/manual_fixture sources cannot satisfy readiness",
            ],
        },
        "current_source_availability": {
            "summary": "6/6 current source groups available",
            "evidence_status": "SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE",
            "control_note": "Current captured sources can support internal diligence, but they do not clear licensed historical readiness, quote verification, legal approval, or executable-economics gates.",
            "ready_for_internal_diligence": True,
            "ready_for_investment_or_client_use": False,
            "groups": [
                {"label": "CME Bitcoin options", "status": "available", "provider": "Databento CME", "license_label": "licensed_vendor_api_databento", "row_count": 1135, "sha256": "a" * 64},
                {"label": "Rates/fee curves", "status": "available", "provider": "FRED rates CSV", "license_label": "public_reference_rates", "row_count": 1, "sha256": "b" * 64},
            ],
        },
        "latest_run": {
            "run_id": "btcvol-20260515-121752",
            "as_of_cst": "2026-05-15 12:17:52 CDT",
            "btc_spot": 79682.66,
            "btc_per_share": 0.0005679118586151414,
            "quality_score": 100,
            "quality_grade": "green",
            "freshness_grade": "green",
            "dislocations": 7,
            "quote_review_candidates": 2,
            "evidence_bundle_sha256": "b81186890ba7163c92b13407552d6ef83a83b32f98a0f8885187acd5c64c31ef",
            "report_path": "artifacts/institutional/data/reports/latest.md",
            "dashboard_path": "artifacts/institutional/dashboard/index.html",
            "evidence_bundle_path": "artifacts/institutional/data/normalized/run/bundle.zip",
            "evidence_manifest_path": "artifacts/institutional/data/normalized/run/evidence_manifest.json",
            "candidate_ledger_path": "artifacts/institutional/data/normalized/run/candidate_triage.jsonl",
            "quote_evidence_ledger_path": "artifacts/institutional/data/normalized/run/quote_evidence.jsonl",
        },
        "top_candidates": [
            {
                "rank": 1,
                "candidate": "IBIT 0D ATM vs Deribit 1D ATM",
                "priority": "high",
                "direction": "IBIT cheap vs Deribit",
                "gross_iv_diff_vol_pts": -13.13,
                "recommended_workflow": "draft two-counterparty indicative RFQ review (internal only)",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            },
            {
                "rank": 2,
                "candidate": "IBIT 11D ATM vs Deribit 14D ATM",
                "priority": "medium",
                "direction": "IBIT cheap vs Deribit",
                "gross_iv_diff_vol_pts": -3.76,
                "recommended_workflow": "add to RFQ review queue",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            }
        ],
        "quote_evidence_ledger": {
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            "valid_count": 2,
            "invalid_count": 1,
            "records": [
                {
                    "rfq_id": "rfq-a",
                    "candidate_id": "candidate-a",
                    "structure": "IBIT 5D ATM vs Deribit 4D ATM",
                    "counterparty": "Dealer <A>",
                    "bid_iv": 0.41,
                    "ask_iv": 0.45,
                    "mid_iv": 0.43,
                    "spread_vol_pts": 4.0,
                    "execution_confidence": "manual-indicative",
                    "source_confidence": "manual-indicative-rfq",
                    "status": "indicative",
                    "publishability": "not-investor-publishable",
                    "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                }
            ],
            "summary": {
                "quote_verified_records": 0,
                "manual_indicative_records": 2,
                "trade_verified_records": 0,
                "invalid_records": 1,
                "candidate_count": 1,
                "quote_verified_candidates": 0,
                "trade_verified_candidates": 0,
                "controls": [
                    "Two distinct counterparties required for quote-verified candidate status",
                    "same-dealer duplicate quotes cannot promote a candidate",
                    "Quote evidence remains not investor-publishable until legal wrapper approval",
                ],
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "structure": "IBIT 5D ATM vs Deribit 4D ATM",
                        "indicative_quote_count": 2,
                        "trade_verified_count": 0,
                        "stage": "two-demo-indicative-quotes",
                        "publishability": "not-investor-publishable",
                        "avg_mid_iv": 43.5,
                        "counterparties": ["Dealer A", "Dealer B"],
                        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                    }
                ],
            },
        },
        "backtest_research": {
            "available": True,
            "path": "artifacts/institutional/backtests/vol-spread-backtest-multitenor-v1.md",
            "summary_json_path": "artifacts/institutional/backtests/vol-spread-backtest-multitenor-v1.json",
            "bytes": 2048,
            "sha256": "c0ffee90ba7163c92b13407552d6ef83a83b32f98a0f8885187acd5c64c31ef",
            "summary_json_sha256": "baddad90ba7163c92b13407552d6ef83a83b32f98a0f8885187acd5c64c31ef",
            "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
            "tenor": "7d",
            "snapshot_count": 14,
            "trade_count": 2,
            "gross_pnl": 54000.0,
            "max_drawdown": 0.0,
            "win_rate": 1.0,
            "sample_gate": "insufficient-history",
            "control_note": "Point-in-time research only using synthetic fills; not executable economics and not an investment conclusion.",
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
        },
        "historical_source_diagnostics": {
            "available": True,
            "coverage_ready": False,
            "source_coverage_tracker": [
                {"label": "IBIT", "status": "fixture-only"},
                {"label": "Deribit", "status": "fixture-only"},
            ],
            "manifest_path": "artifacts/institutional/historical/manifests/historical-source-manifest-fixtures-v1.json",
            "manifest_sha256": "238ace299e2a2d6af4f8ecc223149346a876d0184df7fca40fb0ae052bf3d939",
            "source_count": 2,
            "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
            "control_note": "Historical backtests use hashed point-in-time sources; synthetic fills only; not executable economics.",
            "required_source_labels": ["IBIT", "Deribit", "CME", "BTC reference", "IBIT holdings", "Rates/fees"],
            "coverage_summary": "2/6 required source groups covered",
            "coverage_ready": False,
            "missing_required_source_labels": ["CME", "BTC reference", "IBIT holdings", "Rates/fees"],
            "coverage_matrix": [
                {"label": "IBIT", "status": "covered", "source_count": 1},
                {"label": "Deribit", "status": "covered", "source_count": 1},
                {"label": "CME", "status": "missing", "source_count": 0},
                {"label": "BTC reference", "status": "missing", "source_count": 0},
                {"label": "IBIT holdings", "status": "missing", "source_count": 0},
                {"label": "Rates/fees", "status": "missing", "source_count": 0},
            ],
            "source_tracker_summary": "2 covered · 4 missing · 1 fixture-only",
            "source_coverage_tracker": [
                {
                    "label": "IBIT",
                    "status": "fixture-only",
                    "source_count": 1,
                    "provider": "manual_fixture",
                    "coverage_start": "2025-01-02T15:30:00Z",
                    "coverage_end": "2025-01-02T15:30:00Z",
                    "row_count": 493,
                    "sha256": "a803cdd9ba7804417eeb2a97af06064db7757b0ec919fb9acc4bfc0b89acd430",
                    "license_label": "manual_fixture",
                    "redistribution": "internal_only",
                    "execution_confidence": "screen_only_not_executable",
                    "blocker": "Fixture source; replace with licensed historical export before investment use.",
                    "next_action": "Expand IBIT historical coverage window and retain manifest hashes for replay.",
                },
                {
                    "label": "Deribit",
                    "status": "covered",
                    "source_count": 1,
                    "provider": "Deribit public API replay",
                    "coverage_start": "2025-01-02T15:30:00Z",
                    "coverage_end": "2025-01-02T15:30:00Z",
                    "row_count": 936,
                    "sha256": "238ace299e2a2d6af4f8ecc223149346a876d0184df7fca40fb0ae052bf3d939",
                    "license_label": "public_api_screen_data",
                    "redistribution": "internal_only",
                    "execution_confidence": "screen_only_not_executable",
                    "blocker": "No blocker captured; verify license scope and expand date coverage before investment use.",
                    "next_action": "Load Deribit historical options replay source with raw-file hash and coverage dates.",
                },
                {
                    "label": "CME",
                    "status": "missing",
                    "source_count": 0,
                    "provider": "missing",
                    "coverage_start": None,
                    "coverage_end": None,
                    "row_count": None,
                    "sha256": None,
                    "license_label": "missing",
                    "redistribution": "internal_only",
                    "execution_confidence": "screen_only_not_executable",
                    "blocker": "No CME source captured in latest historical manifest.",
                    "next_action": "Configure licensed CME/Databento historical source; keep screen-only until quote/trade evidence exists.",
                },
            ],
            "sources": [
                {
                    "source_name": "IBIT OPRA historical options fixture",
                    "provider": "manual_fixture",
                    "venue": "OPRA",
                    "instrument_scope": "IBIT options",
                    "license_label": "manual_fixture",
                    "redistribution": "internal_only",
                    "execution_confidence": "screen_only_not_executable",
                    "coverage_start": "2025-01-02T15:30:00Z",
                    "coverage_end": "2025-01-02T15:30:00Z",
                    "sha256": "a803cdd9ba7804417eeb2a97af06064db7757b0ec919fb9acc4bfc0b89acd430",
                    "raw_path": "artifacts/institutional/historical/raw/fixtures/ibit_opra_fixture_2025-01-02.csv",
                    "coverage_labels": ["IBIT"],
                    "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
                }
            ],
        },
        "business_model_sequence": [
            "Research/evidence engine",
            "Treasury and miner hedge structuring",
            "Partner-led RFQ and execution support",
            "Principal risk sleeve only after chosen legal wrapper",
        ],
        "case_studies": {
            "treasury": {
                "title": "Corporate BTC treasury hedge case study",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                "hedged_btc": 350,
                "unhedged_btc": 650,
                "spot_value_usd": 80000000,
                "floor_price": 60000,
                "cap_price": 100000,
                "protected_value_at_floor_usd": 21000000,
                "quote_control": "Premium and executable levels require two-counterparty quote verification.",
            },
            "miner": {
                "title": "Miner runway protection case study",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                "hedged_monthly_btc": 60,
                "six_month_production_btc": 720,
                "floor_price": 56000,
                "monthly_revenue_at_spot_usd": 9600000,
                "monthly_floor_revenue_on_hedged_btc_usd": 3360000,
                "cash_runway_months_before_hedge": 3.75,
                "quote_control": "Indicative economics require quote verification before investor/client use.",
                "warnings": ["Hedge only conservative production; avoid overhedging and collateral stress."],
            },
        },
        "quote_verification_board": {
            "title": "Quote Verification Demo Board",
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            "control": "Manual demo workflow only; no RFQ is sent and no executable quote is implied.",
            "summary": {
                "screen_only": 2,
                "reviewed": 0,
                "rfq_package_drafted": 0,
                "indicative_quote_1": 1,
                "indicative_quote_2": 0,
                "quote_verified": 0,
                "trade_verified": 0,
            },
            "rows": [
                {
                    "rank": 1,
                    "candidate": "IBIT 5D ATM vs Deribit 4D ATM",
                    "stage": "screen_only",
                    "quote_count": 0,
                    "next_action": "Internal candidate review; draft RFQ package only after approval.",
                    "publishability": "internal-only",
                    "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                },
                {
                    "rank": 2,
                    "candidate": "IBIT 12D ATM vs Deribit 15D ATM",
                    "stage": "indicative_quote_1",
                    "quote_count": 1,
                    "next_action": "Collect second independent indicative quote before any quote-verified label.",
                    "publishability": "internal-only",
                    "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                },
            ],
        },
    }


def test_render_investor_site_contains_option_b_sections_and_controls():
    html = render_investor_site_html(_site_data())

    assert "BTC Treasury &amp; Miner Hedging Desk" in html
    assert "purpose-built cross-venue volatility evidence engine" in html
    assert "SCREEN-ONLY · NOT EXECUTABLE" in html
    assert "Executive proof status" in html
    assert "Verifier PASS" in html
    assert "1/6 groups ready" in html
    assert "External use" in html
    assert "Internal evidence prototype. Public screen/model data only" in html
    assert "No RFQ sent. No executable quote" in html
    assert "Latest Static Evidence Snapshot" in html
    assert "Static as of run timestamp; not real-time market data." in html
    assert "Treasury Hedge Case Study" in html
    assert "350 BTC hedged" in html
    assert "Scenario outputs are model-estimated and screen-only" in html
    assert "ILLUSTRATIVE MODEL OUTPUT" in html
    assert "illustrative floor-protected sleeve" in html
    assert "illustrative floor /" in html
    assert "model-estimated monthly floor revenue" in html
    assert "Static screen evidence only; not live market data" in html
    assert "Institutional Readiness Gate" in html
    assert "0/4 readiness gates passed" in html
    assert "NOT READY FOR INVESTMENT/CLIENT USE" in html
    assert "licensed historical source coverage incomplete" in html
    assert "two-counterparty quote evidence missing" in html
    assert "counsel-approved wrapper missing" in html
    assert "Readiness Remediation Plan" in html
    assert "Replace IBIT/Deribit fixtures and missing groups with licensed/replay-ready source manifests." in html
    assert "Accumulate enough point-in-time replay observations for sample-gate pass thresholds." in html
    assert "Capture two distinct external indicative quote records for at least one candidate." in html
    assert "Obtain counsel-approved business/legal wrapper before external investor/client use." in html
    assert "Licensed Source Intake Contract" in html
    assert "SCREEN-ONLY · INTAKE CONTRACT · NOT EXECUTABLE" in html
    assert "IBIT options history" in html
    assert "accepted formats: csv, jsonl, parquet" in html
    assert "available_ts, expiration, strike, option_type, bid, ask, volume, open_interest, source_ref" in html
    assert "No future available_ts relative to decision_ts" in html
    assert "Fixture/manual_fixture sources cannot satisfy readiness" in html
    assert "Source Intake Validation" in html
    assert "1/6 source groups structurally valid" in html
    assert "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE" in html
    assert "BTC reference history: fixture/manual_fixture sources cannot satisfy readiness" in html
    assert "Licensed Source Intake Contract" in html
    assert "Required licensed source groups and accepted formats" in html
    assert "Validation gates" in html
    assert "Current Source Availability" in html
    assert "6/6 current source groups available" in html
    assert "CME Bitcoin options · available" in html
    assert "Databento CME" in html
    assert "1135" in html
    assert "do not clear licensed historical readiness" in html
    assert "SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE" in html
    assert "overflow-wrap:anywhere" in html
    assert "@media (max-width:560px)" in html
    assert "Premium and executable levels require two-counterparty quote verification." in html
    assert "Miner Runway Protection" in html
    assert "60 BTC/month hedged" in html
    assert "3.75 months pre-hedge cash runway" in html
    assert "Indicative economics require quote verification before investor/client use." in html
    assert "Quote Verification Demo Board" in html
    assert "Manual demo workflow only; no RFQ is sent and no executable quote is implied." in html
    assert "Screen-only" in html
    assert "Indicative quote 1" in html
    assert "Internal RFQ draft" in html
    assert "Post-trade record verified" in html
    assert "After approval and manual outreach, counterparties may provide indicative quotes." in html
    assert "Evidence ledger may mark candidate quote-verified after required indicative quote records are captured." in html
    assert "Post-trade verification only occurs after an actual external execution record exists; this demo has no trading capability." in html
    assert "Quote Evidence Ledger" in html
    assert "Two distinct counterparties required for quote-verified candidate status" in html
    assert "same-dealer duplicate quotes cannot promote a candidate" in html
    assert "Backtest / Research Evidence" in html
    assert "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE" in html
    assert "synthetic fills" in html
    assert "insufficient-history" in html
    assert "1d scenario" in html
    assert "7d scenario" in html
    assert "30d scenario" in html
    assert "$49,000.00" in html
    assert "$2,050.00" in html
    assert "50.0%" in html
    assert "Cost/trade" in html
    assert "$250.00" in html
    assert "0.25 vol pts slippage" in html
    assert "strict chronology enforced" in html
    assert "strategy sees prior history only" in html
    assert "vol-spread-backtest-multitenor-v1.md" in html
    assert "vol-spread-backtest-multitenor-v1.json" in html
    assert "not an investment conclusion" in html
    assert "Backtest Robustness" in html
    assert "Sample gate pass count" in html
    assert "0 / 3" in html
    assert "Minimum gates" in html
    assert "30 snapshots / 20 synthetic trades" in html
    assert "Effective cost/trade" in html
    assert "$2,750.00" in html
    assert "Source Diagnostics" in html
    assert "Operational source coverage tracker" in html
    assert "Historical source files and replay controls" in html
    assert "<details class=\"disclosure\"><summary>Operational source coverage tracker</summary>" in html
    assert "Historical source manifest" in html
    assert "IBIT OPRA historical options fixture" in html
    assert "manual_fixture" in html
    assert "screen_only_not_executable" in html
    assert "historical-source-manifest-fixtures-v1.json" in html
    assert "Historical backtests use hashed point-in-time sources" in html
    assert "2/6 required source groups covered" in html
    assert "Coverage gaps" in html
    assert "CME missing" in html
    assert "BTC reference missing" in html
    assert "IBIT covered" in html
    assert "Deribit covered" in html
    assert "Rates/fees missing" in html
    assert "Operational source coverage tracker" in html
    assert "2 covered · 4 missing · 1 fixture-only" in html
    assert "IBIT · fixture-only" in html
    assert "Provider: manual_fixture" in html
    assert "Rows: 493" in html
    assert "Fixture source; replace with licensed historical export before investment use." in html
    assert "Expand IBIT historical coverage window and retain manifest hashes for replay." in html
    assert "CME · missing" in html
    assert "Configure licensed CME/Databento historical source; keep screen-only until quote/trade evidence exists." in html
    assert "Trade Now" not in html
    assert "Live RFQ" not in html
    assert "2 demo records" in html
    assert "0 real records" in html
    assert "Quote-verified candidates: 0" in html
    assert "0 trade-verified candidates" in html
    assert "Not investor-publishable" in html
    assert "Dealer &lt;A&gt;" in html
    assert "Dealer <A>" not in html
    assert "Demo/manual indicative rows do not count as real quote-verified evidence" in html
    assert "TRADE_VERIFIED" not in html
    assert "RFQ_PACKAGE_DRAFTED" not in html
    assert "Evidence Room" in html
    assert "Desk / Platform First" in html
    assert "Fund sleeve later" in html
    assert "IBIT 0D ATM vs Deribit 1D ATM" in html
    assert "draft two-counterparty indicative RFQ review (internal only)" in html
    assert "candidate review list" in html
    assert "RFQ review queue" not in html
    assert "b81186890ba7163c92b13407552d6ef83a83b32f98a0f8885187acd5c64c31ef" in html
    assert "Execute Trade" not in html
    assert "Place Order" not in html


def test_render_investor_site_escapes_dynamic_values():
    data = _site_data()
    data["latest_run"]["run_id"] = "<script>alert(1)</script>"
    data["top_candidates"][0]["candidate"] = "<img src=x onerror=alert(1)>"

    html = render_investor_site_html(data)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img" not in html
    assert "&lt;img" in html


def test_write_investor_site_creates_index_html(tmp_path):
    output = write_investor_site(_site_data(), tmp_path / "index.html")

    assert output.exists()
    assert "BTC Vol Desk" in output.read_text(encoding="utf-8")


def test_case_study_money_values_have_adjacent_control_pills():
    html = render_investor_site_html(_site_data())

    for phrase in (
        "$21,000,000.00 illustrative floor-protected sleeve",
        "$60,000.00 illustrative floor / $100,000.00 cap",
        "$3,360,000.00 model-estimated monthly floor revenue",
    ):
        idx = html.index(phrase)
        local_container_start = html.rfind("<div", 0, idx)
        local_container_end = html.find("</div>", idx)
        local = html[local_container_start:local_container_end]
        assert "ILLUSTRATIVE MODEL OUTPUT" in local
        assert "NOT A QUOTE" in local
        assert "NOT EXECUTABLE" in local
        assert "NOT INVESTOR-PUBLISHABLE" in local


def test_backtest_money_values_have_adjacent_control_pills():
    html = render_investor_site_html(_site_data())

    for phrase in ("$49,000.00", "$2,050.00", "$250.00", "$2,750.00"):
        idx = html.index(phrase)
        local_container_start = html.rfind("<div class='kpi'", 0, idx)
        if local_container_start == -1:
            local_container_start = html.rfind('<div class="kpi"', 0, idx)
        local_container_end = html.find("</div></div>", idx)
        local = html[local_container_start:local_container_end]
        assert "ILLUSTRATIVE MODEL OUTPUT" in local
        assert "NOT A QUOTE" in local
        assert "NOT EXECUTABLE" in local
        assert "NOT INVESTOR-PUBLISHABLE" in local


def test_case_studies_render_suppressed_when_spot_unavailable():
    data = _site_data()
    data["case_studies"] = {
        "available": False,
        "control_note": "BTC spot unavailable — case studies suppressed to avoid fallback economics.",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
    }

    html = render_investor_site_html(data)

    assert "BTC spot unavailable — case studies suppressed" in html
    assert "Treasury Hedge Case Study" not in html
    assert "Miner Runway Protection" not in html
