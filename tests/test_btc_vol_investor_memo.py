from pathlib import Path

from institutional_btc_vol.investor_memo import render_investor_memo_markdown, write_investor_memo


def _site_data():
    return {
        "positioning": "BTC Treasury & Miner Hedging Desk",
"full_positioning": "BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "latest_run": {
            "run_id": "btcvol-test",
            "as_of_cst": "2026-05-15 13:15:00 CDT",
            "btc_spot": 80000.0,
            "btc_per_share": 0.000567,
            "quality_score": 100,
            "quality_grade": "green",
            "freshness_grade": "green",
            "dislocations": 7,
            "quote_review_candidates": 3,
            "evidence_bundle_sha256": "abc123",
            "report_path": "artifacts/institutional/data/reports/latest.md",
            "evidence_bundle_path": "artifacts/institutional/data/normalized/run/bundle.zip",
            "evidence_manifest_path": "artifacts/institutional/data/normalized/run/evidence_manifest.json",
            "candidate_ledger_path": "artifacts/institutional/data/normalized/run/candidate_triage.jsonl",
        },
        "top_candidates": [
            {
                "rank": 1,
                "candidate": "IBIT 5D ATM vs Deribit 4D ATM",
                "gross_iv_diff_vol_pts": 3.47,
                "priority": "review",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            }
        ],
        "case_studies": {
            "treasury": {
                "title": "Corporate BTC treasury hedge case study",
                "hedged_btc": 350,
                "protected_value_at_floor_usd": 21_000_000,
                "floor_price": 60_000,
                "cap_price": 100_000,
                "quote_control": "Premium and executable levels require two-counterparty quote verification.",
            },
            "miner": {
                "title": "Miner runway protection case study",
                "hedged_monthly_btc": 60,
                "monthly_floor_revenue_on_hedged_btc_usd": 3_360_000,
                "cash_runway_months_before_hedge": 3.75,
                "quote_control": "Indicative economics require quote verification before investor/client use.",
            },
        },
        "quote_verification_board": {
            "title": "Quote Verification Demo Board",
            "control": "Manual demo workflow only; no RFQ is sent and no executable quote is implied.",
            "summary": {
                "screen_only": 5,
                "reviewed": 0,
                "rfq_package_drafted": 0,
                "indicative_quote_1": 0,
                "indicative_quote_2": 0,
                "quote_verified": 0,
                "trade_verified": 0,
            },
        },
        "business_model_sequence": [
            "Research/evidence engine",
            "Treasury and miner hedge structuring",
            "Partner-led RFQ and execution support",
            "Principal risk sleeve only after chosen legal wrapper",
        ],
    }


def test_render_investor_memo_contains_diligence_sections_and_controls():
    memo = render_investor_memo_markdown(_site_data())

    assert memo.startswith("# BTC Treasury & Miner Hedging Desk Investor Memo")
    assert "SCREEN-ONLY · NOT EXECUTABLE" in memo
    assert "Not a client portal. Not an execution venue. Not a fund offering." in memo
    assert "## Thesis" in memo
    assert "## Evidence Snapshot" in memo
    assert "Run ID: `btcvol-test`" in memo
    assert "Bundle SHA-256: `abc123`" in memo
    assert "## Treasury Case Study" in memo
    assert "350 BTC hedged" in memo
    assert "Illustrative model output — protected value: $21,000,000.00 illustrative floor-protected sleeve" in memo
    assert "Illustrative model output — monthly floor revenue" in memo
    assert "## Miner Case Study" in memo
    assert "60 BTC/month hedged" in memo
    assert "## Quote Verification Workflow" in memo
    assert "Manual demo workflow only; no RFQ is sent and no executable quote is implied." in memo
    assert "Post-trade verification only occurs after an actual external execution record exists; this memo does not imply execution." in memo
    assert "## Business Model Sequence" in memo
    assert "## Investor Ask" in memo
    assert "## Evidence Room" in memo
    assert "## Limitations / Gating Items" in memo
    assert "CME remains unavailable until licensed/vendor/broker feed is configured." in memo
    assert "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow." in memo
    assert "Internal evidence prototype. Public screen/model data only" in memo
    assert "Configured-source quality" in memo
    assert "Current screen-source availability" in memo
    assert "gross IV observations only" in memo
    assert "Submit RFQ" not in memo
    assert "Execute Trade" not in memo


def test_write_investor_memo_creates_markdown_file(tmp_path):
    output = tmp_path / "investor-packet" / "memo.md"

    written = write_investor_memo(_site_data(), output)

    assert written == output
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# BTC Treasury & Miner Hedging Desk Investor Memo")
