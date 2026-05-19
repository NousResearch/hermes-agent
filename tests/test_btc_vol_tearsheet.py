from pathlib import Path

from institutional_btc_vol.investor_tearsheet import render_tearsheet_markdown, write_tearsheet


def _data():
    return {
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "positioning": "BTC Treasury & Miner Hedging Desk",
        "full_positioning": "BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine",
        "latest_run": {
            "run_id": "run-tear-001",
            "as_of_cst": "2026-05-15 14:10:00 CDT",
            "btc_spot": 80123.45,
            "quality_grade": "green",
            "freshness_grade": "green",
            "dislocations": 7,
            "quote_review_candidates": 3,
            "evidence_bundle_sha256": "abc123def456",
        },
        "top_candidates": [
            {
                "candidate": "IBIT 5D ATM vs Deribit 4D ATM",
                "gross_iv_diff_vol_pts": 3.47,
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            }
        ],
        "quote_evidence_ledger": {
            "summary": {
                "quote_verified_records": 0,
                "manual_indicative_records": 2,
                "quote_verified_candidates": 0,
                "trade_verified_candidates": 0,
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "structure": "IBIT 5D ATM vs Deribit 4D ATM",
                        "indicative_quote_count": 2,
                        "stage": "two-demo-indicative-quotes",
                        "publishability": "not-investor-publishable",
                    }
                ],
            }
        },
    }


def test_render_tearsheet_markdown_is_one_page_investor_summary_with_controls():
    markdown = render_tearsheet_markdown(_data())

    assert markdown.startswith("# BTC Treasury & Miner Hedging Desk — One-Page Tear Sheet")
    assert "BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine" in markdown
    assert "SCREEN-ONLY · NOT EXECUTABLE" in markdown
    assert "Not a client portal. Not an execution venue. Not a fund offering." in markdown
    assert "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow." in markdown
    assert "Internal evidence prototype. Public screen/model data only" in markdown
    assert "## Evidence Snapshot" in markdown
    assert "Run ID: `run-tear-001`" in markdown
    assert "BTC Reference: `$80,123.45`" in markdown
    assert "Configured-source quality: `GREEN`" in markdown
    assert "Current screen-source availability: `Current screen/vendor captures partial: CME unavailable`" in markdown
    assert "Overall evidence readiness" in markdown
    assert "Screen-only dislocations: `7`" in markdown
    assert "Quote-review flags: `3`" in markdown
    assert "## Why Now" in markdown
    assert "## Initial ICP" in markdown
    assert "## Evidence Operations" in markdown
    assert "2 demo/manual indicative quote records" in markdown
    assert "0 real quote-verified records" in markdown
    assert "0 quote-verified candidates" in markdown
    assert "0 trade-verified candidates" in markdown
    assert "not-investor-publishable" in markdown
    assert "## Diligence Ask" in markdown
    assert "Submit RFQ" not in markdown
    assert "Execute Trade" not in markdown
    assert "Request Quote" not in markdown


def test_write_tearsheet_writes_file(tmp_path):
    output = tmp_path / "tear-sheet.md"

    path = write_tearsheet(_data(), output)

    assert path == output
    text = output.read_text(encoding="utf-8")
    assert "One-Page Tear Sheet" in text
    assert "abc123def456" in text
