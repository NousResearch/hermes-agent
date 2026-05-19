from institutional_btc_vol.dashboard import render_dashboard_html


def test_render_dashboard_html_contains_status_cards_curves_and_screen_only_label():
    html = render_dashboard_html(
        run_id="btcvol-20260514-220000",
        as_of_cst="2026-05-14 22:00:00 CDT",
        btc_spot=81770.86,
        btc_per_share=0.000564212717166,
        deribit_atm_rows=[
            {"expiry": "2026-05-15", "dte": 1, "native_symbol": "BTC-15MAY26-81000-C", "iv_mark": 0.3285},
            {"expiry": "2026-05-29", "dte": 15, "native_symbol": "BTC-29MAY26-81000-P", "iv_mark": 0.3601},
        ],
        ibit_atm_rows=[
            {"expiry": "2026-05-15", "dte": 1, "native_symbol": "IBIT-2026-05-15-46.0-C", "iv_mark": 0.3480},
        ],
        dislocations=[
            {
                "candidate": "IBIT 1D ATM vs Deribit 1D ATM",
                "gross_iv_diff_vol_pts": 1.95,
                "confidence": "screen-only",
                "next_action": "watch",
            },
            {
                "candidate": "IBIT 6D ATM vs Deribit 4D ATM",
                "gross_iv_diff_vol_pts": 3.58,
                "confidence": "screen-only",
                "next_action": "quote review",
            },
        ],
        quality_warnings=["CME source missing until licensed/vendor/broker feed is configured"],
        quality_score={
            "score": 95,
            "grade": "green",
            "core_sources_live": True,
            "material_warning_count": 0,
            "notes": ["CME missing is expected pre-license"],
        },
        trend_summary={
            "run_count": 2,
            "latest_run_id": "btcvol-20260514-215245",
            "btc_change": 150.25,
            "quality_change": 5,
            "quote_review_change": 1,
            "latest_quality_grade": "green",
            "btc_series": [81620.61, 81770.86],
            "quality_series": [90, 95],
            "spread_7d_change_vol_pts": 1.25,
            "spread_7d_series": [2.0, 3.25],
            "latest_deribit_30d_iv": 0.39,
            "latest_ibit_30d_iv": 0.405,
        },
        quote_evidence={
            "valid_records": [
                {
                    "rfq_id": "RFQ-009",
                    "structure": "IBIT vs Deribit 30D ATM straddle",
                    "counterparty": "Desk A",
                    "execution_confidence": "quote-verified",
                    "status": "indicative",
                    "spread_vol_pts": 1.7,
                    "is_investor_publishable": False,
                }
            ],
            "invalid_records": [],
            "summary": {"total": 1, "quote_verified": 1, "trade_verified": 0, "invalid": 0},
        },
        candidate_triage=[
            {
                "rank": 1,
                "candidate": "IBIT 7D ATM vs Deribit 7D ATM",
                "gross_iv_diff_vol_pts": 7.74,
                "priority": "high",
                "direction": "IBIT rich vs Deribit",
                "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
                "recommended_workflow": "draft two-counterparty indicative RFQ review (internal only)",
            }
        ],
        market_diagnostics=[
            {
                "source": "Deribit BTC options",
                "option_markets": 936,
                "valid_bid_ask": 920,
                "missing_bid_ask": 16,
                "crossed_markets": 0,
                "wide_markets": 0,
                "missing_iv": 0,
                "grade": "green",
                "notes": ["No bid/ask or IV anomalies detected"],
            },
            {
                "source": "IBIT Nasdaq option-chain",
                "option_markets": 493,
                "valid_bid_ask": 450,
                "missing_bid_ask": 43,
                "crossed_markets": 1,
                "wide_markets": 7,
                "missing_iv": 0,
                "grade": "yellow",
                "notes": ["1 crossed market", "7 wide markets"],
            },
        ],
        freshness={
            "grade": "yellow",
            "stale_sources": ["Nasdaq IBIT options"],
            "missing_sources": [],
            "sources": {
                "Deribit": {"status": "fresh", "age_minutes": 2.0, "max_age_minutes": 15},
                "Nasdaq IBIT options": {"status": "stale", "age_minutes": 18.0, "max_age_minutes": 15},
                "iShares IBIT holdings": {"status": "fresh", "age_minutes": 720.0, "max_age_minutes": 1440},
            },
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        },
        recent_runs=[
            {
                "run_id": "btcvol-20260514-215245",
                "as_of_cst": "2026-05-14 21:52:45 CDT",
                "btc_spot": 81770.86,
                "dislocations": 7,
                "quote_review_candidates": 1,
            }
        ],
    )

    assert "BTC Vol Desk Monitor" in html
    assert "SCREEN-ONLY" in html
    assert "btcvol-20260514-220000" in html
    assert "$81,770.86" in html
    assert "0.000564212717" in html
    assert "Deribit" in html
    assert "IBIT / ETF" in html
    assert "CME" in html
    assert "RFQ" in html
    assert "IBIT 6D ATM vs Deribit 4D ATM" in html
    assert "3.58 vol pts" in html
    assert "Quote Review" in html
    assert "RFQ-001" in html
    assert "Internal RFQ Template Inventory" in html
    assert "Internal template" in html
    assert "Review required" in html
    assert "Internal evidence prototype" in html
    assert "Screen-only comparison; not executable economics" in html
    assert "Live" not in html
    assert "CME source missing" in html
    assert "Data Quality" in html
    assert "95 / 100" in html
    assert "GREEN" in html
    assert "Core sources captured" in html
    assert "Trend Snapshot" in html
    assert "$150.25" in html
    assert "+5" in html
    assert "+1" in html
    assert "IV Benchmark Trends" in html
    assert "7D spread change" in html
    assert "30D Deribit" in html
    assert "RFQ Quote Evidence" in html
    assert "quote-verified" in html
    assert "Not investor-publishable" in html
    assert "Candidate Triage" in html
    assert "IBIT 7D ATM vs Deribit 7D ATM" in html
    assert "draft two-counterparty indicative RFQ review (internal only)" in html
    assert "Market Diagnostics" in html
    assert "IBIT Nasdaq option-chain" in html
    assert "493" in html
    assert "1 crossed market" in html
    assert "7 wide markets" in html
    assert "Source Freshness" in html
    assert "Nasdaq IBIT options" in html
    assert "18.0 min" in html
    assert "SCREEN-ONLY · NOT EXECUTABLE" in html
    assert "Recent Runs" in html
    assert "btcvol-20260514-215245" in html
    assert "1 quote-review" in html


def test_render_dashboard_html_escapes_dynamic_values():
    html = render_dashboard_html(
        run_id="<script>alert('x')</script>",
        as_of_cst="2026-05-14 22:00:00 CDT",
        btc_spot=None,
        btc_per_share=None,
        deribit_atm_rows=[],
        ibit_atm_rows=[],
        dislocations=[{"candidate": "<b>bad</b>", "gross_iv_diff_vol_pts": 0, "confidence": "screen-only", "next_action": "watch"}],
        quality_warnings=["<img src=x onerror=alert(1)>"],
        quality_score={"score": 10, "grade": "<red>", "core_sources_live": False, "material_warning_count": 1, "notes": ["<bad>"]},
        quote_evidence={"valid_records": [], "invalid_records": [{"errors": ["<bad>"]}], "summary": {"total": 1, "quote_verified": 0, "trade_verified": 0, "invalid": 1}},
        recent_runs=[{"run_id": "<svg/onload=alert(1)>", "as_of_cst": "now", "btc_spot": None, "dislocations": 0, "quote_review_candidates": 0}],
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>bad</b>" not in html
    assert "&lt;b&gt;bad&lt;/b&gt;" in html
    assert "<img" not in html



def test_render_dashboard_html_marks_cme_available_when_databento_rows_present():
    html = render_dashboard_html(
        run_id="btcvol-cme",
        as_of_cst="2026-05-17 12:55:38 CDT",
        btc_spot=78504.03,
        btc_per_share=0.0005679118586151414,
        deribit_atm_rows=[],
        ibit_atm_rows=[],
        dislocations=[],
        quality_warnings=["CME Databento normalized rows: 1605", "CME Databento BBO rows: 5000"],
        cme_rows=1605,
    )

    assert "CME" in html
    assert "Available" in html
    assert "1,605 Databento rows" in html
    assert "licensed/vendor feed required" not in html
