from institutional_btc_vol.trends import build_trend_summary


def test_build_trend_summary_computes_btc_quality_and_review_deltas():
    runs = [
        {
            "run_id": "btcvol-1",
            "as_of_cst": "2026-05-14 22:35:08 CDT",
            "btc_spot": 81500.0,
            "quality_score": 95,
            "quality_grade": "green",
            "dislocations": 6,
            "quote_review_candidates": 1,
        },
        {
            "run_id": "btcvol-2",
            "as_of_cst": "2026-05-14 22:42:06 CDT",
            "btc_spot": 81650.0,
            "quality_score": 100,
            "quality_grade": "green",
            "dislocations": 7,
            "quote_review_candidates": 2,
        },
    ]

    summary = build_trend_summary(runs)

    assert summary["run_count"] == 2
    assert summary["latest_run_id"] == "btcvol-2"
    assert summary["btc_change"] == 150.0
    assert summary["quality_change"] == 5
    assert summary["quote_review_change"] == 1
    assert summary["latest_quality_grade"] == "green"
    assert summary["btc_series"] == [81500.0, 81650.0]
    assert summary["quality_series"] == [95, 100]


def test_build_trend_summary_accepts_newest_first_manifest_rows():
    summary = build_trend_summary([
        {"run_id": "btcvol-20260514-225121", "as_of_cst": "2026-05-14 22:51:21 CDT", "btc_spot": 81515.0, "quality_score": 100, "quality_grade": "green", "quote_review_candidates": 2},
        {"run_id": "btcvol-20260514-224130", "as_of_cst": "2026-05-14 22:41:30 CDT", "btc_spot": 81542.0, "quality_score": 75, "quality_grade": "yellow", "quote_review_candidates": 3},
    ])

    assert summary["latest_run_id"] == "btcvol-20260514-225121"
    assert summary["latest_quality_grade"] == "green"
    assert summary["quality_change"] == 25
    assert summary["quote_review_change"] == -1


def test_build_trend_summary_handles_missing_or_single_run():
    assert build_trend_summary([])["run_count"] == 0

    summary = build_trend_summary([
        {"run_id": "btcvol-1", "btc_spot": None, "quality_score": None, "quote_review_candidates": 0}
    ])

    assert summary["run_count"] == 1
    assert summary["btc_change"] is None
    assert summary["quality_change"] is None
    assert summary["quote_review_change"] is None
