from institutional_btc_vol.trends import build_trend_summary, extract_iv_benchmarks


def test_extract_iv_benchmarks_selects_nearest_tenors_and_spreads_in_vol_points():
    deribit_rows = [
        {"dte": 1, "iv_mark": 0.30},
        {"dte": 8, "iv_mark": 0.35},
        {"dte": 22, "iv_mark": 0.38},
    ]
    ibit_rows = [
        {"dte": 1, "iv_mark": 0.34},
        {"dte": 6, "iv_mark": 0.37},
        {"dte": 29, "iv_mark": 0.41},
    ]

    benchmarks = extract_iv_benchmarks(deribit_rows, ibit_rows, tenors=(1, 7, 30))

    assert benchmarks["deribit_1d_iv"] == 0.30
    assert benchmarks["ibit_1d_iv"] == 0.34
    assert benchmarks["spread_1d_vol_pts"] == 4.0
    assert benchmarks["deribit_7d_iv"] == 0.35
    assert benchmarks["ibit_7d_iv"] == 0.37
    assert benchmarks["spread_7d_vol_pts"] == 2.0
    assert benchmarks["deribit_30d_iv"] == 0.38
    assert benchmarks["ibit_30d_iv"] == 0.41
    assert benchmarks["spread_30d_vol_pts"] == 3.0


def test_build_trend_summary_includes_iv_benchmark_changes_from_manifest_rows():
    runs = [
        {
            "run_id": "btcvol-20260514-230000",
            "as_of_cst": "2026-05-14 23:00:00 CDT",
            "btc_spot": 81000,
            "quality_score": 100,
            "quote_review_candidates": 1,
            "quality_grade": "green",
            "spread_7d_vol_pts": 2.0,
            "deribit_30d_iv": 0.38,
            "ibit_30d_iv": 0.41,
        },
        {
            "run_id": "btcvol-20260514-235000",
            "as_of_cst": "2026-05-14 23:50:00 CDT",
            "btc_spot": 81200,
            "quality_score": 100,
            "quote_review_candidates": 2,
            "quality_grade": "green",
            "spread_7d_vol_pts": 3.25,
            "deribit_30d_iv": 0.39,
            "ibit_30d_iv": 0.405,
        },
    ]

    summary = build_trend_summary(runs)

    assert summary["spread_7d_change_vol_pts"] == 1.25
    assert summary["spread_7d_series"] == [2.0, 3.25]
    assert summary["latest_deribit_30d_iv"] == 0.39
    assert summary["latest_ibit_30d_iv"] == 0.405
