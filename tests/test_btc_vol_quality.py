from institutional_btc_vol.quality import compute_quality_score


def test_compute_quality_score_green_when_core_sources_live_and_no_material_warnings():
    score = compute_quality_score(
        deribit_rows=936,
        ibit_rows=493,
        btc_per_share=0.000564212717166,
        cme_available=False,
        quality_warnings=["CME source missing until licensed/vendor/broker feed is configured"],
        dislocations=7,
        quote_review_candidates=1,
    )

    assert score["grade"] == "green"
    assert score["score"] >= 80
    assert score["core_sources_live"] is True
    assert score["cme_available"] is False
    assert "CME missing is expected pre-license" in score["notes"]


def test_compute_quality_score_yellow_when_ibit_missing_or_warnings_accumulate():
    score = compute_quality_score(
        deribit_rows=936,
        ibit_rows=0,
        btc_per_share=0.000564212717166,
        cme_available=False,
        quality_warnings=[
            "CME source missing until licensed/vendor/broker feed is configured",
            "IBIT options fetch failed: 403",
        ],
        dislocations=0,
        quote_review_candidates=0,
    )

    assert score["grade"] == "yellow"
    assert score["core_sources_live"] is False
    assert "IBIT options missing" in score["notes"]


def test_compute_quality_score_red_when_deribit_or_btc_share_missing():
    score = compute_quality_score(
        deribit_rows=0,
        ibit_rows=493,
        btc_per_share=None,
        cme_available=False,
        quality_warnings=["Deribit fetch failed", "iShares BTC/share parse failed"],
        dislocations=0,
        quote_review_candidates=0,
    )

    assert score["grade"] == "red"
    assert score["score"] < 50
    assert "Deribit missing" in score["notes"]
    assert "BTC/share missing" in score["notes"]


def test_compute_quality_score_penalizes_stale_source_freshness():
    score = compute_quality_score(
        deribit_rows=12,
        ibit_rows=493,
        btc_per_share=0.000564212717166,
        cme_available=False,
        quality_warnings=["CME source missing until licensed/vendor/broker feed is configured"],
        dislocations=7,
        quote_review_candidates=1,
        freshness={
            "grade": "yellow",
            "stale_sources": ["Nasdaq IBIT options"],
            "missing_sources": [],
        },
    )

    assert score["grade"] == "yellow"
    assert score["freshness_grade"] == "yellow"
    assert "Freshness warning: Nasdaq IBIT options stale" in score["notes"]


def test_compute_quality_score_red_when_core_source_freshness_missing():
    score = compute_quality_score(
        deribit_rows=12,
        ibit_rows=493,
        btc_per_share=0.000564212717166,
        cme_available=False,
        quality_warnings=["CME source missing until licensed/vendor/broker feed is configured"],
        dislocations=7,
        quote_review_candidates=1,
        freshness={
            "grade": "red",
            "stale_sources": [],
            "missing_sources": ["Deribit"],
        },
    )

    assert score["grade"] == "red"
    assert score["score"] < 50
    assert "Freshness missing: Deribit" in score["notes"]
