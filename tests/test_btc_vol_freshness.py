from datetime import datetime, timedelta, timezone

from institutional_btc_vol.freshness import evaluate_source_freshness


def test_evaluate_source_freshness_marks_recent_core_sources_green():
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    result = evaluate_source_freshness(
        as_of=now,
        sources={
            "Deribit": now - timedelta(minutes=2),
            "Nasdaq IBIT options": now - timedelta(minutes=4),
            "iShares IBIT holdings": now - timedelta(hours=12),
        },
    )

    assert result["grade"] == "green"
    assert result["stale_sources"] == []
    assert result["missing_sources"] == []
    assert result["sources"]["Deribit"]["status"] == "fresh"
    assert result["sources"]["iShares IBIT holdings"]["max_age_minutes"] == 1440


def test_evaluate_source_freshness_degrades_stale_and_missing_core_sources():
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    result = evaluate_source_freshness(
        as_of=now,
        sources={
            "Deribit": now - timedelta(minutes=45),
            "Nasdaq IBIT options": None,
            "iShares IBIT holdings": now - timedelta(days=2),
        },
    )

    assert result["grade"] == "red"
    assert "Deribit" in result["stale_sources"]
    assert "iShares IBIT holdings" in result["stale_sources"]
    assert result["missing_sources"] == ["Nasdaq IBIT options"]
    assert result["sources"]["Nasdaq IBIT options"]["status"] == "missing"


def test_evaluate_source_freshness_accepts_iso_timestamp_strings():
    now = "2026-05-15T12:00:00+00:00"
    result = evaluate_source_freshness(
        as_of=now,
        sources={"Deribit": "2026-05-15T11:58:30+00:00"},
        max_age_minutes={"Deribit": 5},
    )

    assert result["grade"] == "green"
    assert result["sources"]["Deribit"]["age_minutes"] == 1.5
