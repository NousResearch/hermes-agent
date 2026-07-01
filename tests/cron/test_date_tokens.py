"""Tests for deterministic cron date-token expansion."""

from datetime import datetime

from cron.date_tokens import _now_pt, expand_cron_date_tokens


def test_expands_all_supported_date_tokens():
    now = datetime(2026, 7, 1, 12, 0)
    text = " ".join(
        (
            "$(date +%Y-%m-%d)",
            "$(date +%F)",
            "$(date +%Y%m%d)",
            "$(date +%Y-%m)",
            "$(date +%Y)",
            r"$(date +\%Y-\%m-\%d)",
            "<TODAY>",
            "{TODAY}",
            "{{ TODAY }}",
            "<WEEK_ENDING_SUNDAY>",
            "{WEEK_ENDING_SUNDAY}",
            "{{ WEEK_ENDING_SUNDAY }}",
            "<YYYY-MM-DD>",
            "{YYYY-MM-DD}",
            "YYYY-MM-DD",
            "YYYYMMDD",
            "YYYY-MM",
        )
    )

    result = expand_cron_date_tokens(text, now=now)

    assert "$(date" not in result
    assert "TODAY" not in result
    assert "WEEK_ENDING_SUNDAY" not in result
    assert "YYYY" not in result
    assert result.count("2026-07-01") == 9
    assert "20260701" in result
    assert result.endswith("2026-07")


def test_week_ending_sunday_is_same_day_on_sunday():
    result = expand_cron_date_tokens(
        "<WEEK_ENDING_SUNDAY>", now=datetime(2026, 7, 5, 9, 0)
    )
    assert result == "2026-07-05"


def test_week_ending_sunday_uses_coming_sunday_midweek():
    result = expand_cron_date_tokens(
        "<WEEK_ENDING_SUNDAY>", now=datetime(2026, 7, 1, 9, 0)
    )
    assert result == "2026-07-05"


def test_plain_text_and_malformed_tokens_are_unchanged():
    text = "No dates here. Keep $(date +%Q), <TODAYS>, and YYYY-MM-DDx unchanged."
    assert expand_cron_date_tokens(text, now=datetime(2026, 7, 1)) == text


def test_empty_values_are_unchanged():
    assert expand_cron_date_tokens("") == ""


def test_default_clock_is_pacific_time():
    now = _now_pt()
    assert now.tzinfo is not None
    assert getattr(now.tzinfo, "key", None) == "America/Los_Angeles"
