"""Tests for hermes_cli/timefmt.py — relative_time and boundary conditions."""

import time


def test_null_timestamp_returns_question_mark():
    from hermes_cli.timefmt import relative_time
    assert relative_time(0) == "?"
    assert relative_time(None) == "?"
    assert relative_time("") == "?"


def test_just_now_for_fresh_timestamps():
    from hermes_cli.timefmt import relative_time
    assert relative_time(time.time()) == "just now"
    assert relative_time(time.time() - 30) == "just now"


def test_minutes_ago():
    from hermes_cli.timefmt import relative_time
    assert relative_time(time.time() - 120) == "2m ago"
    assert relative_time(time.time() - 59 * 60) == "59m ago"


def test_hours_ago():
    from hermes_cli.timefmt import relative_time
    assert relative_time(time.time() - 3600) == "1h ago"
    assert relative_time(time.time() - 23 * 3600) == "23h ago"


def test_yesterday():
    from hermes_cli.timefmt import relative_time
    assert relative_time(time.time() - 86400) == "yesterday"
    assert relative_time(time.time() - 172799) == "yesterday"


def test_days_ago():
    from hermes_cli.timefmt import relative_time
    assert relative_time(time.time() - 2 * 86400) == "2d ago"
    assert relative_time(time.time() - 6 * 86400) == "6d ago"


def test_older_than_a_week_uses_date_format():
    from hermes_cli.timefmt import relative_time
    result = relative_time(time.time() - 8 * 86400)
    # Should be YYYY-MM-DD format
    assert len(result) == 10
    assert result[4] == "-" and result[7] == "-"


def test_future_timestamps_still_format():
    from hermes_cli.timefmt import relative_time
    result = relative_time(time.time() + 86400 * 400)
    assert len(result) == 10  # YYYY-MM-DD
