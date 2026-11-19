"""``agent.retry_utils.reset_instant_from_message`` — the absolute ``resets <when>`` grammar.

A weekly plan wall names a calendar instant ("You've hit your weekly limit · resets Nov 20, 3pm
(Asia/Tokyo)"), which the relative "resets in 4hr" grammar cannot answer. The parser is only
allowed to answer when the text is clearly about a limit window and the instant is plausible;
everything else returns None so callers fall back to a bounded backoff instead of guessing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.retry_utils import reset_delay_from_message, reset_instant_from_message

JST = ZoneInfo("Asia/Tokyo")


def _epoch(year: int, month: int, day: int, hour: int, minute: int = 0, tz=JST) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=tz).timestamp()


def test_weekly_wall_sentence_resolves_to_its_named_instant():
    now = _epoch(2026, 11, 19, 9)
    text = ("Provider said: HTTP 429: This request would exceed your account's rate limit. "
            "Please try again later.): You've hit your weekly limit · resets Nov 20, 3pm (Asia/Tokyo)")
    assert reset_instant_from_message(text, now=now) == _epoch(2026, 11, 20, 15)


def test_iso_instant_with_utc_designator():
    now = _epoch(2026, 11, 19, 9)
    text = "quota exhausted; resets 2026-11-20T06:00:00Z"
    assert reset_instant_from_message(text, now=now) == _epoch(2026, 11, 20, 15)


def test_iso_instant_without_zone_reads_the_host_zone():
    now = _epoch(2026, 11, 19, 9)
    expected = datetime(2026, 11, 20, 6, 0, tzinfo=datetime.now().astimezone().tzinfo).timestamp()
    assert reset_instant_from_message("rate limit resets 2026-11-20 06:00", now=now) == expected


@pytest.mark.parametrize("hour, minute, meridiem, expected_hour", [
    (12, 0, "am", 0), (12, 0, "pm", 12), (3, 0, "pm", 15), (3, 30, "am", 3), (11, 5, "am", 11),
])
def test_twelve_hour_clock_conversion(hour, minute, meridiem, expected_hour):
    now = _epoch(2026, 11, 19, 9)
    text = f"rate limit resets Nov 20, {hour}" + (f":{minute:02d}" if minute else "") + f"{meridiem} (Asia/Tokyo)"
    assert reset_instant_from_message(text, now=now) == _epoch(2026, 11, 20, expected_hour, minute)


def test_relative_wall_is_left_to_the_relative_parser():
    now = _epoch(2026, 11, 19, 9)
    text = "You've hit your weekly limit · resets in 4hr"
    assert reset_instant_from_message(text, now=now) is None
    assert reset_delay_from_message(text) == 4 * 3600  # unchanged in-turn behaviour


@pytest.mark.parametrize("text", [
    "the parser flag resets Nov 20, 3pm (Asia/Tokyo)",       # no limit word before the verb
    "rate limit resets Nov 20, 3pm (Not/AZone)",             # unreadable zone
    "rate limit resets Nov 20",                              # no time of day
    "rate limit resets 3pm (Asia/Tokyo)",                    # no date
    "rate limit resets Nov 20, 25pm (Asia/Tokyo)",           # impossible clock
    "rate limit resets Nov 32, 3pm (Asia/Tokyo)",            # impossible day
])
def test_untrustworthy_shapes_return_none(text):
    assert reset_instant_from_message(text, now=_epoch(2026, 11, 19, 9)) is None


def test_hints_outside_the_plausible_horizon_are_ignored():
    now = _epoch(2026, 11, 19, 9)
    # More than a day past: a stale window statement, not this attempt's boundary.
    assert reset_instant_from_message("rate limit resets Nov 12, 3pm (Asia/Tokyo)", now=now) is None
    # Branch far beyond the two-week horizon: not trusted as a plan boundary.
    assert reset_instant_from_message("rate limit resets Dec 30, 3pm (Asia/Tokyo)", now=now) is None


def test_year_rollover_uses_the_next_occurrence():
    now = _epoch(2026, 12, 31, 9)
    assert reset_instant_from_message("rate limit resets Jan 2, 3pm (Asia/Tokyo)", now=now) == \
        _epoch(2027, 1, 2, 15)


def test_empty_and_unrelated_messages():
    assert reset_instant_from_message("", now=_epoch(2026, 11, 19, 9)) is None
    assert reset_instant_from_message(None, now=_epoch(2026, 11, 19, 9)) is None
    assert reset_instant_from_message("build succeeded in 42s", now=_epoch(2026, 11, 19, 9)) is None


def test_now_defaults_to_the_wall_clock():
    """Callers that pass no ``now`` still get the provider's instant for a near-future message."""
    when = datetime.now(UTC) + timedelta(hours=2)
    text = f"rate limit resets {when.day:02d} {when.hour:02d}:00"
    # ISO form: unambiguous and independent of month names.
    iso = f"rate limit resets {when.strftime('%Y-%m-%dT%H:%M')}"
    assert reset_instant_from_message(iso) is not None
    assert reset_instant_from_message(f"rate limit resets {text}") is None  # no month: not trusted
