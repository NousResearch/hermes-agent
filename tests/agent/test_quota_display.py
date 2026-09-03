"""Tests for the `display.quota` mode vocabulary shared by /quota and the TUI."""

import pytest

from agent.quota_display import (
    DEFAULT_QUOTA_MODE,
    QUOTA_MODES,
    describe_quota_mode,
    format_reset_in,
    normalize_quota_mode,
    quota_usage,
    render_quota_menu,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("session", "session"),
        ("SESSION", "session"),
        ("  5h  ", "session"),
        ("short", "session"),
        ("both", "both"),
        ("all", "both"),
        ("weekly", "weekly"),
        ("week", "weekly"),
        ("tightest", "tightest"),
        ("off", "off"),
        ("none", "off"),
        ("hidden", "off"),
    ],
)
def test_normalize_resolves_names_and_aliases(raw, expected):
    assert normalize_quota_mode(raw) == expected


def test_yaml_booleans_read_as_toggles():
    # `quota: false` should behave like the other display switches.
    assert normalize_quota_mode(False) == "off"
    assert normalize_quota_mode(True) == DEFAULT_QUOTA_MODE


@pytest.mark.parametrize("raw", ["", "nonsense", "5m", 42, None, []])
def test_unrecognized_values_resolve_to_none(raw):
    # None, not a silent default: callers show usage rather than picking a mode
    # the user did not ask for.
    assert normalize_quota_mode(raw) is None


def test_every_canonical_mode_round_trips_and_is_described():
    for mode in QUOTA_MODES:
        assert normalize_quota_mode(mode) == mode
        assert describe_quota_mode(mode) != mode


def test_usage_line_lists_every_mode():
    usage = quota_usage()

    for mode in QUOTA_MODES:
        assert mode in usage
    assert "status" in usage


LIVE = {"session": (100, "2h 13m"), "weekly": (81, "5d 0h")}


def test_menu_lists_every_mode_with_the_segment_it_produces():
    menu = render_quota_menu("session", LIVE)

    for mode in QUOTA_MODES:
        assert f"/quota {mode}" in menu
    # One glyph per segment: the trailing window is appended bare, exactly as
    # the status bar renders it.
    assert "◔ 100% 2h 13m · 81% 5d 0h" in menu
    assert menu.count("◔ 100% 2h 13m · ◔") == 0


def test_menu_marks_the_current_mode_only():
    menu = render_quota_menu("weekly", LIVE)
    marked = [ln for ln in menu.splitlines() if ln.strip().startswith("▸")]

    assert len(marked) == 1
    assert "/quota weekly" in marked[0]
    assert "currently: weekly" in menu


def test_menu_falls_back_to_sample_numbers_without_a_snapshot():
    menu = render_quota_menu("session")

    assert "sample numbers" in menu
    assert "◔ 100% 2h 13m" in menu


def test_menu_says_when_the_examples_are_the_users_own_limits():
    assert "your current limits" in render_quota_menu("session", LIVE)


def test_menu_tightest_example_follows_the_scarcer_window():
    assert "◔ 81% 5d 0h" in render_quota_menu("tightest", LIVE)
    roomy_weekly = {"session": (12, "40m"), "weekly": (90, "6d 2h")}
    tightest_line = [
        ln for ln in render_quota_menu("tightest", roomy_weekly).splitlines() if "/quota tightest" in ln
    ][0]

    assert "12%" in tightest_line


def test_format_reset_in_matches_the_read_outs_resolution():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)

    assert format_reset_in(now + timedelta(minutes=12)) == "12m"
    assert format_reset_in(now + timedelta(minutes=165)) == "2h 45m"
    assert format_reset_in(now + timedelta(minutes=7380)) == "5d 3h"
    assert format_reset_in(now - timedelta(hours=1)) == "now"
    assert format_reset_in(None) == ""
    assert format_reset_in("not-a-date") == ""


def test_format_reset_in_accepts_the_iso_strings_the_rpc_returns():
    from datetime import datetime, timedelta, timezone

    iso = (datetime.now(timezone.utc) + timedelta(minutes=90)).isoformat()

    assert format_reset_in(iso) == "1h 30m"
