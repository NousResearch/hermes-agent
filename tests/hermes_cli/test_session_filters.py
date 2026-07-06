"""Tests for hermes_cli.session_filters — CLI time/filter parsing for
`hermes sessions prune` / `hermes sessions archive`."""

import time
from argparse import Namespace
from datetime import datetime

import pytest

from hermes_cli.session_filters import (
    build_prune_filters,
    describe_filters,
    parse_duration_seconds,
    parse_point_in_time,
)


def _ns(**kwargs):
    defaults = dict(
        older_than=None, newer_than=None, before=None, after=None,
        source=None, title=None, end_reason=None, cwd=None,
        min_messages=None, max_messages=None,
        model=None, provider=None, user=None, chat_id=None, chat_type=None,
        branch=None, min_tokens=None, max_tokens=None, min_cost=None,
        max_cost=None, min_tool_calls=None, max_tool_calls=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestParseDurationSeconds:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("30m", 1800),
            ("5h", 18000),
            ("2d", 172800),
            ("1w", 604800),
            ("90", 90 * 86400),   # bare number = days (back-compat)
            ("1.5h", 5400),
            ("10 min", 600),
            ("2 hours", 7200),
        ],
    )
    def test_valid(self, value, expected):
        assert parse_duration_seconds(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "abc", "5x", "2026-07-05", "h5"])
    def test_invalid_returns_none(self, value):
        assert parse_duration_seconds(value) is None


class TestParsePointInTime:
    def test_duration_is_relative_to_now(self):
        ts = parse_point_in_time("5h", "--before")
        assert ts == pytest.approx(time.time() - 18000, abs=5)

    def test_iso_date(self):
        ts = parse_point_in_time("2026-07-05", "--before")
        assert ts == datetime(2026, 7, 5).timestamp()

    def test_iso_datetime(self):
        ts = parse_point_in_time("2026-07-05 14:30", "--after")
        assert ts == datetime(2026, 7, 5, 14, 30).timestamp()

    def test_invalid_raises_with_flag_name(self):
        with pytest.raises(ValueError, match="--older-than"):
            parse_point_in_time("nonsense", "--older-than")


class TestBuildPruneFilters:
    def test_newer_than_sets_lower_bound_only(self):
        f = build_prune_filters(_ns(newer_than="5h"))
        assert f["started_before"] is None
        assert f["started_after"] == pytest.approx(time.time() - 18000, abs=5)
        assert f["older_than_days"] is None  # no implicit 90d cap

    def test_older_than_bare_days(self):
        f = build_prune_filters(_ns(older_than="90"))
        assert f["started_before"] == pytest.approx(
            time.time() - 90 * 86400, abs=5
        )
        assert f["started_after"] is None

    def test_window_before_and_after(self):
        f = build_prune_filters(_ns(after="10h", before="2h"))
        assert f["started_after"] < f["started_before"]

    def test_inverted_window_rejected(self):
        with pytest.raises(ValueError, match="Empty time window"):
            build_prune_filters(_ns(after="2h", before="10h"))

    def test_tighter_bound_wins(self):
        # --older-than 1d and --before 5h both set the upper bound;
        # 1d ago is earlier (tighter for "older than") so it wins.
        f = build_prune_filters(_ns(older_than="1d", before="5h"))
        assert f["started_before"] == pytest.approx(
            time.time() - 86400, abs=5
        )

    def test_passthrough_filters(self):
        f = build_prune_filters(
            _ns(source="cli", title="smoke", end_reason="done",
                cwd="/tmp/x", min_messages=1, max_messages=9)
        )
        assert f["source"] == "cli"
        assert f["title_like"] == "smoke"
        assert f["end_reason"] == "done"
        assert f["cwd_prefix"] == "/tmp/x"
        assert f["min_messages"] == 1
        assert f["max_messages"] == 9

    def test_passthrough_extended_filters(self):
        f = build_prune_filters(
            _ns(model="sonnet", provider="openrouter", user="alice",
                chat_id="c-9", chat_type="group", branch="feature/x",
                min_tokens=100, max_tokens=5000, min_cost=0.01,
                max_cost=2.5, min_tool_calls=1, max_tool_calls=40)
        )
        assert f["model_like"] == "sonnet"
        assert f["provider"] == "openrouter"
        assert f["user_id"] == "alice"
        assert f["chat_id"] == "c-9"
        assert f["chat_type"] == "group"
        assert f["branch_like"] == "feature/x"
        assert f["min_tokens"] == 100
        assert f["max_tokens"] == 5000
        assert f["min_cost"] == 0.01
        assert f["max_cost"] == 2.5
        assert f["min_tool_calls"] == 1
        assert f["max_tool_calls"] == 40

    def test_describe_filters_extended(self):
        f = build_prune_filters(_ns(model="gpt-5", provider="nous",
                                    max_cost=0.5))
        desc = describe_filters(f)
        assert "model contains 'gpt-5'" in desc
        assert "provider 'nous'" in desc
        assert "<= $0.5" in desc

    def test_describe_filters_mentions_active_parts(self):
        f = build_prune_filters(_ns(newer_than="5h", source="cli"))
        desc = describe_filters(f)
        assert "started after" in desc
        assert "source 'cli'" in desc

    def test_describe_filters_empty(self):
        f = build_prune_filters(_ns())
        assert describe_filters(f) == "no filters (all ended sessions)"
