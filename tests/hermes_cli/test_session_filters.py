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


class TestParsePointInTime:
    def test_duration_is_relative_to_now(self):
        ts = parse_point_in_time("5h", "--before")
        assert ts == pytest.approx(time.time() - 18000, abs=5)


    def test_invalid_raises_with_flag_name(self):
        with pytest.raises(ValueError, match="--older-than"):
            parse_point_in_time("nonsense", "--older-than")


class TestBuildPruneFilters:

    def test_older_than_bare_days(self):
        f = build_prune_filters(_ns(older_than="90"))
        assert f["last_active_before"] == pytest.approx(
            time.time() - 90 * 86400, abs=5
        )
        assert f["started_before"] is None
        assert f["started_after"] is None




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




