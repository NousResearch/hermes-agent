"""Tests for hermes_cli/loop.py — parser, interval, and LoopManager."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.loop import (
    LoopState,
    LoopManager,
    _parse_interval,
    _parse_loop_command,
    format_interval,
    _format_countdown,
)


# ──────────────────────────────────────────────────────────────────────
# _parse_interval
# ──────────────────────────────────────────────────────────────────────


class TestParseInterval:

    def test_seconds(self):
        assert _parse_interval("60s") == 60
        assert _parse_interval("90sec") == 90
        assert _parse_interval("120seconds") == 120

    def test_minutes(self):
        assert _parse_interval("5m") == 300
        assert _parse_interval("30min") == 1800
        assert _parse_interval("1minute") == 60
        assert _parse_interval("2minutes") == 120

    def test_hours(self):
        assert _parse_interval("1h") == 3600
        assert _parse_interval("2hr") == 7200
        assert _parse_interval("3hours") == 10800

    def test_days(self):
        assert _parse_interval("1d") == 86400
        assert _parse_interval("2days") == 172800

    def test_bare_number(self):
        assert _parse_interval("120") == 120
        assert _parse_interval("0") == 0

    def test_whitespace(self):
        assert _parse_interval(" 5m ") == 300

    def test_invalid(self):
        assert _parse_interval("") is None
        assert _parse_interval("abc") is None
        assert _parse_interval("5x") is None
        assert _parse_interval(None) is None


# ──────────────────────────────────────────────────────────────────────
# format_interval
# ──────────────────────────────────────────────────────────────────────


class TestFormatInterval:

    def test_seconds(self):
        assert format_interval(30) == "30s"
        assert format_interval(59) == "59s"

    def test_minutes(self):
        assert format_interval(60) == "1m"
        assert format_interval(300) == "5m"
        assert format_interval(90) == "1m 30s"

    def test_hours(self):
        assert format_interval(3600) == "1h"
        assert format_interval(5400) == "1h 30m"

    def test_days(self):
        assert format_interval(86400) == "1d"
        assert format_interval(90000) == "1d 1h"


# ──────────────────────────────────────────────────────────────────────
# _format_countdown
# ──────────────────────────────────────────────────────────────────────


class TestFormatCountdown:

    def test_zero(self):
        assert _format_countdown(0) == "next: now"
        assert _format_countdown(-5) == "next: now"

    def test_seconds(self):
        assert _format_countdown(30) == "next: 30s"

    def test_minutes(self):
        assert _format_countdown(150) == "next: 2m 30s"

    def test_hours(self):
        assert _format_countdown(3661) == "next: 1h 1m"


# ──────────────────────────────────────────────────────────────────────
# _parse_loop_command
# ──────────────────────────────────────────────────────────────────────


class TestParseLoopCommand:

    def test_empty(self):
        assert _parse_loop_command("")["action"] == "status"
        assert _parse_loop_command(None)["action"] == "status"

    def test_status(self):
        assert _parse_loop_command("list")["action"] == "status"
        assert _parse_loop_command("status")["action"] == "status"

    def test_pause_all(self):
        assert _parse_loop_command("pause")["action"] == "pause_all"

    def test_pause_one(self):
        r = _parse_loop_command("pause #a3f1")
        assert r["action"] == "pause"
        assert r["uid"] == "a3f1"

    def test_resume_all(self):
        assert _parse_loop_command("resume")["action"] == "resume_all"

    def test_resume_one(self):
        r = _parse_loop_command("resume #b2c3")
        assert r["action"] == "resume"
        assert r["uid"] == "b2c3"

    def test_clear_all(self):
        assert _parse_loop_command("clear")["action"] == "clear_all"
        assert _parse_loop_command("stop")["action"] == "clear_all"
        assert _parse_loop_command("done")["action"] == "clear_all"

    def test_clear_one(self):
        r = _parse_loop_command("clear #d4e5")
        assert r["action"] == "clear"
        assert r["uid"] == "d4e5"

    def test_set_with_interval(self):
        r = _parse_loop_command("5m check server")
        assert r["action"] == "set"
        assert r["interval"] == 300
        assert r["prompt"] == "check server"

    def test_set_with_every_prefix(self):
        r = _parse_loop_command("every 30m check logs")
        assert r["action"] == "set"
        assert r["interval"] == 1800
        assert r["prompt"] == "check logs"

    def test_set_interval_min_clamp(self):
        r = _parse_loop_command("10s quick check")
        assert r["action"] == "set"
        assert r["interval"] == 60  # clamped to MIN_INTERVAL_SECONDS

    def test_bare_interval_no_prompt(self):
        r = _parse_loop_command("5m")
        assert r["action"] == "error"
        assert "Missing prompt" in r["message"]

    def test_unknown_subcommand(self):
        r = _parse_loop_command("foobar")
        assert r["action"] == "error"
        assert "Unknown subcommand" in r["message"]


# ──────────────────────────────────────────────────────────────────────
# LoopState serialization
# ──────────────────────────────────────────────────────────────────────


class TestLoopState:

    def test_roundtrip(self):
        s = LoopState(
            prompt="test", id="abc123", interval_seconds=300,
            status="active", last_fired_at=100.0, created_at=50.0,
            turns_completed=3,
        )
        raw = s.to_json()
        s2 = LoopState.from_json(raw)
        assert s2.prompt == "test"
        assert s2.id == "abc123"
        assert s2.interval_seconds == 300
        assert s2.status == "active"
        assert s2.last_fired_at == 100.0
        assert s2.turns_completed == 3

    def test_from_json_defaults(self):
        s = LoopState.from_json('{"prompt": "x"}')
        assert s.prompt == "x"
        assert s.id is None
        assert s.interval_seconds == 300
        assert s.status == "active"


# ──────────────────────────────────────────────────────────────────────
# LoopManager (mocked SessionDB)
# ──────────────────────────────────────────────────────────────────────


class TestLoopManager:

    def _make_db(self):
        """Return a mock SessionDB with in-memory meta store."""
        store = {}
        db = MagicMock()
        db.get_meta = lambda k: store.get(k)
        db.set_meta = lambda k, v: store.__setitem__(k, v) if v else store.pop(k, None)
        return db

    @patch("hermes_cli.loop._get_session_db")
    def test_add_creates_loop(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        state = mgr.add("check server", interval_seconds=120)
        assert state.prompt == "check server"
        assert state.interval_seconds == 120
        assert state.id is not None
        assert len(state.id) == 6
        assert state.status == "active"
        assert mgr.is_active()

    @patch("hermes_cli.loop._get_session_db")
    def test_add_empty_prompt_raises(self, mock_db_fn):
        mock_db_fn.return_value = self._make_db()
        mgr = LoopManager(session_id="test-sid")
        with pytest.raises(ValueError, match="empty"):
            mgr.add("")

    @patch("hermes_cli.loop._get_session_db")
    def test_pause_and_resume(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        state = mgr.add("test")

        paused = mgr.pause(uid=state.id)
        assert len(paused) == 1
        assert paused[0].status == "paused"
        assert not mgr.is_active()

        resumed = mgr.resume(uid=state.id)
        assert len(resumed) == 1
        assert resumed[0].status == "active"
        assert mgr.is_active()

    @patch("hermes_cli.loop._get_session_db")
    def test_clear_one(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        s1 = mgr.add("first")
        s2 = mgr.add("second")
        assert len(mgr.all_states) == 2

        count = mgr.clear(uid=s1.id)
        assert count == 1
        assert len(mgr.all_states) == 1
        assert s2.id in mgr.all_states

    @patch("hermes_cli.loop._get_session_db")
    def test_clear_all(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        mgr.add("first")
        mgr.add("second")
        assert len(mgr.all_states) == 2

        count = mgr.clear()
        assert count == 2
        assert len(mgr.all_states) == 0

    @patch("hermes_cli.loop._get_session_db")
    def test_pause_nonexistent(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        assert mgr.pause(uid="nonexistent") == []

    @patch("hermes_cli.loop._get_session_db")
    def test_status_line_empty(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        assert "No active loops" in mgr.status_line()

    @patch("hermes_cli.loop._get_session_db")
    def test_status_line_with_loops(self, mock_db_fn):
        db = self._make_db()
        mock_db_fn.return_value = db
        mgr = LoopManager(session_id="test-sid")
        state = mgr.add("check server", interval_seconds=300)
        state.last_fired_at = time.time() - 100  # 100s ago

        line = mgr.status_line()
        assert f"#{state.id}" in line
        assert "every 5m" in line
        assert "check server" in line
        assert "active" in line or "running" in line or "stopped" in line
