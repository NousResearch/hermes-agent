"""Tests for hermes_cli/loop.py — persistent timer-driven loops."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the loop-module's DB cache for each test.
    from hermes_cli import loop

    loop._DB_CACHE.clear()
    yield home
    loop._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_interval
# ──────────────────────────────────────────────────────────────────────


class TestParseInterval:
    def test_minutes(self):
        from hermes_cli.loop import _parse_interval

        assert _parse_interval("5m") == 300
        assert _parse_interval("30m") == 1800
        assert _parse_interval("1min") == 60
        assert _parse_interval("2 mins") == 120

    def test_hours(self):
        from hermes_cli.loop import _parse_interval

        assert _parse_interval("1h") == 3600
        assert _parse_interval("2hr") == 7200
        assert _parse_interval("3 hours") == 10800

    def test_days(self):
        from hermes_cli.loop import _parse_interval

        assert _parse_interval("1d") == 86400
        assert _parse_interval("2 days") == 172800

    def test_invalid_returns_none(self):
        from hermes_cli.loop import _parse_interval

        assert _parse_interval("") is None
        assert _parse_interval("abc") is None
        assert _parse_interval("5x") is None
        assert _parse_interval("5") is None


# ──────────────────────────────────────────────────────────────────────
# LoopState serialization
# ──────────────────────────────────────────────────────────────────────


class TestLoopState:
    def test_to_json_roundtrip(self):
        from hermes_cli.loop import LoopState

        state = LoopState(
            prompt="check deployment",
            interval_seconds=300,
            status="active",
            last_fired_at=12345.0,
            created_at=12340.0,
            turns_completed=3,
        )
        raw = state.to_json()
        restored = LoopState.from_json(raw)
        assert restored.prompt == state.prompt
        assert restored.interval_seconds == state.interval_seconds
        assert restored.status == state.status
        assert restored.last_fired_at == state.last_fired_at
        assert restored.created_at == state.created_at
        assert restored.turns_completed == state.turns_completed

    def test_from_json_defaults(self):
        from hermes_cli.loop import LoopState

        restored = LoopState.from_json('{"prompt": "test"}')
        assert restored.prompt == "test"
        assert restored.interval_seconds == 300
        assert restored.status == "active"
        assert restored.last_fired_at == 0.0
        assert restored.created_at == 0.0
        assert restored.turns_completed == 0


# ──────────────────────────────────────────────────────────────────────
# LoopManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestLoopManager:
    def test_no_loop_initial(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-1")
        assert mgr.state is None
        assert not mgr.is_active()
        assert "No active loop" in mgr.status_line()

    def test_set_creates_state(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("check deployment")
        assert state.prompt == "check deployment"
        assert state.status == "active"
        assert state.interval_seconds == 300
        assert state.turns_completed == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "check deployment" in mgr.status_line()

    def test_set_with_interval(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-3")
        state = mgr.set("check deployment", interval_seconds=600)
        assert state.interval_seconds == 600

    def test_set_rejects_empty(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-4")
        with pytest.raises(ValueError):
            mgr.set("")
        with pytest.raises(ValueError):
            mgr.set("   ")

    def test_pause_and_resume(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-5")
        mgr.set("loop text")
        mgr.pause(reason="user-paused")
        assert mgr.state.status == "paused"
        assert not mgr.is_active()

        mgr.resume()
        assert mgr.state.status == "active"
        assert mgr.is_active()

    def test_clear(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="test-sid-6")
        mgr.set("loop")
        mgr.clear()
        assert mgr.state is None
        assert not mgr.is_active()

    def test_persistence_across_managers(self, hermes_home):
        """A second manager on the same session sees the loop."""
        from hermes_cli.loop import LoopManager

        mgr1 = LoopManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = LoopManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.prompt == "do the thing"
        assert mgr2.is_active()

    def test_status_line_paused(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="status-sid")
        mgr.set("test")
        mgr.pause()
        line = mgr.status_line()
        assert "paused" in line.lower()
        assert "test" in line

    def test_status_line_none(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="empty-sid")
        assert "No active loop" in mgr.status_line()


# ──────────────────────────────────────────────────────────────────────
# evaluate_after_turn
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateAfterTurn:
    def test_inactive_no_state(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-1")
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

    def test_inactive_when_paused(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-2")
        mgr.set("test")
        mgr.pause()
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

    def test_interval_not_elapsed(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-3")
        mgr.set("test", interval_seconds=300)
        # Simulate a recent fire
        mgr.state.last_fired_at = time.time()
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["should_continue"] is False
        assert d["verdict"] == "fired"
        assert "interval not elapsed" in d["reason"]

    def test_interval_elapsed_first_fire(self, hermes_home):
        """First fire: last_fired_at is 0, so interval is always 'elapsed'."""
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-4", default_max_turns=5)
        mgr.set("check deployment")
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["should_continue"] is True
        assert d["verdict"] == "fired"
        assert "[Loop check] check deployment" == d["continuation_prompt"]
        assert mgr.state.turns_completed == 1
        assert mgr.state.last_fired_at > 0

    def test_interval_elapsed_subsequent_fire(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-5", default_max_turns=5)
        mgr.set("check deployment", interval_seconds=300)
        # First fire
        mgr.evaluate_after_turn(user_initiated=True)
        assert mgr.state.turns_completed == 1
        # Reset last_fired_at to simulate interval elapsed
        mgr.state.last_fired_at = 0
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["should_continue"] is True
        assert d["verdict"] == "fired"
        assert mgr.state.turns_completed == 2

    def test_budget_exhausted(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-6", default_max_turns=3)
        mgr.set("hard loop")
        # Call 1: turns_completed 0→1, interval passes, fire
        mgr.evaluate_after_turn(user_initiated=True)
        mgr.state.last_fired_at = 0
        # Call 2: turns_completed 1→2, interval passes, fire
        mgr.evaluate_after_turn(user_initiated=True)
        mgr.state.last_fired_at = 0
        # Call 3: turns_completed 2→3, budget 3 >= 3 → pause
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["should_continue"] is False
        assert d["verdict"] == "budget"
        assert mgr.state.status == "paused"
        assert "turns used" in d["message"]

    def test_user_initiated_false_increments_turns(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-7", default_max_turns=10)
        mgr.set("test")
        d = mgr.evaluate_after_turn(user_initiated=False)
        assert d["should_continue"] is True
        # All turns increment turns_completed
        assert mgr.state.turns_completed == 1

    def test_message_on_fired(self, hermes_home):
        from hermes_cli.loop import LoopManager

        mgr = LoopManager(session_id="eval-sid-8", default_max_turns=5)
        mgr.set("check logs")
        d = mgr.evaluate_after_turn(user_initiated=True)
        assert d["message"] != ""
        assert "Loop check" in d["message"]
        assert "check logs" in d["message"]


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_loop_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("loop")
    assert cmd is not None
    assert cmd.name == "loop"


def test_loop_command_category():
    from hermes_cli.commands import COMMANDS_BY_CATEGORY

    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/loop" in session_cmds
