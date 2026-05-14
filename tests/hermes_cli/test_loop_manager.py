"""Tests for hermes_cli/loop.py — background non‑blocking loop scheduler."""

from __future__ import annotations

import queue
import time
from unittest.mock import patch, MagicMock

import pytest

# Convenience imports for scheduler tests
from hermes_cli.loop import save_loop, load_loop


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
    def test_seconds(self):
        from hermes_cli.loop import _parse_interval

        assert _parse_interval("10s") == 10
        assert _parse_interval("30s") == 30
        assert _parse_interval("1sec") == 1
        assert _parse_interval("5 seconds") == 5
        assert _parse_interval("30") == 30  # bare number
        assert _parse_interval("0") == 0

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

    def test_shutdown_stops_scheduler(self, hermes_home):
        from hermes_cli.loop import LoopManager

        pq = queue.Queue()
        mgr = LoopManager(session_id="shutdown-sid")
        mgr.set("test", pending_input=pq, is_idle=lambda: True)
        assert mgr.is_running()

        mgr.shutdown()
        assert not mgr.is_running()


# ──────────────────────────────────────────────────────────────────────
# LoopScheduler
# ──────────────────────────────────────────────────────────────────────


class TestLoopScheduler:
    """Tests for the non‑blocking background scheduler.

    The scheduler replaces the old blocking evaluate_after_turn().
    It runs a daemon thread that only injects prompts when the agent
    is idle (is_idle() returns True).
    """

    def test_start_and_stop(self, hermes_home):
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        state = LoopState(prompt="test", interval_seconds=60)
        sched = LoopScheduler(
            "sid-1", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        assert not sched.running
        sched.start()
        assert sched.running
        sched.stop()
        # Give the thread a moment to exit
        time.sleep(0.05)
        assert not sched.running

    def test_stop_is_idempotent(self, hermes_home):
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        state = LoopState(prompt="test", interval_seconds=60)
        sched = LoopScheduler(
            "sid-2", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        sched.start()
        sched.stop()
        sched.stop()  # idempotent
        time.sleep(0.05)
        assert not sched.running

    def test_first_tick_fires_when_idle(self, hermes_home):
        """First tick: last_fired_at=0, so interval always 'elapsed'.
        With is_idle=True, prompt should be injected."""
        from hermes_cli.loop import LoopScheduler, LoopState, save_loop

        pq = queue.Queue()
        state = LoopState(prompt="check deploy", interval_seconds=300,
                          last_fired_at=0.0, status="active")
        save_loop("sid-3", state)

        sched = LoopScheduler(
            "sid-3", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        sched._tick()  # manual tick — bypasses sleep
        sched.stop()

        # Prompt should have been injected (without [Loop check] prefix)
        try:
            msg = pq.get(timeout=1)
        except queue.Empty:
            msg = None
        assert msg is not None
        assert "check deploy" == msg

    def test_tick_skips_when_busy(self, hermes_home):
        """When is_idle returns False, no prompt should be injected."""
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        state = LoopState(prompt="check deploy", interval_seconds=300,
                          last_fired_at=0.0, status="active")
        save_loop("sid-4", state)

        sched = LoopScheduler(
            "sid-4", state,
            pending_input=pq,
            is_idle=lambda: False,  # agent is BUSY
        )
        sched._tick()

        # Nothing should be in the queue
        assert pq.empty()
        sched.stop()

    def test_tick_skips_when_interval_not_elapsed(self, hermes_home):
        """When last_fired_at is recent, prompt should NOT fire."""
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        state = LoopState(prompt="check deploy", interval_seconds=300,
                          last_fired_at=time.time(),  # just fired
                          status="active")
        save_loop("sid-5", state)

        sched = LoopScheduler(
            "sid-5", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        sched._tick()

        assert pq.empty()
        sched.stop()

    def test_tick_updates_fire_time_and_turns(self, hermes_home):
        from hermes_cli.loop import LoopScheduler, LoopState, load_loop

        pq = queue.Queue()
        state = LoopState(prompt="test", interval_seconds=60,
                          last_fired_at=0.0, turns_completed=0,
                          status="active")
        save_loop("sid-6", state)

        sched = LoopScheduler(
            "sid-6", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        before = time.time()
        sched._tick()
        after = time.time()
        sched.stop()

        # Reload from DB to verify persistence
        reloaded = load_loop("sid-6")
        assert reloaded is not None
        assert reloaded.turns_completed == 1
        assert reloaded.last_fired_at >= before
        assert reloaded.last_fired_at <= after

    def test_tick_skips_paused(self, hermes_home):
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        state = LoopState(prompt="test", interval_seconds=60,
                          last_fired_at=0.0, status="paused")
        save_loop("sid-7", state)

        sched = LoopScheduler(
            "sid-7", state,
            pending_input=pq,
            is_idle=lambda: True,
        )
        sched._tick()
        assert pq.empty()
        sched.stop()

    def test_on_message_callback(self, hermes_home):
        """on_message was removed from _tick — verify no callback is called."""
        from hermes_cli.loop import LoopScheduler, LoopState

        pq = queue.Queue()
        messages = []

        state = LoopState(prompt="test", interval_seconds=60,
                          last_fired_at=0.0, status="active")
        save_loop("sid-8", state)

        sched = LoopScheduler(
            "sid-8", state,
            pending_input=pq,
            is_idle=lambda: True,
            on_message=lambda msg: messages.append(msg),
        )
        sched._tick()
        sched.stop()

        # on_message callback was removed from _tick
        assert len(messages) == 0

    def test_scheduler_integrated_via_loop_manager(self, hermes_home):
        """LoopManager.set() with callbacks starts the scheduler."""
        from hermes_cli.loop import LoopManager

        pq = queue.Queue()
        mgr = LoopManager(session_id="int-sid")
        assert not mgr.is_running()

        mgr.set("check", pending_input=pq, is_idle=lambda: True)
        assert mgr.is_running()

        mgr.shutdown()
        assert not mgr.is_running()

    def test_pause_stops_scheduler(self, hermes_home):
        from hermes_cli.loop import LoopManager

        pq = queue.Queue()
        mgr = LoopManager(session_id="pause-sid")
        mgr.set("test", pending_input=pq, is_idle=lambda: True)
        assert mgr.is_running()

        mgr.pause()
        assert not mgr.is_running()
        assert mgr.state.status == "paused"

    def test_clear_stops_scheduler(self, hermes_home):
        from hermes_cli.loop import LoopManager

        pq = queue.Queue()
        mgr = LoopManager(session_id="clear-sid")
        mgr.set("test", pending_input=pq, is_idle=lambda: True)
        assert mgr.is_running()

        mgr.clear()
        assert not mgr.is_running()
        assert mgr.state is None

    def test_resume_restarts_scheduler(self, hermes_home):
        from hermes_cli.loop import LoopManager

        pq = queue.Queue()
        mgr = LoopManager(session_id="resume-sid")
        mgr.set("test", pending_input=pq, is_idle=lambda: True)
        mgr.pause()
        assert not mgr.is_running()

        mgr.resume(pending_input=pq, is_idle=lambda: True)
        assert mgr.is_running()

        mgr.shutdown()


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
