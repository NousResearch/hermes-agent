"""Tests for the gateway's orphan goal recovery sweep (#81109).

On boot (and periodically) the gateway scans the profiles it serves and
enqueues continuation turns for orphaned active goals through the adapter
FIFO — the same path the post-turn goal hook uses.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _set_goal(home, session_id: str, *, owner_pid=None):
    from hermes_state import SessionDB
    from hermes_cli.goals import GoalState, save_goal

    db = SessionDB(db_path=Path(home) / "state.db")
    state = GoalState(
        goal=f"goal for {session_id}",
        status="active",
        turns_used=0,
        max_turns=5,
        created_at=time.time() - 600,
        last_turn_at=time.time() - 600,
    )
    if owner_pid is not None:
        state.owner_pid = owner_pid
        state.last_owner_seen_at = time.time() - 60
    else:
        state.last_owner_seen_at = time.time() - 3600
    save_goal(session_id, state, db=db)
    db.close()


class _RecordingAdapter:
    """Minimal adapter that records pending-message FIFO writes."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        class _R:
            success = True

        return _R()


def _make_runner(hermes_home, session_id: str, src):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={src.platform: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=src.platform,
        chat_type="dm",
        # The routing index persists origin via origin_json; a recovered
        # entry has it populated (this is what the sweep needs to find the
        # adapter + FIFO key for an orphaned session).
        origin=src,
    )

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    store._generate_session_key.return_value = build_session_key(src)
    store._entries = {build_session_key(src): session_entry}
    store._lock = MagicMock()

    def _ensure_loaded_locked():
        return None

    store._ensure_loaded_locked = _ensure_loaded_locked
    runner.session_store = store
    runner._session_db = None

    adapter = _RecordingAdapter()
    runner.adapters[src.platform] = adapter
    return runner, adapter, session_entry


def _telegram_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


@pytest.mark.asyncio
async def test_boot_sweep_enqueues_continuation_for_orphaned_goal(
    hermes_home, monkeypatch
):
    """An orphaned goal (dead owner pid) with a persisted gateway session is
    enqueued through the adapter FIFO on boot."""
    src = _telegram_source()
    session_key = build_session_key(src)
    runner, adapter, session_entry = _make_runner(hermes_home, "goal-sess-1", src)

    _set_goal(hermes_home, "goal-sess-1", owner_pid=4_000_000)

    # Serve only this home.
    monkeypatch.setattr(
        runner, "_goal_recovery_homes", lambda: [str(hermes_home)]
    )

    enqueued = await runner._run_goal_recovery_sweep()
    assert enqueued == 1
    event = adapter._pending_messages.get(session_key)
    assert event is not None
    assert "[Continuing toward your standing goal]" in event.text
    assert "goal for goal-sess-1" in event.text
    # The claim was cleared once enqueued — the loop is now active-running.
    from hermes_cli.goals import load_goal

    assert load_goal("goal-sess-1").orphaned is False


@pytest.mark.asyncio
async def test_boot_sweep_skips_goal_with_live_owner(hermes_home, monkeypatch):
    """A goal whose owner pid is still alive is never enqueued (no
    double-fire)."""
    src = _telegram_source()
    session_key = build_session_key(src)
    runner, adapter, _entry = _make_runner(hermes_home, "goal-live-1", src)

    _set_goal(hermes_home, "goal-live-1", owner_pid=os.getpid())

    monkeypatch.setattr(
        runner, "_goal_recovery_homes", lambda: [str(hermes_home)]
    )

    enqueued = await runner._run_goal_recovery_sweep()
    assert enqueued == 0
    assert adapter._pending_messages.get(session_key) is None


@pytest.mark.asyncio
async def test_boot_sweep_skips_paused_goal(hermes_home, monkeypatch):
    """A paused goal is untouched even with a dead owner pid."""
    src = _telegram_source()
    runner, adapter, _entry = _make_runner(hermes_home, "goal-paused-1", src)

    from hermes_state import SessionDB
    from hermes_cli.goals import GoalState, save_goal

    db = SessionDB(db_path=Path(hermes_home) / "state.db")
    state = GoalState(
        goal="paused goal",
        status="paused",
        turns_used=0,
        max_turns=5,
        created_at=time.time() - 600,
        last_turn_at=time.time() - 600,
    )
    state.owner_pid = 4_000_000
    state.last_owner_seen_at = time.time() - 60
    save_goal("goal-paused-1", state, db=db)
    db.close()

    monkeypatch.setattr(
        runner, "_goal_recovery_homes", lambda: [str(hermes_home)]
    )

    enqueued = await runner._run_goal_recovery_sweep()
    assert enqueued == 0
    from hermes_cli.goals import load_goal

    loaded = load_goal("goal-paused-1")
    assert loaded is not None
    assert loaded.status == "paused"
    assert loaded.orphaned is False


@pytest.mark.asyncio
async def test_boot_sweep_skips_session_without_gateway_origin(
    hermes_home, monkeypatch
):
    """A goal whose session has no persisted gateway origin (pure CLI/TUI
    row) is not this gateway's to drive — skipped."""
    src = _telegram_source()
    runner, adapter, _entry = _make_runner(hermes_home, "goal-cli-1", src)

    _set_goal(hermes_home, "goal-cli-1", owner_pid=4_000_000)

    # No matching entry in the routing index for this session id.
    runner.session_store._entries = {}

    monkeypatch.setattr(
        runner, "_goal_recovery_homes", lambda: [str(hermes_home)]
    )

    enqueued = await runner._run_goal_recovery_sweep()
    assert enqueued == 0


@pytest.mark.asyncio
async def test_sweep_loop_interval_from_config(hermes_home, monkeypatch):
    """The sweep interval resolves from goals.recovery_interval_minutes
    (0 disables the periodic loop)."""
    runner, _adapter, _entry = _make_runner(
        hermes_home, "goal-cfg-1", _telegram_source()
    )
    # Runner.config is a GatewayConfig dataclass; the goals.* knobs come
    # from the user config via hermes_cli.config.load_config.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"recovery_interval_minutes": 0, "recovery_enabled": True}},
    )
    assert runner._goal_recovery_interval_minutes() == 0
    assert runner._goal_recovery_enabled() is True
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"recovery_interval_minutes": 7, "recovery_enabled": False}},
    )
    assert runner._goal_recovery_interval_minutes() == 7
    assert runner._goal_recovery_enabled() is False
