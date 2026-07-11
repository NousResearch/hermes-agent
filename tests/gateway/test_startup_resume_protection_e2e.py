"""Real-seam integration test for boot-resume interrupt protection (#284).

2026-07-10 20:17 live E2E FAILED while all 8 unit tests were green: the units
injected ``runner._startup_resume_active`` directly and tested the busy-path
branch in isolation, so the marker's *lifetime* vs the real dispatch chain was
never exercised.  This test crosses the production seam:

  _run_startup_resume_event (real wrapper)
    -> BasePlatformAdapter.handle_message (REAL, not mocked)
      -> _start_session_processing -> _process_message_background
        -> message handler (fake agent turn, blocks like `sleep 90`)
  user MessageEvent -> adapter.handle_message (REAL busy path)
    -> runner._handle_active_session_busy_message (real)

No direct attribute injection anywhere: if the marker dies before the turn
does (the live failure), these tests go red for the same reason production
failed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL  # noqa: E402


class RecordingAdapter(BasePlatformAdapter):
    """Real adapter machinery; only the wire transport is stubbed."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(str(content))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id: str = "777", user_id: str = "user1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_id=user_id,
    )


def _user_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m1",
    )


def _make_runner(session_key: str) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    entry = SimpleNamespace(session_key=session_key, session_id="sess-1")
    session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={session_key: entry},
        switch_session=MagicMock(),
        clear_resume_pending=MagicMock(),
    )
    session_store._ensure_loaded_locked = lambda: None
    runner.session_store = session_store
    runner._session_db = MagicMock()
    runner._session_db._db = MagicMock()
    runner._session_db._db.get_compression_lock_holder.return_value = None
    # Bind the REAL turn-exit chokepoint (marker lifecycle lives there) —
    # stubbing it would hide exactly the seam this file exists to test.
    runner._active_session_leases = {}
    runner._running_agent_tasks = {}
    runner._persist_active_agents = lambda: None
    runner._release_running_agent_state = (
        GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    )
    return runner


@pytest.mark.asyncio
async def test_marker_survives_for_the_lifetime_of_the_real_turn():
    """THE incident, deterministically.

    The recovery turn blocks (like `sleep 90`).  34 simulated seconds later a
    user message hits the REAL adapter busy path.  The marker must still be
    set — pre-fix it was discarded as soon as the wrapper's await chain
    detached from the actual turn, so interrupt mode killed the recovery.
    """
    adapter = RecordingAdapter()
    ev = _user_event("")  # synthetic resume event shape
    ev.internal = True
    sk = build_session_key(ev.source)

    runner = _make_runner(sk)
    runner.adapters[Platform.TELEGRAM] = adapter

    turn_started = asyncio.Event()
    turn_gate = asyncio.Event()
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 1, "max_iterations": 300, "current_tool": "terminal",
    }

    async def fake_agent_turn(event):
        # what _handle_message does: claim the running slot, run the turn
        runner._running_agents[sk] = parent
        turn_started.set()
        await turn_gate.wait()          # <- the sleep 90 window
        runner._release_running_agent_state(sk)
        return "recovery report posted"

    adapter.set_message_handler(fake_agent_turn)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    resume_task = asyncio.create_task(
        runner._run_startup_resume_event(adapter, ev, sk)
    )
    await asyncio.wait_for(turn_started.wait(), timeout=5)
    # Give the wrapper every chance to (wrongly) discard the marker early.
    await asyncio.sleep(0.1)

    # The marker-lifetime contract: turn alive => marker set.
    assert runner._session_in_startup_resume(sk) is True, (
        "marker discarded while the recovery turn is still running — "
        "this is the 2026-07-10 20:17 live failure"
    )

    # Ace's "How's it going?" through the REAL adapter entry point.
    poke = _user_event("How's it going?")
    await adapter.handle_message(poke)

    parent.interrupt.assert_not_called()
    acks = [s for s in adapter.sent if "restarted" in s.lower() or "Interrupting" in s]
    assert acks, f"no busy ack sent; sent={adapter.sent!r}"
    assert any("restarted" in s.lower() for s in acks), (
        f"old interrupt ack sent instead of restart-aware ack: {acks!r}"
    )
    assert not any("Interrupting" in s for s in acks)

    # Recovery completes -> protection window closes.
    turn_gate.set()
    await asyncio.wait_for(resume_task, timeout=5)
    assert runner._session_in_startup_resume(sk) is False


@pytest.mark.asyncio
async def test_protection_window_closes_after_recovery():
    """After the recovery turn finishes, a user message interrupts normally
    again — the guard must not outlive the turn."""
    adapter = RecordingAdapter()
    ev = _user_event("")
    ev.internal = True
    sk = build_session_key(ev.source)

    runner = _make_runner(sk)
    runner.adapters[Platform.TELEGRAM] = adapter

    turn_gate = asyncio.Event()
    turn_started = asyncio.Event()
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 2, "max_iterations": 300, "current_tool": None,
    }

    async def fake_agent_turn(event):
        runner._running_agents[sk] = parent
        turn_started.set()
        await turn_gate.wait()
        runner._release_running_agent_state(sk)
        return "done"

    adapter.set_message_handler(fake_agent_turn)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    resume_task = asyncio.create_task(
        runner._run_startup_resume_event(adapter, ev, sk)
    )
    await asyncio.wait_for(turn_started.wait(), timeout=5)
    turn_gate.set()
    await asyncio.wait_for(resume_task, timeout=5)

    # New ordinary turn (not a recovery) — now a follow-up must interrupt.
    turn_gate2 = asyncio.Event()
    started2 = asyncio.Event()

    async def ordinary_turn(event):
        runner._running_agents[sk] = parent
        started2.set()
        await turn_gate2.wait()
        runner._release_running_agent_state(sk)
        return "ok"

    adapter.set_message_handler(ordinary_turn)
    await adapter.handle_message(_user_event("start ordinary work"))
    await asyncio.wait_for(started2.wait(), timeout=5)

    await adapter.handle_message(_user_event("actually stop"))
    parent.interrupt.assert_called_once()
    turn_gate2.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_marker_survives_wrapper_cancellation():
    """H1-cancel: if the _run_startup_resume_event wrapper task is CANCELLED
    (startup-restore teardown, timeout gating) while the shielded turn keeps
    running, the finally must NOT strip protection from the still-running
    recovery turn."""
    adapter = RecordingAdapter()
    ev = _user_event("")
    ev.internal = True
    sk = build_session_key(ev.source)

    runner = _make_runner(sk)
    runner.adapters[Platform.TELEGRAM] = adapter

    turn_started = asyncio.Event()
    turn_gate = asyncio.Event()
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 1, "max_iterations": 300, "current_tool": "terminal",
    }

    async def fake_agent_turn(event):
        runner._running_agents[sk] = parent
        turn_started.set()
        await turn_gate.wait()
        runner._release_running_agent_state(sk)
        return "recovery report"

    adapter.set_message_handler(fake_agent_turn)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    resume_task = asyncio.create_task(
        runner._run_startup_resume_event(adapter, ev, sk)
    )
    await asyncio.wait_for(turn_started.wait(), timeout=5)
    await asyncio.sleep(0.05)

    # The wrapper is cancelled; the shielded inner turn keeps running.
    resume_task.cancel()
    try:
        await resume_task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)

    assert sk in runner._running_agents, "turn should still be running (shielded)"
    assert runner._session_in_startup_resume(sk) is True, (
        "wrapper cancellation stripped protection from a live recovery turn"
    )

    poke = _user_event("How's it going?")
    await adapter.handle_message(poke)
    parent.interrupt.assert_not_called()

    turn_gate.set()
    await asyncio.sleep(0.1)
    assert runner._session_in_startup_resume(sk) is False, (
        "marker must clear once the turn actually finishes"
    )


@pytest.mark.asyncio
async def test_marker_survives_session_task_lookup_miss():
    """H1a: adapter._session_tasks entry rotated/absent when the wrapper looks
    it up -> wrapper awaits nothing and its finally runs immediately.  The
    still-running turn must stay protected."""
    adapter = RecordingAdapter()
    ev = _user_event("")
    ev.internal = True
    sk = build_session_key(ev.source)

    runner = _make_runner(sk)
    runner.adapters[Platform.TELEGRAM] = adapter

    turn_started = asyncio.Event()
    turn_gate = asyncio.Event()
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 1, "max_iterations": 300, "current_tool": "terminal",
    }

    async def fake_agent_turn(event):
        runner._running_agents[sk] = parent
        # Simulate the rotation: the owner-task entry disappears from the
        # adapter map while the turn continues (drain-task swap, key rotation).
        adapter._session_tasks.pop(sk, None)
        turn_started.set()
        await turn_gate.wait()
        runner._release_running_agent_state(sk)
        return "recovery report"

    adapter.set_message_handler(fake_agent_turn)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    resume_task = asyncio.create_task(
        runner._run_startup_resume_event(adapter, ev, sk)
    )
    await asyncio.wait_for(turn_started.wait(), timeout=5)
    # Let the wrapper's handle_message return + its finally (if buggy) run.
    await asyncio.sleep(0.1)

    assert runner._session_in_startup_resume(sk) is True, (
        "task-lookup miss discarded the marker while the turn is running — "
        "the 2026-07-10 20:17 live failure mode"
    )

    poke = _user_event("How's it going?")
    await adapter.handle_message(poke)
    parent.interrupt.assert_not_called()
    assert any("restarted" in s.lower() for s in adapter.sent), adapter.sent

    turn_gate.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_marker_survives_sentinel_phase_fast_dispatch():
    """AEGIS-RIG live trace (2026-07-11 08:54): the adapter task the wrapper
    shields is NOT the agent turn.  The gateway handler spawns the turn on its
    own per-message task and RETURNS, so the wrapper's finally ran ~7ms after
    dispatch while the slot still held the scheduler's pre-claim SENTINEL; the
    real turn registered its agent ~40ms later.  Discarding the marker on
    sentinel stripped protection from every recovery turn on this timing.

    Sequence: pre-claim sentinel (as _schedule_resume_pending_sessions does)
    -> wrapper dispatch where the handler returns immediately and the turn
    runs detached -> wrapper finally sees SENTINEL -> marker must survive ->
    poke during the running turn gets the demotion, not an interrupt.
    """
    adapter = RecordingAdapter()
    ev = _user_event("")
    ev.internal = True
    sk = build_session_key(ev.source)

    runner = _make_runner(sk)
    runner.adapters[Platform.TELEGRAM] = adapter

    turn_running = asyncio.Event()
    turn_gate = asyncio.Event()
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 1, "max_iterations": 300, "current_tool": "terminal",
    }

    detached: list[asyncio.Task] = []

    async def detached_turn():
        # the real agent registers AFTER the dispatch path has fully unwound
        await asyncio.sleep(0.04)
        runner._running_agents[sk] = parent
        turn_running.set()
        await turn_gate.wait()
        runner._release_running_agent_state(sk)

    async def fast_handler(event):
        # production shape: spawn the turn, return immediately
        detached.append(asyncio.create_task(detached_turn()))
        return None

    adapter.set_message_handler(fast_handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    # scheduler pre-claim: slot holds the sentinel before the wrapper spawns
    runner._running_agents[sk] = _AGENT_PENDING_SENTINEL
    runner._running_agents_ts[sk] = 0.0

    await runner._run_startup_resume_event(adapter, ev, sk)
    # Wrapper has fully returned during the sentinel window.  The contract:
    # the SLOT is handed back (the late-arrival drain's re-dispatch must be
    # able to claim it — test_auto_resume_runs_agent_exactly_once_through_
    # full_path), but the protection MARKER must stay armed for the incoming
    # recovery turn.
    assert runner._session_in_startup_resume(sk) is True, (
        "wrapper finally discarded the marker during the SENTINEL phase — "
        "the AEGIS-RIG 2026-07-11 live failure"
    )
    assert runner._running_agents.get(sk) is not _AGENT_PENDING_SENTINEL, (
        "pre-claimed slot must be handed back so the drain re-dispatch can claim it"
    )

    await asyncio.wait_for(turn_running.wait(), timeout=5)
    assert runner._session_in_startup_resume(sk) is True

    # With the turn now live, the adapter holds its session guard (in
    # production the per-message task keeps it for the turn's duration; our
    # fast handler returned early, so re-install it and a live owner task so
    # the poke routes to the busy path instead of a fresh dispatch).
    adapter._active_sessions[sk] = asyncio.Event()
    adapter._session_tasks[sk] = detached[0]

    poke = _user_event("How's it going?")
    await adapter.handle_message(poke)
    parent.interrupt.assert_not_called()
    assert any("restarted" in s.lower() for s in adapter.sent), adapter.sent

    turn_gate.set()
    await asyncio.sleep(0.1)
    assert runner._session_in_startup_resume(sk) is False, (
        "marker must clear at the turn chokepoint once the turn finishes"
    )
