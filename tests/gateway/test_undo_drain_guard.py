"""AC7: /undo and /redo must refuse while a /stop'd turn is still DRAINING.

Motivating follow-up (Ace, 2026-07-14): after /stop, the running-agent slot is
released immediately (so the session stays responsive), but the turn coroutine
keeps appending transcript rows until it reaches its next cooperative interrupt
checkpoint. During that window the running-agent guard does NOT fire, so a naive
/undo rewinds into a transcript the draining turn is still writing — producing a
landing the drain immediately clobbers (the exact undo-clobber incident).

The fix: _interrupt_and_clear_session records the still-mid-flight turn in
runner._draining_turns; /undo and /redo consult it via _session_turn_draining()
and refuse while the task is not done (pruning the entry once it finishes).
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from unittest.mock import AsyncMock

import pytest

import hermes_undo
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    session_store._db = db
    hermes_undo._session_db = db
    hermes_undo.clear_state()
    yield session_store
    db.close()
    hermes_undo.clear_state()


def _source(chat_id="chat-1"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


def _event(text, source):
    return MessageEvent(text=text, source=source, message_id="m1")


def _runner(store):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    return runner


class _FakeTask:
    """Stand-in for an asyncio.Task with a controllable done() state."""

    def __init__(self, done: bool):
        self._done = done

    def done(self) -> bool:
        return self._done


def _seed(db, sid):
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1 " * 50)
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2 " * 50)


def test_undo_refuses_while_turn_draining(store):
    """AC7: a not-done drain task makes /undo refuse WITHOUT rewinding."""
    src = _source("draining")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)
    runner._draining_turns = {build_session_key(src): _FakeTask(done=False)}

    before = store._db.get_messages(entry.session_id, include_inactive=False)
    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    after = store._db.get_messages(entry.session_id, include_inactive=False)

    # Refusal message, and the transcript was NOT rewound (same active rows).
    assert "still finishing" in msg or "can't run yet" in msg
    assert "Undid" not in msg
    assert len(before) == len(after), "the drain guard must not rewind the transcript"


def test_redo_refuses_while_turn_draining(store):
    src = _source("draining-redo")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)
    runner._draining_turns = {build_session_key(src): _FakeTask(done=False)}

    msg = asyncio.run(runner._handle_redo_command(_event("/redo", src)))
    assert "still finishing" in msg or "can't run yet" in msg
    assert "Redid" not in msg


def test_undo_proceeds_and_prunes_when_drain_done(store):
    """A done drain task is pruned on access and /undo proceeds normally."""
    src = _source("drain-done")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)
    key = build_session_key(src)
    runner._draining_turns = {key: _FakeTask(done=True)}

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))

    # Undo ran (not refused) and the stale drain entry was pruned.
    assert "still finishing" not in msg
    assert "Undid" in msg
    assert key not in runner._draining_turns, "a done drain entry must be pruned on access"


def test_undo_proceeds_when_no_drain_registered(store):
    """Baseline: no drain marker → /undo behaves exactly as before."""
    src = _source("no-drain")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)
    # no _draining_turns attribute at all → fail-safe path

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    assert "Undid" in msg


def test_drain_guard_fails_open_on_error(store):
    """AC7 fail-open: a task whose done() raises must NOT wedge /undo."""
    src = _source("drain-raise")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)

    class _BoomTask:
        def done(self):
            raise RuntimeError("introspection boom")

    runner._draining_turns = {build_session_key(src): _BoomTask()}

    # Must not raise, must not refuse — falls through to a real undo.
    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    assert "Undid" in msg


def test_stop_registers_only_a_not_done_drain(store):
    """The capture in _interrupt_and_clear_session registers a drain ONLY when
    the turn's task is still mid-flight; a normal (done) turn exit must NOT.

    Drives the real _interrupt_and_clear_session release path with a controlled
    _running_agent_tasks entry, asserting the _draining_turns registration is
    keyed on task.done().
    """
    src = _source("cap")
    key = build_session_key(src)
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)

    # --- case A: mid-flight task (a real /stop) → registers a drain ---
    from unittest.mock import MagicMock

    runner = _runner(store)
    runner._draining_turns = {}
    runner._running_agents = {key: MagicMock()}
    runner._running_agents_ts = {key: 0.0}
    runner._running_agent_tasks = {key: _FakeTask(done=False)}
    runner._pending_messages = {}
    runner._active_session_leases = {}
    runner._evict_cached_agent = lambda *_a, **_k: None
    runner._adapter_for_source = lambda *_a, **_k: None
    runner._invalidate_session_run_generation = lambda *a, **k: 1
    runner.session_store = store

    asyncio.run(
        runner._interrupt_and_clear_session(
            key, src, interrupt_reason="Stop requested", invalidation_reason="stop_command"
        )
    )
    assert key in runner._draining_turns, "a mid-flight /stop must register a drain"

    # --- case B: already-done task (normal turn exit) → no drain registered ---
    runner2 = _runner(store)
    runner2._draining_turns = {}
    runner2._running_agents = {key: MagicMock()}
    runner2._running_agents_ts = {key: 0.0}
    runner2._running_agent_tasks = {key: _FakeTask(done=True)}
    runner2._pending_messages = {}
    runner2._active_session_leases = {}
    runner2._evict_cached_agent = lambda *_a, **_k: None
    runner2._adapter_for_source = lambda *_a, **_k: None
    runner2._invalidate_session_run_generation = lambda *a, **k: 1
    runner2.session_store = store

    asyncio.run(
        runner2._interrupt_and_clear_session(
            key, src, interrupt_reason="Stop requested", invalidation_reason="stop_command"
        )
    )
    assert key not in runner2._draining_turns, (
        "a done task (normal turn exit) must NOT register a false drain"
    )


def test_stop_sets_persist_superseded_on_live_agent(store):
    """Append-time generation gate (Phase 2): /stop sets _persist_superseded on
    the live agent so _flush_messages_to_session_db suppresses the zombie's
    continued rows. Drives the real _interrupt_and_clear_session path."""
    from unittest.mock import MagicMock

    src = _source("supersede")
    key = build_session_key(src)
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)

    live_agent = MagicMock()
    runner = _runner(store)
    runner._draining_turns = {}
    runner._running_agents = {key: live_agent}
    runner._running_agents_ts = {key: 0.0}
    runner._running_agent_tasks = {key: _FakeTask(done=False)}
    runner._pending_messages = {}
    runner._active_session_leases = {}
    runner._evict_cached_agent = lambda *_a, **_k: None
    runner._adapter_for_source = lambda *_a, **_k: None
    runner._invalidate_session_run_generation = lambda *a, **k: 1
    runner.session_store = store

    asyncio.run(
        runner._interrupt_and_clear_session(
            key, src, interrupt_reason="Stop requested", invalidation_reason="stop_command"
        )
    )
    assert live_agent._persist_superseded is True, (
        "/stop must flag the live agent so its continued rows are suppressed"
    )


def test_reaper_direct_invalidate_leaves_persist_superseded_unset(store):
    """req-4 / R6: the stale-agent reaper calls _invalidate_session_run_generation
    DIRECTLY (not via _interrupt_and_clear_session), so it must NOT set
    _persist_superseded — the reaper path is scoped out of the gate by design.
    Guards against a future refactor silently flagging the reaper path."""
    from unittest.mock import MagicMock

    src = _source("reaper")
    key = build_session_key(src)
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)

    live_agent = MagicMock()
    live_agent._persist_superseded = False
    runner = _runner(store)
    runner._session_run_generation = {}

    # The reaper path: a bare _invalidate_session_run_generation call.
    runner._invalidate_session_run_generation(key, reason="stale_running_agent_eviction")

    # It must NOT have touched the agent's flag (it doesn't even see the agent).
    assert live_agent._persist_superseded is False, (
        "the reaper's direct _invalidate must NOT set _persist_superseded (R6 scope-out)"
    )


