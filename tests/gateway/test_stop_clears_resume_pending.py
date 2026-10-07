"""Regression: /stop retires the restart-recovery marker.

A gateway restart that interrupts a turn marks the session ``resume_pending``; only a turn that
completes successfully clears it. A /stop ends the (often resumed) turn as "interrupted", so
without this the marker survived and the NEXT restart auto-resumed the work the user had
explicitly stopped. It is cleared in ``_interrupt_and_clear_session`` (where the busy-path and
dispatched /stop routes converge) and up front in the /stop handler, which also covers a /stop with
nothing running. Both are driven for real here (stubs as in test_agent_loop_stopped_hook.py).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _Store:
    """A session with one resume marker, identified as the real store identifies it."""

    def __init__(self, session_key, marker="A"):
        self._key = session_key
        self.marker = marker
        self.cleared = []

    def get_or_create_session(self, source):
        return SimpleNamespace(session_key=self._key, session_id="s1")

    def peek_resume_marker(self, session_key):
        return self.marker

    def clear_resume_pending(self, session_key, *, only_marker=None):
        if self.marker is None or (only_marker is not None and only_marker != self.marker):
            return False
        self.cleared.append(session_key)
        self.marker = None
        return True


def _setup():
    from gateway.run import GatewayRunner

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    key = build_session_key(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send=AsyncMock())}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {key: MagicMock()}
    runner._pending_messages = {}
    runner._invalidate_session_run_generation = lambda *a, **kw: None
    runner._release_running_agent_state = lambda *a, **kw: None
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner.session_store = _Store(key)
    return runner, source, key


@pytest.mark.asyncio
async def test_busy_path_stop_clears_resume_pending():
    """/stop while the agent is running: the busy-path handler (the common case)."""
    runner, source, key = _setup()
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    await runner._busy_stop_command(event, key, source)
    assert runner.session_store.cleared == [key]


@pytest.mark.asyncio
async def test_dispatched_stop_with_running_agent_clears_resume_pending():
    """The /stop handler reached with an agent under the caller's own key (the fallback route)."""
    runner, source, key = _setup()
    await runner._handle_stop_command(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))
    assert key in runner.session_store.cleared


@pytest.mark.asyncio
async def test_idle_stop_with_nothing_running_clears_resume_pending(monkeypatch):
    """Nothing running anywhere, yet the marker can still be set (auto-resume skipped because the
    adapter was not ready, or the restart-loop breaker tripped): /stop must retire it anyway."""
    import tools.async_delegation as async_delegation

    monkeypatch.setattr(async_delegation, "interrupt_for_session", lambda **kw: False)
    runner, source, key = _setup()
    runner._running_agents = {}
    reply = await runner._handle_stop_command(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))
    assert reply == t("gateway.stop.no_active")
    assert runner.session_store.cleared == [key]


@pytest.mark.asyncio
async def test_other_interrupts_keep_the_marker():
    """Only a user's stop retires it; e.g. the /new fast path or a shutdown interrupt must not."""
    runner, source, key = _setup()
    await runner._interrupt_and_clear_session(key, source, interrupt_reason="some other reason",
                                              invalidation_reason="test")
    assert runner.session_store.cleared == []


@pytest.mark.asyncio
async def test_stop_survives_a_failing_clear():
    runner, source, key = _setup()

    def _boom(session_key, **kw):
        raise RuntimeError("store unavailable")

    runner.session_store.clear_resume_pending = _boom
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    assert await runner._busy_stop_command(event, key, source)


class _SuccessorAdapter:
    """An adapter whose interrupt await lets a successor run set its own fresh marker."""

    def __init__(self, store):
        self.store = store
        self.send = AsyncMock()

    async def interrupt_session_activity(self, session_key, chat_id):
        self.store.marker = "B"


@pytest.mark.asyncio
async def test_successor_marker_survives_the_stop_tail():
    """Marker A, /stop starts, a successor sets marker B during the stop's awaits, /stop finishes:
    B must survive (the stop retires only the marker it saw)."""
    runner, source, key = _setup()
    runner.adapters = {Platform.TELEGRAM: _SuccessorAdapter(runner.session_store)}
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    await runner._busy_stop_command(event, key, source)
    assert runner.session_store.marker == "B"
    assert runner.session_store.cleared == []


@pytest.mark.asyncio
async def test_stop_with_no_marker_leaves_a_later_one():
    """Nothing to retire when /stop began: a marker set meanwhile is not the stop's to clear."""
    runner, source, key = _setup()
    runner.session_store.marker = None
    runner.adapters = {Platform.TELEGRAM: _SuccessorAdapter(runner.session_store)}
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    await runner._busy_stop_command(event, key, source)
    assert runner.session_store.marker == "B"


# The store's side: each marker has its own identity, and a compare-clear removes only that one.

def _real_store(tmp_path):
    from gateway.session import SessionStore
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _real_session(store):
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    return store.get_or_create_session(source).session_key


def test_each_marker_has_its_own_identity(tmp_path):
    store = _real_store(tmp_path)
    key = _real_session(store)
    assert store.peek_resume_marker(key) is None
    store.mark_resume_pending(key)
    first = store.peek_resume_marker(key)
    store.mark_resume_pending(key)
    assert first and store.peek_resume_marker(key) not in (None, first)


def test_compare_clear_keeps_a_newer_marker(tmp_path):
    store = _real_store(tmp_path)
    key = _real_session(store)
    store.mark_resume_pending(key)
    stale = store.peek_resume_marker(key)
    store.mark_resume_pending(key)
    assert store.clear_resume_pending(key, only_marker=stale) is False
    assert store._entries[key].resume_pending is True
    assert store.clear_resume_pending(key, only_marker=store.peek_resume_marker(key)) is True
    assert store._entries[key].resume_pending is False
    assert store._entries[key].resume_marker_id is None


def test_identical_timestamps_are_still_told_apart(tmp_path):
    store = _real_store(tmp_path)
    key = _real_session(store)
    store.mark_resume_pending(key)
    stale = store.peek_resume_marker(key)
    at = store._entries[key].last_resume_marked_at
    store.mark_resume_pending(key)
    store._entries[key].last_resume_marked_at = at   # the same instant, a different marker
    assert store.clear_resume_pending(key, only_marker=stale) is False


def test_pre_upgrade_marker_is_identified_by_its_time(tmp_path):
    """A row written before the token existed still clears by the marker the stop saw."""
    store = _real_store(tmp_path)
    key = _real_session(store)
    store.mark_resume_pending(key)
    store._entries[key].resume_marker_id = None
    seen = store.peek_resume_marker(key)
    assert seen and seen.startswith("legacy:")
    assert store.clear_resume_pending(key, only_marker=seen) is True


def test_marker_identity_survives_a_save_and_load(tmp_path):
    from gateway.session import SessionEntry
    store = _real_store(tmp_path)
    key = _real_session(store)
    store.mark_resume_pending(key)
    entry = store._entries[key]
    again = SessionEntry.from_dict(entry.to_dict())
    assert again.resume_marker_id == entry.resume_marker_id
