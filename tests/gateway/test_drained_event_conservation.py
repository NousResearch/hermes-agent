"""Conservation of already-drained accepted events across transcription/shutdown/cancellation.

``_run_agent_drain_pending`` pops the oldest ACCEPTED event (A), promotes the next
overflow item (B) into the freed slot, then awaits voice transcription.  That await
is a real suspension point: a restart/shutdown can set ``_draining`` while A is owned
only by the local coroutine, and the task can be cancelled (or a transcription error
can unwind it) before A is handed to the next turn.  In every case A must be returned
to the queue in admission order (A slot, B overflow head, C behind B) — never dropped.

A restage is only legitimate while the run generation is still current: a ``/new`` or
``/stop`` invalidates the generation and clears the conversation scope while the drain
is suspended, so the old A must NOT be replayed over a successor's slot.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.platforms.event import MessageType
from gateway.session import SessionEntry, SessionSource

KEY = "agent:main:discord:group:1513247605675790346:117431298246705156"
_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id="1513247605675790346",
    chat_type="group",
    user_id="117431298246705156",
    scope_id="1480524732964278294",
)


def _source() -> SessionSource:
    return SessionSource(**_ORIGIN, message_id="1552671843494666330")


def _voice_event(text: str, message_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_source(),
        message_id=message_id,
        message_type=MessageType.VOICE,
        media_urls=["synthetic-voice.ogg"],
        media_types=["audio/ogg"],
    )


def _text_event(text: str, message_id: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id=message_id)


class _Adapter(BasePlatformAdapter):
    """Minimal real adapter: the real queue owner (slot + get_pending_message)."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_runner(monkeypatch) -> tuple:
    import gateway.run as gr

    runner = gr.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._pending_messages = {}
    runner._is_user_authorized = lambda s: True
    runner._session_db = MagicMock()
    runner._reply_anchor_for_event = lambda e: None
    runner._cache_session_source = lambda k, s: None
    runner._set_session_env = lambda c: None
    runner._restart_requested = False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._turn_leases = None

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key=KEY, session_id="sess-1",
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD, chat_type="group",
    )
    store.load_transcript.return_value = []
    store.has_platform_message_id.return_value = False
    runner.session_store = store

    adapter = _Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter._pending_messages = {}
    adapter.gateway_runner = runner
    runner._delivery_adapter_for = lambda src: adapter
    return runner, adapter


def _overflow(runner) -> list:
    return runner._peek_session_state(KEY).conversation.queued_events


def _admit(runner, adapter, *events) -> None:
    """Admit A/B/C through the REAL admission receipt: slot head then FIFO tail."""
    for event in events:
        assert runner._queue_or_replace_pending_event(KEY, event) is True


class _SttBarrier:
    """A patched STT seam the drain suspends on: entered → release, or raise."""

    def __init__(self, *, fail: bool = False):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._fail = fail

    async def __call__(self, event, adapter, source, text, *, log_context, metadata=None):
        self.entered.set()
        await self.release.wait()
        if self._fail:
            raise RuntimeError("transcription blew up")
        return ("A transcript", ["t"])


def _install_barrier(runner, barrier: _SttBarrier) -> None:
    runner._transcribe_and_echo_pending_voice = barrier


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: accepted events conserved across the transcription suspension
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("scenario", ["late_draining", "cancelled", "errored", "normal", "entry_draining", "eventless_no_adapter"])
@pytest.mark.asyncio
async def test_drain_suspension_conserves_accepted_events(monkeypatch, tmp_path, scenario):
    """A/B/C admitted, A popped and B promoted, then the STT await is interrupted.

    ``late_draining`` — ``_draining`` flips while suspended: A must be front-restaged
    (slot A, overflow [B, C]) and the drain must still run no new model turn.
    ``cancelled`` / ``errored`` — the suspension unwinds: A conserved before re-raising.
    ``normal`` — the happy path consumes A and leaves B staged, C behind it.
    ``entry_draining`` — the entry guard leaves every bucket untouched (no dequeue).
    """
    runner, adapter = _make_runner(monkeypatch)
    a, b, c = (
        _voice_event("A voice caption", "A1"),
        _text_event("B follow-up", "B1"),
        _text_event("C follow-up", "C1"),
    )
    _admit(runner, adapter, a, b, c)
    assert adapter._pending_messages[KEY] is a
    assert _overflow(runner) == [b, c]

    result = {"interrupted": True, "messages": []}

    if scenario == "eventless_no_adapter":
        generation = runner._begin_session_run_generation(KEY)
        drained = await runner._run_agent_drain_pending(
            {"pending_steer": "leftover steer"}, None, _source(), KEY)
        assert drained == (None, "leftover steer")
        assert runner._current_session_run_generation(KEY) == generation
        assert adapter._pending_messages[KEY] is a
        assert _overflow(runner) == [b, c]
        return

    if scenario == "entry_draining":
        runner._draining = True
        pending_event, pending = await runner._run_agent_drain_pending(result, adapter, _source(), KEY)
        assert (pending_event, pending) == (None, None)
        # Untouched: nothing was dequeued, A/B/C keep their places.
        assert adapter._pending_messages[KEY] is a
        assert _overflow(runner) == [b, c]
        return

    barrier = _SttBarrier(fail=(scenario == "errored"))
    _install_barrier(runner, barrier)
    task = asyncio.create_task(runner._run_agent_drain_pending(result, adapter, _source(), KEY))
    await asyncio.wait_for(barrier.entered.wait(), timeout=5.0)

    # The real drain popped A and staged B into the freed slot before suspending.
    assert adapter._pending_messages[KEY] is b
    assert _overflow(runner) == [c]

    if scenario == "late_draining":
        runner._draining = True
        barrier.release.set()
        pending_event, pending = await asyncio.wait_for(task, timeout=5.0)
        # No new model turn runs during shutdown...
        assert (pending_event, pending) == (None, None)
        # ...but A is conserved in front of the promoted B, C behind it.
        assert adapter._pending_messages[KEY] is a
        assert _overflow(runner) == [b, c]
        await _assert_shutdown_spool_conserves(tmp_path, adapter, runner, a, b, c)
    elif scenario == "cancelled":
        assert not task.done()
        assert task.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter._pending_messages[KEY] is a
        assert _overflow(runner) == [b, c]
    elif scenario == "errored":
        barrier.release.set()
        with pytest.raises(RuntimeError, match="transcription blew up"):
            await task
        assert adapter._pending_messages[KEY] is a
        assert _overflow(runner) == [b, c]
    else:  # normal
        barrier.release.set()
        pending_event, pending = await asyncio.wait_for(task, timeout=5.0)
        assert pending_event is a and pending == "A transcript"
        # A is consumed by the turn; B stays staged, C behind it.
        assert adapter._pending_messages[KEY] is b
        assert _overflow(runner) == [c]


async def _assert_shutdown_spool_conserves(tmp_path, adapter, runner, a, b, c) -> None:
    """Compose the REAL shutdown writer + REAL SQLite recovery over the restaged queue."""
    from gateway.shutdown_flush import (
        flush_overflow_to_file, flush_pending_to_file, recover_pending_spool,
    )
    from hermes_state import SessionDB

    assert flush_pending_to_file(dict(adapter._pending_messages), reason="shutdown") == 1
    assert flush_overflow_to_file({KEY: list(_overflow(runner))}, reason="shutdown") == 2

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("sess-1", "gateway")
        recovered, _held = recover_pending_spool(
            db, session_resolver=lambda key, not_after=None: ("sess-1", db))
        assert recovered == 3
        contents = [m["content"] for m in db.get_messages("sess-1")]
        assert contents == [a.text, b.text, c.text]
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: restage is fenced by the run-generation owner
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("scenario", [
    "cancel_no_invalidate",
    "cancel_invalidate",
    "normal_invalidate",
    "late_draining_invalidate",
    "steer_invalidate",
    "error_invalidate",
])
@pytest.mark.asyncio
async def test_drain_restage_fenced_by_run_generation(monkeypatch, scenario):
    """Restage/return is fenced by the run-generation owner.

    ``cancel_no_invalidate`` (original case 1) — shutdown without generation bump:
    cancelling restages A (slot A, overflow [B, C]).
    ``cancel_invalidate`` (original case 2) — ``/new`` / ``/stop`` invalidation:
    the successor's slot/overflow are untouched and the stale A is NOT revived.
    ``normal_invalidate`` / ``late_draining_invalidate`` / ``steer_invalidate`` —
    generation invalidated during the await: the drain returns (None, None) and the
    successor D/E are untouched.
    ``error_invalidate`` — generation invalidated + transcription error: the error
    re-raises and the successor D/E are untouched.
    """
    runner, adapter = _make_runner(monkeypatch)
    generation = runner._begin_session_run_generation(KEY)
    a, b, c = (
        _voice_event("A voice caption", "A1"),
        _text_event("B follow-up", "B1"),
        _text_event("C follow-up", "C1"),
    )
    _admit(runner, adapter, a, b, c)

    invalidate = scenario != "cancel_no_invalidate"
    fail = scenario == "error_invalidate"
    steer = "old steer" if scenario == "steer_invalidate" else None

    barrier = _SttBarrier(fail=fail)
    _install_barrier(runner, barrier)
    result = {"interrupted": True, "messages": [], "pending_steer": steer}
    task = asyncio.create_task(runner._run_agent_drain_pending(
        result, adapter, _source(), KEY, run_generation=generation))
    await asyncio.wait_for(barrier.entered.wait(), timeout=5.0)

    # The real drain popped A and staged B into the freed slot before suspending.
    assert adapter._pending_messages[KEY] is b
    assert _overflow(runner) == [c]

    if invalidate:
        # A real /new-style boundary: bump the generation and clear the conversation scope,
        # then install the successor work the new conversation owns via REAL admission.
        runner._invalidate_session_run_generation(KEY, reason="test")
        runner._clear_conversation_scope(KEY, reason="test")
        # The promoted B belongs to the old conversation; pop it so the successor
        # takes the slot through real admission (not a direct assignment).
        adapter._pending_messages.pop(KEY, None)
        successor, later = _text_event("successor", "D1"), _text_event("later", "E1")
        assert runner._queue_or_replace_pending_event(KEY, successor) is True
        assert runner._queue_or_replace_pending_event(KEY, later) is True

    if scenario.startswith("cancel"):
        # Assert real cancellation injection: task is pending, cancel() returns True.
        assert not task.done()
        assert task.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await task
        if invalidate:
            # Stale A must NOT be revived over successor D/E.
            assert adapter._pending_messages[KEY].text == "successor"
            assert [e.text for e in _overflow(runner)] == ["later"]
            assert a not in _overflow(runner)
            assert adapter._pending_messages[KEY] is not a
        else:
            # No invalidation: A is conserved in front of promoted B, C behind it.
            assert adapter._pending_messages[KEY] is a
            assert _overflow(runner) == [b, c]
    elif scenario == "late_draining_invalidate":
        runner._draining = True
        barrier.release.set()
        pending_event, pending = await asyncio.wait_for(task, timeout=5.0)
        assert (pending_event, pending) == (None, None)
        assert adapter._pending_messages[KEY].text == "successor"
        assert [e.text for e in _overflow(runner)] == ["later"]
    else:
        barrier.release.set()
        if scenario == "error_invalidate":
            with pytest.raises(RuntimeError, match="transcription blew up"):
                await task
        else:
            pending_event, pending = await asyncio.wait_for(task, timeout=5.0)
            assert (pending_event, pending) == (None, None)
        assert adapter._pending_messages[KEY].text == "successor"
        assert [e.text for e in _overflow(runner)] == ["later"]
