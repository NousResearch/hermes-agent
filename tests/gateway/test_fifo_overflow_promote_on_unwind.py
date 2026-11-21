"""Regression tests for the eager FIFO overflow promotion on turn unwind.

``_run_agent_drain_pending`` promotes the overflow head only when the turn produced a result
(``if result and adapter and session_key``).  A turn that unwinds through an exception, a
cancellation or an early exit never reaches that site, so events sitting in
``SessionState.conversation.queued_events`` (the #28503 FIFO overflow) are orphaned until a NEW
inbound event for a NON-busy session triggers the lazy ``_rescue_orphaned_overflow`` (#99882).
For machine-generated follow-ups — plugin ``inject_message`` turns, async-delegation completions,
``/queue`` items — that next inbound may never arrive.

``GatewayRunner._promote_orphaned_overflow_on_unwind`` promotes the head into the adapter's
pending slot at unwind, so the adapter's existing finally-block late-arrival drain
(``_finish_session_task`` → ``_spawn_drain_task``) dispatches it as a fresh turn.  It is a no-op
on the success path, where the slot is already staged.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    PlatformConfig,
)
from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _text_event(text: str, msg_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=MagicMock(chat_id="123", platform=Platform.TELEGRAM, profile=None),
        message_id=msg_id,
    )


def _runner(adapter) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    runner._draining = False
    runner._adapter_for_source = lambda _s: adapter
    return runner


def _overflow(runner, session_key):
    return runner._session_state(session_key).conversation.queued_events


class TestPromoteOrphanedOverflowOnUnwind:
    def test_unwind_promotes_overflow_head_into_empty_slot_in_fifo_order(self):
        adapter = _StubAdapter()
        runner = _runner(adapter)
        session_key = "telegram:user:1"
        e1, e2 = _text_event("queued-1", "q1"), _text_event("queued-2", "q2")
        _overflow(runner, session_key).extend([e1, e2])
        assert session_key not in adapter._pending_messages

        promoted = runner._promote_orphaned_overflow_on_unwind(session_key, MagicMock())

        assert promoted is True
        # Oldest first: the head moves into the slot, the rest stay queued behind it (#28503).
        assert adapter._pending_messages[session_key] is e1
        assert _overflow(runner, session_key) == [e2]

    def test_unwind_noop_when_slot_already_staged(self):
        """Success-path guarantee: _promote_queued_event already staged the next-up event."""
        adapter = _StubAdapter()
        runner = _runner(adapter)
        session_key = "telegram:user:2"
        staged = _text_event("staged", "s0")
        adapter._pending_messages[session_key] = staged
        e1 = _text_event("queued-1", "q1")
        _overflow(runner, session_key).append(e1)

        promoted = runner._promote_orphaned_overflow_on_unwind(session_key, MagicMock())

        assert promoted is False
        assert adapter._pending_messages[session_key] is staged
        assert _overflow(runner, session_key) == [e1]

    def test_unwind_noop_when_overflow_empty(self):
        adapter = _StubAdapter()
        runner = _runner(adapter)
        session_key = "telegram:user:3"

        promoted = runner._promote_orphaned_overflow_on_unwind(session_key, MagicMock())

        assert promoted is False
        assert session_key not in adapter._pending_messages

    def test_unwind_noop_while_draining(self):
        """Shutdown flush owns pending state — promotion must not race it."""
        adapter = _StubAdapter()
        runner = _runner(adapter)
        runner._draining = True
        session_key = "telegram:user:4"
        e1 = _text_event("queued-1", "q1")
        _overflow(runner, session_key).append(e1)

        promoted = runner._promote_orphaned_overflow_on_unwind(session_key, MagicMock())

        assert promoted is False
        assert session_key not in adapter._pending_messages
        assert _overflow(runner, session_key) == [e1]

    def test_unwind_noop_when_adapter_missing(self):
        runner = _runner(None)
        session_key = "telegram:user:5"
        e1 = _text_event("queued-1", "q1")
        _overflow(runner, session_key).append(e1)

        promoted = runner._promote_orphaned_overflow_on_unwind(session_key, MagicMock())

        assert promoted is False
        assert _overflow(runner, session_key) == [e1]


class TestCleanupTurnTasksCallsPromotion:
    @pytest.mark.asyncio
    async def test_cleanup_turn_tasks_promotes_when_turn_did_not_drain(self):
        """The real unwind path must reach the promotion, not just the helper in isolation."""
        adapter = _StubAdapter()
        runner = _runner(adapter)
        session_key = "telegram:user:6"
        e1 = _text_event("queued-1", "q1")
        _overflow(runner, session_key).append(e1)

        runner._release_running_agent_state = MagicMock()
        runner._await_stream_task = AsyncMock()
        runner._update_runtime_status = MagicMock()

        turn_ctx = TurnContext(
            source=MagicMock(),
            session_key=session_key,
            run_generation=1,
            stream_consumer_holder=[None],
            streaming_tts_consumer_holder=[None],
        )

        async def _done_task():
            task = asyncio.create_task(asyncio.sleep(0))
            await task
            return task

        await runner._run_agent_cleanup_turn_tasks(
            turn_ctx,
            progress_task=None,
            log_task=None,
            interrupt_monitor=await _done_task(),
            _notify_task=await _done_task(),
            tracking_task=await _done_task(),
            stream_task=None,
        )

        assert adapter._pending_messages[session_key] is e1
        assert _overflow(runner, session_key) == []


class TestAdapterDrainsPromotedEvent:
    @pytest.mark.asyncio
    async def test_adapter_finally_drains_promoted_event(self):
        """The adapter side of the contract: a handler that dies with the slot staged still drains.

        Mirrors what the runner does at unwind — the promoted event lands in
        ``_pending_messages`` before the handler unwinds — and asserts the adapter's ``finally``
        (``_finish_session_task``) hands it to a fresh turn via ``_spawn_drain_task``.
        """
        adapter = _StubAdapter()
        session_key = "telegram:user:7"
        promoted = _text_event("queued-1", "q1")

        async def _dying_handler(event):
            adapter._pending_messages[session_key] = promoted
            raise RuntimeError("turn unwound")

        adapter._message_handler = _dying_handler
        adapter._spawn_drain_task = MagicMock()

        await adapter._process_message_background(_text_event("first", "m1"), session_key)

        adapter._spawn_drain_task.assert_called_once_with(promoted, session_key)
