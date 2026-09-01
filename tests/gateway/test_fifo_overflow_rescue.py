"""Regression tests for #99882: FIFO overflow orphan rescue.

When a follow-up is demoted to /queue during compression-in-flight,
it lands in SessionState.conversation.queued_events (overflow) with
the current turn's event occupying adapter._pending_messages[session_key]
(slot).  After the slot's turn completes, _promote_queued_event moves
the overflow head into the slot.  When that drain never runs — the
compression window ended through an exit that skipped the promotion
site — the overflow is silently orphaned: never dispatched, never
persisted, never logged.

The rescue in GatewayRunner._rescue_orphaned_overflow stages one orphan
into the slot on the next idle arrival, so FIFO order (#28503) holds.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    PlatformConfig,
)
from gateway.run import GatewayRunner


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


class TestRescueOrphanedOverflow:
    def test_moves_overflow_head_to_empty_slot(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        # Minimal session_state with queued_events
        adapter = _StubAdapter()
        session_key = "telegram:user:1"
        # Two overflow items orphaned after slot turn completed
        runner._session_state(session_key).conversation.queued_events.extend(
            [_text_event("orphan-1", "o1"), _text_event("orphan-2", "o2")]
        )
        # Slot empty (session went idle)
        assert session_key not in adapter._pending_messages

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued == 1
        # Slot now holds the oldest orphan
        assert adapter._pending_messages[session_key].text == "orphan-1"
        # Remaining orphan stays in overflow
        overflow = runner._session_state(session_key).conversation.queued_events
        assert len(overflow) == 1
        assert overflow[0].text == "orphan-2"

    def test_noop_when_slot_occupied(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:2"
        runner._session_state(session_key).conversation.queued_events.append(
            _text_event("orphan", "o1")
        )
        adapter._pending_messages[session_key] = _text_event("busy-slot", "slot")

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued == 0
        assert adapter._pending_messages[session_key].text == "busy-slot"
        assert len(runner._session_state(session_key).conversation.queued_events) == 1

    def test_noop_when_no_overflow(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:3"

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued == 0
        assert session_key not in adapter._pending_messages

    def test_fifo_order_preserved_across_rescue_and_new_message(self):
        """Oldest orphan runs first, new arrival last — FIFO (#28503)."""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = _StubAdapter()
        session_key = "telegram:user:4"

        # Two orphans from the lost window
        runner._session_state(session_key).conversation.queued_events.extend(
            [_text_event("orphan-1", "o1"), _text_event("orphan-2", "o2")]
        )

        # New message arrives for idle session — rescue stages orphan-1
        rescued = runner._rescue_orphaned_overflow(session_key, adapter)
        assert rescued == 1
        # Simulate the caller enqueueing the new message behind the rescued chain
        new_event = _text_event("new-msg", "new1")
        runner._session_state(session_key).conversation.queued_events.append(new_event)

        # Drain order: slot (orphan-1), then overflow[0] (orphan-2), then new-msg
        assert adapter._pending_messages[session_key].text == "orphan-1"
        overflow_texts = [e.text for e in runner._session_state(session_key).conversation.queued_events]
        assert overflow_texts == ["orphan-2", "new-msg"]
