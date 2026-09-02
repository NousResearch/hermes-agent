"""Regression tests for #99882: FIFO overflow orphan rescue.

When a follow-up is demoted to /queue during compression-in-flight,
it lands in SessionState.conversation.queued_events (overflow) with
the current turn's event occupying adapter._pending_messages[session_key]
(slot).  After the slot's turn completes, _promote_queued_event moves
the overflow head into the slot.  When that drain never runs — the
busy window ended through an exit that skipped the promotion site
(/stop, turn exception, generation bump) — the overflow is silently
orphaned: never dispatched, never persisted, never logged.

The rescue in GatewayRunner._rescue_orphaned_overflow pops the oldest
orphan for the caller to run as the current turn and stages the next
orphan in the slot, so FIFO order (#28503) holds and nothing runs twice.
"""

from unittest.mock import MagicMock

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


def _runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    return runner


class TestRescueOrphanedOverflow:
    def test_single_orphan_is_returned_and_removed_from_both_stores(self):
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:1"
        runner._session_state(session_key).conversation.queued_events.append(
            _text_event("orphan-1", "o1")
        )
        assert session_key not in adapter._pending_messages

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued is not None and rescued.text == "orphan-1"
        # The rescued event runs as the current turn, so it must NOT also
        # sit in the slot — the post-turn drain would run it a second time.
        assert session_key not in adapter._pending_messages
        assert runner._session_state(session_key).conversation.queued_events == []

    def test_two_orphans_return_oldest_and_stage_next_in_slot(self):
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:1b"
        runner._session_state(session_key).conversation.queued_events.extend(
            [_text_event("orphan-1", "o1"), _text_event("orphan-2", "o2")]
        )

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued is not None and rescued.text == "orphan-1"
        # Slot now holds the NEXT orphan so the drain continues the chain.
        assert adapter._pending_messages[session_key].text == "orphan-2"
        assert runner._session_state(session_key).conversation.queued_events == []

    def test_noop_when_slot_occupied(self):
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:2"
        runner._session_state(session_key).conversation.queued_events.append(
            _text_event("orphan", "o1")
        )
        adapter._pending_messages[session_key] = _text_event("busy-slot", "slot")

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued is None
        assert adapter._pending_messages[session_key].text == "busy-slot"
        assert len(runner._session_state(session_key).conversation.queued_events) == 1

    def test_noop_when_no_overflow(self):
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:3"

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)

        assert rescued is None
        assert session_key not in adapter._pending_messages

    def test_fifo_order_preserved_across_rescue_and_new_message(self):
        """Oldest orphan runs first, new arrival last — FIFO (#28503).

        Mirrors the idle-arrival call site: rescue → _enqueue_fifo(new).
        """
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:4"
        runner._session_state(session_key).conversation.queued_events.extend(
            [_text_event("orphan-1", "o1"), _text_event("orphan-2", "o2")]
        )

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)
        assert rescued is not None and rescued.text == "orphan-1"
        runner._enqueue_fifo(session_key, _text_event("new-msg", "new1"), adapter)

        # Drain order after this turn: slot (orphan-2), then overflow (new-msg)
        assert adapter._pending_messages[session_key].text == "orphan-2"
        overflow_texts = [
            e.text for e in runner._session_state(session_key).conversation.queued_events
        ]
        assert overflow_texts == ["new-msg"]

    def test_single_orphan_then_new_message_lands_in_slot(self):
        """With one orphan the slot is free after rescue, so the incoming
        message must go to the slot (not overflow) or the drain never sees it."""
        runner = _runner()
        adapter = _StubAdapter()
        session_key = "telegram:user:5"
        runner._session_state(session_key).conversation.queued_events.append(
            _text_event("orphan-1", "o1")
        )

        rescued = runner._rescue_orphaned_overflow(session_key, adapter)
        assert rescued is not None and rescued.text == "orphan-1"
        runner._enqueue_fifo(session_key, _text_event("new-msg", "new1"), adapter)

        assert adapter._pending_messages[session_key].text == "new-msg"
        assert runner._session_state(session_key).conversation.queued_events == []
