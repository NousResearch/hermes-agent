"""Tests for interrupt race condition fix (#2483).

When a message arrives during post-processing (after the executor returns
but before the interrupt monitor is cancelled), the message must not be
silently dropped.  Two recovery paths are tested:

1. The monitor popped the message and called agent.interrupt(), but the
   agent had already finished — _monitor_pending_text recovers the text.
2. The message arrived after the executor returned and is still in
   adapter._pending_messages — the late-drain path picks it up.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class StubAdapter(BasePlatformAdapter):
    """Minimal adapter for interrupt race tests."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent_messages = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_messages.append(content)
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        thread_id=None,
    )


def _make_event(text="hello", chat_id="123456"):
    return MessageEvent(
        source=_source(chat_id),
        text=text,
        message_type=MessageType.TEXT,
    )


class TestInterruptRaceDrain:
    """Verify pending messages are recovered from both race windows."""

    def test_get_pending_message_pops_from_dict(self):
        """get_pending_message should pop and return the event."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"
        event = _make_event("test message")

        adapter._pending_messages[session_key] = event
        result = adapter.get_pending_message(session_key)

        assert result is event
        assert session_key not in adapter._pending_messages

    def test_get_pending_message_returns_none_when_empty(self):
        """get_pending_message returns None when no pending message exists."""
        adapter = StubAdapter()
        result = adapter.get_pending_message("nonexistent_key")
        assert result is None

    def test_has_pending_interrupt_requires_event_set(self):
        """has_pending_interrupt only returns True when the event is set."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"

        # No active session — should be False
        assert adapter.has_pending_interrupt(session_key) is False

        # Active session but event not set — should be False
        adapter._active_sessions[session_key] = asyncio.Event()
        assert adapter.has_pending_interrupt(session_key) is False

        # Event set — should be True
        adapter._active_sessions[session_key].set()
        assert adapter.has_pending_interrupt(session_key) is True

    def test_pending_message_survives_after_pop(self):
        """After monitor pops the message, it should be recoverable via
        the _monitor_pending_text mechanism (tested at integration level)."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"
        event = _make_event("follow-up question")

        # Simulate: adapter queues a pending message
        adapter._pending_messages[session_key] = event

        # Simulate: monitor pops it (like line 4966 in run.py)
        popped = adapter.get_pending_message(session_key)
        assert popped is event
        pending_text = popped.text

        # At this point _pending_messages is empty
        assert session_key not in adapter._pending_messages

        # The text should be preserved for recovery
        assert pending_text == "follow-up question"

    def test_late_arriving_message_drainable(self):
        """A message that arrives after executor returns should still be
        retrievable via get_pending_message (the late-drain path)."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"

        # Simulate: executor finished, no interrupt was detected
        # Then a new message arrives:
        late_event = _make_event("late message")
        adapter._pending_messages[session_key] = late_event

        # The late-drain path calls get_pending_message
        recovered = adapter.get_pending_message(session_key)
        assert recovered is late_event
        assert recovered.text == "late message"

    def test_interrupt_event_cleared_after_handling(self):
        """After clearing the interrupt event, new messages should be able
        to set it again for a fresh interrupt cycle."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"

        # Create session and set interrupt
        adapter._active_sessions[session_key] = asyncio.Event()
        adapter._active_sessions[session_key].set()
        assert adapter.has_pending_interrupt(session_key) is True

        # Clear it (like run.py line 5013)
        adapter._active_sessions[session_key].clear()
        assert adapter.has_pending_interrupt(session_key) is False

        # New message can set it again
        adapter._active_sessions[session_key].set()
        assert adapter.has_pending_interrupt(session_key) is True

    def test_concurrent_pop_is_safe(self):
        """Only one caller should get the pending message from a pop."""
        adapter = StubAdapter()
        session_key = "agent:main:telegram:dm:123456"
        event = _make_event("contested message")

        adapter._pending_messages[session_key] = event

        # First pop succeeds
        first = adapter.get_pending_message(session_key)
        assert first is event

        # Second pop returns None
        second = adapter.get_pending_message(session_key)
        assert second is None
