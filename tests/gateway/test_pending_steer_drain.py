"""tests for leftover steer and pending message queue preservation in gateway drain."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, Platform
from gateway.run_turn import GatewayTurnMixin


class _StubPlatformAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(None, Platform.TELEGRAM)
        self._pending_messages = {}
        self._active_sessions = {}

    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


from gateway.session_state import SessionState


class _DummyRunner(GatewayTurnMixin):
    def __init__(self):
        self._draining = False
        self._sessions = {}

    def _sessions_map(self):
        return self._sessions

    def _session_state(self, session_key: str) -> SessionState:
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionState()
        return self._sessions[session_key]

    def _peek_session_state(self, session_key: str) -> SessionState | None:
        return self._sessions.get(session_key)

    def _overflow_queue(self, session_key: str):
        state = self._peek_session_state(session_key)
        return state.conversation.queued_events if state else None

    def _enqueue_fifo(self, session_key: str, queued_event: MessageEvent, adapter: any) -> None:
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if pending_slot is None:
            return
        if session_key in pending_slot:
            self._session_state(session_key).conversation.queued_events.append(queued_event)
        else:
            pending_slot[session_key] = queued_event
        queued_event._gateway_accepted = True

    def _promote_queued_event(
        self, session_key: str, adapter: any, pending_event: MessageEvent | None
    ) -> MessageEvent | None:
        overflow = self._overflow_queue(session_key)
        if not overflow:
            return pending_event
        if pending_event is None:
            return overflow.pop(0)
        if adapter is not None and hasattr(adapter, "_pending_messages"):
            adapter._pending_messages[session_key] = overflow.pop(0)
        return pending_event

    def _pending_event_audio_paths(self, event):
        return []

    def _status_action_label(self):
        return "drain"


def _make_source(chat_id: str = "123") -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


@pytest.mark.asyncio
async def test_leftover_steer_alone_delivers_as_pending():
    runner = _DummyRunner()
    adapter = _StubPlatformAdapter()
    source = _make_source()
    session_key = "telegram:123"

    result = {
        "final_response": "done",
        "pending_steer": "focus on pricing details",
    }

    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, session_key)
    assert pending_event is None
    assert pending == "focus on pricing details"


@pytest.mark.asyncio
async def test_leftover_steer_preserves_concurrent_pending_event():
    runner = _DummyRunner()
    adapter = _StubPlatformAdapter()
    source = _make_source()
    session_key = "telegram:123"

    background_event = MessageEvent(
        text="background job completed",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
    )
    adapter._pending_messages[session_key] = background_event

    result = {
        "final_response": "done",
        "pending_steer": "check edge cases first",
    }

    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, session_key)
    assert pending_event is None
    assert pending == "check edge cases first"
    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key] is background_event


@pytest.mark.asyncio
async def test_leftover_steer_preserves_fifo_order_with_overflow():
    runner = _DummyRunner()
    adapter = _StubPlatformAdapter()
    source = _make_source()
    session_key = "telegram:123"

    event1 = MessageEvent(text="first queued", message_type=MessageType.TEXT, source=source)
    event2 = MessageEvent(text="second queued", message_type=MessageType.TEXT, source=source)
    event3 = MessageEvent(text="third queued", message_type=MessageType.TEXT, source=source)

    runner._enqueue_fifo(session_key, event1, adapter)
    runner._enqueue_fifo(session_key, event2, adapter)
    runner._enqueue_fifo(session_key, event3, adapter)

    assert adapter._pending_messages[session_key] is event1
    assert runner._overflow_queue(session_key) == [event2, event3]

    result = {
        "final_response": "done",
        "pending_steer": "steer prompt",
    }

    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, session_key)
    assert pending_event is None
    assert pending == "steer prompt"

    assert adapter._pending_messages[session_key] is event1
    assert runner._overflow_queue(session_key) == [event2, event3]


@pytest.mark.asyncio
async def test_leftover_steer_with_interrupt_enqueues_behind_interrupt_message():
    runner = _DummyRunner()
    adapter = _StubPlatformAdapter()
    source = _make_source()
    session_key = "telegram:123"

    result = {
        "final_response": "interrupted",
        "interrupted": True,
        "interrupt_message": "stop right there",
        "pending_steer": "next time do that",
    }

    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, session_key)
    assert pending_event is None
    assert pending == "stop right there"
    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key].text == "next time do that"
