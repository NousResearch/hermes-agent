"""Unit tests for gateway follow-up runtime helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_followup_runtime_service import (
    clear_gateway_pending_interrupt,
    deliver_gateway_first_response_before_followup,
    extract_gateway_pending_followup,
    queue_gateway_pending_followup_for_later,
)
from gateway.config import Platform
from gateway.session import SessionSource


def test_extract_gateway_pending_followup_uses_interrupt_message_when_queue_is_empty():
    adapter = SimpleNamespace()
    logger = MagicMock()

    followup = extract_gateway_pending_followup(
        result={"interrupted": True, "interrupt_message": "继续说"},
        adapter=adapter,
        session_key="session-1",
        dequeue_pending_event_text=lambda adapter_obj, session_key: (None, None),
        logger=logger,
    )

    assert followup is not None
    assert followup.text == "继续说"
    assert followup.was_interrupted is True


def test_extract_gateway_pending_followup_discards_command_text():
    adapter = SimpleNamespace()
    logger = MagicMock()
    pending_event = SimpleNamespace(text="/stop", metadata=None)

    followup = extract_gateway_pending_followup(
        result={"interrupted": False},
        adapter=adapter,
        session_key="session-1",
        dequeue_pending_event_text=lambda adapter_obj, session_key: (pending_event, "/stop"),
        logger=logger,
    )

    assert followup is None


def test_clear_gateway_pending_interrupt_clears_active_session_flag():
    interrupt_event = MagicMock()
    adapter = SimpleNamespace(_active_sessions={"session-1": interrupt_event})

    clear_gateway_pending_interrupt(
        adapter=adapter,
        session_key="session-1",
    )

    interrupt_event.clear.assert_called_once_with()


def test_queue_gateway_pending_followup_for_later_builds_text_event():
    queued = {}

    def _queue_message(session_key, event):
        queued["session_key"] = session_key
        queued["event"] = event

    adapter = SimpleNamespace(queue_message=_queue_message)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
    )
    pending_event = SimpleNamespace(message_id="pending-1")
    fallback_event = SimpleNamespace(message_id="fallback-1")

    queue_gateway_pending_followup_for_later(
        adapter=adapter,
        session_key="session-1",
        pending_text="后面继续",
        source=source,
        pending_event=pending_event,
        fallback_event=fallback_event,
    )

    assert queued["session_key"] == "session-1"
    assert queued["event"].text == "后面继续"
    assert queued["event"].message_id == "pending-1"


@pytest.mark.asyncio
async def test_deliver_gateway_first_response_before_followup_skips_streamed_or_suppressed():
    adapter = SimpleNamespace(send=AsyncMock())

    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "[[NO_REPLY]]", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=None,
        fallback_event=None,
        stream_consumer=SimpleNamespace(already_sent=False),
        logger=MagicMock(),
    )
    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "done", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=None,
        fallback_event=None,
        stream_consumer=SimpleNamespace(already_sent=True),
        logger=MagicMock(),
    )

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_gateway_first_response_before_followup_sends_visible_text():
    adapter = SimpleNamespace(send=AsyncMock())
    pending_event = SimpleNamespace(metadata={"thread_id": "7"})

    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "先回这条", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=pending_event,
        fallback_event=None,
        stream_consumer=SimpleNamespace(already_sent=False),
        logger=MagicMock(),
    )

    adapter.send.assert_awaited_once_with(
        "12345",
        "先回这条",
        metadata={"thread_id": "7"},
    )
