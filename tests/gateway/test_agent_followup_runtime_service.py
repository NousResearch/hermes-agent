"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Unit tests for gateway follow-up runtime helpers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_followup_runtime_service import (
    clear_gateway_pending_interrupt,
    deliver_gateway_first_response_before_followup,
    extract_gateway_pending_followup,
    process_gateway_pending_followup,
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
    pending_event = SimpleNamespace(
        message_id="pending-1",
        raw_message={"source": "pending"},
        metadata={"explicit_addressed": True, "address_reason": "bot_mention"},
        reply_to_message_id="bot-msg-1",
        reply_to_text="上一条",
        media_urls=["/tmp/test.png"],
        media_sources=["/tmp/test.png"],
        media_types=["image/png"],
    )
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
    assert queued["event"].raw_message == {"source": "pending"}
    assert queued["event"].metadata["explicit_addressed"] is True
    assert queued["event"].reply_to_message_id == "bot-msg-1"
    assert queued["event"].reply_to_text == "上一条"
    assert queued["event"].media_urls == ["/tmp/test.png"]

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
        history_len=0,
        empty_response_fallback=lambda _kind: None,
        logger=MagicMock(),
    )
    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "done", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=None,
        fallback_event=None,
        stream_consumer=SimpleNamespace(already_sent=True),
        history_len=0,
        empty_response_fallback=lambda _kind: None,
        logger=MagicMock(),
    )

    adapter.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_deliver_gateway_first_response_before_followup_sends_visible_text():
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="sent-1")),
        _record_successful_response_context=MagicMock(),
    )
    pending_event = SimpleNamespace(metadata={"thread_id": "7"})

    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "先回这条", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=pending_event,
        fallback_event=None,
        stream_consumer=SimpleNamespace(already_sent=False),
        history_len=0,
        empty_response_fallback=lambda _kind: None,
        logger=MagicMock(),
    )

    adapter.send.assert_awaited_once_with(
        "12345",
        "先回这条",
        metadata={"thread_id": "7"},
    )
    adapter._record_successful_response_context.assert_called_once_with(
        pending_event,
        ["sent-1"],
    )

@pytest.mark.asyncio
async def test_deliver_gateway_first_response_before_followup_uses_empty_fallback_without_reopening_context():
    fallback_event = SimpleNamespace(metadata={"explicit_addressed": True})
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="sent-1")),
        _record_successful_response_context=MagicMock(),
    )

    await deliver_gateway_first_response_before_followup(
        result={"interrupted": False, "final_response": "(empty)", "suppress_reply": False},
        adapter=adapter,
        chat_id="12345",
        pending_event=None,
        fallback_event=fallback_event,
        stream_consumer=SimpleNamespace(already_sent=False),
        history_len=0,
        empty_response_fallback=lambda kind: "我在，你继续说。" if kind == "empty" else None,
        logger=MagicMock(),
    )

    adapter.send.assert_awaited_once_with(
        "12345",
        "我在，你继续说。",
        metadata={"explicit_addressed": True, "skip_successful_response_context": True},
    )
    adapter._record_successful_response_context.assert_not_called()
    assert fallback_event.metadata["skip_successful_response_context"] is True

@pytest.mark.asyncio
async def test_process_gateway_pending_followup_returns_none_when_nothing_pending():
    result = await process_gateway_pending_followup(
        result={"messages": []},
        adapter=SimpleNamespace(),
        session_key="session-1",
        dequeue_pending_event_text=lambda *_: (None, None),
        logger=MagicMock(),
        interrupt_depth=0,
        max_interrupt_depth=3,
        source=SimpleNamespace(),
        fallback_event=None,
        chat_id="12345",
        stream_consumer=None,
        history=[],
        current_response_fallback={"final_response": "fallback", "messages": []},
        empty_response_fallback=lambda _kind: None,
        recurse_followup=AsyncMock(),
    )

    assert result is None

@pytest.mark.asyncio
async def test_process_gateway_pending_followup_queues_when_depth_cap_reached(monkeypatch):
    followup = SimpleNamespace(event=SimpleNamespace(message_id="pending-1"), text="继续", was_interrupted=True)
    queue_mock = MagicMock()
    clear_mock = MagicMock()
    recurse_mock = AsyncMock()

    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.extract_gateway_pending_followup",
        lambda **kwargs: followup,
    )
    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.clear_gateway_pending_interrupt",
        lambda **kwargs: clear_mock(**kwargs),
    )
    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.queue_gateway_pending_followup_for_later",
        lambda **kwargs: queue_mock(**kwargs),
    )

    result = await process_gateway_pending_followup(
        result={"messages": ["done"], "interrupted": True},
        adapter=SimpleNamespace(),
        session_key="session-1",
        dequeue_pending_event_text=lambda *_: (None, None),
        logger=MagicMock(),
        interrupt_depth=3,
        max_interrupt_depth=3,
        source=SimpleNamespace(),
        fallback_event=SimpleNamespace(message_id="fallback-1"),
        chat_id="12345",
        stream_consumer=None,
        history=[{"role": "user", "content": "hi"}],
        current_response_fallback={"final_response": "fallback", "messages": []},
        empty_response_fallback=lambda _kind: None,
        recurse_followup=recurse_mock,
    )

    assert result == {"final_response": "fallback", "messages": []}
    clear_mock.assert_called_once()
    queue_mock.assert_called_once()
    recurse_mock.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_gateway_pending_followup_delivers_and_recurses(monkeypatch):
    pending_event = SimpleNamespace(message_id="pending-1", raw_message="原文", metadata={"thread_id": "7"})
    followup = SimpleNamespace(event=pending_event, text="继续", was_interrupted=False)
    recurse_mock = AsyncMock(return_value={"final_response": "continued", "messages": ["next"]})
    deliver_mock = AsyncMock()
    clear_mock = MagicMock()

    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.extract_gateway_pending_followup",
        lambda **kwargs: followup,
    )
    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.clear_gateway_pending_interrupt",
        lambda **kwargs: clear_mock(**kwargs),
    )
    monkeypatch.setattr(
        "gateway.agent_followup_runtime_service.deliver_gateway_first_response_before_followup",
        deliver_mock,
    )

    result = await process_gateway_pending_followup(
        result={"messages": [{"role": "assistant", "content": "done"}], "final_response": "done"},
        adapter=SimpleNamespace(),
        session_key="session-1",
        dequeue_pending_event_text=lambda *_: (None, None),
        logger=MagicMock(),
        interrupt_depth=1,
        max_interrupt_depth=3,
        source=SimpleNamespace(chat_id="12345"),
        fallback_event=SimpleNamespace(message_id="fallback-1"),
        chat_id="12345",
        stream_consumer=SimpleNamespace(already_sent=False),
        history=[{"role": "user", "content": "hi"}],
        current_response_fallback={"final_response": "fallback", "messages": []},
        empty_response_fallback=lambda _kind: None,
        recurse_followup=recurse_mock,
    )

    assert result == {"final_response": "continued", "messages": ["next"]}
    clear_mock.assert_called_once()
    deliver_mock.assert_awaited_once()
    recurse_mock.assert_awaited_once_with(
        followup,
        [{"role": "assistant", "content": "done"}],
    )
