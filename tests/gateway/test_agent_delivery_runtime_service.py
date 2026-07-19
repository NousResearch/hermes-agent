"""DEAD path: not imported by gateway/run.py — contract-only unit tests.
See gateway/RUNTIME_SERVICES.md.
"""
from unittest.mock import ANY, AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_delivery_runtime_service import finalize_gateway_agent_delivery


@pytest.mark.asyncio
async def test_finalize_gateway_agent_delivery_returns_visible_response():
    should_send_voice_reply = MagicMock(return_value=False)
    send_voice_reply = AsyncMock()
    deliver_media_from_response = AsyncMock()
    event = object()

    result = await finalize_gateway_agent_delivery(
        agent_result={},
        suppress_reply=False,
        response="hello",
        agent_messages=[{"role": "assistant", "content": "hello"}],
        event=event,
        platform="qq",
        adapters={},
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )

    assert result == "hello"
    should_send_voice_reply.assert_called_once_with(
        event,
        "hello",
        [{"role": "assistant", "content": "hello"}],
        already_sent=False,
    )
    send_voice_reply.assert_not_awaited()
    deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_gateway_agent_delivery_sends_voice_before_streaming_short_circuit():
    should_send_voice_reply = MagicMock(return_value=True)
    send_voice_reply = AsyncMock()
    deliver_media_from_response = AsyncMock()
    adapter = object()
    event = object()

    result = await finalize_gateway_agent_delivery(
        agent_result={"already_sent": True},
        suppress_reply=False,
        response="hello",
        agent_messages=[],
        event=event,
        platform="qq",
        adapters={"qq": adapter},
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )

    assert result is None
    should_send_voice_reply.assert_called_once_with(
        event,
        "hello",
        [],
        already_sent=True,
    )
    send_voice_reply.assert_awaited_once_with(event, "hello")
    deliver_media_from_response.assert_awaited_once_with("hello", event, adapter)


@pytest.mark.asyncio
async def test_finalize_gateway_agent_delivery_suppresses_without_side_effects():
    should_send_voice_reply = MagicMock()
    send_voice_reply = AsyncMock()
    deliver_media_from_response = AsyncMock()

    result = await finalize_gateway_agent_delivery(
        agent_result={},
        suppress_reply=True,
        response="hidden",
        agent_messages=[],
        event=object(),
        platform="qq",
        adapters={"qq": object()},
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )

    assert result is None
    should_send_voice_reply.assert_not_called()
    send_voice_reply.assert_not_awaited()
    deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_gateway_agent_delivery_skips_media_when_response_empty():
    should_send_voice_reply = MagicMock(return_value=False)
    send_voice_reply = AsyncMock()
    deliver_media_from_response = AsyncMock()

    result = await finalize_gateway_agent_delivery(
        agent_result={"already_sent": True},
        suppress_reply=False,
        response="",
        agent_messages=None,
        event=object(),
        platform="qq",
        adapters={"qq": object()},
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )

    assert result is None
    should_send_voice_reply.assert_called_once_with(
        ANY,
        "",
        [],
        already_sent=True,
    )
    send_voice_reply.assert_not_awaited()
    deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_gateway_agent_delivery_marks_synthetic_fallback_on_event_metadata():
    should_send_voice_reply = MagicMock(return_value=False)
    send_voice_reply = AsyncMock()
    deliver_media_from_response = AsyncMock()
    event = SimpleNamespace(metadata={"existing": True})

    result = await finalize_gateway_agent_delivery(
        agent_result={"synthetic_fallback": True},
        suppress_reply=False,
        response="刚才这轮接口空转了",
        agent_messages=[],
        event=event,
        platform="qq",
        adapters={},
        should_send_voice_reply=should_send_voice_reply,
        send_voice_reply=send_voice_reply,
        deliver_media_from_response=deliver_media_from_response,
    )

    assert result == "刚才这轮接口空转了"
    assert event.metadata["existing"] is True
    assert event.metadata["skip_successful_response_context"] is True
