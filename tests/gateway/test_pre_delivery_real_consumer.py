"""Real GatewayStreamConsumer tests for the strict pre-delivery boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.run_turn_runner import TurnRunner
from gateway.pre_delivery import PreDeliveryGate



def _adapter():
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    Adapter = type("BoundaryAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    Adapter.__abstractmethods__ = frozenset()
    adapter = Adapter.__new__(Adapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.draft_calls = []

    async def send_draft(*, chat_id, draft_id, content, metadata=None):
        adapter.draft_calls.append(content)
        return SendResult(success=True, message_id=None)

    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: True
    adapter.send_draft = send_draft
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    return adapter


@pytest.mark.asyncio
async def test_quarantine_holds_delta_commentary_and_segment_break():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("assistant text")
    consumer.on_commentary("commentary")
    await asyncio.sleep(0.08)
    consumer.on_segment_break()
    await asyncio.sleep(0.08)
    consumer.suppress_final_delivery()
    consumer.finish()
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("result_shape", [
    {"failed": True, "completed": True},
    {"interrupted": True, "completed": True},
    {"completed": False},
    {"completed": True, "final_response": ""},
])
async def test_quarantine_stops_non_success_terminal_shapes(result_shape):
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    consumer.suppress_final_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("buffered but unvalidated")
    consumer.finish()
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_then_finish_delivers_only_approved_rewrite():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("unvalidated draft")
    await asyncio.sleep(0.08)
    consumer.release_content_delivery()
    consumer.finish("approved final")
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_awaited()
    assert adapter.send.call_args.kwargs["content"] == "approved final"


@pytest.mark.asyncio
async def test_release_is_ordered_after_queued_commentary():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    # Queue all events before the consumer gets a chance to drain.  The release
    # must not flip a shared flag ahead of the earlier commentary event.
    consumer.on_delta("unvalidated draft")
    consumer.on_commentary("unvalidated commentary")
    consumer.release_content_delivery()
    consumer.finish("approved final")
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_awaited_once()
    assert adapter.send.call_args.kwargs["content"] == "approved final"


@pytest.mark.asyncio
@pytest.mark.parametrize("result_shape", [
    {"failed": True, "completed": True},
    {"interrupted": True, "completed": True},
    {"completed": False},
    {"completed": True, "final_response": ""},
])
async def test_turn_runner_suppresses_non_success_terminal_shapes(result_shape):
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    runner = SimpleNamespace(pre_delivery_gate=PreDeliveryGate(
        mode="strict", policy=lambda **_: {"allowed": True, "status": "passed", "final_text": "ok"},
    ))
    ctx = SimpleNamespace(
        result_holder=[None],
        source=SimpleNamespace(platform="test", chat_id="chat-1"),
        session_key="session-1",
        run_generation=1,
        inbound_message_id="turn-1",
    )
    turn = TurnRunner(runner, ctx)
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("buffered but unvalidated")
    turn._finish_stream_consumer(
        {"final_response": "buffered but unvalidated", "messages": [], **result_shape},
        [], consumer,
    )
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_quarantine_holds_stream_is_message_transport():
    adapter = _adapter()
    adapter.draft_stream_is_message = True
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("first segment")
    await asyncio.sleep(0.06)
    consumer.on_segment_break()
    consumer.on_delta(" second segment")
    await asyncio.sleep(0.06)
    consumer.suppress_final_delivery()
    consumer.finish()
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
