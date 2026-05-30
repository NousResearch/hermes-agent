"""Tests for WeCom native reply streaming in GatewayStreamConsumer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _WeComNativeAdapter:
    MAX_MESSAGE_LENGTH = 4096
    SUPPORTS_NATIVE_STREAMING_REPLIES = True

    def __init__(self, *, native_results=None):
        self.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_final")
        )
        self.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_edit")
        )
        self.draft_calls = []
        self.send_draft = AsyncMock(return_value=SimpleNamespace(success=True))
        self.draft_supported = False
        self._native_results = list(native_results or [])
        self.native_calls = []

    async def send_stream_chunk(self, **kwargs):
        self.native_calls.append(kwargs)
        if self._native_results:
            return self._native_results.pop(0)
        return SimpleNamespace(success=True)

    def supports_draft_streaming(self, chat_type=None, metadata=None) -> bool:
        return bool(self.draft_supported)


def _reply_metadata():
    return {
        "reply": {
            "enterprise_id": "ent_1",
            "chat_type": "dm",
            "receive_id": "user_1",
            "msgid": "wecom_msg_1",
        }
    }


@pytest.mark.asyncio
async def test_native_reply_transport_requires_adapter_capability_and_reply_context():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls, "expected native send_stream_chunk calls"
    adapter.edit_message.assert_not_called()
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_reply_transport_not_used_without_reply_context():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=False, error="missing_reply_context")]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
        metadata={},
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls == []
    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "Hello ▉"


@pytest.mark.asyncio
async def test_native_final_success_sets_final_content_delivered():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=True), SimpleNamespace(success=True)]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    final_calls = [call for call in adapter.native_calls if call["finalize"] is True]
    assert len(final_calls) == 1


@pytest.mark.asyncio
async def test_native_final_success_sends_single_finish_frame_when_done_arrives_with_first_delta():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=True), SimpleNamespace(success=True)]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=100, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    consumer.finish()
    await consumer.run()

    final_calls = [call for call in adapter.native_calls if call["finalize"] is True]
    assert len(final_calls) == 1


@pytest.mark.asyncio
async def test_native_partial_success_falls_back_tail_only():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Hello "},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_partial_success_confirmed_prefix_len_falls_back_tail_only():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"confirmed_prefix_len": len("Hello ")},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_partial_success_confirmed_prefix_len_larger_than_content_clamps_visible_prefix():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"confirmed_prefix_len": len("Hello world") + 100},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_reply_transport_takes_precedence_over_draft_capability():
    adapter = _WeComNativeAdapter()
    adapter.draft_supported = True
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        ),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls
    adapter.send_draft.assert_not_awaited()
    adapter.edit_message.assert_not_called()


@pytest.mark.asyncio
async def test_native_failure_before_visible_content_sends_full_final_once():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=False, error="native_unavailable")]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello world")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert len(adapter.native_calls) == 1
    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "Hello world"


@pytest.mark.asyncio
async def test_segment_break_native_failure_flushes_tail_without_duplicate():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Hello "},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.on_segment_break()
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"


@pytest.mark.asyncio
async def test_ambiguous_native_delivered_prefix_mismatch_suppresses_blind_full_resend():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Mismatch"},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert [call["content"] for call in adapter.native_calls] == ["Hello ", "Hello world"]
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
