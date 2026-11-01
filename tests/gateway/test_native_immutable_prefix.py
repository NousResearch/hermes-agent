"""Regression tests for adapters whose native stream cannot retract accepted bytes."""

from itertools import pairwise
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class ImmutableNativeAdapter:
    SUPPORTS_NATIVE_STREAMING = True
    NATIVE_STREAM_IMMUTABLE_PREFIX = True

    def supports_native_streaming(self, **_kwargs):
        return True

    async def send_stream_frame(self, *_args, **_kwargs):
        return True


@pytest.mark.asyncio
async def test_native_interim_frames_are_raw_prefixes_without_cursor_or_fence_closers():
    consumer = GatewayStreamConsumer(
        ImmutableNativeAdapter(), "synthetic", StreamConsumerConfig(transport="native", chat_type="dm", cursor="▉")
    )
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._send_frame = AsyncMock(return_value=True)

    for frame in ("synthetic text ", "synthetic text `ab", "synthetic text `abc` tail"):
        await consumer._native_push(frame, finalize=False, is_turn_final=False)

    sent = [call.args[0] for call in consumer._send_frame.await_args_list]
    assert sent == ["synthetic text ", "synthetic text `ab", "synthetic text `abc` tail"]
    assert all(right.startswith(left) for left, right in pairwise(sent))


@pytest.mark.asyncio
async def test_native_divergence_closes_prefix_then_buffers_complete_final():
    consumer = GatewayStreamConsumer(
        ImmutableNativeAdapter(), "synthetic", StreamConsumerConfig(transport="native", chat_type="dm", cursor="")
    )
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._last_sent_text = "acknowledged prefix"
    consumer._send_frame = AsyncMock(return_value=True)

    result = await consumer._native_push(
        "rewritten complete answer", finalize=True, is_turn_final=True
    )

    assert result is False
    assert consumer._send_frame.await_args.args[0] == "acknowledged prefix"
    assert consumer._send_frame.await_args.kwargs["finalize"] is True
    assert consumer.cfg.buffer_only is True
    assert consumer._use_native_streaming is False
    assert consumer._fallback_final_send is True
    assert consumer._fallback_prefix == "acknowledged prefix"
    assert consumer._final_response_sent is False


@pytest.mark.asyncio
async def test_native_api_failure_closes_acknowledged_prefix_and_defers_fallback():
    consumer = GatewayStreamConsumer(
        ImmutableNativeAdapter(), "synthetic", StreamConsumerConfig(transport="native", chat_type="dm", cursor="")
    )
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._last_sent_text = "acknowledged prefix"
    consumer._send_frame = AsyncMock(side_effect=[False, True])

    result = await consumer._native_push(
        "acknowledged prefix plus continuation", finalize=False, is_turn_final=False
    )

    assert result is False
    assert [call.args[0] for call in consumer._send_frame.await_args_list] == [
        "acknowledged prefix plus continuation", "acknowledged prefix"
    ]
    assert consumer._send_frame.await_args_list[1].kwargs["finalize"] is True
    assert consumer.cfg.buffer_only is True
    assert consumer._fallback_final_send is True
    assert consumer._fallback_prefix == "acknowledged prefix"


@pytest.mark.asyncio
async def test_default_native_adapter_keeps_existing_frame_behavior():
    class DefaultNativeAdapter(ImmutableNativeAdapter):
        NATIVE_STREAM_IMMUTABLE_PREFIX = False

    consumer = GatewayStreamConsumer(
        DefaultNativeAdapter(), "synthetic", StreamConsumerConfig(transport="native", chat_type="dm")
    )
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._last_sent_text = "prior"
    consumer._send_frame = AsyncMock(return_value=True)

    result = await consumer._native_push("prior and more", finalize=True, is_turn_final=True)

    assert result is True
    assert consumer._send_frame.await_args.args[0] == "prior and more"
