"""Follow-up contracts for native final-delivery reconciliation."""

from __future__ import annotations

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter


@pytest.mark.asyncio
async def test_native_final_tail_matches_preamble_but_not_unsent_augmentation():
    adapter = _make_native_streaming_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(chat_type="dm", cursor="", buffer_threshold=1),
    )

    consumer.on_delta("Let me check.\n")
    consumer.on_delta("Answer")
    consumer.finish("Answer\nVerified footer")
    await asyncio.create_task(consumer.run())

    transmitted = "".join(frame["text"] for frame in adapter.frames if frame["finalize"])
    assert transmitted == "Let me check.\nAnswer"
    assert consumer.delivered_final_matches("Answer") is True
    assert consumer.delivered_final_matches("Answer\nVerified footer") is False
