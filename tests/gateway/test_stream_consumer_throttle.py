"""Intermediate updates obey the interval even after the cumulative buffer fills."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import stream_consumer
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,flood", [("edit", False), ("edit", True), ("draft", False)])
@pytest.mark.parametrize("boundary", [False, True], ids=["done", "segment"])
async def test_run_throttles_cumulative_updates_but_not_finalization(monkeypatch, transport, flood, boundary):
    clock = SimpleNamespace(now=0.1)
    monkeypatch.setattr(stream_consumer, "time", SimpleNamespace(monotonic=lambda: clock.now))
    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.message_len_fn_for_chat.return_value = len
    adapter.max_message_length_for_chat.return_value = 4096
    adapter.streaming_overflow_limit.return_value = 4096
    adapter.supports_draft_streaming.return_value = True
    adapter.prefers_fresh_final_streaming.return_value = False
    updates = []
    finals = []

    async def send(*, content, metadata=None, **kwargs):
        (finals if metadata and metadata.get("notify") else updates).append((clock.now, content))
        return SendResult(success=True, message_id="preview")

    async def edit(*, content, finalize=False, **kwargs):
        (finals if finalize else updates).append((clock.now, content))
        if flood and not finalize and len(updates) == 2:
            return SendResult(success=False, error="flood_control:6", retry_after=6)
        return SendResult(success=True, message_id="preview")

    async def draft(*, content, **kwargs):
        updates.append((clock.now, content))
        return SendResult(success=True)

    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(side_effect=edit)
    adapter.send_draft = AsyncMock(side_effect=draft)
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=1.0, buffer_threshold=32, cursor=" ▉", transport=transport,
    ))
    text = "A" * 64
    consumer.on_delta(text)
    times = iter([0.15, 0.2, 1.2, 1.3, 2.3, 3.3, 3.35])
    final_text = None

    async def next_tick(delay):
        nonlocal text, final_text
        if final_text is not None:
            consumer.finish()
            return
        clock.now = next(times)
        text += " more"
        consumer.on_delta(" more")
        if clock.now == 3.35:
            if boundary:
                final_text = text
                consumer.on_segment_break()
            else:
                final_text = text + " authoritative final"
                consumer.finish(final_text)

    # Drive real queue drains and transport calls without wall-clock races.
    monkeypatch.setattr(stream_consumer, "asyncio", SimpleNamespace(
        sleep=next_tick, CancelledError=asyncio.CancelledError,
    ))
    await consumer.run()

    expected_times = [0.1, 1.2, 3.3] if flood else [0.1, 1.2, 2.3, 3.3]
    assert [when for when, _ in updates] == expected_times
    assert finals == [(3.35, final_text)]
    assert all(len(content) >= consumer.cfg.buffer_threshold for _, content in updates)
    if not boundary:
        assert consumer.final_content_delivered
        assert consumer.delivered_final_matches(final_text) is True
