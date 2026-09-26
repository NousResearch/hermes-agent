"""A flood-refused streaming edit honours the server's ``retry_after`` (#116312).

Telegram answers a flood-refused ``editMessageText`` with the seconds to wait. The stream
consumer used to ignore it and only double its own 0.8s interval, so every one of the three
allowed strikes was spent — and three more refused requests earned — inside the first ~5s of
a penalty that is usually 9s or longer.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _flood(wait: float):
    return SimpleNamespace(success=False, error=f"flood_control:{wait}", retry_after=wait)


async def _stream(adapter, edit_interval: float, deltas: int, gap: float):
    consumer = GatewayStreamConsumer(
        adapter, "chat_1", StreamConsumerConfig(edit_interval=edit_interval, buffer_threshold=1, cursor=" ▉"))
    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(gap)
    for i in range(deltas):
        consumer.on_delta(f" w{i}")
        await asyncio.sleep(gap)
    consumer.finish()
    await task
    return consumer


@pytest.mark.asyncio
async def test_flood_refused_edit_waits_the_servers_retry_after_before_the_next_interim_edit():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=_flood(9.0))

    consumer = await _stream(adapter, edit_interval=0.02, deltas=6, gap=0.05)

    # The first refused interim edit reads retry_after=9s; every later interim tick (all inside
    # that window) is skipped instead of being sent into the same penalty.
    # Interim previews carry the cursor; the finalize path's cursor-strip edit does not.
    interim_edits = [c for c in adapter.edit_message.await_args_list if "▉" in c.kwargs["content"]]
    assert len(interim_edits) == 1
    assert consumer._current_edit_interval >= 9.0


@pytest.mark.asyncio
async def test_flood_without_retry_after_keeps_doubling_backoff():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood_control:9"))

    consumer = await _stream(adapter, edit_interval=0.02, deltas=6, gap=0.05)

    interim_edits = [c for c in adapter.edit_message.await_args_list if "▉" in c.kwargs["content"]]
    assert len(interim_edits) >= 2  # 0.04s, 0.08s, 0.16s: the legacy doubling ladder keeps firing
    assert consumer._current_edit_interval <= 10.0
    assert consumer._flood_strikes >= 2
