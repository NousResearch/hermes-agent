"""/stop or /new during a streamed reply on the edit transport: the abandoned preview is re-edited
without the live cursor, never left ending in it, and nothing else is delivered for the dead turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

CURSOR = " ▉"


async def _stream_then_reset(text: str, cursor: str = CURSOR):
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.MAX_MESSAGE_LENGTH = 4096
    current = {"ok": True}
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=1, cursor=cursor), run_still_current=lambda: current["ok"])
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(text)
    await async_wait_until(lambda: adapter.send.await_count, what="first preview sent")
    assert adapter.send.await_count == 1
    current["ok"] = False  # /stop or /new bumped the run generation
    consumer.on_delta(" stale tail")
    await asyncio.wait_for(task, 5)
    return adapter, consumer


@pytest.mark.asyncio
async def test_reset_mid_stream_strips_the_cursor_from_the_preview():
    adapter, consumer = await _stream_then_reset("partial answer")
    assert adapter.send.await_args.kwargs["content"].endswith(CURSOR)
    shown = adapter.edit_message.await_args.kwargs["content"]
    assert shown == "partial answer"
    assert "stale tail" not in shown and adapter.send.await_count == 1
    assert consumer._final_response_sent is False and consumer._already_sent is True


@pytest.mark.asyncio
async def test_reset_mid_code_block_keeps_the_closing_fence():
    adapter, _ = await _stream_then_reset("look:\n```py\nx = 1")
    assert adapter.edit_message.await_args.kwargs["content"] == "look:\n```py\nx = 1\n```"


@pytest.mark.asyncio
async def test_reset_without_a_cursor_makes_no_extra_edit():
    adapter, _ = await _stream_then_reset("partial answer", cursor="")
    adapter.edit_message.assert_not_awaited()
