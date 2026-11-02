"""Exercise raw immutable frames through the real consumer loop."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class RawAdapter:
    NATIVE_STREAM_IMMUTABLE_PREFIX = True
    SUPPORTS_NATIVE_TOOL_PROGRESS = False
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = 30000

    def __init__(self):
        self.frames = []
        self.accepted = asyncio.Queue()
        self.send = AsyncMock(return_value=SendResult(success=True, message_id="ordinary"))
        self.fail_frame_text: str | None = None
        self.failed = asyncio.Event()

    async def send_stream_frame(self, text, **kwargs):
        self.frames.append((text, kwargs.get("finalize", False)))
        if text == self.fail_frame_text:
            self.failed.set()
            return False
        await self.accepted.put(text)
        return True


@pytest.mark.asyncio
async def test_real_loop_preserves_split_markdown_whitespace_and_final():
    adapter = RawAdapter()
    consumer = GatewayStreamConsumer(adapter, "synthetic", StreamConsumerConfig(
        edit_interval=0, buffer_threshold=1, chat_type="dm", cursor="▉"),
        initial_reply_to_id="synthetic-inbound")
    consumer._resolve_native_streaming = lambda: True
    task = asyncio.create_task(consumer.run())
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == ""
    cumulative = ""
    for chunk in ["正文 ", "`ab", "c`\n\n", "```python\n", "x = 1\n", "```\n结束"]:
        cumulative += chunk
        consumer.on_delta(chunk)
        assert await asyncio.wait_for(adapter.accepted.get(), 3) == cumulative
    consumer.finish(cumulative)
    await asyncio.wait_for(task, 3)
    assert adapter.frames[-1] == (cumulative, True)
    adapter.send.assert_not_awaited()
    assert consumer.final_content_delivered


@pytest.mark.asyncio
async def test_native_api_failure_in_split_inline_code_sends_only_exact_final_tail():
    adapter = RawAdapter()
    consumer = GatewayStreamConsumer(adapter, "synthetic", StreamConsumerConfig(
        edit_interval=0, buffer_threshold=1, chat_type="dm", cursor=""))
    consumer._resolve_native_streaming = lambda: True
    failed = "prefix `open"
    final = failed + " inline` suffix"
    adapter.fail_frame_text = failed
    task = asyncio.create_task(consumer.run())
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == ""
    consumer.on_delta("prefix ")
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == "prefix "
    consumer.on_delta("`open")
    await asyncio.wait_for(adapter.failed.wait(), 3)
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == "prefix "
    adapter.send.assert_not_awaited()
    consumer.finish(final)
    await asyncio.wait_for(task, 3)
    assert adapter.frames[-1] == ("prefix ", True)
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "`open inline` suffix"


@pytest.mark.asyncio
async def test_cancelled_stale_run_seals_real_adapter_frame_without_ordinary_send():
    adapter = RawAdapter()
    current = True
    consumer = GatewayStreamConsumer(adapter, "synthetic", StreamConsumerConfig(
        edit_interval=0, buffer_threshold=1, chat_type="dm", cursor=""),
        run_still_current=lambda: current)
    consumer._resolve_native_streaming = lambda: True
    task = asyncio.create_task(consumer.run())
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == ""
    consumer.on_delta("partial `inline")
    assert await asyncio.wait_for(adapter.accepted.get(), 3) == "partial `inline"
    current = False
    await asyncio.wait_for(task, 3)
    assert adapter.frames[-1] == ("partial `inline", True)
    adapter.send.assert_not_awaited()
