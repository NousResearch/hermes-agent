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

@pytest.mark.asyncio
async def test_reset_while_final_send_fails_does_not_retry_the_dead_turn():
    """A /stop during the final send must not retry stale text after it was rejected."""
    current = {"ok": True}
    attempts = []

    async def send(*, chat_id, content, reply_to=None, metadata=None):
        attempts.append(content)
        if len(attempts) == 1:
            current["ok"] = False  # /stop arrived during the awaited send
            return SimpleNamespace(success=False, message_id=None, error="rejected")
        return SimpleNamespace(success=True, message_id="too-late")

    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock()
    consumer = GatewayStreamConsumer(
        adapter, "chat", run_still_current=lambda: current["ok"],
    )
    consumer.on_delta("answer that must not be re-sent after stop")
    consumer.finish()

    await consumer.run()

    assert attempts == ["answer that must not be re-sent after stop"]
    assert consumer.final_response_sent is False


@pytest.mark.asyncio
async def test_reset_while_the_final_edit_is_refused_sends_no_fallback_after_new():
    """A preview is on screen; /new lands while the final edit is in flight and that edit is refused.
    The fallback that would send the unseen tail as a new message must not run for the dead turn."""
    current = {"ok": True}
    sends: list[str] = []

    async def send(*, chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        return SimpleNamespace(success=True, message_id=f"m{len(sends)}")

    async def edit_message(*, chat_id, message_id, content, finalize=False, metadata=None):
        if finalize:
            current["ok"] = False  # /new bumped the generation while the final edit was in flight
            return SimpleNamespace(success=False, message_id=None, error="Bad Request: message can't be edited")
        return SimpleNamespace(success=True, message_id=message_id)

    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(side_effect=edit_message)
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=1, cursor=CURSOR), run_still_current=lambda: current["ok"])
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("the dead turn's")
    await async_wait_until(lambda: sends, what="first preview sent")
    consumer.on_delta(" whole answer")
    consumer.finish()
    await asyncio.wait_for(task, 5)

    assert sends == ["the dead turn's" + CURSOR], sends
    assert consumer.final_response_sent is False
