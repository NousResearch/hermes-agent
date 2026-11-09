"""A streamed send that timed out after the request left may already be in the chat.

The adapter reports it as a non-retryable timeout (the rule ``_send_with_retry`` uses to refuse a
resend). The stream consumer must count that text as shown: sending it again, or editing a preview
up to it, puts a second copy of the answer in front of the user.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import DEFAULT_STREAMING_CURSOR
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

_TIMED_OUT = SimpleNamespace(success=False, message_id=None, error="Timed out", retryable=False)
# A reset after the write: no "timeout" in the text, the adapter marks the result uncertain instead.
_RESET = SimpleNamespace(success=False, message_id=None, retryable=False,
                         error="Server disconnected without sending a response.",
                         raw_response={"delivery_uncertain": True})


def _landing_adapter(delivered: list, *, fail_first: bool = True, failure=_TIMED_OUT) -> MagicMock:
    """Every send reaches the chat; the first one reports a post-write failure."""
    async def send(**kw):
        delivered.append(kw["content"])
        if fail_first and len(delivered) == 1:
            return failure
        return SimpleNamespace(success=True, message_id=f"m{len(delivered)}")

    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_TIMED_OUT, _RESET], ids=["timeout", "reset_after_write"])
@pytest.mark.parametrize("shape", ["short_final", "streamed_preview"])
async def test_timed_out_stream_send_is_shown_once_and_nothing_is_lost(shape, failure):
    """A short answer whose only send times out goes out once and counts as the turn's delivery
    (the gateway's normal final send stays suppressed). A timed-out preview is followed by its
    unseen tail only, never by a second full copy."""
    delivered: list = []
    consumer = GatewayStreamConsumer(
        _landing_adapter(delivered, failure=failure), "chat",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    if shape == "short_final":
        consumer.on_delta("Short answer.")
        consumer.finish()
        await asyncio.wait_for(consumer.run(), timeout=10)
        assert delivered == ["Short answer."]
        final = "Short answer."
    else:
        consumer.on_delta("first part ")
        task = asyncio.create_task(consumer.run())
        await async_wait_until(lambda: delivered, timeout=5)  # the preview send went out
        consumer.on_delta("second part.")
        consumer.finish()
        await asyncio.wait_for(task, timeout=10)
        assert delivered == ["first part ", "second part."]
        final = "first part second part."

    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(final) is not False


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["returned", "raised"])
async def test_timed_out_fresh_final_does_not_edit_the_preview_up_to_the_answer(raises):
    """Preview shown, then the fresh-final send times out (it may be on screen), whether the
    adapter returns the timeout or raises it: the preview must not be edited into a second full
    copy, and nothing is sent a third time."""
    delivered: list = []
    adapter = _landing_adapter(delivered, fail_first=False)
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(edit_interval=0.01))
    await consumer._send_or_edit("Here is the")

    async def lost_ack(**kw):
        delivered.append(kw["content"])
        if raises:
            raise TimeoutError("Timed out")
        return _TIMED_OUT

    adapter.send.side_effect = lost_ack
    adapter.prefers_fresh_final_streaming = MagicMock(return_value=True)

    assert await consumer._send_or_edit("Here is the full answer.", finalize=True) is True

    assert delivered == ["Here is the", "Here is the full answer."]
    adapter.edit_message.assert_not_called()
    adapter.delete_message.assert_not_called()


def _telegram_flagged(adapter: MagicMock) -> MagicMock:
    """Carry the real Telegram adapter's stream-fallback flags onto the double."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    for flag in ("RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK", "FALLBACK_ON_FINAL_EDIT_FLOOD"):
        setattr(adapter, flag, getattr(TelegramAdapter, flag))
    return adapter


@pytest.mark.asyncio
async def test_timed_out_preview_with_no_more_text_is_not_sent_again_on_telegram():
    """The preview already held the whole answer and its send timed out (it may be on screen, its
    id is unknown so it cannot be replaced): finishing with nothing new settles the turn without
    Telegram's empty-fallback resend putting a second copy beside it."""
    delivered: list = []
    consumer = GatewayStreamConsumer(
        _telegram_flagged(_landing_adapter(delivered)), "chat",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta("The whole answer.")
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)  # the preview send went out
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert delivered == ["The whole answer."]
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("The whole answer.") is not False


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["", DEFAULT_STREAMING_CURSOR], ids=["no_cursor", "default_cursor"])
@pytest.mark.parametrize("rest,expected", [
    ("y = 2\n```\nDone.", "```python\ny = 2\n```\nDone."),
    ("```", None),  # only the closing fence follows: the closed preview already showed it all
    ("", None),  # nothing follows: the preview was the whole answer
])
async def test_timed_out_preview_inside_open_code_block_sends_only_the_unseen_code(
        rest, expected, cursor):
    """The preview was shown with a display-only closing fence (the cursor, when on, sits inside
    it); more code may stream after it. The unseen tail is all that follows, never the whole
    answer again, and a bare closing fence is never a message of its own."""
    delivered: list = []
    consumer = GatewayStreamConsumer(
        _telegram_flagged(_landing_adapter(delivered)), "chat",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=cursor))
    consumer.on_delta("Code:\n```python\nx = 1\n")
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)
    consumer.on_delta(rest)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert delivered[1:] == ([expected] if expected else []), delivered
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_timed_out_continuation_is_not_followed_by_the_whole_answer_again():
    """The preview and the tail sent after it both time out after landing: each went out once and
    the turn counts as delivered, so the gateway's normal final send does not repeat it all."""
    delivered: list = []
    adapter = _telegram_flagged(_landing_adapter(delivered))
    adapter.send.side_effect = lambda **kw: delivered.append(kw["content"]) or _TIMED_OUT
    consumer = GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta("first part ")
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)
    consumer.on_delta("second part.")
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert delivered == ["first part ", "second part."]
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("first part second part.") is not False


@pytest.mark.asyncio
async def test_timed_out_continuation_that_telegram_splits_again_is_not_counted_as_shown():
    """3,000 periods fit one fallback chunk raw but escape to two MarkdownV2 messages: when that send
    times out, the second message was never attempted, so the turn must stay undelivered for the
    gateway's final send instead of silently losing that text."""
    from plugins.platforms.telegram.telegram_markdown import to_markdown_v2

    delivered: list = []
    adapter = _telegram_flagged(_landing_adapter(delivered))
    adapter.format_message = to_markdown_v2
    adapter.send.side_effect = lambda **kw: delivered.append(kw["content"]) or _TIMED_OUT
    consumer = GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta("first part ")
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)
    consumer.on_delta("." * 3000)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert consumer.final_content_delivered is False
