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

from hermes_agent.gateway.config import DEFAULT_STREAMING_CURSOR
from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
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
    # Counted as shown on a guess: the gateway hands it to the ledger's marked redelivery.
    assert consumer.delivery_uncertain is True


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
    from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

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
@pytest.mark.parametrize("preview,rest,expected", [
    ("Code:\n```python\ndef f():\n    result = fo", "o()\n    return result\n```",
     "```python\n    result = foo()\n    return result\n```"),
    ("Code:\n```python\ndef f():\n", "    return 1\n```", "```python\n    return 1\n```"),
    ("```python\nx = fo", "o(1)\n```", "```python\nx = foo(1)\n```"),
], ids=["indented_line_cut_mid_identifier", "cut_at_line_start", "cut_on_first_code_line"])
async def test_timed_out_preview_cut_mid_code_line_resends_the_whole_line(preview, rest, expected):
    """A preview cut inside a code line: the continuation repeats that whole line with its
    indentation, so one message holds the complete line; a cut at a line start repeats nothing."""
    delivered: list = []
    consumer = GatewayStreamConsumer(
        _telegram_flagged(_landing_adapter(delivered)), "chat",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta(preview)
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)
    consumer.on_delta(rest)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert delivered[1:] == [expected], delivered


@pytest.mark.asyncio
@pytest.mark.parametrize("tail_raises", [False, True], ids=["returned", "raised"])
async def test_timed_out_continuation_is_not_followed_by_the_whole_answer_again(tail_raises):
    """The preview and the tail sent after it both time out after landing (the tail's timeout
    returned or raised): each went out once and the turn counts as delivered, so the gateway's
    normal final send does not repeat it all."""
    delivered: list = []
    adapter = _telegram_flagged(_landing_adapter(delivered))

    def lost_ack(**kw):
        delivered.append(kw["content"])
        if tail_raises and len(delivered) > 1:
            raise TimeoutError("Timed out")
        return _TIMED_OUT

    adapter.send.side_effect = lost_ack
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
    assert consumer.delivery_uncertain is True


def _timing_out_telegram(delivered: list, *, outcomes: tuple = (), then=None) -> MagicMock:
    """A stream double whose sends go through the real TelegramAdapter send path (rich messages
    on) to a bot whose every send reaches Telegram and then times out. ``outcomes`` scripts the
    first bot calls (``None`` times out after landing) and ``then`` every later one; an exception
    outcome is a refusal: nothing lands."""
    from hermes_agent.gateway.config import PlatformConfig
    from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter
    from telegram.error import TimedOut

    pending = list(outcomes)

    def lost_ack(text):
        outcome = pending.pop(0) if pending else then
        if isinstance(outcome, Exception):
            raise outcome
        delivered.append(text)
        raise TimedOut("Timed out")

    telegram = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"rich_messages": True}))
    telegram._telegram_chat_outbound_slot_secs = 0
    bot = telegram._bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=lambda **kw: lost_ack(kw["text"]))
    bot.do_api_request = AsyncMock(
        side_effect=lambda method, api_kwargs: lost_ack(api_kwargs["rich_message"]["markdown"]))
    adapter = _telegram_flagged(_landing_adapter(delivered))
    adapter.format_message = telegram.format_message
    adapter._resume_partial_send = telegram._resume_partial_send

    async def send(*, chat_id, content, reply_to=None, metadata=None):
        return await telegram.send(chat_id, content, reply_to, metadata)

    adapter.send.side_effect = send
    return adapter


async def _stream_then_tail(adapter: MagicMock, delivered: list, tail: str) -> GatewayStreamConsumer:
    consumer = GatewayStreamConsumer(
        adapter, "123", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta("first part ")
    task = asyncio.create_task(consumer.run())
    await async_wait_until(lambda: delivered, timeout=5)
    consumer.on_delta(tail)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)
    return consumer


_SPLIT = "." * 3000  # fits one stream send raw; escapes to two MarkdownV2 messages


async def _send_split_answer(site: str, adapter: MagicMock, delivered: list):
    """Drive ``_SPLIT`` through one of the stream's sends that may land: the first send (as the
    whole short answer, or as a preview), the fallback continuation, the fresh final and the
    empty-fallback final."""
    if site == "fallback_final":
        return await _stream_then_tail(adapter, delivered, _SPLIT)
    consumer = GatewayStreamConsumer(
        adapter, "123", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    if site == "short_final":
        consumer.on_delta(_SPLIT)
        consumer.finish()
        await asyncio.wait_for(consumer.run(), timeout=10)
    elif site == "preview":
        consumer.on_delta(_SPLIT)
        task = asyncio.create_task(consumer.run())
        await async_wait_until(lambda: delivered, timeout=5)
        consumer.on_delta(" tail")
        consumer.finish()
        await asyncio.wait_for(task, timeout=10)
    elif site == "fresh_final":
        assert await consumer._try_fresh_final(_SPLIT) is True
    else:
        assert await consumer._send_empty_fallback_final(_SPLIT) == "ambiguous"
    return consumer


_SITES = ["short_final", "preview", "fallback_final", "fresh_final", "empty_fallback"]


@pytest.mark.asyncio
@pytest.mark.parametrize("site", _SITES)
async def test_timed_out_send_that_telegram_splits_again_sends_the_rest_once(site):
    """The first of the two messages times out (it may be on screen); the second was never
    attempted. Every period goes out exactly once: the timed-out head is not repeated and the
    never-attempted tail is not lost behind a send counted as shown."""
    delivered: list = []
    consumer = await _send_split_answer(site, _timing_out_telegram(delivered), delivered)

    assert sum(d.count("\\.") for d in delivered) == len(_SPLIT), [len(d) for d in delivered]
    if site in {"short_final", "preview", "fallback_final"}:
        assert consumer.final_content_delivered is True


@pytest.mark.asyncio
@pytest.mark.parametrize("site", _SITES)
async def test_refused_rest_of_a_timed_out_split_send_is_not_counted_as_shown(site):
    """The never-attempted second message is then refused outright (so is every later send):
    that text is not on screen, so the send is not counted as shown and the gateway's own final
    send still runs."""
    from telegram.error import BadRequest

    delivered: list = []
    lands = 2 if site == "fallback_final" else 1  # the fallback follows a timed-out preview
    adapter = _timing_out_telegram(delivered, outcomes=(None,) * lands, then=BadRequest("Chat not found"))
    if site == "fresh_final":
        consumer = GatewayStreamConsumer(adapter, "123", StreamConsumerConfig())
        assert await consumer._try_fresh_final(_SPLIT) is False
    elif site == "empty_fallback":
        consumer = GatewayStreamConsumer(adapter, "123", StreamConsumerConfig())
        assert await consumer._send_empty_fallback_final(_SPLIT) == "failed"
    else:
        consumer = await _send_split_answer(site, adapter, delivered)
        assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_timed_out_rich_continuation_is_counted_as_shown():
    """The same tail with a table goes out as ONE sendRichMessage (its MarkdownV2 form would not
    fit one message): nothing was left unattempted, so its timeout counts as shown and the gateway
    does not resend the whole answer."""
    delivered: list = []
    tail = "| a | b |\n|---|---|\n| 1 | 2 |\n\n" + "." * 3000
    consumer = await _stream_then_tail(_timing_out_telegram(delivered), delivered, tail)

    assert delivered[1:] == [tail]
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_TIMED_OUT, _RESET], ids=["timeout", "reset_after_write"])
async def test_oversized_first_head_with_lost_ack_sends_only_the_unsent_tail(failure):
    """The sealed overflow head landed but has no ACK/message id. Finish the
    split without sending the head a second time inside the whole answer."""
    delivered: list[str] = []
    adapter = _landing_adapter(delivered, failure=failure)
    adapter.MAX_MESSAGE_LENGTH = 650  # consumer safe limit: 550
    adapter.truncate_message = MagicMock(
        side_effect=lambda text, limit: [text[:limit], text[limit:]])
    answer = "H" * 550 + "T" * 250
    consumer = GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta(answer)
    consumer.finish()
    await asyncio.wait_for(consumer.run(), timeout=10)

    assert len(delivered) == 2, delivered
    assert "".join(delivered) == answer  # head once, then only the unseen tail
    assert consumer.final_content_delivered is True
    assert consumer.delivery_uncertain is True  # ledger may issue a marked retry


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks,expected", [
    # The only send is the final; it ends in a code block the display closed.
    ((["Here:\n```python\nprint(1)"]), ["Here:\n```python\nprint(1)\n```"]),
    # The preview stopped inside the language tag: the rest reopens the model's own opener.
    ((["```pyt", "hon\nx = 1\n```"]), ["```pyt\n```", "```python\nx = 1\n```"]),
], ids=["final_in_open_code_block", "preview_cut_in_language_tag"])
async def test_timed_out_send_in_a_code_block_shows_only_what_the_model_wrote(chunks, expected):
    """Telegram (finalize edit required): a timed-out send whose code fence the display closed
    counts as shown, so the gateway does not resend the whole answer unlabelled, and a
    continuation never invents a code line out of a split fence tag."""
    from hermes_agent.gateway.run_turn import GatewayTurnMixin

    delivered: list = []
    adapter = _telegram_flagged(_landing_adapter(delivered))
    adapter.REQUIRES_EDIT_FINALIZE = True
    consumer = GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta(chunks[0])
    task = asyncio.create_task(consumer.run())
    for rest in chunks[1:]:
        await async_wait_until(lambda: delivered, timeout=5)
        consumer.on_delta(rest)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert delivered == expected
    assert GatewayTurnMixin._run_agent_stream_confirmed_final_delivery(consumer, "".join(chunks))
    assert consumer.delivery_uncertain is True  # the ledger's resend is labelled
