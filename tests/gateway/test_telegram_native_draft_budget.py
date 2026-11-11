"""Telegram native draft previews: one shared draft/typing budget, a short RetryAfter keeps drafts on,
the UTF-16 bound holds for astral text, and the opt-in ``plain_drafts`` frames only ever grow.

Real ``GatewayStreamConsumer`` + real ``TelegramAdapter``; only the Bot transport is a fake.
"""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Flood(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter(**extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.send_chat_action = AsyncMock(return_value=True)
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=9))
    return adapter


@pytest.mark.asyncio
async def test_drafts_and_typing_share_one_per_chat_budget():
    """20 calls per 5 s per chat across sendMessageDraft and sendChatAction: frame 21 is skipped
    locally (no request), typing is quiet while drafts spend the budget, another chat is unaffected."""
    adapter = _adapter()
    for i in range(20):
        assert (await adapter.send_draft("123", 7, f"frame {i}")).success
    skipped = await adapter.send_draft("00123", 7, "frame 20")
    assert skipped.success and skipped.raw_response == {"skipped": True} and skipped.retry_after > 0
    assert adapter._bot.send_message_draft.await_count == 20
    await adapter.send_typing("123")
    adapter._bot.send_chat_action.assert_not_awaited()
    assert not (await adapter.send_draft("456", 8, "other chat")).raw_response


@pytest.mark.asyncio
@pytest.mark.parametrize("plain", [False, True], ids=["default", "plain_drafts"])
async def test_astral_draft_respects_the_utf16_limit(plain):
    adapter = _adapter(plain_drafts=plain)
    content = "😀" * 2100  # 2,100 code points, 4,200 UTF-16 units
    await adapter.send_draft("123", 7, content)
    sent = adapter._bot.send_message_draft.call_args.kwargs["text"]
    assert utf16_len(sent) <= adapter.MAX_MESSAGE_LENGTH
    if plain:
        assert sent == "😀" * 2048  # a raw prefix: no chunk marker, no split surrogate


async def _stream(adapter, deltas, final, *, cursor=" ▉"):
    consumer = GatewayStreamConsumer(adapter, "123", StreamConsumerConfig(
        transport="draft", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor=cursor))
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        before = adapter._bot.send_message_draft.await_count
        consumer.on_delta(delta)
        for _ in range(200):
            if adapter._bot.send_message_draft.await_count > before:
                break
            await asyncio.sleep(0.01)
    consumer.finish(final)
    await asyncio.wait_for(task, timeout=10)
    return consumer


@pytest.mark.asyncio
async def test_short_retry_after_keeps_native_drafts_for_the_rest_of_the_reply():
    """A 1 s RetryAfter on one frame pauses previews; it never switches the reply to edit streaming."""
    adapter = _adapter()
    adapter._bot.send_message_draft.side_effect = [True, _Flood(1), True, True, True, True, True, True]
    final = "first part, second part, third part, done."
    consumer = await _stream(adapter, ["first part,", " second part,", " third part,", " done."], final)
    assert consumer._use_draft_streaming is True
    assert adapter._bot.send_message_draft.await_count >= 3  # previews resumed after the window
    adapter._bot.edit_message_text.assert_not_called()
    assert adapter._bot.send_message.await_count == 1  # exactly one durable final


@pytest.mark.asyncio
@pytest.mark.parametrize("plain", [True, False], ids=["plain_drafts", "default"])
async def test_plain_drafts_frames_only_grow(plain):
    """``plain_drafts``: raw text, no cursor, no synthetic fence, so every frame is a prefix of the
    next. Default: MarkdownV2 previews as before (formatting parity), still without the cursor."""
    adapter = _adapter(plain_drafts=plain)
    deltas = ["Code:\n```python\nprint(", "'hello')\n```\nAnd **bold", " text** done."]
    final = "".join(deltas)
    await _stream(adapter, deltas, final)
    frames = [c.kwargs for c in adapter._bot.send_message_draft.await_args_list]
    assert frames and all("▉" not in f["text"] for f in frames)
    if plain:
        texts = [f["text"] for f in frames]
        assert all("parse_mode" not in f for f in frames)
        assert all(b.startswith(a) for a, b in itertools.pairwise(texts)), texts
        assert texts[-1] == final
    else:
        assert all(f.get("parse_mode") for f in frames)
    # The persistent final keeps its normal MarkdownV2 rendering either way.
    assert adapter._bot.send_message.await_args.kwargs.get("parse_mode")
