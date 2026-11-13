"""Telegram native draft previews: one shared draft/typing budget, a short RetryAfter keeps drafts on,
the UTF-16 bound holds for astral text, and the opt-in ``plain_drafts`` frames only ever grow.

Real ``GatewayStreamConsumer`` + real ``TelegramAdapter``; only the Bot transport is a fake.
"""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import utf16_len
from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter
from tests.waiting import async_wait_until


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
        await async_wait_until(lambda: adapter._bot.send_message_draft.await_count > before, timeout=5,
                               what=f"a draft frame for {delta!r}")
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


@pytest.mark.asyncio
async def test_plain_drafts_wins_over_rich_drafts():
    adapter = _adapter(plain_drafts=True, rich_messages=True, rich_drafts=True)
    adapter._bot.do_api_request = AsyncMock(return_value=True)
    content = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert (await adapter.send_draft("123", 7, content)).success
    adapter._bot.do_api_request.assert_not_awaited()  # no sendRichMessageDraft
    assert adapter._bot.send_message_draft.await_args.kwargs == {"chat_id": 123, "draft_id": 7, "text": content}


@pytest.mark.asyncio
async def test_dm_topic_typing_retry_spends_its_own_budget_slot():
    """The topic-less typing retry is a second request: it claims a slot, and is skipped when none is left."""
    adapter = _adapter()
    adapter._dm_topic_fallback = lambda metadata: True
    adapter._message_thread_id_for_typing = lambda thread_id: 5
    adapter._bot.send_chat_action.side_effect = [RuntimeError("Bad Request: message thread not found"), True]
    for i in range(18):
        await adapter.send_draft("123", 7, f"frame {i}")
    await adapter.send_typing("123", metadata={"thread_id": "5"})
    assert adapter._bot.send_chat_action.await_count == 2  # 18 drafts + 2 typing = the 20-call window
    assert (await adapter.send_draft("123", 7, "over")).raw_response == {"skipped": True}

    adapter2 = _adapter()
    adapter2._dm_topic_fallback = lambda metadata: True
    adapter2._message_thread_id_for_typing = lambda thread_id: 5
    adapter2._bot.send_chat_action.side_effect = RuntimeError("Bad Request: message thread not found")
    for i in range(19):
        await adapter2.send_draft("123", 7, f"frame {i}")
    await adapter2.send_typing("123", metadata={"thread_id": "5"})
    assert adapter2._bot.send_chat_action.await_count == 1  # no slot left for the retry


def test_plain_drafts_reaches_the_adapter_from_config_yaml(tmp_path, monkeypatch):
    """Both documented nestings load through the gateway's config loader into the adapter."""
    import json

    from hermes_agent.gateway.config import Platform, load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for doc in ({"telegram": {"extra": {"plain_drafts": True}}},
                {"platforms": {"telegram": {"extra": {"plain_drafts": True}}}}):
        (tmp_path / "config.yaml").write_text(json.dumps(doc))
        platform_config = load_gateway_config().platforms[Platform.TELEGRAM]
        assert TelegramAdapter(platform_config).DRAFT_STREAM_PREFIX_STABLE is True, doc
    (tmp_path / "config.yaml").write_text(json.dumps({"telegram": {"extra": {"plain_drafts": False}}}))
    assert TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM]).DRAFT_STREAM_PREFIX_STABLE is False
