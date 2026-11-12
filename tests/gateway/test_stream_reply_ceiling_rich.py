"""A streamed reply on a rich-enabled Telegram chat stays inside the 8-message cap (#152012).

With rich transport on, the stream consumer seals heads at the rich cap (32,768 chars). A prose head
has no rich construct, so the adapter delivers the seal as a MarkdownV2 split, capped at the reply's
budget with its notice last. That capped seal is the reply's last text: its tail must not start a new
live preview, or the consumer later forgets the cut and the gateway re-sends the whole reply beside
the eight messages already shown.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


def _body(chars: int) -> str:
    lines, i, n = [], 0, 0
    while n < chars:
        line = " ".join(f"w{i}x{j}" for j in range(9))
        lines.append(line)
        n += len(line) + 1
        i += 1
    return "\n".join(lines)[:chars]


def _rich_adapter(screen: list) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={"rich_messages": True}))
    ids = iter(range(1, 999))

    async def do_api_request(method, api_kwargs=None, **_kw):
        if method == "editMessageText":
            return True
        screen.append(mid := next(ids))
        return SimpleNamespace(message_id=mid)

    async def send_message(text="", **_kw):
        screen.append(mid := next(ids))
        return SimpleNamespace(message_id=mid)

    bot = MagicMock()
    bot.do_api_request = AsyncMock(side_effect=do_api_request)
    bot.send_message = AsyncMock(side_effect=send_message)
    bot.edit_message_text = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=kw.get("message_id")))
    bot.send_chat_action = AsyncMock()
    bot.delete_message = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


@pytest.mark.asyncio
async def test_a_slow_huge_rich_stream_is_cut_once_and_not_sent_again(monkeypatch):
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    monkeypatch.setattr(telegram_adapter, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    screen: list = []
    adapter = _rich_adapter(screen)
    final = "| metric | value |\n|---|---|\n| rows | many |\n\n" + _body(415_000)
    consumer = GatewayStreamConsumer(adapter, "111", StreamConsumerConfig(
        edit_interval=0.05, buffer_threshold=40, cursor=" ▉"))
    task = asyncio.create_task(consumer.run())
    for start in range(0, len(final), 1500):
        consumer.on_delta(final[start:start + 1500])
        await asyncio.sleep(0.01)  # health: allow HX013 -- the pace of the stream IS the behaviour (seals between deltas)
    consumer.finish(final)
    await asyncio.wait_for(task, 60)

    assert len(screen) <= adapter.MAX_SPLIT_MESSAGES
    # The cut delivery is the turn's final: the gateway's normal send must not post it again.
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(final) is True
