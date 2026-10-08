"""Native Telegram previews preserve prefixes without changing final delivery.

Exercise the real consumer and adapter together; only the bot transport is fake.
Queue acknowledgements synchronise input with delivered frames, without asserting
wall-clock latency or claiming to reproduce a Telegram client's animation.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class RecordingBot:
    def __init__(self, *, reject_drafts=False):
        self.reject_drafts = reject_drafts
        self.drafts = []
        self.messages = []
        self.edits = []
        self.rich_messages = []
        self.frames = asyncio.Queue()
        self.deliveries = asyncio.Queue()

    async def send_message_draft(self, **kwargs):
        self.drafts.append(kwargs)
        self.frames.put_nowait(kwargs)
        return not self.reject_drafts

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        self.deliveries.put_nowait(kwargs)
        return SimpleNamespace(message_id=123)

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(message_id=123)

    async def do_api_request(self, endpoint, *, api_kwargs):
        self.rich_messages.append((endpoint, api_kwargs))
        return SimpleNamespace(message_id=123)

    async def send_chat_action(self, **kwargs):
        return True


def _consumer(bot, *, rich_messages=False):
    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="fake-token",
        extra={"rich_messages": rich_messages, "rich_drafts": False},
    ))
    adapter._bot = bot
    consumer = GatewayStreamConsumer(
        adapter, "12345",
        StreamConsumerConfig(
            transport="draft", chat_type="dm", edit_interval=0.05,
            buffer_threshold=1, cursor=" ▉",
        ),
    )
    return adapter, consumer


@pytest.mark.asyncio
async def test_burst_drafts_are_plain_prefixes_and_final_retains_rich_content():
    bot = RecordingBot()
    _adapter, consumer = _consumer(bot, rich_messages=True)
    first = "**Steps**\n\n```python\nprint("
    middle = "'ready')\n"
    tail = "```\n\n| Item | Status |\n|---|---|\n| stream | complete |"
    complete = first + middle + tail
    final = complete + "\n\nVerification details: " + "detail " * 800 + "complete."

    consumer.on_delta(first)
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(bot.frames.get(), timeout=5)
        # Several producer deltas arrive together: the next preview must catch up
        # to all of them instead of slowly replaying a fixed character allowance.
        for delta in (middle, tail):
            consumer.on_delta(delta)
        await asyncio.wait_for(bot.frames.get(), timeout=5)
        consumer.finish(final)
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            await task

    assert [frame["text"] for frame in bot.drafts] == [first, complete]
    assert len({frame["draft_id"] for frame in bot.drafts}) == 1
    assert all("parse_mode" not in frame for frame in bot.drafts)
    assert bot.messages == []
    assert bot.edits == []
    assert bot.rich_messages == [(
        "sendRichMessage",
        {
            "chat_id": 12345,
            "rich_message": {"markdown": final},
        },
    )]
    assert consumer.delivered_final_matches(final)


@pytest.mark.asyncio
async def test_rejected_native_draft_falls_back_without_losing_final_content():
    bot = RecordingBot(reject_drafts=True)
    adapter, consumer = _consumer(bot)
    first = "A complete **preview**"
    final = first + "\n\nFinal details: [reference](https://example.com)."

    consumer.on_delta(first)
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(bot.deliveries.get(), timeout=5)
        consumer.finish(final)
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            await task

    assert len(bot.drafts) == 1
    assert len(bot.messages) == 1
    assert bot.rich_messages == []
    assert bot.edits[-1]["text"] == adapter.format_message(final)
    assert consumer.delivered_final_matches(final)
