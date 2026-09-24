"""A truncated Telegram preview is not acknowledgement of the complete answer."""
import asyncio
import re
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class FloodError(Exception):
    retry_after = 120

    def __str__(self):
        return "Flood control exceeded. Retry in 120 seconds"


class PreviewBot:
    def __init__(self, final_flood, rich_enabled):
        self.messages = {}
        self.preview_written = asyncio.Event()
        self.flood = False
        self.final_flood = final_flood
        self.rich_enabled = rich_enabled
        self.seed_written = asyncio.Event()

    async def send_message(self, **kwargs):
        if self.flood and self.final_flood:
            raise FloodError()
        mid = len(self.messages) + 1
        self.messages[mid] = kwargs['text']
        self.seed_written.set()
        return SimpleNamespace(message_id=mid)

    async def edit_message_text(self, **kwargs):
        if self.flood:
            raise FloodError()
        # A failed legacy head-seal can also leave the whole buffer for the next
        # interim update. Let that raw capped preview land after the refusal.
        if not self.rich_enabled and len(kwargs['text']) > 3000 and ' (1/2)' not in kwargs['text']:
            raise FloodError()
        self.messages[kwargs['message_id']] = kwargs['text']
        if len(kwargs['text']) > 3000:
            self.preview_written.set()

    async def send_message_draft(self, **kwargs):
        raise AssertionError('Group chats must use edit streaming')

    async def do_api_request(self, **kwargs):
        raise AssertionError('Plain paragraphs do not need rich rendering')

    async def delete_message(self, **kwargs):
        self.messages.pop(kwargs['message_id'], None)


@pytest.mark.asyncio
@pytest.mark.parametrize('final_flood', [True, False])
@pytest.mark.parametrize('rich_enabled', [True, False])
@pytest.mark.parametrize('fenced', [True, False])
async def test_truncated_preview_never_confirms_unseen_tail(final_flood, rich_enabled, fenced):
    bot = PreviewBot(final_flood, rich_enabled)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='offline-test', extra={'rich_messages': rich_enabled}))
    adapter._bot = bot
    consumer = GatewayStreamConsumer(
        adapter=adapter, chat_id='123',
        config=StreamConsumerConfig(edit_interval=0, buffer_threshold=1, chat_type='group'),
    )
    lines = ['Start of answer'] + [f'Paragraph {i:04d} with its unique content' for i in range(180)] + ['END OF ANSWER']
    text = '\n'.join(lines)
    if fenced:
        text = '```text\n' + text + '\n```'
    task = asyncio.create_task(consumer.run())
    # Seed a short message first, then exceed the legacy edit cap while still below
    # the rich-capable transport's accumulation budget.
    consumer.on_delta(text[:100])
    try:
        await asyncio.wait_for(bot.seed_written.wait(), 5)
        consumer.on_delta(text[100:])
        await asyncio.wait_for(bot.preview_written.wait(), 5)
        assert 'END OF ANSWER' not in ''.join(bot.messages.values())
        bot.flood = True
        consumer.finish(text)
        await asyncio.wait_for(task, 10)
        if final_flood:
            assert not consumer.final_content_delivered
            assert not consumer.final_response_sent
        else:
            assert 'END OF ANSWER' in ''.join(bot.messages.values())
            assert consumer.final_response_sent
            rendered = re.sub(r'\\(.)', r'\1', '\n'.join(bot.messages.values()))
            for line in lines:
                assert rendered.count(line) == 1
    finally:
        if not task.done():
            task.cancel()
            await task


@pytest.mark.asyncio
async def test_short_flood_retry_preserves_partial_preview_receipt():
    from unittest.mock import AsyncMock, MagicMock

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='offline-test'))
    adapter._bot = MagicMock()
    flood = FloodError()
    flood.retry_after = 0.001
    adapter._bot.edit_message_text = AsyncMock(side_effect=[flood, None])
    result = await adapter.edit_message('123', '1', 'preview text\n' * 700, finalize=False)
    assert not result.success
    assert result.raw_response['partial_overflow'] is True
    assert result.raw_response['delivered_prefix'] == adapter._bot.edit_message_text.call_args.kwargs['text']
