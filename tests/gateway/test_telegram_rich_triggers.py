"""Offline synthetic protocol tests; not evidence of any live Telegram payload.

Harness and mention-parity control adapted from the test-only review contribution:
https://github.com/NousResearch/hermes-agent/pull/132858#issuecomment-6232845484
The canonical runner isolates each file. Gateway conftest installs PTB mocks
before collection; discard those only here, before importing the adapter.
"""
import asyncio
import importlib
import json
import re
import sys
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

for name in list(sys.modules):
    if name == "telegram" or name.startswith("telegram."):
        del sys.modules[name]
telegram = importlib.import_module("telegram")
from telegram import Update
from telegram.ext import Application, ExtBot
from telegram.request import BaseRequest
from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


class ClosedRequest(BaseRequest):
    """No socket transport. Only synthetic initialization getMe is permitted."""
    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        assert url.endswith('/getMe'), 'Unexpected bot request blocked'
        return 200, json.dumps({'ok': True, 'result': {
            'id': 900, 'is_bot': True, 'first_name': 'Synthetic',
            'username': 'synthetic_bot'}}).encode()


def update(bot, text=None, rich=True, kind='message', **fields):
    msg = {'message_id': 10, 'date': 1700000000,
           'chat': {'id': 100, 'type': 'private', 'first_name': 'Synthetic'},
           'from': {'id': 42, 'is_bot': False, 'first_name': 'Synthetic'}}
    if text is not None:
        msg['text'] = text
    if rich:
        msg['rich_message'] = {'blocks': [{'type': 'paragraph', 'text': 'Rich α\n第二行'}]}
    msg.update(fields)
    return Update.de_json({'update_id': 1000, kind: msg}, bot)


@pytest_asyncio.fixture
async def harness():
    assert telegram.__version__ == '22.8'
    bot = ExtBot('900:SYNTHETIC_NOT_A_REAL_TOKEN', request=ClosedRequest(),
                 get_updates_request=ClosedRequest())
    app = Application.builder().bot(bot).updater(None).build()
    await app.initialize()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, extra={
        'allow_from': ['42'], 'group_allow_from': ['42'], 'require_mention': False}))
    adapter._bot = bot
    adapter._running = True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._on_platform_update = AsyncMock()
    adapter.handle_message = AsyncMock()
    errors = []
    async def error_handler(u, context):
        errors.append(context.error)
    app.add_error_handler(error_handler)
    adapter._register_handlers(app)
    yield adapter, app, bot, errors
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)
    await app.shutdown()


async def flush(adapter):
    tasks = list(adapter._pending_text_batch_tasks.values())
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.parametrize('rich', [False, True])
@pytest.mark.asyncio
async def test_group_mention_parity(harness, rich):
    adapter, app, bot, errors = harness
    adapter.config.extra['require_mention'] = True
    adapter._text_batch_delay_seconds = 0
    text = '@synthetic_bot hello'
    u = update(bot, text=None if rich else text, rich=rich,
        chat={'id': -100, 'type': 'supergroup', 'title': 'Synthetic'},
        **({'rich_message': {'blocks': [{'type': 'paragraph', 'text': text}]}} if rich else {}))
    await app.process_update(u)
    await flush(adapter)
    assert not errors
    assert adapter.handle_message.await_count == 1
    assert adapter.handle_message.call_args.args[0].text == 'hello'


@pytest.mark.parametrize('blocks,expected', [
    ([{'type': 'paragraph', 'text': '😀 @synthetic_bot hello'}], True),
    ([{'type': 'paragraph', 'text': {'type': 'bold', 'text': '@synthetic_bot hello'}}], True),
    ([{'type': 'paragraph', 'text': 'hello @other_bot'}], False),
    ([{'type': 'paragraph', 'text': 'foo@synthetic_bot.example'}], False),
    ([{'type': 'paragraph', 'text': '@synthetic_bot_suffix'}], False),
    ([{'type': 'paragraph', 'text': '/reset@synthetic_bot'}], False),
    ([{'type': 'paragraph', 'text': '`@synthetic_bot`'}], False),
    ([{'type': 'paragraph', 'text': 'https://example.org/?q=@synthetic_bot'}], False),
    ([{'type': 'paragraph', 'text': {'type': 'url', 'text': '@synthetic_bot', 'url': 'https://example.org'}}], False),
    ([{'type': 'paragraph', 'text': {'type': 'code', 'text': '@synthetic_bot'}}], False),
    ([{'type': 'code', 'text': '@synthetic_bot'}], False),
    ([{'type': 'paragraph', 'text': ['@synthetic_', 'bot']}], False),
    ([{'type': 'paragraph', 'text': 'https://example.org/wake'}], False),
    ([{'type': 'paragraph', 'text': {'type': 'code', 'text': 'wake'}}], False),
    ([{'type': 'table', 'cells': [[{'text': '@synthetic_bot hello'}]]}], True),
    ([{'type': 'list', 'items': [{'blocks': [{'type': 'paragraph', 'text': '@synthetic_bot hello'}]}]}], True),
    ([{'type': 'blockquote', 'blocks': [{'type': 'paragraph', 'text': '@synthetic_bot hello'}]}], True),
    ([{'type': 'unknown', 'text': '@synthetic_bot hello'}], False),
    ([{'type': [], 'text': '@synthetic_bot hello'}], False),
])
@pytest.mark.asyncio
async def test_rich_trigger_boundaries(harness, blocks, expected):
    adapter, app, bot, errors = harness
    adapter.config.extra['require_mention'] = True
    adapter._mention_patterns = [re.compile('wake')]
    adapter._text_batch_delay_seconds = 0
    original = update(bot, chat={'id': -100, 'type': 'supergroup', 'title': 'Synthetic'},
                      rich_message={'blocks': blocks})
    await app.process_update(original)
    await flush(adapter)
    assert not errors
    assert adapter.handle_message.await_count == int(expected)
    if expected:
        event = adapter.handle_message.call_args.args[0]
        assert event.raw_message is original.message
        assert original.message.text is None
        assert event.message_type == MessageType.TEXT


@pytest.mark.parametrize('rich', [False, True])
@pytest.mark.parametrize('mode', ['wake', 'foreign', 'unauthorized', 'topic', 'guest_chat'])
@pytest.mark.asyncio
async def test_rich_plain_policy_parity(harness, rich, mode):
    adapter, app, bot, errors = harness
    adapter.config.extra.update(require_mention=True, observe_unmentioned_group_messages=True,
                                group_allowed_chats=['-100'])
    adapter._observe_unmentioned_group_message = Mock()
    adapter._text_batch_delay_seconds = 0
    text = {'wake': 'wake hello', 'foreign': 'wake @other_bot hello'}.get(mode, '@synthetic_bot hello')
    adapter._mention_patterns = [re.compile('wake')]
    adapter._note_bot_username('synthetic_bot')  # avoid the identity refresh network path
    adapter.config.extra.update({
        'unauthorized': {'guest_mode': True},
        'guest_chat': {'guest_mode': True, 'group_allowed_chats': ['-200']},
        'topic': {'allowed_topics': ['7']},
    }.get(mode, {}))
    fields = {
        'unauthorized': {'from': {'id': 43, 'is_bot': False, 'first_name': 'Unknown'}},
        'topic': {'message_thread_id': 8, 'is_topic_message': True},
    }.get(mode, {})
    u = update(bot, text=None if rich else text, rich=rich,
               chat={'id': -100, 'type': 'supergroup', 'title': 'Synthetic'},
               **({'rich_message': {'blocks': [{'type': 'paragraph', 'text': text}]}} if rich else {}),
               **fields)
    await app.process_update(u)
    await flush(adapter)
    assert not errors
    assert adapter.handle_message.await_count == int(mode in {'wake', 'guest_chat'})
    if mode in {'unauthorized', 'topic'}:
        adapter._observe_unmentioned_group_message.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_utf16_entities_still_authoritative(harness):
    adapter, app, bot, errors = harness
    adapter.config.extra['require_mention'] = True
    adapter._text_batch_delay_seconds = 0
    for message_id, offset in [(10, 3), (11, 2)]:
        await app.process_update(update(bot, rich=False, message_id=message_id,
            text='😀 @synthetic_bot hello',
            entities=[{'type': 'mention', 'offset': offset, 'length': 14}],
            chat={'id': -100, 'type': 'supergroup', 'title': 'Synthetic'}))
        await flush(adapter)
    assert not errors
    assert adapter.handle_message.await_count == 1


