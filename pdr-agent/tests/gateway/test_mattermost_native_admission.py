"""Mattermost posts pass native admission and run a turn, with or without attachments.

Contract:
V1 A plain Mattermost text post (no attachment) is admitted and runs a turn.
V2 A Mattermost post with an attachment is admitted, runs a turn and keeps its media.
"""
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult, cache_image_from_bytes_async
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

SENDER = 'user_123'
PNG_BYTES = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32


def _posted_event(post_id, message, file_ids=()):
    post = {'id': post_id, 'user_id': SENDER, 'channel_id': 'dm_chan', 'message': message,
            'file_ids': list(file_ids)}
    return {'event': 'posted', 'data': {'post': json.dumps(post), 'channel_type': 'D', 'sender_name': '@alice'}}


@pytest.mark.asyncio
async def test_real_mattermost_posts_run_a_turn_through_native_admission(tmp_path, monkeypatch):
    """Contract: V1, V2"""
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    monkeypatch.setenv('MATTERMOST_ALLOWED_USERS', SENDER)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token='test-token',
                                               extra={'url': 'https://mm.example.com'}))
    adapter._bot_user_id, adapter._bot_username = 'bot_user_id', 'hermes-bot'
    adapter.gateway_runner, runner.adapters = runner, {Platform.MATTERMOST: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='mm')
    runner._wire_adapter_handlers(adapter)
    image_path = await cache_image_from_bytes_async(PNG_BYTES, '.png')
    sent, turns = [], []

    async def download_attachments(file_ids):
        if not file_ids:
            return [], []
        return [image_path], ['image/png']

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    async def turn(event, *_args, **_kwargs):
        turns.append((event.text, len(event.media_urls), list(event.media_types)))
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(adapter, '_download_attachments', download_attachments)
    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    try:
        await adapter._handle_ws_event(_posted_event('p1', 'hello'))
        await async_wait_until(lambda: len(sent) >= 1, timeout=20, what='reply to a plain text post')
        await adapter._handle_ws_event(_posted_event('p2', 'look at this', file_ids=['f1']))
        await async_wait_until(lambda: len(sent) >= 2, timeout=20, what='reply to a post with an image')
        assert sent == ['ok', 'ok'], sent
        assert turns == [('hello', 0, []), ('look at this', 1, ['image/png'])], turns
    finally:
        authority.db.close()
