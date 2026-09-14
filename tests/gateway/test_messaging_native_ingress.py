"""Transport facts remain explicit and command routing keeps its native identity."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


@pytest.mark.parametrize('chat_type,private', [('private', True), ('supergroup', False), ('channel', False)])
@pytest.mark.parametrize('edited', [False, True])
def test_telegram_transport_privacy_edit_and_bot(chat_type, private, edited):
    from tests.gateway.test_telegram_reply_quote import _make_adapter, _make_message

    adapter, message = _make_adapter(), _make_message(text='/group 1 status')
    message.chat.type = chat_type
    message.chat.title = 'Test'
    message.edit_date = 1 if edited else None
    message.from_user.is_bot = True
    event = adapter._build_message_event(message, MessageType.COMMAND)
    assert event.source.is_one_to_one is private
    assert event.source.message_is_edit is edited
    assert event.source.is_bot is True
    assert event.message_id == '1001'
    assert event.text == '/group 1 status'
    assert event.message_type == MessageType.COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize('private', [False, True])
@pytest.mark.parametrize('edited', [False, True])
async def test_discord_native_message_provenance(monkeypatch, private, edited):
    from plugins.platforms.discord import adapter as module
    from tests.gateway.test_discord_double_dispatch import _TextChannel, _make_message

    class DM:
        id = 100
        guild = None

    monkeypatch.setattr(module.discord, 'DMChannel', DM)
    monkeypatch.setenv('DISCORD_REQUIRE_MENTION', 'false')
    monkeypatch.setenv('DISCORD_AUTO_THREAD', 'false')
    monkeypatch.setenv('DISCORD_HISTORY_BACKFILL', 'false')
    adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token='test-token'))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    adapter._text_batch_delay_seconds = 0
    adapter._is_allowed_user = lambda *args, **kwargs: True
    adapter.handle_message = AsyncMock()
    message = _make_message(channel=DM() if private else _TextChannel(), content='/group 1 status')
    message.edited_at = 1 if edited else None
    message.author.bot = True
    await adapter._handle_message(message)
    event = adapter.handle_message.await_args.args[0]
    assert event.source.is_one_to_one is private
    assert event.source.message_is_edit is edited
    assert event.source.is_bot is True
    assert event.message_type == MessageType.COMMAND
    assert event.text == '/group 1 status'


@pytest.mark.parametrize('private', [False, True])
def test_discord_slash_event_is_fresh_and_native(monkeypatch, private):
    from plugins.platforms.discord import adapter as module
    from tests.gateway.test_discord_double_dispatch import _TextChannel

    class DM:
        id = 100

    monkeypatch.setattr(module.discord, 'DMChannel', DM)
    adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token='test-token'))
    event = adapter._build_slash_event(
        SimpleNamespace(channel=DM() if private else _TextChannel(), channel_id=100, guild_id=None,
                        user=SimpleNamespace(id=7, display_name='Alice')), '/group 1 status')
    assert event.source.is_one_to_one is private
    assert event.source.message_is_edit is False
    assert event.message_type == MessageType.COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize('edited', [False, True])
async def test_signal_keeps_attachment_presence_when_bytes_are_ignored(monkeypatch, edited):
    from tests.gateway.test_signal import _make_signal_adapter

    adapter = _make_signal_adapter(monkeypatch, ignore_attachments=True)
    adapter.handle_message = AsyncMock()
    data = {'message': '/group 1 status', 'attachments': [{'id': 'file-1'}]}
    envelope = {'sourceNumber': '+15559876543', 'timestamp': 1700000000000}
    envelope.update({'editMessage': {'dataMessage': data}} if edited else {'dataMessage': data})
    await adapter._handle_envelope({'envelope': envelope})
    event = adapter.handle_message.await_args.args[0]
    assert event.source.is_one_to_one is True
    assert event.source.message_is_edit is edited
    assert event.source.message_had_attachments is True
    assert not event.media_urls
    assert event.text == '/group 1 status'


@pytest.mark.asyncio
@pytest.mark.parametrize('is_group,private', [(None, False), (False, True), (True, False), ('false', False)])
@pytest.mark.parametrize('owner', [False, True])
async def test_whatsapp_requires_explicit_private_fact(monkeypatch, is_group, private, owner):
    from tests.gateway.test_whatsapp_from_owner import _make_adapter, _dm_payload

    adapter = _make_adapter()
    adapter._should_process_message = lambda data: True
    event = await adapter._build_message_event(_dm_payload(
        isGroup=is_group, fromMe=True, fromOwner=owner, isEdited=True,
        nativeType='protocolMessage:MESSAGE_EDIT', body='/group 1 status'))
    assert event is not None
    assert event.source.is_one_to_one is private
    assert event.source.is_bot is (not owner)
    assert event.source.message_is_edit is True
    assert event.metadata['message_is_edit'] is True


@pytest.mark.asyncio
async def test_mattermost_does_not_invent_private_proof_for_failed_attachment():
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter(PlatformConfig(enabled=True, token='test-token', extra={'url': 'https://example.invalid'}))
    adapter._download_attachments = AsyncMock(return_value=([], []))
    adapter.handle_message = AsyncMock()
    await adapter._handle_ws_event({'event': 'posted', 'data': {
        'channel_type': 'D', 'sender_name': 'Alice',
        'post': json.dumps({'id': 'post-1', 'user_id': 'alice', 'channel_id': 'channel-1',
                            'message': '/group 1 status', 'file_ids': ['file-1']})}})
    event = adapter.handle_message.await_args.args[0]
    assert event.source.message_had_attachments is True
    assert event.source.is_one_to_one is None
    assert not event.media_urls
    assert event.message_type == MessageType.COMMAND


def test_transport_facts_are_not_rehydrated_from_serialized_sessions():
    from gateway.config import Platform
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id='1', chat_type='dm',
                           is_one_to_one=True, message_is_edit=True, message_had_attachments=True)
    persisted = source.to_dict()
    assert not {'is_one_to_one', 'message_is_edit', 'message_had_attachments'} & persisted.keys()
    restored = SessionSource.from_dict(persisted)
    assert restored.is_one_to_one is None
    assert restored.message_is_edit is False
    assert restored.message_had_attachments is False
