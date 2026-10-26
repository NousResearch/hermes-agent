"""An attachment an adapter could not cache locally (a remote URL left in ``media_urls``, as DingTalk
always does) is dropped with a note; the message and its other media are still admitted."""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import prepare_native

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_remote_media_url_is_dropped_with_a_note_instead_of_refusing_the_message(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='remote-media')
    local = tmp_path / 'clip.mp4'
    local.write_bytes(b'video-bytes')
    event = MessageEvent(text='see these', message_type=MessageType.PHOTO, message_id='m1',
                         source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                         media_urls=['https://static.dingtalk.com/media/abc.png', str(local)],
                         media_types=['image/png', 'video/mp4'])
    try:
        payload = await prepare_native(runner, event)
        envelope = payload['native_text_v1']
        assert payload['text'].startswith('see these\n') and 'image/png' in payload['text']
        assert [media['size'] for media in envelope['media']] == [len(b'video-bytes')]
        assert envelope['event']['media_types'] == ['video/mp4']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_an_attachment_only_message_stores_just_the_note(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='remote-media-only')
    event = MessageEvent(text='', message_type=MessageType.PHOTO, message_id='m2',
                         source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                         media_urls=['https://static.dingtalk.com/media/abc.png'], media_types=['image/png'])
    try:
        payload = await prepare_native(runner, event)
        assert payload['text'] == '[The user sent an attachment (image/png) that could not be retrieved.]'
    finally:
        authority.db.close()
