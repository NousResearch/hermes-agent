"""Native media capture (copy + SHA-256 + fsyncs) runs off the owner loop, under the admission lease."""
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import restore_native
from hermes_state_runtime import get_session_admission

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_native_media_capture_runs_off_loop_and_commits_owned_bytes(tmp_path, monkeypatch):
    import gateway.session_envelope as envelope
    import gateway.session_ingress_media as media
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    loop_thread, captured = threading.current_thread(), []
    real_capture = media.capture_native_media

    def capture(paths):
        captured.append((threading.current_thread() is loop_thread, media._lease_scope.get() is not None))
        return real_capture(paths)

    monkeypatch.setattr(envelope, 'capture_native_media', capture)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='media')
    try:
        attachment = tmp_path / 'clip.mp4'
        attachment.write_bytes(b'video-bytes' * 1000)
        event = MessageEvent(text='look', message_type=MessageType.VIDEO, message_id='v1',
                             source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                             media_urls=[str(attachment)], media_types=['video/mp4'])
        receipt = await authority.admit_native(event)
        assert captured == [(False, True)], captured  # worker thread, lease carried by the context
        attachment.write_bytes(b'changed')
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        restored = restore_native(row['payload'], runner)
        from pathlib import Path
        assert Path(restored.media_urls[0]).read_bytes() == b'video-bytes' * 1000
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_native_admission_does_not_rehash_captured_media_on_the_loop(tmp_path, monkeypatch):
    """After the off-loop capture, admission derives the route's source without re-verifying the
    captured bytes on the owner loop (B1-07 remainder); execution still verifies them."""
    import hashlib
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    loop_thread, on_loop = threading.current_thread(), []
    real_digest = hashlib.file_digest

    def digest(*a, **k):
        on_loop.append(threading.current_thread() is loop_thread)
        return real_digest(*a, **k)
    monkeypatch.setattr(hashlib, 'file_digest', digest)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='media')
    try:
        attachment = tmp_path / 'clip.mp4'
        attachment.write_bytes(b'video-bytes' * 1000)
        event = MessageEvent(text='look', message_type=MessageType.VIDEO, message_id='v2',
                             source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                             media_urls=[str(attachment)], media_types=['video/mp4'])
        receipt = await authority.admit_native(event)
        assert receipt.admission_id
        assert True not in on_loop, f'{on_loop.count(True)} captured-media re-hash(es) on the owner loop'
    finally:
        authority.db.close()
