"""Native media capture (copy + SHA-256 + fsyncs) runs off the owner loop, under the admission lease."""
import asyncio
import hashlib
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import restore_native
from hermes_state_runtime import get_session_admission
from tests.waiting import async_wait_until

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
@pytest.mark.parametrize('pause_after_publish', [False, True])
async def test_cancelled_native_capture_releases_unadmitted_bytes(tmp_path, monkeypatch, pause_after_publish):
    import gateway.session_envelope as envelope
    import gateway.session_ingress_media as media
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='cancel-media')
    source_file = tmp_path / 'clip.mp4'
    source_file.write_bytes(b'video-bytes' * 1000)
    event = MessageEvent(text='look', message_type=MessageType.VIDEO, message_id='cancel-v1',
                         source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                         media_urls=[str(source_file)], media_types=['video/mp4'])
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = media.capture_native_media

    def paused(paths):
        try:
            if not pause_after_publish:
                entered.set()
                assert release.wait(10)
            references = original(paths)
            if pause_after_publish:
                entered.set()
                assert release.wait(10)
            return references
        finally:
            finished.set()

    monkeypatch.setattr(envelope, 'capture_native_media', paused)
    try:
        admission = asyncio.create_task(authority.admit_native(event))
        assert await asyncio.to_thread(entered.wait, 10)
        admission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(admission, 3)
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        root = media._media_root()
        captured = root / hashlib.sha256(source_file.read_bytes()).hexdigest() / 'clip.mp4'
        await async_wait_until(lambda: str(captured) not in media._leases and not captured.exists(),
                               timeout=3, what='abandoned capture to be collected')
        assert str(captured) not in media._leases
        assert not captured.exists()
    finally:
        release.set()
        authority.db.close()


@pytest.mark.asyncio
async def test_native_capture_error_releases_published_bytes(tmp_path, monkeypatch):
    import gateway.session_ingress_media as media
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='error-media')
    source_file = tmp_path / 'clip.mp4'
    source_file.write_bytes(b'video-bytes' * 1000)
    target = media._media_root() / hashlib.sha256(source_file.read_bytes()).hexdigest() / source_file.name
    event = MessageEvent(text='look', message_type=MessageType.VIDEO, message_id='error-v1',
                         source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm'),
                         media_urls=[str(source_file)], media_types=['video/mp4'])

    def failed_fsync(_directory):
        raise OSError('disk failed after publish')

    monkeypatch.setattr(media, 'fsync_directory', failed_fsync)
    try:
        with pytest.raises(OSError, match='disk failed after publish'):
            await authority.admit_native(event)
        assert not target.exists()
        assert str(target) not in media._leases
    finally:
        authority.db.close()


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
