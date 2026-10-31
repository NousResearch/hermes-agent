import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import prepare_native, restore_native
from gateway.session_ingress_media import ingress_fingerprint
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize('echo_enabled', [True, False])
@pytest.mark.parametrize('successful', [True, False])
async def test_voice_interrupt_is_transcribed_and_echoed_once_across_native_admission(
    tmp_path, monkeypatch, echo_enabled, successful,
):
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '123')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='test'))
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.send = AsyncMock()
    monkeypatch.setattr(runner, '_should_echo_stt_transcripts', lambda: echo_enabled)
    transcribe = AsyncMock(side_effect=lambda path, *_: (
        ('spoken words', '"spoken words"') if successful else (None, runner._untranscribed_audio_note(path))
    ))
    monkeypatch.setattr(runner, '_transcribe_one_clip', transcribe)
    audio = tmp_path / 'voice.ogg'
    audio.write_bytes(b'voice recording')
    source = adapter.build_source(chat_id='123', user_id='123', chat_type='dm')
    event = MessageEvent(text='caption', source=source, message_id='7',
                         message_type=MessageType.VOICE, media_urls=[str(audio)],
                         media_types=['audio/ogg'])

    await runner._transcribe_and_echo_pending_voice(
        event, adapter, source, event.text, log_context='Voice-busy-interrupt',
    )
    payload = json.loads(json.dumps(await prepare_native(runner, event)))
    audio.unlink()
    restored = restore_native(payload, runner)
    text = await runner._prepare_inbound_message_text(event=restored, source=restored.source, history=[])
    await runner._transcribe_and_echo_pending_voice(
        restored, adapter, restored.source, text, log_context='Voice-drain',
    )

    expected = '"spoken words"' if successful else runner._untranscribed_audio_note(restored.media_urls[0])
    assert text == expected + '\n\ncaption'
    assert Path(restored.media_urls[0]).read_bytes() == b'voice recording'
    assert transcribe.await_count == 1
    assert adapter.send.await_count == int(echo_enabled and successful)


@pytest.mark.asyncio
@pytest.mark.parametrize('record_ingress', [True, False])
@pytest.mark.parametrize('first_busy', [True, False])
async def test_redelivery_reuses_admitted_voice_cache_and_rejects_changed_audio(
    tmp_path, monkeypatch, record_ingress, first_busy,
):
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '123')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='test'))
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.send = AsyncMock()
    monkeypatch.setattr(runner, '_should_echo_stt_transcripts', lambda: True)
    transcribe = AsyncMock(return_value=('words', '"words"'))
    monkeypatch.setattr(runner, '_transcribe_one_clip', transcribe)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='voice')
    source = adapter.build_source(chat_id='123', user_id='123', chat_type='dm')
    timestamp = datetime(2026, 10, 31, tzinfo=UTC)

    async def deliver(name, busy=False, audio_bytes=b'voice recording'):
        audio = tmp_path / name
        audio.write_bytes(audio_bytes)
        event = MessageEvent(text='', source=source, message_id='8', timestamp=timestamp,
                             message_type=MessageType.VOICE, media_urls=[str(audio)], media_types=['audio/ogg'])
        payload = await prepare_native(runner, event)
        assert 'voice_stt' not in payload['native_text_v1']
        assert not hasattr(restore_native(payload, runner), '_gateway_pending_stt_text')
        if record_ingress:
            event._ingress_sha256 = ingress_fingerprint(payload)
        if busy:
            await runner._transcribe_and_echo_pending_voice(event, adapter, source, '', log_context='interrupt')
        return await authority.admit_native(event)

    try:
        first = await deliver('first.ogg', busy=first_busy)
        retry = await deliver('retry.ogg', busy=not first_busy)
        assert first.admission_id == retry.admission_id
        row = get_session_admission(authority.db, admission_id=first.admission_id)
        restored = restore_native(row['payload'], runner)
        assert hasattr(restored, '_gateway_pending_stt_text') is first_busy
        assert transcribe.await_count == adapter.send.await_count == 1
        if first_busy:
            assert await runner._prepare_inbound_message_text(
                event=restored, source=restored.source, history=[],
            ) == '"words"'
            assert transcribe.await_count == adapter.send.await_count == 1
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver('changed.ogg', audio_bytes=b'changed recording')
    finally:
        authority.db.close()
