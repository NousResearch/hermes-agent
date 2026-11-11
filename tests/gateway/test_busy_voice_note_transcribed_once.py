"""A voice note that interrupts a busy turn is transcribed and echoed ONCE, though the session
authority's FIFO runs it from a freshly restored event (#152457).

The busy preempt transcribes the clip for ``agent.interrupt()`` and echoes it; native admission
then re-captures the clip and the FIFO turn restores a new ``MessageEvent`` without that event's
STT cache, so the turn's preprocessing used to pay STT and echo the transcript a second time.
"""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import restore_native
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until

CHAT = '15551234567@s.whatsapp.net'


class _RunningAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, *args):
        self.interrupts.append(args)

    def get_activity_summary(self):
        return {}


async def _busy_rig(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='busy-voice')
    runner._wire_adapter_handlers(adapter)
    sent = []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_should_echo_stt_transcripts', lambda: True)
    monkeypatch.setattr(runner, '_handle_message_with_agent', lambda *a, **k: None)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')
    turn = runner._session_state(runner._session_key_for_source(source)).turn
    turn.agent = _RunningAgent()
    turn.event = MessageEvent(text='long task', source=source, message_id='t0')
    return runner, authority, source, turn.agent, sent


def _rows(authority):
    return [row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


@pytest.mark.asyncio
async def test_busy_voice_note_is_transcribed_and_echoed_once(tmp_path, monkeypatch):
    calls = []

    def transcribe_audio(path, model=None, source=None):
        calls.append(path)
        return {'success': True, 'transcript': 'stop and check the logs', 'provider': 'fake'}

    monkeypatch.setattr('tools.transcription_tools.transcribe_audio', transcribe_audio)
    runner, authority, source, agent, sent = await _busy_rig(tmp_path, monkeypatch)
    clip = tmp_path / 'note.ogg'
    clip.write_bytes(b'OggS voice note bytes')
    try:
        task = asyncio.ensure_future(runner._handle_message(MessageEvent(
            text='', message_type=MessageType.VOICE, source=source, message_id='v1',
            media_urls=[str(clip)], media_types=['audio/ogg'])))
        await async_wait_until(lambda: _rows(authority), timeout=10, what='the voice note admitted')
        task.cancel()
        assert agent.interrupts == [('"stop and check the logs"',)]  # the busy path paid STT once

        # The FIFO turn: what the drain restores from the committed row, through main's preprocessing.
        restored = restore_native(_rows(authority)[0]['payload'], runner)
        text = await runner._prepare_inbound_message_text(event=restored, source=restored.source, history=[])

        assert '"stop and check the logs"' in text
        assert len(calls) == 1, calls
        assert [m for m in sent if 'stop and check the logs' in m] == ['🎙️ "stop and check the logs"']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_same_voice_note_sent_again_while_idle_is_echoed_again(tmp_path, monkeypatch):
    """The hand-off is consumed by the one turn it was for: the clip resent later echoes anew."""
    monkeypatch.setattr('tools.transcription_tools.transcribe_audio',
                        lambda path, model=None, source=None: {'success': True, 'transcript': 'hi', 'provider': 'f'})
    runner, authority, source, _agent, sent = await _busy_rig(tmp_path, monkeypatch)
    clip = tmp_path / 'note.ogg'
    clip.write_bytes(b'OggS same clip')
    try:
        task = asyncio.ensure_future(runner._handle_message(MessageEvent(
            text='', message_type=MessageType.VOICE, source=source, message_id='v1',
            media_urls=[str(clip)], media_types=['audio/ogg'])))
        await async_wait_until(lambda: _rows(authority), timeout=10, what='the voice note admitted')
        task.cancel()
        restored = restore_native(_rows(authority)[0]['payload'], runner)
        await runner._prepare_inbound_message_text(event=restored, source=restored.source, history=[])
        again = MessageEvent(text='', message_type=MessageType.VOICE, source=source, message_id='v2',
                             media_urls=[str(clip)], media_types=['audio/ogg'])
        await runner._prepare_inbound_message_text(event=again, source=source, history=[])

        assert [m for m in sent if '"hi"' in m] == ['🎙️ "hi"', '🎙️ "hi"']
    finally:
        authority.db.close()


async def _admit_busy_voice(runner, authority, source, clip):
    task = asyncio.ensure_future(runner._handle_message(MessageEvent(
        text='', message_type=MessageType.VOICE, source=source, message_id='v1',
        media_urls=[str(clip)], media_types=['audio/ogg'])))
    await async_wait_until(lambda: _rows(authority), timeout=10, what='the voice note admitted')
    task.cancel()
    return restore_native(_rows(authority)[0]['payload'], runner)


@pytest.mark.asyncio
async def test_same_clip_forwarded_to_another_chat_still_echoes_there(tmp_path, monkeypatch):
    """The "already echoed" hand-off belongs to the busy MESSAGE: the same clip arriving idle in
    another chat before that message's FIFO turn runs is echoed in that chat, and only once overall."""
    monkeypatch.setattr('tools.transcription_tools.transcribe_audio',
                        lambda path, model=None, source=None: {'success': True, 'transcript': 'hi', 'provider': 'f'})
    runner, authority, source, _agent, sent = await _busy_rig(tmp_path, monkeypatch)
    clip = tmp_path / 'note.ogg'
    clip.write_bytes(b'OggS forwarded clip')
    other = runner.adapters[Platform.WHATSAPP].build_source(
        chat_id='15557654321@s.whatsapp.net', user_id=CHAT, chat_type='dm')
    try:
        restored = await _admit_busy_voice(runner, authority, source, clip)
        forwarded = MessageEvent(text='', message_type=MessageType.VOICE, source=other, message_id='f1',
                                 media_urls=[str(clip)], media_types=['audio/ogg'])
        await runner._prepare_inbound_message_text(event=forwarded, source=other, history=[])
        assert [m for m in sent if '"hi"' in m] == ['🎙️ "hi"', '🎙️ "hi"']  # busy echo + chat B's own
        await runner._prepare_inbound_message_text(event=restored, source=restored.source, history=[])
        assert [m for m in sent if '"hi"' in m] == ['🎙️ "hi"', '🎙️ "hi"']  # chat A's turn: already shown
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_clip_listed_twice_in_one_message_echoes_once(tmp_path, monkeypatch):
    calls = []

    def transcribe_audio(path, model=None, source=None):
        calls.append(path)
        return {'success': True, 'transcript': 'twice', 'provider': 'f'}

    monkeypatch.setattr('tools.transcription_tools.transcribe_audio', transcribe_audio)
    runner, authority, source, _agent, sent = await _busy_rig(tmp_path, monkeypatch)
    clip = tmp_path / 'note.ogg'
    clip.write_bytes(b'OggS current and quoted media')
    try:
        event = MessageEvent(text='', message_type=MessageType.VOICE, source=source, message_id='d1',
                             media_urls=[str(clip), str(clip)], media_types=['audio/ogg', 'audio/ogg'])
        await runner._prepare_inbound_message_text(event=event, source=source, history=[])

        assert len(calls) == 1
        assert [m for m in sent if '"twice"' in m] == ['🎙️ "twice"']
    finally:
        authority.db.close()
