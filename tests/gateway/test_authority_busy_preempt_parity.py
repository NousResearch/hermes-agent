"""The authority's busy preempt keeps main's busy handler: voice steers, steer folds, busy ack."""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import list_session_admissions

CHAT = '15551234567@s.whatsapp.net'


class _RunningAgent:
    def __init__(self):
        self.steers, self.interrupts = [], []

    def steer(self, text):
        self.steers.append(text)
        return True

    def interrupt(self, *args):
        self.interrupts.append(args)

    def get_activity_summary(self):
        return {}


async def _setup(tmp_path, monkeypatch, mode):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='busy')
    runner._wire_adapter_handlers(adapter)
    sent = []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: mode)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='group')
    key = runner._session_key_for_source(source)
    agent = _RunningAgent()
    turn = runner._session_state(key).turn
    turn.agent = agent
    turn.event = MessageEvent(text='unaddressed chatter', source=source, message_id='t0', reply_expected=False)
    return runner, authority, source, agent, sent, turn


def _admitted(authority):
    return [row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


@pytest.mark.asyncio
async def test_busy_voice_note_steers_with_transcript_folds_and_acks(tmp_path, monkeypatch):
    runner, authority, source, agent, sent, turn = await _setup(tmp_path, monkeypatch, 'steer')

    async def transcribe(event, adapter, source, text, **_):
        return 'TRANSCRIBED voice follow-up', ['TRANSCRIBED voice follow-up']

    monkeypatch.setattr(runner, '_transcribe_and_echo_pending_voice', transcribe)
    monkeypatch.setattr(runner, '_pending_event_audio_paths', lambda event: list(event.media_urls))
    voice = tmp_path / 'note.ogg'
    voice.write_bytes(b'OggS')
    try:
        handled = asyncio.ensure_future(runner._handle_message(MessageEvent(
            text='', message_type=MessageType.VOICE, source=source, message_id='v1', reply_expected=True,
            media_urls=[str(voice)], media_types=['audio/ogg'])))
        done, _ = await asyncio.wait({handled}, timeout=10)
        handled.cancel()
        assert done, 'the voice note was admitted as a turn behind the running one instead of steering'
        assert len(agent.steers) == 1 and agent.steers[0].endswith('TRANSCRIBED voice follow-up'), agent.steers
        assert turn.event.reply_expected is True  # addressed steer folded: a silence marker can't end it
        assert sent, 'no busy ack'
        assert _admitted(authority) == []  # steered text is inside the run, never replayed
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_busy_voice_note_interrupts_with_transcript_in_interrupt_mode(tmp_path, monkeypatch):
    runner, authority, source, agent, sent, _turn = await _setup(tmp_path, monkeypatch, 'interrupt')

    async def transcribe(event, adapter, source, text, **_):
        return 'TRANSCRIBED stop that', ['TRANSCRIBED stop that']

    monkeypatch.setattr(runner, '_transcribe_and_echo_pending_voice', transcribe)
    monkeypatch.setattr(runner, '_pending_event_audio_paths', lambda event: list(event.media_urls))
    monkeypatch.setattr(runner, '_handle_message_with_agent', lambda *a, **k: None)
    voice = tmp_path / 'note.ogg'
    voice.write_bytes(b'OggS')
    try:
        task = asyncio.ensure_future(runner._handle_message(MessageEvent(
            text='', message_type=MessageType.VOICE, source=source, message_id='v2',
            media_urls=[str(voice)], media_types=['audio/ogg'])))
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: _admitted(authority), timeout=10, what='the voice note admitted')
        task.cancel()
        assert agent.interrupts == [('TRANSCRIBED stop that',)], agent.interrupts
        assert sent, 'no busy ack'
    finally:
        authority.db.close()
