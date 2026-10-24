"""Runner-side TTS failures are visible without exposing provider messages."""
import json
from unittest.mock import AsyncMock, patch

import pytest
from gateway.config import Platform
from gateway.run_voice import GatewayVoiceMixin
from tests.gateway.test_base_auto_tts_output_format import _DummyAdapter, _make_voice_event


class Runner(GatewayVoiceMixin):
    def __init__(self, adapter):
        self.adapter = adapter
        adapter.gateway_runner = self

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _reply_anchor_for_event(self, event):
        return event.message_id

    def _thread_metadata_for_source(self, source, anchor=None):
        return {'thread_id': source.thread_id}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['result', 'json', 'exception', 'empty'])
async def test_runner_synthesis_failure_notifies_once(failure):
    adapter = _DummyAdapter(Platform.TELEGRAM)
    runner = Runner(adapter)
    event = _make_voice_event(Platform.TELEGRAM)
    kwargs = {'return_value': json.dumps({'success': False, 'error': 'private-secret'})}
    if failure == 'json':
        kwargs = {'return_value': 'not json private-secret'}
    elif failure == 'exception':
        kwargs = {'side_effect': RuntimeError('private-secret')}
    elif failure == 'empty':
        kwargs = {'return_value': json.dumps({'success': True, 'file_paths': []})}
    with patch('tools.tts_tool.text_to_speech_tool', **kwargs):
        await runner._send_voice_reply(event, 'hello')
        await runner._send_voice_reply(event, 'hello again')
    assert len(adapter.sent) == 1, 'runner synthesis failure must notify once per outage'
    assert 'Voice reply unavailable' in adapter.sent[0]['content']
    assert 'private-secret' not in adapter.sent[0]['content']


@pytest.mark.asyncio
async def test_runner_recovery_resets_notice_and_blank_is_quiet(tmp_path):
    adapter = _DummyAdapter(Platform.TELEGRAM)
    runner = Runner(adapter)
    runner._deliver_voice_reply = AsyncMock()
    event = _make_voice_event(Platform.TELEGRAM)
    failed = json.dumps({'success': False})
    def success(**kwargs):
        from pathlib import Path
        path = Path(kwargs['output_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'local test artifact')
        return json.dumps({'success': True, 'file_path': str(path)})
    with patch('tools.tts_tool.text_to_speech_tool', return_value=failed):
        await runner._send_voice_reply(event, 'first')
    with patch('tools.tts_tool.text_to_speech_tool', side_effect=success):
        await runner._send_voice_reply(event, 'recovered')
    runner._deliver_voice_reply.assert_awaited_once()
    with patch('tools.tts_tool.text_to_speech_tool', return_value=failed):
        await runner._send_voice_reply(event, 'next outage')
    assert len(adapter.sent) == 2
    with patch('tools.tts_tool.text_to_speech_tool') as synth:
        await runner._send_voice_reply(event, '')
    synth.assert_not_called()
    assert len(adapter.sent) == 2


@pytest.mark.asyncio
async def test_cancelled_synthesis_does_not_send_notice():
    import asyncio
    adapter = _DummyAdapter(Platform.TELEGRAM)
    runner = Runner(adapter)
    event = _make_voice_event(Platform.TELEGRAM)
    with patch('tools.tts_tool.text_to_speech_tool', side_effect=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            await runner._send_voice_reply(event, 'hello')
    assert not adapter.sent

