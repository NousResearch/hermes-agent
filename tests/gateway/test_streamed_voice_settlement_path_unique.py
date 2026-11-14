"""Streamed completion audio must be delivered only after its admission commits."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_results import execution_result
from hermes_state_runtime import get_session_admission
from tests.gateway.test_auto_voice_reply_format import _make_adapter, _make_runner
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit, REF


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['completed', 'different_result', 'failed'])
async def test_streamed_completion_delivers_voice_after_commit_without_resending_text(tmp_path, monkeypatch, case):
    db, authority = _authority(tmp_path, monkeypatch, Platform.TELEGRAM)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=False)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode['telegram:123'] = 'voice_only'
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda source: adapter
    authority.runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='123', user_id='human')
    authority.sessions['s'].source = source
    import gateway.session_ingress as ingress
    monkeypatch.setattr(ingress, 'MessageEvent', lambda **kw: MessageEvent(
        text=kw['text'], source=source, internal=True, metadata={'presentation_origin': 'voice'}))
    sent = []
    adapter._send_final_text = AsyncMock(side_effect=lambda *a, **k: sent.append('text'))
    async def voice(event, response):
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert (row['status'], row['outcome']) == ('terminal', 'completed')
        sent.append(('voice', response))
    runner._send_voice_reply = voice
    async def handle(event):
        answer = 'streamed answer'
        result = {'final_response': answer, 'messages': [], 'completed': True, 'already_sent': True,
                  'media_already_delivered': True}
        execution_result.get()['result'] = result
        response = await runner._hmwa_deliver_turn_response(event, source, SimpleNamespace(session_id='s'),
                                                            's', 1, result, [], answer, '', False)
        if case == 'different_result':
            execution_result.get()['terminal_result'] = dict(result)
        if case == 'failed':
            result['failed'] = True
        return response
    runner._handle_message = handle
    with db:
        receipt = await _submit(authority, 'streamed')
        await asyncio.wait_for(authority._drain(REF), 5)
        assert sent == ([('voice', 'streamed answer')] if case == 'completed' else [])


@pytest.mark.asyncio
@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('case', ['completed', 'different_result', 'failed'])
async def test_queued_terminal_result_speaks_only_after_commit(tmp_path, monkeypatch, streamed, case):
    from tests.gateway.test_queued_final_ledger import _chain_runner_and_ctx

    db, authority = _authority(tmp_path, monkeypatch, Platform.TELEGRAM)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=False)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode['telegram:123'] = 'voice_only'
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda source: adapter
    authority.runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='123', user_id='human')
    authority.sessions['s'].source = source
    import gateway.session_ingress as ingress
    monkeypatch.setattr(ingress, 'MessageEvent', lambda **kw: MessageEvent(
        text=kw['text'], source=source, internal=True, metadata={'presentation_origin': 'voice'}))
    monkeypatch.setattr(ingress, 'deliver_response', AsyncMock(return_value=True))
    sent = []
    async def voice(event, response):
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert (row['status'], row['outcome']) == ('terminal', 'completed')
        sent.append(response)
    runner._send_voice_reply = voice
    async def handle(event):
        answer = 'queued answer'
        result = {'final_response': answer, 'messages': [], 'completed': True,
                  'already_sent': streamed, 'media_already_delivered': True}
        cls, chain, ctx, terminal = _chain_runner_and_ctx(result)
        terminal.internal = True
        terminal.metadata = {'presentation_origin': 'voice'}
        execution_result.get()['result'] = result
        if case == 'different_result':
            execution_result.get()['terminal_result'] = dict(result)
        elif case == 'failed':
            result['failed'] = True
        merged = await cls._run_agent_queued_followup(
            chain, ctx, adapter=None, pending='queued', pending_event=terminal,
            response='opening', result={'interrupted': True, 'messages': []}, stream_task=None)
        return await runner._hmwa_deliver_turn_response(
            event, source, SimpleNamespace(session_id='s'), 's', 1, merged, [], answer, '', False)
    runner._handle_message = handle
    with db:
        receipt = await _submit(authority, 'queued')
        await asyncio.wait_for(authority._drain(REF), 5)
        assert sent == (['queued answer'] if case == 'completed' else [])
        if case == 'completed':
            assert ingress.deliver_response.await_count == (0 if streamed else 1)
