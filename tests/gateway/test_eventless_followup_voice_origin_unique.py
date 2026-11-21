from unittest.mock import AsyncMock

import pytest

from gateway.session_ingress import executing_admission
from tests.gateway.test_gateway_silence_tokens import _event, _runner
from tests.gateway.test_queued_final_ledger import _chain_runner_and_ctx


@pytest.mark.asyncio
async def test_eventless_followup_preserves_opening_voice_for_settlement(monkeypatch, tmp_path):
    answer = {'final_response': 'terminal answer', 'messages': [], 'tools': [],
              'history_offset': 0, 'last_prompt_tokens': 0, 'api_calls': 1, 'failed': False}
    cls, chain, ctx, _ = _chain_runner_and_ctx(answer)
    ctx.channel_prompt = None
    merged = await cls._run_agent_queued_followup(
        chain, ctx, adapter=None, pending='steer', pending_event=None,
        response='first', result={'interrupted': True, 'messages': []}, stream_task=None)
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=merged)
    runner._send_voice_reply = AsyncMock()
    opener = _event(internal=True)
    opener.metadata['presentation_origin'] = 'voice'
    token = executing_admission.set(True)
    try:
        await runner._handle_message_with_agent(opener, opener.source,
            'agent:main:telegram:group:-1001:12345', 1)
    finally:
        executing_admission.reset(token)
    assert opener.metadata.get('presentation_origin') == 'voice'
    assert getattr(opener, '_deferred_voice_response', None) == 'terminal answer'


@pytest.mark.asyncio
async def test_eventless_deep_followup_inherits_nearest_queued_event():
    cls, chain, ctx, middle = _chain_runner_and_ctx({
        'final_response': 'answer', 'messages': [],
        'queued_terminal_inbound_id': None, '_queued_terminal_event': None,
    })
    result = await cls._run_agent_queued_followup(
        chain, ctx, adapter=None, pending='middle', pending_event=middle,
        response='first', result={'interrupted': True, 'messages': []}, stream_task=None)
    assert result['_queued_terminal_event'] is middle
