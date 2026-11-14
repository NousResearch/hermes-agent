"""Queued answers use the terminal turn's trusted voice origin, not the opener's."""
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_ingress import executing_admission
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source
from tests.gateway.test_queued_final_ledger import _chain_runner_and_ctx


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "opening_voice,terminal_type,terminal_internal,terminal_origin,expected",
    [
        (True, MessageType.TEXT, False, None, False),
        (False, MessageType.TEXT, True, 'voice', True),
        (True, MessageType.TEXT, True, None, False),
        (False, MessageType.TEXT, False, 'voice', False),
        (True, MessageType.VOICE, False, None, False),
    ],
)
async def test_queued_terminal_drives_settlement_voice_candidate(
    monkeypatch, tmp_path, opening_voice, terminal_type, terminal_internal, terminal_origin, expected,
):
    terminal = MessageEvent(text='queued', source=_source(), message_id='terminal',
                            message_type=terminal_type, internal=terminal_internal,
                            metadata={'presentation_origin': terminal_origin} if terminal_origin else {})
    answer = {'final_response': 'terminal answer', 'messages': [], 'tools': [],
              'history_offset': 0, 'last_prompt_tokens': 0, 'api_calls': 1, 'failed': False}
    cls, chain, ctx, _ = _chain_runner_and_ctx(answer)
    merged = await cls._run_agent_queued_followup(
        chain, ctx, adapter=None, pending='queued', pending_event=terminal,
        response='first', result={'interrupted': True, 'messages': []}, stream_task=None)

    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=merged)
    runner._send_voice_reply = AsyncMock()
    opener = _event(internal=opening_voice)
    if opening_voice:
        opener.metadata['presentation_origin'] = 'voice'
    token = executing_admission.set(True)
    try:
        await runner._handle_message_with_agent(opener, opener.source,
            'agent:main:telegram:group:-1001:12345', 1)
    finally:
        executing_admission.reset(token)
    assert (getattr(opener, '_deferred_voice_response', None) == 'terminal answer') is expected
    assert opener.message_type == terminal_type
    assert opener.internal is terminal_internal
    runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_deepest_queued_event_wins():
    deepest = MessageEvent(text='last', source=_source(), message_type=MessageType.VOICE)
    cls, chain, ctx, middle = _chain_runner_and_ctx({
        'final_response': 'last answer', 'messages': [],
        'queued_terminal_inbound_id': 'last', '_queued_terminal_event': deepest})
    result = await cls._run_agent_queued_followup(
        chain, ctx, adapter=None, pending='middle', pending_event=middle,
        response='first', result={'interrupted': True, 'messages': []}, stream_task=None)
    assert result['_queued_terminal_event'] is deepest
