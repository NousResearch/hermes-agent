"""Terminal ACP responses describe the exact admission, not the session's latest turn."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp import RequestError
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent


def agent_with_admission():
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()
    agent._gateway.rpc.return_value = {'admission_id': 'current'}
    agent._conn = AsyncMock()
    return agent


async def complete(agent, admission_id, outcome, text, **payload):
    await agent._project({'session_id': 's', 'type': 'message.complete',
                          'admission_id': admission_id,
                          'payload': {'outcome': outcome, 'text': text, **payload}})


@pytest.mark.asyncio
@pytest.mark.parametrize('text,visible', [
    ('Operation interrupted: waiting for model response (0.3s elapsed).', ''),
    ('Partial answer', 'Partial answer'),
])
async def test_cancelled_admission_suppresses_only_interrupt_metadata(text, visible):
    agent = agent_with_admission()
    await complete(agent, 'other', 'completed', '')
    task = asyncio.create_task(agent.prompt([TextContentBlock(type='text', text='hello')], 's'))
    await asyncio.sleep(0)
    assert not task.done()
    await complete(agent, 'current', 'cancelled', text)
    response = await asyncio.wait_for(task, 2)
    assert response.stop_reason == 'cancelled'
    emitted = ''.join(call.kwargs['update'].content.text
                      for call in agent._conn.session_update.await_args_list)
    assert emitted == visible
    assert 'other' in agent._terminals and 'current' not in agent._terminals


@pytest.mark.asyncio
async def test_refused_reattach_after_a_replay_gap_still_fails_the_pump():
    """A gap re-attaches that session (test_gateway_replay_gap); a refused re-attach ends the wait as
    before: the pump fails, the next call redials."""
    from hermes_cli.gateway_client import GatewayRPCError
    agent = agent_with_admission()
    agent._snapshots['s'] = {'replay_epoch': 'e', 'last_sequence': 9}
    agent._gateway.events = asyncio.Queue()
    agent._gateway.events.put_nowait({'method': 'event', 'params': {
        'session_id': 's', 'type': 'session.replay_gap', 'payload': {'reason': 'subscriber_overflow'}}})
    agent._gateway.rpc.side_effect = GatewayRPCError('not_found')
    await asyncio.wait_for(agent._events(), 2)
    assert isinstance(agent._failure, GatewayRPCError) and str(agent._failure) == 'not_found'


@pytest.mark.asyncio
async def test_failed_admission_raises_without_poisoning_the_next_success():
    agent = agent_with_admission()
    await complete(agent, 'current', 'failed', 'The admitted turn failed.')
    with pytest.raises(RequestError) as failed:
        await agent.prompt([TextContentBlock(type='text', text='hello')], 's')
    assert failed.value.to_error_obj()['data']['details'] == 'admitted_turn_failed'
    assert 'current' not in agent._terminals
    await complete(agent, 'current', 'completed', 'Operation interrupted: quoted model text')
    response = await agent.prompt([TextContentBlock(type='text', text='retry')], 's')
    assert response.stop_reason == 'end_turn'
    assert agent._conn.session_update.await_args.kwargs['update'].content.text == (
        'Operation interrupted: quoted model text')


@pytest.mark.asyncio
async def test_failed_admission_error_carries_the_completion_cause():
    """The editor learns why the turn failed; code, message and data.details stay what they were."""
    agent = agent_with_admission()
    await complete(agent, 'current', 'failed', 'Provider said: HTTP 429: slow down\n',
                   failure_reason='rate_limit')
    with pytest.raises(RequestError) as failed:
        await agent.prompt([TextContentBlock(type='text', text='hello')], 's')
    assert failed.value.to_error_obj() == {'code': -32603, 'message': 'Internal error', 'data': {
        'details': 'admitted_turn_failed', 'failure_reason': 'rate_limit',
        'message': 'Provider said: HTTP 429: slow down'}}


@pytest.mark.asyncio
async def test_failed_terminal_receipt_error_has_only_the_details():
    """A retry's terminal receipt carries no completion text: the error is the bare one."""
    agent = agent_with_admission()
    agent._gateway.rpc.return_value = {'admission_id': 'current', 'status': 'terminal', 'outcome': 'failed'}
    with pytest.raises(RequestError) as failed:
        await agent.prompt([TextContentBlock(type='text', text='hello')], 's')
    assert failed.value.to_error_obj() == {
        'code': -32603, 'message': 'Internal error', 'data': {'details': 'admitted_turn_failed'}}
