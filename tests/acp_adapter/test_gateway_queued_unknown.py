"""Editor prompts fail recoverably when their FIFO becomes unknown."""
import asyncio
from unittest.mock import AsyncMock, call

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('before_ack', [False, True])
async def test_editor_follower_reports_unknown_without_poisoning_other_sessions(before_ack):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {}
    agent._snapshots['other'] = {}
    unknown = {'session_id': 's', 'type': 'session.info', 'payload': {'pending': [
        {'admission_id': 'head', 'status': 'unknown'}, {'admission_id': 'ours', 'status': 'queued'}]}}

    async def submit(method, **params):
        assert method == 'prompt.submit'
        if params['session_id'] == 'other':
            await agent._project({'session_id': 'other', 'type': 'message.complete',
                                  'admission_id': 'other-turn', 'payload': {'outcome': 'completed'}})
            return {'admission_id': 'other-turn'}
        if before_ack:
            await agent._project(unknown)
        return {'admission_id': 'ours'}

    agent._gateway.rpc.side_effect = submit
    submitted = asyncio.create_task(agent.prompt([acp.text_block('queued')], 's'))
    if not before_ack:
        await asyncio.sleep(0)
        await agent._project(unknown)
    with pytest.raises(GatewayClientError, match='unknown_execution: do not resend accepted input'):
        await asyncio.wait_for(submitted, 5)
    agent._gateway.rpc.assert_awaited_once()
    assert agent._failure is None
    assert (await agent.prompt([acp.text_block('independent')], 'other')).stop_reason == 'end_turn'


@pytest.mark.asyncio
async def test_editor_refuses_new_prompt_until_lost_turn_is_resolved():
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await asyncio.wait_for(agent.prompt([acp.text_block('new input')], 's'), 5)
    agent._gateway.rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_resumed_unknown_turn_can_be_discarded_without_leaving_the_editor():
    # The editor needs the same fenced recovery as classic chat.  It must use the
    # unknown row's generation, not the session's current generation.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    lost = 'admission-lost-0123456789abcdef'
    unknown = {'session_id': 's', 'messages': [], 'prompts': [],
               'pending': [{'admission_id': lost, 'status': 'unknown', 'execution_generation': 4}]}
    cleared = {**unknown, 'pending': []}
    agent._snapshots['s'] = unknown
    agent._gateway.rpc.side_effect = lambda method, **params: {
        'session.resume': unknown if len(agent._gateway.rpc.await_args_list) == 1 else cleared,
        'prompt.resolve_unknown': {'admission_id': lost, 'status': 'terminal', 'outcome': 'interrupted'},
    }[method]

    response = await agent.prompt([acp.text_block(f'/discard {lost}')], 's')

    assert response.stop_reason == 'end_turn'
    assert agent._snapshots['s']['pending'] == []
    assert agent._gateway.rpc.await_args_list == [
        call('session.resume', session_id='s'),
        call('prompt.resolve_unknown', session_id='s', admission_id=lost, execution_generation=4),
        call('session.resume', session_id='s'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('yield_between', [False, True])
async def test_editor_own_lost_turn_reports_unknown_not_end_turn(yield_between):
    # The prompt's own started turn is lost: whichever frame wakes the prompt first, the
    # editor gets the unknown-execution error, never a successful end_turn.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {}

    async def lose_worker():
        await agent._project({'session_id': 's', 'type': 'session.info',
                              'payload': {'pending': [{'admission_id': 'ours', 'status': 'unknown'}]}})
        if yield_between:
            # session.info alone must wake and settle the prompt before the completion arrives.
            await async_wait_until(submitted.done, what='the prompt woke on session.info')
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'ours',
                              'payload': {'outcome': 'unknown', 'text': 'Worker execution is unknown.'}})

    agent._gateway.rpc.return_value = {'admission_id': 'ours'}
    submitted = asyncio.create_task(agent.prompt([acp.text_block('work')], 's'))
    await asyncio.sleep(0)
    await lose_worker()
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await asyncio.wait_for(submitted, 5)
