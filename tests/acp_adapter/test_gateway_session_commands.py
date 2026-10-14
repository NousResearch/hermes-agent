"""``/queue``, ``/steer`` and the edit-policy mode on the gateway path, against a fake owner."""
import asyncio
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


def _agent(rpc):
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._snapshots['s'] = {'execution_generation': 7}
    return agent


@pytest.mark.asyncio
async def test_queue_submits_its_text_as_the_next_prompt():
    async def rpc(method, **params):
        assert method == 'prompt.submit' and params['text'] == 'run the linter', (method, params)
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'a',
                              'payload': {'outcome': 'completed', 'text': 'done'}})
        return {'admission_id': 'a'}

    agent = _agent(rpc)
    response = await asyncio.wait_for(agent.prompt([acp.text_block('/queue run the linter')], 's'), 5)
    assert response.stop_reason == 'end_turn'


@pytest.mark.asyncio
async def test_steer_reaches_this_editors_running_turn():
    async def rpc(method, **params):
        assert method == 'session.steer', method
        assert params == {'session_id': 's', 'text': 'use pytest', 'execution_generation': 7}
        return {'status': 'queued', 'text': 'use pytest', 'execution_generation': 7, 'authority_epoch': 1}

    agent = _agent(rpc)
    agent._admissions['s'] = 'running-turn'
    response = await agent.prompt([acp.text_block('/steer use pytest')], 's')
    assert response.stop_reason == 'end_turn'
    assert 'Steer queued for the active turn: use pytest' in \
        agent._conn.session_update.await_args.kwargs['update'].content.text


@pytest.mark.asyncio
async def test_a_refused_steer_is_reported_and_never_resent_as_a_prompt():
    from hermes_cli.gateway_client import GatewayRPCError
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        raise GatewayRPCError('unsupported_control')

    agent = _agent(rpc)
    agent._admissions['s'] = 'running-turn'
    with pytest.raises(GatewayRPCError, match='unsupported_control'):
        await agent.prompt([acp.text_block('/steer use pytest')], 's')
    assert calls == ['session.steer']


@pytest.mark.asyncio
async def test_a_steer_for_a_turn_that_just_ended_becomes_the_next_prompt():
    from hermes_cli.gateway_client import GatewayRPCError
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'session.steer':
            raise GatewayRPCError('stale_generation')
        assert method == 'prompt.submit' and params['text'] == 'use pytest', (method, params)
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'a',
                              'payload': {'outcome': 'completed', 'text': 'done'}})
        return {'admission_id': 'a'}

    agent = _agent(rpc)
    agent._admissions['s'] = 'running-turn'
    response = await asyncio.wait_for(agent.prompt([acp.text_block('/steer use pytest')], 's'), 5)
    assert response.stop_reason == 'end_turn' and calls == ['session.steer', 'prompt.submit']


@pytest.mark.asyncio
async def test_only_the_owners_edit_policy_can_be_selected():
    agent = _agent(AsyncMock())
    await agent.set_session_mode('default', 's')
    await agent.set_config_option('edit_approval_policy', 's', 'ask')
    with pytest.raises(GatewayClientError, match='acp_edit_policy_mutation_unavailable'):
        await agent.set_session_mode('accept_edits', 's')
    agent._gateway.rpc.assert_not_awaited()
