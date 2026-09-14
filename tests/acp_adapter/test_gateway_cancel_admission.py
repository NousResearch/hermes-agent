"""session/cancel cancels the ACP-owned admission, never another surface's running turn."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


def agent_with_queued_prompt():
    agent = GatewayACPAgent()
    # Another viewer's turn is executing under generation 7 when we are queued.
    agent._snapshots['s'] = {'execution_generation': 7}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    return agent


async def queued(agent):
    agent._gateway.rpc.return_value = {'admission_id': 'ours'}
    task = asyncio.create_task(agent.prompt([TextContentBlock(type='text', text='hello')], 's'))
    await asyncio.sleep(0)
    assert not task.done()
    agent._gateway.rpc.reset_mock()
    return task


async def settle(agent, task, outcome):
    await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'ours',
                          'payload': {'outcome': outcome, 'text': ''}})
    return await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_cancel_targets_queued_acp_admission_not_the_running_generation():
    agent = agent_with_queued_prompt()
    task = await queued(agent)
    agent._gateway.rpc.return_value = {'admission_id': 'ours', 'status': 'terminal', 'outcome': 'cancelled'}
    await agent.cancel('s')
    methods = [call.args[0] for call in agent._gateway.rpc.await_args_list]
    assert methods == ['prompt.cancel']
    assert agent._gateway.rpc.await_args.kwargs == {'session_id': 's', 'admission_id': 'ours'}
    assert (await settle(agent, task, 'cancelled')).stop_reason == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_interrupts_only_when_our_admission_is_the_one_running():
    agent = agent_with_queued_prompt()
    task = await queued(agent)

    async def rpc(method, **params):
        if method == 'prompt.cancel':
            raise GatewayClientError('stale_generation')
        if method == 'prompt.receipt':
            return {'admission_id': 'ours', 'status': 'started', 'execution_generation': 8}
        return {}
    agent._gateway.rpc.side_effect = rpc
    await agent.cancel('s')
    calls = {call.args[0]: call.kwargs for call in agent._gateway.rpc.await_args_list}
    assert calls['session.interrupt'] == {'session_id': 's', 'execution_generation': 8}
    assert (await settle(agent, task, 'cancelled')).stop_reason == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_without_an_acp_prompt_leaves_other_surfaces_turn_alone():
    agent = agent_with_queued_prompt()
    await agent.cancel('s')
    agent._gateway.rpc.assert_not_awaited()
