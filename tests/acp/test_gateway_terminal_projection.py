"""Terminal ACP responses describe the exact admission, not the session's latest turn."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


def agent_with_admission():
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()
    agent._gateway.rpc.return_value = {'admission_id': 'current'}
    agent._conn = AsyncMock()
    return agent


async def complete(agent, admission_id, outcome, text):
    await agent._project({'session_id': 's', 'type': 'message.complete',
                          'admission_id': admission_id,
                          'payload': {'outcome': outcome, 'text': text}})


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
async def test_failed_admission_raises_without_poisoning_the_next_success():
    agent = agent_with_admission()
    await complete(agent, 'current', 'failed', 'The admitted turn failed.')
    with pytest.raises(GatewayClientError, match='admitted_turn_failed'):
        await agent.prompt([TextContentBlock(type='text', text='hello')], 's')
    assert 'current' not in agent._terminals
    await complete(agent, 'current', 'completed', 'Operation interrupted: quoted model text')
    response = await agent.prompt([TextContentBlock(type='text', text='retry')], 's')
    assert response.stop_reason == 'end_turn'
    assert agent._conn.session_update.await_args.kwargs['update'].content.text == (
        'Operation interrupted: quoted model text')
