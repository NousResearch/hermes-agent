"""Editor prompts fail recoverably when their FIFO becomes unknown."""
import asyncio
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


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
