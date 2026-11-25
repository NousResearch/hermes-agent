"""Terminal results cannot be replaced by a successor's unknown queue state."""
import asyncio
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
@pytest.mark.parametrize('before_ack', [False, True])
async def test_successor_unknown_does_not_overwrite_completed_editor_prompt(before_ack):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {}

    async def publish_buffered_events():
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'ours',
                              'payload': {'outcome': 'completed', 'text': 'done'}})
        # Both events may be buffered before the prompt coroutine wakes; only
        # the successor has become unknown, not the completed prompt.
        await agent._project({'session_id': 's', 'type': 'session.info',
                              'payload': {'pending': [{'admission_id': 'next', 'status': 'unknown'}]}})

    async def submit(method, **params):
        assert method == 'prompt.submit'
        if before_ack:
            await publish_buffered_events()
        return {'admission_id': 'ours'}

    agent._gateway.rpc.side_effect = submit
    submitted = asyncio.create_task(agent.prompt([acp.text_block('our work')], 's'))
    if not before_ack:
        await asyncio.sleep(0)
        assert agent._admissions['s'] == 'ours'
        await publish_buffered_events()
    response = await asyncio.wait_for(submitted, 5)
    assert response.stop_reason == 'end_turn'
    agent._gateway.rpc.assert_awaited_once()
