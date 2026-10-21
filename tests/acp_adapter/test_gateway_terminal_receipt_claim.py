"""A retried submit's terminal receipt and the live message.complete deliver one reply between them."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent


def _agent(final, *, receipt_gate=None):
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()

    async def rpc(method, **kwargs):
        if method == 'prompt.submit':
            return {'admission_id': 'a1', 'status': 'terminal', 'outcome': 'completed'}
        assert method == 'prompt.receipt'
        if receipt_gate is not None:
            await receipt_gate()
        return {'result': {'final_response': final}}

    agent._gateway.rpc.side_effect = rpc
    agent._conn = AsyncMock()
    return agent


def _complete(text):
    return {'session_id': 's', 'type': 'message.complete', 'admission_id': 'a1',
            'payload': {'outcome': 'completed', 'text': text}}


def _shown(agent):
    return [call.kwargs['update'].content.text for call in agent._conn.session_update.await_args_list]


async def _prompt(agent):
    return await asyncio.wait_for(agent.prompt([TextContentBlock(type='text', text='hello')], 's'), 2)


@pytest.mark.asyncio
async def test_live_completion_during_the_receipt_editor_update_is_not_shown_twice():
    agent = _agent('Final answer')
    updates = []

    async def session_update(**kwargs):
        updates.append(kwargs['update'].content.text)
        if len(updates) == 1:  # the pump projects the queued completion while the receipt is suspended
            await agent._project(_complete('Final answer'))

    agent._conn.session_update.side_effect = session_update
    assert (await _prompt(agent)).stop_reason == 'end_turn'
    assert updates == ['Final answer']
    # The late completion of that admission paints nothing either.
    await agent._project(_complete('Final answer'))
    assert updates == ['Final answer']


@pytest.mark.asyncio
async def test_receipt_shows_only_what_the_live_stream_did_not():
    agent = _agent('Hello world')
    await agent._project({'session_id': 's', 'type': 'message.delta', 'admission_id': 'a1',
                          'payload': {'text': 'Hello '}})
    assert (await _prompt(agent)).stop_reason == 'end_turn'
    assert _shown(agent) == ['Hello ', 'world']


@pytest.mark.asyncio
async def test_completion_that_lands_before_the_receipt_is_the_one_shown():
    async def gate():
        await agent._project(_complete('Final answer'))

    agent = _agent('Final answer', receipt_gate=gate)
    assert (await _prompt(agent)).stop_reason == 'end_turn'
    assert _shown(agent) == ['Final answer']
