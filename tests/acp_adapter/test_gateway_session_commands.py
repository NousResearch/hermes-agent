"""``/queue``, ``/steer`` and the edit-policy mode on the gateway path, against a fake owner."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

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


def _receipt(status, generation=None):
    return {'admission_id': 'mine', 'status': status, 'execution_generation': generation}


@pytest.mark.asyncio
async def test_steer_reaches_this_editors_running_turn():
    async def rpc(method, **params):
        if method == 'prompt.receipt':
            assert params == {'session_id': 's', 'admission_id': 'mine'}, params
            return _receipt('started', 9)
        assert method == 'session.steer', method
        # Our admission's generation, not the session's latest event (7).
        assert params == {'session_id': 's', 'text': 'use pytest', 'execution_generation': 9}
        return {'status': 'queued', 'text': 'use pytest', 'execution_generation': 9, 'authority_epoch': 1}

    agent = _agent(rpc)
    agent._admissions['s'] = 'mine'
    response = await agent.prompt([acp.text_block('/steer use pytest')], 's')
    assert response.stop_reason == 'end_turn'
    assert 'Steer queued for the active turn: use pytest' in \
        agent._conn.session_update.await_args.kwargs['update'].content.text


@pytest.mark.asyncio
async def test_steer_while_this_editors_prompt_waits_behind_another_clients_turn_is_queued():
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'prompt.receipt':
            return _receipt('queued')
        assert method == 'prompt.submit' and params['text'] == 'use pytest', (method, params)
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'a',
                              'payload': {'outcome': 'completed', 'text': 'done'}})
        return {'admission_id': 'a'}

    agent = _agent(rpc)  # generation 7 is the other client's running turn
    agent._admissions['s'] = 'mine'
    response = await asyncio.wait_for(agent.prompt([acp.text_block('/steer use pytest')], 's'), 5)
    assert response.stop_reason == 'end_turn' and calls == ['prompt.receipt', 'prompt.submit']


@pytest.mark.asyncio
async def test_a_refused_steer_is_reported_and_never_resent_as_a_prompt():
    from hermes_cli.gateway_client import GatewayRPCError
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'prompt.receipt':
            return _receipt('started', 7)
        raise GatewayRPCError('unsupported_control')

    agent = _agent(rpc)
    agent._admissions['s'] = 'mine'
    with pytest.raises(GatewayRPCError, match='unsupported_control'):
        await agent.prompt([acp.text_block('/steer use pytest')], 's')
    assert calls == ['prompt.receipt', 'session.steer']


@pytest.mark.asyncio
async def test_a_steer_for_a_turn_that_just_ended_becomes_the_next_prompt():
    from hermes_cli.gateway_client import GatewayRPCError
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'prompt.receipt':
            return _receipt('started', 7)
        if method == 'session.steer':
            raise GatewayRPCError('stale_generation')
        assert method == 'prompt.submit' and params['text'] == 'use pytest', (method, params)
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'a',
                              'payload': {'outcome': 'completed', 'text': 'done'}})
        return {'admission_id': 'a'}

    agent = _agent(rpc)
    agent._admissions['s'] = 'mine'
    response = await asyncio.wait_for(agent.prompt([acp.text_block('/steer use pytest')], 's'), 5)
    assert response.stop_reason == 'end_turn' and calls == ['prompt.receipt', 'session.steer', 'prompt.submit']


@pytest.mark.asyncio
async def test_only_the_owners_edit_policy_can_be_selected():
    agent = _agent(AsyncMock())
    await agent.set_session_mode('default', 's')
    await agent.set_config_option('edit_approval_policy', 's', 'ask')
    with pytest.raises(GatewayClientError, match='acp_edit_policy_mutation_unavailable'):
        await agent.set_session_mode('accept_edits', 's')
    agent._gateway.rpc.assert_not_awaited()


def _context_shown(agent):
    return [(call.kwargs['update'].size, call.kwargs['update'].used)
            for call in agent._conn.session_update.await_args_list
            if call.kwargs['update'].session_update == 'usage_update']


def _boundary_agent(messages):
    """/reset and /compress commit, then evict the owner's agent: ``session.stats`` has no window."""
    async def rpc(method, **params):
        if method == 'session.resume':
            return {'execution_generation': 8, 'messages': messages}
        assert method == 'session.stats', method
        return {'input_tokens': 0, 'context_tokens': 0, 'context_length': None}

    agent = _agent(rpc)
    agent._usage['s'] = {'context_length': 200000, 'context_tokens': 150000}
    agent._mutations = MagicMock(apply=AsyncMock(return_value={'status': 'applied', 'summary': {}}))
    return agent


@pytest.mark.asyncio
async def test_reset_empties_the_editors_context_indicator():
    agent = _boundary_agent([])
    await agent.prompt([acp.text_block('/reset')], 's')
    assert _context_shown(agent) == [(200000, 0)]


@pytest.mark.asyncio
async def test_compress_shows_the_compressed_transcripts_occupancy():
    from agent.model_metadata import estimate_request_tokens_rough
    kept = [{'role': 'user', 'content': 'summary of the earlier turns ' * 40}]
    agent = _boundary_agent(kept)
    await agent.prompt([acp.text_block('/compress')], 's')
    assert _context_shown(agent) == [(200000, estimate_request_tokens_rough(kept))]
