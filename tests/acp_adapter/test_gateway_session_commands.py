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
async def test_a_steer_that_ended_the_voice_chat_is_reported_and_never_resent_as_a_prompt():
    """TUI-26: ``/steer stop`` in a LOCAL voice chat ends the chat on the owner (``voice_stopped``);
    sent on as a prompt it would admit ``stop`` as a turn."""
    from agent.i18n import t
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'prompt.receipt':
            return _receipt('started', 9)
        return {'voice_stopped': True}

    agent = _agent(rpc)
    agent._admissions['s'] = 'mine'
    response = await agent.prompt([acp.text_block('/steer stop')], 's')
    assert response.stop_reason == 'end_turn'
    assert calls == ['prompt.receipt', 'session.steer']
    assert t('cli.voice.stop_phrase_typed') in agent._conn.session_update.await_args.kwargs['update'].content.text


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
async def test_modes_and_the_config_option_set_the_owners_edit_policy():
    """ACP-7: main's modes are the session's ``edit_policy`` on the owner, one fenced mutation each."""
    sent = []

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'revision': 3, 'execution_generation': 7}
        if method == 'session.info':
            return {}
        assert method == 'session.mutate', method
        sent.append((params['operation'], params['payload'], params['expected_generation']))
        return {'session_id': 's', 'operation': 'edit_policy', 'revision': 4}

    agent = _agent(rpc)
    await agent.set_session_mode('accept_edits', 's')
    await agent.set_config_option('edit_approval_policy', 's', 'session')
    await agent.set_session_mode('default', 's')
    assert sent == [('edit_policy', {'policy': 'workspace_session'}, 7), ('edit_policy', {'policy': 'session'}, 7),
                    ('edit_policy', {'policy': 'ask'}, 7)]
    sent.clear()
    with pytest.raises(GatewayClientError, match='invalid_params'):
        await agent.set_session_mode('yolo', 's')
    # Another config id is accepted and means nothing here, as on main.
    await agent.set_config_option('theme', 's', 'dark')
    assert sent == []



@pytest.mark.asyncio
async def test_the_editors_own_pick_projected_before_the_mutation_answers_is_harmless():
    """The owner publishes ``session.info`` inside the commit, before ``session.mutate`` answers, so the
    pump may project the editor's own pick first: at most one ``current_mode_update`` of the picked
    mode, and the pump keeps running."""
    async def rpc(method, **params):
        if method == 'session.resume':
            return {'revision': 3, 'execution_generation': 7}
        await agent._project({'session_id': 's', 'type': 'session.info',
                              'payload': {'edit_approval_policy': params['payload']['policy']}})
        return {'session_id': 's', 'operation': 'edit_policy', 'revision': 4}

    agent = _agent(rpc)
    agent._modes['s'] = 'default'
    await agent.set_session_mode('dont_ask', 's')
    shown = [call.kwargs['update'] for call in agent._conn.session_update.await_args_list]
    assert [(u.session_update, u.current_mode_id) for u in shown] == [('current_mode_update', 'dont_ask')], shown
    assert agent._modes['s'] == 'dont_ask' and agent._failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize('pick', [lambda a: a.set_session_mode('dont_ask', 's'),
                                  lambda a: a.set_config_option('edit_approval_policy', 's', 'session')])
async def test_a_mode_pick_during_a_turn_is_refused_with_the_reason(pick):
    """The owner changes the edit policy on an idle session only; the editor is told why, by name."""
    from hermes_cli.gateway_client import GatewayRPCError

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'revision': 3, 'execution_generation': 7}
        raise GatewayRPCError('session_busy')

    agent = _agent(rpc)
    agent._modes['s'] = 'default'
    with pytest.raises(acp.RequestError) as refused:
        await pick(agent)
    assert 'idle' in str(refused.value.data or refused.value), refused.value
    shown = [call.kwargs['update'] for call in agent._conn.session_update.await_args_list]
    assert any('idle' in getattr(getattr(u, 'content', None), 'text', '') for u in shown), shown
    assert agent._modes['s'] == 'default'


@pytest.mark.asyncio
async def test_a_mode_pick_behind_a_lost_turn_says_to_resolve_it_not_to_wait():
    """unknown_execution: no turn is running, so "after this turn" would never come true."""
    from hermes_cli.gateway_client import GatewayRPCError

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'revision': 3, 'execution_generation': 7}
        raise GatewayRPCError('unknown_execution')

    agent = _agent(rpc)
    agent._snapshots['s']['pending'] = [{'admission_id': 'lost-1', 'status': 'unknown'}]
    with pytest.raises(acp.RequestError) as refused:
        await agent.set_session_mode('dont_ask', 's')
    text = str(refused.value.data or refused.value)
    assert 'after this turn' not in text and '/discard lost-1' in text, text


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
