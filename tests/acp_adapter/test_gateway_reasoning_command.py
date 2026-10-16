"""``/reasoning`` on the gateway path: an ACP editor sets this conversation's level, against a fake owner."""
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_session_meta import COMMANDS, available_commands_update, help_text
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_mutations import slash_mutation


def _agent(rpc):
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._snapshots['s'] = {'execution_generation': 7}
    return agent


def _replies(agent):
    return [call.kwargs['update'].content.text.strip() for call in agent._conn.session_update.await_args_list
            if call.kwargs['update'].session_update == 'agent_message_chunk']


def test_slash_mutation_names_the_session_level():
    assert slash_mutation('/reasoning', 'low') == ('reasoning', {'effort': 'low'})
    assert slash_mutation('/reasoning', '  XHigh ') == ('reasoning', {'effort': 'xhigh'})
    assert slash_mutation('/reasoning', 'none') == ('reasoning', {'effort': 'none'})


@pytest.mark.parametrize('arg', ['', 'hgih', 'low high', '--global low', 'low --global', '--global'])
def test_slash_mutation_refuses_unknown_levels_and_options(arg):
    with pytest.raises(GatewayClientError, match='unsupported_reasoning_options'):
        slash_mutation('/reasoning', arg)


def test_reasoning_is_advertised_to_editors():
    assert 'reasoning' in COMMANDS
    assert 'reasoning' in {c.name for c in available_commands_update().available_commands}
    assert '/reasoning' in help_text()


@pytest.mark.asyncio
async def test_reasoning_sets_the_level_through_the_owners_session_mutation():
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        if method == 'session.resume':
            return {'execution_generation': 7, 'revision': 3, 'messages': []}
        assert method == 'session.mutate', method
        return {'status': 'applied'}

    agent = _agent(rpc)
    response = await agent.prompt([acp.text_block('/reasoning low')], 's')
    assert response.stop_reason == 'end_turn'
    mutate = [params for method, params in calls if method == 'session.mutate']
    assert len(mutate) == 1
    assert mutate[0]['operation'] == 'reasoning' and mutate[0]['payload'] == {'effort': 'low'}
    assert mutate[0]['expected_generation'] == 7
    assert 'prompt.submit' not in [method for method, _ in calls]  # a command, never a model turn
    assert _replies(agent) == ['Reasoning effort for this session: low']


@pytest.mark.asyncio
async def test_bare_reasoning_reports_the_current_level_and_changes_nothing():
    async def rpc(method, **params):
        assert method == 'session.info', method
        return {'reasoning_effort': 'high'}

    agent = _agent(rpc)
    await agent.prompt([acp.text_block('/reasoning')], 's')
    assert _replies(agent) == ['Reasoning effort: high']


@pytest.mark.asyncio
@pytest.mark.parametrize('arg', ['hgih', '--global low'])
async def test_a_refused_level_reaches_no_gateway_verb(arg):
    agent = _agent(AsyncMock())
    with pytest.raises(GatewayClientError, match='unsupported_reasoning_options'):
        await agent.prompt([acp.text_block(f'/reasoning {arg}')], 's')
    agent._gateway.rpc.assert_not_awaited()
