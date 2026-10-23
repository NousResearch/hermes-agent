"""Zed renders its model and effort pickers only from ``configOptions`` (#143547 ACP-24)."""
import pytest
from acp.schema import ModelInfo, SessionModelState

from acp_adapter.server import HermesACPAgent


@pytest.mark.asyncio
async def test_effort_pick_applies_through_the_reasoning_mutation_and_returns_the_pickers(monkeypatch):
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: SessionModelState(
        available_models=[ModelInfo(model_id='openrouter:fixture', name='fixture')], current_model_id='openrouter:fixture'))
    info = {'model': 'fixture', 'provider': 'openrouter', 'reasoning_effort': 'medium'}
    mutations = []

    class RPC:
        async def rpc(self, method, **params):
            if method == 'session.mutate':
                mutations.append((params['operation'], params['payload']))
                info['reasoning_effort'] = params['payload']['effort']
                return {'status': 'applied'}
            if method == 'session.info':
                return dict(info)
            return {'revision': 1, 'execution_generation': 1, 'info': dict(info)}

    agent = HermesACPAgent()
    agent._gateway = RPC()
    agent._snapshots['s'] = {'revision': 1, 'execution_generation': 1}
    reply = await agent.set_config_option(config_id='reasoning_effort', session_id='s', value='high')
    assert mutations == [('reasoning', {'effort': 'high'})]
    pickers = {option.category: option.current_value for option in reply.config_options}
    assert pickers == {'mode': 'ask', 'model': 'openrouter:fixture', 'thought_level': 'high'}
