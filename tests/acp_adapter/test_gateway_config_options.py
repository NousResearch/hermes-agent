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


def test_the_effort_picker_offers_default_only_while_no_level_is_set():
    from acp_adapter.gateway_session_meta import config_options
    from acp_adapter.modes import DEFAULT_MODE

    def effort(level):
        option = config_options(None, {'reasoning_effort': level}, DEFAULT_MODE)[-1]
        return option.current_value, [choice.value for choice in option.options]
    # ``default`` (a reset) reads back as the profile's level: offering it would make the pick jump.
    current, values = effort('medium')
    assert current == 'medium' and 'default' not in values
    current, values = effort('')
    assert current == 'default' and values[0] == 'default'


def test_every_model_row_publishes_the_effort_levels_and_the_level_in_force():
    """A client that offers a picker per model reads ``ModelInfo._meta``: the session-level
    ``config_options`` option only describes the model in force, so other rows would go without."""
    from acp_adapter.gateway_session_meta import with_reasoning_meta

    state = SessionModelState(
        available_models=[ModelInfo(model_id='custom:fixture', name='fixture'),
                          ModelInfo(model_id='custom:other', name='other')],
        current_model_id='custom:fixture')
    rows = {row.model_id: row.field_meta for row in
            with_reasoning_meta(state, {'reasoning_effort': 'high'}).available_models}
    levels = [choice['value'] for choice in rows['custom:fixture']['reasoningEfforts']]
    assert levels[:3] == ['none', 'minimal', 'low'] and 'default' not in levels
    assert all('label' in choice for choice in rows['custom:fixture']['reasoningEfforts'])
    # One effort per session: running on any of these models uses it, so every row reports it.
    assert rows['custom:fixture']['reasoningEffort'] == 'high'
    assert rows['custom:other']['reasoningEffort'] == 'high'
    assert rows['custom:other']['reasoningEfforts'] == rows['custom:fixture']['reasoningEfforts']


def test_no_level_set_leaves_the_picker_unselected_rather_than_inventing_one():
    from acp_adapter.gateway_session_meta import with_reasoning_meta

    state = SessionModelState(available_models=[ModelInfo(model_id='custom:fixture', name='fixture')],
                              current_model_id='custom:fixture')
    # Unset, a reset level, and a level this owner does not accept all leave the row without a value.
    for info in ({}, {'reasoning_effort': 'default'}, {'reasoning_effort': 'turbo'}):
        assert 'reasoningEffort' not in with_reasoning_meta(state, info).available_models[0].field_meta, info
