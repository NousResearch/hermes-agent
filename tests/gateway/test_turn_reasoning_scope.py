"""One-turn choices must not become cached fallback session edits."""
import copy
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from agent.route_binding import bind_route_entry
from gateway.session_api_turn import api_execution
from gateway.session_cli_turn import cli_turn_scope
from hermes_constants import resolve_reasoning_config
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve
from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _run_turn


@pytest.mark.parametrize('surface', ['cli', 'api'])
@pytest.mark.parametrize('restore_during_turn', [False, True])
def test_two_complete_turns_on_cached_fallback(surface, restore_during_turn):
    config = {'agent': {'reasoning_effort': 'medium', 'reasoning_overrides': {
        'primary': 'high', 'fallback': 'low'}}}
    primary = resolve_reasoning_config(config, 'primary')
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    runner = _gateway_runner()
    runner._resolve_session_reasoning_config.return_value = primary
    seen = []

    def run(turn, active_agent, *args):
        if restore_during_turn and len(seen) == 0:
            active_agent._rate_limited_until = 0
            with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()), \
                 patch('agent.agent_runtime_helpers._primary_quota_reopened_early', return_value=False):
                assert active_agent._restore_primary_runtime()
        seen.append(copy.deepcopy(active_agent.reasoning_config))
        return {'final_response': 'ok', 'messages': []}

    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config), \
         patch('gateway.run_turn_runner.TurnRunner._resolve_turn_agent', return_value=(agent, True)), \
         patch('gateway.run_turn_runner.TurnRunner._run_conversation_with_approval', run), \
         patch('gateway.session_cli_turn.one_turn_route', side_effect=lambda t, p, m, r: (m, r)), \
         patch('gateway.session_api_turn.prepare_api_runtime', side_effect=lambda m, r: (m, r)):
        assert agent._try_activate_fallback()
        agent._rate_limited_until = 0 if restore_during_turn else float('inf')
        scope = cli_turn_scope('adm-1', {'model_override': {'reasoning': 'xhigh'}}) if surface == 'cli' else nullcontext()
        token = api_execution.set({
            'history': None, 'content': 'clean text',
            'settings': {'model_options': {'reasoning_effort': 'xhigh'}},
        }) if surface == 'api' else None
        try:
            with scope:
                result, _ = _run_turn(runner, 'adm-1')
                assert result['final_response'] == 'ok'
        finally:
            if token is not None:
                api_execution.reset(token)
        result, _ = _run_turn(runner, 'adm-2')
        assert result['final_response'] == 'ok'
        assert seen[0]['effort'] == 'xhigh'
        assert seen[1]['effort'] == ('high' if restore_during_turn else 'low')
        agent._rate_limited_until = 0
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            if agent._fallback_activated:
                assert agent._restore_primary_runtime()
        assert agent.reasoning_config == primary


@pytest.mark.parametrize('edit', ['assign', 'same', 'in_place', 'none'])
def test_real_edit_during_override_survives_fallback_and_primary_restore(edit):
    from agent.route_binding import turn_reasoning_override
    config = {'agent': {'reasoning_effort': 'low'}}
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config(config, 'primary')
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        with turn_reasoning_override(agent, {'enabled': True, 'effort': 'xhigh'}):
            if edit == 'assign':
                agent.reasoning_config = {'enabled': True, 'effort': 'high'}
            elif edit == 'same':
                agent.reasoning_config = agent.reasoning_config
            elif edit == 'none':
                agent.reasoning_config = None
            else:
                agent.reasoning_config['effort'] = 'medium'
            expected = copy.deepcopy(agent.reasoning_config)
            assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'fallback-2'}, 'openrouter', 'fallback-2')
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
        assert agent.reasoning_config == expected
