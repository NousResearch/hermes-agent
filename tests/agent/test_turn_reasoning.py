"""A turn selection uses native serialization, continuation and rejection/fallback recovery."""
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

from agent.turn_reasoning import _active_config, _reasoning_scope
from tests.agent.test_length_continuation_thinking_exhaustion import (
    _full_response, _run, _thinking_only_length_response,
)

pytest_plugins = ['tests.agent.test_length_continuation_thinking_exhaustion']


def _effort(kwargs):
    if 'reasoning_effort' in kwargs:
        effort = kwargs['reasoning_effort']
        return {'enabled': effort != 'none', 'effort': effort}
    return (kwargs.get('extra_body') or {}).get('reasoning') or kwargs.get('reasoning')


def test_selection_survives_tool_free_retry_and_intentional_continuation_off(loop_agent):
    loop_agent.provider = 'custom'
    loop_agent.model = 'wire-reasoning'
    loop_agent.reasoning_config = {'enabled': True, 'effort': 'high'}
    loop_agent.client.chat.completions.create.side_effect = [_thinking_only_length_response(), _full_response('done')]
    with _reasoning_scope(loop_agent, 'high', {'enabled': False}):
        result = _run(loop_agent, 'write a long report')
        assert result['completed'] is True
        calls = loop_agent.client.chat.completions.create.call_args_list
        assert _effort(calls[0].kwargs) is not None, calls[0].kwargs
        assert _effort(calls[0].kwargs)['effort'] == 'high'
        assert _effort(calls[1].kwargs)['enabled'] is False
        assert loop_agent.reasoning_config == {'enabled': True, 'effort': 'high'}
    assert loop_agent.reasoning_config == {'enabled': False}


def test_selected_effort_rejection_uses_native_level_recovery(loop_agent):
    from tests.agent.test_reasoning_disable_recovery import _FakeApiError
    loop_agent.provider = 'custom'
    loop_agent.model = 'wire-reasoning'
    loop_agent.reasoning_config = {'enabled': True, 'effort': 'max'}
    loop_agent.client.chat.completions.create.side_effect = [
        _FakeApiError(400, "reasoning.effort 'max' is unsupported"), _full_response('recovered')]
    with _reasoning_scope(loop_agent, 'max', {'enabled': False}):
        result = _run(loop_agent, 'hello')
        assert result['completed'] is True
        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 2
        assert _effort(calls[0].kwargs) is not None, calls[0].kwargs
        assert _effort(calls[0].kwargs)['effort'] == 'max'
        assert _effort(calls[1].kwargs) is None
        assert loop_agent._reasoning_effort_rejected is True
        assert not getattr(loop_agent, '_reasoning_disable_rejected', False)


def test_native_fallback_and_snapshot_restore_hold_only_this_turn():
    from tests.agent.test_provider_fallback import _make_agent, _mock_client
    agent = _make_agent(fallback_model=[{'provider': 'openai-codex', 'model': 'gpt-5.5'}])
    baseline = {'enabled': False}
    agent.reasoning_config = {'enabled': True, 'effort': 'max'}
    agent._primary_runtime['reasoning_config'] = dict(agent.reasoning_config)
    with (patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_client(), 'gpt-5.5')),
          _reasoning_scope(agent, 'max', baseline)):
        assert agent._try_activate_fallback() is True
        assert agent.model == 'gpt-5.5'
        assert agent.reasoning_config == {'enabled': True, 'effort': 'xhigh'}
        with patch('agent.agent_runtime_helpers._rebuild_primary_client'):
            agent._rate_limited_until = 0
            assert agent._restore_primary_runtime() is True
        assert agent.reasoning_config['effort'] == 'max'
    assert agent.reasoning_config == baseline
    assert agent._primary_runtime['reasoning_config'] == baseline
    assert _active_config(agent) is None


def test_real_loop_rate_limit_fallback_sends_selected_effort(loop_agent):
    from tests.agent.test_provider_fallback import _mock_client
    from tests.agent.test_reasoning_disable_recovery import _FakeApiError
    loop_agent.provider = 'custom'
    loop_agent.model = 'wire-primary'
    loop_agent.reasoning_config = {'enabled': True, 'effort': 'high'}
    loop_agent._fallback_chain = [{'provider': 'custom', 'model': 'wire-fallback',
                                   'api_mode': 'chat_completions'}]
    primary = loop_agent.client
    primary.chat.completions.create.side_effect = _FakeApiError(429, 'Rate limit exceeded')
    fallback = _mock_client()
    fallback.chat.completions.create.return_value = _full_response('fallback answer')
    with (patch('agent.auxiliary_client.resolve_provider_client', return_value=(fallback, 'wire-fallback')),
          _reasoning_scope(loop_agent, 'high', {'enabled': False})):
        result = _run(loop_agent, 'hello')
        assert result['completed'] is True
        assert _effort(primary.chat.completions.create.call_args.kwargs)['effort'] == 'high'
        assert fallback.chat.completions.create.call_count == 1
        assert _effort(fallback.chat.completions.create.call_args.kwargs)['effort'] == 'high'
        assert loop_agent.model == 'wire-fallback'
    assert loop_agent.reasoning_config == {'enabled': False}


def test_concurrent_agents_and_nested_child_have_no_selection_leak():
    from types import SimpleNamespace
    barrier = threading.Barrier(2)
    def run(effort):
        agent = SimpleNamespace(provider='openai-codex', model='gpt-6.1-sol', api_mode='codex_responses')
        child = SimpleNamespace(provider='openai-codex', model='gpt-6.1-sol', api_mode='codex_responses')
        with _reasoning_scope(agent, effort, None):
            barrier.wait(timeout=5)
            assert _active_config(agent)['effort'] == effort
            assert _active_config(child) is None
        assert _active_config(agent) is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ['low', 'high']))
