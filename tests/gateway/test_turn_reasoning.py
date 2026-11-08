"""Value contract at the gateway reasoning boundary, using real plugin dispatch and local policies."""
from types import SimpleNamespace
import threading

import pytest

from gateway.run_turn_reasoning import select_reasoning
from gateway.session_policy import build_policy, pin_reasoning_effort
from hermes_constants import parse_reasoning_effort


def _turn(tmp_path, raw='none', pinned=True):
    config = {'agent': {'reasoning_effort': raw}}
    if pinned:
        pin_reasoning_effort(config, raw, session=True)
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'gpt-6.1-sol'}, config)
    ctx = SimpleNamespace(session_id='s', session_key='acp:s', inbound_message_id='admission',
                          title_user_message='hello', message='wrapped hello', history=[])
    return SimpleNamespace(_ctx=ctx, _runner=SimpleNamespace(_peek_session_state=lambda k: None)), policy


@pytest.fixture
def manager(monkeypatch):
    from hermes_cli import plugins
    pm = plugins.PluginManager()
    pm._discovered = True
    monkeypatch.setattr(plugins, '_plugin_manager', pm)
    return pm


def _select(turn, policy, baseline):
    return select_reasoning(turn, policy, baseline, 'gpt-6.1-sol',
                            {'provider': 'openai-codex', 'api_mode': 'codex_responses'}, 'acp')


@pytest.mark.parametrize('result', [None, False, {}, {'effort': 'high'}, 'auto', 'HIGH', ' high ', 5])
def test_invalid_or_absent_value_is_native_baseline(manager, tmp_path, result):
    manager._hooks['select_turn_reasoning'] = [lambda **p: result]
    turn, policy = _turn(tmp_path)
    baseline = {'enabled': False}
    assert _select(turn, policy, baseline) == (baseline, None)


@pytest.mark.parametrize('raw,pin', [('none', True), ('none', False), (False, False), ('disabled', False), ('high', True)])
def test_callback_receives_original_provenance_without_host_or_secret_handles(manager, tmp_path, raw, pin):
    seen = []
    def callback(**p):
        seen.append(p)
    manager._hooks['select_turn_reasoning'] = [callback]
    turn, policy = _turn(tmp_path, raw, pin)
    _select(turn, policy, parse_reasoning_effort(raw))
    p = seen[0]
    assert p['reasoning']['raw_effort'] == raw
    assert p['reasoning']['pinned'] is pin
    assert p['reasoning']['source'] == ('session' if pin else 'default')
    assert p['user_message'] == 'hello' and p['platform'] == 'acp'
    assert not {'agent', 'gateway', 'runner', 'api_key', 'base_url', 'config'} & p.keys()


def test_valid_value_is_clamped_and_payload_mutation_cannot_change_baseline(manager, tmp_path):
    def callback(**p):
        p['reasoning']['config']['enabled'] = True
        return 'none'
    manager._hooks['select_turn_reasoning'] = [callback]
    turn, policy = _turn(tmp_path)
    baseline = {'enabled': False}
    assert _select(turn, policy, baseline) == ({'enabled': True, 'effort': 'low'}, 'none')
    assert baseline == {'enabled': False}


def test_request_choice_wins_even_when_callback_returns_an_effort(manager, tmp_path):
    from gateway.session_cli_turn import cli_turn_scope
    turn, policy = _turn(tmp_path)
    manager._hooks['select_turn_reasoning'] = [lambda **p: 'max']
    with cli_turn_scope('admission', {'model_override': {'model': 'gpt-6.1-sol', 'reasoning': 'low'}}):
        assert _select(turn, policy, {'enabled': True, 'effort': 'low'}) == ({'enabled': True, 'effort': 'low'}, None)


@pytest.mark.parametrize('raw', ['low', 'medium', 'high', 'xhigh', 'max', False, 'disabled'])
def test_explicit_manual_pin_wins(manager, tmp_path, raw):
    manager._hooks['select_turn_reasoning'] = [lambda **p: 'max']
    turn, policy = _turn(tmp_path, raw)
    baseline = parse_reasoning_effort(raw)
    assert _select(turn, policy, baseline) == (baseline, None)


def test_plugin_absent_and_live_session_override_keep_baseline(manager, tmp_path):
    turn, policy = _turn(tmp_path)
    baseline = {'enabled': True, 'effort': 'medium'}
    assert _select(turn, policy, baseline) == (baseline, None)
    turn._runner._peek_session_state = lambda k: SimpleNamespace(
        conversation=SimpleNamespace(reasoning_override=baseline))
    manager._hooks['select_turn_reasoning'] = [lambda **p: 'high']
    assert _select(turn, policy, baseline) == (baseline, None)


@pytest.mark.parametrize('failure', ['exception', 'timeout'])
def test_failed_callbacks_keep_native_baseline(manager, tmp_path, monkeypatch, failure):
    release = threading.Event()
    def callback(**p):
        if failure == 'exception':
            raise ValueError('test failure')
        assert release.wait(5)
        return 'high'
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, '_resolve_hook_callback_timeout', lambda: .01)
    manager._hooks['select_turn_reasoning'] = [callback]
    turn, policy = _turn(tmp_path)
    try:
        assert _select(turn, policy, {'enabled': False}) == ({'enabled': False}, None)
    finally:
        release.set()


def test_session_reporting_keeps_the_pinned_none_during_selection():
    from agent.voice_turn_route import session_runtime_view
    from gateway.session_config import _reasoning
    agent = SimpleNamespace(reasoning_config={'enabled': True, 'effort': 'high'}, _voice_route_state=None,
                            _turn_reasoning_own={'reasoning_config': {'enabled': False}})
    assert session_runtime_view(agent).reasoning_config == {'enabled': False}
    assert _reasoning(agent, None, {})['value'] == 'none'
