"""CLI launch pins fail instead of silently resolving a startup fallback."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.session_worker_construct import route_runtime
from gateway.session_policy import build_policy
from hermes_cli.auth import AuthError


@pytest.mark.parametrize('source,launch_request,pinned', [
    ('oneshot', {'model': 'example'}, True),
    ('oneshot', {'provider': 'primary'}, True),
    ('cli', {'model': 'example'}, True),
    ('oneshot', {}, False),
    ('oneshot', {'provider': 'auto'}, False),
    ('gui', {'model': 'example'}, False),
])
def test_launch_pin_preserves_or_walks_fallback(monkeypatch, tmp_path, source, launch_request, pinned):
    calls = []
    def primary(**kwargs):
        calls.append('primary')
        raise AuthError('not authenticated')
    def fallback(config, **kwargs):
        calls.append('fallback')
        return {'provider': 'fallback', 'request_overrides': {'extra_body': {'team': 'test'}}}, {'model': 'alternate'}
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', primary)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_with_fallback', fallback)
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime', lambda **kwargs: None)
    frozen = {'model': {'provider': 'primary', 'default': 'example'}}
    policy = build_policy({'source': source, 'cwd': str(tmp_path), **launch_request}, frozen)
    frozen, _ = policy.effective()
    if pinned:
        with pytest.raises(AuthError):
            route_runtime(policy, frozen, 'example', None)
        assert calls == ['primary']
    else:
        runtime, entry = route_runtime(policy, frozen, 'example', None)
        assert calls == ['fallback']
        assert entry['model'] == 'alternate'
        assert runtime['request_overrides'] == {'extra_body': {'team': 'test'}}


def test_oneshot_route_keeps_custom_request_overrides(tmp_path, monkeypatch):
    """The gateway-backed -z route forwards the resolver's custom extra body."""
    from gateway.run import _runtime_agent_kwargs
    overrides = {'extra_body': {'user': 'proxy-user'}}
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime',
                        lambda **kwargs: {'provider': 'custom', 'request_overrides': overrides})
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'example'},
                          {'model': {'provider': 'custom', 'default': 'example'}})
    frozen, model = policy.effective()
    runtime, entry = route_runtime(policy, frozen, model, None)
    assert entry is None
    assert _runtime_agent_kwargs(runtime)['request_overrides'] == overrides


def test_real_oneshot_resolution_pinned_vs_default(tmp_path, monkeypatch):
    """A real missing-key route fails when pinned; its unpinned sibling resolves the fallback."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    config = {'model': {'provider': 'google', 'default': 'gemini-test'},
              'fallback_providers': [{'provider': 'openrouter', 'model': 'example/fallback'}]}
    from hermes_cli import runtime_provider
    original = runtime_provider.resolve_runtime_provider
    calls = []
    def tracked(**kwargs):
        calls.append(kwargs.get('requested'))
        return original(**kwargs)
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', tracked)
    for pin in (False, True):
        calls.clear()
        params = {'source': 'oneshot', 'cwd': str(tmp_path)}
        if pin:
            params.update(model='gemini-test', provider='google')
        policy = build_policy(params, config)
        frozen, model = policy.effective()
        if pin:
            with pytest.raises(AuthError):
                route_runtime(policy, frozen, model, None)
            assert calls == ['google']
        else:
            runtime, entry = route_runtime(policy, frozen, model, None)
            assert runtime['provider'] == 'openrouter'
            assert entry['model'] == 'example/fallback'


def test_one_turn_gateway_override_does_not_release_launch_pin(tmp_path, monkeypatch):
    from gateway.session_cli_turn import one_turn_runtime
    from gateway.session_worker_construct import cli_launch_route_pinned
    from hermes_cli.model_switch import ModelSwitchResult
    config = {'model': {'provider': 'openrouter', 'default': 'original'}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'original'}, config)
    frozen, model = policy.effective()
    before = json.dumps(frozen, sort_keys=True)
    monkeypatch.setattr('hermes_cli.model_switch.switch_model', lambda **kwargs: ModelSwitchResult(
        success=True, new_model='once', target_provider='openrouter', api_key='',
        base_url='https://openrouter.ai/api/v1', api_mode='chat_completions', request_overrides={}))
    monkeypatch.setattr('hermes_cli.config.get_compatible_custom_providers', lambda config: [])
    once, _ = one_turn_runtime({'model': 'once', 'provider': 'openrouter'}, model,
                               {'provider': 'openrouter', 'api_key': 'test-key',
                                'base_url': 'https://openrouter.ai/api/v1', 'api_mode': 'chat_completions'}, frozen)
    assert once == 'once'
    assert json.dumps(frozen, sort_keys=True) == before
    assert cli_launch_route_pinned(policy, frozen, model)


def test_followed_profile_failure_uses_current_route(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    original = {'model': {'provider': 'google', 'default': 'first'}}
    live = {'model': {'provider': 'openrouter', 'default': 'second'}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'first',
                           'follow_profile_config': True}, original)
    monkeypatch.setattr('gateway.session_policy.profile_config', lambda authority: live)
    result = _resolution_failure_result(RuntimeError('current route diagnostic'), policy,
                                        SimpleNamespace(platform=Platform.LOCAL), 'test', object())
    assert 'current route diagnostic' in result['final_response']


def test_worker_bootstrap_releases_followed_away_launch_route(tmp_path, monkeypatch):
    from gateway.session_managed_worker import _bootstrap
    from gateway.session_policy import restore_policy
    from gateway.session_worker_construct import cli_launch_route_pinned
    original = {'model': {'provider': 'google', 'default': 'first'}}
    live = {'model': {'provider': 'openrouter', 'default': 'second'}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'first',
                           'follow_profile_config': True}, original)
    monkeypatch.setattr('gateway.session_policy.profile_config', lambda authority: live)
    monkeypatch.setattr('gateway.session_worker_construct.construct_inputs', lambda *args: {})
    monkeypatch.setattr('gateway.session_policy.route_launch_key', lambda *args: None)
    monkeypatch.setattr('gateway.session_ingress.row_turn_author', lambda *args: None)
    monkeypatch.setattr('gateway.session_managed_worker._session_yolo', lambda *args: False)
    source = SimpleNamespace(chat_id='chat', user_id='user')
    authority = SimpleNamespace(profile_id='test', sessions={'sid': SimpleNamespace(source=source, route='route')})
    frame = _bootstrap(authority, SimpleNamespace(session_id='sid'), {'payload': {'text': 'hello'}}, policy, {})
    hydrated = restore_policy(frame['policy'])
    frozen, model = hydrated.effective()
    assert not cli_launch_route_pinned(hydrated, frozen, model)
    assert frozen['_cli_launch_route_released'] is True
    assert '_cli_launch_route_released' not in json.loads(policy.config_json)


def test_mutated_and_profile_replaced_route_no_longer_holds_cli_pin(tmp_path):
    from gateway.session_worker_construct import cli_launch_route_pinned
    config = {'model': {'provider': 'google', 'default': 'gemini-test'}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'gemini-test'}, config)
    frozen, model = policy.effective()
    assert cli_launch_route_pinned(policy, frozen, model)
    changed = {**frozen, '_cli_launch_route_released': True}
    switched = replace(policy, config_json=json.dumps(changed), model='new-model')
    assert not cli_launch_route_pinned(switched, changed, 'new-model')
    followed = {**frozen, 'model': {'provider': 'openrouter', 'default': 'other'}}
    assert not cli_launch_route_pinned(policy, followed, 'other')


def test_pinned_failure_result_never_echoes_error(tmp_path, caplog):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'gemini-test'},
                          {'model': {'provider': 'google', 'default': 'gemini-test'}})
    secret = 'Bearer fake-token-ABC123 https://user:pass@host/?api_key=fake-key'
    try:
        raise AuthError(secret)
    except AuthError as exc:
        result = _resolution_failure_result(exc, policy, SimpleNamespace(platform=Platform.LOCAL), 'test')
    rendered = repr(result) + caplog.text
    assert 'Provider credentials could not be resolved' in rendered
    for fragment in ('fake-token', 'ABC123', 'user:pass', 'fake-key'):
        assert fragment not in rendered


@pytest.mark.parametrize(('error', 'expected'), [
    (AuthError('untrusted detail', code='invalid_provider'), 'Unknown provider'),
    (AuthError('untrusted detail', relogin_required=True), 're-authenticate'),
])
def test_pinned_failure_retains_actionable_diagnosis(tmp_path, error, expected):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'example'},
                          {'model': {'provider': 'google', 'default': 'example'}})
    result = _resolution_failure_result(error, policy, SimpleNamespace(platform=Platform.LOCAL), 'test')
    assert expected in result['final_response']
    assert 'untrusted detail' not in repr(result)


def test_pinned_quota_keeps_structured_reset_window(tmp_path):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    from hermes_cli.auth_codex import _codex_quota_exhausted_error
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'example'},
                          {'model': {'provider': 'openai-codex', 'default': 'example'}})
    result = _resolution_failure_result(_codex_quota_exhausted_error(604800), policy,
                                        SimpleNamespace(platform=Platform.LOCAL), 'test')
    assert '604800s' in result['final_response']
    assert 'Wait a moment' not in result['final_response']


def test_pinned_transient_auth_is_not_misreported_as_bad_credentials(tmp_path):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'example'},
                          {'model': {'provider': 'nous', 'default': 'example'}})
    exc = AuthError('untrusted detail', code='upstream_blocked', retryable=True)
    result = _resolution_failure_result(exc, policy, SimpleNamespace(platform=Platform.LOCAL), 'test')
    assert 'temporarily unavailable' in result['final_response']
    assert result['failure_retryable'] is True
    assert 'untrusted detail' not in repr(result)


def test_pinned_launch_credential_error_preserves_safe_reason(tmp_path):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    from hermes_state_runtime import RuntimeStoreError
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'example'},
                          {'model': {'provider': 'google', 'default': 'example'}})
    exc = RuntimeStoreError('launch_credentials_unavailable', field='model.api_key',
                            detail='changed in config.yaml since this session started')
    result = _resolution_failure_result(exc, policy, SimpleNamespace(platform=Platform.LOCAL), 'test')
    assert 'model.api_key' in result['final_response']
    assert 'changed in config.yaml since this session started' in result['final_response']


def test_pinned_local_server_unavailable_keeps_failure_classification(tmp_path):
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_result
    from hermes_cli.runtime_provider_custom import LocalRuntimeUnavailableError
    policy = build_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'local'},
                          {'model': {'provider': 'custom', 'default': 'local'}})
    result = _resolution_failure_result(LocalRuntimeUnavailableError('secret', retryable=True), policy,
                                        SimpleNamespace(platform=Platform.LOCAL), 'test')
    assert result['failure_reason'] == 'local_runtime_unavailable'
    assert result['failure_retryable'] is True
    assert 'secret' not in repr(result)
