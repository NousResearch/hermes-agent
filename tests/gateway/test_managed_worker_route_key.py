"""A managed worker without a launch key (every Kanban run) resolves its route's credential the
way the in-process runner does (``_resolve_policy_agent_runtime``), so an ``anthropic_messages``
provider other than Anthropic gets its own key instead of ``Bearer `` (#139155). The key comes from
the worker's own profile home, never the default one, and a bound launch key still wins."""
import json
import secrets
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_managed_turn_contract import assignment

BASE_URL = 'https://api.minimax.io/anthropic'
KEY_ENV = 'KB5_MINIMAX_TEST_KEY'


def _minimax_frame(tmp_path, monkeypatch, api_key=None):
    frame, _ = assignment(tmp_path, monkeypatch)
    config = json.loads(frame['policy']['config_json'])
    config['model'] = {'default': 'MiniMax-M2.7', 'provider': 'custom:minimax', 'base_url': BASE_URL,
                       'api_mode': 'anthropic_messages'}
    config['custom_providers'] = [{'name': 'minimax', 'base_url': BASE_URL, 'key_env': KEY_ENV,
                                   'api_mode': 'anthropic_messages'}]
    request = {k: v for k, v in json.loads(frame['policy']['request_json']).items() if k not in ('provider', 'base_url')}
    policy = {**frame['policy'], 'model': 'MiniMax-M2.7', 'config_json': json.dumps(config),
              'request_json': json.dumps(request)}
    return {**frame, 'policy': policy, 'api_key': api_key}


def _execute(frame, monkeypatch, base=object, built=None, result=None, sent=None):
    from agent import managed_worker as worker
    built = {} if built is None else built

    class Agent(base):
        def __init__(self, **kwargs):
            built.update(kwargs)
            if base is not object:
                super().__init__(**kwargs)
            self.session_prompt_tokens = self.session_completion_tokens = 0
        def run_conversation(self, text, **kwargs):
            return dict(result or {'final_response': 'ok', 'completed': True})

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
        def approval(self, data): pass
        def clarify(self, questions): pass

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    sent = [] if sent is None else sent
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: sent.append((kind, payload))))
    return built


def test_keyless_worker_resolves_its_custom_provider_key(tmp_path, monkeypatch):
    key = 'fake-' + secrets.token_hex(8)
    monkeypatch.setenv(KEY_ENV, key)
    built = _execute(_minimax_frame(tmp_path, monkeypatch), monkeypatch)
    assert built['provider'] == 'custom:minimax' and built['base_url'] == BASE_URL
    assert built['api_key'] == key


def test_keyless_worker_reads_its_own_profile_home_not_the_default(tmp_path, monkeypatch):
    """The owner launches a secondary profile's worker with that profile's env (``_worker_env``);
    the worker's resolution then sees its own key, never the launch (default) profile's."""
    import os
    from gateway.session_managed_worker import _worker_env
    default = tmp_path / '.hermes'
    profile = default / 'profiles' / 'kb5'
    profile.mkdir(parents=True)
    own, other = 'fake-own-' + secrets.token_hex(8), 'fake-default-' + secrets.token_hex(8)
    (default / '.env').write_text(f'{KEY_ENV}={other}\n')
    (profile / '.env').write_text(f'{KEY_ENV}={own}\n')
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(default))
    monkeypatch.setenv(KEY_ENV, other)
    env = _worker_env(SimpleNamespace(profile_id=str(profile)))
    assert env is not None and env['HERMES_HOME'] == str(profile)
    for name in list(os.environ):
        monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    built = _execute(_minimax_frame(tmp_path, monkeypatch), monkeypatch)
    assert built['api_key'] == own


def test_bound_launch_key_wins_over_the_route_key(tmp_path, monkeypatch):
    launch = 'fake-launch-' + secrets.token_hex(8)
    monkeypatch.setenv(KEY_ENV, 'fake-route-' + secrets.token_hex(8))
    built = _execute(_minimax_frame(tmp_path, monkeypatch, api_key=launch), monkeypatch)
    assert built['api_key'] == launch


def test_primary_auth_failure_never_borrows_a_fallback_providers_key(tmp_path, monkeypatch):
    """A fallback entry's credential belongs to another provider; the worker's primary route never
    carries it (the agent's own fallback chain switches provider, model and key together)."""
    calls = []

    def resolve(config, **kwargs):
        calls.append(kwargs)
        return {'provider': 'openrouter', 'api_key': 'fake-other-provider'}, {'provider': 'openrouter', 'model': 'x'}
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_with_fallback', resolve)
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime', lambda **kw: None)
    built = _execute(_minimax_frame(tmp_path, monkeypatch), monkeypatch)
    assert calls and built['api_key'] is None


def _kanban_failures():
    from hermes_cli import kanban_db as kb
    from hermes_cli.auth import AuthError
    from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE
    return [(AuthError('quota', provider='openai-codex', code=CODEX_RATE_LIMITED_CODE), kb.KANBAN_RATE_LIMIT_EXIT_CODE),
            (AuthError('relogin', provider='nous', relogin_required=True), kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE)]


@pytest.mark.parametrize('error, expected', _kanban_failures())
def test_keyless_worker_credential_failure_keeps_the_kanban_start_exit(tmp_path, monkeypatch, error, expected):
    """An AuthError from the key-less worker's own route resolution goes through the worker's real
    start-failure path (``execute`` -> ``_construct_agent`` -> ``record_start_failure``): a quota wall
    books 75, a re-login 78, and the agent is never built."""
    from gateway import session_kanban
    from tests.gateway.test_kanban_result_exit import _bound_claim
    path, context = _bound_claim(tmp_path / 'kanban', monkeypatch)
    frame = _minimax_frame(tmp_path, monkeypatch)
    frame = {**frame, 'policy': {**frame['policy'], 'kanban_json': json.dumps(context)}}

    def resolve(**kwargs):
        raise error
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime', resolve)
    # Only the HERMES_KANBAN_* environment binding is skipped (it writes os.environ directly).
    monkeypatch.setattr(session_kanban, 'bind_worker_context', lambda frame: None)
    built = {}
    with pytest.raises(type(error)):
        _execute(frame, monkeypatch, built=built)
    assert built == {}
    assert session_kanban.worker_exit_code(path, context) == expected


FALLBACK = {'provider': 'openrouter', 'model': 'fallback-model'}


def _with_turn_chain(frame, chain):
    """The owner's per-turn ``construct_v1``: an ordinary session's refreshed profile chain, which the
    frozen snapshot (no fallbacks here) does not carry."""
    request = json.loads(frame['policy']['request_json'])
    request['construct_v1'] = {'fallback_model': chain, 'ephemeral_system_prompt': None, 'prefill_messages': None}
    return {**frame, 'policy': {**frame['policy'], 'request_json': json.dumps(request)}}


def _primary_auth_failure(monkeypatch, fallback_ok):
    from hermes_cli.auth import AuthError
    calls = []

    def resolve(requested=None, **kwargs):
        calls.append(requested)
        if requested == FALLBACK['provider'] and fallback_ok:
            return {'provider': 'openrouter', 'api_key': 'fake-fallback', 'base_url': 'https://openrouter.ai/api/v1'}
        raise AuthError('relogin', provider='nous', relogin_required=True)
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime', lambda **kw: None)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', resolve)
    return calls


def test_primary_auth_failure_reaches_the_turns_refreshed_fallback_chain(tmp_path, monkeypatch):
    """An expired primary with a fallback the user added after session creation: the worker resolves
    against the chain supplied for this turn, so the agent is built (key-less, its own chain switches
    provider, model and key together) instead of a start failure."""
    calls = _primary_auth_failure(monkeypatch, fallback_ok=True)
    built = _execute(_with_turn_chain(_minimax_frame(tmp_path, monkeypatch), [FALLBACK]), monkeypatch)
    assert FALLBACK['provider'] in calls
    assert built['fallback_model'] == [FALLBACK] and built['api_key'] is None


def test_primary_auth_failure_with_an_unusable_turn_chain_keeps_the_kanban_start_exit(tmp_path, monkeypatch):
    from gateway import session_kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli.auth import AuthError
    from tests.gateway.test_kanban_result_exit import _bound_claim
    path, context = _bound_claim(tmp_path / 'kanban', monkeypatch)
    frame = _with_turn_chain(_minimax_frame(tmp_path, monkeypatch), [FALLBACK])
    frame = {**frame, 'policy': {**frame['policy'], 'kanban_json': json.dumps(context)}}
    _primary_auth_failure(monkeypatch, fallback_ok=False)
    monkeypatch.setattr(session_kanban, 'bind_worker_context', lambda frame: None)
    built = {}
    with pytest.raises(AuthError):
        _execute(frame, monkeypatch, built=built)
    assert built == {}
    assert session_kanban.worker_exit_code(path, context) == kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE


def test_keyless_worker_on_a_fallback_never_sends_its_key_to_the_primary_endpoint(tmp_path, monkeypatch):
    """Resolution landing on a configured fallback: the real agent's primary (MiniMax, Anthropic
    protocol) client is built without the fallback provider's key. Asserted where the HTTP client
    is constructed; this does not drive a turn through the fallback."""
    fallback_key = 'fake-fallback-' + secrets.token_hex(8)
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr('hermes_cli.runtime_provider_custom._resolve_named_custom_runtime', lambda **kw: None)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_with_fallback', lambda config, **kw: (
        {'provider': 'openrouter', 'api_key': fallback_key, 'base_url': 'https://openrouter.ai/api/v1'},
        {'provider': 'openrouter', 'model': 'fallback-model'}))
    clients = []

    def build_anthropic_client(api_key, base_url=None, timeout=None, **kwargs):
        clients.append((api_key, base_url))
        return SimpleNamespace()
    monkeypatch.setattr('agent.anthropic_adapter.build_anthropic_client', build_anthropic_client)
    from run_agent import AIAgent
    built = _execute(_minimax_frame(tmp_path, monkeypatch), monkeypatch, base=AIAgent)
    primary = [key for key, base_url in clients if base_url == BASE_URL]
    assert primary and fallback_key not in primary
    assert all(fallback_key != key for key, _ in clients)
    assert built['api_key'] is None
