"""A managed worker without a launch key (every Kanban run) resolves its route's credential the
way the in-process runner does (``_resolve_policy_agent_runtime``), so an ``anthropic_messages``
provider other than Anthropic gets its own key instead of ``Bearer `` (#139155). The key comes from
the worker's own profile home, never the default one, and a bound launch key still wins."""
import json
import secrets
import threading
from types import SimpleNamespace

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


def _execute(frame, monkeypatch):
    from agent import managed_worker as worker
    built = {}

    class Agent:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.session_prompt_tokens = self.session_completion_tokens = 0
        def run_conversation(self, text, **kwargs):
            return {'final_response': 'ok', 'completed': True}

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
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: None))
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
