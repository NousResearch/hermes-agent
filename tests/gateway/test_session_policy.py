"""Fresh local policy is frozen and scoped, not process launcher state."""
import os
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_policy_selects_surface_and_isolates_cwd(tmp_path):
    from gateway.session_policy import build_policy, policy_scope
    from agent.runtime_cwd import resolve_agent_cwd
    from tools.terminal_scope import terminal_env
    from hermes_state_runtime import RuntimeStoreError

    cfg = {'platform_toolsets': {'cli': ['terminal']}}
    before = dict(os.environ)
    policies = []
    for source in ('cli', 'tui', 'gui'):
        cwd = tmp_path / source
        cwd.mkdir()
        policies.append(build_policy({'source': source, 'cwd': str(cwd), 'model': source}, cfg))
    assert policies[2].platform == 'desktop'
    assert 'desktop_ui' in policies[2].toolsets
    assert all('desktop_ui' not in p.toolsets for p in policies[:2])
    # Platform-gated toolsets (catalog) ride on the Desktop surface only, like the native TUI factory.
    assert 'catalog' in policies[2].toolsets and all('catalog' not in p.toolsets for p in policies[:2])
    cfg['platform_toolsets']['cli'].clear()
    assert 'terminal' in policies[0].toolsets

    def run(policy):
        with policy_scope(policy):
            assert str(resolve_agent_cwd()) == policy.cwd
            assert terminal_env('TERMINAL_CWD') == policy.cwd
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, policies))
    assert dict(os.environ) == before
    for params in ({'source': 'cron'}, {'toolsets': ['not-a-toolset']}, {'cwd': '.'},
                   {'provider': 12}, {'skills': ['x']}, {'toolsets': ['desktop_ui'], 'source': 'cli'}):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            build_policy(params, {})


def test_launch_policy_reaches_real_turn_runner(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'session_policy_peer.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=130)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'policy-receipt.json').read_text())
    assert receipt['cwd_effects'] and receipt['same_agents'] and receipt['no_spill']
    print(json.dumps(receipt))


def test_launch_options_are_frozen_and_validated(tmp_path):
    from gateway.session_policy import build_policy
    from hermes_constants import parse_reasoning_effort
    from hermes_state_runtime import RuntimeStoreError
    cfg = {'agent': {'max_turns': 8, 'reasoning_effort': 'low'}}
    params = dict(cwd=str(tmp_path), provider='custom', base_url='http://127.0.0.1:1234/v1',
                  model='fixture', reasoning='high', max_turns=3, ignore_rules=True)
    policy = build_policy(params, cfg)
    assert policy.provider == 'custom' and policy.base_url == params['base_url']
    assert policy.ignore_rules and policy.max_turns == 3
    assert policy.reasoning_config == parse_reasoning_effort('high')
    cfg['agent']['reasoning_effort'] = 'none'
    assert policy.reasoning_config == parse_reasoning_effort('high')
    assert build_policy(dict(cwd=str(tmp_path), ignore_rules=False), cfg).ignore_rules is False
    for bad in ({'max_turns': True}, {'max_turns': 0}, {'reasoning': 'garbage'},
                {'ignore_rules': 'false'}, {'base_url': 'http://user:secret@localhost/v1'}):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            build_policy(dict(cwd=str(tmp_path), **bad), cfg)


def test_explicit_key_is_private_and_missing_after_restart_fails_closed(tmp_path):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.session_policy import build_policy, bind_launch_key, launch_key
    from hermes_state_runtime import RuntimeStoreError
    authority = SimpleNamespace(instance_id='owned', profile_id='profile', epoch=1)
    sibling = SimpleNamespace(instance_id='sibling', profile_id='profile', epoch=1)
    raw = 'UNIQUE-PRIVATE-LAUNCH-KEY'
    before = dict(os.environ)
    params = dict(cwd=str(tmp_path), model='fixture', api_key=raw)
    private = {}
    policy = build_policy(params, {'model': {'api_key': raw}}, private_secrets=private)
    policy = bind_launch_key(authority, 'session-a', policy, raw, config_secrets=private)
    assert policy.config(authority)['model']['api_key'] == raw
    assert raw not in json.dumps(asdict(policy))
    assert launch_key(authority, policy) == raw
    retry_private = {}
    retry = build_policy(params, {'model': {'api_key': raw}}, private_secrets=retry_private)
    assert bind_launch_key(authority, 'session-a', retry, raw, config_secrets=retry_private) == policy
    for other in (sibling, SimpleNamespace(instance_id='owned', profile_id='profile', epoch=2)):
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            launch_key(other, policy)
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            policy.config(other)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_launch_key(authority, 'session-a', build_policy(params, {}), 'different')
    assert dict(os.environ) == before


def test_ordinary_daemon_cli_launch_policy(tmp_path):
    import subprocess
    import sys
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c',
        'import json,sys; from pathlib import Path; '
        'from tests.gateway.fixtures.cli_launch_policy_probe import probe; '
        'print(json.dumps(probe(Path(sys.argv[1]))))', str(tmp_path)], cwd=root,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    print(json.dumps(json.loads(result.stdout.splitlines()[-1])))


def test_null_config_sections_read_as_absent(tmp_path):
    # A fresh install's config.yaml carries a bare ``gateway:`` (YAML null); the managed-worker gate chains
    # ``.get('gateway', {}).get(...)`` on the frozen policy and must read the default, not crash the turn.
    from gateway.session_policy import build_policy, LocalSessionPolicy
    from dataclasses import replace
    import json

    cfg = {'gateway': None, 'display': None, 'platform_toolsets': {'cli': ['terminal']}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path)}, cfg)
    assert policy.config().get('gateway', {}).get('managed_workers') is None
    legacy = replace(policy, config_json=json.dumps(cfg))  # persisted before normalization
    assert isinstance(legacy, LocalSessionPolicy)
    assert legacy.config().get('display', {}).get('busy_input_mode', 'interrupt') == 'interrupt'



def test_frozen_route_keeps_its_endpoint_after_live_config_edit(tmp_path, monkeypatch):
    """R2-M2: the frozen policy's config-derived endpoint is the one its frozen credential
    belongs to; a later ``model.base_url`` edit must not redirect that credential."""
    import json
    from types import SimpleNamespace
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_cli.config_effective import load_user_config_effective
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))

    def write(url, key):
        (home / 'config.yaml').write_text(json.dumps(
            {'model': {'provider': 'custom', 'base_url': url, 'api_key': key, 'default': 'm'}}))
    write('http://127.0.0.1:1/v1', 'sk-frozen-endpoint-one')
    private = {}
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, load_user_config_effective(home / 'config.yaml'),
                          private_secrets=private)
    authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=None)
    policy = bind_launch_key(authority, 'sid', policy, None, config_secrets=private)
    write('http://127.0.0.1:2/v1', 'sk-new-endpoint-two')
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
    runner = SimpleNamespace(session_authority=authority)
    _, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=SimpleNamespace())
    assert (runtime['base_url'], runtime['api_key']) == ('http://127.0.0.1:1/v1', 'sk-frozen-endpoint-one')


def test_frozen_bare_custom_route_keeps_its_endpoint_pool_and_launch_key_wins(tmp_path, monkeypatch):
    """The frozen ``model.api_key`` is config, not a launch key: a URL-matched credential pool still
    serves the frozen endpoint (refresh/rotation on 401/429), while a bound launch key wins (R2-M1)."""
    import json
    from types import SimpleNamespace
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_cli.config_effective import load_user_config_effective
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    url = 'http://127.0.0.1:1/v1'
    (home / 'config.yaml').write_text(json.dumps(
        {'model': {'provider': 'custom', 'base_url': url, 'api_key': 'sk-frozen-config', 'default': 'm'}}))
    pool = object()
    monkeypatch.setattr('hermes_cli.runtime_provider._try_resolve_from_custom_pool', lambda base_url, *a, **k: {
        'provider': 'custom', 'api_mode': 'chat_completions', 'base_url': base_url, 'api_key': 'sk-pooled',
        'source': 'pool:custom', 'credential_pool': pool} if base_url.rstrip('/') == url else None)
    authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=None)
    runner = SimpleNamespace(session_authority=authority)
    runtimes = []
    for sid, launch in (('config-keyed', None), ('launch-keyed', 'sk-launch-explicit')):
        private = {}
        params = {'cwd': str(tmp_path), 'model': 'm'} | ({'api_key': launch} if launch else {})
        policy = build_policy(params, load_user_config_effective(home / 'config.yaml'), private_secrets=private)
        policy = bind_launch_key(authority, sid, policy, launch, config_secrets=private)
        monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source, p=policy: p)
        runtimes.append(GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=SimpleNamespace())[1])
    assert (runtimes[0]['base_url'], runtimes[0]['credential_pool']) == (url, pool)
    assert (runtimes[1]['api_key'], runtimes[1]['credential_pool']) == ('sk-launch-explicit', None)


def test_lazy_info_reports_the_free_tier_pinned_model(tmp_path, monkeypatch):
    """A not-yet-built session with no launch model/provider runs on ``nous/welcome`` on the free
    tier (the agent build pins it), so ``session.info`` says so; an explicit launch model stands."""
    from dataclasses import replace
    from types import SimpleNamespace
    from gateway.session_local import _lazy_model
    from gateway.session_policy import build_policy
    import hermes_cli.anon_auth as anon_auth
    authority = SimpleNamespace(runner=None, profile_id='fixture')
    monkeypatch.setattr(anon_auth, 'free_tier_route', lambda: True)
    configured = replace(build_policy({'cwd': str(tmp_path)}, {'model': {'default': 'cfg-model'}}), model='cfg-model')
    assert _lazy_model(authority, configured) == anon_auth.GUEST_MODEL
    assert _lazy_model(authority, build_policy({'cwd': str(tmp_path), 'model': 'pick'}, {})) == 'pick'
    monkeypatch.setattr(anon_auth, 'free_tier_route', lambda: False)
    assert _lazy_model(authority, configured) == 'cfg-model'
