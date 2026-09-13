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
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
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


