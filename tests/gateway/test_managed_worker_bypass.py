"""Bypass sessions execute in the admission-owned worker, which binds the safe policy first."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _policy(tmp_path, **extra):
    from gateway.session_local import _bypass_policy
    return _bypass_policy({'cwd': str(tmp_path), 'model': 'safe-fixture', 'provider': 'custom',
                           'base_url': 'http://127.0.0.1:9/v1', **extra}, private_secrets={})


def test_bypass_policy_routes_to_managed_worker_and_bootstrap_carries_flags(tmp_path, monkeypatch):
    from gateway import session_managed_worker as smw
    from gateway.session_policy import build_policy
    from agent.managed_worker import validate_bootstrap
    from hermes_state_runtime import RuntimeStoreError

    safe = _policy(tmp_path, safe_mode=True)
    config_only = _policy(tmp_path, ignore_user_config=True)
    ordinary = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {'platform_toolsets': {'cli': ['terminal']}})
    opted = build_policy({'cwd': str(tmp_path), 'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'},
                         {'gateway': {'managed_workers': True}})
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    ref = SimpleNamespace(session_id='sid')
    for policy, expected in ((safe, safe), (config_only, config_only), (ordinary, None), (opted, opted)):
        monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source, p=policy: p)
        assert smw.managed_policy(authority, ref) is expected
    # A bypass launch never depends on the profile's gateway.managed_workers opt-in.
    assert 'gateway' not in safe.config() or safe.config()['gateway'].get('managed_workers') is not True
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    row = {'payload': {'text': 'SAFE_PROBE'}}
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    frame = smw._bootstrap(authority, ref, row, safe, scope)
    assert frame['safe_mode'] is True and frame['ignore_user_config'] is True
    assert isinstance(frame['policy']['config_json'], str)
    wire = json.loads(json.dumps(frame))
    assert validate_bootstrap(wire) is wire
    frame = smw._bootstrap(authority, ref, row, config_only, scope)
    assert frame['safe_mode'] is False and frame['ignore_user_config'] is True
    for bad in ({'safe_mode': 1}, {'ignore_user_config': 'yes'}, {'safe_mode': True, 'ignore_user_config': False}):
        with pytest.raises(ValueError):
            validate_bootstrap({**frame, **bad})


@pytest.mark.parametrize('mode', ['safe', 'config'])
def test_worker_binds_bypass_policy_before_runtime_imports(tmp_path, mode):
    """The frozen explicit config is what the worker's config readers return; the profile's
    config.yaml (malformed here) is never opened and plugin discovery never runs."""
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: [unterminated\n', encoding='utf-8')
    plugin = home / 'plugins' / 'sentinel'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: sentinel\nversion: 1.0.0\nkind: standalone\n', encoding='utf-8')
    (plugin / '__init__.py').write_text("import os; from pathlib import Path\nPath(os.environ['HERMES_HOME'], 'plugin-executed').touch()\ndef register(ctx): pass\n", encoding='utf-8')
    script = tmp_path / 'probe.py'
    script.write_text(f'''
import json, os, sys, threading
sys.path.insert(0, {str(root)!r})
from agent.managed_worker import bind_bypass_policy
home = os.environ['HERMES_HOME']
frame = {{'safe_mode': {mode == 'safe'!r}, 'ignore_user_config': True,
         'policy': {{'config_json': json.dumps({{'model': {{'default': 'safe-fixture', 'provider': 'custom'}},
                    'plugins': {{'enabled': ['sentinel']}}, 'agent': {{'max_turns': 7}}}})}}}}
opened = []
sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == 'open' and home in str(args[0]) else None)
bind_bypass_policy(frame)
from hermes_cli.config import load_config, read_raw_config
from hermes_cli.plugins import discover_plugins, get_plugin_manager
discover_plugins()
cfg = load_config()
raw = read_raw_config()
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(1) as pool:
    helper = pool.submit(lambda: load_config()['agent']['max_turns']).result()
print(json.dumps({{'max_turns': cfg['agent']['max_turns'], 'raw_model': raw['model'], 'helper': helper,
    'opened': opened, 'plugins': sorted(get_plugin_manager()._plugins),
    'executed': os.path.exists(os.path.join(home, 'plugin-executed'))}}))
''', encoding='utf-8')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, str(script)], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt['max_turns'] == 7 and receipt['helper'] == 7
    assert receipt['raw_model'] == {'default': 'safe-fixture', 'provider': 'custom'}
    assert not [p for p in receipt['opened'] if p.endswith('config.yaml') or p.endswith('.env')], receipt
    if mode == 'safe':
        assert receipt['plugins'] == [] and receipt['executed'] is False, receipt
    else:
        assert receipt['executed'] is True, receipt  # config-only keeps plugins


def test_worker_outbox_directory_is_portable_and_distinct_per_execution(tmp_path):
    """execution_id ('admission-worker:<hex>') is a durable ledger key, not a path: ':' is
    illegal in a Windows directory name (WinError 267), so the outbox dir must be derived."""
    from agent.managed_worker import outbox_dir
    first = outbox_dir(tmp_path, 'admission-worker:ef7caea811d14c7092fd261cd167b311')
    second = outbox_dir(tmp_path, 'admission-worker:ef7caea811d14c7092fd261cd167b312')
    assert first.parent == tmp_path / 'worker-outboxes' and first != second
    assert not set(first.name).intersection(':\\/<>"|?*'), first.name
    first.mkdir(parents=True)
    assert first.is_dir()
