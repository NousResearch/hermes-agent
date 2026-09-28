"""Bypass launches freeze a typed isolation policy chosen before any profile read."""
import json
import os

import pytest


def test_bypass_policy_is_typed_frozen_and_safe_implies_ignore(tmp_path):
    from dataclasses import asdict
    from gateway.session_policy import build_policy, restore_policy, CREATE_FIELDS
    from hermes_state_runtime import RuntimeStoreError

    assert {'safe_mode', 'ignore_user_config'} <= CREATE_FIELDS
    cfg = {'platform_toolsets': {'cli': ['terminal']}}
    ordinary = build_policy({'cwd': str(tmp_path), 'model': 'm'}, cfg)
    assert ordinary.safe_mode is False and ordinary.ignore_user_config is False
    safe = build_policy({'cwd': str(tmp_path), 'model': 'm', 'safe_mode': True}, cfg)
    assert safe.safe_mode is True and safe.ignore_user_config is True
    config_only = build_policy({'cwd': str(tmp_path), 'model': 'm', 'ignore_user_config': True}, cfg)
    assert config_only.safe_mode is False and config_only.ignore_user_config is True
    # Safe launches also skip project rules: the request fingerprint still records only what was sent.
    assert safe.ignore_rules is True and json.loads(safe.request_json) == {'cwd': str(tmp_path), 'model': 'm', 'safe_mode': True, 'source': 'cli'}
    restored = restore_policy(asdict(safe))
    assert restored == safe and restored.safe_mode is True
    for bad in ({'safe_mode': 1}, {'safe_mode': 'true'}, {'ignore_user_config': None}):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            build_policy({'cwd': str(tmp_path), **bad}, cfg)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_policy({**asdict(safe), 'safe_mode': 'yes'})
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_policy({**asdict(safe), 'ignore_user_config': False})  # implication must hold in cold receipts too


def test_bypass_session_freezes_code_defaults_without_reading_profile(tmp_path, monkeypatch):
    """The frozen snapshot of a bypass session is code defaults + explicit options; the profile's
    (possibly malformed) config.yaml, .env and plugin toolsets are never consulted."""
    from types import SimpleNamespace
    from gateway.session_local import _bypass_policy
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    import gateway.session_policy, hermes_cli.tools_config, toolsets, tools.terminal_scope  # noqa: F401,E401
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: [unterminated\n  terminal: {backend: docker}\n', encoding='utf-8')
    (home / '.env').write_text('TERMINAL_ENV=docker\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(home))
    # The ordinary owner discovered plugins at startup, long before this launch (contract J:
    # an owner that boots against broken config is a separate limitation).
    discover_plugins()
    opened = []
    active = [True]

    def witness(event, args):
        if event == 'hermes.test.done':
            active[0] = False
        elif active[0] and event == 'open' and str(home) in str(args[0]):
            import traceback
            opened.append((str(args[0]), ''.join(traceback.format_stack(limit=25))))
    import sys
    sys.addaudithook(witness)
    cwd = tmp_path / 'work'
    cwd.mkdir()
    params = {'cwd': str(cwd), 'model': 'safe-fixture', 'provider': 'custom',
              'base_url': 'http://127.0.0.1:9/v1', 'safe_mode': True, 'toolsets': ['terminal']}
    private = {}
    policy = _bypass_policy(params, private_secrets=private)
    sys.audit('hermes.test.done')  # audit hooks cannot be removed; close the witness window
    assert not opened, '\n'.join(p + '\n' + s for p, s in opened)
    config = policy.config()
    assert config['model']['provider'] == 'custom' and config['model']['base_url'] == params['base_url']
    assert config.get('plugins', {}).get('enabled', DEFAULT_CONFIG.get('plugins', {}).get('enabled')) in (None, [])
    from hermes_cli.plugins import get_plugin_toolset_keys_nowait
    assert 'terminal' in policy.toolsets and not set(policy.toolsets) & get_plugin_toolset_keys_nowait()
    terminal = json.loads(policy.terminal_json)
    assert terminal['TERMINAL_ENV'] == 'local' and terminal['TERMINAL_CWD'] == str(cwd)
    assert dict(os.environ).get('HERMES_SAFE_MODE') is None
