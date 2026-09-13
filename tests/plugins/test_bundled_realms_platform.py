"""Unsupported hosts must fail before registration, state, or driver acquisition."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _assert_unsupported_host_is_inert(tmp_path):
    code = r'''
import json, sys
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key
home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
other = [manifest_key(m) for m in collect_directory_manifests() if m.name != 'hermes-realms']
(home / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': ['hermes-realms'], 'disabled': other}}), encoding='utf-8')
def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'}:
        raise AssertionError(event)
sys.addaudithook(audit)
manager = PluginManager()
manager.discover_and_load()
plugin = manager._plugins['hermes-realms']
assert plugin.error and 'Linux' in plugin.error, plugin.error
assert not plugin.hooks_registered
assert not plugin.tools_registered
assert 'realms' not in manager._cli_commands
assert not (home / 'realms').exists()
assert not (home / 'plugin-data' / 'hermes-realms').exists()
'''
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT', 'TEMP', 'TMP') if k in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               APPDATA=str(tmp_path / 'AppData/Roaming'), LOCALAPPDATA=str(tmp_path / 'AppData/Local'),
               HERMES_HOME=str(tmp_path / 'profile'), PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.windows_only
def test_enabled_realms_is_inert_on_windows(tmp_path):
    _assert_unsupported_host_is_inert(tmp_path)


@pytest.mark.macos_only
def test_enabled_realms_is_inert_on_macos(tmp_path):
    _assert_unsupported_host_is_inert(tmp_path)
