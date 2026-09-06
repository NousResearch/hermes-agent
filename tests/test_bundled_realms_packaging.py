"""Exercise built payloads, not a downloader or a source-text snapshot."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import tarfile

import pytest

from tests.test_packaging_build_guard import _build_artifact


@pytest.mark.linux_only
@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
def test_sealed_artifact_contains_loadable_realms_with_viewer_and_notices(tmp_path, kind):
    result = _build_artifact(kind, tmp_path, nix_build=True)
    assert result.returncode == 0, result.stderr
    installed = tmp_path / 'installed'
    if kind == 'wheel':
        with zipfile.ZipFile(next(tmp_path.glob('*.whl'))) as wheel:
            wheel.extractall(installed)
    else:
        with tarfile.open(next(tmp_path.glob('*.tar.gz'))) as sdist:
            sdist.extractall(installed, filter='data')
        installed = next(installed.iterdir())
    plugin = installed / 'plugins' / 'hermes-realms'
    # Data consumed by native discovery, skills, the API and the actual viewer.
    for relative in ('plugin.yaml', 'LICENSE', 'skills/realms/SKILL.md',
                     'dashboard/manifest.json', 'realms/web/viewer.html',
                     'realms/web/THIRD_PARTY.md', 'realms/web/vendor/novnc/LICENSE.txt',
                     'realms/web/vendor/novnc/vendor/pako/LICENSE'):
        assert (plugin / relative).is_file(), f'{kind} omitted {relative}'
    code = r'''
import json, sys
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key
home = get_hermes_home()
home.mkdir(parents=True)
other = [manifest_key(m) for m in collect_directory_manifests() if m.name != 'hermes-realms']
(home / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': [], 'disabled': other}}), encoding='utf-8')
events = []
def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect', 'socket.bind'}:
        events.append(event)
        raise AssertionError(event)
sys.addaudithook(audit)
manager = PluginManager()
manager.discover_and_load()
assert not manager._plugins['hermes-realms'].enabled
assert 'realms.integration' not in sys.modules
assert manager.find_plugin_skill('hermes-realms:realms') is None
assert not (home / 'realms').exists()
manager.unload()
(home / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': ['hermes-realms'], 'disabled': other}}), encoding='utf-8')
manager = PluginManager()
manager.discover_and_load()
loaded = manager._plugins['hermes-realms']
assert loaded.enabled and loaded.error is None, loaded.error
assert 'realm' in loaded.tools_registered
assert 'on_session_identity' in loaded.hooks_registered
manager.invoke_hook('on_session_identity', session_id='conversation-probe', runtime_session_id='runtime-probe',
                    stored_session_id='stored-probe', task_id='task-probe', hermes_home=str(home))
assert manager.find_plugin_skill('hermes-realms:realms').is_file()
assert 'realms' in manager._cli_commands, 'enabled plugin must expose its native setup CLI'
import argparse
entry = manager._cli_commands['realms']
parser = argparse.ArgumentParser(prog='hermes realms')
entry['setup_fn'](parser)
args = parser.parse_args(['list'])
assert entry['handler_fn'](args) == 0
install_args = parser.parse_args(['install-driver', '--archive', str(home / 'untrusted.tar.gz')])
from importlib import import_module
package = entry["handler_fn"].__module__.rpartition(".")[0]
driver_path = import_module(package + ".config").driver_path
assert install_args.target is None
(home / 'untrusted.tar.gz').write_bytes(b'not the pinned release')
assert entry['handler_fn'](install_args) == 1
assert not driver_path().exists()
integration = import_module(package + ".integration")
get_integration, requirements_available = integration.get_integration, integration.requirements_available
service = get_integration()
assert service.home == home
assert service.owners.resolve(runtime_session_id='runtime-probe') == service.owners.resolve(stored_session_id='stored-probe')
assert service.driver_executable.is_relative_to(home), 'driver must live in writable profile data, not the sealed package'
from importlib import import_module
package = entry["handler_fn"].__module__.rpartition(".")[0]
driver_path = import_module(package + ".config").driver_path
assert driver_path() == service.driver_executable
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
other_home = home.parent / 'other-profile'
token = set_hermes_home_override(other_home)
try:
    assert driver_path().is_relative_to(other_home)
finally:
    reset_hermes_home_override(token)
assert not requirements_available(), 'no driver has been installed'
assert service.manager.list() == []
# The server serves these very assets from the sealed package.
get_profile_viewer = import_module(package + ".bridge").get_profile_viewer
from fastapi.testclient import TestClient
viewer = get_profile_viewer(home)
viewer.origin = 'http://testserver'
with TestClient(viewer.app) as client:
    response = client.get('/assets/vendor/novnc/core/rfb.js')
    assert response.status_code == 200, response.text
manager.unload()
assert not events, events
assert not (Path(sys.argv[1]) / 'vendor' / 'cua-driver').exists()
print('sealed wheel: native plugin, skill and noVNC work without checkout')
'''
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'home' / 'profile'),
               PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1')
    # Sealed package code is immutable; all state must remain in the profile.
    for path in plugin.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    plugin.chmod(0o555)
    try:
        completed = subprocess.run([sys.executable, '-c', code, str(plugin)], cwd=tmp_path,
                                   env=env, text=True, capture_output=True, timeout=60)
    finally:
        plugin.chmod(0o755)
        for path in plugin.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert completed.returncode == 0, completed.stdout + completed.stderr
