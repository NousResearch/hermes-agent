"""Real discovery of the shipped plugin, without a compositor or host input."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('settings', [{}, {'enabled': []},
    {'enabled': ['hermes-realms'], 'disabled': ['hermes-realms']}])
def test_realms_is_inventoried_but_inert_until_enabled(tmp_path, settings):
    code = r'''
import json
from pathlib import Path
import sys
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key

home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
# Leave unrelated bundled providers out of this integration test; Realms still
# goes through the production scanner, config gate, importer and registrations.
other = [manifest_key(m) for m in collect_directory_manifests()
         if m.name != 'hermes-realms']
settings = json.loads(sys.argv[1])
settings['disabled'] = other + settings.get('disabled', [])
(home / 'config.yaml').write_text(json.dumps({'plugins': settings}), encoding='utf-8')
events = []
def no_process_or_network(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect', 'socket.bind'}:
        events.append(event)
        raise AssertionError(f'discovery must not launch or download: {event}')
sys.addaudithook(no_process_or_network)
manager = PluginManager()
manager.discover_and_load()
rows = {p['name']: p for p in manager.list_plugins()}
assert 'hermes-realms' in rows, 'Realms must ship in the native plugin inventory'
assert rows['hermes-realms']['source'] == 'bundled'
assert not rows['hermes-realms']['enabled']
assert manager.find_plugin_skill('hermes-realms:realms') is None
assert 'realms.integration' not in sys.modules
assert not (home / 'realms').exists()
assert not (home / 'plugin-data' / 'hermes-realms').exists()
assert not manager._plugins['hermes-realms'].hooks_registered
assert not manager._plugins['hermes-realms'].tools_registered
assert not manager._plugins['hermes-realms'].commands_registered
assert 'realms' not in manager._cli_commands
manager.unload()
assert not events, events
print('bundled Realms is inert by default')
'''
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT', 'TEMP', 'TMP') if key in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               APPDATA=str(tmp_path / 'AppData/Roaming'), LOCALAPPDATA=str(tmp_path / 'AppData/Local'),
               HERMES_HOME=str(tmp_path / 'profile'), PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, '-c', code, json.dumps(settings)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
