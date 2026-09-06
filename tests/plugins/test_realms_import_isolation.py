"""Trusted discovery must not execute or replace a project-owned realms module."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"


@pytest.mark.linux_only
@pytest.mark.parametrize("foreign", ["path", "preloaded"])
@pytest.mark.parametrize("first", ["native", "dashboard"])
def test_real_discovery_binds_its_own_runtime(tmp_path, foreign, first):
    code = r'''
import importlib, json, pickle, sys, types
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key
from hermes_cli import web_server, web_server_dashboard as dashboard
from fastapi import FastAPI
from fastapi.testclient import TestClient
plugin_root = Path(sys.argv[1]).resolve()
home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
if sys.argv[4] == 'standalone':
    target = home / 'plugins/hermes-realms'
    target.parent.mkdir()
    target.symlink_to(plugin_root, target_is_directory=True)
other = [manifest_key(m) for m in collect_directory_manifests() if m.name != 'hermes-realms']
(home / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': ['hermes-realms'], 'disabled': other}}))
canary = Path.cwd() / 'canary'
(Path.cwd() / 'realms.py').write_text("from pathlib import Path\nPath('canary').touch()\n")
foreign = None
if sys.argv[2] == 'preloaded':
    foreign = types.ModuleType('realms')
    foreign.sentinel = object()
    sys.modules['realms'] = foreign
before = list(sys.path)
manager = PluginManager()
app = FastAPI()
web_server.app = app
web_server._dashboard_plugins_cache = None
rows = dashboard._discover_dashboard_plugins()
row = next(r for r in rows if r['name'] == 'hermes-realms')
web_server._get_dashboard_plugins = lambda: [row]
def native():
    manager.discover_and_load()
    loaded = manager._plugins['hermes-realms']
    assert loaded.enabled and not loaded.error, loaded.error
    assert loaded.tools_registered and loaded.hooks_registered
    assert manager.find_plugin_skill('hermes-realms:realms') is not None
def api():
    dashboard._mount_plugin_api_routes()
    with TestClient(app) as client:
        response = client.get('/api/plugins/hermes-realms/realms')
        assert response.status_code == 200, response.text
        assert response.json()['realms'] == []
try:
    actions = {'native': native, 'dashboard': api}
    actions[sys.argv[3]]()
    actions['dashboard' if sys.argv[3] == 'native' else 'native']()
    assert not canary.exists(), 'project realms.py executed during trusted discovery'
    assert sys.modules.get('realms') is foreign, 'foreign namespace was claimed or replaced'
    assert sys.path == before, 'plugin changed the process import search path'
    modules = [m for m in list(sys.modules.values()) if getattr(m, '__file__', None)
               and Path(m.__file__).resolve() == plugin_root / 'realms/integration.py']
    assert len(modules) == 1, [m.__name__ for m in modules]
    module = modules[0]
    assert module.__spec__.origin == str(plugin_root / 'realms/integration.py')
    config = importlib.import_module(module.__package__ + '.config').Config()
    assert type(pickle.loads(pickle.dumps(config))) is type(config)
    print(json.dumps({'runtime': module.__name__, 'origin': module.__file__, 'first': sys.argv[3], 'foreign': sys.argv[2]}))
finally:
    manager.unload()
    assert not canary.exists(), 'project realms.py executed during trusted discovery'
'''
    env = {key: os.environ[key] for key in (
        "PATH", "LANG", "TZ", "SYSTEMROOT", "TEMP", "TMP"
    ) if key in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               HERMES_HOME=str(tmp_path / "profile"), PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code, str(PLUGIN), foreign, first, "bundled"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())

@pytest.mark.linux_only
@pytest.mark.parametrize("entry", ["cli.py", "realms/launch.py", "realms/bootstrap.py", "realms/supervisor.py"])
def test_file_entrypoints_ignore_foreign_namespace(tmp_path, entry):
    code = r'''
import json, os, runpy, sys, types
from pathlib import Path
root, entry = Path(sys.argv[1]), sys.argv[2]
foreign = types.ModuleType('realms')
sys.modules['realms'] = foreign
Path('realms.py').write_text("raise AssertionError('foreign runtime executed')")
runtime = Path.cwd() / 'runtime'
runtime.mkdir()
args = {
    'cli.py': ['--home', str(Path.cwd() / 'profile'), 'list'],
    'realms/launch.py': [],
    'realms/bootstrap.py': ['publish', str(runtime)],
    'realms/supervisor.py': ['cleanup', str(Path.cwd() / 'profile'), 'r-' + '0' * 24],
}[entry]
sys.argv = [str(root / entry), *args]
before = list(sys.path)
try:
    runpy.run_path(str(root / entry), run_name='__main__')
except SystemExit as exc:
    assert exc.code == (2 if entry == 'realms/launch.py' else 0), exc.code
assert sys.modules['realms'] is foreign
assert sys.path == before
if entry == 'realms/bootstrap.py':
    assert json.loads((runtime / 'display.json').read_text()) == {}
'''
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "profile"))
    result = subprocess.run([sys.executable, "-c", code, str(PLUGIN), entry],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
