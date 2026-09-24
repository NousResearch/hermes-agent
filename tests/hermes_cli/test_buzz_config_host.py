"""Real bundled discovery and host ASGI boundary; no service or frontend startup."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter('buzz')  # provenance, actual runtime import


def test_actual_bundled_discovery():
    from hermes_cli.web_server_dashboard import _discover_dashboard_plugins
    root = Path(__file__).resolve().parents[2]
    matches = [p for p in _discover_dashboard_plugins() if p['name'] == 'buzz-platform']
    assert len(matches) == 1, 'actual bundled Buzz Dashboard is absent'
    plugin = matches[0]
    assert plugin['source'] == 'bundled'
    assert Path(plugin['_dir']).resolve() == root / 'plugins/platforms/buzz/dashboard'
    assert plugin['_api_file'] == 'plugin_api.py'
    assert plugin['has_api'] is True
    assert plugin['tab']['hidden'] is True


HOST_PROBE = r"""
import json, os, sys
from pathlib import Path
from fastapi.testclient import TestClient
home = Path(os.environ['HERMES_HOME'])
path = home / 'config.yaml'
scenario = sys.argv[1]
# Fresh process imports the real host app: its top-level startup mount performs
# discovery and trust checks. No manual APIRouter or dependency overrides.
from hermes_cli import web_server as host
from hermes_cli import web_server_dashboard as dashboard
assert host.app.dependency_overrides == {}
assert Path(host.__file__).resolve().is_relative_to(Path.cwd())
assert host._SESSION_TOKEN == 'synthetic-host-session'
client = TestClient(host.app, base_url='http://127.0.0.1')
headers = {'X-Hermes-Session-Token': 'synthetic-host-session'}
url = '/api/plugins/buzz-platform/policy'
body = {'policy': {'allow_all_users': False}}
before = path.read_bytes()
assert client.put(url, json=body).status_code == 401
assert client.put(url, json=body, headers={'X-Hermes-Session-Token': 'wrong'}).status_code == 401
assert path.read_bytes() == before
if scenario == 'disabled-startup':
    assert client.put(url, json=body, headers=headers).status_code == 404
    assert path.read_bytes() == before
    assert 'hermes_dashboard_plugin_buzz-platform' not in sys.modules
elif scenario == 'enabled-and-revoked':
    assert 'hermes_dashboard_plugin_buzz-platform' in sys.modules
    response = client.put(url, json=body, headers=headers)
    assert response.status_code == 200, (response.status_code, response.text)
    assert path.read_bytes() != before
    assert client.get(url, headers=headers).json()['policy']['allow_all_users'] is False
    import yaml
    config = yaml.safe_load(path.read_text())
    assert config['gateway']['platforms']['buzz']['extra']['allow_all_users'] is False
    config['plugins'] = {'disabled': ['buzz-platform']}
    path.write_text(yaml.safe_dump(config))
    disabled = path.read_bytes()
    assert client.put(url, json={'policy': {'allow_all_users': True}}, headers=headers).status_code == 404
    assert path.read_bytes() == disabled
    assert client.put(url, json=body).status_code == 401
    assert client.get('/dashboard-plugins/buzz-platform/manifest.json').status_code == 404
elif scenario == 'assets':
    prefix = '/dashboard-plugins/buzz-platform/'
    root = Path.cwd() / 'plugins/platforms/buzz/dashboard'
    for asset in ['manifest.json', 'dist/index.js', 'dist/style.css', 'assets/BuzzLogo24px.svg']:
        response = client.get(prefix + asset)
        assert response.status_code == 200, (asset, response.text)
        assert response.content == (root / asset).read_bytes()
    assert client.get(prefix + '%2e%2e%2fmanifest.json').status_code == 403
    assert client.get(prefix + 'plugin_api.py').status_code == 404
elif scenario in ('project', 'untrusted-user', 'trusted-user'):
    name = 'host-boundary-probe'
    assert any(p['name'] == name for p in host._get_dashboard_plugins())
    imported = (home / 'executed').exists()
    response = client.put('/api/plugins/' + name + '/write', headers=headers)
    if scenario == 'trusted-user':
        assert imported and response.status_code == 200, response.text
        assert (home / 'written').read_text() == 'yes'
    else:
        assert not imported, 'untrusted API imported'
        # Project has no mounted endpoint: SPA GET fallback yields 405 for PUT.
        assert response.status_code == (405 if scenario == 'project' else 404), (response.status_code, response.text)
        assert not (home / 'written').exists()
        assert 'hermes_dashboard_plugin_' + name not in sys.modules
print(json.dumps({'scenario': scenario, 'host': host.__file__, 'native_auth': True, 'manual_router': False}))
"""


@pytest.mark.parametrize('scenario', ['enabled-and-revoked', 'disabled-startup', 'assets',
                                      'project', 'untrusted-user', 'trusted-user'])
def test_real_host_boundary(tmp_path, scenario):
    home = tmp_path / 'home'
    home.mkdir()
    managed = tmp_path / 'managed'
    managed.mkdir()
    config = {'buzz': {'allow_all_users': True}, 'plugins': {'enabled': ['buzz-platform']}}
    if scenario == 'disabled-startup':
        config['plugins']['disabled'] = ['buzz-platform']
    if scenario == 'trusted-user':
        config['plugins']['enabled'].append('host-boundary-probe')
    (home / 'config.yaml').write_text(json.dumps(config))
    (home / '.env').write_text('')
    env = dict(os.environ, HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(managed),
               HERMES_DASHBOARD_SESSION_TOKEN='synthetic-host-session')
    env.pop('HERMES_DESKTOP', None)
    env.pop('HERMES_ENABLE_PROJECT_PLUGINS', None)
    cwd = Path(__file__).resolve().parents[2]
    if scenario in ('project', 'untrusted-user', 'trusted-user'):
        if scenario == 'project':
            # Real project discovery follows cwd. Keep the actual source on
            # PYTHONPATH but execute under a synthetic project root.
            cwd = tmp_path / 'project'
            cwd.mkdir()
            root = cwd / '.hermes/plugins'
            env['HERMES_ENABLE_PROJECT_PLUGINS'] = 'true'
        else:
            root = home / 'plugins'
        dashboard = root / 'probe/dashboard'
        dashboard.mkdir(parents=True)
        (dashboard / 'manifest.json').write_text(json.dumps({
            'name': 'host-boundary-probe', 'api': 'plugin_api.py'}))
        (dashboard / 'plugin_api.py').write_text(
            "import os\nfrom pathlib import Path\nfrom fastapi import APIRouter\n"
            "home = Path(os.environ['HERMES_HOME'])\nhome.joinpath('executed').write_text('yes')\n"
            "router = APIRouter()\n@router.put('/write')\ndef write():\n"
            "    home.joinpath('written').write_text('yes')\n    return {'ok': True}\n")
    probe = HOST_PROBE.replace('assert Path(host.__file__).resolve().is_relative_to(Path.cwd())',
                              'assert Path(host.__file__).resolve().is_relative_to(Path(' + repr(str(Path(__file__).resolve().parents[2])) + '))')
    result = subprocess.run([sys.executable, '-c', probe, scenario], cwd=cwd,
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
