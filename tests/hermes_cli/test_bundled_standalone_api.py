"""The dashboard may inventory bundled standalone packages without importing them."""
import pytest
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import web_server_dashboard as dashboard
from hermes_cli.config import load_config, save_config


def test_disabled_bundled_api_is_not_imported(tmp_path, monkeypatch):
    # Use the actual shipped package and API importer, with an isolated ASGI app.
    app = FastAPI()
    monkeypatch.setattr(web_server, 'app', app)
    monkeypatch.setattr(web_server, '_dashboard_plugins_cache', None)
    monkeypatch.setattr(dashboard, 'get_process_hermes_home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = load_config()
    config['plugins'] = {'enabled': [], 'disabled': []}
    save_config(config)
    rows = dashboard._discover_dashboard_plugins()
    realms = next(row for row in rows if row['name'] == 'hermes-realms')
    # Only Realms is relevant here; the scan above is the actual bundled scanner.
    monkeypatch.setattr(web_server, '_get_dashboard_plugins', lambda: [realms])
    sys.modules.pop('hermes_dashboard_plugin_hermes-realms', None)
    dashboard._mount_plugin_api_routes()
    assert 'hermes_dashboard_plugin_hermes-realms' not in sys.modules
    assert not any(getattr(route, 'path', '').startswith('/api/plugins/hermes-realms') for route in app.routes)


@pytest.mark.linux_only
def test_bundled_standalone_api_uses_native_opt_in_gate(tmp_path, monkeypatch):
    # Use the actual shipped package and API importer, with an isolated ASGI app.
    app = FastAPI()
    monkeypatch.setattr(web_server, 'app', app)
    monkeypatch.setattr(web_server, '_dashboard_plugins_cache', None)
    monkeypatch.setattr(dashboard, 'get_process_hermes_home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = load_config()
    config['plugins'] = {'enabled': [], 'disabled': []}
    save_config(config)
    rows = dashboard._discover_dashboard_plugins()
    realms = next(row for row in rows if row['name'] == 'hermes-realms')
    # Only Realms is relevant here; the scan above is the actual bundled scanner.
    monkeypatch.setattr(web_server, '_get_dashboard_plugins', lambda: [realms])
    sys.modules.pop('hermes_dashboard_plugin_hermes-realms', None)
    dashboard._mount_plugin_api_routes()
    assert 'hermes_dashboard_plugin_hermes-realms' not in sys.modules
    assert not any(getattr(route, 'path', '').startswith('/api/plugins/hermes-realms') for route in app.routes)

    config['plugins']['enabled'] = ['hermes-realms']
    save_config(config)
    dashboard._mount_plugin_api_routes()
    assert 'hermes_dashboard_plugin_hermes-realms' in sys.modules
    app.middleware('http')(web_server._plugin_api_runtime_gate)
    @app.middleware('http')
    async def authenticated(request, call_next):
        request.state.token_authenticated = True
        return await call_next(request)
    with TestClient(app) as client:
        response = client.get('/api/plugins/hermes-realms/realms')
        assert response.status_code == 200, response.text
        assert response.json()['realms'] == []
        # Removing the allow-list entry must revoke already-mounted routes too.
        config['plugins']['enabled'] = []
        save_config(config)
        assert client.get('/api/plugins/hermes-realms/realms').status_code == 404
