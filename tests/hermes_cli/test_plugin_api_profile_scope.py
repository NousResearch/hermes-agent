"""Regression for #120310: plugin API dispatch must bind the selected profile."""

import json
from pathlib import Path

from starlette.testclient import TestClient

from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
from hermes_cli import web_server
import hermes_cli.web_server_dashboard as dashboard
import tui_gateway.launch_profile_policy as launch_policy


def test_plugin_api_secret_scope_across_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_cli import profiles

    launch_home = tmp_path / "home"
    other_home = launch_home / "profiles" / "other"
    for home, value in ((launch_home, "launch-value"), (other_home, "other-value")):
        home.mkdir(parents=True)
        (home / ".env").write_text(f"PLUGIN_SCOPE_TEST_KEY={value}\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - scope-test\n", encoding="utf-8"
        )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: launch_home / "profiles")

    plugin_dir = launch_home / "plugins" / "scope-test" / "dashboard"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.json").write_text(json.dumps({
        "name": "scope-test", "api": "plugin_api.py", "entry": "dist/index.js"
    }), encoding="utf-8")
    (plugin_dir / "plugin_api.py").write_text(
        "from fastapi import APIRouter\n"
        "from agent.secret_scope import get_secret\n"
        "from hermes_constants import get_hermes_home\n"
        "router = APIRouter()\n"
        "@router.get('/sync')\n"
        "def sync():\n"
        "    return {'secret': get_secret('PLUGIN_SCOPE_TEST_KEY'), 'home': str(get_hermes_home())}\n"
        "@router.get('/async')\n"
        "async def async_route():\n"
        "    return {'secret': get_secret('PLUGIN_SCOPE_TEST_KEY'), 'home': str(get_hermes_home())}\n",
        encoding="utf-8",
    )

    original_routes = list(web_server.app.router.routes)
    prior_multiplex = is_multiplex_active()
    prior_snapshot = launch_policy._snapshot
    web_server._dashboard_plugins_cache = None
    try:
        dashboard._mount_plugin_api_routes()
        mounted = [route for route in web_server.app.router.routes if route not in original_routes]
        assert mounted
        for route in mounted:
            web_server.app.router.routes.remove(route)
        web_server.app.router.routes[:0] = mounted  # ahead of the SPA catch-all
        launch_policy.activate_multi_profile_hosting()
        assert get_secret_raises_unscoped()
        with TestClient(web_server.app) as client:
            for kind in ("sync", "async"):
                path = f"/api/plugins/scope-test/{kind}"
                assert client.get(path).status_code == 401
                headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
                for profile, home, secret in (
                    (None, launch_home, "launch-value"),
                    ("other", other_home, "other-value"),
                    (None, launch_home, "launch-value"),
                ):
                    response = client.get(path, params={"profile": profile} if profile else {}, headers=headers)
                    assert response.status_code == 200, response.text
                    assert response.json() == {"secret": secret, "home": str(home)}
                assert client.get(path, params={"profile": "missing"}, headers=headers).status_code == 404
            for config in (
                "plugins:\n  enabled:\n    - scope-test\n  disabled:\n    - scope-test\n",
                "plugins:\n  enabled: []\n",
            ):
                (other_home / "config.yaml").write_text(config, encoding="utf-8")
                for kind in ("sync", "async"):
                    path = f"/api/plugins/scope-test/{kind}"
                    assert client.get(path, params={"profile": "other"}, headers=headers).status_code == 404
                    response = client.get(path, headers=headers)
                    assert response.status_code == 200, response.text
                    assert response.json() == {"secret": "launch-value", "home": str(launch_home)}
    finally:
        web_server.app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None
        launch_policy._snapshot = prior_snapshot
        set_multiplex_active(prior_multiplex)


def get_secret_raises_unscoped():
    from agent.secret_scope import UnscopedSecretError
    try:
        get_secret("PLUGIN_SCOPE_TEST_KEY")
    except UnscopedSecretError:
        return True
    return False
