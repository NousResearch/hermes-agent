"""A plugin installed and enabled after the web server imported must serve its backend API on the
first request — no restart — and stop serving once disabled."""

import json

import pytest

from hermes_cli import web_server


@pytest.fixture
def client(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    original_routes = list(app.router.routes)
    web_server._dashboard_plugins_cache = None
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    try:
        yield c
    finally:
        app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None


def _install_late_plugin(home, *, enabled: bool) -> None:
    dash = home / "plugins" / "late-probe" / "dashboard"
    dash.mkdir(parents=True)
    (dash / "manifest.json").write_text(json.dumps({"name": "late-probe", "api": "plugin_api.py"}))
    (dash / "plugin_api.py").write_text(
        "from fastapi import APIRouter\nrouter = APIRouter()\n"
        "@router.get('/ping')\nasync def ping():\n    return {'pong': True}\n"
    )
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("plugins", {})["enabled"] = ["late-probe"] if enabled else []
    save_config(cfg)


def test_plugin_enabled_after_startup_serves_without_restart(client):
    from hermes_constants import get_hermes_home

    assert client.get("/api/plugins/late-probe/ping").status_code == 404
    _install_late_plugin(get_hermes_home(), enabled=True)

    first = client.get("/api/plugins/late-probe/ping", follow_redirects=False)
    assert (first.status_code, first.json()) == (200, {"pong": True})
    assert client.get("/api/plugins/late-probe/missing").status_code == 404

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg["plugins"]["enabled"], cfg["plugins"]["disabled"] = [], ["late-probe"]
    save_config(cfg)
    assert client.get("/api/plugins/late-probe/ping").status_code == 404


def test_installed_but_not_enabled_plugin_is_never_imported(client):
    from hermes_constants import get_hermes_home

    _install_late_plugin(get_hermes_home(), enabled=False)
    (get_hermes_home() / "plugins" / "late-probe" / "dashboard" / "plugin_api.py").write_text(
        "raise SystemExit('imported a not-enabled plugin')\n")
    assert client.get("/api/plugins/late-probe/ping").status_code == 404
