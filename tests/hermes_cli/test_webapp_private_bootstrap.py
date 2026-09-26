"""Webapp's public HTML must not grant the privileged RPC/host-file credential."""
import json
import re
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.mark.parametrize("surface,gated,headless", [
    ("webapp", False, False), ("webapp", True, False), ("dashboard", False, False), ("serve", False, True),
])
def test_bootstrap_token_visibility(monkeypatch, tmp_path, surface, gated, headless):
    from hermes_cli import web_server as server
    from hermes_cli import web_server_profiles
    from hermes_cli.web_server_dashboard import mount_spa

    serving_profile = "profile</script>"
    monkeypatch.setattr(web_server_profiles, "serving_profile_name", lambda: serving_profile)
    (tmp_path / "index.html").write_text("<html><head></head><body>Webapp</body></html>")
    application = FastAPI()
    application.state.ui_surface = surface
    application.state.auth_required = gated
    monkeypatch.setattr(server, "app", application)
    monkeypatch.setattr(server, "WEB_DIST", tmp_path)
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1" if headless else "0")
    mount_spa(application)
    client = TestClient(application)
    for path in ("/", "/chat", "/settings", "/index.html"):
        response = client.get(path)
        should_inject = not gated and surface != "webapp" and path != "/index.html" and (not headless or path == "/")
        assert (server._SESSION_TOKEN in response.text) is should_inject
        assert server._SESSION_TOKEN not in str(response.headers)
        if not headless and (path != "/index.html" or surface == "webapp"):
            profile_js = json.dumps(serving_profile).replace("</", "<\\/")
            assert f"window.__HERMES_DASHBOARD_PROFILE__={profile_js};" in response.text
        if surface == "webapp" and not headless:
            assert 'window.__HERMES_UI_SURFACE__="webapp"' in response.text


@pytest.mark.parametrize("operation", ["shell", "reattach", "close", "rpc", "file"])
def test_html_cannot_authorize_privileged_sibling_routes(monkeypatch, tmp_path, operation):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "index.html").write_text("<html><head></head><body>Webapp</body></html>")
    # Real middleware, routers and auth gates; don't start the unrelated lifespan.
    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "testclient", raising=False)
    spa = FastAPI()
    spa.state.ui_surface = "webapp"
    spa.state.auth_required = False
    monkeypatch.setattr(server, "WEB_DIST", tmp_path)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(spa)
    html = TestClient(spa).get("/").text
    match = re.search(r'__HERMES_SESSION_TOKEN__\s*=\s*("[^"]*")', html)
    exposed = json.loads(match[1]) if match else ""
    client = TestClient(server.app, base_url="http://testclient")
    if operation == "file":
        response = client.get("/api/fs/read-text", params={"path": str(tmp_path / "index.html")},
                              headers={"X-Hermes-Session-Token": exposed})
        assert response.status_code == 401
    else:
        query = {"token": exposed}
        path = "/api/ws" if operation == "rpc" else "/api/host-terminal"
        if operation in {"reattach", "close"}:
            query["attach"] = "known-shell"
        if operation == "close":
            query["action"] = "close"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(path + "?" + urlencode(query), headers={"host": "testclient"}):
                pass
        assert exc.value.code == 4401
