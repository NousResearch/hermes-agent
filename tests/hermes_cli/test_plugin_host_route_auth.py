"""Regression for #140499: upstream admission survives credential-free hosting."""
import os
from types import SimpleNamespace

import pytest

from hermes_cli.plugin_host import PluginHost
from tests.hermes_cli import test_plugin_secondary_profile_host as secondary

daemon = secondary.daemon

_API = '''
import os
from fastapi import APIRouter, Request
from hermes_cli.web_deps import late

router = APIRouter()

@router.put("/config")
async def config(request: Request):
    late("_require_token")(request)
    return {"pid": os.getpid(), "saved": True,
            "credentials": [k for k in request.headers if k in
                {"authorization", "cookie", "x-hermes-session-token", "x-hermes-gateway-ticket"}]}
'''


@pytest.mark.platforms("any")
def test_secondary_guarded_mutation_uses_upstream_admission(daemon, monkeypatch):
    monkeypatch.setattr(PluginHost, "asgi_request", secondary._REAL_ASGI_REQUEST)
    secondary._beta_plugin(daemon)
    api = daemon.beta / "plugins" / secondary.NAME / "dashboard" / "plugin_api.py"
    api.write_text(_API, encoding="utf-8")
    url = f"/api/plugins/{secondary.NAME}/config"
    try:
        response = daemon.client.put(url, headers=daemon.beta_ticket())
        assert response.status_code == 200, response.text
        assert response.json()["saved"] is True
        assert response.json()["pid"] != os.getpid()
        assert response.json()["credentials"] == []
        assert daemon.client.put(url, params={"profile": "beta"}).status_code == 401
    finally:
        secondary._shutdown_hosts()


@pytest.mark.platforms("any")
def test_host_admission_does_not_leak_or_come_from_http_input(tmp_path):
    api = tmp_path / "api"
    api.mkdir()
    (api / "plugin_api.py").write_text(_API, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    host = PluginHost(SimpleNamespace(home_path=home, _plugins={}))
    args = ("guard-probe", str(api), "plugin_api.py", "PUT", "/config", "authenticated=true",
            [("X-Hosted-Http-Authenticated", "true")], b'{"authenticated":true}')
    try:
        assert host.asgi_request(*args, authenticated=True)["status"] == 200
        for marker in (False, None, "true", 1):
            assert host.asgi_request(*args, authenticated=marker)["status"] == 401
        assert host.asgi_request(*args)["status"] == 401
        assert host.asgi_request(*args, authenticated=True)["status"] == 200
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_launch_host_preserves_native_admission(daemon, monkeypatch):
    monkeypatch.setattr(PluginHost, "asgi_request", secondary._REAL_ASGI_REQUEST)
    secondary._install(daemon.launch, "launch-guard")
    secondary._plugins(daemon.launch, enabled=["launch-guard"], isolation="host")
    api = daemon.launch / "plugins" / "launch-guard" / "dashboard" / "plugin_api.py"
    api.write_text(_API, encoding="utf-8")
    try:
        response = daemon.client.put("/api/plugins/launch-guard/config", headers=daemon.owner)
        assert response.status_code == 200, response.text
        assert response.json()["pid"] != os.getpid()
    finally:
        secondary._shutdown_hosts()


def test_admission_matches_normal_guard_and_preserves_unguarded_routes():
    from fastapi import FastAPI, HTTPException
    from starlette.requests import Request
    from hermes_cli.plugin_host_http_auth import admitted_for_sensitive_route
    from hermes_cli import web_server

    app = FastAPI()
    scope = {"type": "http", "method": "PUT", "path": "/config", "headers": [], "app": app}
    assert not admitted_for_sensitive_route(Request(scope))
    assert admitted_for_sensitive_route(Request({**scope, "headers": [
        (web_server._SESSION_HEADER_NAME.lower().encode(), web_server._SESSION_TOKEN.encode())]}))
    app.state.auth_required = True
    assert not admitted_for_sensitive_route(Request(scope))
    assert admitted_for_sensitive_route(Request({**scope, "state": {"session": object()}}))
    assert admitted_for_sensitive_route(Request({**scope, "state": {"native_http_principal": object()}}))
    for marker in (None, False, "true", 1):
        with pytest.raises(HTTPException) as error:
            web_server._require_token(Request({**scope, "state": {"hosted_http_authenticated": marker}}))
        assert error.value.status_code == 401
