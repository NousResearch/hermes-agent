"""Regression for #140499: upstream admission survives credential-free hosting."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth.native_http import HEADER
from hermes_cli.plugin_host import PluginHost
from tests.hermes_cli import test_plugin_secondary_profile_host as secondary

daemon = secondary.daemon

_API = '''
import os
from fastapi import APIRouter, Request
from hermes_cli.web_deps import late
from hermes_constants import get_hermes_home

router = APIRouter()

@router.put("/config")
async def config(request: Request):
    late("_require_token")(request)
    return {"pid": os.getpid(), "saved": True,
            "home": str(get_hermes_home()),
            "native": getattr(request.state, "hosted_native_http_authenticated", False) is True,
            "credentials": [k for k in request.headers if k in
                {"authorization", "cookie", "x-hermes-session-token", "x-hermes-gateway-ticket"}]}
'''


@pytest.mark.platforms("any")
@pytest.mark.parametrize("profile", ["default", "beta"])
def test_hosted_route_preserves_native_admission(daemon, monkeypatch, profile):
    monkeypatch.setattr(PluginHost, "asgi_request", secondary._REAL_ASGI_REQUEST)
    if profile == "beta":
        secondary._beta_plugin(daemon)
        home, name, other = daemon.beta, secondary.NAME, "default"
    else:
        home, name, other = daemon.launch, "launch-guard", "beta"
        secondary._install(home, name)
        secondary._plugins(home, enabled=[name], isolation="host")
    api = home / "plugins" / name / "dashboard" / "plugin_api.py"
    api.write_text(_API, encoding="utf-8")
    url = f"/api/plugins/{name}/config"
    store = web_server.app.state.gateway_runner.session_ticket_store

    def ticket():
        return {HEADER: store.mint(profile_id=str(home), subject="desktop", purpose="native-http")}

    try:
        grant = ticket()
        response = daemon.client.put(url, headers=grant)
        assert response.status_code == 200, response.text
        native = response.json()
        assert native["saved"] is True and native["native"] is True
        assert native["pid"] != os.getpid()
        assert Path(native["home"]).resolve() == home.resolve()
        assert native["credentials"] == []

        assert daemon.client.put(url, headers=grant).status_code == 401
        assert daemon.client.put(url, headers={**ticket(), "Origin": ""}).status_code == 401
        assert daemon.client.put(url, params={"profile": other}, headers=ticket()).status_code == 403

        params = {"profile": profile} if profile == "beta" else {}
        spoof = {"hosted_native_http_authenticated": "true"}
        assert daemon.client.put(url, params=params, headers=spoof).status_code == 401
        browser = daemon.client.put(url, params=params, headers={
            **daemon.owner, **spoof, "Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"})
        assert browser.status_code == 200, browser.text
        assert browser.json()["pid"] == native["pid"]
        assert browser.json()["saved"] is True and browser.json()["native"] is False
        if profile == "beta":
            assert browser.json()["credentials"] == []
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
            [("X-Hosted-Http-Authenticated", "true"), ("hosted_native_http_authenticated", "true")],
            b'{"authenticated":true,"native_authenticated":true}')
    try:
        admitted = host.asgi_request(*args, authenticated=True, native_authenticated=True)
        assert admitted["status"] == 200
        assert json.loads(admitted["body"])["native"] is True
        for marker in (False, None, "true", 1):
            assert host.asgi_request(*args, authenticated=marker, native_authenticated=True)["status"] == 401
            admitted = host.asgi_request(*args, authenticated=True, native_authenticated=marker)
            assert admitted["status"] == 200
            assert json.loads(admitted["body"])["native"] is False
        assert host.asgi_request(*args)["status"] == 401
        admitted = host.asgi_request(*args, authenticated=True)
        assert admitted["status"] == 200
        assert json.loads(admitted["body"])["native"] is False
    finally:
        host.shutdown()


def test_native_admission_requires_the_worker_home(tmp_path):
    from starlette.requests import Request
    from hermes_cli.plugin_host_http_auth import admitted_for_native_http

    home, other = tmp_path / "home", tmp_path / "other"
    grant = {"purpose": "native-http", "profile_id": str(home)}
    scope = {"type": "http", "headers": [], "state": {"native_http_principal": grant}}
    assert admitted_for_native_http(Request(scope), home)
    assert not admitted_for_native_http(Request(scope), other)
    assert not admitted_for_native_http(Request({**scope, "headers": [(b"origin", b"")]}), home)
    for invalid in (None, {}, {**grant, "purpose": "native-profile-registry"}):
        assert not admitted_for_native_http(Request({**scope, "state": {"native_http_principal": invalid}}), home)


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
