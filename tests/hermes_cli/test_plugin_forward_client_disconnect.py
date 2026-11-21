"""Regression for #155514: disconnected plugin uploads do not reach the host."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from hermes_cli import plugin_host_http_auth, plugins
from hermes_cli import web_server_dashboard as dashboard
from hermes_cli import web_server_plugin_profiles as profiles


def _scope():
    path = "/api/plugins/example/upload"
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
    }


async def _disconnect():
    return {"type": "http.disconnect"}


@pytest.fixture
def secondary_host(monkeypatch, tmp_path):
    host = Mock()
    monkeypatch.setattr(profiles, "_secondary_hosted_plugin", lambda name: {
        "_dir": tmp_path, "_api_file": "api.py",
    })
    monkeypatch.setattr(profiles, "secondary_profile_home", lambda: tmp_path)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: Mock(_plugin_host=lambda: host))
    monkeypatch.setattr(plugin_host_http_auth, "admitted_for_sensitive_route", lambda request: False)

    async def run_hosted(home, name, call):
        call(1.0)
        return Response(status_code=200)

    runner = AsyncMock(side_effect=run_hosted)
    monkeypatch.setattr(profiles, "run_hosted_request", runner)
    return host, runner


@pytest.mark.asyncio
async def test_secondary_disconnect_does_not_call_host(secondary_host):
    host, runner = secondary_host
    response = await profiles.forward_secondary_plugin_request(Request(_scope(), _disconnect), "example")
    assert response.status_code == 499
    runner.assert_not_awaited()
    host.asgi_request.assert_not_called()


@pytest.mark.asyncio
async def test_dashboard_disconnect_does_not_call_host(monkeypatch, tmp_path):
    host = Mock(home=tmp_path)
    monkeypatch.setattr(dashboard, "_hosted_plugin_for_request", lambda name: (host, str(tmp_path), "api.py"))
    app = FastAPI()
    app.dependency_overrides[dashboard._plugin_route_secret_scope] = lambda: None
    dashboard._mount_hosted_plugin_api(app, {"name": "example"}, "api.py")
    messages = []

    async def send(message):
        messages.append(message)

    await app(_scope(), _disconnect, send)
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 499
    host.asgi_request.assert_not_called()


@pytest.mark.asyncio
async def test_secondary_normal_body_reaches_host(secondary_host):
    host, runner = secondary_host
    payload = b"upload\x00content"

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    response = await profiles.forward_secondary_plugin_request(Request(_scope(), receive), "example")
    assert response.status_code == 200
    runner.assert_awaited_once()
    assert host.asgi_request.call_args.args[7] == payload
