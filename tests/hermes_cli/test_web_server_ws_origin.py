"""Authenticated gateway upgrades reject malformed and cross-site web origins."""

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def remote_dashboard(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "bound_host", "100.64.0.10", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "trusted_public_hosts", frozenset(), raising=False)
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    clear_providers()
    _reset_for_tests()
    register_provider(StubAuthProvider())
    client = TestClient(web_server.app, base_url="http://100.64.0.10:9119")
    # TestClient's relative WS URLs use testserver rather than its HTTP base_url.
    client.headers["Host"] = "100.64.0.10:9119"
    try:
        login = client.get("/auth/login?provider=stub", follow_redirects=False)
        assert login.status_code == 302
        state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
        callback = client.get(
            "/auth/callback", params={"code": "stub_code", "state": state}, follow_redirects=False
        )
        assert callback.status_code == 302
        yield client
    finally:
        client.close()
        clear_providers()
        _reset_for_tests()


def _ticket(client):
    response = client.post("/api/auth/ws-ticket")
    assert response.status_code == 200
    return response.json()["ticket"]


def test_authenticated_allowed_origin_completes_gateway_handshake(remote_dashboard):
    assert remote_dashboard.get("/api/status").status_code == 200
    for origin in (
        "http://100.64.0.10:9119",
        "file://",
        "null",
        "app://hermes",
    ):
        ticket = _ticket(remote_dashboard)
        with remote_dashboard.websocket_connect(
            f"/api/ws?ticket={ticket}", headers={"Origin": origin}
        ) as connection:
            assert connection.receive_json()["params"]["type"] == "gateway.ready"
            connection.send_json({"jsonrpc": "2.0", "id": "origin-ping", "method": "gateway.ping"})
            assert connection.receive_json() == {
                "jsonrpc": "2.0", "id": "origin-ping", "result": {"ok": True}
            }
        # Reconnects must mint a fresh ticket, never reuse an accepted one.
        with pytest.raises(WebSocketDisconnect) as rejected:
            with remote_dashboard.websocket_connect(
                f"/api/ws?ticket={ticket}", headers={"Origin": origin}
            ):
                pass
        assert rejected.value.code == 4401


def test_gateway_upgrade_preserves_auth_host_and_origin_boundaries(remote_dashboard, monkeypatch):
    for headers in (
        {"Origin": "https://evil.test"},
        {"Origin": "http://127.0.0.1:52133"},
        {"Origin": "http://localhost:61307"},
        {"Origin": "http://[::1]:52133"},
        {"Origin": "http://localhost.evil.test:52133"},
        {"Origin": "http://evil.test@localhost:52133"},
        {"Origin": "http://localhost:notaport"},
        {"Origin": "http://[::1].evil.test:52133"},
        {"Origin": "http://[::1"},
        {"Origin": "https://[not-an-ip]:52133"},
        {"Origin": "http://testclient:52133"},
        {"Host": "evil.test:9119", "Origin": "http://127.0.0.1:52133"},
    ):
        with pytest.raises(WebSocketDisconnect) as rejected:
            with remote_dashboard.websocket_connect(f"/api/ws?ticket={_ticket(remote_dashboard)}", headers=headers):
                pass
        assert rejected.value.code == 4403

    # Even an authenticated HTTP cookie session is not a WS-upgrade credential.
    for query in ("", "?ticket=invalid", f"?token={web_server._SESSION_TOKEN}"):
        with pytest.raises(WebSocketDisconnect) as rejected:
            with remote_dashboard.websocket_connect(
                f"/api/ws{query}", headers={"Origin": "null"}
            ):
                pass
        assert rejected.value.code == 4401

    # Public --insecure binds keep their existing cross-origin policy.
    monkeypatch.setattr(web_server.app.state, "auth_required", False)
    with pytest.raises(WebSocketDisconnect) as rejected:
        with remote_dashboard.websocket_connect(
            f"/api/ws?token={web_server._SESSION_TOKEN}", headers={"Origin": "http://127.0.0.1:52133"}
        ):
            pass
    assert rejected.value.code == 4403
