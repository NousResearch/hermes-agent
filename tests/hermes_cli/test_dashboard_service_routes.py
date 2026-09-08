from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_routers import service_proxy
from plugins.platforms.a2a.adapter import A2ARequestHandler


class _EchoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return None

    def _reply(self, body: bytes = b"") -> None:
        payload = json.dumps({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "cookie": self.headers.get("Cookie"),
            "dashboard_token": self.headers.get("X-Hermes-Dashboard-Token"),
            "forwarded_host": self.headers.get("X-Forwarded-Host"),
            "forwarded_proto": self.headers.get("X-Forwarded-Proto"),
            "forwarded_prefix": self.headers.get("X-Forwarded-Prefix"),
            "body": body.decode("utf-8"),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "upstream=must-not-escape")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        self._reply()

    def do_POST(self):  # noqa: N802
        self._reply(self.rfile.read(int(self.headers.get("Content-Length", "0"))))


@pytest.fixture
def echo_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def public_dashboard():
    previous = {
        key: getattr(web_server.app.state, key, None)
        for key in ("bound_host", "bound_port", "auth_required", "trusted_public_hosts")
    }
    web_server.app.state.bound_host = "agent.example.com"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    web_server.app.state.trusted_public_hosts = frozenset({"agent.example.com"})
    try:
        yield TestClient(web_server.app, base_url="https://agent.example.com")
    finally:
        for key, value in previous.items():
            setattr(web_server.app.state, key, value)


def _route_config(name: str, port: int) -> dict:
    return {
        "gateway": {"platforms": {
            name: {"enabled": True, "extra": {"public_route": True, "port": port}},
        }}
    }


def test_disabled_service_route_returns_json_instead_of_oauth_redirect(
    public_dashboard, monkeypatch
):
    monkeypatch.setattr(service_proxy, "load_config", lambda: {})

    response = public_dashboard.get("/a2a/.well-known/agent-card.json", follow_redirects=False)

    assert response.status_code == 404
    assert response.json() == {"error": "service route is not enabled"}


def test_service_route_prefix_does_not_open_similar_dashboard_path(
    public_dashboard, monkeypatch
):
    monkeypatch.setattr(service_proxy, "load_config", lambda: {})

    response = public_dashboard.get("/a2attack", follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["location"]


def test_a2a_public_route_requires_its_own_service_credential(
    public_dashboard, echo_server, monkeypatch
):
    monkeypatch.setattr(service_proxy, "load_config", lambda: _route_config("a2a", echo_server))
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

    response = public_dashboard.get("/a2a/.well-known/agent-card.json")

    assert response.status_code == 404
    assert response.json() == {"error": "service route is not enabled"}


def test_a2a_route_preserves_service_auth_and_advertised_prefix(
    public_dashboard, echo_server, monkeypatch
):
    monkeypatch.setattr(service_proxy, "load_config", lambda: _route_config("a2a", echo_server))
    monkeypatch.setenv("A2A_PEER_TOKENS", "nova:test-token")
    monkeypatch.delenv("A2A_PORT", raising=False)

    response = public_dashboard.post(
        "/a2a/?mode=send",
        headers={"Authorization": "Bearer test-token"},
        json={"jsonrpc": "2.0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == "/?mode=send"
    assert payload["authorization"] == "Bearer test-token"
    assert payload["forwarded_host"] == "agent.example.com"
    assert payload["forwarded_proto"] == "https"
    assert payload["forwarded_prefix"] == "/a2a"
    assert json.loads(payload["body"]) == {"jsonrpc": "2.0"}
    assert "set-cookie" not in response.headers


def test_api_route_strips_dashboard_credentials_and_keeps_api_key(
    public_dashboard, echo_server, monkeypatch
):
    monkeypatch.setattr(
        service_proxy, "load_config", lambda: _route_config("api_server", echo_server))
    monkeypatch.delenv("API_SERVER_PORT", raising=False)

    response = public_dashboard.get(
        "/hermes-api/api/sessions?limit=1",
        headers={
            "Authorization": "Bearer api-key",
            "Cookie": "hermes_session_at=private",
            "X-Hermes-Dashboard-Token": "private",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == "/api/sessions?limit=1"
    assert payload["authorization"] == "Bearer api-key"
    assert payload["cookie"] is None
    assert payload["dashboard_token"] is None


def test_a2a_public_url_includes_forwarded_prefix(monkeypatch):
    monkeypatch.delenv("A2A_PUBLIC_URL", raising=False)
    handler = object.__new__(A2ARequestHandler)
    handler.headers = {
        "X-Forwarded-Host": "agent.example.com",
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Prefix": "/a2a",
    }

    assert handler._request_public_url() == "https://agent.example.com/a2a/"
