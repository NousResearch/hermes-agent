"""A sign-in begun on this machine's own address while the listener declares a public URL.

The provider returns to ``<public>/auth/callback``, and the state cookie set where sign-in started
never reaches it from ``127.0.0.1``: the browser's sign-in, and a native app's, failed with a missing
PKCE state cookie. Such a start goes to the same path and query on the public URL first. A request a
proxy forwarded (Tailscale Serve, a reverse proxy) is never moved.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers

PUBLIC = "https://box.tailnet.ts.net"
LOOPBACK = "http://127.0.0.1:9120"
NATIVE_QUERY = ("provider=nous&code_challenge=" + "a" * 43 + "&code_challenge_method=S256"
                "&redirect_uri=http%3A%2F%2F127.0.0.1%3A50123%2Fhermes-browser-callback&state=s1")


@pytest.fixture
def public_webapp(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "127.0.0.1"), ("bound_port", 9120), ("auth_required", True),
                       ("trusted_public_hosts", frozenset({"box.tailnet.ts.net"})),
                       ("ui_surface", "webapp"), ("listener_public_url", PUBLIC)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    clear_providers()
    yield
    clear_providers()


def _client(base_url: str = LOOPBACK, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(web_server.app, base_url=base_url, client=(peer, 50000), follow_redirects=False)


@pytest.mark.parametrize("path", ["/auth/login?provider=nous&next=%2Fsessions", f"/auth/native/authorize?{NATIVE_QUERY}"])
def test_a_sign_in_started_on_loopback_moves_to_the_public_url_first(public_webapp, path):
    response = _client().get(path)
    assert response.status_code == 302
    assert response.headers["location"] == f"{PUBLIC}{path}"
    assert not any("pkce" in name for name in response.cookies), "no state cookie is set on 127.0.0.1"


@pytest.mark.parametrize("client_kwargs,headers", [
    pytest.param({"base_url": PUBLIC}, {}, id="https-public-host"),
    pytest.param({}, {"X-Forwarded-For": "100.101.102.103"}, id="forwarded-by-a-proxy"),
    pytest.param({}, {"X-Forwarded-Host": "box.tailnet.ts.net"}, id="forwarded-host"),
    pytest.param({"peer": "192.168.1.20"}, {}, id="a-lan-peer"),
])
def test_a_forwarded_or_public_request_is_never_moved(public_webapp, client_kwargs, headers):
    response = _client(**client_kwargs).get("/auth/login?provider=nous", headers=headers)
    assert response.status_code != 302 or not response.headers["location"].startswith(PUBLIC)


def test_without_a_public_url_nothing_moves(public_webapp, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "listener_public_url", "", raising=False)
    response = _client().get("/auth/login?provider=nous")
    assert response.status_code != 302 or not response.headers["location"].startswith("https://")
