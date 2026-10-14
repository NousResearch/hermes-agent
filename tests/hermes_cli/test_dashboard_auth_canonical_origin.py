"""Loopback OAuth entry canonicalizes before allocating host-only PKCE cookies."""
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import routes
from plugins.dashboard_auth.nous import NousDashboardAuthProvider

PUBLIC = "https://dashboard.example.test:8443"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
@pytest.mark.parametrize("path", [
    "/login?next=%2Fsessions",
    "/auth/login?provider=nous&next=%2Fsessions",
    "/auth/native/authorize?provider=nous&code_challenge=abc&code_challenge_method=S256&redirect_uri=http%3A%2F%2F127.0.0.1%3A45678%2Fcb&state=desktop",
])
def test_loopback_entry_moves_before_setting_cookie(monkeypatch, host, path):
    monkeypatch.setattr(routes._prefix_mod, "resolve_public_url", lambda: PUBLIC)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app, base_url="http://127.0.0.1:9119", follow_redirects=False)
    response = client.get(path, headers={"Host": f"{host}:9119"})
    assert response.status_code == 302
    assert response.headers["location"] == PUBLIC + path
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"].startswith("no-store")


def test_canonical_origin_sets_pkce_for_same_origin_callback(monkeypatch):
    monkeypatch.setattr(routes._prefix_mod, "resolve_public_url", lambda: PUBLIC)
    provider = NousDashboardAuthProvider(client_id="agent:test", portal_url="https://portal.example.test")
    monkeypatch.setattr(routes, "get_provider", lambda name: provider)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app, base_url=PUBLIC, follow_redirects=False)
    login = client.get("/login")
    assert login.status_code == 200
    response = client.get("/auth/login?provider=nous")
    assert response.status_code == 302
    target = urlparse(response.headers["location"])
    assert target.netloc == "portal.example.test"
    assert parse_qs(target.query)["redirect_uri"] == [PUBLIC + "/auth/callback"]
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("__Host-hermes_session_pkce=")
    assert "Secure" in cookie and "HttpOnly" in cookie


def test_expired_browser_callback_restarts_without_redeeming_code(monkeypatch):
    monkeypatch.setattr(routes._prefix_mod, "resolve_public_url", lambda: PUBLIC)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app, base_url=PUBLIC, follow_redirects=False)
    callback = "/auth/callback?code=expired-code&state=expired-state&next=https://evil.example"
    browser = client.get(callback, headers={"Accept": "text/html"})
    assert browser.status_code == 302
    assert browser.headers["location"] == PUBLIC + "/login?reason=pkce_expired"
    assert "set-cookie" not in browser.headers
    api = client.get(callback, headers={"Accept": "application/json"})
    assert api.status_code == 400
    assert api.json()["detail"] == "Missing PKCE state cookie"
