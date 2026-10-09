"""Native Desktop's cookie-session writes through the real dashboard auth gate.

Regression from the PR #93508 review: installed Desktop builds reach password-gated
gateways (and Cloud agents signed in through the silent cookie flow) with Electron
``net.request``, which sends the session cookie but never an Origin. Every write,
ws-ticket mints included, got 403 and Desktop read that as an expired session. The
header shapes below are what Electron 40.10.2 sends on the wire. A browser page's
write always carries Origin, so page-shaped requests must stay refused.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import plugins.dashboard_auth.basic as basic_plugin
from hermes_cli import web_server
from hermes_cli.config import read_raw_config
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit

HTTPS_REMOTE = "https://agent.example.com"
PLAIN_HTTP_REMOTE = "http://192.168.50.220:9119"
# Electron net.request with useSessionCookies: Chromium marks secure targets with
# Sec-Fetch-Site none and sends no Fetch Metadata at all to a plain-http remote.
NATIVE_HEADERS = {
    HTTPS_REMOTE: {"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "empty"},
    PLAIN_HTTP_REMOTE: {},
}
CONFIG_WRITE = {"config": {"display": {"skin": "mono"}}}


@pytest.fixture(scope="module")
def password_hash():
    return basic_plugin.hash_password("hunter2")


@pytest.fixture
def password_gate(password_hash, monkeypatch):
    """An all-interfaces gateway gated by the bundled username/password provider."""
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9119), ("auth_required", True),
                       ("trusted_public_hosts", frozenset())):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    clear_providers()
    register_provider(basic_plugin.BasicAuthProvider(
        username="admin", password_hash=password_hash, secret=b"s" * 32))
    # The login limiter counts successful sign-ins too, and every test signs in once.
    _reset_password_rate_limit()
    yield
    clear_providers()


def _signed_in(base_url: str) -> TestClient:
    client = TestClient(web_server.app, base_url=base_url)
    login = client.post("/auth/password-login",
                        json={"provider": "basic", "username": "admin", "password": "hunter2"})
    assert login.status_code == 200, login.text
    return client


@pytest.mark.parametrize("base_url", [HTTPS_REMOTE, PLAIN_HTTP_REMOTE])
def test_native_cookie_session_mints_tickets_writes_config_and_refreshes(password_gate, base_url):
    client = _signed_in(base_url)
    native = NATIVE_HEADERS[base_url]

    ticket = client.post("/api/auth/ws-ticket", headers=native)
    assert ticket.status_code == 200, ticket.text
    assert ticket.json()["ticket"]

    saved = client.put("/api/config", headers=native, json=CONFIG_WRITE)
    assert saved.status_code == 200, saved.text
    assert read_raw_config()["display"]["skin"] == "mono"

    # The access cookie lapses first; the next write must rotate the session, not 403.
    for cookie in list(client.cookies.jar):
        if cookie.name.endswith(SESSION_AT_COOKIE):
            client.cookies.delete(cookie.name, domain=cookie.domain, path=cookie.path)
    refreshed = client.post("/api/auth/ws-ticket", headers=native)
    assert refreshed.status_code == 200, refreshed.text
    assert any(name.endswith(SESSION_AT_COOKIE) for name in client.cookies)


@pytest.mark.parametrize("base_url,headers", [
    pytest.param(HTTPS_REMOTE, {"Origin": "https://evil.example", "Sec-Fetch-Site": "none"},
                 id="none-with-foreign-origin"),
    pytest.param(HTTPS_REMOTE, {"Origin": "https://blog.example.com", "Sec-Fetch-Site": "same-site"},
                 id="same-site-sibling"),
    pytest.param(HTTPS_REMOTE, {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
                 id="cross-site"),
    pytest.param(HTTPS_REMOTE, {"Origin": "null", "Sec-Fetch-Site": "same-site"},
                 id="no-referrer-sibling-form"),
    pytest.param(HTTPS_REMOTE, {"Sec-Fetch-Site": "same-site"}, id="same-site-without-origin"),
    pytest.param(HTTPS_REMOTE, {"Sec-Fetch-Site": "cross-site"}, id="cross-site-without-origin"),
    pytest.param(PLAIN_HTTP_REMOTE, {"Origin": "http://192.168.50.220:8080"}, id="http-sibling-port"),
    pytest.param(PLAIN_HTTP_REMOTE, {"Origin": "null"}, id="http-no-referrer-form"),
])
def test_page_shaped_cookie_writes_stay_refused(password_gate, base_url, headers):
    client = _signed_in(base_url)

    ticket = client.post("/api/auth/ws-ticket", headers=headers)
    assert ticket.status_code == 403, ticket.text

    saved = client.put("/api/config", headers=headers, json=CONFIG_WRITE)
    assert saved.status_code == 403, saved.text
    assert read_raw_config().get("display", {}).get("skin") != "mono"
