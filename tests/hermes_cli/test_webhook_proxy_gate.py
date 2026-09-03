"""Tests for the platform-webhook proxy gate.

Regression context: on a single-public-port deployment (e.g. a Fly.io app
that only exposes the dashboard's port), a platform adapter's own webhook
server (LINE on 8646, Feishu on its own port, etc.) is never reachable from
the internet directly. Before this gate, those requests hit THIS app's
global dashboard-auth middleware instead — which has no route for
``/line/webhook`` and, under the OAuth gate, bounced every request to
``/login`` (or 401'd it), silently dropping the webhook. See
docs in ``hermes-config/skills/realife/realife-line-management`` (private
repo) for the live incident this reproduces.

These tests mock the internal HTTP call (``httpx.AsyncClient.request``) so
they never depend on a real listener on the platform's port.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app():
    """Same shape as test_dashboard_auth_middleware.gated_app: OAuth-gated,
    non-loopback bind, one stub provider registered."""
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_unmatched_path_still_hits_the_auth_gate(gated_app):
    """Regression guard: only the allowlisted platform paths bypass auth —
    everything else must keep redirecting to /login exactly as before."""
    r = gated_app.post("/not/a/webhook", json={}, follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("location", "")


@pytest.mark.parametrize(
    "path,platform",
    [
        ("/line/webhook", "line"),
        ("/line/webhook/health", "line"),
        ("/feishu/webhook", "feishu"),
        ("/wecom/callback", "wecom_callback"),
    ],
)
def test_platform_webhook_bypasses_the_auth_gate_and_proxies(
    monkeypatch, gated_app, path, platform
):
    """A known platform-webhook path skips the cookie gate entirely and is
    forwarded to the adapter's own port — no session, no /login bounce."""
    captured: dict = {}

    async def fake_request(self, method, url, *, content=None, headers=None):
        captured["method"] = method
        captured["url"] = str(url)
        captured["content"] = content
        captured["headers"] = dict(headers or {})
        return httpx.Response(
            200,
            content=b"OK",
            headers={"content-type": "text/plain", "x-upstream": "adapter"},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    r = gated_app.post(
        path,
        content=b'{"events": []}',
        headers={
            "X-Line-Signature": "deadbeef",
            "Content-Type": "application/json",
            "Cookie": "should=not-leak-to-upstream-unmodified",
        },
        follow_redirects=False,
    )

    assert r.status_code == 200, r.text
    assert r.text == "OK"
    assert r.headers.get("x-upstream") == "adapter"

    assert captured["method"] == "POST"
    assert captured["url"].startswith(f"http://127.0.0.1:")
    assert captured["url"].endswith(path)
    assert captured["content"] == b'{"events": []}'
    # The adapter's own signature check must still see the header.
    assert captured["headers"].get("x-line-signature") == "deadbeef"
    # Hop-by-hop / rewritten headers must not leak through verbatim.
    assert "host" not in captured["headers"]
    assert "content-length" not in captured["headers"]


def test_line_webhook_forwards_query_string(monkeypatch, gated_app):
    captured: dict = {}

    async def fake_request(self, method, url, *, content=None, headers=None):
        captured["url"] = str(url)
        return httpx.Response(200, content=b"ok", request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    gated_app.get("/line/webhook/health?probe=1", follow_redirects=False)
    assert captured["url"].endswith("/line/webhook/health?probe=1")


def test_unreachable_adapter_returns_502_not_a_login_bounce(monkeypatch, gated_app):
    """If the adapter's port isn't listening (crashed / still starting), the
    caller must see a clean 502 — never a 401/302 that looks like an auth
    failure to LINE's retry logic."""

    async def fake_request(self, method, url, *, content=None, headers=None):
        raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    r = gated_app.post("/line/webhook", json={}, follow_redirects=False)
    assert r.status_code == 502
    assert r.json()["detail"] == "line webhook adapter unreachable"


def test_resolve_webhook_proxy_port_prefers_config_over_default(monkeypatch):
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"gateway": {"platforms": {"line": {"port": 19999}}}},
    )
    assert web_server._resolve_webhook_proxy_port("line") == 19999


def test_resolve_webhook_proxy_port_falls_back_to_default_on_bad_config(monkeypatch):
    monkeypatch.setattr(web_server, "load_config", lambda: (_ for _ in ()).throw(OSError()))
    assert web_server._resolve_webhook_proxy_port("line") == 8646
