"""Operator sign-in notice (env ``HERMES_DASHBOARD_LOGIN_NOTICE``) on the dashboard's sign-in pages.

Set, the notice is shown (escaped, URLs as links) under the subtitle of ``/login`` and of the native provider
chooser, and the gate never skips those pages: no auto-SSO bounce for a browser, and the native authorize route
renders its chooser even for a single provider. Unset, nothing changes.
"""
from __future__ import annotations

import base64
import hashlib
import html

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SSO_ATTEMPT_COOKIE
from hermes_cli.dashboard_auth.login_page import render_login_html
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

PAGE = "https://example.com/privacy"
NOTICE = ("Example Corp's assistant for the ops team. Sign-in is limited to the ops group. "
          "What you send it is kept 30 days. How your data is used: " + PAGE)


@pytest.fixture
def gated():
    clear_providers()
    register_provider(StubAuthProvider())
    state = web_server.app.state
    prev = (getattr(state, "bound_host", None), getattr(state, "bound_port", None), getattr(state, "auth_required", None))
    state.bound_host, state.bound_port, state.auth_required = "fly-app.fly.dev", 443, True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev", follow_redirects=False)
    yield client
    clear_providers()
    state.bound_host, state.bound_port, state.auth_required = prev


def _pkce():
    verifier = base64.urlsafe_b64encode(b"desktop-verifier-secret-material-0123456789abcd").rstrip(b"=").decode()
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def _native(client, **extra):
    return client.get("/auth/native/authorize", params={
        "code_challenge": _pkce(), "code_challenge_method": "S256",
        "redirect_uri": "http://127.0.0.1:53999/cb", "state": "s", **extra})


def test_no_notice_by_default(gated, monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_LOGIN_NOTICE", raising=False)
    assert '<p class="notice">' not in render_login_html()


def test_the_notice_is_escaped_and_its_url_is_a_link(gated, monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_LOGIN_NOTICE", NOTICE)
    out = render_login_html()
    expect = html.escape(NOTICE.split(PAGE)[0])
    assert (f'<p class="notice">{expect}<a href="{PAGE}" rel="noopener noreferrer">{PAGE}</a></p>') in out
    assert "Example Corp&#x27;s assistant" in out and "kept 30 days" in out
    # the notice sits under the subtitle, above the provider buttons
    assert out.index('class="subtitle"') < out.index('<p class="notice">') < out.index('class="provider-btn"')


def test_markup_in_the_notice_is_never_rendered(gated, monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_LOGIN_NOTICE", '<script>alert(1)</script> see "https://x.example/a?b=1&c=<i>"')
    out = render_login_html()
    assert "<script>alert(1)</script>" not in out and "&lt;script&gt;" in out
    assert 'href="https://x.example/a?b=1&amp;c="' in out and "<i>" not in out


def test_browser_lands_on_the_notice_page_instead_of_auto_sso(gated, monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_LOGIN_NOTICE", NOTICE)
    r = gated.get("/sessions")
    assert r.status_code == 302 and r.headers["location"].startswith("/login")
    assert not any(SSO_ATTEMPT_COOKIE in c for c in r.headers.get_list("set-cookie"))
    page = gated.get(r.headers["location"])
    assert page.status_code == 200 and '<p class="notice">' in page.text and PAGE in page.text
    assert 'class="provider-btn" href="/auth/login?provider=stub' in page.text


def test_without_the_notice_auto_sso_is_unchanged(gated, monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_LOGIN_NOTICE", raising=False)
    r = gated.get("/sessions")
    assert r.status_code == 302 and r.headers["location"].startswith("/auth/login?provider=stub")


def test_native_sign_in_shows_the_notice_before_the_single_provider(gated, monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_LOGIN_NOTICE", NOTICE)
    r = _native(gated)
    assert r.status_code == 200 and '<p class="notice">' in r.text and PAGE in r.text
    assert r.text.count('class="provider-btn"') == 1
    href = html.unescape(r.text.split('class="provider-btn" href="', 1)[1].split('"', 1)[0])
    assert href.startswith("/auth/native/authorize?") and "provider=stub" in href
    follow = gated.get(href)
    assert follow.status_code == 302 and "code=stub_code" in follow.headers["location"]


def test_native_sign_in_without_the_notice_auto_selects(gated, monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_LOGIN_NOTICE", raising=False)
    r = _native(gated)
    assert r.status_code == 302 and "code=stub_code" in r.headers["location"]
