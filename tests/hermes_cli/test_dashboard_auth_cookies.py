"""Tests for the dashboard-auth cookie helpers."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from hermes_cli.dashboard_auth.cookies import (
    PKCE_COOKIE,
    SESSION_AT_COOKIE,
    SESSION_RT_COOKIE,
    clear_pkce_cookie,
    clear_session_cookies,
    read_pkce_cookie,
    read_session_cookies,
    set_pkce_cookie,
    set_session_cookies,
)


def _build_app(use_https: bool = True):
    app = FastAPI()

    @app.get("/set")
    def set_endpoint():
        r = Response("ok")
        set_session_cookies(
            r, access_token="AT", refresh_token="RT",
            access_token_expires_in=3600, use_https=use_https,
        )
        return r

    @app.get("/set-pkce")
    def set_pkce():
        r = Response("ok")
        set_pkce_cookie(r, payload="provider=stub;state=s;verifier=v",
                        use_https=use_https)
        return r

    @app.get("/clear")
    def clear():
        r = Response("ok")
        clear_session_cookies(r)
        clear_pkce_cookie(r)
        return r

    return app


def test_session_cookies_are_httponly_samesite_lax_secure_in_https():
    client = TestClient(_build_app(use_https=True))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"{SESSION_AT_COOKIE}="))
    rt = next(c for c in cookies if c.startswith(f"{SESSION_RT_COOKIE}="))
    for c in (at, rt):
        assert "HttpOnly" in c
        assert "samesite=lax" in c.lower()
        assert "Secure" in c
        assert "Path=/" in c


def test_session_cookies_omit_secure_when_http():
    client = TestClient(_build_app(use_https=False))
    r = client.get("/set")
    for c in r.headers.get_list("set-cookie"):
        if c.startswith(f"{SESSION_AT_COOKIE}=") or c.startswith(f"{SESSION_RT_COOKIE}="):
            assert "Secure" not in c, f"Cookie unexpectedly Secure: {c}"


def test_session_cookies_have_30day_rt_and_token_ttl_at():
    client = TestClient(_build_app(use_https=True))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"{SESSION_AT_COOKIE}="))
    rt = next(c for c in cookies if c.startswith(f"{SESSION_RT_COOKIE}="))
    assert "Max-Age=3600" in at
    assert "Max-Age=2592000" in rt  # 30 days = 30 * 86400


def test_clear_session_cookies_emits_expired_at_and_rt():
    client = TestClient(_build_app())
    r = client.get("/clear")
    cookies = r.headers.get_list("set-cookie")
    assert any(
        c.startswith(f"{SESSION_AT_COOKIE}=") and "Max-Age=0" in c
        for c in cookies
    )
    assert any(
        c.startswith(f"{SESSION_RT_COOKIE}=") and "Max-Age=0" in c
        for c in cookies
    )


def test_pkce_cookie_short_ttl_and_path_root():
    client = TestClient(_build_app(use_https=True))
    r = client.get("/set-pkce")
    pkce = next(
        c for c in r.headers.get_list("set-cookie")
        if c.startswith(f"{PKCE_COOKIE}=")
    )
    assert "HttpOnly" in pkce
    assert "Max-Age=600" in pkce  # 10 minutes
    assert "Path=/" in pkce
    assert "Secure" in pkce


def test_read_session_cookies_from_request():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(
            b"cookie",
            f"{SESSION_AT_COOKIE}=at_value; {SESSION_RT_COOKIE}=rt_value".encode(),
        )],
    }
    req = Request(scope)
    at, rt = read_session_cookies(req)
    assert at == "at_value"
    assert rt == "rt_value"


def test_read_session_cookies_missing_returns_none():
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    assert read_session_cookies(req) == (None, None)


def test_read_pkce_cookie_round_trip():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"cookie", f"{PKCE_COOKIE}=state=s;verifier=v".encode())],
    }
    req = Request(scope)
    assert read_pkce_cookie(req) == "state=s"  # NB: cookie value stops at ';'


def test_detect_https_via_scheme():
    """``detect_https`` reads from request.url.scheme.

    Under uvicorn proxy_headers=True the scheme is rewritten from
    ``X-Forwarded-Proto``; that's an integration concern, not unit.
    """
    from hermes_cli.dashboard_auth.cookies import detect_https
    http_req = Request({
        "type": "http", "method": "GET", "path": "/", "scheme": "http",
        "headers": [], "server": ("x", 80),
    })
    https_req = Request({
        "type": "http", "method": "GET", "path": "/", "scheme": "https",
        "headers": [], "server": ("x", 443),
    })
    assert detect_https(http_req) is False
    assert detect_https(https_req) is True
