"""Tests for the dashboard-auth cookie helpers."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from hermes_cli.dashboard_auth.cookies import (
    PKCE_COOKIE,
    SESSION_AT_COOKIE,
    SESSION_PROVIDER_COOKIE,
    SESSION_RT_COOKIE,
    clear_pkce_cookie,
    clear_session_cookies,
    read_pkce_cookie,
    read_session_cookies,
    read_session_provider,
    set_pkce_cookie,
    set_session_cookies,
)


def _build_app(use_https: bool = True, prefix: str = ""):
    app = FastAPI()

    @app.get("/set")
    def set_endpoint():
        r = Response("ok")
        set_session_cookies(
            r, access_token="AT", refresh_token="RT",
            access_token_expires_in=3600, use_https=use_https,
            prefix=prefix, provider="nous",
        )
        return r

    @app.get("/set-pkce")
    def set_pkce():
        r = Response("ok")
        set_pkce_cookie(r, payload="provider=stub;state=s;verifier=v",
                        use_https=use_https, prefix=prefix)
        return r

    @app.get("/clear")
    def clear():
        r = Response("ok")
        clear_session_cookies(r, prefix=prefix)
        clear_pkce_cookie(r, prefix=prefix)
        return r

    return app


# Cookie name resolution helpers used throughout — the bare name resolves
# to a request-shape-dependent variant (__Host- / __Secure- / bare).
# Tests pin a specific shape so a regression in the name-resolution
# logic fails loudly rather than silently breaking sessions.


def test_session_cookies_use_host_prefix_on_https_direct():
    """HTTPS + no proxy prefix → __Host- prefix (strongest spec
    hardening: bound to exact origin, requires Path=/, requires Secure)."""
    client = TestClient(_build_app(use_https=True, prefix=""))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"__Host-{SESSION_AT_COOKIE}="))
    rt = next(c for c in cookies if c.startswith(f"__Host-{SESSION_RT_COOKIE}="))
    provider = next(c for c in cookies if c.startswith(f"__Host-{SESSION_PROVIDER_COOKIE}=nous"))
    for c in (at, rt, provider):
        assert "HttpOnly" in c
        assert "samesite=lax" in c.lower()
        assert "Secure" in c
        assert "Path=/" in c


def test_session_cookies_use_secure_prefix_when_proxied():
    """HTTPS + /hermes prefix → __Secure- prefix (__Host- forbids
    Path != "/"; __Secure- keeps the Secure-required hardening)."""
    client = TestClient(_build_app(use_https=True, prefix="/hermes"))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"__Secure-{SESSION_AT_COOKIE}="))
    assert "Path=/hermes" in at
    assert "Secure" in at
    # __Host- variant must NOT be emitted on the prefix path.
    assert not any(
        c.startswith(f"__Host-{SESSION_AT_COOKIE}=") for c in cookies
    )


def test_session_cookies_use_bare_name_on_http():
    """Loopback HTTP dev: __Host- / __Secure- both require Secure, which
    we can't set on HTTP. Use bare cookie names."""
    client = TestClient(_build_app(use_https=False))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    # Bare name present; no __Host- / __Secure- variant emitted.
    assert any(c.startswith(f"{SESSION_AT_COOKIE}=") for c in cookies)
    assert not any(
        c.startswith(f"__Host-{SESSION_AT_COOKIE}=")
        or c.startswith(f"__Secure-{SESSION_AT_COOKIE}=")
        for c in cookies
    )
    # No Secure flag (HTTP).
    at = next(c for c in cookies if c.startswith(f"{SESSION_AT_COOKIE}="))
    assert "Secure" not in at










def test_read_session_cookies_from_request_secure_prefix():
    """Reader also finds cookies set with the __Secure- variant
    (HTTPS behind a proxy prefix)."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(
            b"cookie",
            f"__Secure-{SESSION_AT_COOKIE}=at_value; "
            f"__Secure-{SESSION_RT_COOKIE}=rt_value".encode(),
        )],
    }
    req = Request(scope)
    at, rt = read_session_cookies(req)
    assert at == "at_value"
    assert rt == "rt_value"




