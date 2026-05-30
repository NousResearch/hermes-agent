"""Tests for the dashboard-auth cookie helpers."""
from __future__ import annotations

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


def _build_app(use_https: bool = True, prefix: str = ""):
    app = FastAPI()

    @app.get("/set")
    def set_endpoint():
        r = Response("ok")
        set_session_cookies(
            r, access_token="AT", refresh_token="RT",
            access_token_expires_in=3600, use_https=use_https,
            prefix=prefix,
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
    for c in (at, rt):
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


def test_session_cookies_have_30day_rt_and_token_ttl_at():
    client = TestClient(_build_app(use_https=True))
    r = client.get("/set")
    cookies = r.headers.get_list("set-cookie")
    at = next(c for c in cookies if c.startswith(f"__Host-{SESSION_AT_COOKIE}="))
    rt = next(c for c in cookies if c.startswith(f"__Host-{SESSION_RT_COOKIE}="))
    assert "Max-Age=3600" in at
    assert "Max-Age=2592000" in rt  # 30 days = 30 * 86400


def test_clear_session_cookies_emits_expired_at_and_rt():
    """``clear_session_cookies`` emits Max-Age=0 deletions for every
    plausible cookie-name variant under the active prefix so we flush
    stale cookies that an older deploy may have set under a different
    prefix."""
    client = TestClient(_build_app())
    r = client.get("/clear")
    cookies = r.headers.get_list("set-cookie")
    # At least one variant of each session cookie should be deleted.
    assert any(
        SESSION_AT_COOKIE in c and "Max-Age=0" in c for c in cookies
    )
    assert any(
        SESSION_RT_COOKIE in c and "Max-Age=0" in c for c in cookies
    )


def test_pkce_cookie_short_ttl_and_path_root():
    client = TestClient(_build_app(use_https=True))
    r = client.get("/set-pkce")
    pkce = next(
        c for c in r.headers.get_list("set-cookie")
        if PKCE_COOKIE in c
    )
    assert "HttpOnly" in pkce
    assert "Max-Age=600" in pkce  # 10 minutes
    assert "Path=/" in pkce
    assert "Secure" in pkce


def test_read_session_cookies_from_request_bare_name():
    """Reader accepts the bare name (loopback) by default."""
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


def test_read_session_cookies_from_request_host_prefix():
    """Reader also finds cookies set with the __Host- variant
    (HTTPS direct deploy)."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(
            b"cookie",
            f"__Host-{SESSION_AT_COOKIE}=at_value; "
            f"__Host-{SESSION_RT_COOKIE}=rt_value".encode(),
        )],
    }
    req = Request(scope)
    at, rt = read_session_cookies(req)
    assert at == "at_value"
    assert rt == "rt_value"


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


def test_read_session_cookies_strict_https_direct_rejects_bare_fallback():
    """HTTPS direct deploys must authenticate only the __Host- variant."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(
            b"cookie",
            f"{SESSION_AT_COOKIE}=bare_at; __Host-{SESSION_AT_COOKIE}=host_at".encode(),
        )],
    }
    req = Request(scope)
    at, rt = read_session_cookies(req, use_https=True, prefix="")
    assert at == "host_at"
    assert rt is None


def test_read_session_cookies_strict_https_proxied_rejects_bare_fallback():
    """HTTPS prefix deploys must authenticate only the __Secure- variant."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(
            b"cookie",
            f"{SESSION_AT_COOKIE}=bare_at; __Host-{SESSION_AT_COOKIE}=host_at".encode(),
        )],
    }
    req = Request(scope)
    assert read_session_cookies(req, use_https=True, prefix="/hermes") == (None, None)


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
