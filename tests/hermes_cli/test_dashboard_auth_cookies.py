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
        clear_pkce_cookie(r, use_https=use_https, prefix=prefix)
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
    assert "; Secure" not in at










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





# ---------------------------------------------------------------------------
# PKCE cookie set/clear contract — the OAuth round trip crosses sites, so the
# attribute shape (SameSite / Secure) is load-bearing, not cosmetic. These
# tests pin the full Set-Cookie header shape for both origins so a regression
# in either direction (cookie dropped by Chromium mid-redirect, or a stale
# cookie surviving a clear) fails loudly.
# ---------------------------------------------------------------------------


def test_pkce_cookie_https_is_samesite_none_secure():
    """HTTPS: the PKCE cookie must be SameSite=None + Secure.

    The cookie is set on the /auth/login 302 and must survive the
    cross-site redirect chain through the IDP back to /auth/callback.
    Chromium intermittently drops SameSite=Lax cookies set on a 302 in a
    cross-site chain (crbug 40508226); SameSite=None is the fix.
    """
    client = TestClient(_build_app(use_https=True, prefix=""))
    r = client.get("/set-pkce")
    cookies = r.headers.get_list("set-cookie")
    pkce = next(c for c in cookies if c.startswith(f"__Host-{PKCE_COOKIE}="))
    assert "samesite=none" in pkce.lower()
    assert "; Secure" in pkce
    assert "HttpOnly" in pkce


def test_pkce_cookie_http_stays_lax_without_secure():
    """Loopback HTTP dev: SameSite=None requires Secure, which HTTP can't
    carry — so the setter degrades to bare-name Lax without Secure."""
    client = TestClient(_build_app(use_https=False, prefix=""))
    r = client.get("/set-pkce")
    cookies = r.headers.get_list("set-cookie")
    pkce = next(c for c in cookies if c.startswith(f"{PKCE_COOKIE}="))
    assert "samesite=lax" in pkce.lower()
    assert "; Secure" not in pkce


def test_clear_pkce_cookie_https_matches_set_shape():
    """HTTPS clear: every name variant is deleted with SameSite=None +
    Secure — matching the HTTPS setter so the browser honours the
    deletion for whichever variant was actually set."""
    client = TestClient(_build_app(use_https=True, prefix=""))
    cookies = client.get("/clear").headers.get_list("set-cookie")
    for name in (f"__Host-{PKCE_COOKIE}", f"__Secure-{PKCE_COOKIE}", PKCE_COOKIE):
        deletion = next(c for c in cookies if c.startswith(f'{name}="'))
        assert "Max-Age=0" in deletion
        assert "samesite=none" in deletion.lower()
        assert "; Secure" in deletion


def test_clear_pkce_cookie_http_bare_deletion_is_insecure_lax():
    """HTTP clear: the bare-name deletion must mirror the HTTP setter's
    shape (Lax, no Secure). A Secure deletion can be ignored by browsers
    on a plain-HTTP origin, leaving a stale PKCE cookie behind. The
    __Host-/__Secure- variants require Secure to be valid at all, so
    those deletions keep it regardless of origin."""
    client = TestClient(_build_app(use_https=False, prefix=""))
    cookies = client.get("/clear").headers.get_list("set-cookie")
    bare = next(
        c for c in cookies
        if c.startswith(f'{PKCE_COOKIE}="')
        and not c.startswith("__")
    )
    assert "Max-Age=0" in bare
    assert "samesite=lax" in bare.lower()
    assert "; Secure" not in bare
    for name in (f"__Host-{PKCE_COOKIE}", f"__Secure-{PKCE_COOKIE}"):
        deletion = next(c for c in cookies if c.startswith(f'{name}="'))
        assert "Max-Age=0" in deletion
        assert "; Secure" in deletion


def test_clear_session_cookies_prefixed_deletions_carry_secure():
    """__Host-/__Secure- deletions must carry Secure (and __Host- Path=/):
    browsers reject a prefixed Set-Cookie that violates its prefix rules,
    so an insecure deletion for __Host-hermes_session_at is silently
    ignored and the session cookie survives logout on HTTPS origins."""
    client = TestClient(_build_app(use_https=True, prefix=""))
    cookies = client.get("/clear").headers.get_list("set-cookie")
    for name in (SESSION_AT_COOKIE, SESSION_RT_COOKIE, SESSION_PROVIDER_COOKIE):
        host = next(c for c in cookies if c.startswith(f'__Host-{name}="'))
        assert "; Secure" in host
        assert "Path=/;" in host or host.rstrip().endswith("Path=/")
        secure = next(c for c in cookies if c.startswith(f'__Secure-{name}="'))
        assert "; Secure" in secure
        bare = next(
            c for c in cookies
            if c.startswith(f'{name}="') and not c.startswith("__")
        )
        # Bare-name deletion mirrors the bare setter (Lax, no Secure) so
        # it still works on plain-HTTP origins.
        assert "; Secure" not in bare
        assert "Max-Age=0" in bare
