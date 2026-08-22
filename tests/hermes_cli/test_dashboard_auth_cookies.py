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


# ---------------------------------------------------------------------------
# PKCE cookie: regression for #83832
# ---------------------------------------------------------------------------
#
# The PKCE payload is a flat ``key=value;key=value`` string. Without
# URL-encoding, RFC 6265's unquoted ``;`` terminates the cookie-value
# on the wire and the browser only stores the first segment. The
# OIDC callback then sees a partial payload and fails with
# "Missing PKCE state cookie". The fix URL-encodes the payload
# in the setter and decodes it in routes.py before the ``;`` split.
# These tests pin both the wire shape and the round-trip.


def test_set_pkce_cookie_url_encodes_payload_to_avoid_rfc6265_split():
    """The wire-level Set-Cookie value must not contain a literal,
    unquoted ``;`` (the RFC 6265 cookie-value terminator). The
    embedded ``;`` separators in the payload become ``%3B`` so
    every segment of the PKCE payload survives the browser round
    trip intact.

    Regression for #83832 — the OIDC callback was failing with
    "Missing PKCE state cookie" because the browser truncated
    the cookie at the first ``;``.
    """
    from urllib.parse import unquote
    client = TestClient(_build_app(use_https=True, prefix=""))
    r = client.get("/set-pkce")
    pkce_set = next(
        c for c in r.headers.get_list("set-cookie")
        if c.startswith(f"__Host-{PKCE_COOKIE}=")
    )
    # Take just the cookie name=value pair, ignore the attributes.
    pkce_value = pkce_set.split(";", 1)[0]
    # No unquoted literal ``;`` left in the value (i.e. before any
    # attribute separator). The payload is
    # ``provider=stub;state=s;verifier=v`` — encoded as
    # ``provider%3Dstub%3Bstate%3Ds%3Bverifier%3Dv``.
    assert ";" not in pkce_value.split("=", 1)[1], (
        f"unquoted ; leaked into the cookie value: {pkce_value!r}"
    )
    # Round-trip the URL-encoding back to the original payload.
    decoded = unquote(pkce_value.split("=", 1)[1])
    assert decoded == "provider=stub;state=s;verifier=v", (
        f"URL-encoded payload didn't round-trip to the original: "
        f"got {decoded!r}"
    )


def test_pkce_cookie_round_trip_preserves_all_segments():
    """End-to-end: the browser stores the Set-Cookie, the server
    reads it back via ``read_pkce_cookie``, the OAuth callback
    in routes.py decodes the URL-encoded value and parses every
    segment. Pre-fix this would lose ``state`` and ``verifier``
    because the browser truncated at the first ``;``.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest_dashboard_auth import StubAuthProvider  # type: ignore
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from urllib.parse import unquote

    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    try:
        client = TestClient(
            web_server.app, base_url="https://fly-app.fly.dev",
        )
        # /auth/login sets the PKCE cookie with provider / state / verifier
        # packed by the login handler. Capture both the PKCE value and
        # the state that the IDP saw.
        r1 = client.get(
            "/auth/login?provider=stub", follow_redirects=False,
        )
        assert r1.status_code == 302
        pkce_set = next(
            c for c in r1.headers.get_list("set-cookie")
            if "hermes_session_pkce" in c
        )
        # Pull just the name=value portion so we can echo it back as
        # a Cookie header.
        pkce_kv = pkce_set.split(";", 1)[0]
        # Confirm the setter URL-encoded the value.
        encoded_value = pkce_kv.split("=", 1)[1]
        decoded_value = unquote(encoded_value)
        # The login handler packs provider, state, and verifier
        # into the payload. All three must survive intact.
        assert "provider=stub" in decoded_value
        assert "state=" in decoded_value
        assert "verifier=" in decoded_value
        # And the encoded wire value must NOT have a literal, unquoted ``;``
        # between segments.
        assert ";" not in encoded_value, (
            f"literal ; in wire cookie value: {encoded_value!r}"
        )

        # Round-trip via /auth/callback — the success path confirms
        # the callback decoded the URL-encoded value and matched
        # the state. (302 to the post-login page = success.)
        state = r1.headers["location"].split("state=")[1]
        r2 = client.get(
            f"/auth/callback?code=stub_code&state={state}",
            headers={"cookie": pkce_kv},
            follow_redirects=False,
        )
        assert r2.status_code == 302, (
            f"OIDC callback failed — the PKCE cookie round trip is broken. "
            f"Body: {r2.text!r}"
        )
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required


def test_pkce_callback_works_when_next_query_includes_encoded_path():
    """The ``next=`` segment carries a URL-encoded path (e.g. a
    relative URL containing ``;`` from a query parameter on the
    post-login target). The setter URL-encodes the whole payload
    (so the ``;`` in the next= value doesn't trip RFC 6265), and
    the reader decodes the next= value back to its original form
    for the redirect."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest_dashboard_auth import StubAuthProvider  # type: ignore
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from urllib.parse import quote, unquote

    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    try:
        client = TestClient(
            web_server.app, base_url="https://fly-app.fly.dev",
        )
        # next= with a /sessions?view=recent&project=foo target.
        # The login handler URL-encodes the next= value once, then
        # the setter URL-encodes the whole payload. The reader
        # decodes the payload back, and the routes callback then
        # passes the next= through.
        next_target = "/sessions?view=recent&project=foo"
        r1 = client.get(
            f"/auth/login?provider=stub&next={quote(next_target, safe='')}",
            follow_redirects=False,
        )
        assert r1.status_code == 302
        pkce_kv = next(
            c for c in r1.headers.get_list("set-cookie")
            if "hermes_session_pkce" in c
        ).split(";", 1)[0]
        # Drive the callback — must succeed (302 to the post-login
        # target, NOT a 400 "Missing PKCE state cookie").
        state = r1.headers["location"].split("state=")[1]
        r2 = client.get(
            f"/auth/callback?code=stub_code&state={state}",
            headers={"cookie": pkce_kv},
            follow_redirects=False,
        )
        assert r2.status_code == 302, (
            f"callback failed with next= present: {r2.text!r}"
        )
        # The post-login redirect carries the next= target.
        assert next_target in r2.headers.get("location", "") or \
            unquote(next_target) in unquote(r2.headers.get("location", "")), (
            f"post-login redirect didn't carry the next= target: "
            f"{r2.headers.get('location')!r}"
        )
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required




