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
# PKCE cookie: regression for #83832 (field case: Traefik+Authentik chain)
# ---------------------------------------------------------------------------
#
# The PKCE payload is a flat ``key=value;key=value`` string. A raw ``;``
# is a cookie-attribute terminator, so Python's http.cookies emits the
# value in RFC 6265 quoted form with each ``;`` escaped as the
# backslash-octal ``\073``. Mainstream browsers echo that form back
# verbatim and Python decodes it — but ``"`` and ``\`` are outside the
# plain cookie-octet set, and cookie-aware proxy hops that parse and
# re-emit the Cookie header (verified for Go's net/http, common in
# Go-based proxy/IDP middleware) reject the value and drop the cookie
# entirely. The callback
# then 400s with "Missing PKCE state cookie" even though the browser
# sent the cookie. The fix URL-encodes the payload in the setter so the
# wire value contains only cookie-octets, and every reader decodes via
# cookies.parse_pkce_payload(). These tests pin the wire shape and the
# round trip.


def test_set_pkce_cookie_url_encodes_payload_to_avoid_rfc6265_split():
    """The wire-level cookie value must contain only plain RFC 6265
    cookie-octets: no raw ``;`` (attribute terminator), no ``"`` and no
    ``\\`` (the http.cookies quoted form that strict cookie-aware proxy
    parsers — verified for Go's net/http — reject, dropping the whole
    cookie).

    Regression for #83832 / the Traefik+Authentik support case: the
    callback failed with "Missing PKCE state cookie" because a proxy
    hop dropped the quoted ``\\073`` form.
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
    wire = pkce_value.split("=", 1)[1]
    # No unquoted literal ``;`` in the value (attribute terminator). The
    # payload ``provider=stub;state=s;verifier=v`` is encoded as
    # ``provider%3Dstub%3Bstate%3Ds%3Bverifier%3Dv``.
    assert ";" not in wire, (
        f"unquoted ; leaked into the cookie value: {pkce_value!r}"
    )
    # The real field failure (Traefik/Authentik): the http.cookies quoted
    # form ``"...\073..."`` is not made of cookie-octets, and Go's
    # net/http drops any cookie whose value contains ``"`` or ``\``.
    # Pin the whole value to the plain RFC 6265 cookie-octet set.
    assert '"' not in wire and "\\" not in wire, (
        f"non-cookie-octet chars leaked into the wire value: {wire!r}"
    )
    cookie_octets = (
        "!#$%&'()*+-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    )
    assert all(ch in cookie_octets for ch in wire), (
        f"non-cookie-octet chars in the wire value: {wire!r}"
    )
    # Round-trip the URL-encoding back to the original payload.
    decoded = unquote(wire)
    assert decoded == "provider=stub;state=s;verifier=v", (
        f"URL-encoded payload didn't round-trip to the original: "
        f"got {decoded!r}"
    )


def test_parse_pkce_payload_old_format_cookie_survives_rolling_upgrade():
    """Mixed-version window (10-minute PKCE TTL): a cookie minted by a
    pre-encoding server arrives at the new reader — after starlette's
    cookie-header unquoting — as the FLAT form with raw ``;`` between
    segments and a single-encoded ``next``. The reader must split it
    as-is, NOT payload-decode it first: decoding early would turn an
    old ``next`` value containing ``%3B`` into a bogus delimiter and
    truncate the post-login target.
    """
    from hermes_cli.dashboard_auth.cookies import parse_pkce_payload

    old = (
        "provider=stub;state=s123;verifier=v456;"
        "next=%2Fsessions%3Fx%3Da%3Bb%26project%3Dfoo"
    )
    parts = parse_pkce_payload(old)
    assert parts == {
        "provider": "stub",
        "state": "s123",
        "verifier": "v456",
        # Preserved verbatim — still single-encoded, exactly what the
        # old reader produced; the downstream next-validator unquotes.
        "next": "%2Fsessions%3Fx%3Da%3Bb%26project%3Dfoo",
    }, f"old-format cookie mis-parsed: {parts!r}"


def test_parse_pkce_payload_new_format_round_trips_setter_encoding():
    """The new encoded wire form (no raw ``;`` possible — it is %3B)
    decodes back to the exact original payload segments."""
    from urllib.parse import quote

    from hermes_cli.dashboard_auth.cookies import parse_pkce_payload

    payload = "provider=stub;state=s123;verifier=v456;next=%2Fsessions"
    wire = quote(payload, safe="")
    assert ";" not in wire
    parts = parse_pkce_payload(wire)
    assert parts == {
        "provider": "stub",
        "state": "s123",
        "verifier": "v456",
        "next": "%2Fsessions",
    }, f"new-format wire value mis-parsed: {parts!r}"


def test_pkce_cookie_round_trip_preserves_all_segments():
    """End-to-end: the browser stores the Set-Cookie, the server
    reads it back via ``read_pkce_cookie``, the OAuth callback
    in routes.py decodes the URL-encoded value and parses every
    segment. Pre-fix, the quoted ``\\073`` wire form was dropped
    whole by strict proxy-hop cookie parsers, so the callback saw
    no PKCE cookie at all.
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
        # The post-login redirect carries EXACTLY the original target:
        # login-side single-encode + setter whole-payload encode must be
        # symmetrically undone by parse_pkce_payload + the validator's
        # unquote. Pin the exact byte shape — a relaxed substring match
        # would hide an encode/decode imbalance.
        assert r2.headers.get("location") == next_target, (
            f"post-login redirect didn't carry the exact next= target: "
            f"{r2.headers.get('location')!r} != {next_target!r}"
        )
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required





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
