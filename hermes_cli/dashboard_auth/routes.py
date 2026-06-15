"""HTTP routes for the dashboard-auth OAuth round trip.

Mounted at root (no prefix) by ``web_server.py``. The router does not
auto-gate; gating is performed by ``gated_auth_middleware``, which
allowlists everything under ``/auth/*`` and ``/api/auth/providers``.

The routes:

  GET  /login              → server-rendered login page
  GET  /auth/login?provider=N → 302 to IDP, sets PKCE cookie
  GET  /auth/callback?code,state → completes login, sets session cookies
  POST /auth/logout        → clears cookies, best-effort revoke
  GET  /api/auth/providers → list registered providers (login bootstrap)
  GET  /api/auth/me        → current Session as JSON (auth-required)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from hermes_cli.dashboard_auth import (
    get_provider,
    list_providers,
)
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import (
    InvalidCodeError,
    InvalidCredentialsError,
    ProviderError,
)
from hermes_cli.dashboard_auth.cookies import (
    clear_pkce_cookie,
    clear_session_cookies,
    detect_https,
    read_pkce_cookie,
    read_session_cookies,
    set_pkce_cookie,
    set_session_cookies,
)
from hermes_cli.dashboard_auth.login_page import render_login_html

_log = logging.getLogger(__name__)

router = APIRouter()


def _redirect_uri(request: Request) -> str:
    """Reconstruct the absolute callback URL the IDP redirects back to.

    Three resolution tiers:

      1. ``HERMES_DASHBOARD_PUBLIC_URL`` env var or
         ``dashboard.public_url`` in config.yaml — when set, this is
         the complete authority (scheme + host + optional path prefix)
         and we append ``/auth/callback`` verbatim. ``X-Forwarded-Prefix``
         is IGNORED on this code path because the operator has declared
         the public URL — we no longer need to guess from proxy headers,
         and stacking the prefix on top would double-prefix the common
         case where the prefix is already baked into ``public_url``.
         Relief valve for deploys behind reverse proxies whose forwarded
         headers aren't reliable.

      2. ``X-Forwarded-Prefix: /hermes`` (Mission Control deploys) — we
         prepend the prefix to the path FastAPI's ``url_for`` produces
         (it doesn't natively honour this header — it isn't part of the
         Starlette/uvicorn proxy_headers set).

      3. Bare ``request.url_for("auth_callback")`` — under uvicorn's
         ``proxy_headers=True`` this picks up the public https URL from
         ``X-Forwarded-Host`` plus ``X-Forwarded-Proto``. Fly.io's
         default path.
    """
    from urllib.parse import urlparse, urlunparse

    from hermes_cli.dashboard_auth.prefix import (
        prefix_from_request,
        resolve_public_url,
    )

    # Tier 1: operator-declared public URL.
    public_url = resolve_public_url()
    if public_url:
        # ``public_url`` is the complete authority (possibly with a
        # path prefix already baked in). Append the auth callback path
        # verbatim. ``resolve_public_url`` already stripped any trailing
        # slash so we don't produce ``//auth/callback`` double-slashes.
        return f"{public_url}/auth/callback"

    # Tier 2 + 3: reconstruct from the request URL, optionally with
    # X-Forwarded-Prefix layered on top of the path.
    base = str(request.url_for("auth_callback"))
    prefix = prefix_from_request(request)
    if not prefix:
        return base
    parsed = urlparse(base)
    return urlunparse(parsed._replace(path=f"{prefix}{parsed.path}"))


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _prefix(request: Request) -> str:
    """Resolve the X-Forwarded-Prefix header for the active request.

    Local indirection so the routes pass a consistent value to the
    cookie helpers (cookie name + Path attribute) and the gate's
    redirect builders (login_url construction). See
    ``hermes_cli.dashboard_auth.prefix`` for the normalisation rules.
    """
    from hermes_cli.dashboard_auth.prefix import prefix_from_request
    return prefix_from_request(request)


# ---------------------------------------------------------------------------
# Public: login page (server-rendered HTML, no SPA bundle)
# ---------------------------------------------------------------------------


@router.get("/login", name="login_page")
async def login_page(request: Request) -> HTMLResponse:
    # Read the ``next=`` query the gate's ``_unauth_response`` set on
    # the redirect URL. Validate against the same same-origin rules the
    # callback applies (defence in depth — the gate already filters,
    # but /login is reachable directly too).
    next_path = _validate_post_login_target(
        request.query_params.get("next", "")
    )
    # Desktop system-browser sign-in threads its loopback handoff target
    # through /login when the gateway has more than one provider (the
    # single-provider case skips the page and hits /auth/login directly).
    # Validated to a loopback URL here; /auth/login re-validates on click.
    app_redirect = _validate_app_redirect(
        request.query_params.get("app_redirect", "")
    )
    app_state = request.query_params.get("app_state", "") if app_redirect else ""
    return HTMLResponse(
        render_login_html(
            next_path=next_path,
            app_redirect=app_redirect,
            app_state=app_state,
        ),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Public: provider list for the login-page bootstrap
# ---------------------------------------------------------------------------


@router.get("/api/auth/providers", name="auth_providers")
async def api_auth_providers() -> Any:
    providers = list_providers()
    if not providers:
        # Q13: fail-closed when zero providers are registered.
        return JSONResponse(
            {"detail": "no auth providers registered"},
            status_code=503,
        )
    return {
        "providers": [
            {
                "name": p.name,
                "display_name": p.display_name,
                "supports_password": bool(
                    getattr(p, "supports_password", False)
                ),
            }
            for p in providers
        ],
        # Capability flag for native apps: this gateway can complete the
        # OAuth round trip in the system browser and hand the session back
        # to a loopback redirect (see app_handoff + /api/auth/desktop-exchange).
        # The desktop app feature-detects on this before opening the
        # system browser, and falls back to its embedded login window when
        # absent — so older gateways and the Nous-hosted path keep working.
        "app_handoff": True,
    }


# ---------------------------------------------------------------------------
# Public: OAuth round trip
# ---------------------------------------------------------------------------


@router.get("/auth/login", name="auth_login")
async def auth_login(
    request: Request,
    provider: str,
    next: str = "",
    app_redirect: str = "",
    app_state: str = "",
):
    p = get_provider(provider)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider!r}",
        )

    try:
        ls = p.start_login(redirect_uri=_redirect_uri(request))
    except ProviderError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider,
            reason="provider_unreachable",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Provider unreachable: {e}",
        )

    audit_log(
        AuditEvent.LOGIN_START,
        provider=provider,
        ip=_client_ip(request),
    )

    resp = RedirectResponse(url=ls.redirect_url, status_code=302)
    # Pack the provider name into the PKCE cookie so the callback can
    # find it without a separate cookie. Provider may or may not have
    # already included a ``provider=`` segment.
    pkce = ls.cookie_payload.get("hermes_session_pkce", "")
    if "provider=" not in pkce:
        pkce = f"provider={provider};{pkce}" if pkce else f"provider={provider}"
    # Carry ``next=`` through the round trip in the PKCE cookie. Real
    # IDPs only echo back ``code`` + ``state`` on the callback URL, so
    # query-string transport would lose the value — the cookie is the
    # only server-controlled channel that survives. Validate before we
    # store it so an attacker who reaches /auth/login directly with
    # ``next=//evil.example`` can't poison the cookie.
    safe_next = _validate_post_login_target(next)
    if safe_next:
        from urllib.parse import quote
        pkce = f"{pkce};next={quote(safe_next, safe='')}"
    # Carry the native-app loopback handoff target through the round trip
    # the same way ``next`` rides: real IDPs only echo ``code`` + ``state``
    # on the callback, so the server-set PKCE cookie is the only channel
    # that survives. ``app_redirect`` is validated to a loopback http URL
    # so an attacker who reaches /auth/login directly can't redirect the
    # one-time handoff code to a host they control. ``app_state`` is the
    # app's own CSRF/correlation nonce, echoed back verbatim.
    safe_app_redirect = _validate_app_redirect(app_redirect)
    if safe_app_redirect:
        from urllib.parse import quote
        pkce = f"{pkce};app_redirect={quote(safe_app_redirect, safe='')}"
        if app_state:
            pkce = f"{pkce};app_state={quote(app_state, safe='')}"
    set_pkce_cookie(
        resp, payload=pkce, use_https=detect_https(request),
        prefix=_prefix(request),
    )
    return resp


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
):
    pkce_raw = read_pkce_cookie(request)
    if not pkce_raw:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            reason="missing_pkce_cookie",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="Missing PKCE state cookie",
        )

    # Parse ``provider=...;state=...;verifier=...;next=...`` — the
    # ``next`` segment is optional (only present when /auth/login was
    # given a next= query). All keys live in the same flat namespace;
    # ``next`` carries a URL-encoded path so it never contains ``;``.
    parts = dict(
        seg.split("=", 1) for seg in pkce_raw.split(";") if "=" in seg
    )
    provider_name = parts.get("provider", "")
    expected_state = parts.get("state", "")
    verifier = parts.get("verifier", "")
    # Read next= from the cookie ONLY. The IDP doesn't echo next= back
    # on the callback URL (it only carries ``code`` + ``state``), so any
    # next= query parameter on the callback URL is attacker-controlled
    # and MUST be ignored.
    next_from_cookie = parts.get("next", "")
    # Native-app loopback handoff target + nonce, set by /auth/login when
    # the desktop app drives the flow. Same cookie-only trust model as
    # ``next``: re-validated below before use.
    app_redirect_from_cookie = parts.get("app_redirect", "")
    app_state_from_cookie = parts.get("app_state", "")

    p = get_provider(provider_name)
    if p is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider in cookie: {provider_name!r}",
        )

    if error:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="idp_error",
            error=error,
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail=f"OAuth error from provider: {error} ({error_description})",
        )

    if not state or state != expected_state:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="state_mismatch",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="OAuth state mismatch (CSRF check failed)",
        )

    try:
        session = p.complete_login(
            code=code,
            state=state,
            code_verifier=verifier,
            redirect_uri=_redirect_uri(request),
        )
    except InvalidCodeError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="invalid_code",
            ip=_client_ip(request),
        )
        raise HTTPException(status_code=400, detail=f"Invalid code: {e}")
    except ProviderError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="provider_unreachable",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Provider unreachable: {e}",
        )

    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider=provider_name,
        user_id=session.user_id,
        email=session.email,
        org_id=session.org_id,
        ip=_client_ip(request),
    )

    # Native-app system-browser handoff. This callback ran in the user's
    # real browser, so the OS passkey / Touch ID / WebAuthn prompt could
    # fire (the whole point — issue #42448). But the session cookie must
    # land in the APP, not the browser. So rather than set cookies here we
    # mint a single-use handoff code bound to the session and 302 to the
    # app's loopback listener carrying only that code; the app trades it at
    # ``POST /api/auth/desktop-exchange`` for the cookies. No session
    # cookie is set on the browser, so no stray session is left behind. An
    # invalid/absent app_redirect simply falls through to the normal
    # browser-cookie path below.
    if app_redirect_from_cookie:
        safe_app_redirect = _validate_app_redirect(app_redirect_from_cookie)
        if safe_app_redirect:
            from urllib.parse import (
                unquote,
                urlencode,
                urlparse,
                urlunparse,
            )

            from hermes_cli.dashboard_auth.app_handoff import mint_handoff

            handoff_code = mint_handoff(
                access_token=session.access_token,
                refresh_token=session.refresh_token,
                expires_at=session.expires_at,
            )
            app_state = (
                unquote(app_state_from_cookie)
                if app_state_from_cookie else ""
            )
            parsed = urlparse(safe_app_redirect)
            handoff_query = urlencode(
                {"code": handoff_code, "state": app_state}
            )
            sep = "&" if parsed.query else ""
            target = urlunparse(
                parsed._replace(query=f"{parsed.query}{sep}{handoff_query}")
            )
            resp = RedirectResponse(url=target, status_code=302)
            clear_pkce_cookie(resp, prefix=_prefix(request))
            return resp

    expires_in = max(60, session.expires_at - int(time.time()))
    # Honour the ``next=`` value the gate's _unauth_response set in the
    # /login redirect URL and that /auth/login persisted into the PKCE
    # cookie. We re-validate against the same-origin rules here — the
    # cookie is server-set so this is defence in depth, but a regression
    # that lets attacker-controlled bytes into the cookie would otherwise
    # produce an open redirect.
    landing = _validate_post_login_target(next_from_cookie) or "/"
    resp = RedirectResponse(url=landing, status_code=302)
    set_session_cookies(
        resp,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        access_token_expires_in=expires_in,
        use_https=detect_https(request),
        prefix=_prefix(request),
    )
    clear_pkce_cookie(resp, prefix=_prefix(request))
    return resp


def _validate_post_login_target(raw: str) -> str:
    """Return ``raw`` if it's a safe same-origin path, else empty string.

    The ``next`` query param survives a full OAuth round trip — the gate
    encodes it into the /login redirect, the login page emits it back into
    /auth/login, and the IDP preserves it across /authorize/callback. We
    have to re-validate here because the value came back in via the
    URL (an attacker could craft a /auth/callback URL with their own
    ``next=https://evil.example``).
    """
    if not raw:
        return ""
    from urllib.parse import unquote
    decoded = unquote(raw)
    if not decoded.startswith("/") or decoded.startswith("//"):
        return ""
    # Don't loop back to login pages or auth flow.
    if any(
        decoded == p or decoded.startswith(p)
        for p in ("/login", "/auth/", "/api/auth/")
    ):
        return ""
    # Reject any ``/api/*`` target. The gate's ``_safe_next_target``
    # already filters these out before they reach the cookie, but a
    # malicious or stale ``next=`` value that re-enters via the
    # callback URL must not be honoured: a successful redirect to an
    # API endpoint renders raw JSON in the browser address bar — never
    # a useful post-login destination, and indistinguishable from an
    # attacker trying to weaponise the redirect.
    if decoded == "/api" or decoded.startswith("/api/"):
        return ""
    return decoded


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _validate_app_redirect(raw: str) -> str:
    """Return ``raw`` (URL-decoded) if it's a safe loopback http URL, else "".

    The desktop app passes ``app_redirect`` so ``/auth/callback`` can bounce
    the one-time handoff code to its ephemeral local listener. This value
    survives the full OAuth round trip in the server-set PKCE cookie, but it
    enters via the URL at ``/auth/login`` — so an attacker could craft
    ``/auth/login?provider=…&app_redirect=https://evil.example/grab`` to try
    to exfiltrate a handoff code. We therefore accept ONLY an ``http://``
    URL whose host is a loopback literal and which carries an explicit port
    (the app's ephemeral listener) and an absolute path. That confines the
    handoff code to a listener on the same machine.
    """
    if not raw:
        return ""
    from urllib.parse import unquote, urlparse

    decoded = unquote(raw)
    try:
        parsed = urlparse(decoded)
    except ValueError:
        return ""
    # Loopback listeners are plain http; never accept https (or anything
    # else) here — the only valid target is a local socket.
    if parsed.scheme != "http":
        return ""
    if (parsed.hostname or "") not in _LOOPBACK_HOSTS:
        return ""
    # Credentials in the authority are never legitimate here.
    if parsed.username or parsed.password:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port is None:
        return ""
    if not parsed.path.startswith("/"):
        return ""
    return decoded


# ---------------------------------------------------------------------------
# Public: native-app session handoff (desktop system-browser sign-in)
# ---------------------------------------------------------------------------


class _DesktopExchangeBody(BaseModel):
    code: str
    state: str = ""


@router.post("/api/auth/desktop-exchange", name="auth_desktop_exchange")
async def auth_desktop_exchange(request: Request, body: _DesktopExchangeBody):
    """Trade a one-time handoff code for the dashboard session cookies.

    Completes the desktop system-browser sign-in: the app obtained the
    handoff code on its loopback listener after ``/auth/callback`` ran in
    the user's real browser. The app POSTs the code here through its OAuth
    session partition; we set the same HttpOnly session cookies the browser
    flow would (``set_session_cookies``), which land in the app's cookie
    jar. All token material stays server-side — the app never sees a token,
    only the opaque code.

    Allowlisted past the auth gate (it is the thing that establishes the
    session). The code is single-use, short-TTL, and 256-bit, so a missing
    or stale code is the only meaningful failure and brute force is
    infeasible — no rate limiting needed.
    """
    from hermes_cli.dashboard_auth.app_handoff import (
        HandoffInvalid,
        consume_handoff,
    )

    try:
        material = consume_handoff(body.code)
    except HandoffInvalid:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            reason="invalid_handoff_code",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired handoff code",
        )

    expires_in = max(60, int(material["expires_at"]) - int(time.time()))
    resp = JSONResponse({"ok": True})
    set_session_cookies(
        resp,
        access_token=str(material["access_token"]),
        refresh_token=str(material["refresh_token"]),
        access_token_expires_in=expires_in,
        use_https=detect_https(request),
        prefix=_prefix(request),
    )
    return resp


# ---------------------------------------------------------------------------
# Public: password (non-redirect) login
# ---------------------------------------------------------------------------
#
# Brute-force throttle. The OAuth flow has no guessable secret on our side
# (the IDP owns credentials), but ``/auth/password-login`` accepts a
# password we verify locally, so it's a credential-stuffing target. A
# simple in-process sliding-window limiter per client IP raises the cost
# of online guessing without any external dependency. It is intentionally
# best-effort: process-local (resets on restart), and behind a trusting
# proxy the IP is the proxy's unless X-Forwarded-For is set — which is why
# this is defence-in-depth on top of the provider's own constant-time
# verify, not the only line of defence.

_PW_RATE_MAX_ATTEMPTS = 10
_PW_RATE_WINDOW_SEC = 60.0
_pw_attempts: Dict[str, Deque[float]] = defaultdict(deque)
_pw_attempts_lock = threading.Lock()


def _password_rate_limited(ip: str) -> bool:
    """True if ``ip`` has exceeded the password-login attempt budget.

    Sliding window: prune attempts older than the window, then check the
    count. Records the attempt timestamp when allowed. An empty IP (no
    discernible client) shares a single bucket — fail-safe toward
    throttling rather than letting unattributable traffic through
    unmetered.
    """
    now = time.monotonic()
    cutoff = now - _PW_RATE_WINDOW_SEC
    key = ip or "_unknown_"
    with _pw_attempts_lock:
        bucket = _pw_attempts[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _PW_RATE_MAX_ATTEMPTS:
            return True
        bucket.append(now)
        return False


def _reset_password_rate_limit() -> None:
    """Test-only: clear all rate-limit buckets."""
    with _pw_attempts_lock:
        _pw_attempts.clear()


class _PasswordLoginBody(BaseModel):
    provider: str
    username: str
    password: str
    next: str = ""


@router.post("/auth/password-login", name="auth_password_login")
async def auth_password_login(request: Request, body: _PasswordLoginBody):
    """Authenticate a username/password against a password provider.

    Mirrors the cookie-minting tail of ``/auth/callback`` but skips the
    PKCE/state/code machinery (those are OAuth-only). On success sets the
    session cookies and returns JSON ``{"ok": true, "next": <path>}`` —
    the credential form POSTs via fetch and navigates client-side, so a
    302 (which fetch follows opaquely) is the wrong shape here.

    Failure modes, all deliberately generic so the endpoint can't be used
    as a username oracle or a provider-enumeration oracle:
      * unknown provider / provider lacks password support → 404
      * bad credentials → 401 ("Invalid credentials")
      * backing store unreachable → 503
      * too many attempts from this IP → 429
    """
    ip = _client_ip(request)
    if _password_rate_limited(ip):
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=body.provider,
            reason="rate_limited",
            ip=ip,
        )
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again shortly.",
        )

    p = get_provider(body.provider)
    if p is None or not getattr(p, "supports_password", False):
        # Don't leak which providers exist or which support passwords —
        # same 404 whether the provider is unknown or OAuth-only.
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=body.provider,
            reason="unknown_password_provider",
            ip=ip,
        )
        raise HTTPException(status_code=404, detail="Unknown provider")

    try:
        session = p.complete_password_login(
            username=body.username, password=body.password
        )
    except InvalidCredentialsError:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=body.provider,
            reason="invalid_credentials",
            ip=ip,
        )
        # Generic message — never distinguish unknown-user from wrong-password.
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except NotImplementedError:
        # supports_password was True but the method isn't actually
        # implemented — a provider bug, not a client error.
        raise HTTPException(status_code=500, detail="Provider misconfigured")
    except ProviderError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=body.provider,
            reason="provider_unreachable",
            ip=ip,
        )
        raise HTTPException(status_code=503, detail=f"Provider unreachable: {e}")

    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider=body.provider,
        user_id=session.user_id,
        email=session.email,
        org_id=session.org_id,
        ip=ip,
    )

    expires_in = max(60, session.expires_at - int(time.time()))
    landing = _validate_post_login_target(body.next) or "/"
    resp = JSONResponse({"ok": True, "next": landing})
    set_session_cookies(
        resp,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        access_token_expires_in=expires_in,
        use_https=detect_https(request),
        prefix=_prefix(request),
    )
    return resp


@router.post("/auth/logout", name="auth_logout")
async def auth_logout(request: Request):
    _at, rt = read_session_cookies(request)
    if rt:
        # Best-effort revoke. Try every provider so a session minted by
        # any registered provider is revoked correctly. Failures are
        # logged but never raised.
        for provider in list_providers():
            try:
                provider.revoke_session(refresh_token=rt)
            except Exception as e:  # noqa: BLE001 — best-effort
                _log.warning(
                    "dashboard-auth: revoke on %r failed: %s",
                    provider.name, e,
                )

    sess = getattr(request.state, "session", None)
    audit_log(
        AuditEvent.LOGOUT,
        provider=(sess.provider if sess else "unknown"),
        user_id=(sess.user_id if sess else ""),
        ip=_client_ip(request),
    )

    prefix = _prefix(request)
    resp = RedirectResponse(url=f"{prefix}/login", status_code=302)
    clear_session_cookies(resp, prefix=prefix)
    clear_pkce_cookie(resp, prefix=prefix)
    return resp


# ---------------------------------------------------------------------------
# Auth-required: identity probe for the SPA
# ---------------------------------------------------------------------------


@router.get("/api/auth/me", name="auth_me")
async def api_auth_me(request: Request):
    """Return the verified session as JSON. Auth-required (gate enforces)."""
    sess = getattr(request.state, "session", None)
    if sess is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "user_id": sess.user_id,
        "email": sess.email,
        "display_name": sess.display_name,
        "org_id": sess.org_id,
        "provider": sess.provider,
        "expires_at": sess.expires_at,
    }


# ---------------------------------------------------------------------------
# Auth-required: WS upgrade ticket (Phase 5)
# ---------------------------------------------------------------------------


@router.post("/api/auth/ws-ticket", name="auth_ws_ticket")
async def api_auth_ws_ticket(request: Request):
    """Mint a short-lived single-use ticket for the authenticated session.

    Browsers cannot set ``Authorization`` on a WebSocket upgrade, so in
    gated mode the SPA POSTs this endpoint to get a ``?ticket=`` value to
    append to ``/api/pty``, ``/api/ws``, ``/api/pub``, or ``/api/events``.

    The ticket has a 30-second TTL and is single-use. Calling this endpoint
    multiple times in quick succession (e.g. one ticket per WS) is the
    expected pattern.
    """
    sess = getattr(request.state, "session", None)
    if sess is None:
        # Middleware should already have rejected, but check defensively.
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Import here so the routes module stays usable in test contexts that
    # don't load the ticket store.
    from hermes_cli.dashboard_auth.ws_tickets import TTL_SECONDS, mint_ticket

    ticket = mint_ticket(user_id=sess.user_id, provider=sess.provider)
    audit_log(
        AuditEvent.WS_TICKET_MINTED,
        provider=sess.provider,
        user_id=sess.user_id,
        ip=_client_ip(request),
    )
    return {"ticket": ticket, "ttl_seconds": TTL_SECONDS}
