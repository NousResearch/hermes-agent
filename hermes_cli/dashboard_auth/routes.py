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
    list_session_providers,
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
    clear_sso_attempt_cookie,
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
    return HTMLResponse(
        render_login_html(next_path=next_path),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Public: provider list for the login-page bootstrap
# ---------------------------------------------------------------------------


@router.get("/api/auth/providers", name="auth_providers")
async def api_auth_providers() -> Any:
    # Advertise only interactive providers; a token-only credential (e.g. drain)
    # is not a sign-in option.
    providers = list_session_providers()
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
    }


# ---------------------------------------------------------------------------
# Public: OAuth round trip
# ---------------------------------------------------------------------------


@router.get("/auth/login", name="auth_login")
async def auth_login(request: Request, provider: str, next: str = ""):
    p = get_provider(provider)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider!r}",
        )
    if not getattr(p, "supports_session", True):
        raise HTTPException(
            status_code=404,
            detail=f"Provider does not support interactive login: {provider!r}",
        )
    if getattr(p, "supports_password", False):
        from urllib.parse import quote

        safe_next = _validate_post_login_target(next)
        login_url = f"{_prefix(request)}/login"
        if safe_next:
            login_url = f"{login_url}?next={quote(safe_next, safe='')}"
        return RedirectResponse(url=login_url, status_code=302)

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
    set_pkce_cookie(
        resp, payload=pkce, use_https=detect_https(request),
        prefix=_prefix(request),
    )
    return resp


# ---------------------------------------------------------------------------
# Public: RFC 8252 native-app authorization (system browser + loopback + PKCE)
# ---------------------------------------------------------------------------


def _validate_loopback_redirect_uri(raw: str) -> str:
    """Return ``raw`` if it is a safe loopback redirect_uri, else raise.

    RFC 8252 §7.3 restricts native-app redirects to the loopback interface.
    We accept only ``http://127.0.0.1[:port]/...`` and ``http://[::1][:port]/...``
    — literal loopback IPs. ``localhost`` is deliberately NOT accepted
    (RFC 8252 §8.3: the name can resolve to a non-loopback address via the
    hosts file or a hostile resolver, so clients "SHOULD use loopback IP
    literals"; the desktop always sends ``127.0.0.1``).
    A non-loopback host would let an attacker who can reach ``/auth/native/
    authorize`` (a public route) turn the gateway's authenticated callback
    into an open redirect that leaks a live authorization code to an
    arbitrary origin — so this check is a security boundary, not ergonomics.
    """
    from urllib.parse import urlparse

    if not raw:
        raise HTTPException(status_code=400, detail="redirect_uri required")
    parsed = urlparse(raw)
    if parsed.scheme != "http":
        raise HTTPException(
            status_code=400,
            detail="native redirect_uri must be http:// on the loopback interface",
        )
    host = (parsed.hostname or "").lower()
    if host not in ("127.0.0.1", "::1"):
        raise HTTPException(
            status_code=400,
            detail=(
                "native redirect_uri host must be a loopback IP literal "
                "(127.0.0.1 / ::1)"
            ),
        )
    return raw


@router.get("/auth/native/authorize", name="auth_native_authorize")
async def auth_native_authorize(
    request: Request,
    provider: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "",
    redirect_uri: str = "",
    state: str = "",
):
    """Begin an RFC 8252 native-app login for the desktop app.

    The desktop opens THIS url in the system browser with its own PKCE
    ``code_challenge`` (S256), a loopback ``redirect_uri``, and a CSRF
    ``state``. We stash a pending broker authorization, then hand off to the
    EXISTING upstream PKCE round trip (``provider.start_login`` → IDP →
    ``/auth/callback``), carrying the broker_state in the same PKCE cookie the
    cookie flow uses. On the callback we mint a loopback code (see
    ``auth_callback``); no browser session cookie is ever set for the desktop.
    """
    # PKCE method must be S256 (RFC 7636 — plain is disallowed for native apps).
    if code_challenge_method.upper() != "S256":
        raise HTTPException(
            status_code=400,
            detail="code_challenge_method must be S256",
        )
    if not code_challenge:
        raise HTTPException(status_code=400, detail="code_challenge required")
    _validate_loopback_redirect_uri(redirect_uri)

    # Resolve the provider. With exactly one session provider registered
    # (the common hosted case) an empty ``provider`` selects it, mirroring
    # the auto-SSO convenience so the desktop needn't hardcode the name.
    p = get_provider(provider) if provider else None
    if p is None and not provider:
        sess_providers = list_session_providers()
        if len(sess_providers) == 1:
            p = sess_providers[0]
    if p is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider: {provider!r}"
        )
    if not getattr(p, "supports_session", True) or getattr(
        p, "supports_password", False
    ):
        # Native PKCE brokering is only meaningful for redirect/OAuth
        # providers; a password provider has no IDP round trip to broker.
        raise HTTPException(
            status_code=400,
            detail=f"Provider does not support native OAuth login: {p.name!r}",
        )

    from hermes_cli.dashboard_auth import native_flow

    try:
        broker_state = native_flow.register_pending(
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            client_state=state,
            client_ip=_client_ip(request),
        )
    except native_flow.NativeFlowError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        ls = p.start_login(redirect_uri=_redirect_uri(request))
    except ProviderError as e:
        raise HTTPException(status_code=503, detail=f"Provider unreachable: {e}")

    audit_log(
        AuditEvent.NATIVE_AUTHORIZE_START,
        provider=p.name,
        ip=_client_ip(request),
    )

    resp = RedirectResponse(url=ls.redirect_url, status_code=302)
    # Thread the provider name + broker_state through the gateway's OWN PKCE
    # cookie so the callback can (a) dispatch to the right provider and (b)
    # find the pending native authorization. The desktop's challenge/state
    # never touch this cookie — only our opaque broker_state does.
    pkce = ls.cookie_payload.get("hermes_session_pkce", "")
    if "provider=" not in pkce:
        pkce = f"provider={p.name};{pkce}" if pkce else f"provider={p.name}"
    pkce = f"{pkce};broker={broker_state}"
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
    # RFC 8252 native-app flow: /auth/native/authorize stashed a broker_state
    # here so this callback can mint a loopback authorization code for the
    # desktop instead of setting browser session cookies. Absent for the
    # ordinary cookie/SPA login.
    broker_state = parts.get("broker", "")

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

    expires_in = max(60, session.expires_at - int(time.time()))

    # RFC 8252 native-app branch: the desktop initiated this via
    # /auth/native/authorize and is waiting on a loopback listener. Mint a
    # one-time gateway authorization code bound to the desktop's PKCE
    # challenge and 302 the SYSTEM BROWSER to the desktop's loopback
    # redirect_uri — no session cookies are set on this response, and the
    # tokens are handed to the desktop only at /auth/native/token. This is
    # what lets the desktop avoid both the embedded webview and cookie auth.
    if broker_state:
        from hermes_cli.dashboard_auth import native_flow

        try:
            pending = native_flow.get_pending(broker_state)
            gw_code = native_flow.complete_pending(
                broker_state, session=session
            )
        except native_flow.NativeFlowError:
            audit_log(
                AuditEvent.NATIVE_TOKEN_FAILURE,
                provider=provider_name,
                reason="pending_not_found",
                ip=_client_ip(request),
            )
            raise HTTPException(
                status_code=400,
                detail="Native login expired or unknown; restart sign-in.",
            )
        from urllib.parse import urlencode

        sep = "&" if "?" in pending.redirect_uri else "?"
        loopback = (
            f"{pending.redirect_uri}{sep}"
            f"{urlencode({'code': gw_code, 'state': pending.client_state})}"
        )
        audit_log(
            AuditEvent.NATIVE_CODE_ISSUED,
            provider=provider_name,
            user_id=session.user_id,
            ip=_client_ip(request),
        )
        resp = RedirectResponse(url=loopback, status_code=302)
        # Clear the PKCE cookie (its job is done) but set NO session cookies:
        # the desktop is not a browser session, it redeems the code for a
        # bearer token it stores itself.
        clear_pkce_cookie(resp, prefix=_prefix(request))
        clear_sso_attempt_cookie(resp, prefix=_prefix(request))
        return resp

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
        provider=session.provider,
    )
    clear_pkce_cookie(resp, prefix=_prefix(request))
    # Clear the one-shot auto-SSO loop-guard marker now that login succeeded,
    # so it never lingers to suppress a future silent attempt after logout.
    clear_sso_attempt_cookie(resp, prefix=_prefix(request))
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
        provider=session.provider,
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
    append to ``/api/pty``, ``/api/console``, ``/api/ws``, ``/api/pub``, or
    ``/api/events``.

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


# ---------------------------------------------------------------------------
# Public: RFC 8252 native-app token exchange (loopback code → bearer tokens)
# ---------------------------------------------------------------------------


class _NativeTokenBody(BaseModel):
    code: str
    code_verifier: str


@router.post("/auth/native/token", name="auth_native_token")
async def auth_native_token(request: Request, body: _NativeTokenBody):
    """Exchange a loopback gateway code + PKCE verifier for bearer tokens.

    The desktop POSTs this from its loopback listener after catching the
    ``?code=`` redirect. We verify ``SHA256(code_verifier) == code_challenge``
    (the challenge captured at ``/auth/native/authorize``), consume the code
    (single use), and return the upstream tokens **in the JSON body** — the
    desktop stores them in the OS keychain and authenticates with
    ``Authorization: Bearer`` thereafter. No cookie is set on this response.

    Failure modes (all deliberately generic — the code is consumed on every
    path so there is no verifier oracle and no replay):
      * unknown / expired / already-redeemed code, or PKCE mismatch → 400
    """
    from hermes_cli.dashboard_auth import native_flow

    try:
        session = native_flow.redeem_code(
            code=body.code, code_verifier=body.code_verifier
        )
    except native_flow.CodeInvalid:
        audit_log(
            AuditEvent.NATIVE_TOKEN_FAILURE,
            reason="invalid_code_or_pkce",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired authorization code.",
        )

    audit_log(
        AuditEvent.NATIVE_TOKEN_SUCCESS,
        provider=session.provider,
        user_id=session.user_id,
        ip=_client_ip(request),
    )
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "token_type": "Bearer",
        "expires_at": session.expires_at,
        "provider": session.provider,
        "user_id": session.user_id,
    }


class _NativeRefreshBody(BaseModel):
    refresh_token: str
    provider: str = ""


@router.post("/auth/native/refresh", name="auth_native_refresh")
async def auth_native_refresh(request: Request, body: _NativeRefreshBody):
    """Rotate a native-app session using the desktop-held refresh token.

    The desktop owns its refresh token (OS keychain) rather than a cookie, so
    it rotates here instead of relying on the gate's transparent cookie
    rotation. Mirrors the middleware's ``_attempt_refresh`` provider stacking:
    tries each session provider until one rotates the token, returning the new
    access/refresh pair **in the JSON body**.

    Failure modes:
      * every provider rejects the RT (dead/expired/reuse-detected) → 401
        ``session_expired`` so the desktop starts a fresh native login;
      * a provider's IDP is unreachable and none rotated → 503.
    """
    from hermes_cli.dashboard_auth import list_session_providers
    from hermes_cli.dashboard_auth.base import RefreshExpiredError

    if not body.refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token required")

    providers = list_session_providers()
    if body.provider:
        providers.sort(key=lambda p: p.name != body.provider)

    unreachable: str | None = None
    for provider in providers:
        try:
            session = provider.refresh_session(refresh_token=body.refresh_token)
        except RefreshExpiredError:
            continue
        except ProviderError as e:
            if unreachable is None:
                unreachable = provider.name
            _log.warning(
                "dashboard-auth: provider %r unreachable during native refresh: %s",
                provider.name, e,
            )
            continue
        audit_log(
            AuditEvent.REFRESH_SUCCESS,
            provider=session.provider,
            user_id=session.user_id,
            ip=_client_ip(request),
        )
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "token_type": "Bearer",
            "expires_at": session.expires_at,
            "provider": session.provider,
            "user_id": session.user_id,
        }

    if unreachable is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Auth provider {unreachable!r} unreachable",
        )
    audit_log(
        AuditEvent.REFRESH_FAILURE,
        reason="all_providers_rejected_rt",
        ip=_client_ip(request),
    )
    return JSONResponse(
        {
            "error": "session_expired",
            "detail": "Refresh token expired or invalid; start a new sign-in.",
        },
        status_code=401,
    )
