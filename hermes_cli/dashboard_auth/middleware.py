"""Auth-gate middleware for the dashboard.

Engaged when ``app.state.auth_required is True``; a no-op otherwise (loopback
mode is handled by the legacy ``_SESSION_TOKEN`` ``auth_middleware``). Allows
the auth-bootstrap routes and static assets through unauthenticated; for
everything else demands a bearer token or a valid session cookie and attaches
the verified :class:`Session` to ``request.state.session``. HTML routes are
redirected to ``/login``; ``/api/*`` routes get a 401 JSON envelope.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from hermes_cli.dashboard_auth import list_session_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, RefreshExpiredError
from hermes_cli.dashboard_auth.cookies import (
    clear_session_cookies, clear_sso_attempt_cookie, detect_https, read_session_cookies,
    read_session_provider, read_sso_attempt_cookie, set_session_cookies,
    set_session_provider_cookie, set_sso_attempt_cookie)
from hermes_cli.dashboard_auth.prefix import prefix_from_request
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from hermes_cli.dashboard_auth.request_utils import (
    access_token_max_age as _expires_in_seconds, client_ip as _client_ip,
    extract_bearer as _extract_bearer, is_safe_next_path, scan_session_providers,
    unreachable_response)

_log = logging.getLogger(__name__)

# Prefix-matched bypass list: auth bootstrap routes and static asset mounts. ``/assets/`` with
# the trailing slash matches ``/assets/foo.css`` but not ``/assetsleak``.
_GATE_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/auth/login", "/auth/callback", "/auth/native/authorize", "/auth/native/token",
    "/auth/native/refresh", "/auth/password-login", "/auth/logout", "/login",
    "/api/auth/providers", "/api/mcp/oauth/callback/",
    "/assets/", "/favicon.ico", "/ds-assets/", "/fonts/", "/fonts-terminal/")


def _path_is_public(path: str) -> bool:
    """:data:`PUBLIC_API_PATHS` (shared with the legacy middleware) matched exactly so
    ``/api/status`` never exposes ``/api/status/extension``; :data:`_GATE_PUBLIC_PREFIXES`
    prefix-matched."""
    return path in PUBLIC_API_PATHS or any(
        path == p or path.startswith(p) for p in _GATE_PUBLIC_PREFIXES)


def _safe_next_target(request: Request) -> str:
    """URL-encoded ``next`` value for the login redirect, or ``""``. Only same-origin paths outside
    the auth flow and ``/api`` are kept (query preserved); dropped deep links fall back to the
    SPA's ``sessionStorage["hermes.lastLocation"]``."""
    path = request.url.path
    if not path or not is_safe_next_path(path):
        return ""
    query = request.url.query
    return quote(f"{path}?{query}" if query else path, safe="")


def _unauth_response(request: Request, *, reason: str) -> Response:
    """API routes -> 401 JSON with ``login_url``; HTML routes -> 302 -> /login. fetch() follows a
    302 opaquely into the cross-origin OAuth dance, so API routes never get redirects; the SPA's
    401 handler navigates to ``login_url`` on ``unauthenticated`` / ``session_expired``."""
    next_param = _safe_next_target(request)
    prefix = prefix_from_request(request)
    login_url = f"{prefix}/login?next={next_param}" if next_param else f"{prefix}/login"
    if request.url.path.startswith("/api/"):
        expired = reason == "invalid_or_expired_session"
        return JSONResponse(
            {"error": "session_expired" if expired else "unauthenticated", "detail": "Unauthorized",
             "reason": reason, "login_url": login_url}, status_code=401)
    return RedirectResponse(url=login_url, status_code=302)


def _auto_sso_response(request: Request) -> Response | None:
    """302 straight to ``/auth/login`` on an unauthenticated HTML load, or ``None``.

    Only for a document load (not ``/api/*``) when exactly one interactive OAuth-style provider is
    registered (a password provider must render the form) and the one-shot loop-guard cookie is
    absent — a present marker means the portal had no session last time: clear it and fall back
    to ``/login`` rather than ping-pong. Convenience, not a security check.
    """
    if request.url.path.startswith("/api/"):
        return None
    if read_sso_attempt_cookie(request):
        resp = _unauth_response(request, reason="no_cookie")
        clear_sso_attempt_cookie(resp, prefix=prefix_from_request(request))
        return resp
    providers = list_session_providers()
    if len(providers) != 1 or getattr(providers[0], "supports_password", False):
        return None
    provider = providers[0]
    prefix = prefix_from_request(request)
    next_param = _safe_next_target(request)
    auth_login = f"{prefix}/auth/login?provider={quote(provider.name, safe='')}"
    if next_param:
        auth_login = f"{auth_login}&next={next_param}"
    resp = RedirectResponse(url=auth_login, status_code=302)
    set_sso_attempt_cookie(resp, use_https=detect_https(request), prefix=prefix)
    audit_log(AuditEvent.LOGIN_START, provider=provider.name, reason="auto_sso",
              ip=_client_ip(request))
    return resp


def _verify_access_token(
    request: Request, *, access_token: str, provider_hint: str | None = None, audit: bool = True):
    """Run ``verify_session`` across the provider stack; Session or ``None``. ``audit=False`` is
    the native-app bearer path (no cookie, no server-side refresh — the desktop rotates via
    ``/auth/native/refresh``); 503-on-outage semantics come from :func:`scan_session_providers`."""
    def _audit_unreachable(provider):
        if audit:
            audit_log(AuditEvent.SESSION_VERIFY_FAILURE, provider=provider.name,
                      reason="provider_unreachable", ip=_client_ip(request))

    return scan_session_providers(
        provider_hint, lambda p: p.verify_session(access_token=access_token),
        phase="verify" if audit else "bearer verify", log=_log, on_unreachable=_audit_unreachable)


async def _serve_refreshed(request: Request, call_next, new_session, provider: str) -> Response:
    """Serve the request under a just-rotated session and write the rotated cookies back. Writing
    the ROTATED RT is mandatory: Portal runs reuse detection, so replaying the stale RT would
    revoke the session."""
    request.state.session = new_session
    response = await call_next(request)
    set_session_cookies(
        response, access_token=new_session.access_token, refresh_token=new_session.refresh_token,
        access_token_expires_in=_expires_in_seconds(new_session), use_https=detect_https(request),
        prefix=prefix_from_request(request), provider=provider)
    audit_log(AuditEvent.REFRESH_SUCCESS, provider=provider, user_id=new_session.user_id,
              ip=_client_ip(request))
    return response


def _session_expired_response(request: Request) -> Response:
    """Refresh failed (or no RT): structured 401/redirect and clear the dead cookies under the
    active prefix so the deletion Path matches the set Path."""
    audit_log(AuditEvent.SESSION_VERIFY_FAILURE, reason="no_provider_recognises",
              ip=_client_ip(request))
    response = _unauth_response(request, reason="invalid_or_expired_session")
    clear_session_cookies(response, prefix=prefix_from_request(request))
    return response


async def gated_auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Engaged only when ``app.state.auth_required is True``."""
    if not getattr(request.app.state, "auth_required", False):
        return await call_next(request)
    # Already authenticated by the token-auth seam (service caller on a registered token
    # route): not a cookie session, must not bounce to /login.
    if getattr(request.state, "token_authenticated", False) or _path_is_public(request.url.path):
        return await call_next(request)
    # RFC 8252 native-app bearer path: the same provider-minted access token the cookie flow
    # stores, verified with the same provider stack, no cookie read or set. A presented-but-
    # invalid bearer gets the structured 401 so the desktop refreshes/re-logs instead of
    # following a cookie redirect.
    bearer = _extract_bearer(request)
    if bearer:
        try:
            bearer_session = _verify_access_token(request, access_token=bearer, audit=False)
        except ProviderError as e:
            return unreachable_response(str(e))
        if bearer_session is not None:
            request.state.session = bearer_session
            return await call_next(request)
        return _unauth_response(request, reason="invalid_or_expired_session")

    at, _rt = read_session_cookies(request)
    provider_hint = read_session_provider(request)
    if not at and not _rt:
        # No session at all: try the silent portal bounce before /login.
        auto = _auto_sso_response(request)
        return auto if auto is not None else _unauth_response(request, reason="no_cookie")
    # An absent AT with a present RT is the COMMON expiry case (the AT cookie's Max-Age tracks
    # the token TTL, so the browser evicts it first) — skip straight to refresh.
    session = None
    if at:
        try:
            session = _verify_access_token(request, access_token=at, provider_hint=provider_hint)
        except ProviderError as e:
            return unreachable_response(str(e))
    if session is None:
        # Rotate via the refresh token before forcing re-login; on success the request is
        # served transparently with the rotated cookies re-set.
        try:
            refreshed = _attempt_refresh(request, refresh_token=_rt, provider_hint=provider_hint)
        except ProviderError as e:
            # Uncertain (provider unreachable), not rejected: keep the cookies.
            return unreachable_response(str(e))
        if refreshed is None:
            return _session_expired_response(request)
        return await _serve_refreshed(request, call_next, *refreshed)

    request.state.session = session
    response = await call_next(request)
    if not provider_hint and session.provider:
        set_session_provider_cookie(
            response, provider=session.provider, use_https=detect_https(request),
            prefix=prefix_from_request(request))
    return response


def _attempt_refresh(request: Request, *, refresh_token, provider_hint: str | None = None):
    """Rotate an expired session via the refresh token; ``(Session, provider_name)`` or ``None``.
    ``RefreshExpiredError`` rejects that candidate only (Basic raises it for foreign opaque tokens
    too); if none succeeds and any raised ``ProviderError`` it is re-raised so the caller returns
    503 without clearing cookies."""
    if not refresh_token:
        return None

    def _audit_failure(reason):
        return lambda provider: audit_log(
            AuditEvent.REFRESH_FAILURE, provider=provider.name, reason=reason,
            ip=_client_ip(request))

    def _refresh(provider):
        new_session = provider.refresh_session(refresh_token=refresh_token)
        return None if new_session is None else (new_session, provider.name)

    return scan_session_providers(
        provider_hint, _refresh, phase="refresh", log=_log, swallow=(RefreshExpiredError,),
        on_swallow=_audit_failure("refresh_expired"),
        on_unreachable=_audit_failure("provider_unreachable"))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'DashboardAuthProvider': ('hermes_cli.dashboard_auth.base', 'DashboardAuthProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
