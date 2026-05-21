"""Auth-gate middleware for the dashboard.

Engaged when ``app.state.auth_required is True``. The gate's job:

  1. Allow a small set of routes through unauthenticated (login page,
     ``/auth/*`` OAuth round trip, ``/api/auth/providers``, static
     assets).
  2. For everything else, demand a valid session cookie and attach the
     verified :class:`Session` to ``request.state.session``.
  3. On HTML routes, redirect missing/invalid cookies to ``/login``.
     On ``/api/*`` routes, return 401 JSON.

The middleware is a no-op when ``auth_required`` is False (loopback
mode); the legacy ``_SESSION_TOKEN`` ``auth_middleware`` handles those
binds.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from hermes_cli.dashboard_auth import list_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError
from hermes_cli.dashboard_auth.cookies import read_session_cookies

_log = logging.getLogger(__name__)

# Paths that bypass the auth gate. Order matters: prefix match.
_GATE_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/auth/login",
    "/auth/callback",
    "/auth/logout",
    "/login",
    "/api/auth/providers",
    "/assets/",
    "/favicon.ico",
    "/ds-assets/",
    "/fonts/",
    "/fonts-terminal/",
)


def _path_is_public(path: str) -> bool:
    return any(
        path == prefix or path.startswith(prefix)
        for prefix in _GATE_PUBLIC_PREFIXES
    )


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _unauth_response(path: str, *, reason: str) -> Response:
    """API routes → 401 JSON; HTML routes → 302 → /login."""
    if path.startswith("/api/"):
        return JSONResponse(
            {"detail": "Unauthorized", "reason": reason},
            status_code=401,
        )
    return RedirectResponse(url="/login", status_code=302)


async def gated_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Engaged only when ``app.state.auth_required is True``.

    No-op pass-through in loopback mode so the legacy auth_middleware can
    handle those binds via ``_SESSION_TOKEN``.
    """
    if not getattr(request.app.state, "auth_required", False):
        return await call_next(request)

    path = request.url.path
    if _path_is_public(path):
        return await call_next(request)

    at, _rt = read_session_cookies(request)
    if not at:
        return _unauth_response(path, reason="no_cookie")

    # Try every registered provider's verify_session in turn. Providers
    # MUST return None for tokens they don't recognise (not raise). This
    # lets multiple providers stack — the first one that recognises a
    # token wins.
    session = None
    for provider in list_providers():
        try:
            session = provider.verify_session(access_token=at)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: provider %r unreachable during verify: %s",
                provider.name, e,
            )
            audit_log(
                AuditEvent.SESSION_VERIFY_FAILURE,
                provider=provider.name,
                reason="provider_unreachable",
                ip=_client_ip(request),
            )
            return JSONResponse(
                {"detail": f"Auth provider {provider.name!r} unreachable"},
                status_code=503,
            )
        if session is not None:
            break

    if session is None:
        audit_log(
            AuditEvent.SESSION_VERIFY_FAILURE,
            reason="no_provider_recognises",
            ip=_client_ip(request),
        )
        return _unauth_response(path, reason="invalid_or_expired_session")

    request.state.session = session
    return await call_next(request)
