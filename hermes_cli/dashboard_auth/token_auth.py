"""Route-agnostic bearer-token auth seam for the dashboard.

Any machine-credential provider plugs in here. A route opts in by registering its exact path via
:func:`register_token_route`; only registered paths are token-authable, so the auth surface of
existing routes never widens. :func:`token_auth_middleware` runs OUTERMOST (installed last) and
owns the decision for a token route: a recognised token attaches ``request.state.token_principal``
+ ``token_authenticated`` (the cookie gates honour that flag and never bounce to /login);
otherwise 401, or 503 when a provider's backing store was unreachable. Fails closed.

WebSocket services reuse the same registry, but only the exact registered path may accept a
Bearer upgrade credential. Browser/query credentials stay unchanged for the
legacy WebSocket paths; a present bearer header never falls back to query auth when it is malformed,
unrecognised, or lacks required scope.
"""
from __future__ import annotations

import logging
import threading
from typing import Awaitable, Callable, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal
from hermes_cli.dashboard_auth.request_utils import (
    client_ip as _client_ip, extract_bearer as extract_bearer_token, unreachable_response)

_log = logging.getLogger(__name__)
_token_routes: dict[str, tuple[str, ...]] = {}
_lock = threading.Lock()
NOT_HANDLED = "not_handled"


def _normalise_scopes(required_scopes: tuple[str, ...] | tuple[()] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(scope.strip() for scope in required_scopes if str(scope).strip())


def _has_required_scopes(principal: TokenPrincipal, required_scopes: tuple[str, ...]) -> bool:
    if not required_scopes:
        return True
    granted = set(principal.scopes)
    return all(scope in granted for scope in required_scopes)


def _audit_token_failure(
    reason: str,
    *,
    path: str,
    ip: str,
    provider: Optional[str] = None,
) -> None:
    audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason=reason, provider=provider, path=path, ip=ip)


def _audit_token_success(*, path: str, ip: str, provider: str) -> None:
    audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, reason="accepted", provider=provider, path=path, ip=ip)


def register_token_route(path: str, *, required_scopes: tuple[str, ...] = ()) -> None:
    """Mark ``path`` (exact match) as token-authable.

    ``required_scopes`` is backwards-compatible metadata: unscoped routes keep working, while a
    route that lists scopes requires the presented principal to satisfy all of them.
    Re-registering the same path is monotonic: scopes are unioned so later empty
    registrations cannot downgrade an earlier protected route.
    """
    scopes = _normalise_scopes(required_scopes)
    with _lock:
        existing = _token_routes.get(path, ())
        merged = tuple(dict.fromkeys((*existing, *scopes)))
        _token_routes[path] = merged


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable (exact match)."""
    with _lock:
        return path in _token_routes


def token_route_required_scopes(path: str) -> tuple[str, ...]:
    """Required scopes for a registered token route, or ``()`` when unregistered/unscoped."""
    with _lock:
        return _token_routes.get(path, ())


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()


def _extract_bearer_value(header_value: str) -> str:
    parts = header_value.split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def authenticate_token_value(token: str) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against ``token``.

    Returns ``(principal, None)`` on success; ``(None, None)`` for no recogniser; ``(None, name)``
    when no provider accepted it AND at least one was unreachable. Never raises.
    """
    if not token:
        return None, None
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning("dashboard-auth: token provider %r unreachable during verify: %s",
                         provider.name, e)
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:  # noqa: BLE001 — a buggy provider must not 500 the gate
            _log.warning("dashboard-auth: token provider %r raised during verify: %s",
                         provider.name, e)
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


def authenticate_token(request: Request) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Request wrapper around :func:`authenticate_token_value` for HTTP callers."""
    token = extract_bearer_token(request)
    if not token:
        return None, None
    return authenticate_token_value(token)


def _request_scope_reason(
    principal: Optional[TokenPrincipal],
    *,
    path: str,
    required_scopes: tuple[str, ...],
) -> Optional[str]:
    if principal is None:
        return None
    if _has_required_scopes(principal, required_scopes):
        return None
    return "insufficient_scope"


async def token_auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Pass-through for unregistered paths; for a token route, valid token -> attach principal +
    flag, unreachable -> 503, else 401. A malformed bearer on a registered route fails closed and
    does not consult any non-bearer credential path."""
    path = request.url.path
    if not is_token_route(path):
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if not auth_header:
        _audit_token_failure("missing_authorization", path=path, ip=_client_ip(request))
        return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)

    token = _extract_bearer_value(auth_header)
    ip = _client_ip(request)
    if not token:
        _audit_token_failure("malformed_authorization", path=path, ip=ip)
        return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)

    principal, unreachable = authenticate_token_value(token)
    if principal is not None:
        required_scopes = token_route_required_scopes(path)
        scope_reason = _request_scope_reason(principal, path=path, required_scopes=required_scopes)
        if scope_reason is None:
            request.state.token_principal = principal
            request.state.token_authenticated = True
            _audit_token_success(path=path, ip=ip, provider=principal.provider)
            return await call_next(request)
        _audit_token_failure(
            scope_reason, path=path, ip=ip, provider=principal.provider,
        )
        return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)

    if unreachable:
        _audit_token_failure("provider_unreachable", path=path, ip=ip, provider=unreachable)
        return unreachable_response(unreachable)

    _audit_token_failure("token_unrecognised", path=path, ip=ip)
    return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)


def _ws_auth_reason(ws) -> tuple[Optional[str], str]:
    """Validate WebSocket bearer auth for registered token routes.

    Returns ``(None, "bearer")`` when a registered route accepts the bearer;
    returns ``(NOT_HANDLED, "bearer")`` when the request has no Authorization
    header so the caller can delegate to the canonical browser/query gate;
    otherwise returns a failure reason token plus ``bearer``.
    """
    path = ws.url.path
    auth_header = ws.headers.get("authorization", "") or ""
    if not auth_header:
        return NOT_HANDLED, "bearer"

    ip = ws.client.host if getattr(ws, "client", None) else ""
    if not is_token_route(path):
        _audit_token_failure("route_not_registered", path=path, ip=ip)
        return "route_not_registered", "bearer"

    token = _extract_bearer_value(auth_header)
    if not token:
        _audit_token_failure("malformed_authorization", path=path, ip=ip)
        return "malformed_authorization", "bearer"

    principal, unreachable = authenticate_token_value(token)
    if principal is not None:
        required_scopes = token_route_required_scopes(path)
        if _has_required_scopes(principal, required_scopes):
            _audit_token_success(path=path, ip=ip, provider=principal.provider)
            return None, "bearer"
        _audit_token_failure("insufficient_scope", path=path, ip=ip, provider=principal.provider)
        return "insufficient_scope", "bearer"

    if unreachable:
        _audit_token_failure("provider_unreachable", path=path, ip=ip, provider=unreachable)
        return "provider_unreachable", "bearer"

    _audit_token_failure("token_unrecognised", path=path, ip=ip)
    return "token_unrecognised", "bearer"
