"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

Any machine-credential provider plugs in here. A route opts in by registering its exact path via
:func:`register_token_route`; only registered paths are token-authable, so the auth surface of
existing routes never widens. :func:`token_auth_middleware` runs OUTERMOST (installed last) and
owns the decision for a token route: a recognised token attaches ``request.state.token_principal``
+ ``token_authenticated`` (the cookie gates honour that flag and never bounce to /login);
otherwise 401, or 503 when a provider's backing store was unreachable. Fails closed.
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

_token_routes: set[str] = set()  # exact paths that accept bearer-token auth
_lock = threading.Lock()


def register_token_route(path: str) -> None:
    """Mark ``path`` (exact match) as token-authable. Idempotent; does NOT make the route public."""
    with _lock:
        _token_routes.add(path)


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable (exact match)."""
    with _lock:
        return path in _token_routes


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()


def authenticate_token(request: Request) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token. Returns ``(principal, None)``
    on success; ``(None, None)`` for no token or no recogniser (401); ``(None, name)`` when no
    provider accepted it AND at least one was unreachable (caller surfaces 503). Never raises."""
    token = extract_bearer_token(request)
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


async def token_auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Pass-through for unregistered paths; for a token route, valid token -> attach principal +
    flag, unreachable -> 503, else 401."""
    path = request.url.path
    if not is_token_route(path):
        return await call_next(request)
    principal, unreachable = authenticate_token(request)
    if principal is not None:
        request.state.token_principal = principal
        request.state.token_authenticated = True
        return await call_next(request)
    if unreachable:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE, provider=unreachable, reason="provider_unreachable",
            path=path, ip=_client_ip(request))
        return unreachable_response(unreachable)

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE, reason="no_provider_recognises_token", path=path,
        ip=_client_ip(request))
    return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)
