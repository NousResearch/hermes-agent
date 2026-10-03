"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

Any machine-credential provider plugs in here. A route opts in by registering its exact path via
:func:`register_token_route`, or a whole router subtree via :func:`register_token_route_prefix`
(for parameterised paths exact matching cannot cover); only registered paths are token-authable,
so the auth surface of existing routes never widens. A registration may demand a scope the
verified principal must carry (403 otherwise), so one service credential cannot open another
surface's routes. :func:`token_auth_middleware` runs OUTERMOST (installed last) and owns the
decision for a token route: a recognised token attaches ``request.state.token_principal``
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

# Exact paths and path prefixes, each mapped to an optional required scope.
_token_routes: dict[str, Optional[str]] = {}
_token_route_prefixes: dict[str, Optional[str]] = {}
_lock = threading.Lock()

# Sentinel distinguishing "no registered route matched" from a matched route
# with no scope requirement (None).
_NO_MATCH = object()


def register_token_route(path: str, *, scope: Optional[str] = None) -> None:
    """Mark ``path`` (exact match) as token-authable. Idempotent; does NOT make the route public.
    With ``scope`` set, a verified principal must also carry it in ``TokenPrincipal.scopes`` or
    the request gets 403 — one stacked service credential cannot open every token route."""
    with _lock:
        _token_routes[path] = scope


def register_token_route_prefix(prefix: str, *, scope: Optional[str] = None) -> None:
    """Mark every path under ``prefix`` as token-authable — for routers with parameterised
    paths (``/tasks/{id}``) that exact matching cannot cover. Otherwise the same semantics as
    :func:`register_token_route`, including the optional required ``scope``."""
    with _lock:
        _token_route_prefixes[prefix.rstrip("/") + "/"] = scope


def _match_token_route(path: str):
    """Required scope for ``path`` (``None`` = any verified principal), or ``_NO_MATCH``.
    Exact registrations win; among prefixes the longest match wins, so the scope boundary never
    depends on plugin load order (a broad prefix cannot shadow a narrower scoped one)."""
    with _lock:
        if path in _token_routes:
            return _token_routes[path]
        matches = [p for p in _token_route_prefixes if path.startswith(p)]
        if matches:
            return _token_route_prefixes[max(matches, key=len)]
    return _NO_MATCH


def is_token_route(path: str) -> bool:
    """True if ``path`` is token-authable (exact or prefix registration)."""
    return _match_token_route(path) is not _NO_MATCH


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()
        _token_route_prefixes.clear()


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
    required_scope = _match_token_route(path)
    if required_scope is _NO_MATCH:
        return await call_next(request)
    principal, unreachable = authenticate_token(request)
    if principal is not None:
        if required_scope and required_scope not in (principal.scopes or ()):
            # Authenticated, but the credential lacks this route's capability
            # — e.g. the drain secret presented to the kanban API. 403 (not
            # 401) so a mis-scoped integration reads "wrong credential for
            # this surface" rather than "bad token".
            audit_log(
                AuditEvent.TOKEN_AUTH_FAILURE,
                provider=principal.provider,
                reason="missing_scope",
                path=path,
                ip=_client_ip(request),
            )
            return JSONResponse(
                {"error": "forbidden", "detail": "Forbidden"},
                status_code=403,
            )
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
