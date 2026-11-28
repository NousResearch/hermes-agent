"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

Any machine-credential provider plugs in here. Plugins bind exact paths to a provider and scopes via
``ctx.register_dashboard_token_route``; legacy :func:`register_token_route` stacks only providers
without bound policies, so a plugin's token cannot authorize unrelated token routes.
Only registered paths are token-authable, so the auth surface of existing routes never widens.
:func:`token_auth_middleware` runs OUTERMOST (installed last) and
owns the decision for a token route: a recognised token attaches ``request.state.token_principal``
+ ``token_authenticated`` (the cookie gates honour that flag and never bounce to /login);
otherwise 401, 403 for insufficient scopes, or 503 when a provider's backing store was unreachable.
Fails closed.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal
from hermes_cli.dashboard_auth.request_utils import (
    client_ip as _client_ip, extract_bearer as extract_bearer_token, unreachable_response)
from hermes_cli.dashboard_auth.registry import snapshot_registration

if TYPE_CHECKING:
    from hermes_cli.plugins_ledger import PluginRegistration

_log = logging.getLogger(__name__)

_token_routes: set[str] = set()  # exact paths that accept bearer-token auth
_token_route_policies: dict[str, TokenRoutePolicy] = {}  # launch-owned, never profile overlays
_lock = threading.Lock()


@dataclass(frozen=True)
class TokenRoutePolicy:
    """One exact path's owning provider, required scopes and launch home."""

    registration: PluginRegistration
    required_scopes: frozenset[str]
    home: str


def bind_token_route(path: str, policy: TokenRoutePolicy, *, previous: Optional[TokenRoutePolicy] = None) -> None:
    """Supersede an equal policy or this plugin's parked carryover, never another live policy."""
    with _lock:
        current = _token_route_policies.get(path)
        if current is not None and current != policy and current is not previous:
            raise ValueError(f"conflicting token route policy for {path!r}")
        _token_route_policies[path] = policy


def restore_token_route(path: str, current: TokenRoutePolicy) -> bool:
    """Identity-conditional inverse for the plugin registration ledger."""
    with _lock:
        if _token_route_policies.get(path) is not current:
            return False
        _token_route_policies.pop(path)
        return True


def register_token_route(path: str) -> None:
    """Mark ``path`` (exact match) as token-authable. Idempotent; does NOT make the route public."""
    with _lock:
        _token_routes.add(path)


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable (exact match)."""
    with _lock:
        return path in _token_routes or path in _token_route_policies


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()
        _token_route_policies.clear()


def authenticate_token(request: Request) -> tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try unbound token providers against the request's bearer token. Returns ``(principal, None)``
    on success; ``(None, None)`` for no token or no recogniser (401); ``(None, name)`` when no
    provider accepted it AND at least one was unreachable (caller surfaces 503). Never raises."""
    token = extract_bearer_token(request)
    if not token:
        return None, None
    with _lock:
        bound_providers = {policy.registration.key for policy in _token_route_policies.values()}
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        if provider.name in bound_providers:
            continue
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning("dashboard-auth: token provider %r unreachable during verify: %s",
                         provider.name, e)
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:
            _log.warning("dashboard-auth: token provider %r raised during verify: %s",
                         provider.name, e)
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


def _authenticate_bound_token(
    request: Request, policy: TokenRoutePolicy,
) -> tuple[Optional[TokenPrincipal], Optional[str]]:
    """Only the process-global owning provider may vouch for a bound route."""
    token = extract_bearer_token(request)
    if not token:
        return None, "missing_token"
    name = policy.registration.key
    provider = snapshot_registration(name)
    if provider is None or not provider.supports_token:
        return None, "token_provider_unavailable"
    if provider is not policy.registration.value:
        return None, "token_provider_replaced"
    try:
        principal = provider.verify_token(token=token)
    except ProviderError as exc:
        _log.warning("dashboard-auth: token provider %r unreachable during verify: %s", name, exc)
        return None, "provider_unreachable"
    except Exception as exc:  # health: allow BLE001 -- third-party verifier failures must fail closed
        _log.warning("dashboard-auth: token provider %r raised during verify: %s", name, exc)
        return None, "provider_error"
    if principal is None:
        return None, "token_rejected"
    if not isinstance(principal, TokenPrincipal) or isinstance(principal.scopes, (str, bytes)):
        return None, "invalid_principal"
    if principal.provider != name:
        return None, "provider_mismatch"
    from hermes_cli.web_server_plugin_profiles import selected_plugin_profile
    from hermes_cli.web_server_profiles import _is_current_profile, _resolve_profile_dir
    from hermes_constants import get_hermes_home, hermes_home_key
    # Resolve routing's home without activating multiplex or hydrating a secondary's secrets.
    try:
        profile = selected_plugin_profile(request.scope)
        home = get_hermes_home() if _is_current_profile(profile) else _resolve_profile_dir(profile.strip())
        if hermes_home_key(home) != policy.home:
            return None, "profile_scope"
    except Exception:  # health: allow BLE001 -- profile resolution errors must be audited refusals
        return None, "profile_scope"
    try:
        covered = policy.required_scopes.issubset(principal.scopes)
    except TypeError:
        return None, "invalid_principal"
    if not covered:
        return None, "insufficient_scope"
    return principal, None


async def token_auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Pass-through for unregistered paths; for a token route, valid token -> attach principal +
    flag, unreachable -> 503, insufficient scope -> 403, else 401."""
    path = request.url.path
    with _lock:
        policy = _token_route_policies.get(path)
        legacy = path in _token_routes
    if policy is None and not legacy:
        return await call_next(request)
    if policy is None:
        principal, unreachable = authenticate_token(request)
        reason = "provider_unreachable" if unreachable else "no_provider_recognises_token"
    else:
        principal, reason = _authenticate_bound_token(request, policy)
        unreachable = policy.registration.key if reason == "provider_unreachable" else None
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
        AuditEvent.TOKEN_AUTH_FAILURE, reason=reason, path=path, ip=_client_ip(request),
        **({"provider": policy.registration.key} if policy is not None else {}))
    if reason == "insufficient_scope":
        return JSONResponse({"error": "insufficient_scope", "detail": "Forbidden"}, status_code=403)
    return JSONResponse({"error": "unauthenticated", "detail": "Unauthorized"}, status_code=401)
