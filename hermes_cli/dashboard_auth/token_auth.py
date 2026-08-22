"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

This is the generic API-token capability (decisions.md Q-C): a reusable seam
that ANY service-to-service / machine-credential provider plugs into, NOT a
drain-specific hook. The drain bearer-secret plugin is merely the first
consumer.

How it fits the existing auth framework:

  * The interactive gate (``gated_auth_middleware``) authenticates a human
    via a session cookie on every non-public route. A service caller has no
    cookie — it presents a bearer token in the ``Authorization`` header on a
    single request. That is what this seam verifies.

  * A route opts in by registering its exact path via
    :func:`register_token_route`. Only registered paths are token-authable;
    everything else is untouched, so this can never accidentally widen the
    auth surface of an existing route. A route may explicitly allow requests
    without a bearer, or the exact legacy dashboard session bearer, to fall
    through to the existing gate.

  * :func:`token_auth_middleware` runs OUTERMOST (installed last in
    ``web_server.py``). For a token route it fully owns the auth decision:
    authenticate via the stacked token providers, attach the verified
    :class:`~hermes_cli.dashboard_auth.base.TokenPrincipal` to
    ``request.state.token_principal`` + set ``request.state.token_authenticated``,
    and pass through; otherwise reject (401 unauthenticated, or 503 when a
    provider's backing store was unreachable). The downstream cookie/session
    gates honour ``token_authenticated`` and skip enforcement, so a
    token-authed service request is never bounced to ``/login``.

  * Fails closed: a token route with no registered token provider, no token,
    or an unrecognised token gets 401 — never an open pass-through.

Provider stacking mirrors ``verify_session``: each ``supports_token`` provider
is consulted in registration order until one returns a principal. A provider
that doesn't recognise the token returns ``None`` and the seam moves on; a
provider whose backing store is unreachable raises ``ProviderError``, which the
seam remembers and surfaces as 503 only if NO provider accepts the token.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal

_log = logging.getLogger(__name__)

# Exact paths that accept non-interactive bearer-token auth. A route registers
# itself here at import/startup; the seam only acts on registered paths. The
# rule is deliberately exact-path only: there is no prefix or template match.
@dataclass(frozen=True)
class _TokenRouteRule:
    methods: frozenset[str] | None = None
    required_scope: str | None = None
    loopback_only: bool = False
    allow_session_fallback: bool = False


_token_routes: dict[str, _TokenRouteRule] = {}
_lock = threading.Lock()


def _normalise_methods(methods) -> frozenset[str] | None:
    if methods is None:
        return None
    if isinstance(methods, str):
        methods = (methods,)
    normalised = []
    for method in methods:
        if not isinstance(method, str) or not method.strip():
            raise ValueError("token route methods must be non-empty strings")
        normalised.append(method.strip().upper())
    if not normalised:
        raise ValueError("token route methods must not be empty")
    return frozenset(normalised)


def register_token_route(
    path: str,
    *,
    methods=None,
    required_scope: str | None = None,
    loopback_only: bool = False,
    allow_session_fallback: bool = False,
) -> None:
    """Mark ``path`` (exact match) as token-authable.

    ``methods=None``, ``required_scope=None``, ``loopback_only=False``, and
    ``allow_session_fallback=False`` preserve the original route-agnostic
    behaviour. Configured methods are normalised to uppercase. Duplicate
    identical registrations are idempotent; a conflicting registration fails
    rather than silently weakening a rule.

    Call at module import / app setup so the seam knows which routes to guard.
    Registering a route does NOT make it public — it makes it authenticate by
    token instead of by session cookie.
    """
    rule = _TokenRouteRule(
        methods=_normalise_methods(methods),
        required_scope=(
            required_scope.strip() if required_scope is not None else None
        ),
        loopback_only=bool(loopback_only),
        allow_session_fallback=bool(allow_session_fallback),
    )
    if rule.required_scope == "":
        raise ValueError("token route required_scope must be non-empty")
    with _lock:
        previous = _token_routes.get(path)
        if previous is None:
            _token_routes[path] = rule
            return
        if previous == rule:
            return
        raise ValueError(
            f"conflicting token route registration for exact path {path!r}: "
            f"existing={previous!r}, requested={rule!r}"
        )


def is_token_route(path: str, method: str | None = None) -> bool:
    """True if ``path`` (and optional ``method``) is token-authable."""
    with _lock:
        rule = _token_routes.get(path)
    if rule is None:
        return False
    if method is None or rule.methods is None:
        return True
    return method.strip().upper() in rule.methods


def _route_for_request(request: Request) -> Optional[_TokenRouteRule]:
    """Return the applicable rule, or ``None`` to fall through to normal auth."""
    path = request.url.path
    with _lock:
        rule = _token_routes.get(path)
    if rule is None:
        return None

    method = getattr(request, "method", "GET")
    if rule.methods is not None and method.upper() not in rule.methods:
        return None

    if rule.loopback_only:
        app_state = getattr(getattr(request, "app", None), "state", None)
        bound_host = getattr(app_state, "bound_host", None)
        peer = getattr(getattr(request, "client", None), "host", None)
        if not _is_loopback_bind(bound_host) or not _is_loopback_peer(peer):
            return None
    return rule


def _is_loopback_bind(value: object) -> bool:
    """Return True only for a known loopback bind value.

    ``localhost`` is a deliberate, existing dashboard bind alias. Other
    hostnames are not resolved here: DNS is not a trustworthy auth boundary.
    Literal IPv4/IPv6 addresses use the standard library's loopback semantics,
    including the complete IPv4 loopback range and IPv6 loopback forms.
    """
    if not isinstance(value, str):
        return False
    host = value.strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_loopback_peer(value: object) -> bool:
    """Return True only for a literal, identifiable loopback request peer."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return ipaddress.ip_address(value.strip()).is_loopback
    except ValueError:
        return False


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def extract_bearer_token(request: Request) -> str:
    """Return the bearer token from the ``Authorization`` header, or "".

    Accepts ``<scheme> <token>`` where scheme is "bearer" (case-insensitive).
    Returns an empty string for a missing/malformed header or a non-bearer
    scheme — the caller treats "" as "no token presented".
    """
    auth = request.headers.get("authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def _has_valid_legacy_session_bearer(request: Request) -> bool:
    """Return True only for the exact native dashboard session bearer.

    This compatibility path is deliberately kept behind a route rule's
    ``allow_session_fallback`` opt-in. It does not accept arbitrary bearer
    headers and it reuses the dashboard's existing process-local session token
    without logging or exposing the secret.
    """
    token = extract_bearer_token(request)
    if not token:
        return False
    from hermes_cli import web_server

    return hmac.compare_digest(
        token.encode("utf-8"), web_server._SESSION_TOKEN.encode("utf-8")
    )


def authenticate_token(
    request: Request,
    *,
    required_scope: str | None = None,
) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token.

    Returns ``(principal, unreachable_provider_name)``:
      * ``(TokenPrincipal, None)`` — a provider recognised and accepted the
        token. With ``required_scope``, providers lacking that scope are
        skipped until a matching principal is found.
      * ``(None, None)`` — no token, or no provider recognised it (reject 401).
      * ``(TokenPrincipal, None)`` lacking ``required_scope`` — one or more
        providers recognised the token, but none had the required scope
        (reject 403).
      * ``(None, name)`` — no provider accepted it AND at least one provider's
        backing store was unreachable (the caller surfaces 503, not 401, so a
        transient outage doesn't read as "bad credentials").

    Never raises: a provider ``ProviderError`` is caught and remembered.
    """
    token = extract_bearer_token(request)
    if not token:
        return None, None
    unreachable: Optional[str] = None
    missing_scope: Optional[TokenPrincipal] = None
    for provider in list_token_providers():
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: token provider %r unreachable during verify: %s",
                provider.name, e,
            )
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:  # noqa: BLE001 — a buggy provider must not 500 the gate
            _log.warning(
                "dashboard-auth: token provider %r raised during verify: %s",
                provider.name, e,
            )
            continue
        if principal is not None:
            if required_scope is None or required_scope in principal.scopes:
                return principal, None
            if missing_scope is None:
                missing_scope = principal
    if missing_scope is not None:
        return missing_scope, None
    return None, unreachable


async def token_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Outermost auth seam for token-authable routes.

    No-op pass-through for any path not registered via
    :func:`register_token_route`. For an applicable exact route rule, token
    auth is the only accepted scheme:

      * valid token  → attach principal + ``token_authenticated`` flag, pass through.
      * valid token without the route's scope  → 403, without token auth.
      * configured session fallback with no bearer, or the exact legacy
        dashboard session bearer → downstream cookie/session/local auth decides.
      * unreachable  → 503 (provider backing store down; not "bad credentials").
      * otherwise    → 401 unauthenticated.

    Runs before the cookie/session gates (installed last in ``web_server.py``).
    The cookie gates honour ``request.state.token_authenticated`` and skip
    enforcement, so a token-authed request is never redirected to ``/login``.
    """
    path = request.url.path
    rule = _route_for_request(request)
    if rule is None:
        return await call_next(request)

    if rule.allow_session_fallback:
        if not request.headers.get("authorization", "").strip():
            return await call_next(request)
        if _has_valid_legacy_session_bearer(request):
            return await call_next(request)

    principal, unreachable = authenticate_token(
        request, required_scope=rule.required_scope
    )
    if principal is not None:
        if rule.required_scope is not None and rule.required_scope not in principal.scopes:
            # Do not let a principal that authenticated but lacks this route's
            # capability bypass the downstream dashboard/session gate.
            request.state.token_authenticated = False
            audit_log(
                AuditEvent.TOKEN_AUTH_FAILURE,
                provider=principal.provider,
                principal=principal.principal,
                reason="missing_scope",
                required_scope=rule.required_scope,
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
            AuditEvent.TOKEN_AUTH_FAILURE,
            provider=unreachable,
            reason="provider_unreachable",
            path=path,
            ip=_client_ip(request),
        )
        return JSONResponse(
            {"detail": f"Auth provider {unreachable!r} unreachable"},
            status_code=503,
        )

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        reason="no_provider_recognises_token",
        path=path,
        ip=_client_ip(request),
    )
    return JSONResponse(
        {"error": "unauthenticated", "detail": "Unauthorized"},
        status_code=401,
    )
