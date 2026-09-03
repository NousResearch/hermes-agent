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
    auth surface of an existing route.

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
import logging
import threading
from typing import Awaitable, Callable, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal

_log = logging.getLogger(__name__)
_INTERNAL_PROVIDER = "dashboard-internal"

# Exact paths that accept non-interactive bearer-token auth. A route registers
# itself here at import/startup; the seam only acts on registered paths.
_token_routes: dict[str, Optional[tuple[str, ...]]] = {}
_lock = threading.Lock()


def register_token_route(
    path: str,
    *,
    allow_internal: bool = False,
    internal_scopes: tuple[str, ...] = (),
) -> None:
    """Mark ``path`` (exact match) as token-authable.

    Idempotent. Call at module import / app setup so the seam knows which
    routes to guard. Registering a route does NOT make it public — it makes
    it authenticate by token instead of by session cookie. The process-internal
    bearer remains forbidden unless ``allow_internal`` is explicit; when
    enabled, only ``internal_scopes`` are attached to that principal.
    """
    policy = tuple(internal_scopes) if allow_internal else None
    with _lock:
        existing = _token_routes.get(path)
        if path in _token_routes and existing != policy:
            raise ValueError(f"conflicting token-route policy for {path!r}")
        _token_routes[path] = policy


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable (exact match)."""
    with _lock:
        return path in _token_routes


def internal_scopes_for_route(path: str) -> Optional[tuple[str, ...]]:
    """Return scopes for the process-internal bearer, or ``None`` if forbidden."""
    with _lock:
        return _token_routes.get(path)


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


def authenticate_token(
    request: Request,
) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token.

    Returns ``(principal, unreachable_provider_name)``:
      * ``(TokenPrincipal, None)`` — a provider recognised and accepted the token.
      * ``(None, None)`` — no token, or no provider recognised it (reject 401).
      * ``(None, name)`` — no provider accepted it AND at least one provider's
        backing store was unreachable (the caller surfaces 503, not 401, so a
        transient outage doesn't read as "bad credentials").

    Never raises: a provider ``ProviderError`` is caught and remembered.
    """
    token = extract_bearer_token(request)
    if not token:
        return None, None
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        if provider.name == _INTERNAL_PROVIDER:
            _log.warning(
                "dashboard-auth: token provider attempted to use reserved name %r; "
                "rejecting",
                provider.name,
            )
            continue
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
            if principal.provider != provider.name:
                _log.warning(
                    "dashboard-auth: token provider %r returned mismatched "
                    "principal provenance %r; rejecting",
                    provider.name,
                    principal.provider,
                )
                continue
            return principal, None
    return None, unreachable


async def token_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Outermost auth seam for token-authable routes.

    No-op pass-through for any path not registered via
    :func:`register_token_route`. For a registered path, token auth is the
    only accepted scheme:

      * opted-in exact process-internal bearer → attach the scoped server principal;
      * valid provider token → attach principal + ``token_authenticated`` flag, pass through.
      * unreachable  → 503 (provider backing store down; not "bad credentials").
      * otherwise    → 401 unauthenticated.

    Runs before the cookie/session gates (installed last in ``web_server.py``).
    The cookie gates honour ``request.state.token_authenticated`` and skip
    enforcement, so a token-authed request is never redirected to ``/login``.
    """
    path = request.url.path
    if not is_token_route(path):
        return await call_next(request)

    token = extract_bearer_token(request)
    internal_scopes = internal_scopes_for_route(path)
    internal_token = getattr(request.app.state, "internal_api_token", None)
    if (
        internal_scopes is not None
        and token
        and isinstance(internal_token, str)
        and internal_token
        and hmac.compare_digest(token.encode(), internal_token.encode())
    ):
        principal = TokenPrincipal(
            principal=_INTERNAL_PROVIDER,
            provider=_INTERNAL_PROVIDER,
            scopes=internal_scopes,
        )
        unreachable = None
    else:
        principal, unreachable = authenticate_token(request)
    if principal is not None:
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
