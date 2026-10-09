"""Request-level helpers shared by the auth routes and both middlewares."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from hermes_cli.dashboard_auth import list_session_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, ProviderError

# Paths a post-login redirect must never land on: the auth flow itself (would loop) and any
# ``/api/*`` target (raw JSON in the address bar, indistinguishable from a weaponised redirect).
_NEXT_DENY_PREFIXES = ("/login", "/auth/", "/api/auth/")


def client_ip(request: Request) -> str:
    """ASGI peer address for rate limits, native pending caps, and auth audit.

    Never parse client-supplied ``X-Forwarded-For`` here: direct clients can
    spoof it. Trusted proxy normalization belongs upstream, where the server
    may rewrite ``request.client`` only for operator-configured trusted peers.
    """
    return request.client.host if request.client else ""


# Bound the synthetic client identifier we persist in audit records so an
# attacker-controlled User-Agent header cannot bloat the audit log.
_MAX_USER_AGENT_LEN = 256

# Attribution only, never a credential: the same User-Agent must map to the same
# id across records so a storm can be grouped by device, while the log keeps no
# raw header (User-Agents embed OS versions and build numbers, and this field is
# attacker-controlled).
_CLIENT_DEVICE_HASH_LEN = 16


def client_device(request) -> str:
    """Stable, non-reversible client id for audit records, derived from the ``User-Agent``.

    Both REFRESH_FAILURE sites audit *before* any token-derived identity resolves, so no
    ``user_id`` exists; the User-Agent is the only per-device signal available on either
    path (the native route and the cookie gate). Hashing keeps attribution ("same device
    vs different device") without persisting a raw, spoofable header.

    Returns ``""`` when the header is absent or empty.
    """
    ua = request.headers.get("user-agent", "")
    if not ua:
        return ""
    digest = hashlib.sha256(ua[:_MAX_USER_AGENT_LEN].encode("utf-8", "replace")).hexdigest()
    return digest[:_CLIENT_DEVICE_HASH_LEN]


def audit_refresh_failure(
    request: Request, *, provider: Optional[str] = None, reason: str,
) -> None:
    """Record one refresh rejection with the client attribution every REFRESH_FAILURE needs.

    Shared by the native refresh route and the cookie-gate middleware: both audit the same
    event before identity resolves, and a client id on only one of them would leave the other
    path unattributable behind a NAT (#98338).
    """
    audit_log(
        AuditEvent.REFRESH_FAILURE, provider=provider, reason=reason,
        device=client_device(request), ip=client_ip(request))


def extract_bearer(request: Request) -> str:
    """``Authorization: Bearer <token>`` value (scheme case-insensitive), or ``""``."""
    parts = request.headers.get("authorization", "").split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def is_safe_next_path(path: str) -> bool:
    """Same-origin post-login target: rejects non-relative and protocol-relative (``//evil``)
    values, the auth routes themselves, and every ``/api`` path."""
    if not path.startswith("/") or path.startswith("//"):
        return False
    if any(path == p or path.startswith(p) for p in _NEXT_DENY_PREFIXES):
        return False
    return not (path == "/api" or path.startswith("/api/"))


def access_token_max_age(session) -> int:
    """Cookie Max-Age for the access token: seconds to ``exp``, floored at 60."""
    return max(60, int(session.expires_at) - int(time.time()))


def unreachable_response(provider_name: str) -> JSONResponse:
    """503 for a transient IDP/backing-store outage (never a forced re-login)."""
    return JSONResponse({"detail": f"Auth provider {provider_name!r} unreachable"}, status_code=503)


def scan_session_providers(
    provider_hint: Optional[str], call: Callable[[DashboardAuthProvider], object], *, phase: str,
    log: logging.Logger, swallow: tuple[type[BaseException], ...] = (),
    on_swallow: Optional[Callable[[DashboardAuthProvider], None]] = None,
    on_unreachable: Optional[Callable[[DashboardAuthProvider], None]] = None):
    """Run ``call`` across the session providers; first non-``None`` result or ``None``.

    The hinted provider goes first (stable sort; a stale/unknown hint leaves registration order
    intact). ``swallow`` exceptions reject that candidate only. A ``ProviderError`` (IDP/JWKS
    unreachable) must NOT abort the chain — the credential may belong to a different, reachable
    provider; it is logged under ``phase`` and, if nothing else succeeds, re-raised as
    ``ProviderError(name)`` so the caller answers 503 instead of forcing a re-login.
    """
    providers = list_session_providers()
    if provider_hint:
        providers.sort(key=lambda provider: provider.name != provider_hint)
    unreachable: Optional[str] = None
    for provider in providers:
        try:
            result = call(provider)
        except swallow:
            if on_swallow is not None:
                on_swallow(provider)
            continue
        except ProviderError as e:
            log.warning("dashboard-auth: provider %r unreachable during %s: %s",
                        provider.name, phase, e)
            if on_unreachable is not None:
                on_unreachable(provider)
            if unreachable is None:
                unreachable = provider.name
            continue
        if result is not None:
            return result
    if unreachable is not None:
        raise ProviderError(unreachable)
    return None
