"""Request-level helpers shared by the auth routes and both middlewares."""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from hermes_cli.dashboard_auth import list_session_providers
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, ProviderError

# Paths a post-login redirect must never land on: the auth flow itself (would loop) and any
# ``/api/*`` target (raw JSON in the address bar, indistinguishable from a weaponised redirect).
_NEXT_DENY_PREFIXES = ("/login", "/auth/", "/api/auth/")


def client_ip(request: Request) -> str:
    """First ``X-Forwarded-For`` hop, else the peer address."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")


def extract_bearer(request: Request) -> str:
    """``Authorization: Bearer <token>`` value (scheme case-insensitive), or ``""``."""
    parts = request.headers.get("authorization", "").split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def _http_origin(value: str) -> tuple[str, str, int] | None:
    """Parse one serialized HTTP origin, never an opaque or URL-like origin."""
    if not value or any(ord(c) <= 32 or ord(c) >= 127 or c in "\\,?#%" for c in value):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None or parsed.path
                or parsed.netloc.endswith(":")):
            return None
        port = parsed.port
        effective_port = port if port is not None else (443 if parsed.scheme == "https" else 80)
        return parsed.scheme, parsed.hostname, effective_port
    except ValueError:
        return None


def cookie_origin_is_allowed(request: Request) -> bool:
    """Cookie writes must come from the dashboard's own origin, not just its site.

    ``SameSite=Lax`` keeps cross-site pages from sending the session cookies but
    still admits same-site siblings (another port or subdomain). The browser's
    ``Sec-Fetch-Site`` verdict decides when present: pages cannot set it, and the
    browser computes it against the URL it actually addressed, so it stays right
    behind a TLS-terminating or Host-rewriting proxy where this server's view of
    its own URL differs. Only a browser without Fetch Metadata falls back to
    comparing Origin with the OAuth public-URL authority, otherwise the request's
    Host and ASGI scheme (uvicorn applies *trusted* proxy headers). Never trust
    raw forwarded headers here or reuse WS's opaque/non-HTTP Origin exemption.
    """
    from hermes_cli.dashboard_auth.prefix import resolve_public_url
    from hermes_cli.web_server import _is_accepted_host

    origins = request.headers.getlist("origin")
    origin = _http_origin(origins[0]) if len(origins) == 1 else None
    if origin is None:
        return False
    bound_host = getattr(request.app.state, "bound_host", None)
    if bound_host and not _is_accepted_host(
        request.headers.get("host", ""), bound_host,
        getattr(request.app.state, "trusted_public_hosts", frozenset()),
    ):
        return False
    fetch_sites = request.headers.getlist("sec-fetch-site")
    if fetch_sites:
        return fetch_sites == ["same-origin"]
    target = urlsplit(resolve_public_url() or str(request.url))
    return origin == _http_origin(f"{target.scheme}://{target.netloc}")


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
