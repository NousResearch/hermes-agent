"""Short-lived native refresh replay, scoped to the concrete provider and caller.

A provider hint only orders discovery: it must neither split one rotating credential's
lock nor let an unrelated provider reuse its result. Raw refresh tokens are never keys.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field

from hermes_cli.dashboard_auth.base import DashboardAuthProvider, RefreshExpiredError, Session
from hermes_cli.dashboard_auth.request_utils import scan_session_providers

_SUCCESS_TTL = 30.0
_FAILURE_TTL = 5.0
_MAX_ENTRIES = 256
_guard = threading.Lock()


@dataclass
class _Flight:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


# Hold the provider itself while caching: replacement (including same-name scoped
# registrations) invalidates identity, and Python cannot recycle its id under a live entry.
_cache: dict[tuple[int, bytes], tuple[float, DashboardAuthProvider, Session | None]] = {}
_flights: dict[tuple[int, bytes], _Flight] = {}


def _prune(now: float) -> None:
    for key, (expires, _, _) in list(_cache.items()):
        if expires <= now:
            del _cache[key]
    while len(_cache) > _MAX_ENTRIES:
        del _cache[min(_cache, key=lambda key: _cache[key][0])]


def _refresh_provider(provider: DashboardAuthProvider, token: str, client_ip: str) -> Session | None:
    digest = hashlib.sha256(client_ip.encode() + b"\0" + token.encode()).digest()
    key = (id(provider), digest)
    with _guard:
        _prune(time.monotonic())
        flight = _flights.setdefault(key, _Flight())
        flight.users += 1
    try:
        with flight.lock:
            with _guard:
                cached = _cache.get(key)
                if cached is not None and cached[0] > time.monotonic():
                    return cached[2]
            try:
                session = provider.refresh_session(refresh_token=token)
            except RefreshExpiredError:
                session = None
            # ProviderError and unexpected execution failures are deliberately not cached.
            with _guard:
                now = time.monotonic()
                _cache[key] = (now + (_SUCCESS_TTL if session is not None else _FAILURE_TTL), provider, session)
                _prune(now)
            return session
    finally:
        with _guard:
            flight.users -= 1
            if flight.users == 0:
                _flights.pop(key, None)


def refresh_native_session(token: str, provider_hint: str, client_ip: str) -> Session | None:
    """Preserve upstream provider fallback/503 behavior while coalescing each actual issuer."""
    return scan_session_providers(
        provider_hint, lambda provider: _refresh_provider(provider, token, client_ip),
        phase="native refresh", log=logging.getLogger(__name__),
    )
