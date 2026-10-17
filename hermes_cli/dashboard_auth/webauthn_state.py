"""Ephemeral challenge/ceremony state for WebAuthn ceremonies (sibling of native_flow).

``begin`` (options) -> ``complete`` (verify) are two browser round trips; the
fido2 server holds exactly one piece of state between them — the ``challenge``
it generated plus the ``user_verification`` requirement — which never reaches
the browser in reusable form. We park that state in memory keyed by an opaque
256-bit handle that rides an HttpOnly cookie (the browser never sees the
challenge bytes), consume it on completion (single use), and bound it by a TTL.

Mirrors native_flow.py exactly: in-memory, process-local (the dashboard is a
single process — documented envelope, same as the password rate limiter),
``time.time`` with an injectable ``now`` so tests can drive expiry, thread lock,
monotonic-alphabet GC on every access, constant-time-safe handles compared by
dict lookup of an unguessable token, and ``_reset_for_tests()``.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Default begin->complete window (config dashboard.webauthn.challenge_ttl_seconds
# may lower 30-300; the resolver clamps, this is the plugin default).
_DEFAULT_TTL_SECONDS = 120
# Global cap so one client (or a flood) cannot grow the store unbounded: each
# ceremony parks a few dozen bytes; a login page poll loop stays bounded anyway.
_MAX_ENTRIES = 1024
# Per-IP cap on concurrent IN-FLIGHT ceremonies: /auth/webauthn/options is a
# public pre-auth route, one spammer must not fill the global store and lock
# out other logins (the per-IP limiter semantics mirror native_flow).
_MAX_CHALLENGES_PER_IP = 8

_lock = threading.Lock()


@dataclass(frozen=True)
class _Challenge:
    """Server-side ceremony state parked between begin and complete."""
    challenge: str  # the fido2-generated challenge, already websafe-encoded
    user_verification: str  # "preferred" | "required"
    kind: str  # "authentication" | "registration"
    client_ip: str
    # The authenticated principal that began a REGISTRATION ceremony (enrolment
    # is never anonymous); empty for authentication ceremonies.
    user_id: str
    expires_at: int


_challenges: dict[str, _Challenge] = {}


class ChallengeError(Exception):
    """Base for ceremony-state failures (unknown/expired/replayed handle)."""


class ChallengeNotFound(ChallengeError):
    """The ceremonial handle is unknown, expired, or already consumed."""


def _gc_locked(now: int) -> None:
    """Drop expired challenges. Caller holds ``_lock``."""
    for k in [k for k, v in _challenges.items() if v.expires_at < now]:
        _challenges.pop(k, None)


def _now(now: Optional[int]) -> int:
    return int(time.time()) if now is None else now


def register_challenge(
    *, challenge: str, user_verification: str, kind: str, client_ip: str = "",
    user_id: str = "", ttl_seconds: Optional[int] = None, now: Optional[int] = None) -> str:
    """Park one ceremony's server state; return the opaque one-shot ``handle``.

    Raises ``ChallengeError`` (fail closed) at global capacity or when
    ``client_ip`` already holds ``_MAX_CHALLENGES_PER_IP`` in-flight entries.
    """
    now = _now(now)
    ttl = _DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    handle = secrets.token_urlsafe(32)
    with _lock:
        _gc_locked(now)
        if _capacity_ok_locked():
            per_ip = sum(1 for v in _challenges.values() if v.client_ip == client_ip)
            if not client_ip or per_ip < _MAX_CHALLENGES_PER_IP:
                _challenges[handle] = _Challenge(
                    challenge=challenge, user_verification=user_verification, kind=kind,
                    client_ip=client_ip, user_id=user_id, expires_at=now + ttl)
                return handle
    raise ChallengeError("webauthn ceremony store at capacity")


def consume_challenge(handle: str, *, kind: str, now: Optional[int] = None) -> _Challenge:
    """Consume (single-use) the ceremony state for ``handle``.

    The entry is popped BEFORE validation so a wrong/garbage handle cannot be
    retried and a stale entry cannot be replayed after expiry. Raises
    ``ChallengeNotFound`` when unknown/expired/wrong-kind (no detail oracle —
    the route answers a generic 401 either way).
    """
    now = _now(now)
    with _lock:
        _gc_locked(now)
        entry = _challenges.pop(handle, None)
    if entry is None or entry.kind != kind:
        raise ChallengeNotFound("unknown, expired, or replayed webauthn ceremony")
    if entry.expires_at < now:
        raise ChallengeNotFound("unknown, expired, or replayed webauthn ceremony")
    return entry


def pending_count(client_ip: str) -> int:
    """Number of in-flight ceremonies for ``client_ip`` (observability/tests)."""
    with _lock:
        _gc_locked(_now(None))
        return sum(1 for v in _challenges.values() if v.client_ip == client_ip)


def _capacity_ok_locked() -> bool:
    return len(_challenges) < _MAX_ENTRIES


def _reset_for_tests() -> None:
    """Test-only: drop all ceremony state."""
    with _lock:
        _challenges.clear()