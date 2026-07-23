"""WS-upgrade auth credentials for gated mode.

Browsers cannot set ``Authorization`` on a WebSocket upgrade. In loopback
mode the legacy ``?token=<_SESSION_TOKEN>`` query param works because the
token is injected into the SPA bundle. In gated mode there is no injected
token — so this module provides two credential shapes:

1. **Single-use browser tickets** (``mint_ticket`` / ``consume_ticket``).
   The SPA gets a fresh ticket via the authenticated REST endpoint
   ``POST /api/auth/ws-ticket`` and passes it as ``?ticket=`` on the WS
   upgrade. Single-use, TTL = 30 seconds — a leaked ticket is uninteresting.

2. **Audience-bound internal capabilities** (``internal_ws_credential`` /
   ``consume_internal_credential``). These authenticate *server-spawned* WS
   clients over loopback. A process-local random root derives distinct,
   multi-use capabilities for the JSON-RPC gateway and each profile/channel
   event sidecar. The root never leaves this module, and a sidecar capability
   cannot be rebound to the broader gateway route. The derived values are
   never injected into HTML or returned by a REST endpoint.

In-memory; the dashboard is a single process so no distributed coordination
is needed. The module exposes a small functional API rather than a class so
tests can patch ``time.time`` cleanly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple

#: Time-to-live for newly-minted tickets in seconds. 30 s is long enough
#: that the SPA can call ``getWsTicket()`` and immediately open the WS,
#: short enough that a leaked ticket is uninteresting.
TTL_SECONDS = 30

_lock = threading.Lock()
_tickets: Dict[str, Tuple[int, Dict[str, Any]]] = {}  # ticket -> (expires_at, info)

#: Process-local derivation key for audience-bound internal capabilities. The
#: key itself never leaves this module; only HMAC-derived values are passed to
#: the server-spawned TUI process. Guarded by ``_lock``.
_internal_credential_key: Optional[bytes] = None

#: Identity recorded for connections that authenticate via the internal
#: credential, so audit logs distinguish them from browser-initiated tickets.
INTERNAL_USER_ID = "server-internal"
INTERNAL_PROVIDER = "server-internal"


class TicketInvalid(Exception):
    """Ticket missing, expired, or already consumed."""


def mint_ticket(*, user_id: str, provider: str) -> str:
    """Generate a one-shot ticket bound to this user identity.

    The returned token is base64url, 43 bytes of entropy (32-byte random
    seed). Stash returns the ``info`` dict to the caller on consume so the
    WS handler can carry the identity forward into its session log.
    """
    ticket = secrets.token_urlsafe(32)
    info = {
        "user_id": user_id,
        "provider": provider,
        "minted_at": int(time.time()),
    }
    with _lock:
        _tickets[ticket] = (int(time.time()) + TTL_SECONDS, info)
        _gc_expired_locked()
    return ticket


def consume_ticket(ticket: str) -> Dict[str, Any]:
    """Validate and consume. Raises :class:`TicketInvalid` on missing/expired/used.

    Single-use semantics: a successful consume immediately removes the
    ticket from the store, so a second call with the same value raises
    ``TicketInvalid("unknown ticket: …")``.
    """
    now = int(time.time())
    with _lock:
        entry = _tickets.pop(ticket, None)
        if entry is None:
            # Truncate ticket value in the error so misuse never logs the
            # secret in full.
            truncated = (ticket[:8] + "…") if ticket else "<empty>"
            raise TicketInvalid(f"unknown ticket: {truncated}")
        expires_at, info = entry
        if expires_at < now:
            raise TicketInvalid("expired")
        return info


def _gc_expired_locked() -> None:
    """Drop expired tickets. Caller must hold ``_lock``."""
    now = int(time.time())
    expired = [t for t, (exp, _) in _tickets.items() if exp < now]
    for t in expired:
        _tickets.pop(t, None)


def _capability_value(key: bytes, *, audience: str, binding: str) -> str:
    """Derive one opaque capability from the process key and its audience."""
    payload = json.dumps(
        [audience, binding], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hmac.new(key, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def internal_ws_credential(*, audience: str, binding: str = "") -> str:
    """Return a stable process-lifetime capability for one internal audience.

    ``audience`` identifies the accepting route class (currently ``gateway``
    or ``sidecar``). ``binding`` narrows that class further, for example to a
    profile/channel pair. The derived value is stable and multi-use so a child
    can reconnect, but changing either field produces a different value.
    """
    if not audience:
        raise ValueError("internal credential audience is required")

    global _internal_credential_key
    with _lock:
        if _internal_credential_key is None:
            _internal_credential_key = secrets.token_bytes(32)
        key = _internal_credential_key
    return _capability_value(key, audience=audience, binding=binding)


def consume_internal_credential(
    value: str, *, audience: str, binding: str = ""
) -> Dict[str, Any]:
    """Validate an audience-bound capability without consuming it.

    Unlike :func:`consume_ticket`, a successful value remains valid for
    reconnects. Validation derives the expected value for the exact audience
    and binding and compares it in constant time. A capability minted for a
    different route, profile, or channel therefore fails closed.
    """
    if not audience:
        raise TicketInvalid("internal credential audience missing")
    with _lock:
        key = _internal_credential_key
    if not value or key is None:
        raise TicketInvalid("no internal credential")
    expected = _capability_value(key, audience=audience, binding=binding)
    if not secrets.compare_digest(value.encode(), expected.encode()):
        raise TicketInvalid("internal credential mismatch")
    return {
        "user_id": INTERNAL_USER_ID,
        "provider": INTERNAL_PROVIDER,
    }


def _reset_for_tests() -> None:
    """Test-only: drop all tickets and the internal credential."""
    global _internal_credential_key
    with _lock:
        _tickets.clear()
        _internal_credential_key = None
