"""Short-lived single-use tickets for WS-upgrade auth in gated mode.

Browsers cannot set ``Authorization`` on a WebSocket upgrade. In loopback
mode the legacy ``?token=<_SESSION_TOKEN>`` query param works because the
token is injected into the SPA bundle. In gated mode there is no injected
token — the SPA gets a fresh ticket via the authenticated REST endpoint
``POST /api/auth/ws-ticket`` and passes that as ``?ticket=`` on the
WS upgrade.

Tickets are single-use, TTL = 30 seconds. In-memory; the dashboard is a
single process so no distributed coordination is needed. The module
exposes a small functional API rather than a class so tests can patch
``time.time`` cleanly.
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any, Dict, Tuple

#: Time-to-live for newly-minted tickets in seconds. 30 s is long enough
#: that the SPA can call ``getWsTicket()`` and immediately open the WS,
#: short enough that a leaked ticket is uninteresting.
TTL_SECONDS = 30

_lock = threading.Lock()
_tickets: Dict[str, Tuple[int, Dict[str, Any]]] = {}  # ticket -> (expires_at, info)


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


def _reset_for_tests() -> None:
    """Test-only: drop all tickets."""
    with _lock:
        _tickets.clear()
