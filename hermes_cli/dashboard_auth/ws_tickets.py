"""WS-upgrade auth credentials for gated mode.

Browsers cannot set ``Authorization`` on a WebSocket upgrade, and gated mode has no token
injected into the SPA, so two credential shapes exist: (1) single-use browser tickets
(``mint_ticket`` / ``consume_ticket``) fetched via authenticated ``POST /api/auth/ws-ticket`` and
passed as ``?ticket=`` on the upgrade — 30 s TTL, a leak is uninteresting; (2) a process-lifetime
internal credential (``internal_ws_credential`` / ``consume_internal_credential``) for
*server-spawned* WS clients (the embedded-TUI PTY child on ``/api/ws`` + ``/api/pub``), which
reuse their attach URL on every reconnect, possibly >30 s after boot — minted once, never expires,
multi-use, never injected into any HTML/SPA (leaves the process only via the child's environment,
so browser XSS cannot read it; grants no more than a ticket). In-memory; ``time.time`` patchable.
"""
from __future__ import annotations

import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple

#: Long enough for ``getWsTicket()`` -> open WS, short enough that a leaked ticket is uninteresting.
TTL_SECONDS = 30

_lock = threading.Lock()
_tickets: Dict[str, Tuple[int, Dict[str, Any]]] = {}  # ticket -> (expires_at, info)
_internal_credential: Optional[str] = None  # lazily minted; guarded by ``_lock``

#: Identity recorded for internal-credential connections (audit logs distinguish them from tickets).
INTERNAL_USER_ID = "server-internal"
INTERNAL_PROVIDER = "server-internal"


class TicketInvalid(Exception):
    """Ticket missing, expired, or already consumed."""


def mint_ticket(*, user_id: str, provider: str) -> str:
    """One-shot base64url ticket (32 random bytes) bound to this identity; ``consume_ticket``
    hands the ``info`` dict back to the WS handler."""
    ticket = secrets.token_urlsafe(32)
    info = {"user_id": user_id, "provider": provider, "minted_at": int(time.time())}
    with _lock:
        _tickets[ticket] = (int(time.time()) + TTL_SECONDS, info)
        _gc_expired_locked()
    return ticket


def consume_ticket(ticket: str) -> Dict[str, Any]:
    """Validate and consume (single-use). Raises :class:`TicketInvalid` on missing/expired/used."""
    now = int(time.time())
    with _lock:
        entry = _tickets.pop(ticket, None)
        if entry is None:
            # Truncated so misuse never logs the secret in full.
            truncated = (ticket[:8] + "…") if ticket else "<empty>"
            raise TicketInvalid(f"unknown ticket: {truncated}")
        expires_at, info = entry
        if expires_at < now:
            raise TicketInvalid("expired")
        return info


def _gc_expired_locked() -> None:
    """Drop expired tickets. Caller must hold ``_lock``."""
    now = int(time.time())
    for t in [t for t, (exp, _) in _tickets.items() if exp < now]:
        _tickets.pop(t, None)


def internal_ws_credential() -> str:
    """Process-lifetime internal WS credential, minted once. Never injected into the SPA or
    returned over REST — only passed to a spawned child via its environment."""
    global _internal_credential
    with _lock:
        if _internal_credential is None:
            _internal_credential = secrets.token_urlsafe(32)
        return _internal_credential


def consume_internal_credential(value: str) -> Dict[str, Any]:
    """Validate an internal credential (NOT single-use); returns the fixed server-internal
    ``{user_id, provider}`` info dict, mirroring ``consume_ticket``. Constant-time compare; any
    value is rejected until a credential has been minted."""
    with _lock:
        expected = _internal_credential
    if not value or expected is None:
        raise TicketInvalid("no internal credential")
    if not secrets.compare_digest(value.encode(), expected.encode()):
        raise TicketInvalid("internal credential mismatch")
    return {"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER}


def _reset_for_tests() -> None:
    """Test-only: drop all tickets and the internal credential."""
    global _internal_credential
    with _lock:
        _tickets.clear()
        _internal_credential = None
