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
_internal_credential: Optional[str] = None  # legacy server-internal credential; guarded by ``_lock``
_internal_credentials: Dict[str, Dict[str, str]] = {}

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
            _internal_credentials[_internal_credential] = {
                "user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER}
        return _internal_credential


def mint_internal_credential(*, user_id: str, provider: str) -> str:
    """Mint a process-lifetime, multi-use credential for one spawned PTY.

    The browser identity was already verified by the ticket consumed on
    ``/api/pty``.  This opaque credential only carries that identity across the
    server-to-child ``/api/ws`` and ``/api/pub`` reconnect boundary; it is never
    returned to the SPA.
    """
    if not user_id or not provider:
        raise ValueError("internal credential identity requires user_id and provider")
    credential = secrets.token_urlsafe(32)
    with _lock:
        _internal_credentials[credential] = {"user_id": user_id, "provider": provider}
    return credential


def consume_internal_credential(value: str) -> Dict[str, Any]:
    """Validate a multi-use server-to-child credential and return its identity.

    The no-argument legacy credential still maps to ``server-internal``; PTY
    credentials map to the authenticated browser user that spawned that PTY.
    """
    with _lock:
        credentials = tuple(_internal_credentials.items())
    if not value or not credentials:
        raise TicketInvalid("no internal credential")
    encoded = value.encode()
    for expected, info in credentials:
        if secrets.compare_digest(encoded, expected.encode()):
            return dict(info)
    raise TicketInvalid("internal credential mismatch")


def _reset_for_tests() -> None:
    """Test-only: drop all tickets and the internal credential."""
    global _internal_credential
    with _lock:
        _tickets.clear()
        _internal_credentials.clear()
        _internal_credential = None
