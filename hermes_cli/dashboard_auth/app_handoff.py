"""One-time session handoff codes for the desktop system-browser flow.

The desktop app cannot complete the OAuth round trip inside an embedded
window: that context cannot raise the OS WebAuthn / passkey / Touch ID
prompt (issue #42448). The fix is to run the IdP ceremony in the user's
real browser via ``shell.openExternal`` and bounce back to a loopback
``http://127.0.0.1:<port>/callback`` the app is listening on (RFC 8252,
"OAuth 2.0 for Native Apps").

That creates a session-handoff problem: ``/auth/callback`` runs in the
system browser, so the HttpOnly session cookie would land in the browser,
not the app. This module bridges the gap. When ``/auth/callback`` detects
an app-handoff request it mints a single-use **handoff code** bound to the
freshly minted :class:`~hermes_cli.dashboard_auth.base.Session`'s cookie
material, and redirects to the app's loopback URL carrying only that code
(never the tokens). The app then trades the code at
``POST /api/auth/desktop-exchange``, which sets the same session cookies
the browser flow would — but on the app's request, so they land in the
app's cookie jar. All token handling stays server-side: the app only ever
sees an opaque, short-lived, single-use code.

The store mirrors :mod:`hermes_cli.dashboard_auth.ws_tickets`: in-memory
(the dashboard is a single process), functional API so tests can patch
``time.time`` cleanly, single-use, short TTL.
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Dict, Tuple

#: Time-to-live for a handoff code in seconds. The code is minted at
#: ``/auth/callback`` and consumed by the app's loopback listener moments
#: later, so the live window is small. 120 s leaves slack for a slow
#: browser redirect or a momentarily busy app without leaving a usable
#: code lying around for long. The code is single-use regardless.
TTL_SECONDS = 120

_lock = threading.Lock()
#: handoff_code -> (expires_at, cookie material). The stored dict carries
#: exactly what ``set_session_cookies`` needs and nothing else — no user
#: identity, no provider — so a leaked store entry reveals only the tokens
#: it would have set as cookies anyway.
_codes: Dict[str, Tuple[int, Dict[str, object]]] = {}


class HandoffInvalid(Exception):
    """Handoff code missing, expired, or already consumed."""


def mint_handoff(
    *, access_token: str, refresh_token: str, expires_at: int
) -> str:
    """Generate a one-shot handoff code bound to a session's cookie material.

    ``expires_at`` is the access token's unix-seconds expiry (the
    ``Session.expires_at`` field); the exchange endpoint recomputes the
    cookie ``Max-Age`` from it at consume time. Returns a base64url code
    with 256 bits of entropy — unguessable, so no rate limiting is needed
    on the exchange endpoint.
    """
    code = secrets.token_urlsafe(32)
    material = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(expires_at),
    }
    with _lock:
        _codes[code] = (int(time.time()) + TTL_SECONDS, material)
        _gc_expired_locked()
    return code


def consume_handoff(code: str) -> Dict[str, object]:
    """Validate and consume a handoff code.

    Single-use: a successful consume removes the code from the store, so a
    replay raises ``HandoffInvalid``. Raises :class:`HandoffInvalid` on a
    missing, expired, or already-consumed code. Returns the cookie-material
    dict (``access_token``, ``refresh_token``, ``expires_at``).
    """
    now = int(time.time())
    with _lock:
        entry = _codes.pop(code, None)
        if entry is None:
            # Truncate so a misused/guessed code never lands in logs in full.
            truncated = (code[:8] + "…") if code else "<empty>"
            raise HandoffInvalid(f"unknown handoff code: {truncated}")
        expires_at, material = entry
        if expires_at < now:
            raise HandoffInvalid("expired")
        return material


def _gc_expired_locked() -> None:
    """Drop expired codes. Caller must hold ``_lock``."""
    now = int(time.time())
    expired = [c for c, (exp, _) in _codes.items() if exp < now]
    for c in expired:
        _codes.pop(c, None)


def _reset_for_tests() -> None:
    """Test-only: drop all handoff codes."""
    with _lock:
        _codes.clear()
