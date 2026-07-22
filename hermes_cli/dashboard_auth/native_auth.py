"""Native-app (RFC 8252) loopback+PKCE login broker for the desktop app.

The desktop app authenticates to a **gated** gateway using the user's *system
browser* instead of an embedded webview, and holds the resulting tokens itself
(sending them as ``Authorization: Bearer``) instead of relying on the HttpOnly
session-cookie jar. This module is the gateway-side broker that makes that
possible without changing the gateway's own upstream IDP contract.

Two PKCE contexts are in play — keep them distinct:

* **gateway ↔ IDP** — the ordinary upstream PKCE the provider's ``start_login``
  generates. In the cookie flow this verifier rides in the ``hermes_session_pkce``
  browser cookie. In the native flow the *system browser* has no such cookie
  (the desktop called ``/auth/native/start`` over its OWN HTTP client), so the
  broker stores the upstream verifier **server-side**, keyed by the upstream
  ``state`` value the IDP echoes back on the callback.
* **desktop ↔ gateway** — a SECOND PKCE pair the desktop generates and keeps to
  itself. Its challenge is registered at ``start`` and verified at ``token``
  redemption, exactly as RFC 8252 prescribes for the loopback code exchange.

Flow (see ``.hermes/plans/2026-07-20-desktop-rfc8252-loopback-auth.md``):

1. **start** (``POST /auth/native/start``, desktop → gateway) — desktop sends its
   loopback ``redirect_uri``, its PKCE ``code_challenge``, and its ``state``. The
   gateway runs the provider's ordinary ``start_login`` against the gateway's own
   ``/auth/callback``, extracts the upstream ``state`` + ``verifier`` from the
   returned cookie payload, and stores a broker record keyed by the upstream
   ``state``. Returns the upstream ``authorization_url``; the desktop opens it in
   the system browser.

2. **callback** — the IDP redirects the *system browser* to the gateway's existing
   ``/auth/callback`` (no cookie present). The callback looks up a broker record
   by the echoed ``state``; if found it completes the upstream login with the
   server-stored verifier, attaches the :class:`Session` via
   :func:`complete_broker`, and 302s the browser to the desktop's loopback
   ``redirect_uri`` carrying a single-use ``code`` + the desktop's ``state``.

3. **token** (``POST /auth/native/token``, desktop → gateway) — desktop redeems the
   single-use ``code`` + its PKCE ``code_verifier``; the broker verifies the
   verifier against the challenge from leg 1 and returns the session tokens as
   JSON. No cookies set.

The returned tokens are the provider-issued ``access_token`` / ``refresh_token``
the cookie flow would have stored, so a subsequent
``Authorization: Bearer <access_token>`` request verifies through the same
``verify_session`` provider stack (see ``middleware.gated_auth_middleware``).

In-memory single-process store, same shape as ``ws_tickets.py``. Time is read via
``time.time`` so tests can monkeypatch it.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

from hermes_cli.dashboard_auth.base import Session

# The broker record lives for the whole interactive login window: the user may
# take a while at the IDP consent screen. 10 minutes matches the PKCE cookie TTL.
BROKER_TTL_SECONDS = 10 * 60
# The redeemable one-time code is short-lived and single-use: the desktop
# redeems it the instant its loopback listener fires.
CODE_TTL_SECONDS = 60


class BrokerError(Exception):
    """A broker record / code was missing, expired, already used, or mismatched.

    Carries an OAuth-style ``error`` code so the route can surface a stable
    machine-readable envelope (``invalid_grant`` / ``invalid_request`` / ...).
    """

    def __init__(self, message: str, *, error: str = "invalid_request") -> None:
        super().__init__(message)
        self.error = error


@dataclass
class _BrokerRecord:
    # ---- gateway ↔ IDP (server-held upstream PKCE) ----
    provider: str
    upstream_verifier: str
    # ---- desktop ↔ gateway (RFC 8252 loopback exchange) ----
    code_challenge: str            # desktop's PKCE challenge, verified at redeem
    code_challenge_method: str
    redirect_uri: str              # desktop's loopback URL the callback 302s to
    desktop_state: str             # echoed back to the desktop on that 302
    # ---- lifecycle ----
    expires_at: int
    session: Optional[Session] = None   # set by complete_broker
    code: Optional[str] = None          # single-use redemption code
    code_expires_at: int = 0


_lock = threading.Lock()
# Keyed by the UPSTREAM ``state`` (what the IDP echoes on the callback).
_records: Dict[str, _BrokerRecord] = {}
# Reverse index: one-time redemption code -> upstream_state.
_code_index: Dict[str, str] = {}


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def is_loopback_redirect_uri(redirect_uri: str) -> bool:
    """True iff ``redirect_uri`` is an ``http`` loopback URL (RFC 8252 §7.3).

    Only ``127.0.0.0/8``, ``::1``, and the literal ``localhost`` are accepted,
    only over ``http``, and a path must be present. Anything else — a public
    host, ``https``, a custom scheme, a bare authority — is rejected so the
    broker can never be aimed at an attacker-controlled redirect.
    """
    try:
        parsed = urlparse(redirect_uri)
    except (ValueError, TypeError):
        return False
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").strip()
    if not host or not parsed.path:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _gc_expired_locked(now: int) -> None:
    """Drop expired broker records + their code-index entries. Holds ``_lock``."""
    dead = [state for state, rec in _records.items() if rec.expires_at < now]
    for state in dead:
        rec = _records.pop(state, None)
        if rec and rec.code:
            _code_index.pop(rec.code, None)


def start_broker(
    *,
    upstream_state: str,
    upstream_verifier: str,
    provider: str,
    code_challenge: str,
    code_challenge_method: str,
    redirect_uri: str,
    desktop_state: str,
) -> None:
    """Register a pending native login, keyed by the upstream ``state``.

    ``upstream_state`` / ``upstream_verifier`` come from the provider's
    ``start_login`` (the gateway↔IDP PKCE). ``code_challenge`` is the desktop's
    own PKCE challenge (S256 only), verified at :func:`redeem_code`.
    """
    if code_challenge_method != "S256":
        raise BrokerError(
            f"unsupported code_challenge_method: {code_challenge_method!r}",
            error="invalid_request",
        )
    if not code_challenge:
        raise BrokerError("code_challenge required", error="invalid_request")
    if not is_loopback_redirect_uri(redirect_uri):
        raise BrokerError(
            "redirect_uri must be an http loopback URL", error="invalid_request"
        )
    if not desktop_state:
        raise BrokerError("state required", error="invalid_request")
    if not upstream_state:
        raise BrokerError("internal: missing upstream state", error="server_error")

    now = int(time.time())
    with _lock:
        _gc_expired_locked(now)
        _records[upstream_state] = _BrokerRecord(
            provider=provider,
            upstream_verifier=upstream_verifier,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            redirect_uri=redirect_uri,
            desktop_state=desktop_state,
            expires_at=now + BROKER_TTL_SECONDS,
        )


def get_broker(upstream_state: str) -> Optional[_BrokerRecord]:
    """Return a live broker record for an upstream ``state``, or None.

    The ``/auth/callback`` handler calls this to decide whether an incoming
    callback belongs to a native (loopback) login vs. the ordinary cookie flow.
    Expired records are evicted on read.
    """
    if not upstream_state:
        return None
    now = int(time.time())
    with _lock:
        rec = _records.get(upstream_state)
        if rec is None:
            return None
        if rec.expires_at < now:
            _records.pop(upstream_state, None)
            if rec.code:
                _code_index.pop(rec.code, None)
            return None
        return rec


def complete_broker(upstream_state: str, session: Session) -> str:
    """Attach the minted ``session`` to a broker record; return a one-time code.

    Called by ``/auth/callback`` when the echoed ``state`` matches a broker
    record. The returned single-use ``code`` is what the browser carries to the
    desktop's loopback listener.
    """
    now = int(time.time())
    code = secrets.token_urlsafe(32)
    with _lock:
        rec = _records.get(upstream_state)
        if rec is None or rec.expires_at < now:
            raise BrokerError("broker session expired", error="invalid_grant")
        if rec.code:  # invalidate any prior code (defensive)
            _code_index.pop(rec.code, None)
        rec.session = session
        rec.code = code
        rec.code_expires_at = now + CODE_TTL_SECONDS
        _code_index[code] = upstream_state
    return code


def redeem_code(*, code: str, code_verifier: str, redirect_uri: str) -> Session:
    """Verify the desktop's PKCE + redirect_uri and return the session tokens.

    Single-use: the code (and its broker record) are removed on success AND on a
    PKCE/redirect mismatch, so a stolen code can't be brute-forced. Raises
    :class:`BrokerError` (``invalid_grant``) on any failure.
    """
    now = int(time.time())
    with _lock:
        upstream_state = _code_index.pop(code, None)
        rec = _records.get(upstream_state) if upstream_state else None
        # Whatever happens next, this code is spent — pop the record too so a
        # failed attempt can't be retried with a guessed verifier.
        if upstream_state is not None:
            _records.pop(upstream_state, None)

    if rec is None or rec.code != code:
        raise BrokerError("unknown or used code", error="invalid_grant")
    if rec.code_expires_at < now:
        raise BrokerError("code expired", error="invalid_grant")
    if rec.session is None:
        raise BrokerError("login not completed", error="invalid_grant")
    if redirect_uri != rec.redirect_uri:
        raise BrokerError("redirect_uri mismatch", error="invalid_grant")
    if not code_verifier:
        raise BrokerError("code_verifier required", error="invalid_grant")

    # PKCE S256: BASE64URL(SHA256(verifier)) must equal the stored challenge.
    computed = _b64url_no_pad(hashlib.sha256(code_verifier.encode("ascii")).digest())
    if not secrets.compare_digest(computed, rec.code_challenge):
        raise BrokerError("PKCE verification failed", error="invalid_grant")

    return rec.session


def _reset_for_tests() -> None:
    """Test-only: drop all broker records + code index."""
    with _lock:
        _records.clear()
        _code_index.clear()
