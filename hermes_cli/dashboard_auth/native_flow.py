"""Gateway-brokered RFC 8252 (OAuth 2.0 for Native Apps) authorization store.

The desktop app is a *native* OAuth client that wants to sign in to a gated
gateway **without an embedded webview and without relying on browser session
cookies**. It cannot be a direct OAuth client of the upstream IDP (Nous
Portal): the Portal ``client_id`` is per-gateway-instance
(``agent:{instance_id}``) and the Portal validates that the ``redirect_uri``
ends in ``/auth/callback`` on the gateway's own public origin — a desktop
loopback ``127.0.0.1`` redirect is rejected. So the **gateway brokers** the
flow: it is the authorization server *to the desktop*, and an OAuth client *to
the Portal*. This is still a textbook RFC 8252 deployment — system browser,
loopback redirect, PKCE, tokens returned to the app (never cookies).

Wire shape (all gateway-side state lives in this module):

  1. Desktop generates its OWN PKCE pair ``(cv_d, cc_d)`` and a ``state``, opens
     a loopback listener on ``127.0.0.1:<port>``, and opens the system browser
     to the gateway's ``GET /auth/native/authorize?...`` carrying ``cc_d``,
     ``state``, and its loopback ``redirect_uri``.
  2. The gateway ``authorize`` route stashes a **pending authorization**
     (``register_pending``) keyed by an opaque ``broker_state`` and runs the
     EXISTING upstream PKCE flow (``provider.start_login`` → Portal
     ``/oauth/authorize`` → gateway ``/auth/callback``). The desktop's
     ``cc_d`` / ``state`` / loopback ``redirect_uri`` ride through the upstream
     round trip inside the gateway's own PKCE cookie, so no desktop secret is
     ever exposed to the Portal.
  3. On the upstream callback the gateway holds a verified :class:`Session`. It
     **mints a one-time gateway authorization code** (``complete_pending``)
     bound to the desktop's ``cc_d``, and 302s the browser to the desktop's
     ``redirect_uri?code=<gw_code>&state=<state>``.
  4. The desktop's loopback listener catches ``gw_code``, then POSTs
     ``/auth/native/token`` with ``gw_code`` + its ``cv_d``. The gateway
     verifies ``SHA256(cv_d) == cc_d`` (``redeem_code``), consumes the code
     (single use), and returns the upstream ``access_token`` /
     ``refresh_token`` / ``expires_at`` **in the JSON body**.
  5. The desktop stores those in the OS keychain and authenticates REST with
     ``Authorization: Bearer <access_token>`` (via the existing ``token_auth``
     seam) and mints ws-tickets the same way — no cookies anywhere.

Security properties this module guarantees:

  * **PKCE binding (RFC 7636).** A gateway code is redeemable only by the client
    that presented the matching ``code_challenge``. An attacker who intercepts
    the loopback ``gw_code`` (e.g. a hostile process racing the redirect) cannot
    exchange it without ``cv_d``, which never leaves the desktop.
  * **Single use.** ``redeem_code`` pops the entry; a replay finds nothing.
  * **Short TTLs.** A pending authorization lives ``_PENDING_TTL`` seconds (the
    interactive login window); a minted code lives ``_CODE_TTL`` seconds (the
    loopback round trip is sub-second). Expired entries are refused and GC'd.
  * **Opaque, high-entropy handles.** ``broker_state`` and ``gw_code`` are
    256-bit ``secrets.token_urlsafe`` values; comparison is constant-time.
  * **No secret logging.** The module stores tokens transiently in memory only
    between callback and redemption; nothing here writes them to disk (the
    audit log strips token fields).

In-memory and process-local: the dashboard is a single process, so no
distributed coordination is needed (mirrors ``ws_tickets``). A functional API
(not a class) keeps ``time.time`` patchable in tests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from hermes_cli.dashboard_auth.base import Session

# TTL for a pending authorization (step 2→3): the whole interactive login,
# including the user typing Portal credentials / approving in the browser.
_PENDING_TTL_SECONDS = 600  # 10 minutes — mirrors the PKCE cookie lifetime.

# TTL for a minted gateway code (step 3→4): only the loopback redirect + the
# desktop's immediate token POST, which is sub-second in practice.
_CODE_TTL_SECONDS = 120  # 2 minutes — generous for a slow local hop.

# Cap the number of concurrent pending/issued entries so a misbehaving or
# malicious client cannot grow the store unbounded. Well above any legitimate
# concurrent-login count for a single desktop user.
_MAX_ENTRIES = 256

# Per-IP cap on concurrent PENDING authorizations. /auth/native/authorize is a
# public (pre-auth) route, so without this a single unauthenticated spammer
# could fill the global store (600s TTL each) and lock out legitimate native
# logins for the pending window. A real desktop runs at most a couple of
# concurrent sign-ins from one address; 8 is generous.
_MAX_PENDING_PER_IP = 8

_lock = threading.Lock()


@dataclass
class _Pending:
    """An in-flight native authorization awaiting the upstream callback.

    Created when the desktop hits ``/auth/native/authorize`` and consumed when
    the upstream ``/auth/callback`` completes and mints the gateway code.
    """

    code_challenge: str  # the DESKTOP's S256 challenge (cc_d), base64url no-pad
    redirect_uri: str  # the desktop's loopback redirect (127.0.0.1:<port>/...)
    client_state: str  # the desktop's own ``state`` (echoed back on redirect)
    client_ip: str  # requester IP at authorize time (per-IP pending cap)
    expires_at: int


@dataclass
class _IssuedCode:
    """A minted one-time gateway authorization code bound to a Session."""

    code_challenge: str  # cc_d — verified against cv_d at redemption
    session: Session
    expires_at: int


# broker_state -> _Pending
_pending: Dict[str, _Pending] = {}
# gw_code -> _IssuedCode
_issued: Dict[str, _IssuedCode] = {}


class NativeFlowError(Exception):
    """Base for native-flow failures (bad/expired/replayed handle, PKCE fail)."""


class PendingNotFound(NativeFlowError):
    """The broker_state is unknown or expired (login window lapsed)."""


class CodeInvalid(NativeFlowError):
    """The gateway code is unknown, expired, already redeemed, or PKCE-mismatched."""


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _s256(verifier: str) -> str:
    """RFC 7636 S256 transform: base64url(sha256(ascii(verifier)))."""
    return _b64url_no_pad(hashlib.sha256(verifier.encode("ascii")).digest())


def _gc_locked(now: int) -> None:
    """Drop expired pending + issued entries. Caller holds ``_lock``."""
    expired_p = [k for k, v in _pending.items() if v.expires_at < now]
    for k in expired_p:
        _pending.pop(k, None)
    expired_c = [k for k, v in _issued.items() if v.expires_at < now]
    for k in expired_c:
        _issued.pop(k, None)


def _capacity_ok_locked() -> bool:
    return (len(_pending) + len(_issued)) < _MAX_ENTRIES


def register_pending(
    *,
    code_challenge: str,
    redirect_uri: str,
    client_state: str,
    client_ip: str = "",
    now: Optional[int] = None,
) -> str:
    """Stash a pending native authorization; return an opaque ``broker_state``.

    Called by ``/auth/native/authorize``. ``code_challenge`` is the DESKTOP's
    S256 challenge (``cc_d``) — we never see the verifier until redemption.
    ``redirect_uri`` is the desktop's loopback callback and ``client_state`` is
    the desktop's own CSRF ``state`` (echoed verbatim on the final redirect).
    ``client_ip`` is the requester's address, used only for the per-IP pending
    cap below.

    The returned ``broker_state`` is what the gateway threads through its OWN
    upstream PKCE round trip (inside the ``hermes_session_pkce`` cookie), so the
    callback can find this entry again via :func:`complete_pending`.

    Raises ``NativeFlowError`` if the store is at capacity or the caller's IP
    already holds ``_MAX_PENDING_PER_IP`` live pending entries (fail closed —
    this is a public pre-auth route, so one spammer must not be able to fill
    the global store and deny sign-in to everyone else).
    """
    now = int(time.time()) if now is None else now
    broker_state = secrets.token_urlsafe(32)
    with _lock:
        _gc_locked(now)
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow authorization store at capacity")
        if client_ip and (
            sum(1 for v in _pending.values() if v.client_ip == client_ip)
            >= _MAX_PENDING_PER_IP
        ):
            raise NativeFlowError(
                "too many pending native authorizations from this address"
            )
        _pending[broker_state] = _Pending(
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            client_state=client_state,
            client_ip=client_ip,
            expires_at=now + _PENDING_TTL_SECONDS,
        )
    return broker_state


def get_pending(broker_state: str, *, now: Optional[int] = None) -> _Pending:
    """Return the pending authorization for ``broker_state`` without consuming it.

    Read-only peek used by the callback to learn the desktop's ``redirect_uri``
    and ``client_state`` for the final 302. Raises :class:`PendingNotFound` if
    unknown or expired (the entry is GC'd on expiry).
    """
    now = int(time.time()) if now is None else now
    with _lock:
        _gc_locked(now)
        entry = _pending.get(broker_state)
        if entry is None:
            raise PendingNotFound("unknown or expired native authorization")
        return entry


def complete_pending(
    broker_state: str,
    *,
    session: Session,
    now: Optional[int] = None,
) -> str:
    """Consume a pending authorization and mint a one-time gateway code.

    Called by ``/auth/callback`` once the upstream :class:`Session` is verified.
    Pops the pending entry (single use), binds a fresh ``gw_code`` to the
    desktop's ``code_challenge`` + the verified ``session``, and returns the
    ``gw_code`` for the loopback redirect.

    Raises :class:`PendingNotFound` if the broker_state is unknown/expired.
    """
    now = int(time.time()) if now is None else now
    with _lock:
        _gc_locked(now)
        pending = _pending.pop(broker_state, None)
        if pending is None:
            raise PendingNotFound("unknown or expired native authorization")
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow code store at capacity")
        gw_code = secrets.token_urlsafe(32)
        _issued[gw_code] = _IssuedCode(
            code_challenge=pending.code_challenge,
            session=session,
            expires_at=now + _CODE_TTL_SECONDS,
        )
    return gw_code


def redeem_code(
    *,
    code: str,
    code_verifier: str,
    now: Optional[int] = None,
) -> Session:
    """Verify PKCE + consume a gateway code; return the bound :class:`Session`.

    Called by ``/auth/native/token``. Enforces:
      * the code exists and is unexpired (else :class:`CodeInvalid`);
      * ``S256(code_verifier) == code_challenge`` in constant time (RFC 7636);
      * single use — the entry is popped BEFORE the PKCE check so a wrong
        verifier cannot be retried against the same code.

    On any failure the code is already consumed (no oracle, no replay).
    """
    now = int(time.time()) if now is None else now
    with _lock:
        _gc_locked(now)
        issued = _issued.pop(code, None)
    # Pop happened under the lock; every return path below has already
    # consumed the code, so a replay (valid or not) finds nothing.
    if issued is None:
        raise CodeInvalid("unknown, expired, or already-redeemed code")
    if issued.expires_at < now:
        raise CodeInvalid("code expired")
    expected = issued.code_challenge
    actual = _s256(code_verifier)
    if not hmac.compare_digest(expected, actual):
        raise CodeInvalid("PKCE verification failed")
    return issued.session


def _reset_for_tests() -> None:
    """Test-only: drop all pending + issued state."""
    with _lock:
        _pending.clear()
        _issued.clear()
