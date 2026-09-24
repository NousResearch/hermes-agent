"""Gateway-brokered RFC 8252 (OAuth 2.0 for Native Apps) authorization store.

The desktop cannot be a direct OAuth client of the upstream IDP (the Portal
``client_id`` is per gateway and only accepts the gateway's ``/auth/callback``),
so the gateway brokers: authorization server *to the desktop*, OAuth client
*to the Portal*. Desktop PKCE pair (cv_d, cc_d) + state -> ``/auth/native/authorize``
stashes a pending entry (:func:`register_pending`) keyed by an opaque
``broker_state`` riding in the gateway's own PKCE cookie -> upstream callback
or password login mints a one-time code bound to cc_d (:func:`complete_pending`)
-> desktop redeems it with cv_d at ``/auth/native/token`` (:func:`redeem_code`).
PKCE binding, single use, short TTLs, 256-bit handles compared in constant time,
no secret logging; in-memory, process-local, ``time.time`` patchable in tests.
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

_PENDING_TTL_SECONDS = 600  # whole interactive login (mirrors the PKCE cookie)
_CODE_TTL_SECONDS = 120  # loopback redirect + immediate token POST only
_MAX_ENTRIES = 256  # global cap so a misbehaving client cannot grow the store unbounded
# Per-IP cap on PENDING entries: /auth/native/authorize is a public pre-auth route, so one
# spammer must not fill the global store and lock out logins.
_MAX_PENDING_PER_IP = 8

_lock = threading.Lock()


@dataclass
class _Pending:
    """In-flight native authorization awaiting the upstream callback."""
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


_pending: Dict[str, _Pending] = {}  # broker_state -> _Pending
_issued: Dict[str, _IssuedCode] = {}  # gw_code -> _IssuedCode


class NativeFlowError(Exception):
    """Base for native-flow failures (bad/expired/replayed handle, PKCE fail)."""


class PendingNotFound(NativeFlowError):
    """The broker_state is unknown or expired (login window lapsed)."""


class CodeInvalid(NativeFlowError):
    """The gateway code is unknown, expired, already redeemed, or PKCE-mismatched."""


def _s256(verifier: str) -> str:
    """RFC 7636 S256 transform: base64url(sha256(ascii(verifier))), no padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _gc_locked(now: int) -> None:
    """Drop expired pending + issued entries. Caller holds ``_lock``."""
    for store in (_pending, _issued):
        for k in [k for k, v in store.items() if v.expires_at < now]:
            store.pop(k, None)


def _capacity_ok_locked() -> bool:
    return (len(_pending) + len(_issued)) < _MAX_ENTRIES


def _now(now: Optional[int]) -> int:
    return int(time.time()) if now is None else now


def _pop_pending_locked(broker_state: str, *, consume: bool) -> _Pending:
    """Look up (optionally consuming) a pending entry; :class:`PendingNotFound` if unknown."""
    entry = (_pending.pop if consume else _pending.get)(broker_state, None)
    if entry is None:
        raise PendingNotFound("unknown or expired native authorization")
    return entry


def register_pending(
    *, code_challenge: str, redirect_uri: str, client_state: str, client_ip: str = "",
    now: Optional[int] = None) -> str:
    """Stash a pending native authorization; return an opaque ``broker_state``. ``code_challenge``
    is the DESKTOP's cc_d. Raises ``NativeFlowError`` (fail closed) at store capacity or when
    ``client_ip`` holds ``_MAX_PENDING_PER_IP`` entries."""
    now = _now(now)
    broker_state = secrets.token_urlsafe(32)
    with _lock:
        _gc_locked(now)
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow authorization store at capacity")
        per_ip = sum(1 for v in _pending.values() if v.client_ip == client_ip)
        if client_ip and per_ip >= _MAX_PENDING_PER_IP:
            raise NativeFlowError("too many pending native authorizations from this address")
        _pending[broker_state] = _Pending(
            code_challenge=code_challenge, redirect_uri=redirect_uri, client_state=client_state,
            client_ip=client_ip, expires_at=now + _PENDING_TTL_SECONDS)
    return broker_state


def get_pending(broker_state: str, *, now: Optional[int] = None) -> _Pending:
    """Peek (without consuming) the pending authorization."""
    with _lock:
        _gc_locked(_now(now))
        return _pop_pending_locked(broker_state, consume=False)


def complete_pending(broker_state: str, *, session: Session, now: Optional[int] = None) -> str:
    """Consume a pending authorization (single use) and mint a one-time gateway code bound to the
    desktop's challenge + ``session``."""
    now = _now(now)
    with _lock:
        _gc_locked(now)
        pending = _pop_pending_locked(broker_state, consume=True)
        if not _capacity_ok_locked():
            raise NativeFlowError("native-flow code store at capacity")
        gw_code = secrets.token_urlsafe(32)
        _issued[gw_code] = _IssuedCode(
            code_challenge=pending.code_challenge, session=session,
            expires_at=now + _CODE_TTL_SECONDS)
    return gw_code


def redeem_code(*, code: str, code_verifier: str, now: Optional[int] = None) -> Session:
    """Verify PKCE + consume a gateway code; return the bound :class:`Session`. The entry is popped
    BEFORE the PKCE check so a wrong verifier cannot be retried (no oracle, no replay)."""
    now = _now(now)
    with _lock:
        _gc_locked(now)
        issued = _issued.pop(code, None)
    if issued is None:
        raise CodeInvalid("unknown, expired, or already-redeemed code")
    if issued.expires_at < now:
        raise CodeInvalid("code expired")
    if not hmac.compare_digest(issued.code_challenge, _s256(code_verifier)):
        raise CodeInvalid("PKCE verification failed")
    return issued.session


def _reset_for_tests() -> None:
    """Test-only: drop all pending + issued state."""
    with _lock:
        _pending.clear()
        _issued.clear()
