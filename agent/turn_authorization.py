"""Opaque per-turn authorization state for trusted gateway metadata.

The bearer value is deliberately unavailable through public attributes, repr,
pickling, or JSON serialization.  Only the MCP call boundary can turn the
current holder into an HTTP header.
"""

from __future__ import annotations

import contextvars
import hmac
import math
import re
import threading
import time
from contextlib import contextmanager
from typing import Iterator
from typing import Any

FIZKO_PERSON_ACCESS_TOKEN_PARAM = "_fizko_person_access_token"
FIZKO_PERSON_ACCESS_TOKEN_EXPIRES_AT_PARAM = "_fizko_person_access_token_expires_at"
FIZKO_PERSON_PRINCIPAL_ID_PARAM = "_fizko_person_principal_id"
FIZKO_PERSON_ADMISSION_ID_PARAM = "_fizko_person_admission_id"

_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{1,16384}\Z")
_PRINCIPAL_ID = re.compile(r"[a-f0-9]{64}\Z")
_ADMISSION_ID = re.compile(r"[a-f0-9]{32}\Z")


class TurnAuthorization:
    """Opaque tri-state authorization: static, personal bearer, or blocked personal descendant."""

    __slots__ = (
        "__admission_id",
        "__admission_lock",
        "__admission_state",
        "__expires_at",
        "__peer",
        "__personal",
        "__principal_id",
        "__token",
    )

    def __init__(
        self,
        token: str | None,
        expires_at: float | None = None,
        *,
        personal: bool = False,
        principal_id: str | None = None,
        admission_id: str | None = None,
    ) -> None:
        self.__token = token
        self.__expires_at = expires_at
        self.__personal = personal
        self.__principal_id = principal_id
        self.__admission_id = admission_id
        self.__admission_lock = threading.Lock()
        self.__admission_state = "pending"
        self.__peer = None

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        expires_at: Any = None,
        principal_id: Any = None,
        admission_id: Any = None,
        require_admission: bool = False,
    ) -> "TurnAuthorization":
        if raw is None:
            if expires_at is not None or principal_id is not None or admission_id is not None:
                raise ValueError("person token metadata requires a bearer token")
            return cls(None)
        if not isinstance(raw, str) or not _BEARER_TOKEN.fullmatch(raw):
            raise ValueError("_fizko_person_access_token must be a non-empty bearer token")
        if expires_at is None:
            raise ValueError("person token expiry is required")
        if principal_id is None:
            raise ValueError("person principal id is required")
        if not isinstance(principal_id, str) or not _PRINCIPAL_ID.fullmatch(principal_id):
            raise ValueError("_fizko_person_principal_id must be a SHA-256 identifier")
        if require_admission and admission_id is None:
            raise ValueError("person admission id is required")
        if admission_id is not None and (
            not isinstance(admission_id, str) or not _ADMISSION_ID.fullmatch(admission_id)
        ):
            raise ValueError("_fizko_person_admission_id must be a private opaque identifier")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
            raise ValueError("_fizko_person_access_token_expires_at must be a Unix timestamp")
        try:
            expiry = float(expires_at)
        except (OverflowError, ValueError):
            raise ValueError(
                "_fizko_person_access_token_expires_at must be a Unix timestamp"
            ) from None
        if not math.isfinite(expiry) or expiry <= 0:
            raise ValueError("_fizko_person_access_token_expires_at must be a Unix timestamp")
        return cls(
            raw,
            expiry,
            personal=True,
            principal_id=principal_id,
            admission_id=admission_id,
        )

    @classmethod
    def blocked(cls) -> "TurnAuthorization":
        """A non-secret fence for work derived from a personal turn."""
        return cls(None, personal=True)

    def __repr__(self) -> str:
        return "<TurnAuthorization [REDACTED]>"

    @property
    def has_token(self) -> bool:
        return self.__token is not None

    @property
    def is_personal(self) -> bool:
        return self.__personal

    @property
    def is_expired(self) -> bool:
        return self.__expires_at is not None and self.__expires_at <= time.time()

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("TurnAuthorization cannot be serialized")

    def same_credential(self, other: object) -> bool:
        if not isinstance(other, TurnAuthorization):
            return False
        if self.__personal != other.__personal:
            return False
        if self.__token is None or other.__token is None:
            return self.__token is other.__token
        return hmac.compare_digest(self.__token, other.__token)

    def same_principal(self, other: object) -> bool:
        """Compare stable owners without exposing either opaque identifier."""
        if not isinstance(other, TurnAuthorization):
            return False
        if self.__personal != other.__personal:
            return False
        if self.__principal_id is None or other.__principal_id is None:
            return self.__principal_id is other.__principal_id
        return hmac.compare_digest(self.__principal_id, other.__principal_id)

    def _fizko_admission_id(self) -> str:
        """Private accounting correlation; never an authenticator."""
        return self.__admission_id or ""

    def _fizko_bind_peer(self, peer: Any) -> None:
        """Pin the peer that supplied this authority, once.

        Admission events carry the private correlation id and must reach only the
        peer that opened the turn -- a session with several attached viewers holds
        a fan-out transport. Deferred work runs outside the request context, so the
        peer is captured here rather than read at emit time.
        """
        if peer is not None and self.__peer is None:
            self.__peer = peer

    def _fizko_peer(self) -> Any:
        return self.__peer

    def _fizko_emit_admission_event(self, status: str, emitter: Any) -> bool:
        """Emit and atomically commit one lifecycle state transition.

        ``started`` and ``terminal`` are each emitted at most once, and a late
        ``started`` can never resurrect an already terminal admission. A failed
        transport write leaves the transition retryable.
        """
        if not self.__admission_id or status not in {"started", "terminal"}:
            return False
        with self.__admission_lock:
            if self.__admission_state == "terminal":
                return False
            if status == "started":
                if self.__admission_state != "pending":
                    return False
            # Gateway transports are terminal sinks: their internal write locks
            # never call back into admission handling. Keep the holder lock over
            # the write so a racing terminal cannot overtake or duplicate started.
            # Transport.write is specified to return bool. Test sinks commonly
            # return None after recording, so only an explicit False means the
            # live delivery failed and should remain retryable.
            if emitter() is False:
                return False
            self.__admission_state = status
            return True

    def _fizko_authorization_header(self) -> str:
        return f"Bearer {self.__token}" if self.__token is not None and not self.is_expired else ""


_CURRENT_TURN_AUTHORIZATION: contextvars.ContextVar[TurnAuthorization | None] = contextvars.ContextVar(
    "hermes_current_turn_authorization", default=None
)


def set_current_turn_authorization(holder: TurnAuthorization):
    return _CURRENT_TURN_AUTHORIZATION.set(holder)


def reset_current_turn_authorization(token) -> None:
    _CURRENT_TURN_AUTHORIZATION.reset(token)


@contextmanager
def without_turn_authorization() -> Iterator[None]:
    """Strip bearer authority while retaining a personal-origin fail-closed fence."""
    current = _CURRENT_TURN_AUTHORIZATION.get()
    replacement = (
        TurnAuthorization.blocked()
        if current is not None and current.is_personal
        else None
    )
    token = _CURRENT_TURN_AUTHORIZATION.set(replacement)
    try:
        yield
    finally:
        _CURRENT_TURN_AUTHORIZATION.reset(token)


def current_fizko_authorization_header() -> str:
    """Authorization override for an opted-in Fizko call; empty means fail closed."""
    holder = _CURRENT_TURN_AUTHORIZATION.get()
    return holder._fizko_authorization_header() if holder is not None else ""


def current_fizko_authorization_state() -> tuple[bool, str]:
    """Whether this is a personal turn and its still-valid Authorization header."""
    holder = _CURRENT_TURN_AUTHORIZATION.get()
    if holder is None or not holder.is_personal:
        return False, ""
    return True, holder._fizko_authorization_header()


def current_turn_authorization() -> TurnAuthorization | None:
    """Return the opaque holder so a transport closure can revalidate it on every attempt."""
    return _CURRENT_TURN_AUTHORIZATION.get()
