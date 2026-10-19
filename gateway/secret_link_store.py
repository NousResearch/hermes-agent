"""In-memory records for one-time secret entry links (``gateway/secret_link.py``).

Records live only in the gateway process: a restart invalidates every open link. A record keeps
``SHA-256(token)``, never the token; lookups compare hashes with ``hmac.compare_digest``. All
state changes run under one lock, so ``open -> used`` is a compare-and-set two parallel submits
cannot both win. Expiry, the per-chat cap, the per-record failure cap and the per-client rate
limit are fixed on purpose (no configuration).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

LINK_TTL_SECONDS = 600
MAX_FAILED_SUBMITS = 5
MAX_OPEN_LINKS_PER_CHAT = 5
CLIENT_REQUESTS_PER_MINUTE = 10
_RATE_WINDOW_SECONDS = 60.0
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}")

OPEN, USED, EXPIRED = "open", "used", "expired"
# What asked for the link: the /secret command, or a skill turn that is waiting for a missing key.
TRIGGER_COMMAND, TRIGGER_MISSING_KEY = "command", "missing-key"


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


@dataclass(eq=False)
class SecretLinkRequest:
    """One link. ``notify`` sends a text to the request's chat; ``on_done(success)`` fires once
    when the record leaves ``open`` for good (the waiting skill turn of a missing-key request)."""
    token_hash: bytes
    chat_key: str
    owner_home: Optional[Path]
    suggested_key: str
    expires_at: float
    notify: Callable[[str], Awaitable[Any]]
    on_done: Optional[Callable[[bool], None]] = None
    trigger: str = TRIGGER_COMMAND
    state: str = OPEN
    nonce: str = ""
    failed_submits: int = 0
    done: bool = field(default=False)

    def __repr__(self) -> str:
        return "<SecretLinkRequest>"


class SecretLinkStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._records: list[SecretLinkRequest] = []
        self._client_hits: dict[str, deque] = {}

    def create(self, *, chat_key: str, owner_home: Optional[Path], suggested_key: str,
               notify: Callable[[str], Awaitable[Any]],
               on_done: Optional[Callable[[bool], None]] = None,
               trigger: str = TRIGGER_COMMAND) -> Optional[tuple[str, SecretLinkRequest]]:
        """New ``(token, record)``, or None when the chat already has the maximum open links."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._expire_due_locked()
            if sum(r.chat_key == chat_key and r.state == OPEN for r in self._records) >= MAX_OPEN_LINKS_PER_CHAT:
                return None
            record = SecretLinkRequest(
                token_hash=_digest(token), chat_key=chat_key, owner_home=owner_home,
                suggested_key=suggested_key, expires_at=self._clock() + LINK_TTL_SECONDS,
                notify=notify, on_done=on_done, trigger=trigger)
            self._records.append(record)
        return token, record

    def lookup(self, token: str) -> Optional[SecretLinkRequest]:
        """The open record for ``token``; None for malformed, unknown, used or expired tokens.
        An expired record found here is marked ``expired`` and removed."""
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
            return None
        digest = _digest(token)
        found = None
        with self._lock:
            for record in self._records:  # no early exit: every hash is compared
                if hmac.compare_digest(record.token_hash, digest):
                    found = record
            if found is None or found.state != OPEN:
                return None
            if self._clock() > found.expires_at:
                self._expire_locked(found)
                self._records.remove(found)
                return None
            return found

    def rotate_nonce(self, record: SecretLinkRequest) -> str:
        nonce = secrets.token_urlsafe(32)
        with self._lock:
            record.nonce = nonce
        return nonce

    def nonce_matches(self, record: SecretLinkRequest, nonce: str) -> bool:
        with self._lock:
            expected = record.nonce
        return bool(expected) and isinstance(nonce, str) and hmac.compare_digest(
            expected.encode("ascii"), nonce.encode("utf-8"))

    def claim(self, record: SecretLinkRequest) -> bool:
        """Atomic ``open -> used``; False when another submit already won or the link expired."""
        with self._lock:
            if record.state != OPEN or self._clock() > record.expires_at:
                return False
            record.state = USED
            return True

    def release(self, record: SecretLinkRequest) -> None:
        """``used -> open`` after a submit that wrote nothing (the link stays usable)."""
        with self._lock:
            if record.state == USED and not record.done:
                record.state = OPEN

    def finish(self, record: SecretLinkRequest, success: bool) -> None:
        """Report the outcome of a used record to its waiter (once)."""
        with self._lock:
            self._report_locked(record, success)

    def record_failure(self, record: SecretLinkRequest) -> bool:
        """Count a failed submit; True when the record is now expired (failure cap reached)."""
        with self._lock:
            record.failed_submits += 1
            if record.failed_submits < MAX_FAILED_SUBMITS:
                return False
            self._expire_locked(record)
            if record in self._records:
                self._records.remove(record)
            return True

    def sweep(self) -> None:
        """Drop used and expired records (the gateway's periodic cleanup)."""
        with self._lock:
            self._expire_due_locked()
            self._records = [r for r in self._records if r.state == OPEN]
            cutoff = self._clock() - _RATE_WINDOW_SECONDS
            for client in list(self._client_hits):
                hits = self._client_hits[client]
                while hits and hits[0] <= cutoff:
                    hits.popleft()
                if not hits:
                    del self._client_hits[client]

    def clear(self) -> None:
        """Drop every record (gateway shutdown); open waiters see a failure."""
        with self._lock:
            for record in self._records:
                self._expire_locked(record)
            self._records = []
            self._client_hits = {}

    def allow_client(self, client: str) -> bool:
        """Sliding one-minute window per client address."""
        now = self._clock()
        with self._lock:
            hits = self._client_hits.setdefault(str(client or ""), deque())
            while hits and hits[0] <= now - _RATE_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= CLIENT_REQUESTS_PER_MINUTE:
                return False
            hits.append(now)
            return True

    def _expire_due_locked(self) -> None:
        now = self._clock()
        for record in self._records:
            if record.state == OPEN and now > record.expires_at:
                self._expire_locked(record)

    def _expire_locked(self, record: SecretLinkRequest) -> None:
        if record.state == OPEN:
            record.state = EXPIRED
        self._report_locked(record, False)

    @staticmethod
    def _report_locked(record: SecretLinkRequest, success: bool) -> None:
        if record.done:
            return
        record.done = True
        if record.on_done is not None:
            record.on_done(success)


STORE = SecretLinkStore()
