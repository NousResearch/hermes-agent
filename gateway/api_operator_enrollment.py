"""One-time operator enrollment codes and token lifecycle.

Lets an already-authorized caller (typically the desktop/CLI owner) mint a
short-lived, single-use pairing code that a new device (e.g. the Navivox
Android app) exchanges for a scoped operator bearer token, without ever
placing that bearer token in a QR code, deep link, or handoff payload.

Mirrors ``gateway/pairing.py``'s salted-hash storage, ``threading.RLock``
guard, and temp-file/atomic-replace persistence conventions, but is kept
intentionally independent of any messaging-platform allowlist: enrollment
codes exist purely to bootstrap :class:`~gateway.api_operator_auth.
OperatorCredentialStore` tokens for the canonical API server.
"""

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from gateway.api_operator_auth import IssuedCredential, OperatorCredentialStore, normalize_scopes
from utils import atomic_replace

# Pairing codes expire five minutes after creation (Global Constraints).
PAIRING_CODE_TTL_SECONDS = 300

# Wrong-code guesses lock further attempts out for an hour, mirroring
# gateway/pairing.py's per-platform lockout. Lockout is global to the store
# (there is one enrollment store per Hermes install) rather than per-code,
# since the attack this defends against — brute-forcing the code space — is
# not scoped to any single pending grant.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 3600

LABEL_MAX_LENGTH = 80
CODE_ENTROPY_BYTES = 24  # secrets.token_urlsafe(24)

STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnrollmentGrant:
    code: str
    label: str
    origin: str
    scopes: tuple[str, ...]
    expires_at: float


@dataclass(frozen=True)
class EnrollmentPreview:
    label: str
    origin: str
    scopes: tuple[str, ...]
    expires_at: float


def _normalize_origin(origin: object) -> str:
    """Normalize an origin string for exact-match comparison.

    Origins carry no path/query worth parsing further, so normalization is
    deliberately narrow: trim surrounding whitespace, drop one trailing
    slash, and case-fold. This keeps comparison exact (no scheme/host
    equivalence heuristics) while tolerating the trivial formatting
    variance a client's URL library might introduce.
    """
    text = str(origin or "").strip()
    if text.endswith("/"):
        text = text[:-1]
    return text.lower()


def _hash_code(code: str, salt: bytes) -> str:
    return hashlib.sha256(salt + code.encode("utf-8")).hexdigest()


def _empty_store() -> dict:
    return {
        "version": STORE_SCHEMA_VERSION,
        "enrollments": {},
        "failed_attempts": 0,
        "lockout_until": None,
    }


class OperatorEnrollmentStore:
    """One-time pairing-code enrollment that mints scoped operator tokens.

    ``now`` is an injectable clock (defaults to :func:`time.time`) so tests
    can advance time deterministically instead of sleeping past the TTL /
    lockout window.

    Concurrency invariant matches :class:`OperatorCredentialStore`: one
    long-lived instance per store path within a process. The internal
    ``threading.RLock`` serializes read-modify-write cycles across threads
    sharing that instance; it does not coordinate multiple processes.
    """

    def __init__(
        self,
        path: Path,
        credentials: OperatorCredentialStore,
        now: Callable[[], float] = time.time,
    ):
        self._path = Path(path)
        self._credentials = credentials
        self._now = now
        self._lock = threading.RLock()

    def create(self, label: str, origin: str, scopes: Iterable[str]) -> EnrollmentGrant:
        """Mint a new one-time pairing code.

        Raises ``ValueError`` (via :func:`normalize_scopes`) for an unknown
        scope domain — callers at the HTTP layer are expected to translate
        that into a 400 response.
        """
        normalized_scopes = normalize_scopes(scopes)
        clean_label = str(label or "").strip()[:LABEL_MAX_LENGTH]
        normalized_origin = _normalize_origin(origin)
        code = secrets.token_urlsafe(CODE_ENTROPY_BYTES)
        salt = os.urandom(16)
        entry_id = secrets.token_hex(8)
        created_at = self._now()
        expires_at = created_at + PAIRING_CODE_TTL_SECONDS

        with self._lock:
            data = self._load()
            # Only these fields are ever persisted for an enrollment entry —
            # the raw code itself is never written to disk.
            data["enrollments"][entry_id] = {
                "hash": _hash_code(code, salt),
                "salt": salt.hex(),
                "label": clean_label,
                "origin": normalized_origin,
                "scopes": list(normalized_scopes),
                "created_at": created_at,
                "expires_at": expires_at,
                "consumed_at": None,
            }
            self._save(data)

        return EnrollmentGrant(
            code=code,
            label=clean_label,
            origin=normalized_origin,
            scopes=normalized_scopes,
            expires_at=expires_at,
        )

    def inspect(self, code: str, origin: str) -> Optional[EnrollmentPreview]:
        """Preview a pending grant without consuming it.

        Returns ``None`` for any of: malformed input, unknown code, expired
        or already-consumed code, origin mismatch, or an active lockout.
        Every failure path returns the identical ``None`` — nothing about
        *why* a lookup failed is observable from the return value.
        """
        with self._lock:
            entry = self._locate(code, origin, consume=False)
        if entry is None:
            return None
        return EnrollmentPreview(
            label=entry["label"],
            origin=entry["origin"],
            scopes=tuple(entry["scopes"]),
            expires_at=entry["expires_at"],
        )

    def exchange(self, code: str, origin: str) -> Optional[IssuedCredential]:
        """Consume a pending grant and mint a scoped operator token.

        The code is marked consumed inside the same locked read-modify-write
        cycle that located it, so two concurrent callers racing the same
        code can never both succeed. Token issuance is delegated to
        :meth:`OperatorCredentialStore.issue` after that consumption has
        already been durably persisted.
        """
        with self._lock:
            entry = self._locate(code, origin, consume=True)
        if entry is None:
            return None
        return self._credentials.issue(entry["label"], entry["scopes"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _locate(self, code: object, origin: object, *, consume: bool) -> Optional[dict]:
        """Find, validate, and (optionally) consume the entry for ``code``.

        Must be called while holding ``self._lock``. A wrong or unknown code,
        and a matched-but-expired-or-already-consumed code, count as a
        failed attempt toward the store-wide lockout (this is the
        brute-force-guessing surface). A matched code presented with the
        wrong origin does *not* count toward the lockout, is not consumed,
        and leaks nothing back to the caller beyond "not found" — the
        presenter already demonstrated knowledge of the code; only the
        origin binding failed.
        """
        if not isinstance(code, str) or not code:
            return None

        normalized_origin = _normalize_origin(origin)
        now = self._now()

        data = self._load()
        if self._is_locked_out(data, now):
            return None

        matched_id = None
        matched_entry = None
        for entry_id, entry in data["enrollments"].items():
            if not isinstance(entry, dict):
                continue
            salt_hex = entry.get("salt")
            hash_hex = entry.get("hash")
            if not isinstance(salt_hex, str) or not isinstance(hash_hex, str):
                continue
            try:
                salt = bytes.fromhex(salt_hex)
            except ValueError:
                continue
            if hmac.compare_digest(_hash_code(code, salt), hash_hex):
                matched_id, matched_entry = entry_id, entry
                break

        if matched_entry is None:
            self._record_failed_attempt(data, now)
            self._save(data)
            return None

        expires_at = matched_entry.get("expires_at")
        is_expired = not isinstance(expires_at, (int, float)) or expires_at <= now
        is_consumed = matched_entry.get("consumed_at") is not None
        if is_expired or is_consumed:
            self._record_failed_attempt(data, now)
            self._save(data)
            return None

        if matched_entry.get("origin") != normalized_origin:
            return None  # No lockout bump, no consumption, no leak.

        result = {
            "label": matched_entry.get("label") or "",
            "origin": matched_entry.get("origin") or "",
            "scopes": list(matched_entry.get("scopes") or []),
            "expires_at": expires_at,
        }

        if consume:
            matched_entry["consumed_at"] = now
            data["enrollments"][matched_id] = matched_entry
            self._save(data)

        return result

    @staticmethod
    def _is_locked_out(data: dict, now: float) -> bool:
        locked_until = data.get("lockout_until")
        return isinstance(locked_until, (int, float)) and now < locked_until

    @staticmethod
    def _record_failed_attempt(data: dict, now: float) -> None:
        failures = data.get("failed_attempts", 0)
        if not isinstance(failures, int):
            failures = 0
        failures += 1
        if failures >= MAX_FAILED_ATTEMPTS:
            data["lockout_until"] = now + LOCKOUT_SECONDS
            failures = 0
        data["failed_attempts"] = failures

    def _load(self) -> dict:
        if not self._path.exists():
            return _empty_store()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_store()
        if not isinstance(raw, dict):
            return _empty_store()

        enrollments = raw.get("enrollments")
        if not isinstance(enrollments, dict):
            enrollments = {}

        failed_attempts = raw.get("failed_attempts", 0)
        if not isinstance(failed_attempts, int) or isinstance(failed_attempts, bool):
            failed_attempts = 0

        lockout_until = raw.get("lockout_until")
        if not isinstance(lockout_until, (int, float)):
            lockout_until = None

        return {
            "version": STORE_SCHEMA_VERSION,
            "enrollments": enrollments,
            "failed_attempts": failed_attempts,
            "lockout_until": lockout_until,
        }

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_path, 0o600)
            atomic_replace(tmp_path, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass  # Windows doesn't support chmod the same way.
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
