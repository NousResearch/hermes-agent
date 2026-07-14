"""Hermes operator authorization vocabulary.

Defines the scope domains, scope normalization, and the credential/principal
types shared by the canonical API server's operator-token authentication and
credential-issuance paths (pairing, storage, and per-request scope checks).

Kept independent of aiohttp so it can be imported by storage and enrollment
modules without pulling in the HTTP server.
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
from typing import Iterable, Optional

from utils import atomic_replace

VALID_SCOPE_DOMAINS = frozenset({
    "chat", "sessions", "profiles", "providers", "skills",
    "tools", "memory", "tasks", "gateway", "settings",
})


def normalize_scopes(values: Iterable[str]) -> tuple[str, ...]:
    scopes = {str(value).strip().lower() for value in values if str(value).strip()}
    for scope in scopes:
        if scope == "*":
            continue
        domain, separator, access = scope.partition(":")
        if separator != ":" or domain not in VALID_SCOPE_DOMAINS or access not in {"read", "write"}:
            raise ValueError(f"unknown scope: {scope}")
    if "*" in scopes:
        return ("*",)
    return tuple(sorted(scopes))


@dataclass(frozen=True)
class AuthPrincipal:
    credential_id: str
    scopes: tuple[str, ...]
    is_superuser: bool = False

    def allows(self, required: str) -> bool:
        return self.is_superuser or "*" in self.scopes or required in self.scopes


@dataclass(frozen=True)
class IssuedCredential:
    credential_id: str
    token: str
    label: str
    scopes: tuple[str, ...]
    created_at: float


@dataclass(frozen=True)
class CredentialSummary:
    credential_id: str
    label: str
    scopes: tuple[str, ...]
    created_at: float
    revoked_at: float | None


STORE_SCHEMA_VERSION = 1
TOKEN_PREFIX = "hop_"
CREDENTIAL_ID_PREFIX = "hoc_"
LABEL_MAX_LENGTH = 80


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_record(record: object) -> Optional[dict]:
    """Interpret one stored credential record, failing closed.

    Returns a record dict with a validated, normalized ``scopes`` tuple and a
    numeric ``created_at`` when the record is structurally sound, or ``None``
    when it is malformed or forward-incompatible. Any single corrupt record is
    dropped from the active set rather than propagated, so one bad entry can
    never raise out of (and thereby DoS) ``authenticate`` or
    ``list_credentials``.

    A record is dropped when it is not a dict, its ``token_hash`` is not an
    ASCII string, its ``scopes`` is not a list of known scope strings, or its
    ``created_at`` is not numeric.
    """
    if not isinstance(record, dict):
        return None

    token_hash = record.get("token_hash")
    if not isinstance(token_hash, str) or not token_hash.isascii():
        return None

    raw_scopes = record.get("scopes")
    if not isinstance(raw_scopes, list) or not all(
        isinstance(scope, str) for scope in raw_scopes
    ):
        return None
    try:
        scopes = normalize_scopes(raw_scopes)
    except ValueError:
        return None

    created_at = record.get("created_at")
    if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
        return None

    revoked_at = record.get("revoked_at")
    if revoked_at is not None and not isinstance(revoked_at, (int, float)):
        return None

    return {
        "token_hash": token_hash,
        "label": record.get("label", "") if isinstance(record.get("label"), str) else "",
        "scopes": scopes,
        "created_at": float(created_at),
        "revoked_at": revoked_at,
    }


class OperatorCredentialStore:
    """Persists hashed, revocable operator bearer tokens.

    Storage is versioned JSON at the configured path. Only SHA-256 token
    hashes are ever written to disk; the raw token is returned exactly once,
    from :meth:`issue`. Persistence goes through a temp-file-then-atomic-replace
    sequence with owner-only (``0600``) permissions, mirroring
    ``gateway/pairing.py``.

    Concurrency invariant: this class assumes a single, long-lived instance per
    store path within one process (the way ``APIServerAdapter`` wires it in
    Task 3). The internal ``threading.RLock`` serializes read-modify-write
    cycles only across threads sharing that one instance. Instantiating two
    stores over the same path — or running two processes against it — is
    unsupported: their writes are not coordinated and concurrent :meth:`issue`
    or :meth:`revoke` calls can lose updates. No cross-process file locking is
    implemented by design.

    Records are parsed defensively on every read: any structurally invalid or
    forward-incompatible record is skipped rather than raised, so a single
    corrupt entry cannot poison authentication or listing for the valid ones.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()

    def issue(self, label: str, scopes: Iterable[str]) -> IssuedCredential:
        normalized_scopes = normalize_scopes(scopes)
        clean_label = str(label or "").strip()[:LABEL_MAX_LENGTH]
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        credential_id = CREDENTIAL_ID_PREFIX + secrets.token_hex(8)
        created_at = time.time()

        with self._lock:
            data = self._load()
            data["credentials"][credential_id] = {
                "token_hash": _hash_token(token),
                "label": clean_label,
                "scopes": list(normalized_scopes),
                "created_at": created_at,
                "revoked_at": None,
            }
            self._save(data)

        return IssuedCredential(
            credential_id=credential_id,
            token=token,
            label=clean_label,
            scopes=normalized_scopes,
            created_at=created_at,
        )

    def authenticate(self, token: str) -> Optional[AuthPrincipal]:
        if not isinstance(token, str) or not token:
            return None
        candidate_hash = _hash_token(token)

        with self._lock:
            data = self._load()
            for credential_id, raw_record in data["credentials"].items():
                record = _parse_record(raw_record)
                if record is None or record["revoked_at"] is not None:
                    continue
                if hmac.compare_digest(candidate_hash, record["token_hash"]):
                    return AuthPrincipal(
                        credential_id=credential_id,
                        scopes=record["scopes"],
                        is_superuser=False,
                    )
        return None

    def list_credentials(self) -> list[CredentialSummary]:
        with self._lock:
            data = self._load()
            records = list(data["credentials"].items())

        summaries = []
        for credential_id, raw_record in records:
            record = _parse_record(raw_record)
            if record is None:
                continue
            summaries.append(
                CredentialSummary(
                    credential_id=credential_id,
                    label=record["label"],
                    scopes=record["scopes"],
                    created_at=record["created_at"],
                    revoked_at=record["revoked_at"],
                )
            )
        summaries.sort(key=lambda summary: summary.created_at)
        return summaries

    def revoke(self, credential_id: str) -> bool:
        with self._lock:
            data = self._load()
            raw_record = data["credentials"].get(credential_id)
            if _parse_record(raw_record) is None:
                return False
            record = data["credentials"][credential_id]
            if record.get("revoked_at") is not None:
                return False
            record["revoked_at"] = time.time()
            self._save(data)
        return True

    def _load(self) -> dict:
        if not self._path.exists():
            return {"version": STORE_SCHEMA_VERSION, "credentials": {}}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": STORE_SCHEMA_VERSION, "credentials": {}}
        if not isinstance(raw, dict):
            return {"version": STORE_SCHEMA_VERSION, "credentials": {}}
        credentials = raw.get("credentials")
        if not isinstance(credentials, dict):
            credentials = {}
        return {"version": STORE_SCHEMA_VERSION, "credentials": credentials}

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
