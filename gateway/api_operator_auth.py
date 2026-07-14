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


class OperatorCredentialStore:
    """Persists hashed, revocable operator bearer tokens.

    Storage is versioned JSON at the configured path. Only SHA-256 token
    hashes are ever written to disk; the raw token is returned exactly once,
    from :meth:`issue`. All read-modify-write cycles are guarded by a single
    ``threading.RLock`` and persisted via a temp-file-then-atomic-replace
    sequence with owner-only (``0600``) permissions, mirroring
    ``gateway/pairing.py``.
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
        if not token:
            return None
        candidate_hash = _hash_token(token)

        with self._lock:
            data = self._load()
            for credential_id, record in data["credentials"].items():
                if not isinstance(record, dict):
                    continue
                if record.get("revoked_at") is not None:
                    continue
                stored_hash = record.get("token_hash")
                if not isinstance(stored_hash, str):
                    continue
                if hmac.compare_digest(candidate_hash, stored_hash):
                    scopes = normalize_scopes(record.get("scopes") or [])
                    return AuthPrincipal(
                        credential_id=credential_id,
                        scopes=scopes,
                        is_superuser=False,
                    )
        return None

    def list_credentials(self) -> list[CredentialSummary]:
        with self._lock:
            data = self._load()
            records = list(data["credentials"].items())

        summaries = []
        for credential_id, record in records:
            if not isinstance(record, dict):
                continue
            summaries.append(
                CredentialSummary(
                    credential_id=credential_id,
                    label=record.get("label", ""),
                    scopes=normalize_scopes(record.get("scopes") or []),
                    created_at=record.get("created_at", 0.0),
                    revoked_at=record.get("revoked_at"),
                )
            )
        summaries.sort(key=lambda summary: summary.created_at)
        return summaries

    def revoke(self, credential_id: str) -> bool:
        with self._lock:
            data = self._load()
            record = data["credentials"].get(credential_id)
            if not isinstance(record, dict) or record.get("revoked_at") is not None:
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
        self._path.parent.mkdir(parents=True, exist_ok=True)
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
