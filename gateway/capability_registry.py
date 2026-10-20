"""Local, fail-closed specialist capability declarations.

The registry stores configured declarations and resolves exact advisory scope.
It does not route work, create profiles, invoke models, or contact providers.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

_MAX_EXPIRY_TIMESTAMP = 253402300799
_PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,95}$")
_REASON_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS capability_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    permissions_hash TEXT NOT NULL,
    domain TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    evidence_class TEXT NOT NULL,
    requested_permissions_json TEXT NOT NULL,
    expires_at INTEGER,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_capability_profile_resolution
ON capability_profiles(signature_hash, evidence_class, status);

CREATE TABLE IF NOT EXISTS specialist_profile_revocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revocation_hash TEXT NOT NULL UNIQUE,
    capability_profile_id INTEGER NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    permissions_hash TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_specialist_profile_revocation
ON specialist_profile_revocations(
    capability_profile_id, profile_id, signature_hash, permissions_hash
);

CREATE TRIGGER IF NOT EXISTS capability_profiles_no_update
BEFORE UPDATE ON capability_profiles BEGIN
    SELECT RAISE(ABORT, 'capability_profiles is append-only');
END;
CREATE TRIGGER IF NOT EXISTS capability_profiles_no_delete
BEFORE DELETE ON capability_profiles BEGIN
    SELECT RAISE(ABORT, 'capability_profiles is append-only');
END;
CREATE TRIGGER IF NOT EXISTS specialist_profile_revocations_no_update
BEFORE UPDATE ON specialist_profile_revocations BEGIN
    SELECT RAISE(ABORT, 'specialist_profile_revocations is append-only');
END;
CREATE TRIGGER IF NOT EXISTS specialist_profile_revocations_no_delete
BEFORE DELETE ON specialist_profile_revocations BEGIN
    SELECT RAISE(ABORT, 'specialist_profile_revocations is append-only');
END;
"""


def _canonical_tokens(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple of strings")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must contain only non-empty strings")
    return tuple(sorted(set(values)))


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _kanban_db():
    """Load board storage only when registry I/O begins."""
    from hermes_cli import kanban_db

    return kanban_db


def _expiry_timestamp(expires_at: datetime | int | float | None) -> int | None:
    if expires_at is None:
        return None
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            raise ValueError("expires_at datetime must be timezone-aware")
        value = int(expires_at.timestamp())
    elif isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise TypeError("expires_at must be a timestamp or timezone-aware datetime")
    else:
        value = int(expires_at)
    if value > _MAX_EXPIRY_TIMESTAMP:
        raise ValueError("expires_at is outside the supported range")
    return value


def _unexpired(expires_at: object, *, now: int) -> bool:
    if expires_at is None:
        return True
    return isinstance(expires_at, int) and not isinstance(expires_at, bool) and now < expires_at <= _MAX_EXPIRY_TIMESTAMP


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    """Canonical advisory scope declared by a specialist profile."""

    domain: str
    actions: tuple[str, ...]
    evidence_class: str
    requested_permissions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be a non-empty string")
        if not isinstance(self.evidence_class, str) or not self.evidence_class.strip():
            raise ValueError("evidence_class must be a non-empty string")
        object.__setattr__(self, "actions", _canonical_tokens(self.actions, field="actions"))
        object.__setattr__(
            self,
            "requested_permissions",
            _canonical_tokens(self.requested_permissions, field="requested_permissions"),
        )

    @property
    def signature_hash(self) -> str:
        return _hash_payload(
            {
                "actions": self.actions,
                "domain": self.domain,
                "evidence_class": self.evidence_class,
            }
        )

    @property
    def permissions_hash(self) -> str:
        return _hash_payload({"requested_permissions": self.requested_permissions})


@dataclass(frozen=True, slots=True)
class RegistryResolution:
    """Fail-closed resolution; only ``active_match`` carries a profile."""

    status: Literal["active_match", "no_match", "ambiguous", "unavailable"]
    profile: str | None
    reason: str


class CapabilityRegistry:
    """Persist and resolve configured specialist capability declarations."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        board: str | None = None,
        configured_profiles: Mapping[str, CapabilitySignature] | None = None,
    ) -> None:
        declarations = dict(configured_profiles or {})
        for profile_id, signature in declarations.items():
            self._validate_profile_id(profile_id)
            if not isinstance(signature, CapabilitySignature):
                raise TypeError("configured profile declarations must be CapabilitySignature values")
        self._db_path = db_path
        self._board = board
        self._configured_profiles = declarations

    @staticmethod
    def _validate_profile_id(profile_id: object) -> str:
        if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
            raise ValueError("profile_id must be a bounded canonical identifier")
        return profile_id

    @contextmanager
    def _connection(self) -> Iterator[object]:
        """Open the board-local registry connection after idempotent schema setup."""
        with _kanban_db().connect_closing(self._db_path, board=self._board) as conn:
            conn.executescript(_SCHEMA)
            yield conn

    def register_configured_profile(
        self,
        profile_id: str,
        *,
        expires_at: datetime | int | float | None = None,
    ) -> int:
        """Return one active row for the exact trusted declaration generation.

        A revocation permanently hides only its immutable row. Calling this
        trusted registration boundary again after revocation appends a new
        generation; repeated calls then reuse that unrevoked generation.
        """
        self._validate_profile_id(profile_id)
        signature = self._configured_profiles.get(profile_id)
        if signature is None:
            raise ValueError("profile_id is not present in configured specialist declarations")
        created_at = int(time.time())
        expires_at_timestamp = _expiry_timestamp(expires_at)
        with self._connection() as conn:
            with _kanban_db().write_txn(conn):
                existing = conn.execute(
                    """
                    SELECT profiles.id FROM capability_profiles AS profiles
                    WHERE profiles.profile_id = ?
                      AND profiles.signature_hash = ?
                      AND profiles.permissions_hash = ?
                      AND profiles.status = 'active'
                      AND profiles.expires_at IS ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM specialist_profile_revocations AS revocations
                          WHERE revocations.capability_profile_id = profiles.id
                            AND revocations.profile_id = profiles.profile_id
                            AND revocations.signature_hash = profiles.signature_hash
                            AND revocations.permissions_hash = profiles.permissions_hash
                      )
                    LIMIT 1
                    """,
                    (
                        profile_id,
                        signature.signature_hash,
                        signature.permissions_hash,
                        expires_at_timestamp,
                    ),
                ).fetchone()
                if existing is not None:
                    return int(existing["id"])
                cursor = conn.execute(
                    """
                    INSERT INTO capability_profiles (
                        profile_id, signature_hash, permissions_hash, domain,
                        actions_json, evidence_class, requested_permissions_json,
                        expires_at, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        profile_id,
                        signature.signature_hash,
                        signature.permissions_hash,
                        signature.domain,
                        json.dumps(signature.actions, separators=(",", ":")),
                        signature.evidence_class,
                        json.dumps(signature.requested_permissions, separators=(",", ":")),
                        expires_at_timestamp,
                        created_at,
                    ),
                )
                return int(cursor.lastrowid)

    def add_active(self, **_: object) -> None:
        """Reject the former arbitrary active-profile entry point."""
        raise ValueError("direct arbitrary active profile registration is disabled; use configured declarations")

    def revoke(
        self,
        *,
        declaration_id: int,
        profile_id: str,
        signature: CapabilitySignature,
        reason_code: str,
        now: int | None = None,
    ) -> str:
        """Append an immutable revocation for one exact declaration row."""
        if isinstance(declaration_id, bool) or not isinstance(declaration_id, int) or declaration_id <= 0:
            raise ValueError("declaration_id must be a positive integer")
        self._validate_profile_id(profile_id)
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        if not isinstance(reason_code, str) or not _REASON_CODE_RE.fullmatch(reason_code):
            raise ValueError("reason_code must be a bounded canonical code")
        created_at = int(time.time()) if now is None else now
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise ValueError("now must be an integer timestamp")
        with self._connection() as conn:
            with _kanban_db().write_txn(conn):
                declaration = conn.execute(
                    """
                    SELECT profiles.id, revocations.revocation_hash
                    FROM capability_profiles AS profiles
                    LEFT JOIN specialist_profile_revocations AS revocations
                      ON revocations.capability_profile_id = profiles.id
                     AND revocations.profile_id = profiles.profile_id
                     AND revocations.signature_hash = profiles.signature_hash
                     AND revocations.permissions_hash = profiles.permissions_hash
                    WHERE profiles.id = ?
                      AND profiles.profile_id = ?
                      AND profiles.signature_hash = ?
                      AND profiles.permissions_hash = ?
                      AND profiles.status = 'active'
                    LIMIT 1
                    """,
                    (
                        declaration_id,
                        profile_id,
                        signature.signature_hash,
                        signature.permissions_hash,
                    ),
                ).fetchone()
                if declaration is None:
                    raise ValueError("declaration_id does not match the exact active capability declaration")
                if declaration["revocation_hash"] is not None:
                    return str(declaration["revocation_hash"])
                receipt = _hash_payload(
                    {
                        "capability_profile_id": declaration_id,
                        "permissions_hash": signature.permissions_hash,
                        "profile_id": profile_id,
                        "reason_code": reason_code,
                        "signature_hash": signature.signature_hash,
                    }
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO specialist_profile_revocations (
                        revocation_hash, capability_profile_id, profile_id,
                        signature_hash, permissions_hash, reason_code, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt,
                        declaration_id,
                        profile_id,
                        signature.signature_hash,
                        signature.permissions_hash,
                        reason_code,
                        created_at,
                    ),
                )
        return receipt

    def resolve(self, signature: CapabilitySignature) -> RegistryResolution:
        """Resolve exactly one active, unexpired, non-expanding local profile."""
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT profile_id, signature_hash, permissions_hash, domain,
                           actions_json, evidence_class, requested_permissions_json, expires_at
                    FROM capability_profiles AS profiles
                    WHERE status = 'active' AND signature_hash = ?
                      AND permissions_hash = ? AND evidence_class = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM specialist_profile_revocations AS revocations
                          WHERE revocations.capability_profile_id = profiles.id
                            AND revocations.profile_id = profiles.profile_id
                            AND revocations.signature_hash = profiles.signature_hash
                            AND revocations.permissions_hash = profiles.permissions_hash
                      )
                    ORDER BY profile_id, id
                    """,
                    (
                        signature.signature_hash,
                        signature.permissions_hash,
                        signature.evidence_class,
                    ),
                ).fetchall()
        except Exception as exc:
            return RegistryResolution(
                status="unavailable",
                profile=None,
                reason=f"local capability registry unavailable: {type(exc).__name__}",
            )

        now = int(time.time())
        matches: set[str] = set()
        for row in rows:
            if not _unexpired(row["expires_at"], now=now):
                continue
            try:
                stored = CapabilitySignature(
                    domain=row["domain"],
                    actions=tuple(json.loads(row["actions_json"])),
                    evidence_class=row["evidence_class"],
                    requested_permissions=tuple(json.loads(row["requested_permissions_json"])),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                stored.signature_hash != row["signature_hash"]
                or stored.permissions_hash != row["permissions_hash"]
                or stored.domain != signature.domain
                or stored.actions != signature.actions
                or stored.requested_permissions != signature.requested_permissions
            ):
                continue
            matches.add(row["profile_id"])

        if len(matches) == 1:
            return RegistryResolution(
                "active_match",
                next(iter(matches)),
                "exact active capability profile matched locally",
            )
        if len(matches) > 1:
            return RegistryResolution("ambiguous", None, "multiple active capability profiles matched the requested scope")
        return RegistryResolution("no_match", None, "no active unexpired profile matched the requested scope")
