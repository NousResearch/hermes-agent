"""Private credentials for reciprocal hosted Group Chat control.

The room's home gateway stores only a SHA-256 commitment for each scoped
control credential. A participating peer stores the corresponding bearer
credential and validated home endpoint in the gateway-wide ``state.db``.
Credentials are deliberately absent from reprs, status mappings, and errors.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from gateway.hosted_room_peer import (
    HostedRoomPeerError,
    gateway_room_grant_secret,
    validate_room_link_url,
)


TOKEN_BYTES = 32
MAX_TOKEN_CHARS = 512
MAX_PEER_LINKS = 1024
MAX_LOAD_LINKS = MAX_PEER_LINKS
MAX_ROOM_NAME_CHARS = 200
MAX_ROOM_MEMBERS = 64
MAX_CONTROL_COMMANDS = 4096
CONTROL_COMMAND_RETENTION_SECONDS = 30 * 24 * 60 * 60
ROOM_LIFETIME_EXPIRES_AT = 253_402_300_799.0
_JOURNAL_MODE_LOCK_RETRIES = 5

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PEER_STATUSES = frozenset({"active", "expired", "revoked", "quarantined"})
_DUMMY_DIGEST = hashlib.sha256(b"invalid-hosted-room-control-token").digest()


def control_retry_attempt_id(command_id: Any, task_id: Any) -> str:
    """Return one stable, bounded Retry identity across direct/worker paths."""

    command = str(command_id or "")
    if command.startswith("worker:"):
        command = command[len("worker:") :]
    material = f"{command}|{str(task_id or '')}".encode("utf-8")
    return f"room-retry:{hashlib.sha256(material).hexdigest()}"


class HostedRoomControlError(ValueError):
    """Raised when a reciprocal room-control record is invalid or conflicts."""


class HostedRoomControlConflictError(HostedRoomControlError):
    """Raised when immutable control-link identity changes unexpectedly."""


@dataclass(frozen=True)
class IssuedRoomControlToken:
    """One-time credential returned by a room's authority gateway."""

    room_id: str
    member_id: str
    authority_gateway_id: str
    authority_epoch: int
    control_token: str = field(repr=False)
    status: str
    created_at: float
    expires_at: float

    def as_status(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "member_id": self.member_id,
            "authority_gateway_id": self.authority_gateway_id,
            "authority_epoch": self.authority_epoch,
            "status": self.status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class StoredPeerRoomControl:
    """Private peer-side route back to a room's authority gateway."""

    room_id: str
    member_id: str
    home_url: str
    transport_security: str
    authority_gateway_id: str
    authority_epoch: int
    room_name: str
    member_count: int
    control_token: str = field(repr=False)
    status: str
    created_at: float
    updated_at: float
    expires_at: float
    revoked_at: float | None = None

    def as_status(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "member_id": self.member_id,
            "home_url": self.home_url,
            "transport_security": self.transport_security,
            "authority_gateway_id": self.authority_gateway_id,
            "authority_epoch": self.authority_epoch,
            "room_name": self.room_name,
            "member_count": self.member_count,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            **({"revoked_at": self.revoked_at} if self.revoked_at is not None else {}),
        }


@dataclass(frozen=True)
class PeerRoomControlSave:
    link: StoredPeerRoomControl
    idempotent: bool


@dataclass(frozen=True)
class PeerRoomControlLoad:
    links: tuple[StoredPeerRoomControl, ...]
    quarantined: int
    truncated: bool


@dataclass(frozen=True)
class RoomControlCommandPlan:
    task_ids: tuple[str, ...]
    result: dict[str, Any] | None
    idempotent: bool


@dataclass(frozen=True)
class PendingRoomControlRetry:
    command_id: str
    room_id: str
    member_id: str
    task_ids: tuple[str, ...]


def default_db_path() -> Path:
    """Use the same gateway-wide state database as hosted room authority."""

    from gateway.hosted_rooms import default_db_path as hosted_room_db_path

    return hosted_room_db_path()


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise HostedRoomControlError(f"invalid {label}")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise HostedRoomControlError(f"invalid {label}")
    return normalized


def _authority_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HostedRoomControlError("invalid authority_epoch")
    return value


def _timestamp(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise HostedRoomControlError(f"invalid {label}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HostedRoomControlError(f"invalid {label}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise HostedRoomControlError(f"invalid {label}")
    return parsed


def _room_name(value: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if not normalized or len(normalized) > MAX_ROOM_NAME_CHARS:
        raise HostedRoomControlError("invalid room_name")
    return normalized


def _member_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HostedRoomControlError("invalid member_count")
    if not 1 <= value <= MAX_ROOM_MEMBERS:
        raise HostedRoomControlError("invalid member_count")
    return value


def _normalize_home_url(value: Any) -> tuple[str, str]:
    try:
        return validate_room_link_url(value)
    except HostedRoomPeerError as exc:
        raise HostedRoomControlError("invalid home control endpoint") from exc


def _token_is_strong(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or not _TOKEN_RE.fullmatch(value)
    ):
        return False
    try:
        padding = "=" * (-len(value) % 4)
        material = base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError):
        return False
    return len(material) >= TOKEN_BYTES


def _control_token(value: Any) -> str:
    if not _token_is_strong(value):
        raise HostedRoomControlError("invalid control credential")
    return value


def _derived_control_token(
    *,
    room_id: str,
    member_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    request_id: str,
) -> str:
    material = "\0".join((
        "hermes-room-control-v1",
        room_id,
        member_id,
        authority_gateway_id,
        str(authority_epoch),
        request_id,
    )).encode("utf-8")
    digest = hmac.new(gateway_room_grant_secret(), material, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _secure_db_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError:
            pass


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_control_tokens (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            request_id TEXT NOT NULL,
            token_hash BLOB NOT NULL CHECK (length(token_hash) = 32),
            status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            revoked_at REAL,
            PRIMARY KEY (
                room_id, member_id, authority_gateway_id, authority_epoch
            )
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_control_tokens_room
           ON hosted_room_control_tokens(room_id, status, expires_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_peer_controls (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            room_name TEXT NOT NULL,
            member_count INTEGER NOT NULL CHECK (member_count BETWEEN 1 AND 64),
            home_url TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            control_token TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('active', 'expired', 'revoked', 'quarantined')
            ),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            revoked_at REAL,
            quarantine_reason TEXT,
            PRIMARY KEY (room_id, member_id)
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_peer_controls_status
           ON hosted_room_peer_controls(status, expires_at, updated_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_control_commands (
            command_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            action TEXT NOT NULL,
            task_ids_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
            result_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_control_commands_retention
           ON hosted_room_control_commands(state, updated_at, command_id)"""
    )


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    home = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_control_tokens)")
    }
    peer = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_peer_controls)")
    }
    commands = {
        row[1]
        for row in conn.execute("PRAGMA table_info(hosted_room_control_commands)")
    }
    return (
        {
            "room_id",
            "member_id",
            "authority_gateway_id",
            "authority_epoch",
            "request_id",
            "token_hash",
            "status",
            "created_at",
            "updated_at",
            "expires_at",
            "revoked_at",
        }.issubset(home)
        and {
            "room_id",
            "member_id",
            "room_name",
            "member_count",
            "home_url",
            "transport_security",
            "authority_gateway_id",
            "authority_epoch",
            "control_token",
            "status",
            "created_at",
            "updated_at",
            "expires_at",
            "revoked_at",
            "quarantine_reason",
        }.issubset(peer)
        and {
            "command_id",
            "room_id",
            "member_id",
            "action",
            "task_ids_json",
            "state",
            "result_json",
            "created_at",
            "updated_at",
        }.issubset(commands)
    )


def _migrate_schema(conn: sqlite3.Connection) -> None:
    home = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_control_tokens)")
    }
    if home and "request_id" not in home:
        conn.execute(
            """ALTER TABLE hosted_room_control_tokens
               ADD COLUMN request_id TEXT NOT NULL DEFAULT 'legacy'"""
        )


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = Path(db_path)
    _secure_db_file(path)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        for attempt in range(_JOURNAL_MODE_LOCK_RETRIES):
            try:
                apply_wal_with_fallback(
                    conn, db_label="state.db (hosted room controls)"
                )
                break
            except sqlite3.OperationalError as exc:
                if (
                    str(exc).lower() != "database is locked"
                    or attempt + 1 == _JOURNAL_MODE_LOCK_RETRIES
                ):
                    raise
                time.sleep(0.01 * (2**attempt))
        conn.execute("PRAGMA secure_delete=ON")
        if not _schema_is_current(conn):
            conn.execute("BEGIN IMMEDIATE")
            _migrate_schema(conn)
            _initialize_schema(conn)
            if not _schema_is_current(conn):
                raise HostedRoomControlError(
                    "hosted room control schema is incompatible"
                )
            conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    _secure_db_file(path)
    return conn


@contextmanager
def _transaction(
    db_path: Path | str, *, immediate: bool = False
) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _active_room_scope(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    member_id: str | None = None,
) -> bool:
    table = conn.execute(
        """SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='hosted_rooms'"""
    ).fetchone()
    if table is None:
        return False
    row = conn.execute(
        """SELECT members_json FROM hosted_rooms
                WHERE room_id=? AND authority_gateway_id=?
                  AND authority_epoch=? AND disbanded_at IS NULL""",
        (room_id, authority_gateway_id, authority_epoch),
    ).fetchone()
    if row is None:
        return False
    if member_id is None:
        return True
    try:
        members = json.loads(str(row["members_json"]))
    except Exception:
        return False
    return any(
        isinstance(member, dict)
        and str(member.get("member_id") or member.get("profile") or "") == member_id
        for member in members
    )


def issue_home_control_token(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    expires_at: Any,
    request_id: Any | None = None,
    now: float | None = None,
) -> IssuedRoomControlToken:
    """Create one replay-safe member credential and retain only its hash."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    authority_gateway_id = _identifier(
        authority_gateway_id, label="authority_gateway_id"
    )
    authority_epoch = _authority_epoch(authority_epoch)
    normalized_request_id = (
        _identifier(request_id, label="request_id")
        if request_id is not None
        else f"one-shot:{secrets.token_hex(16)}"
    )
    created_at = _timestamp(time.time() if now is None else now, label="now")
    expires_at = _timestamp(expires_at, label="expires_at")
    if expires_at <= created_at:
        raise HostedRoomControlError("control credential expiry must be in the future")

    control_token = (
        _derived_control_token(
            room_id=room_id,
            member_id=member_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
            request_id=normalized_request_id,
        )
        if request_id is not None
        else secrets.token_urlsafe(TOKEN_BYTES)
    )
    token_hash = hashlib.sha256(control_token.encode("ascii")).digest()
    with _transaction(db_path, immediate=True) as conn:
        if not _active_room_scope(
            conn,
            room_id=room_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
            member_id=member_id,
        ):
            raise HostedRoomControlError(
                "active Group Chat authority scope is unavailable"
            )
        existing = conn.execute(
            """SELECT request_id, status, expires_at
                 FROM hosted_room_control_tokens
                WHERE room_id=? AND member_id=? AND authority_gateway_id=?
                  AND authority_epoch=?""",
            (room_id, member_id, authority_gateway_id, authority_epoch),
        ).fetchone()
        if existing is not None and existing["status"] == "active":
            if (
                str(existing["request_id"]) == normalized_request_id
                and float(existing["expires_at"]) == expires_at
            ):
                return IssuedRoomControlToken(
                    room_id=room_id,
                    member_id=member_id,
                    authority_gateway_id=authority_gateway_id,
                    authority_epoch=authority_epoch,
                    control_token=control_token,
                    status="active",
                    created_at=created_at,
                    expires_at=expires_at,
                )
            if (
                str(existing["request_id"]) != "legacy"
                and float(existing["expires_at"]) > created_at
            ):
                raise HostedRoomControlConflictError(
                    "an active control credential already exists for this scope"
                )
        conn.execute(
            """INSERT INTO hosted_room_control_tokens(
                   room_id, member_id, authority_gateway_id, authority_epoch,
                   request_id, token_hash, status, created_at, updated_at, expires_at,
                   revoked_at
               ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)
               ON CONFLICT(
                   room_id, member_id, authority_gateway_id, authority_epoch
               ) DO UPDATE SET
                   token_hash=excluded.token_hash,
                   request_id=excluded.request_id,
                   status='active',
                   created_at=excluded.created_at,
                   updated_at=excluded.updated_at,
                   expires_at=excluded.expires_at,
                   revoked_at=NULL""",
            (
                room_id,
                member_id,
                authority_gateway_id,
                authority_epoch,
                normalized_request_id,
                token_hash,
                created_at,
                created_at,
                expires_at,
            ),
        )
    return IssuedRoomControlToken(
        room_id=room_id,
        member_id=member_id,
        authority_gateway_id=authority_gateway_id,
        authority_epoch=authority_epoch,
        control_token=control_token,
        status="active",
        created_at=created_at,
        expires_at=expires_at,
    )


def verify_home_control_token(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    control_token: Any,
    now: float | None = None,
) -> bool:
    """Verify one exact live room/member/authority scope in constant time."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    authority_gateway_id = _identifier(
        authority_gateway_id, label="authority_gateway_id"
    )
    authority_epoch = _authority_epoch(authority_epoch)
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    token_shape_valid = _token_is_strong(control_token)
    token_material = (
        control_token.encode("ascii") if token_shape_valid else b"invalid-credential"
    )
    candidate_hash = hashlib.sha256(token_material).digest()

    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT token_hash, status, expires_at
                 FROM hosted_room_control_tokens
                WHERE room_id=? AND member_id=? AND authority_gateway_id=?
                  AND authority_epoch=?""",
            (room_id, member_id, authority_gateway_id, authority_epoch),
        ).fetchone()
        room_active = _active_room_scope(
            conn,
            room_id=room_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
            member_id=member_id,
        )

    stored_hash: bytes = _DUMMY_DIGEST
    eligible = False
    if row is not None:
        raw_hash = row["token_hash"]
        if isinstance(raw_hash, bytes) and len(raw_hash) == 32:
            stored_hash = raw_hash
            try:
                stored_expiry = float(row["expires_at"])
            except (TypeError, ValueError):
                stored_expiry = float("nan")
            eligible = bool(
                row["status"] == "active"
                and math.isfinite(stored_expiry)
                and stored_expiry > timestamp
                and room_active
                and token_shape_valid
            )
    digest_matches = hmac.compare_digest(stored_hash, candidate_hash)
    return bool(eligible and digest_matches)


def revoke_home_control_tokens(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any | None = None,
    authority_gateway_id: Any | None = None,
    authority_epoch: Any | None = None,
    now: float | None = None,
) -> int:
    """Idempotently revoke matching home-side credential commitments."""

    clauses = ["room_id=?", "status='active'"]
    params: list[Any] = [_identifier(room_id, label="room_id")]
    if member_id is not None:
        clauses.append("member_id=?")
        params.append(_identifier(member_id, label="member_id"))
    if authority_gateway_id is not None:
        clauses.append("authority_gateway_id=?")
        params.append(_identifier(authority_gateway_id, label="authority_gateway_id"))
    if authority_epoch is not None:
        clauses.append("authority_epoch=?")
        params.append(_authority_epoch(authority_epoch))
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            f"""UPDATE hosted_room_control_tokens
                    SET status='revoked', updated_at=?, revoked_at=?
                  WHERE {" AND ".join(clauses)}""",
            (timestamp, timestamp, *params),
        )
        return cursor.rowcount


def revoke_home_control_token_value(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    control_token: Any,
    now: float | None = None,
) -> int:
    """Idempotently revoke the exact bearer, including response-lost retries."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    token_hash = hashlib.sha256(_control_token(control_token).encode("ascii")).digest()
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        rows = conn.execute(
            """SELECT token_hash, status FROM hosted_room_control_tokens
                WHERE room_id=? AND member_id=?""",
            (room_id, member_id),
        ).fetchall()
        matched = next(
            (
                row
                for row in rows
                if hmac.compare_digest(bytes(row["token_hash"]), token_hash)
            ),
            None,
        )
        if matched is None:
            raise HostedRoomControlError("control credential is invalid")
        if str(matched["status"]) == "revoked":
            return 0
        changed = conn.execute(
            """UPDATE hosted_room_control_tokens
                  SET status='revoked', updated_at=?, revoked_at=?
                WHERE room_id=? AND member_id=? AND token_hash=?
                  AND status='active'""",
            (timestamp, timestamp, room_id, member_id, token_hash),
        )
        if changed.rowcount not in {0, 1}:
            raise HostedRoomControlError("control credential changed more than once")
        return changed.rowcount


def _peer_link_from_row(row: sqlite3.Row) -> StoredPeerRoomControl:
    room_id = _identifier(row["room_id"], label="room_id")
    member_id = _identifier(row["member_id"], label="member_id")
    authority_gateway_id = _identifier(
        row["authority_gateway_id"], label="authority_gateway_id"
    )
    authority_epoch = _authority_epoch(row["authority_epoch"])
    room_name = _room_name(row["room_name"])
    member_count = _member_count(row["member_count"])
    home_url, transport_security = _normalize_home_url(row["home_url"])
    if row["transport_security"] != transport_security:
        raise HostedRoomControlError(
            "stored control endpoint classification is invalid"
        )
    status = str(row["status"] or "")
    if status not in _PEER_STATUSES:
        raise HostedRoomControlError("stored control status is invalid")
    created_at = _timestamp(row["created_at"], label="created_at")
    updated_at = _timestamp(row["updated_at"], label="updated_at")
    expires_at = _timestamp(row["expires_at"], label="expires_at")
    revoked_at = row["revoked_at"]
    if revoked_at is not None:
        revoked_at = _timestamp(revoked_at, label="revoked_at")
    control_token = _control_token(row["control_token"])
    return StoredPeerRoomControl(
        room_id=room_id,
        member_id=member_id,
        home_url=home_url,
        transport_security=transport_security,
        authority_gateway_id=authority_gateway_id,
        authority_epoch=authority_epoch,
        room_name=room_name,
        member_count=member_count,
        control_token=control_token,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        expires_at=expires_at,
        revoked_at=revoked_at,
    )


def save_peer_control_link(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    home_url: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    room_name: Any,
    member_count: Any,
    control_token: Any,
    expires_at: Any,
    allow_rotation: bool = False,
    now: float | None = None,
) -> PeerRoomControlSave:
    """Persist a private peer link, rejecting any immutable identity drift."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    home_url, transport_security = _normalize_home_url(home_url)
    authority_gateway_id = _identifier(
        authority_gateway_id, label="authority_gateway_id"
    )
    authority_epoch = _authority_epoch(authority_epoch)
    room_name = _room_name(room_name)
    member_count = _member_count(member_count)
    control_token = _control_token(control_token)
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    expires_at = _timestamp(expires_at, label="expires_at")
    if expires_at <= timestamp:
        raise HostedRoomControlError("control link expiry must be in the future")

    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            """SELECT * FROM hosted_room_peer_controls
                WHERE room_id=? AND member_id=?""",
            (room_id, member_id),
        ).fetchone()
        if existing is not None and str(existing["status"]) in {"expired", "revoked"}:
            conn.execute(
                "DELETE FROM hosted_room_peer_controls WHERE room_id=? AND member_id=?",
                (room_id, member_id),
            )
            existing = None
        if existing is not None:
            try:
                stored = _peer_link_from_row(existing)
            except Exception as exc:
                conn.execute(
                    """UPDATE hosted_room_peer_controls
                          SET status='quarantined', quarantine_reason='invalid_stored_link',
                              updated_at=?
                        WHERE room_id=? AND member_id=?""",
                    (timestamp, room_id, member_id),
                )
                raise HostedRoomControlConflictError(
                    "stored control link is quarantined"
                ) from exc
            same_token = hmac.compare_digest(stored.control_token, control_token)
            if not (
                stored.home_url == home_url
                and stored.authority_gateway_id == authority_gateway_id
                and stored.authority_epoch == authority_epoch
            ):
                raise HostedRoomControlConflictError(
                    "control link conflicts with stored authority"
                )
            rotating = not same_token or stored.expires_at != expires_at
            if rotating and allow_rotation is not True:
                raise HostedRoomControlConflictError(
                    "control link conflicts with stored authority"
                )
            if (
                rotating
                or stored.room_name != room_name
                or stored.member_count != member_count
            ):
                conn.execute(
                    """UPDATE hosted_room_peer_controls
                          SET room_name=?, member_count=?, control_token=?,
                              expires_at=?, status='active', revoked_at=NULL,
                              quarantine_reason=NULL, updated_at=?
                        WHERE room_id=? AND member_id=?""",
                    (
                        room_name,
                        member_count,
                        control_token,
                        expires_at,
                        timestamp,
                        room_id,
                        member_id,
                    ),
                )
                stored = _peer_link_from_row(
                    conn.execute(
                        """SELECT * FROM hosted_room_peer_controls
                            WHERE room_id=? AND member_id=?""",
                        (room_id, member_id),
                    ).fetchone()
                )
            return PeerRoomControlSave(link=stored, idempotent=not rotating)

        count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_peer_controls WHERE status='active'"
        ).fetchone()[0]
        if int(count) >= MAX_PEER_LINKS:
            raise HostedRoomControlError("stored control link limit reached")
        conn.execute(
            """INSERT INTO hosted_room_peer_controls(
                   room_id, member_id, room_name, member_count,
                   home_url, transport_security,
                   authority_gateway_id, authority_epoch, control_token,
                   status, created_at, updated_at, expires_at, revoked_at,
                   quarantine_reason
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL, NULL)""",
            (
                room_id,
                member_id,
                room_name,
                member_count,
                home_url,
                transport_security,
                authority_gateway_id,
                authority_epoch,
                control_token,
                timestamp,
                timestamp,
                expires_at,
            ),
        )
        stored = _peer_link_from_row(
            conn.execute(
                """SELECT * FROM hosted_room_peer_controls
                    WHERE room_id=? AND member_id=?""",
                (room_id, member_id),
            ).fetchone()
        )
    return PeerRoomControlSave(link=stored, idempotent=False)


def load_peer_control_links(
    db_path: Path | str,
    *,
    limit: int = MAX_LOAD_LINKS,
    include_inactive: bool = False,
    now: float | None = None,
) -> PeerRoomControlLoad:
    """Load a bounded page and quarantine malformed private records."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_LOAD_LINKS
    ):
        raise HostedRoomControlError(f"limit must be between 1 and {MAX_LOAD_LINKS}")
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    links: list[StoredPeerRoomControl] = []
    quarantined = 0
    with _transaction(db_path, immediate=True) as conn:
        rows = conn.execute(
            """SELECT rowid, * FROM hosted_room_peer_controls
                ORDER BY updated_at DESC, room_id ASC, member_id ASC
                LIMIT ?""",
            (limit + 1,),
        ).fetchall()
        truncated = len(rows) > limit
        for row in rows[:limit]:
            if row["status"] == "quarantined":
                continue
            try:
                link = _peer_link_from_row(row)
            except Exception:
                conn.execute(
                    """UPDATE hosted_room_peer_controls
                          SET status='quarantined', quarantine_reason='invalid_stored_link',
                              updated_at=? WHERE rowid=?""",
                    (timestamp, row["rowid"]),
                )
                quarantined += 1
                continue
            if link.status == "active" and link.expires_at <= timestamp:
                conn.execute(
                    """UPDATE hosted_room_peer_controls
                          SET status='expired', updated_at=? WHERE rowid=?""",
                    (timestamp, row["rowid"]),
                )
                link = StoredPeerRoomControl(**{
                    **link.__dict__,
                    "status": "expired",
                    "updated_at": timestamp,
                })
            if include_inactive or link.status == "active":
                links.append(link)
    return PeerRoomControlLoad(
        links=tuple(links), quarantined=quarantined, truncated=truncated
    )


def revoke_peer_control_links(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any | None = None,
    now: float | None = None,
) -> int:
    """Idempotently revoke one peer link or every link for a room."""

    clauses = ["room_id=?", "status NOT IN ('revoked', 'quarantined')"]
    params: list[Any] = [_identifier(room_id, label="room_id")]
    if member_id is not None:
        clauses.append("member_id=?")
        params.append(_identifier(member_id, label="member_id"))
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            f"""UPDATE hosted_room_peer_controls
                    SET status='revoked', updated_at=?, revoked_at=?
                  WHERE {" AND ".join(clauses)}""",
            (timestamp, timestamp, *params),
        )
        return cursor.rowcount


def delete_peer_control_links(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any | None = None,
) -> int:
    """Erase peer-side bearer material after reciprocal revocation."""

    clauses = ["room_id=?"]
    params: list[Any] = [_identifier(room_id, label="room_id")]
    if member_id is not None:
        clauses.append("member_id=?")
        params.append(_identifier(member_id, label="member_id"))
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            f"DELETE FROM hosted_room_peer_controls WHERE {' AND '.join(clauses)}",
            params,
        )
        return cursor.rowcount


def update_peer_control_metadata(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    room_name: Any,
    member_count: Any,
    now: float | None = None,
) -> bool:
    """Refresh presentation-only metadata after an authenticated home read."""

    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        changed = conn.execute(
            """UPDATE hosted_room_peer_controls
                  SET room_name=?, member_count=?, updated_at=?
                WHERE room_id=? AND member_id=? AND authority_gateway_id=?
                  AND authority_epoch=? AND status='active'""",
            (
                _room_name(room_name),
                _member_count(member_count),
                timestamp,
                _identifier(room_id, label="room_id"),
                _identifier(member_id, label="member_id"),
                _identifier(authority_gateway_id, label="authority_gateway_id"),
                _authority_epoch(authority_epoch),
            ),
        )
        if changed.rowcount not in {0, 1}:
            raise HostedRoomControlError("control metadata changed more than once")
        return changed.rowcount == 1


def peer_reservation_matches(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    target_profile: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    now: float | None = None,
) -> bool:
    """Bind a reciprocal link to the live target-side RoomLink reservation."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    target_profile = _identifier(target_profile, label="target_profile")
    authority_gateway_id = _identifier(
        authority_gateway_id, label="authority_gateway_id"
    )
    authority_epoch = _authority_epoch(authority_epoch)
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path) as conn:
        table = conn.execute(
            """SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='hosted_room_peer_reservations'"""
        ).fetchone()
        if table is None:
            return False
        row = conn.execute(
            """SELECT 1 FROM hosted_room_peer_reservations
                WHERE room_id=? AND member_id=? AND target_profile=?
                  AND authority_gateway_id=? AND authority_epoch=?
                  AND expires_at>? AND revoked_at IS NULL""",
            (
                room_id,
                member_id,
                target_profile,
                authority_gateway_id,
                authority_epoch,
                timestamp,
            ),
        ).fetchone()
    return row is not None


def begin_control_retry(
    db_path: Path | str,
    *,
    command_id: Any,
    room_id: Any,
    member_id: Any,
    task_ids: Any,
    now: float | None = None,
) -> RoomControlCommandPlan:
    """Freeze one remote retry delivery to one bounded task set."""

    command_id = _identifier(command_id, label="command_id")
    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    if not isinstance(task_ids, (list, tuple)) or len(task_ids) > 8:
        raise HostedRoomControlError("retry task_ids must contain at most 8 tasks")
    frozen = tuple(_identifier(value, label="task_id") for value in task_ids)
    if len(set(frozen)) != len(frozen):
        raise HostedRoomControlError("retry task_ids must be unique")
    encoded = json.dumps(frozen, ensure_ascii=True, separators=(",", ":"))
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        if conn.execute(
            """SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='hosted_rooms'"""
        ).fetchone():
            conn.execute(
                """DELETE FROM hosted_room_control_commands
                     WHERE room_id IN (
                         SELECT room_id FROM hosted_rooms
                          WHERE disbanded_at IS NOT NULL
                     )"""
            )
        existing = conn.execute(
            "SELECT * FROM hosted_room_control_commands WHERE command_id=?",
            (command_id,),
        ).fetchone()
        if existing is not None:
            stored_tasks = tuple(
                _identifier(value, label="task_id")
                for value in json.loads(str(existing["task_ids_json"]))
            )
            if (
                str(existing["room_id"]) != room_id
                or str(existing["member_id"]) != member_id
                or str(existing["action"]) != "retry"
                or (frozen and stored_tasks != frozen)
            ):
                raise HostedRoomControlConflictError(
                    "control command conflicts with its durable retry plan"
                )
            result = (
                json.loads(str(existing["result_json"]))
                if existing["state"] == "completed" and existing["result_json"]
                else None
            )
            return RoomControlCommandPlan(
                task_ids=stored_tasks,
                result=result,
                idempotent=True,
            )
        conn.execute(
            """DELETE FROM hosted_room_control_commands
                 WHERE command_id IN (
                     SELECT command_id FROM hosted_room_control_commands
                      WHERE state='completed' AND updated_at<?
                      ORDER BY updated_at, command_id
                      LIMIT ?
                 )""",
            (timestamp - CONTROL_COMMAND_RETENTION_SECONDS, MAX_CONTROL_COMMANDS),
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_control_commands"
        ).fetchone()[0]
        if int(count) >= MAX_CONTROL_COMMANDS:
            conn.execute(
                """DELETE FROM hosted_room_control_commands
                     WHERE command_id IN (
                         SELECT command_id FROM hosted_room_control_commands
                          WHERE state='completed'
                          ORDER BY updated_at, command_id
                          LIMIT ?
                     )""",
                (int(count) - MAX_CONTROL_COMMANDS + 1,),
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM hosted_room_control_commands"
            ).fetchone()[0]
        if int(count) >= MAX_CONTROL_COMMANDS:
            raise HostedRoomControlError("stored control command limit reached")
        if not frozen:
            raise HostedRoomControlError("This Group Chat has no failed work to retry.")
        conn.execute(
            """INSERT INTO hosted_room_control_commands (
                   command_id, room_id, member_id, action, task_ids_json,
                   state, result_json, created_at, updated_at
               ) VALUES (?, ?, ?, 'retry', ?, 'pending', NULL, ?, ?)""",
            (command_id, room_id, member_id, encoded, timestamp, timestamp),
        )
    return RoomControlCommandPlan(task_ids=frozen, result=None, idempotent=False)


def load_pending_control_retries(
    db_path: Path | str,
    *,
    room_id: Any,
    limit: int = 8,
) -> tuple[PendingRoomControlRetry, ...]:
    """Load a bounded retry queue for the process that owns the room lease."""

    room_id = _identifier(room_id, label="room_id")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise HostedRoomControlError("pending retry limit must be between 1 and 64")
    pending: list[PendingRoomControlRetry] = []
    timestamp = time.time()
    invalid_result = json.dumps(
        {"action": "retry", "error": "invalid_stored_plan", "retried": 0},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    with _transaction(db_path, immediate=True) as conn:
        rows = conn.execute(
            """SELECT command_id, room_id, member_id, task_ids_json
                 FROM hosted_room_control_commands
                WHERE room_id=? AND action='retry' AND state='pending'
                ORDER BY updated_at, created_at, command_id
                LIMIT ?""",
            (room_id, limit),
        ).fetchall()
        for row in rows:
            try:
                raw_task_ids = json.loads(str(row["task_ids_json"]))
                if not isinstance(raw_task_ids, list):
                    raise HostedRoomControlError("stored retry task_ids are invalid")
                task_ids = tuple(
                    _identifier(value, label="task_id") for value in raw_task_ids
                )
                if (
                    not task_ids
                    or len(task_ids) > 8
                    or len(set(task_ids)) != len(task_ids)
                ):
                    raise HostedRoomControlError("stored retry task_ids are invalid")
                pending.append(
                    PendingRoomControlRetry(
                        command_id=_identifier(row["command_id"], label="command_id"),
                        room_id=_identifier(row["room_id"], label="room_id"),
                        member_id=_identifier(row["member_id"], label="member_id"),
                        task_ids=task_ids,
                    )
                )
            except Exception:
                conn.execute(
                    """UPDATE hosted_room_control_commands
                          SET state='completed', result_json=?, updated_at=?
                        WHERE command_id=? AND state='pending'""",
                    (invalid_result, timestamp, row["command_id"]),
                )
    return tuple(pending)


def defer_control_retry(
    db_path: Path | str,
    *,
    command_id: Any,
    now: float | None = None,
) -> bool:
    """Rotate a still-pending command behind newer work after a retry failure."""

    command_id = _identifier(command_id, label="command_id")
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        changed = conn.execute(
            """UPDATE hosted_room_control_commands
                  SET updated_at=?
                WHERE command_id=? AND action='retry' AND state='pending'""",
            (timestamp, command_id),
        )
    return changed.rowcount == 1


def complete_control_retry(
    db_path: Path | str,
    *,
    command_id: Any,
    result: Mapping[str, Any],
    lease: Any | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Commit one remote retry result idempotently."""

    command_id = _identifier(command_id, label="command_id")
    encoded = json.dumps(
        dict(result), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise HostedRoomControlError("control command result is too large")
    timestamp = _timestamp(time.time() if now is None else now, label="now")
    with _transaction(db_path, immediate=True) as conn:
        if lease is not None:
            from gateway import hosted_room_driver

            hosted_room_driver.require_active_lease_in_transaction(
                conn,
                lease,
                now=timestamp,
            )
        row = conn.execute(
            "SELECT state, result_json FROM hosted_room_control_commands WHERE command_id=?",
            (command_id,),
        ).fetchone()
        if row is None:
            raise HostedRoomControlError("control retry plan is missing")
        if row["state"] == "completed":
            if str(row["result_json"] or "") != encoded:
                raise HostedRoomControlConflictError(
                    "control retry result changed after completion"
                )
            return json.loads(encoded)
        changed = conn.execute(
            """UPDATE hosted_room_control_commands
                  SET state='completed', result_json=?, updated_at=?
                WHERE command_id=? AND state='pending'""",
            (encoded, timestamp, command_id),
        )
        if changed.rowcount != 1:
            raise HostedRoomControlConflictError(
                "control retry completion raced another result"
            )
    return json.loads(encoded)
