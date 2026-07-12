"""Fail-closed core for a privileged, token-owning Discord edge process.

The core accepts only an already parsed :class:`DiscordEdgeRequest`.  It has
no socket parser, Discord SDK, token loader, URL, HTTP method, or free-form
dispatcher.  A future Unix service may host it and inject a token-owning
transport plus a live Discord permission prover.

The durable journal is deliberately committed to ``dispatching`` *before* a
mutation is attempted.  A crash from that point onward is an uncertain
outcome, so the same idempotency key is never sent again automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.discord_edge_protocol import (
    DiscordEdgeCapability,
    DiscordEdgeErrorCode,
    DiscordEdgeOperation,
    DiscordEdgeProtocolError,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeReconciliationQuery,
    DiscordEdgeRequest,
    DiscordEdgeThreadReadback,
    DiscordPublicTarget,
    DiscordPublicTargetType,
    SignedDiscordEdgeEnvelope,
    canonical_json_bytes,
    parse_request_for_reconciliation,
    sign_receipt,
    verify_receipt,
    verify_request_capability,
    verify_request_capability_for_reconciliation,
)

_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_SCHEMA_VERSION = 2
_LEGACY_JOURNAL_SCHEMA_VERSION = 1
_JOURNAL_MARKER_PREFIX = "discord-edge-journal.v1 "
_MAX_JOURNAL_MARKER_BYTES = 128
_MAX_PROOF_FUTURE_SKEW_MS = 1_000
_RECONCILIATION_PROOF_DEADLINE_MS = 30_000
_UNCERTAIN_RECEIPT_NAMESPACE = uuid.UUID("802db0b2-a512-4c20-b97c-e23c42cd0665")


def _effective_uid() -> int:
    """Return the POSIX effective UID without breaking imports on Windows."""

    getter = getattr(os, "geteuid", None)
    if getter is None:
        raise OSError("Discord edge journal ownership requires POSIX effective UID")
    return int(getter())


class DiscordEdgeJournalState(StrEnum):
    PREPARED = "prepared"
    DISPATCHING = "dispatching"
    VERIFIED = "verified"
    BLOCKED = "blocked"


class DiscordEdgeBlockerCode(StrEnum):
    TARGET_PROOF_UNAVAILABLE = "target_proof_unavailable"
    TARGET_PROOF_INVALID = "target_proof_invalid"
    TARGET_PROOF_MISMATCH = "target_proof_mismatch"
    TARGET_PROOF_STALE = "target_proof_stale"
    TARGET_NOT_PUBLIC = "target_not_public"
    BOT_CANNOT_VIEW = "bot_cannot_view"
    PERMISSION_REVOKED = "permission_revoked"
    REQUEST_DEADLINE_EXPIRED = "request_deadline_expired_before_dispatch"
    CAPABILITY_EXPIRED = "capability_expired_before_dispatch"
    DISPATCH_OUTCOME_UNCERTAIN = "dispatch_outcome_uncertain"
    DISPATCH_RESULT_INVALID = "dispatch_result_invalid"
    DISPATCH_BINDING_MISMATCH = "dispatch_binding_mismatch"
    DISPATCH_BOT_MISMATCH = "dispatch_bot_mismatch"
    READBACK_PENDING = "readback_pending"
    READBACK_UNAVAILABLE = "readback_unavailable"
    READBACK_INVALID = "readback_invalid"
    READBACK_BINDING_MISMATCH = "readback_binding_mismatch"
    READBACK_AUTHOR_MISMATCH = "readback_author_mismatch"
    READBACK_CONTENT_MISMATCH = "readback_content_mismatch"
    READBACK_REPLY_MISMATCH = "readback_reply_mismatch"
    READBACK_THREAD_TARGET_MISMATCH = "readback_thread_target_mismatch"
    READBACK_THREAD_NAME_MISMATCH = "readback_thread_name_mismatch"
    READBACK_THREAD_ARCHIVE_MISMATCH = "readback_thread_archive_mismatch"


class DiscordEdgeRuntimeErrorCode(StrEnum):
    REQUEST_DEADLINE_EXPIRED = "request_deadline_expired"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    JOURNAL_CORRUPT = "journal_corrupt"
    JOURNAL_NOT_INITIALIZED = "journal_not_initialized"
    RECONCILIATION_NOT_AVAILABLE = "reconciliation_not_available"


class DiscordEdgeRuntimeError(RuntimeError):
    """Stable, secret-free runtime failure."""

    def __init__(self, code: DiscordEdgeRuntimeErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _require_snowflake(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _SNOWFLAKE_RE.fullmatch(value)
        or int(value) == 0
    ):
        raise ValueError(f"{label} must be a non-zero Discord snowflake")
    return value


def _require_positive_unix_ms(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_typed_binding(
    operation: object,
    target: object,
) -> tuple[DiscordEdgeOperation, DiscordPublicTarget]:
    if not isinstance(operation, DiscordEdgeOperation):
        raise ValueError("operation must be a DiscordEdgeOperation")
    if not isinstance(target, DiscordPublicTarget):
        raise ValueError("target must be a DiscordPublicTarget")
    return operation, target


@dataclass(frozen=True)
class DiscordLivePublicTargetProof:
    """Fresh mechanical proof derived from live Discord permissions."""

    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    bot_user_id: str
    observed_at_unix_ms: int
    publicly_viewable: bool
    bot_can_view: bool
    bot_has_required_permission: bool

    def __post_init__(self) -> None:
        _require_typed_binding(self.operation, self.target)
        object.__setattr__(
            self,
            "bot_user_id",
            _require_snowflake(self.bot_user_id, "bot_user_id"),
        )
        object.__setattr__(
            self,
            "observed_at_unix_ms",
            _require_positive_unix_ms(
                self.observed_at_unix_ms,
                "observed_at_unix_ms",
            ),
        )
        for name in (
            "publicly_viewable",
            "bot_can_view",
            "bot_has_required_permission",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class DiscordMutationAccepted:
    """Typed evidence that one fixed transport operation was accepted."""

    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    discord_object_id: str
    bot_user_id: str

    def __post_init__(self) -> None:
        _require_typed_binding(self.operation, self.target)
        object.__setattr__(
            self,
            "discord_object_id",
            _require_snowflake(self.discord_object_id, "discord_object_id"),
        )
        object.__setattr__(
            self,
            "bot_user_id",
            _require_snowflake(self.bot_user_id, "bot_user_id"),
        )


@dataclass(frozen=True)
class DiscordMutationReadback:
    """Exact live readback for a mutation returned by the transport."""

    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    discord_object_id: str
    author_user_id: str
    content: str
    reply_to_message_id: str | None = None
    thread: DiscordEdgeThreadReadback | None = None

    def __post_init__(self) -> None:
        _require_typed_binding(self.operation, self.target)
        object.__setattr__(
            self,
            "discord_object_id",
            _require_snowflake(self.discord_object_id, "discord_object_id"),
        )
        object.__setattr__(
            self,
            "author_user_id",
            _require_snowflake(self.author_user_id, "author_user_id"),
        )
        if not isinstance(self.content, str):
            raise ValueError("readback content must be a string")
        if self.reply_to_message_id is not None:
            object.__setattr__(
                self,
                "reply_to_message_id",
                _require_snowflake(
                    self.reply_to_message_id,
                    "reply_to_message_id",
                ),
            )
        if self.operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE:
            if not isinstance(self.thread, DiscordEdgeThreadReadback):
                raise ValueError("thread creation readback requires typed thread evidence")
        elif self.thread is not None:
            raise ValueError("message readback cannot carry thread evidence")


class DiscordPublicTargetProver(Protocol):
    """Operation-specific live proof surface; there is no generic action."""

    def prove_public_message_send(
        self,
        target: DiscordPublicTarget,
        *,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof: ...

    def prove_public_message_edit(
        self,
        target: DiscordPublicTarget,
        *,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof: ...

    def prove_public_thread_create(
        self,
        target: DiscordPublicTarget,
        *,
        has_initial_message: bool,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof: ...

    def prove_public_readback(
        self,
        operation: DiscordEdgeOperation,
        target: DiscordPublicTarget,
        *,
        require_message_history: bool,
        deadline_unix_ms: int,
        now_unix_ms: int,
    ) -> DiscordLivePublicTargetProof: ...


class DiscordEdgeTransport(Protocol):
    """Fixed token-owning transport surface implemented outside this module."""

    def send_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        content: str,
        reply_to_message_id: str | None,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted: ...

    def edit_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        message_id: str,
        content: str,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted: ...

    def create_public_thread(
        self,
        target: DiscordPublicTarget,
        *,
        name: str,
        initial_message: str | None,
        auto_archive_minutes: int | None,
        deadline_unix_ms: int,
    ) -> DiscordMutationAccepted: ...

    def read_public_message(
        self,
        target: DiscordPublicTarget,
        *,
        operation: DiscordEdgeOperation,
        message_id: str,
        expected_reply_to_message_id: str | None,
    ) -> DiscordMutationReadback: ...

    def read_created_public_thread(
        self,
        target: DiscordPublicTarget,
        *,
        thread_id: str,
        expected_content: str,
    ) -> DiscordMutationReadback: ...


@dataclass(frozen=True)
class DiscordEdgeJournalRecord:
    idempotency_key: str
    request_envelope_sha256: str
    request_envelope: DiscordEdgeRequest | None
    request_id: str
    capability_id: str
    request_sha256: str
    content_sha256: str
    state: DiscordEdgeJournalState
    receipt: SignedDiscordEdgeEnvelope | None
    blocker_code: DiscordEdgeBlockerCode | None
    created_at_unix_ms: int
    updated_at_unix_ms: int


@dataclass(frozen=True)
class DiscordEdgeJournalPrepareResult:
    record: DiscordEdgeJournalRecord
    created: bool


@dataclass(frozen=True)
class DiscordEdgeExecutionResult:
    state: DiscordEdgeJournalState
    receipt: SignedDiscordEdgeEnvelope | None
    blocker_code: DiscordEdgeBlockerCode | None
    replayed: bool


@dataclass(frozen=True)
class DiscordEdgeReconciliationResult:
    request: DiscordEdgeRequest
    execution: DiscordEdgeExecutionResult


def _request_envelope_sha256(request: DiscordEdgeRequest) -> str:
    return hashlib.sha256(canonical_json_bytes(request.to_message())).hexdigest()


class DurableDiscordEdgeJournal:
    """SQLite one-use journal with a commit-before-mutation boundary."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 5_000,
        bootstrap: bool = False,
    ) -> None:
        raw_path = Path(path)
        if not raw_path.is_absolute():
            raise ValueError("Discord edge journal path must be absolute")
        journal_path = Path(os.path.normpath(os.fspath(raw_path)))
        if raw_path != journal_path:
            raise ValueError("Discord edge journal path must be normalized")
        if not 1 <= busy_timeout_ms <= 30_000:
            raise ValueError("busy_timeout_ms is outside its allowed range")
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a boolean")
        try:
            resolved_parent = journal_path.parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Discord edge journal parent must already exist") from exc
        if resolved_parent != journal_path.parent:
            raise ValueError("Discord edge journal parent must be canonical and symlink-free")
        parent_stat = os.stat(resolved_parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("Discord edge journal parent must be a directory")
        if parent_stat.st_uid != _effective_uid():
            raise PermissionError("Discord edge journal parent must be owned by the service")
        if stat.S_IMODE(parent_stat.st_mode) != 0o700:
            raise PermissionError(
                "Discord edge journal parent must have exact mode 0700"
            )
        self.path = journal_path
        self.marker_path = Path(f"{journal_path}.initialized")
        self.busy_timeout_ms = busy_timeout_ms
        self._database_identity: tuple[int, int]
        self._marker_identity: tuple[int, int]
        self._marker_id: str
        if bootstrap:
            self._bootstrap_new()
        else:
            self._open_existing()

    @classmethod
    def bootstrap(
        cls,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 5_000,
    ) -> "DurableDiscordEdgeJournal":
        return cls(path, busy_timeout_ms=busy_timeout_ms, bootstrap=True)

    def _secure_create_file(self, path: Path) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        os.close(fd)

    @staticmethod
    def _assert_secure_file(path: Path, *, label: str) -> os.stat_result:
        try:
            file_stat = os.lstat(path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Discord edge {label} disappeared") from exc
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise PermissionError(f"Discord edge {label} must be one regular file")
        if file_stat.st_uid != _effective_uid():
            raise PermissionError(f"Discord edge {label} has the wrong owner")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise PermissionError(
                f"Discord edge {label} must have exact mode 0600"
            )
        return file_stat

    def _assert_secure_database(self) -> None:
        try:
            database_stat = self._assert_secure_file(
                self.path,
                label="journal database",
            )
        except RuntimeError as exc:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                "Discord edge journal database is missing",
            ) from exc
        if (database_stat.st_dev, database_stat.st_ino) != self._database_identity:
            raise RuntimeError("Discord edge journal database identity changed")

    @staticmethod
    def _write_all(fd: int, value: bytes) -> None:
        offset = 0
        while offset < len(value):
            written = os.write(fd, value[offset:])
            if written <= 0:
                raise OSError("short write while creating Discord edge marker")
            offset += written

    def _fsync_file(self, path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_parent(self) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path.parent, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _marker_bytes(marker_id: str) -> bytes:
        return f"{_JOURNAL_MARKER_PREFIX}{marker_id}\n".encode("ascii")

    def _read_marker_file(self) -> str:
        marker_stat = self._assert_secure_file(
            self.marker_path,
            label="journal initialization marker",
        )
        if (marker_stat.st_dev, marker_stat.st_ino) != self._marker_identity:
            raise RuntimeError("Discord edge journal marker identity changed")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.marker_path, flags)
        try:
            opened_stat = os.fstat(fd)
            if (opened_stat.st_dev, opened_stat.st_ino) != self._marker_identity:
                raise RuntimeError("Discord edge journal marker changed while opening")
            value = os.read(fd, _MAX_JOURNAL_MARKER_BYTES + 1)
        finally:
            os.close(fd)
        if len(value) > _MAX_JOURNAL_MARKER_BYTES:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker is oversized",
            )
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker encoding is invalid",
            ) from exc
        if not text.startswith(_JOURNAL_MARKER_PREFIX) or not text.endswith("\n"):
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker shape is invalid",
            )
        raw_marker_id = text[len(_JOURNAL_MARKER_PREFIX) : -1]
        try:
            parsed = uuid.UUID(raw_marker_id)
        except ValueError as exc:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker identifier is invalid",
            ) from exc
        if str(parsed) != raw_marker_id:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker identifier is not canonical",
            )
        return raw_marker_id

    def _assert_secure_marker(self) -> None:
        try:
            marker_id = self._read_marker_file()
        except RuntimeError as exc:
            if not self.marker_path.exists() and not self.marker_path.is_symlink():
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                    "Discord edge journal initialization marker is missing",
                ) from exc
            raise
        if marker_id != self._marker_id:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal marker value changed",
            )

    def _create_marker(self, marker_id: str) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.marker_path, flags, 0o600)
        try:
            self._write_all(fd, self._marker_bytes(marker_id))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _bootstrap_new(self) -> None:
        if (
            self.path.exists()
            or self.path.is_symlink()
            or self.marker_path.exists()
            or self.marker_path.is_symlink()
        ):
            raise FileExistsError(
                "refusing to bootstrap over an existing Discord edge journal path"
            )
        self._secure_create_file(self.path)
        database_stat = self._assert_secure_file(self.path, label="journal database")
        self._database_identity = (database_stat.st_dev, database_stat.st_ino)
        self._assert_secure_companions()
        marker_id = str(uuid.uuid4())
        self._initialize_new(marker_id)
        self._fsync_file(self.path)
        self._create_marker(marker_id)
        marker_stat = self._assert_secure_file(
            self.marker_path,
            label="journal initialization marker",
        )
        self._marker_identity = (marker_stat.st_dev, marker_stat.st_ino)
        self._marker_id = marker_id
        self._fsync_parent()
        self._validate_existing()

    def _open_existing(self) -> None:
        if not self.path.exists() and not self.path.is_symlink():
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                "Discord edge journal database does not exist; explicit bootstrap is required",
            )
        database_stat = self._assert_secure_file(self.path, label="journal database")
        self._database_identity = (database_stat.st_dev, database_stat.st_ino)
        self._assert_secure_companions()
        if not self.marker_path.exists() and not self.marker_path.is_symlink():
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                "Discord edge journal marker does not exist; explicit bootstrap is required",
            )
        marker_stat = self._assert_secure_file(
            self.marker_path,
            label="journal initialization marker",
        )
        self._marker_identity = (marker_stat.st_dev, marker_stat.st_ino)
        self._marker_id = self._read_marker_file()
        self._migrate_existing_if_needed()
        self._validate_existing()

    def _assert_secure_companions(self) -> None:
        for suffix in ("-wal", "-shm"):
            companion = Path(f"{self.path}{suffix}")
            if companion.exists() or companion.is_symlink():
                self._assert_secure_file(
                    companion,
                    label=f"journal companion {suffix}",
                )

    def _connect(self, *, require_marker: bool = True) -> sqlite3.Connection:
        self._assert_secure_database()
        self._assert_secure_companions()
        if require_marker:
            self._assert_secure_marker()
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA trusted_schema=OFF")
            self._assert_secure_database()
            self._assert_secure_companions()
            if require_marker:
                self._assert_secure_marker()
        except BaseException:
            conn.close()
            raise
        return conn

    def _initialize_new(self, marker_id: str) -> None:
        conn = self._connect(require_marker=False)
        try:
            mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise RuntimeError("Discord edge journal requires SQLite WAL mode")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            existing_tables = {
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                     WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """
                ).fetchall()
            }
            if version != 0 or existing_tables:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                    "bootstrap target is not a fresh empty journal",
                )
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE discord_edge_journal_meta_v1 (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    marker_id TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL CHECK (schema_version = 2)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE discord_edge_idempotency_v1 (
                    idempotency_key TEXT PRIMARY KEY,
                    request_envelope_sha256 TEXT NOT NULL,
                    request_envelope_json TEXT,
                    request_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('prepared', 'dispatching', 'verified', 'blocked')
                    ),
                    receipt_json TEXT,
                    blocker_code TEXT,
                    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms > 0),
                    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms > 0),
                    CHECK (length(request_envelope_sha256) = 64),
                    CHECK (
                        request_envelope_json IS NULL
                        OR length(request_envelope_json) > 0
                    ),
                    CHECK (length(request_sha256) = 64),
                    CHECK (length(content_sha256) = 64),
                    CHECK (
                        (state = 'prepared' AND receipt_json IS NULL
                            AND blocker_code IS NULL)
                        OR state = 'dispatching'
                        OR (state = 'verified' AND receipt_json IS NOT NULL
                            AND blocker_code IS NULL)
                        OR (state = 'blocked' AND receipt_json IS NOT NULL
                            AND blocker_code IS NOT NULL)
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE discord_edge_receipt_history_v1 (
                    idempotency_key TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence > 0),
                    receipt_json TEXT NOT NULL,
                    recorded_at_unix_ms INTEGER NOT NULL
                        CHECK (recorded_at_unix_ms > 0),
                    PRIMARY KEY (idempotency_key, sequence),
                    FOREIGN KEY (idempotency_key)
                        REFERENCES discord_edge_idempotency_v1(idempotency_key)
                        ON DELETE RESTRICT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO discord_edge_journal_meta_v1 (
                    singleton, marker_id, schema_version
                ) VALUES (1, ?, ?)
                """,
                (marker_id, _JOURNAL_SCHEMA_VERSION),
            )
            conn.execute(f"PRAGMA user_version={_JOURNAL_SCHEMA_VERSION}")
            conn.execute("COMMIT")
            checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0:
                raise RuntimeError("Discord edge journal bootstrap checkpoint failed")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()
        self._assert_secure_database()
        self._assert_secure_companions()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
        return frozenset(str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})"))

    @classmethod
    def _expected_table_columns(
        cls,
        *,
        include_request_envelope: bool,
    ) -> dict[str, frozenset[str]]:
        idempotency_columns = {
            "idempotency_key",
            "request_envelope_sha256",
            "request_id",
            "capability_id",
            "request_sha256",
            "content_sha256",
            "state",
            "receipt_json",
            "blocker_code",
            "created_at_unix_ms",
            "updated_at_unix_ms",
        }
        if include_request_envelope:
            idempotency_columns.add("request_envelope_json")
        return {
            "discord_edge_journal_meta_v1": frozenset(
                {"singleton", "marker_id", "schema_version"}
            ),
            "discord_edge_idempotency_v1": frozenset(idempotency_columns),
            "discord_edge_receipt_history_v1": frozenset(
                {
                    "idempotency_key",
                    "sequence",
                    "receipt_json",
                    "recorded_at_unix_ms",
                }
            ),
        }

    @staticmethod
    def _user_tables(conn: sqlite3.Connection) -> frozenset[str]:
        return frozenset(
            str(row[0])
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        )

    def _migrate_existing_if_needed(self) -> None:
        """Atomically add durable request envelopes to an exact v1 journal.

        Existing rows intentionally retain ``NULL`` envelopes.  They remain
        usable only when a caller still possesses the original signed request;
        the new lookup-only reconciliation API never guesses or backfills it.
        """

        conn = self._connect()
        try:
            mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == _JOURNAL_SCHEMA_VERSION:
                return
            if mode != "wal" or version != _LEGACY_JOURNAL_SCHEMA_VERSION:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                    "Discord edge journal lacks a supported initialized schema",
                )
            expected = self._expected_table_columns(
                include_request_envelope=False
            )
            if self._user_tables(conn) != frozenset(expected) or any(
                self._table_columns(conn, table) != columns
                for table, columns in expected.items()
            ):
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "legacy Discord edge journal schema is invalid",
                )
            metadata = [
                (str(row[0]), int(row[1]))
                for row in conn.execute(
                    """
                    SELECT marker_id, schema_version
                      FROM discord_edge_journal_meta_v1
                     WHERE singleton = 1
                    """
                ).fetchall()
            ]
            if metadata != [(self._marker_id, _LEGACY_JOURNAL_SCHEMA_VERSION)]:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "legacy Discord edge journal marker does not match",
                )
            quick_check = conn.execute("PRAGMA quick_check(1)").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "legacy Discord edge journal integrity check failed",
                )

            self._begin(conn)
            current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current_version != _LEGACY_JOURNAL_SCHEMA_VERSION:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "legacy Discord edge journal changed during migration",
                )
            conn.execute(
                """
                ALTER TABLE discord_edge_idempotency_v1
                ADD COLUMN request_envelope_json TEXT
                """
            )
            conn.execute(
                """
                ALTER TABLE discord_edge_journal_meta_v1
                RENAME TO discord_edge_journal_meta_legacy_v1
                """
            )
            conn.execute(
                """
                CREATE TABLE discord_edge_journal_meta_v1 (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    marker_id TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL CHECK (schema_version = 2)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO discord_edge_journal_meta_v1 (
                    singleton, marker_id, schema_version
                ) VALUES (1, ?, ?)
                """,
                (self._marker_id, _JOURNAL_SCHEMA_VERSION),
            )
            conn.execute("DROP TABLE discord_edge_journal_meta_legacy_v1")
            conn.execute(f"PRAGMA user_version={_JOURNAL_SCHEMA_VERSION}")
            self._commit(conn)
            checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0:
                raise RuntimeError("Discord edge journal migration checkpoint failed")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()
        self._assert_secure_database()
        self._assert_secure_companions()
        self._fsync_file(self.path)

    def _validate_existing(self) -> None:
        conn = self._connect()
        try:
            mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if mode != "wal" or version != _JOURNAL_SCHEMA_VERSION:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED,
                    "Discord edge journal lacks its initialized WAL/schema marker",
                )
            expected = self._expected_table_columns(include_request_envelope=True)
            if self._user_tables(conn) != frozenset(expected):
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "Discord edge journal table set is invalid",
                )
            if any(
                self._table_columns(conn, table) != columns
                for table, columns in expected.items()
            ):
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "Discord edge journal column set is invalid",
                )
            metadata = [
                (str(row[0]), int(row[1]))
                for row in conn.execute(
                    """
                    SELECT marker_id, schema_version
                      FROM discord_edge_journal_meta_v1
                     WHERE singleton = 1
                    """
                ).fetchall()
            ]
            if metadata != [(self._marker_id, _JOURNAL_SCHEMA_VERSION)]:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "Discord edge journal durable marker does not match its database",
                )
            quick_check = conn.execute("PRAGMA quick_check(1)").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "Discord edge journal integrity check failed",
                )
        finally:
            conn.close()

    def _begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        self._assert_secure_database()
        self._assert_secure_companions()

    def _commit(self, conn: sqlite3.Connection) -> None:
        conn.execute("COMMIT")
        self._assert_secure_database()
        self._assert_secure_companions()

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @staticmethod
    def _select(
        conn: sqlite3.Connection,
        idempotency_key: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT idempotency_key, request_envelope_sha256, request_id,
                   request_envelope_json, capability_id, request_sha256,
                   content_sha256, state, receipt_json, blocker_code,
                   created_at_unix_ms, updated_at_unix_ms
              FROM discord_edge_idempotency_v1
             WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()

    @staticmethod
    def _record(row: sqlite3.Row) -> DiscordEdgeJournalRecord:
        try:
            state = DiscordEdgeJournalState(row["state"])
            blocker = (
                DiscordEdgeBlockerCode(row["blocker_code"])
                if row["blocker_code"] is not None
                else None
            )
            receipt = None
            if row["receipt_json"] is not None:
                decoded = json.loads(row["receipt_json"])
                receipt = SignedDiscordEdgeEnvelope.from_mapping(
                    decoded,
                    code=DiscordEdgeErrorCode.INVALID_RECEIPT,
                    label="journal receipt",
                )
            request_envelope = None
            serialized_request = row["request_envelope_json"]
            if serialized_request is not None:
                if not isinstance(serialized_request, str) or not serialized_request:
                    raise ValueError("invalid journal request envelope")
                decoded_request = json.loads(serialized_request)
                request_envelope = parse_request_for_reconciliation(decoded_request)
                if (
                    canonical_json_bytes(request_envelope.to_message()).decode("utf-8")
                    != serialized_request
                ):
                    raise ValueError("journal request envelope is not canonical")
            for field in (
                "request_envelope_sha256",
                "request_sha256",
                "content_sha256",
            ):
                if not _SHA256_RE.fullmatch(str(row[field])):
                    raise ValueError("invalid journal digest")
            record = DiscordEdgeJournalRecord(
                idempotency_key=str(row["idempotency_key"]),
                request_envelope_sha256=str(row["request_envelope_sha256"]),
                request_envelope=request_envelope,
                request_id=str(row["request_id"]),
                capability_id=str(row["capability_id"]),
                request_sha256=str(row["request_sha256"]),
                content_sha256=str(row["content_sha256"]),
                state=state,
                receipt=receipt,
                blocker_code=blocker,
                created_at_unix_ms=int(row["created_at_unix_ms"]),
                updated_at_unix_ms=int(row["updated_at_unix_ms"]),
            )
            if request_envelope is not None:
                exact_envelope_binding = (
                    _request_envelope_sha256(request_envelope),
                    request_envelope.request_id,
                    request_envelope.capability.payload.get("capability_id"),
                    request_envelope.intent.idempotency_key,
                    request_envelope.intent.request_sha256,
                    request_envelope.intent.content_sha256,
                )
                exact_record_binding = (
                    record.request_envelope_sha256,
                    record.request_id,
                    record.capability_id,
                    record.idempotency_key,
                    record.request_sha256,
                    record.content_sha256,
                )
                if exact_envelope_binding != exact_record_binding:
                    raise ValueError("journal request envelope binding mismatch")
            return record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal record is invalid",
            ) from exc
        except DiscordEdgeProtocolError as exc:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "Discord edge journal receipt is invalid",
            ) from exc

    @staticmethod
    def _assert_exact_binding(
        record: DiscordEdgeJournalRecord,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
    ) -> None:
        actual = (
            record.request_envelope_sha256,
            record.request_id,
            record.capability_id,
            record.request_sha256,
            record.content_sha256,
        )
        expected = (
            _request_envelope_sha256(request),
            request.request_id,
            capability.capability_id,
            request.intent.request_sha256,
            request.intent.content_sha256,
        )
        if actual != expected:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.IDEMPOTENCY_CONFLICT,
                "idempotency key is already bound to another exact request",
            )

    @staticmethod
    def _can_rebind_prepared_request(
        record: DiscordEdgeJournalRecord,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
    ) -> bool:
        """Allow only a strictly newer writer envelope for the same intent.

        A ``prepared`` row has not crossed the mutation boundary.  Rebinding
        it lets the writer recover an expired request without changing the
        idempotent Discord mutation.  Strictly increasing capability issue
        time is the anti-rollback order: after the CAS, the superseded request
        cannot bind the row again even if its deadline has not yet expired.
        """

        previous = record.request_envelope
        if previous is None or record.state is not DiscordEdgeJournalState.PREPARED:
            return False
        previous_intent = previous.intent
        exact_intent = (
            previous_intent.operation is request.intent.operation
            and previous_intent.target == request.intent.target
            and dict(previous_intent.payload) == dict(request.intent.payload)
            and previous_intent.idempotency_key == request.intent.idempotency_key
            and previous_intent.request_sha256 == request.intent.request_sha256
            and previous_intent.content_sha256 == request.intent.content_sha256
        )
        if not exact_intent:
            return False
        previous_issued_at = previous.capability.payload.get("issued_at_unix_ms")
        if (
            isinstance(previous_issued_at, bool)
            or not isinstance(previous_issued_at, int)
            or previous_issued_at <= 0
        ):
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "prepared request capability issue time is invalid",
            )
        return capability.issued_at_unix_ms > previous_issued_at

    @staticmethod
    def _rebind_prepared_request(
        conn: sqlite3.Connection,
        record: DiscordEdgeJournalRecord,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        now_unix_ms: int,
    ) -> None:
        serialized_request = canonical_json_bytes(request.to_message()).decode("utf-8")
        cursor = conn.execute(
            """
            UPDATE discord_edge_idempotency_v1
               SET request_envelope_sha256 = ?, request_envelope_json = ?,
                   request_id = ?, capability_id = ?, request_sha256 = ?,
                   content_sha256 = ?, updated_at_unix_ms = ?
             WHERE idempotency_key = ? AND state = 'prepared'
                   AND receipt_json IS NULL AND blocker_code IS NULL
                   AND request_envelope_sha256 = ? AND request_id = ?
                   AND capability_id = ? AND request_sha256 = ?
                   AND content_sha256 = ?
            """,
            (
                _request_envelope_sha256(request),
                serialized_request,
                request.request_id,
                capability.capability_id,
                request.intent.request_sha256,
                request.intent.content_sha256,
                now_unix_ms,
                request.intent.idempotency_key,
                record.request_envelope_sha256,
                record.request_id,
                record.capability_id,
                record.request_sha256,
                record.content_sha256,
            ),
        )
        if cursor.rowcount != 1:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.IDEMPOTENCY_CONFLICT,
                "prepared request changed before exact rebind",
            )

    def prepare(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalPrepareResult:
        conn = self._connect()
        try:
            self._begin(conn)
            row = self._select(conn, request.intent.idempotency_key)
            created = row is None
            if row is None:
                conn.execute(
                    """
                    INSERT INTO discord_edge_idempotency_v1 (
                        idempotency_key, request_envelope_sha256,
                        request_envelope_json, request_id, capability_id,
                        request_sha256, content_sha256, state, receipt_json,
                        blocker_code, created_at_unix_ms, updated_at_unix_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', NULL, NULL, ?, ?)
                    """,
                    (
                        request.intent.idempotency_key,
                        _request_envelope_sha256(request),
                        canonical_json_bytes(request.to_message()).decode("utf-8"),
                        request.request_id,
                        capability.capability_id,
                        request.intent.request_sha256,
                        request.intent.content_sha256,
                        now_unix_ms,
                        now_unix_ms,
                    ),
                )
                row = self._select(conn, request.intent.idempotency_key)
            assert row is not None
            record = self._record(row)
            if (
                not created
                and self._can_rebind_prepared_request(record, request, capability)
            ):
                self._rebind_prepared_request(
                    conn,
                    record,
                    request,
                    capability,
                    now_unix_ms=now_unix_ms,
                )
                row = self._select(conn, request.intent.idempotency_key)
                assert row is not None
                record = self._record(row)
            self._assert_exact_binding(record, request, capability)
            self._commit(conn)
            return DiscordEdgeJournalPrepareResult(record, created)
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def claim_dispatching(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        now_unix_ms: int,
    ) -> tuple[DiscordEdgeJournalRecord, bool]:
        conn = self._connect()
        try:
            self._begin(conn)
            row = self._select(conn, request.intent.idempotency_key)
            if row is None:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "prepared journal record disappeared",
                )
            record = self._record(row)
            self._assert_exact_binding(record, request, capability)
            claimed = False
            if record.state is DiscordEdgeJournalState.PREPARED:
                cursor = conn.execute(
                    """
                    UPDATE discord_edge_idempotency_v1
                       SET state = 'dispatching', updated_at_unix_ms = ?
                     WHERE idempotency_key = ? AND state = 'prepared'
                    """,
                    (now_unix_ms, request.intent.idempotency_key),
                )
                claimed = cursor.rowcount == 1
            row = self._select(conn, request.intent.idempotency_key)
            assert row is not None
            record = self._record(row)
            self._commit(conn)
            return record, claimed
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def finalize_blocked(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        receipt: SignedDiscordEdgeEnvelope,
        blocker_code: DiscordEdgeBlockerCode,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        return self._persist_receipt(
            request,
            capability,
            receipt,
            blocker_code,
            target_state=DiscordEdgeJournalState.BLOCKED,
            required_state=DiscordEdgeJournalState.PREPARED,
            now_unix_ms=now_unix_ms,
        )

    def record_accepted_unverified(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        receipt: SignedDiscordEdgeEnvelope,
        blocker_code: DiscordEdgeBlockerCode,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        return self._persist_receipt(
            request,
            capability,
            receipt,
            blocker_code,
            target_state=DiscordEdgeJournalState.DISPATCHING,
            required_state=DiscordEdgeJournalState.DISPATCHING,
            now_unix_ms=now_unix_ms,
        )

    def record_dispatch_uncertain(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        receipt: SignedDiscordEdgeEnvelope,
        blocker_code: DiscordEdgeBlockerCode,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        return self._persist_receipt(
            request,
            capability,
            receipt,
            blocker_code,
            target_state=DiscordEdgeJournalState.DISPATCHING,
            required_state=DiscordEdgeJournalState.DISPATCHING,
            now_unix_ms=now_unix_ms,
        )

    @staticmethod
    def _append_receipt_history(
        conn: sqlite3.Connection,
        *,
        idempotency_key: str,
        receipt_json: str,
        recorded_at_unix_ms: int,
    ) -> None:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1
              FROM discord_edge_receipt_history_v1
             WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if row is None:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "receipt history sequence could not be allocated",
            )
        conn.execute(
            """
            INSERT INTO discord_edge_receipt_history_v1 (
                idempotency_key, sequence, receipt_json, recorded_at_unix_ms
            ) VALUES (?, ?, ?, ?)
            """,
            (
                idempotency_key,
                int(row[0]),
                receipt_json,
                recorded_at_unix_ms,
            ),
        )

    def _persist_receipt(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        receipt: SignedDiscordEdgeEnvelope,
        blocker_code: DiscordEdgeBlockerCode | None,
        *,
        target_state: DiscordEdgeJournalState,
        required_state: DiscordEdgeJournalState,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        serialized = canonical_json_bytes(receipt.to_message()).decode("utf-8")
        conn = self._connect()
        try:
            self._begin(conn)
            row = self._select(conn, request.intent.idempotency_key)
            if row is None:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "journal record disappeared before receipt persistence",
                )
            record = self._record(row)
            self._assert_exact_binding(record, request, capability)
            if record.state is required_state and record.receipt is None:
                cursor = conn.execute(
                    """
                    UPDATE discord_edge_idempotency_v1
                       SET state = ?, receipt_json = ?, blocker_code = ?,
                           updated_at_unix_ms = ?
                     WHERE idempotency_key = ? AND state = ?
                           AND receipt_json IS NULL
                    """,
                    (
                        target_state.value,
                        serialized,
                        blocker_code.value if blocker_code is not None else None,
                        now_unix_ms,
                        request.intent.idempotency_key,
                        required_state.value,
                    ),
                )
                if cursor.rowcount == 1:
                    self._append_receipt_history(
                        conn,
                        idempotency_key=request.intent.idempotency_key,
                        receipt_json=serialized,
                        recorded_at_unix_ms=now_unix_ms,
                    )
            row = self._select(conn, request.intent.idempotency_key)
            assert row is not None
            record = self._record(row)
            self._commit(conn)
            return record
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def upgrade_accepted_unverified(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        expected_receipt: SignedDiscordEdgeEnvelope,
        verified_receipt: SignedDiscordEdgeEnvelope,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        expected_json = canonical_json_bytes(expected_receipt.to_message()).decode("utf-8")
        verified_json = canonical_json_bytes(verified_receipt.to_message()).decode("utf-8")
        conn = self._connect()
        try:
            self._begin(conn)
            row = self._select(conn, request.intent.idempotency_key)
            if row is None:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "journal record disappeared during readback reconciliation",
                )
            record = self._record(row)
            self._assert_exact_binding(record, request, capability)
            if (
                record.state is DiscordEdgeJournalState.DISPATCHING
                and record.receipt is not None
                and canonical_json_bytes(record.receipt.to_message()).decode("utf-8")
                == expected_json
            ):
                cursor = conn.execute(
                    """
                    UPDATE discord_edge_idempotency_v1
                       SET state = 'verified', receipt_json = ?, blocker_code = NULL,
                           updated_at_unix_ms = ?
                     WHERE idempotency_key = ? AND state = 'dispatching'
                           AND receipt_json = ?
                    """,
                    (
                        verified_json,
                        now_unix_ms,
                        request.intent.idempotency_key,
                        expected_json,
                    ),
                )
                if cursor.rowcount == 1:
                    self._append_receipt_history(
                        conn,
                        idempotency_key=request.intent.idempotency_key,
                        receipt_json=verified_json,
                        recorded_at_unix_ms=now_unix_ms,
                    )
            row = self._select(conn, request.intent.idempotency_key)
            assert row is not None
            record = self._record(row)
            self._commit(conn)
            return record
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def replace_accepted_unverified(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        expected_receipt: SignedDiscordEdgeEnvelope,
        replacement_receipt: SignedDiscordEdgeEnvelope,
        blocker_code: DiscordEdgeBlockerCode,
        now_unix_ms: int,
    ) -> DiscordEdgeJournalRecord:
        """Replace one staged accepted receipt with more exact readback evidence."""

        expected_json = canonical_json_bytes(expected_receipt.to_message()).decode(
            "utf-8"
        )
        replacement_json = canonical_json_bytes(
            replacement_receipt.to_message()
        ).decode("utf-8")
        conn = self._connect()
        try:
            self._begin(conn)
            row = self._select(conn, request.intent.idempotency_key)
            if row is None:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "journal record disappeared during accepted evidence replacement",
                )
            record = self._record(row)
            self._assert_exact_binding(record, request, capability)
            if (
                record.state is DiscordEdgeJournalState.DISPATCHING
                and record.receipt is not None
                and canonical_json_bytes(record.receipt.to_message()).decode("utf-8")
                == expected_json
            ):
                cursor = conn.execute(
                    """
                    UPDATE discord_edge_idempotency_v1
                       SET receipt_json = ?, blocker_code = ?,
                           updated_at_unix_ms = ?
                     WHERE idempotency_key = ? AND state = 'dispatching'
                           AND receipt_json = ?
                    """,
                    (
                        replacement_json,
                        blocker_code.value,
                        now_unix_ms,
                        request.intent.idempotency_key,
                        expected_json,
                    ),
                )
                if cursor.rowcount == 1:
                    self._append_receipt_history(
                        conn,
                        idempotency_key=request.intent.idempotency_key,
                        receipt_json=replacement_json,
                        recorded_at_unix_ms=now_unix_ms,
                    )
            row = self._select(conn, request.intent.idempotency_key)
            assert row is not None
            record = self._record(row)
            self._commit(conn)
            return record
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def get(self, idempotency_key: str) -> DiscordEdgeJournalRecord | None:
        conn = self._connect()
        try:
            row = self._select(conn, idempotency_key)
            return self._record(row) if row is not None else None
        finally:
            conn.close()

    def get_for_reconciliation(
        self,
        query: DiscordEdgeReconciliationQuery,
    ) -> DiscordEdgeJournalRecord:
        """Return one exact durable outcome without accepting mutation input."""

        if not isinstance(query, DiscordEdgeReconciliationQuery):
            raise TypeError(
                "reconciliation lookup accepts only DiscordEdgeReconciliationQuery"
            )
        record = self.get(query.idempotency_key)
        if (
            record is None
            or record.request_envelope is None
            or record.state is DiscordEdgeJournalState.PREPARED
            or not query.matches_request(record.request_envelope)
        ):
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE,
                "no exact durable Discord edge outcome is available",
            )
        return record

    def receipt_history(
        self,
        idempotency_key: str,
    ) -> tuple[SignedDiscordEdgeEnvelope, ...]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT receipt_json
                  FROM discord_edge_receipt_history_v1
                 WHERE idempotency_key = ?
                 ORDER BY sequence ASC
                """,
                (idempotency_key,),
            ).fetchall()
            receipts: list[SignedDiscordEdgeEnvelope] = []
            for row in rows:
                try:
                    decoded = json.loads(str(row[0]))
                    receipts.append(
                        SignedDiscordEdgeEnvelope.from_mapping(
                            decoded,
                            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
                            label="journal receipt history",
                        )
                    )
                except (json.JSONDecodeError, DiscordEdgeProtocolError) as exc:
                    raise DiscordEdgeRuntimeError(
                        DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                        "Discord edge receipt history is invalid",
                    ) from exc
            return tuple(receipts)
        finally:
            conn.close()


class DiscordEdgeRuntime:
    """Execute fixed public Discord mutations behind one-use capabilities."""

    def __init__(
        self,
        *,
        writer_public_key: Ed25519PublicKey,
        edge_private_key: Ed25519PrivateKey,
        journal: DurableDiscordEdgeJournal,
        target_prover: DiscordPublicTargetProver,
        transport: DiscordEdgeTransport,
        clock_ms: Callable[[], int] | None = None,
        max_proof_age_ms: int = 5_000,
    ) -> None:
        if not isinstance(writer_public_key, Ed25519PublicKey):
            raise TypeError("writer_public_key must be Ed25519PublicKey")
        if not isinstance(edge_private_key, Ed25519PrivateKey):
            raise TypeError("edge_private_key must be Ed25519PrivateKey")
        if not isinstance(journal, DurableDiscordEdgeJournal):
            raise TypeError("journal must be DurableDiscordEdgeJournal")
        if not 1 <= max_proof_age_ms <= 30_000:
            raise ValueError("max_proof_age_ms is outside its allowed range")
        proof_methods = (
            "prove_public_message_send",
            "prove_public_message_edit",
            "prove_public_thread_create",
            "prove_public_readback",
        )
        transport_methods = (
            "send_public_message",
            "edit_public_message",
            "create_public_thread",
            "read_public_message",
            "read_created_public_thread",
        )
        if any(not callable(getattr(target_prover, name, None)) for name in proof_methods):
            raise TypeError("target_prover lacks the fixed live-proof surface")
        if any(not callable(getattr(transport, name, None)) for name in transport_methods):
            raise TypeError("transport lacks the fixed Discord operation surface")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self.writer_public_key = writer_public_key
        self.edge_private_key = edge_private_key
        self.journal = journal
        self.target_prover = target_prover
        self.transport = transport
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self.max_proof_age_ms = max_proof_age_ms
        self._dispatch_state_lock = threading.RLock()
        self._active_dispatches: set[str] = set()

    @staticmethod
    def _active_dispatch_error() -> DiscordEdgeRuntimeError:
        return DiscordEdgeRuntimeError(
            DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE,
            "matching Discord dispatch is active; reconciliation remains pending",
        )

    def _claim_dispatching_with_active_marker(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        now_unix_ms: int,
    ) -> tuple[DiscordEdgeJournalRecord, bool]:
        """Commit the durable claim and publish its in-process marker atomically."""

        key = request.intent.idempotency_key
        with self._dispatch_state_lock:
            if key in self._active_dispatches:
                raise self._active_dispatch_error()
            self._active_dispatches.add(key)
            try:
                record, claimed = self.journal.claim_dispatching(
                    request,
                    capability,
                    now_unix_ms=now_unix_ms,
                )
            except BaseException:
                self._active_dispatches.discard(key)
                raise
            if not claimed:
                self._active_dispatches.discard(key)
            return record, claimed

    def _clear_active_dispatch(self, idempotency_key: str) -> None:
        with self._dispatch_state_lock:
            self._active_dispatches.discard(idempotency_key)

    def _resolve_unreceipted_dispatch_replay(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
    ) -> DiscordEdgeExecutionResult:
        """Persist uncertainty only when no dispatch is active in this process."""

        key = request.intent.idempotency_key
        with self._dispatch_state_lock:
            current = self.journal.get(key)
            if key in self._active_dispatches:
                raise self._active_dispatch_error()
            if current is None:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "dispatching journal record disappeared",
                )
            if (
                current.state is DiscordEdgeJournalState.DISPATCHING
                and current.receipt is None
            ):
                return self._dispatch_uncertain(
                    request,
                    capability,
                    current.blocker_code
                    or DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN,
                    occurred_at_unix_ms=current.updated_at_unix_ms,
                    replayed=True,
                )
            return self._result_from_record(
                current,
                request,
                capability,
                replayed=True,
                now_unix_ms=max(self.clock_ms(), current.updated_at_unix_ms),
            )

    def execute(self, request: DiscordEdgeRequest) -> DiscordEdgeExecutionResult:
        if not isinstance(request, DiscordEdgeRequest):
            raise TypeError("Discord edge runtime accepts only parsed DiscordEdgeRequest")
        now = self.clock_ms()
        with self._dispatch_state_lock:
            existing = self.journal.get(request.intent.idempotency_key)
        if (
            existing is not None
            and existing.state is not DiscordEdgeJournalState.PREPARED
        ):
            historical_capability = verify_request_capability_for_reconciliation(
                request,
                self.writer_public_key,
            )
            historical = self.journal.prepare(
                request,
                historical_capability,
                now_unix_ms=now,
            ).record
            if (
                historical.state is DiscordEdgeJournalState.DISPATCHING
                and historical.receipt is None
            ):
                return self._resolve_unreceipted_dispatch_replay(
                    request,
                    historical_capability,
                )
            if (
                historical.state is DiscordEdgeJournalState.DISPATCHING
                and historical.receipt is not None
            ):
                historical_receipt = verify_receipt(
                    historical.receipt,
                    self.edge_private_key.public_key(),
                    expected_request=request,
                    expected_capability=historical_capability,
                    now_unix_ms=now,
                )
                if (
                    historical_receipt.outcome
                    is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
                ):
                    return self._reconcile_accepted_unverified_record(
                        request,
                        historical_capability,
                        historical,
                        now_unix_ms=now,
                    )
            return self._result_from_record(
                historical,
                request,
                historical_capability,
                replayed=True,
                now_unix_ms=now,
            )
        if request.deadline_unix_ms <= now:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.REQUEST_DEADLINE_EXPIRED,
                "parsed request deadline expired before edge execution",
            )
        capability = verify_request_capability(
            request,
            self.writer_public_key,
            now_unix_ms=now,
        )
        prepared = self.journal.prepare(request, capability, now_unix_ms=now)
        if prepared.record.state is not DiscordEdgeJournalState.PREPARED:
            if (
                prepared.record.state is DiscordEdgeJournalState.DISPATCHING
                and prepared.record.receipt is None
            ):
                return self._resolve_unreceipted_dispatch_replay(
                    request,
                    capability,
                )
            return self._result_from_record(
                prepared.record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now,
            )

        proof, blocker, failed = self._live_proof(request, now_unix_ms=now)
        if blocker is not None:
            return self._block_before_dispatch(
                request,
                capability,
                blocker,
                failed=failed,
                replayed=not prepared.created,
                now_unix_ms=now,
            )
        assert proof is not None

        dispatch_now = self.clock_ms()
        if request.deadline_unix_ms <= dispatch_now:
            return self._block_before_dispatch(
                request,
                capability,
                DiscordEdgeBlockerCode.REQUEST_DEADLINE_EXPIRED,
                failed=True,
                replayed=not prepared.created,
                now_unix_ms=dispatch_now,
            )
        if capability.expires_at_unix_ms <= dispatch_now:
            return self._block_before_dispatch(
                request,
                capability,
                DiscordEdgeBlockerCode.CAPABILITY_EXPIRED,
                failed=True,
                replayed=not prepared.created,
                now_unix_ms=dispatch_now,
            )
        blocker = self._proof_blocker(request, proof, now_unix_ms=dispatch_now)
        if blocker is not None:
            return self._block_before_dispatch(
                request,
                capability,
                blocker,
                failed=False,
                replayed=not prepared.created,
                now_unix_ms=dispatch_now,
            )

        record, claimed = self._claim_dispatching_with_active_marker(
            request,
            capability,
            now_unix_ms=dispatch_now,
        )
        if not claimed:
            if (
                record.state is DiscordEdgeJournalState.DISPATCHING
                and record.receipt is None
            ):
                return self._resolve_unreceipted_dispatch_replay(
                    request,
                    capability,
                )
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=dispatch_now,
            )

        return self._execute_claimed_dispatch(request, capability, proof, record)

    def _execute_claimed_dispatch(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        proof: DiscordLivePublicTargetProof,
        record: DiscordEdgeJournalRecord,
    ) -> DiscordEdgeExecutionResult:
        try:
            try:
                accepted = self._dispatch(request)
            except Exception:
                return self._dispatch_uncertain(
                    request,
                    capability,
                    DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN,
                    occurred_at_unix_ms=record.updated_at_unix_ms,
                    replayed=False,
                )

            accepted_blocker = self._accepted_blocker(request, proof, accepted)
            if accepted_blocker is not None:
                return self._dispatch_uncertain(
                    request,
                    capability,
                    accepted_blocker,
                    occurred_at_unix_ms=record.updated_at_unix_ms,
                    replayed=False,
                )
            assert isinstance(accepted, DiscordMutationAccepted)

            # Persist signed accepted evidence before the first readback.  A crash
            # from this point can reconcile the exact Discord object without ever
            # dispatching the mutation again.
            staged = self._accepted_unverified(
                request,
                capability,
                accepted,
                DiscordEdgeBlockerCode.READBACK_PENDING,
                replayed=False,
            )
            assert staged.receipt is not None

            try:
                readback = self._readback(request, accepted)
            except Exception:
                return staged
            readback_blocker = self._readback_blocker(
                request,
                proof,
                accepted,
                readback,
            )
            if readback_blocker is not None:
                return self._accepted_unverified(
                    request,
                    capability,
                    accepted,
                    readback_blocker,
                    replayed=False,
                    expected_receipt=staged.receipt,
                )

            occurred = self.clock_ms()
            receipt = sign_receipt(
                self.edge_private_key,
                request,
                capability,
                outcome=DiscordEdgeReceiptOutcome.VERIFIED,
                discord_object_id=accepted.discord_object_id,
                bot_user_id=accepted.bot_user_id,
                adapter_accepted=True,
                readback_verified=True,
                readback_content_sha256=request.intent.content_sha256,
                readback_thread=readback.thread,
                occurred_at_unix_ms=occurred,
            )
            record = self.journal.upgrade_accepted_unverified(
                request,
                capability,
                expected_receipt=staged.receipt,
                verified_receipt=receipt,
                now_unix_ms=occurred,
            )
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=False,
                now_unix_ms=occurred,
            )
        finally:
            self._clear_active_dispatch(request.intent.idempotency_key)

    def reconcile(
        self,
        query: DiscordEdgeReconciliationQuery,
    ) -> DiscordEdgeReconciliationResult:
        """Resolve one journaled outcome without carrying mutation authority.

        This path never calls ``_dispatch``.  It may persist deterministic
        uncertainty for a pre-acceptance crash or perform live, read-only
        proof/readback for an already accepted Discord object.
        """

        if not isinstance(query, DiscordEdgeReconciliationQuery):
            raise TypeError(
                "reconciliation accepts only DiscordEdgeReconciliationQuery"
            )
        with self._dispatch_state_lock:
            record = self.journal.get_for_reconciliation(query)
            if query.idempotency_key in self._active_dispatches:
                raise self._active_dispatch_error()
        request = record.request_envelope
        assert request is not None
        capability = verify_request_capability_for_reconciliation(
            request,
            self.writer_public_key,
        )
        now = self.clock_ms()
        if (
            record.state is DiscordEdgeJournalState.DISPATCHING
            and record.receipt is None
        ):
            execution = self._dispatch_uncertain(
                request,
                capability,
                record.blocker_code
                or DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN,
                occurred_at_unix_ms=record.updated_at_unix_ms,
                replayed=True,
            )
        elif (
            record.state is DiscordEdgeJournalState.DISPATCHING
            and record.receipt is not None
        ):
            receipt = verify_receipt(
                record.receipt,
                self.edge_private_key.public_key(),
                expected_request=request,
                expected_capability=capability,
                now_unix_ms=now,
            )
            if receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED:
                execution = self._reconcile_accepted_unverified_record(
                    request,
                    capability,
                    record,
                    now_unix_ms=now,
                )
            else:
                execution = self._result_from_record(
                    record,
                    request,
                    capability,
                    replayed=True,
                    now_unix_ms=now,
                )
        else:
            execution = self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now,
            )
        return DiscordEdgeReconciliationResult(
            request=request,
            execution=execution,
        )

    def reconcile_accepted_unverified(
        self,
        request: DiscordEdgeRequest,
    ) -> DiscordEdgeExecutionResult:
        """Read back an accepted mutation without ever dispatching it again."""

        if not isinstance(request, DiscordEdgeRequest):
            raise TypeError("reconciliation accepts only parsed DiscordEdgeRequest")
        now = self.clock_ms()
        existing = self.journal.get(request.intent.idempotency_key)
        if existing is None or existing.state is DiscordEdgeJournalState.PREPARED:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE,
                "no durable accepted_unverified outcome exists for reconciliation",
            )
        capability = verify_request_capability_for_reconciliation(
            request,
            self.writer_public_key,
        )
        record = self.journal.prepare(
            request,
            capability,
            now_unix_ms=now,
        ).record
        if record.state is DiscordEdgeJournalState.VERIFIED:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now,
            )
        return self._reconcile_accepted_unverified_record(
            request,
            capability,
            record,
            now_unix_ms=now,
        )

    def _reconcile_accepted_unverified_record(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        record: DiscordEdgeJournalRecord,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeExecutionResult:
        if (
            record.state is not DiscordEdgeJournalState.DISPATCHING
            or record.receipt is None
        ):
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE,
                "journal outcome is not accepted_unverified",
            )
        accepted_receipt = verify_receipt(
            record.receipt,
            self.edge_private_key.public_key(),
            expected_request=request,
            expected_capability=capability,
            now_unix_ms=now_unix_ms,
        )
        if (
            accepted_receipt.outcome
            is not DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
            or accepted_receipt.discord_object_id is None
            or accepted_receipt.bot_user_id is None
        ):
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE,
                "journal outcome lacks accepted mutation evidence",
            )

        try:
            proof = self.target_prover.prove_public_readback(
                request.intent.operation,
                request.intent.target,
                require_message_history=(
                    request.intent.operation
                    is not DiscordEdgeOperation.PUBLIC_THREAD_CREATE
                    or bool(request.intent.content)
                ),
                deadline_unix_ms=(
                    now_unix_ms + _RECONCILIATION_PROOF_DEADLINE_MS
                ),
                now_unix_ms=now_unix_ms,
            )
        except Exception:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now_unix_ms,
            )
        blocker = self._proof_blocker(
            request,
            proof,
            now_unix_ms=self.clock_ms(),
        )
        if blocker is not None:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now_unix_ms,
            )
        accepted = DiscordMutationAccepted(
            operation=request.intent.operation,
            target=request.intent.target,
            discord_object_id=accepted_receipt.discord_object_id,
            bot_user_id=accepted_receipt.bot_user_id,
        )
        if self._accepted_blocker(request, proof, accepted) is not None:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=now_unix_ms,
            )
        try:
            readback = self._readback(request, accepted)
        except Exception:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=self.clock_ms(),
            )
        if self._readback_blocker(request, proof, accepted, readback) is not None:
            return self._result_from_record(
                record,
                request,
                capability,
                replayed=True,
                now_unix_ms=self.clock_ms(),
            )

        occurred = self.clock_ms()
        verified_receipt = sign_receipt(
            self.edge_private_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.VERIFIED,
            discord_object_id=accepted.discord_object_id,
            bot_user_id=accepted.bot_user_id,
            adapter_accepted=True,
            readback_verified=True,
            readback_content_sha256=request.intent.content_sha256,
            readback_thread=readback.thread,
            occurred_at_unix_ms=occurred,
        )
        upgraded = self.journal.upgrade_accepted_unverified(
            request,
            capability,
            expected_receipt=record.receipt,
            verified_receipt=verified_receipt,
            now_unix_ms=occurred,
        )
        return self._result_from_record(
            upgraded,
            request,
            capability,
            replayed=True,
            now_unix_ms=occurred,
        )

    def _live_proof(
        self,
        request: DiscordEdgeRequest,
        *,
        now_unix_ms: int,
    ) -> tuple[
        DiscordLivePublicTargetProof | None,
        DiscordEdgeBlockerCode | None,
        bool,
    ]:
        try:
            if request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
                proof = self.target_prover.prove_public_message_send(
                    request.intent.target,
                    deadline_unix_ms=request.deadline_unix_ms,
                    now_unix_ms=now_unix_ms,
                )
            elif request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT:
                proof = self.target_prover.prove_public_message_edit(
                    request.intent.target,
                    deadline_unix_ms=request.deadline_unix_ms,
                    now_unix_ms=now_unix_ms,
                )
            else:
                proof = self.target_prover.prove_public_thread_create(
                    request.intent.target,
                    has_initial_message=bool(request.intent.content),
                    deadline_unix_ms=request.deadline_unix_ms,
                    now_unix_ms=now_unix_ms,
                )
        except Exception:
            if request.deadline_unix_ms <= self.clock_ms():
                return (
                    None,
                    DiscordEdgeBlockerCode.REQUEST_DEADLINE_EXPIRED,
                    True,
                )
            return None, DiscordEdgeBlockerCode.TARGET_PROOF_UNAVAILABLE, True
        blocker = self._proof_blocker(
            request,
            proof,
            now_unix_ms=self.clock_ms(),
        )
        return proof if blocker is None else None, blocker, False

    def _proof_blocker(
        self,
        request: DiscordEdgeRequest,
        proof: object,
        *,
        now_unix_ms: int,
    ) -> DiscordEdgeBlockerCode | None:
        if not isinstance(proof, DiscordLivePublicTargetProof):
            return DiscordEdgeBlockerCode.TARGET_PROOF_INVALID
        if (
            proof.operation is not request.intent.operation
            or proof.target != request.intent.target
        ):
            return DiscordEdgeBlockerCode.TARGET_PROOF_MISMATCH
        age = now_unix_ms - proof.observed_at_unix_ms
        if age > self.max_proof_age_ms or age < -_MAX_PROOF_FUTURE_SKEW_MS:
            return DiscordEdgeBlockerCode.TARGET_PROOF_STALE
        if not proof.publicly_viewable:
            return DiscordEdgeBlockerCode.TARGET_NOT_PUBLIC
        if not proof.bot_can_view:
            return DiscordEdgeBlockerCode.BOT_CANNOT_VIEW
        if not proof.bot_has_required_permission:
            return DiscordEdgeBlockerCode.PERMISSION_REVOKED
        return None

    def _block_before_dispatch(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        blocker: DiscordEdgeBlockerCode,
        *,
        failed: bool,
        replayed: bool,
        now_unix_ms: int,
    ) -> DiscordEdgeExecutionResult:
        outcome = (
            DiscordEdgeReceiptOutcome.FAILED_BEFORE_DISPATCH
            if failed
            else DiscordEdgeReceiptOutcome.BLOCKED_BEFORE_DISPATCH
        )
        receipt = sign_receipt(
            self.edge_private_key,
            request,
            capability,
            outcome=outcome,
            discord_object_id=None,
            bot_user_id=None,
            adapter_accepted=False,
            readback_verified=False,
            readback_content_sha256=None,
            blocker_code=blocker.value,
            occurred_at_unix_ms=now_unix_ms,
        )
        record = self.journal.finalize_blocked(
            request,
            capability,
            receipt,
            blocker,
            now_unix_ms=now_unix_ms,
        )
        return self._result_from_record(
            record,
            request,
            capability,
            replayed=replayed,
            now_unix_ms=now_unix_ms,
        )

    def _dispatch(self, request: DiscordEdgeRequest) -> DiscordMutationAccepted:
        payload = request.intent.payload
        target = request.intent.target
        if request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
            return self.transport.send_public_message(
                target,
                content=str(payload["content"]),
                reply_to_message_id=payload.get("reply_to_message_id"),
                deadline_unix_ms=request.deadline_unix_ms,
            )
        if request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT:
            return self.transport.edit_public_message(
                target,
                message_id=str(payload["message_id"]),
                content=str(payload["content"]),
                deadline_unix_ms=request.deadline_unix_ms,
            )
        return self.transport.create_public_thread(
            target,
            name=str(payload["name"]),
            initial_message=payload.get("initial_message"),
            auto_archive_minutes=payload.get("auto_archive_minutes"),
            deadline_unix_ms=request.deadline_unix_ms,
        )

    @staticmethod
    def _accepted_blocker(
        request: DiscordEdgeRequest,
        proof: DiscordLivePublicTargetProof,
        accepted: object,
    ) -> DiscordEdgeBlockerCode | None:
        if not isinstance(accepted, DiscordMutationAccepted):
            return DiscordEdgeBlockerCode.DISPATCH_RESULT_INVALID
        if (
            accepted.operation is not request.intent.operation
            or accepted.target != request.intent.target
        ):
            return DiscordEdgeBlockerCode.DISPATCH_BINDING_MISMATCH
        if accepted.bot_user_id != proof.bot_user_id:
            return DiscordEdgeBlockerCode.DISPATCH_BOT_MISMATCH
        if (
            request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT
            and accepted.discord_object_id != request.intent.payload["message_id"]
        ):
            return DiscordEdgeBlockerCode.DISPATCH_BINDING_MISMATCH
        return None

    def _readback(
        self,
        request: DiscordEdgeRequest,
        accepted: DiscordMutationAccepted,
    ) -> DiscordMutationReadback:
        if request.intent.operation in {
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
        }:
            return self.transport.read_public_message(
                request.intent.target,
                operation=request.intent.operation,
                message_id=accepted.discord_object_id,
                expected_reply_to_message_id=(
                    request.intent.payload.get("reply_to_message_id")
                    if request.intent.operation
                    is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
                    else None
                ),
            )
        return self.transport.read_created_public_thread(
            request.intent.target,
            thread_id=accepted.discord_object_id,
            expected_content=request.intent.content,
        )

    @staticmethod
    def _readback_blocker(
        request: DiscordEdgeRequest,
        proof: DiscordLivePublicTargetProof,
        accepted: DiscordMutationAccepted,
        readback: object,
    ) -> DiscordEdgeBlockerCode | None:
        if not isinstance(readback, DiscordMutationReadback):
            return DiscordEdgeBlockerCode.READBACK_INVALID
        if (
            readback.operation is not request.intent.operation
            or readback.target != request.intent.target
            or readback.discord_object_id != accepted.discord_object_id
        ):
            return DiscordEdgeBlockerCode.READBACK_BINDING_MISMATCH
        if readback.author_user_id != proof.bot_user_id:
            return DiscordEdgeBlockerCode.READBACK_AUTHOR_MISMATCH
        if readback.content != request.intent.content:
            return DiscordEdgeBlockerCode.READBACK_CONTENT_MISMATCH
        if (
            request.intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
            and readback.reply_to_message_id
            != request.intent.payload.get("reply_to_message_id")
        ):
            return DiscordEdgeBlockerCode.READBACK_REPLY_MISMATCH
        if request.intent.operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE:
            thread = readback.thread
            assert isinstance(thread, DiscordEdgeThreadReadback)
            if (
                thread.target.target_type
                is not DiscordPublicTargetType.PUBLIC_GUILD_THREAD
                or thread.target.guild_id != request.intent.target.guild_id
                or thread.target.channel_id != accepted.discord_object_id
                or thread.target.parent_channel_id != request.intent.target.channel_id
            ):
                return DiscordEdgeBlockerCode.READBACK_THREAD_TARGET_MISMATCH
            if thread.name != request.intent.payload["name"]:
                return DiscordEdgeBlockerCode.READBACK_THREAD_NAME_MISMATCH
            requested_archive = request.intent.payload.get("auto_archive_minutes")
            if (
                requested_archive is not None
                and thread.auto_archive_minutes != requested_archive
            ):
                return DiscordEdgeBlockerCode.READBACK_THREAD_ARCHIVE_MISMATCH
        return None

    def _accepted_unverified(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        accepted: DiscordMutationAccepted,
        blocker: DiscordEdgeBlockerCode,
        *,
        replayed: bool,
        expected_receipt: SignedDiscordEdgeEnvelope | None = None,
    ) -> DiscordEdgeExecutionResult:
        occurred = self.clock_ms()
        receipt = sign_receipt(
            self.edge_private_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
            discord_object_id=accepted.discord_object_id,
            bot_user_id=accepted.bot_user_id,
            adapter_accepted=True,
            readback_verified=False,
            readback_content_sha256=None,
            blocker_code=blocker.value,
            occurred_at_unix_ms=occurred,
        )
        if expected_receipt is None:
            record = self.journal.record_accepted_unverified(
                request,
                capability,
                receipt,
                blocker,
                now_unix_ms=occurred,
            )
        else:
            record = self.journal.replace_accepted_unverified(
                request,
                capability,
                expected_receipt=expected_receipt,
                replacement_receipt=receipt,
                blocker_code=blocker,
                now_unix_ms=occurred,
            )
        return self._result_from_record(
            record,
            request,
            capability,
            replayed=replayed,
            now_unix_ms=occurred,
        )

    def _dispatch_uncertain(
        self,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        blocker: DiscordEdgeBlockerCode,
        *,
        occurred_at_unix_ms: int,
        replayed: bool,
    ) -> DiscordEdgeExecutionResult:
        receipt_id = str(
            uuid.uuid5(
                _UNCERTAIN_RECEIPT_NAMESPACE,
                "|".join(
                    (
                        request.request_id,
                        capability.capability_id,
                        request.intent.idempotency_key,
                    )
                ),
            )
        )
        receipt = sign_receipt(
            self.edge_private_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
            discord_object_id=None,
            bot_user_id=None,
            adapter_accepted=None,
            readback_verified=False,
            readback_content_sha256=None,
            blocker_code=blocker.value,
            occurred_at_unix_ms=occurred_at_unix_ms,
            receipt_id=receipt_id,
        )
        record = self.journal.record_dispatch_uncertain(
            request,
            capability,
            receipt,
            blocker,
            now_unix_ms=occurred_at_unix_ms,
        )
        return self._result_from_record(
            record,
            request,
            capability,
            replayed=replayed,
            now_unix_ms=max(self.clock_ms(), occurred_at_unix_ms),
        )

    def _result_from_record(
        self,
        record: DiscordEdgeJournalRecord,
        request: DiscordEdgeRequest,
        capability: DiscordEdgeCapability,
        *,
        replayed: bool,
        now_unix_ms: int,
    ) -> DiscordEdgeExecutionResult:
        receipt = record.receipt
        if receipt is not None:
            try:
                verified_receipt = verify_receipt(
                    receipt,
                    self.edge_private_key.public_key(),
                    expected_request=request,
                    expected_capability=capability,
                    now_unix_ms=now_unix_ms,
                )
            except DiscordEdgeProtocolError as exc:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "stored edge receipt failed signature or exact binding verification",
                ) from exc
            expected_outcomes = {
                DiscordEdgeJournalState.DISPATCHING: frozenset(
                    {
                        DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
                        DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
                    }
                ),
                DiscordEdgeJournalState.VERIFIED: frozenset(
                    {DiscordEdgeReceiptOutcome.VERIFIED}
                ),
                DiscordEdgeJournalState.BLOCKED: frozenset(
                    {
                        DiscordEdgeReceiptOutcome.BLOCKED_BEFORE_DISPATCH,
                        DiscordEdgeReceiptOutcome.FAILED_BEFORE_DISPATCH,
                    }
                ),
            }
            if verified_receipt.outcome not in expected_outcomes.get(record.state, ()):
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "stored receipt outcome does not match its journal state",
                )
            expected_blocker = (
                record.blocker_code.value if record.blocker_code is not None else None
            )
            if verified_receipt.blocker_code != expected_blocker:
                raise DiscordEdgeRuntimeError(
                    DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                    "stored receipt blocker does not match its journal state",
                )
        elif record.state in {
            DiscordEdgeJournalState.VERIFIED,
            DiscordEdgeJournalState.BLOCKED,
        }:
            raise DiscordEdgeRuntimeError(
                DiscordEdgeRuntimeErrorCode.JOURNAL_CORRUPT,
                "terminal journal state is missing its signed receipt",
            )
        blocker = record.blocker_code
        if record.state is DiscordEdgeJournalState.DISPATCHING and blocker is None:
            blocker = DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
        return DiscordEdgeExecutionResult(
            state=record.state,
            receipt=receipt,
            blocker_code=blocker,
            replayed=replayed,
        )


__all__ = [
    "DiscordEdgeBlockerCode",
    "DiscordEdgeExecutionResult",
    "DiscordEdgeJournalRecord",
    "DiscordEdgeJournalState",
    "DiscordEdgeReconciliationResult",
    "DiscordEdgeRuntime",
    "DiscordEdgeRuntimeError",
    "DiscordEdgeRuntimeErrorCode",
    "DiscordEdgeTransport",
    "DiscordLivePublicTargetProof",
    "DiscordMutationAccepted",
    "DiscordMutationReadback",
    "DiscordPublicTargetProver",
    "DurableDiscordEdgeJournal",
]
