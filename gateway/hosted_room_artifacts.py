"""Private, task-scoped output artifacts for hosted Group Chat turns."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from gateway.hosted_room_attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
    MAX_GATEWAY_BLOB_BYTES,
    MAX_MESSAGE_ATTACHMENT_BYTES,
)

_SCOPE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
# A lost acknowledgement response can be replayed for one day without keeping
# file metadata or constructor work forever.
ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS = 24 * 60 * 60
UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS = 30 * 24 * 60 * 60
EXPIRED_SOURCE_RECEIPT_RETENTION_SECONDS = 30 * 24 * 60 * 60
ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH = 256
# Dispatch grants expire within one day; source ACK receipts can remain useful
# for 30 days. Keep the execution fence for another full receipt horizon after
# its last producer/retirement activity, then prune only when no artifact row
# still references that logical task.
GENERATION_FENCE_RETENTION_SECONDS = 60 * 24 * 60 * 60
GENERATION_FENCE_PRUNE_BATCH = 256


class RoomArtifactError(ValueError):
    """A hosted-room output artifact failed its scope or integrity contract."""


def _absolute_artifact_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _is_symlink_or_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _raise_symlink_error(
    path: Path | str,
    error: OSError,
    *,
    dir_fd: int | None = None,
) -> None:
    try:
        info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        raise error
    if _is_symlink_or_reparse(info):
        raise RoomArtifactError("Symbolic links cannot be shared.") from error
    raise error


def _open_artifact_path_posix(candidate: Path) -> int:
    read_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_flags = read_flags | os.O_DIRECTORY
    file_flags = read_flags | getattr(os, "O_NONBLOCK", 0)
    parts = candidate.parts
    directory_fd = os.open(candidate.anchor, directory_flags)
    try:
        for part in parts[1:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                _raise_symlink_error(part, exc, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        if len(parts) == 1:
            return os.open(candidate.anchor, file_flags)
        try:
            return os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as exc:
            _raise_symlink_error(parts[-1], exc, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _windows_artifact_components(candidate: Path) -> Iterator[Path]:
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        yield current


def _reject_windows_reparse_components(candidate: Path) -> None:
    for component in _windows_artifact_components(candidate):
        info = os.lstat(component)
        if _is_symlink_or_reparse(info):
            raise RoomArtifactError("Symbolic links cannot be shared.")


def _descriptor_matches_path(descriptor: int, path: Path) -> bool:
    try:
        path_info = os.stat(path, follow_symlinks=False)
        descriptor_info = os.fstat(descriptor)
    except OSError:
        return False
    return not _is_symlink_or_reparse(path_info) and os.path.samestat(
        path_info,
        descriptor_info,
    )


def _open_artifact_path_windows(candidate: Path) -> int:
    _reject_windows_reparse_components(candidate)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    descriptor = os.open(candidate, flags)
    try:
        _reject_windows_reparse_components(candidate)
        if not _descriptor_matches_path(descriptor, candidate):
            raise RoomArtifactError(
                "Artifact path changed while it was being validated."
            )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def open_room_artifact_path(path: Path | str) -> Iterator[tuple[Path, int]]:
    """Open one path without following symlinks and keep its descriptor pinned."""

    candidate = _absolute_artifact_path(path)
    if os.name == "nt":
        descriptor = _open_artifact_path_windows(candidate)
    else:
        descriptor = _open_artifact_path_posix(candidate)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RoomArtifactError("artifact must be a bounded regular file")
        yield candidate, descriptor
    finally:
        os.close(descriptor)


def validate_open_room_artifact_path(path: Path | str, descriptor: int) -> Path:
    """Bind path-based policy results to the object held by ``descriptor``."""

    candidate = _absolute_artifact_path(path)
    if not _descriptor_matches_path(descriptor, candidate):
        raise RoomArtifactError("Artifact path changed while it was being validated.")
    return candidate


@dataclass(frozen=True)
class RoomArtifactScope:
    """Immutable coordinates for one admitted member turn."""

    room_id: str
    task_id: str
    execution_generation: int
    member_id: str
    target_profile: str
    home_install_id: str
    target_install_id: str
    authority_gateway_id: str
    authority_epoch: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoomArtifactScope":
        required = {
            "room_id",
            "task_id",
            "execution_generation",
            "member_id",
            "target_profile",
            "home_install_id",
            "target_install_id",
            "authority_gateway_id",
            "authority_epoch",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise RoomArtifactError("room artifact scope fields are invalid")
        strings = {
            key: str(value[key]).strip()
            for key in required
            if key not in {"execution_generation", "authority_epoch"}
        }
        if not all(strings.values()) or any(
            len(item) > 255 or _SCOPE_ID_RE.fullmatch(item) is None
            for item in strings.values()
        ):
            raise RoomArtifactError("room artifact scope contains an invalid identifier")
        generation = value["execution_generation"]
        epoch = value["authority_epoch"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 1
        ):
            raise RoomArtifactError("room artifact scope generation is invalid")
        return cls(
            **strings,
            execution_generation=generation,
            authority_epoch=epoch,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "task_id": self.task_id,
            "execution_generation": self.execution_generation,
            "member_id": self.member_id,
            "target_profile": self.target_profile,
            "home_install_id": self.home_install_id,
            "target_install_id": self.target_install_id,
            "authority_gateway_id": self.authority_gateway_id,
            "authority_epoch": self.authority_epoch,
        }

    @property
    def key(self) -> str:
        payload = json.dumps(self.as_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def lineage_mapping(self) -> dict[str, Any]:
        """Return the stable task scope shared by every execution generation."""

        return {
            key: value
            for key, value in self.as_mapping().items()
            if key != "execution_generation"
        }

    @property
    def lineage_key(self) -> str:
        return hashlib.sha256(self.lineage_json.encode()).hexdigest()

    @property
    def lineage_json(self) -> str:
        return json.dumps(
            self.lineage_mapping(), sort_keys=True, separators=(",", ":")
        )


_CURRENT_SCOPE: ContextVar[RoomArtifactScope | None] = ContextVar(
    "hosted_room_artifact_scope",
    default=None,
)


def _gateway_artifact_db_path(path: Path) -> Path:
    """Collapse a named profile's state path onto the gateway-wide root DB."""
    profile_home = path.parent
    if profile_home.parent.name == "profiles":
        return profile_home.parent.parent / path.name
    return path


def _lineage_identity_sql(scope_json: str) -> str:
    """Return canonical lineage JSON using SQLite's built-in JSON functions."""

    keys = (
        "authority_epoch",
        "authority_gateway_id",
        "home_install_id",
        "member_id",
        "room_id",
        "target_install_id",
        "target_profile",
        "task_id",
    )
    arguments = ", ".join(
        f"'{key}', json_extract({scope_json}, '$.{key}')" for key in keys
    )
    return (f"CASE WHEN json_extract({scope_json}, '$.kind')='classic' THEN "
            f"json_object('export_id',json_extract({scope_json}, '$.export_id'),'kind','classic') "
            f"ELSE json_object({arguments}) END")


def bind_room_artifact_scope(scope: RoomArtifactScope) -> Token:
    return _CURRENT_SCOPE.set(scope)


def reset_room_artifact_scope(token: Token) -> None:
    _CURRENT_SCOPE.reset(token)


def current_room_artifact_scope() -> RoomArtifactScope | None:
    return _CURRENT_SCOPE.get()


class RoomArtifactOutbox:
    """Durable private bytes awaiting canonical import by the room home."""

    def __init__(self, db_path: Path | str, *, root: Path | str | None = None) -> None:
        requested_db_path = Path(db_path)
        self.db_path = (
            requested_db_path
            if root is not None
            else _gateway_artifact_db_path(requested_db_path)
        )
        self.root = Path(root or self.db_path.parent / "hosted-room-artifact-outbox")
        self.blob_root = self.root / "blobs"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blob_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in (self.root, self.blob_root):
            with suppress(OSError):
                os.chmod(path, 0o700)
        now = time.time()
        with self._connect() as conn:
            self._initialize(conn)
        self.retry_scheduled_cleanups()
        self._reclaim_pending_acknowledged_blobs(now=now)
        self.prune_acknowledged_receipts(now=now)
        self.prune_unacknowledged_artifacts(now=now)
        self.prune_generation_fences(now=now)
        cutoff = now - 3600
        with self._connect() as conn:
            referenced = {
                str(row["blob_name"])
                for row in conn.execute(
                    """SELECT blob_name FROM hosted_room_output_artifacts
                       WHERE acknowledged_at IS NULL"""
                ).fetchall()
            }
        for path in self.blob_root.iterdir():
            try:
                if (
                    path.is_file()
                    and path.name not in referenced
                    and path.stat().st_mtime < cutoff
                ):
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def _connect(self) -> sqlite3.Connection:
        from hermes_state import apply_wal_with_fallback

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(conn, db_label="state.db (hosted room artifact outbox)")
        return conn

    def _reclaim_blob_rows(
        self,
        rows: Sequence[sqlite3.Row],
        *,
        now: float,
        best_effort: bool,
    ) -> int:
        reclaimed_ids: list[str] = []
        first_error: OSError | None = None
        for row in rows:
            try:
                (self.blob_root / str(row["blob_name"])).unlink(missing_ok=True)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
            else:
                reclaimed_ids.append(str(row["artifact_id"]))
        if reclaimed_ids:
            placeholders = ",".join("?" for _ in reclaimed_ids)
            with self._lock, self._connect() as conn:
                self._initialize(conn)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    f"""UPDATE hosted_room_output_artifacts
                           SET blob_reclaimed_at=?
                         WHERE artifact_id IN ({placeholders})
                           AND acknowledged_at IS NOT NULL""",
                    (now, *reclaimed_ids),
                )
                conn.commit()
        if first_error is not None and not best_effort:
            raise first_error
        return len(reclaimed_ids)

    def _reclaim_pending_acknowledged_blobs(self, *, now: float) -> int:
        with self._connect() as conn:
            self._initialize(conn)
            rows = conn.execute(
                """SELECT artifact_id, blob_name
                     FROM hosted_room_output_artifacts
                    WHERE acknowledged_at IS NOT NULL
                      AND blob_reclaimed_at IS NULL
                    ORDER BY acknowledged_at, artifact_id
                    LIMIT ?""",
                (ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH,),
            ).fetchall()
        return self._reclaim_blob_rows(rows, now=now, best_effort=True)

    def prune_acknowledged_receipts(self, *, now: float | None = None) -> int:
        """Retire one bounded batch of expired lost-ack retry receipts."""

        current = time.time() if now is None else float(now)
        cutoff = current - ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT artifact_id, blob_name, blob_reclaimed_at
                     FROM hosted_room_output_artifacts
                    WHERE acknowledged_at IS NOT NULL
                      AND (
                          (receipt_expires_at IS NOT NULL AND receipt_expires_at<=?)
                          OR (receipt_expires_at IS NULL AND acknowledged_at<=?)
                      )
                    ORDER BY acknowledged_at, artifact_id
                    LIMIT ?""",
                (current, cutoff, ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH),
            ).fetchall()
            if rows:
                conn.executemany(
                    "DELETE FROM hosted_room_output_artifacts WHERE artifact_id=?",
                    ((str(row["artifact_id"]),) for row in rows),
                )
                conn.commit()
        for row in rows:
            if row["blob_reclaimed_at"] is None:
                with suppress(OSError):
                    (self.blob_root / str(row["blob_name"])).unlink(missing_ok=True)
        return len(rows)

    def prune_unacknowledged_artifacts(self, *, now: float | None = None) -> int:
        """Reclaim abandoned bytes while retaining an exact ACK tombstone."""
        current = time.time() if now is None else float(now)
        cutoff = current - UNACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT artifact_id, blob_name, scope_json
                     FROM hosted_room_output_artifacts
                    WHERE acknowledged_at IS NULL AND created_at<=?
                      AND json_extract(scope_json, '$.kind') IS NULL
                    ORDER BY created_at, artifact_id
                    LIMIT ?""",
                (cutoff, ACKNOWLEDGED_ARTIFACT_PRUNE_BATCH),
            ).fetchall()
            for row in rows:
                scope = RoomArtifactScope.from_mapping(json.loads(row["scope_json"]))
                self._retire_generation(conn, scope)
                message_event_id = (
                    f"dmessage:{scope.task_id.removeprefix('dtask:')}"
                )
                conn.execute(
                    """UPDATE hosted_room_output_artifacts
                          SET acknowledged_at=?, ack_message_event_id=?,
                              receipt_expires_at=?
                        WHERE artifact_id=? AND acknowledged_at IS NULL""",
                    (
                        current,
                        message_event_id,
                        current + EXPIRED_SOURCE_RECEIPT_RETENTION_SECONDS,
                        row["artifact_id"],
                    ),
                )
            if rows:
                conn.commit()
        if rows:
            self._reclaim_pending_acknowledged_blobs(now=current)
        return len(rows)

    def prune_generation_fences(self, *, now: float | None = None) -> int:
        """Bound retired lineage receipts after every producer horizon closes."""

        current = time.time() if now is None else float(now)
        cutoff = current - GENERATION_FENCE_RETENTION_SECONDS
        artifact_identity = _lineage_identity_sql("artifact.scope_json")
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""SELECT fence.lineage_identity
                       FROM hosted_room_output_generation_fences AS fence
                      WHERE fence.updated_at<=?
                        AND NOT EXISTS (
                            SELECT 1 FROM hosted_room_output_artifacts AS artifact
                             WHERE {artifact_identity}=fence.lineage_identity
                        )
                      ORDER BY fence.updated_at, fence.lineage_identity
                      LIMIT ?""",
                (cutoff, GENERATION_FENCE_PRUNE_BATCH),
            ).fetchall()
            if rows:
                conn.executemany(
                    """DELETE FROM hosted_room_output_generation_fences
                        WHERE lineage_identity=?""",
                    ((str(row["lineage_identity"]),) for row in rows),
                )
                conn.commit()
        return len(rows)

    @staticmethod
    def _initialize(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS hosted_room_output_artifacts (
                artifact_id TEXT PRIMARY KEY,
                scope_key TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                mime TEXT NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                blob_name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                acknowledged_at REAL,
                ack_message_event_id TEXT,
                receipt_expires_at REAL,
                cleanup_required_at REAL,
                blob_reclaimed_at REAL,
                UNIQUE(scope_key, sha256, name)
            )"""
        )
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(hosted_room_output_artifacts)")
        }
        if "blob_reclaimed_at" not in columns:
            conn.execute(
                """ALTER TABLE hosted_room_output_artifacts
                   ADD COLUMN blob_reclaimed_at REAL"""
            )
        if "ack_message_event_id" not in columns:
            conn.execute(
                """ALTER TABLE hosted_room_output_artifacts
                   ADD COLUMN ack_message_event_id TEXT"""
            )
        if "receipt_expires_at" not in columns:
            conn.execute(
                """ALTER TABLE hosted_room_output_artifacts
                   ADD COLUMN receipt_expires_at REAL"""
            )
        if "cleanup_required_at" not in columns:
            conn.execute(
                """ALTER TABLE hosted_room_output_artifacts
                   ADD COLUMN cleanup_required_at REAL"""
            )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_hosted_room_output_scope
               ON hosted_room_output_artifacts(scope_key, created_at)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_hosted_room_output_ack_expiry
               ON hosted_room_output_artifacts(acknowledged_at, artifact_id)
               WHERE acknowledged_at IS NOT NULL"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_hosted_room_output_ack_cleanup
               ON hosted_room_output_artifacts(acknowledged_at, artifact_id)
               WHERE acknowledged_at IS NOT NULL AND blob_reclaimed_at IS NULL"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS hosted_room_output_generation_fences (
                lineage_key TEXT PRIMARY KEY,
                lineage_json TEXT NOT NULL,
                lineage_identity TEXT,
                max_generation INTEGER NOT NULL,
                retired_generation INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )"""
        )
        fence_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(hosted_room_output_generation_fences)"
            )
        }
        if "lineage_identity" not in fence_columns:
            conn.execute(
                """ALTER TABLE hosted_room_output_generation_fences
                   ADD COLUMN lineage_identity TEXT"""
            )
        conn.execute(
            """UPDATE hosted_room_output_generation_fences
                  SET lineage_identity=lineage_json
                WHERE lineage_identity IS NULL"""
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
               idx_hosted_room_output_generation_identity
               ON hosted_room_output_generation_fences(lineage_identity)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_hosted_room_output_generation_expiry
               ON hosted_room_output_generation_fences(updated_at, lineage_identity)"""
        )
        trigger_names = {
            "hosted_room_output_generation_guard_insert",
            "hosted_room_output_generation_guard_update",
            "hosted_room_output_generation_track_insert",
            "hosted_room_output_generation_track_update",
            "hosted_room_output_generation_track_terminal",
            "hosted_room_output_generation_track_delete",
        }
        trigger_placeholders = ", ".join("?" for _ in trigger_names)
        installed_triggers = {
            str(row[0])
            for row in conn.execute(
                f"""SELECT name FROM sqlite_master
                     WHERE type='trigger' AND name IN ({trigger_placeholders})""",
                tuple(sorted(trigger_names)),
            )
        }
        # Existing triggers must learn the non-hosted lineage discriminator once.
        trigger_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='hosted_room_output_generation_track_insert'").fetchone()
        if trigger_sql and "$.kind" not in trigger_sql[0]:
            for name in trigger_names:
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            installed_triggers = set()
        if installed_triggers != trigger_names:
            aggregates: dict[str, tuple[RoomArtifactScope, int, int]] = {}
            rows = conn.execute(
                """SELECT scope_json, acknowledged_at, cleanup_required_at
                     FROM hosted_room_output_artifacts"""
            ).fetchall()
            for row in rows:
                try:
                    scope = RoomArtifactScope.from_mapping(
                        json.loads(row["scope_json"])
                    )
                except Exception:
                    continue
                current = aggregates.get(scope.lineage_json)
                maximum = max(
                    scope.execution_generation,
                    current[1] if current is not None else 0,
                )
                retired = current[2] if current is not None else 0
                if row["acknowledged_at"] is not None or row["cleanup_required_at"] is not None:
                    retired = max(retired, scope.execution_generation)
                aggregates[scope.lineage_json] = (scope, maximum, retired)
            for scope, maximum, retired in aggregates.values():
                conn.execute(
                    """INSERT INTO hosted_room_output_generation_fences
                       (lineage_key, lineage_json, lineage_identity,
                        max_generation, retired_generation, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(lineage_identity) DO UPDATE SET
                           max_generation=MAX(
                               max_generation,
                               excluded.max_generation
                           ),
                           retired_generation=MAX(
                               retired_generation,
                               excluded.retired_generation
                           ),
                           updated_at=MAX(updated_at, excluded.updated_at)""",
                    (
                        scope.lineage_key,
                        scope.lineage_json,
                        scope.lineage_json,
                        maximum,
                        retired,
                        time.time(),
                    ),
                )
        identity = _lineage_identity_sql("NEW.scope_json")
        generation = "CAST(json_extract(NEW.scope_json, '$.execution_generation') AS INTEGER)"
        deleted_identity = _lineage_identity_sql("OLD.scope_json")
        deleted_generation = (
            "CAST(json_extract(OLD.scope_json, '$.execution_generation') AS INTEGER)"
        )
        now = "CAST(strftime('%s', 'now') AS REAL)"
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_guard_insert
                BEFORE INSERT ON hosted_room_output_artifacts
                WHEN EXISTS (
                    SELECT 1 FROM hosted_room_output_generation_fences AS fence
                     WHERE fence.lineage_identity={identity}
                       AND (
                           {generation} < fence.max_generation
                           OR {generation} <= fence.retired_generation
                       )
                )
                BEGIN
                    SELECT RAISE(ABORT, 'stale room artifact execution generation');
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_track_insert
                AFTER INSERT ON hosted_room_output_artifacts
                BEGIN
                    INSERT INTO hosted_room_output_generation_fences
                    (lineage_key, lineage_json, lineage_identity,
                     max_generation, retired_generation, updated_at)
                    VALUES ({identity}, {identity}, {identity}, {generation}, 0, {now})
                    ON CONFLICT(lineage_identity) DO UPDATE SET
                        max_generation=MAX(
                            max_generation,
                            excluded.max_generation
                        ),
                        updated_at=excluded.updated_at;
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_guard_update
                BEFORE UPDATE OF scope_json ON hosted_room_output_artifacts
                WHEN EXISTS (
                    SELECT 1 FROM hosted_room_output_generation_fences AS fence
                     WHERE fence.lineage_identity={identity}
                       AND (
                           {generation} < fence.max_generation
                           OR {generation} <= fence.retired_generation
                       )
                )
                BEGIN
                    SELECT RAISE(ABORT, 'stale room artifact execution generation');
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_track_update
                AFTER UPDATE OF scope_json ON hosted_room_output_artifacts
                BEGIN
                    INSERT INTO hosted_room_output_generation_fences
                    (lineage_key, lineage_json, lineage_identity,
                     max_generation, retired_generation, updated_at)
                    VALUES ({identity}, {identity}, {identity}, {generation}, 0, {now})
                    ON CONFLICT(lineage_identity) DO UPDATE SET
                        max_generation=MAX(
                            max_generation,
                            excluded.max_generation
                        ),
                        updated_at=excluded.updated_at;
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_track_terminal
                AFTER UPDATE OF acknowledged_at, cleanup_required_at
                ON hosted_room_output_artifacts
                WHEN NEW.acknowledged_at IS NOT NULL
                     OR NEW.cleanup_required_at IS NOT NULL
                BEGIN
                    UPDATE hosted_room_output_generation_fences
                       SET max_generation=MAX(max_generation, {generation}),
                           retired_generation=MAX(
                               retired_generation,
                               {generation}
                           ),
                           updated_at={now}
                     WHERE lineage_identity={identity};
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS
                hosted_room_output_generation_track_delete
                BEFORE DELETE ON hosted_room_output_artifacts
                BEGIN
                    UPDATE hosted_room_output_generation_fences
                       SET max_generation=MAX(
                               max_generation,
                               {deleted_generation}
                           ),
                           retired_generation=MAX(
                               retired_generation,
                               {deleted_generation}
                           ),
                           updated_at={now}
                     WHERE lineage_identity={deleted_identity};
                END"""
        )
        conn.commit()

    @staticmethod
    def _admit_generation(
        conn: sqlite3.Connection,
        scope: RoomArtifactScope,
    ) -> None:
        """Advance one logical task generation or reject a stale producer."""

        if scope.as_mapping().get("kind") == "classic":
            from gateway.classic_output_exports import validate_write
            validate_write(conn, scope)

        row = conn.execute(
            """SELECT max_generation, retired_generation
                 FROM hosted_room_output_generation_fences
                WHERE lineage_identity=?""",
            (scope.lineage_json,),
        ).fetchone()
        generation = scope.execution_generation
        if row is not None and (
            generation < int(row["max_generation"])
            or generation <= int(row["retired_generation"])
        ):
            raise RoomArtifactError("room artifact execution generation is stale")
        now = time.time()
        if row is None:
            conn.execute(
                """INSERT INTO hosted_room_output_generation_fences
                   (lineage_key, lineage_json, lineage_identity,
                    max_generation, retired_generation, updated_at)
                   VALUES (?, ?, ?, ?, 0, ?)""",
                (
                    scope.lineage_key,
                    scope.lineage_json,
                    scope.lineage_json,
                    generation,
                    now,
                ),
            )
        else:
            conn.execute(
                """UPDATE hosted_room_output_generation_fences
                      SET max_generation=MAX(max_generation, ?), updated_at=?
                    WHERE lineage_identity=?""",
                (generation, now, scope.lineage_json),
            )

    @staticmethod
    def _retire_generation(
        conn: sqlite3.Connection,
        scope: RoomArtifactScope,
    ) -> None:
        """Fence an admitted generation before its private bytes are retired."""

        now = time.time()
        conn.execute(
            """INSERT INTO hosted_room_output_generation_fences
               (lineage_key, lineage_json, lineage_identity,
                max_generation, retired_generation, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(lineage_identity) DO UPDATE SET
                   max_generation=MAX(max_generation, excluded.max_generation),
                   retired_generation=MAX(
                       retired_generation,
                       excluded.retired_generation
                   ),
                   updated_at=excluded.updated_at""",
            (
                scope.lineage_key,
                scope.lineage_json,
                scope.lineage_json,
                scope.execution_generation,
                scope.execution_generation,
                now,
            ),
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        name = Path(str(value or "")).name.strip()
        if (
            not name
            or name in {".", ".."}
            or len(name) > 255
            or any(char in name for char in ("/", "\\", "\x00", "\n", "\r"))
        ):
            raise RoomArtifactError("artifact name must be a bounded basename")
        return name

    @staticmethod
    def _classify(name: str, data: bytes) -> tuple[str, str]:
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if data.startswith(b"%PDF-"):
            return "pdf", "application/pdf"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image", "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image", "image/jpeg"
        if mime.startswith("image/"):
            raise RoomArtifactError("image bytes do not match their file type")
        if mime == "application/pdf":
            raise RoomArtifactError("PDF bytes do not match their file type")
        if mime.startswith("text/"):
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RoomArtifactError("text artifact is not valid UTF-8") from exc
        return "file", mime

    @staticmethod
    def _manifest(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "artifact_id": str(row["artifact_id"]),
            "kind": str(row["kind"]),
            "name": str(row["name"]),
            "size": int(row["size"]),
            "mime": str(row["mime"]),
            "sha256": str(row["sha256"]),
        }

    def put_open_file(
        self,
        *,
        scope: RoomArtifactScope,
        descriptor: int,
        source_name: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Copy one already-open regular file into private storage."""

        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= MAX_ATTACHMENT_BYTES
        ):
            raise RoomArtifactError("artifact must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            handle.seek(0)
            data = handle.read(MAX_ATTACHMENT_BYTES + 1)
        if len(data) != info.st_size or len(data) > MAX_ATTACHMENT_BYTES:
            raise RoomArtifactError("artifact changed while it was being copied")
        return self.put_bytes(
            scope=scope,
            data=data,
            source_name=source_name,
            name=name,
        )

    def put_bytes(
        self,
        *,
        scope: RoomArtifactScope,
        data: bytes,
        source_name: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Store bounded bytes already read through a trusted backend adapter."""

        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_ATTACHMENT_BYTES:
            raise RoomArtifactError("artifact must be a bounded regular file")
        self.discard_superseded(scope)
        safe_name = self._safe_name(name or source_name)
        kind, mime = self._classify(safe_name, data)
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"rart_{hashlib.sha256((scope.key + digest + safe_name).encode()).hexdigest()[:32]}"
        blob_name = f"blob_{secrets.token_hex(16)}"
        target = self.blob_root / blob_name

        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            self._admit_generation(conn, scope)
            existing = conn.execute(
                """SELECT * FROM hosted_room_output_artifacts
                   WHERE scope_key=? AND sha256=? AND name=?""",
                (scope.key, digest, safe_name),
            ).fetchone()
            if existing is not None:
                return self._manifest(existing)
            if scope.as_mapping().get("kind") == "classic":
                from gateway.classic_output_exports import MAX_FILES
                if conn.execute("SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL").fetchone()[0] >= MAX_FILES:
                    raise RoomArtifactError("Gateway file count quota exceeded; retire existing files first")
            totals = conn.execute(
                """SELECT COUNT(*) AS count, COALESCE(SUM(size), 0) AS bytes
                   FROM hosted_room_output_artifacts
                   WHERE scope_key=? AND acknowledged_at IS NULL""",
                (scope.key,),
            ).fetchone()
            if (
                int(totals["count"]) >= MAX_ATTACHMENTS_PER_MESSAGE
                or int(totals["bytes"]) + len(data) > MAX_MESSAGE_ATTACHMENT_BYTES
            ):
                raise RoomArtifactError("room turn artifact quota exceeded")
            gateway_bytes = int(conn.execute(
                """SELECT COALESCE(SUM(size), 0)
                   FROM hosted_room_output_artifacts
                   WHERE acknowledged_at IS NULL"""
            ).fetchone()[0])
            if gateway_bytes + len(data) > MAX_GATEWAY_BLOB_BYTES:
                raise RoomArtifactError("gateway room artifact quota exceeded")
            temp = self.blob_root / f".tmp-{secrets.token_hex(16)}"
            file_descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(file_descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
                os.chmod(target, 0o600)
                conn.execute(
                    """INSERT INTO hosted_room_output_artifacts
                       (artifact_id, scope_key, scope_json, name, kind, mime, size,
                        sha256, blob_name, created_at, acknowledged_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        artifact_id,
                        scope.key,
                        json.dumps(scope.as_mapping(), sort_keys=True, separators=(",", ":")),
                        safe_name,
                        kind,
                        mime,
                        len(data),
                        digest,
                        blob_name,
                        time.time(),
                    ),
                )
                conn.commit()
            except Exception:
                temp.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise
            row = conn.execute(
                "SELECT * FROM hosted_room_output_artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("stored room artifact could not be reloaded")
            return self._manifest(row)

    def put_path(
        self,
        *,
        scope: RoomArtifactScope,
        path: Path | str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Open and copy one regular file into private storage before returning."""

        with open_room_artifact_path(path) as (candidate, descriptor):
            return self.put_open_file(
                scope=scope,
                descriptor=descriptor,
                source_name=candidate.name,
                name=name,
            )

    def list(self, scope: RoomArtifactScope) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._initialize(conn)
            rows = conn.execute(
                """SELECT * FROM hosted_room_output_artifacts
                   WHERE scope_key=? AND acknowledged_at IS NULL
                   ORDER BY created_at, artifact_id""",
                (scope.key,),
            ).fetchall()
        return [self._manifest(row) for row in rows]

    def read(self, scope: RoomArtifactScope, artifact_id: str) -> tuple[dict[str, Any], bytes]:
        with self._connect() as conn:
            self._initialize(conn)
            row = conn.execute(
                """SELECT * FROM hosted_room_output_artifacts
                   WHERE scope_key=? AND artifact_id=? AND acknowledged_at IS NULL""",
                (scope.key, artifact_id),
            ).fetchone()
        if row is None:
            raise RoomArtifactError("room artifact not found")
        path = self.blob_root / str(row["blob_name"])
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size != int(row["size"]):
                raise RoomArtifactError("room artifact bytes changed")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(MAX_ATTACHMENT_BYTES + 1)
        finally:
            os.close(descriptor)
        if hashlib.sha256(data).hexdigest() != str(row["sha256"]):
            raise RoomArtifactError("room artifact failed SHA-256 validation")
        return self._manifest(row), data

    def retirement_complete(self, scope: RoomArtifactScope) -> bool:
        """Prove exact generation retirement after short-lived ACK rows expire."""

        with self._lock, self._connect() as conn:
            fence = conn.execute(
                """SELECT retired_generation FROM hosted_room_output_generation_fences
                   WHERE lineage_identity=?""",
                (scope.lineage_json,),
            ).fetchone()
            if (
                fence is None
                or int(fence["retired_generation"]) < scope.execution_generation
            ):
                return False
            pending = conn.execute(
                """SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=?
                   AND (acknowledged_at IS NULL OR blob_reclaimed_at IS NULL) LIMIT 1""",
                (scope.key,),
            ).fetchone()
        return pending is None

    def acknowledge(
        self,
        scope: RoomArtifactScope,
        artifact_ids: Sequence[str],
        *,
        message_event_id: str,
    ) -> int:
        expected_event_id = f"dmessage:{scope.task_id.removeprefix('dtask:')}"
        if message_event_id != expected_event_id:
            raise RoomArtifactError(
                "room artifact acknowledgement commitment changed"
            )
        ids = tuple(dict.fromkeys(str(item) for item in artifact_ids if str(item)))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""SELECT artifact_id, blob_name, acknowledged_at,
                           ack_message_event_id, blob_reclaimed_at
                    FROM hosted_room_output_artifacts
                    WHERE scope_key=? AND artifact_id IN ({placeholders})""",
                (scope.key, *ids),
            ).fetchall()
            if len(rows) != len(ids):
                raise RoomArtifactError("room artifact acknowledgement scope changed")
            if any(
                row["ack_message_event_id"] not in {None, message_event_id}
                for row in rows
            ):
                raise RoomArtifactError(
                    "room artifact acknowledgement commitment changed"
                )
            acknowledged_at = time.time()
            changed = conn.execute(
                f"""UPDATE hosted_room_output_artifacts
                       SET acknowledged_at=?, ack_message_event_id=?,
                           receipt_expires_at=?
                    WHERE scope_key=? AND artifact_id IN ({placeholders})
                      AND acknowledged_at IS NULL""",
                (
                    acknowledged_at,
                    message_event_id,
                    acknowledged_at + ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS,
                    scope.key,
                    *ids,
                ),
            ).rowcount
            if changed == 0 and any(
                row["ack_message_event_id"] is None for row in rows
            ):
                conn.execute(
                    f"""UPDATE hosted_room_output_artifacts
                           SET ack_message_event_id=?
                         WHERE scope_key=? AND artifact_id IN ({placeholders})
                           AND acknowledged_at IS NOT NULL
                           AND ack_message_event_id IS NULL""",
                    (message_event_id, scope.key, *ids),
                )
            self._retire_generation(conn, scope)
            conn.commit()
        pending_reclamation = [row for row in rows if row["blob_reclaimed_at"] is None]
        self._reclaim_blob_rows(
            pending_reclamation,
            now=acknowledged_at,
            best_effort=False,
        )
        return int(changed)

    def discard_durably(self, scope: RoomArtifactScope) -> int:
        """Persist exact cleanup intent before attempting private-byte removal."""
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            self._retire_generation(conn, scope)
            conn.execute(
                """UPDATE hosted_room_output_artifacts
                      SET cleanup_required_at=?
                    WHERE scope_key=? AND blob_reclaimed_at IS NULL""",
                (time.time(), scope.key),
            )
            conn.commit()
        return self.discard(scope)

    def retry_scheduled_cleanups(self) -> int:
        """Replay cleanup obligations left by a prior filesystem/DB fault."""
        with self._connect() as conn:
            self._initialize(conn)
            rows = conn.execute(
                """SELECT DISTINCT scope_json
                     FROM hosted_room_output_artifacts
                    WHERE cleanup_required_at IS NOT NULL"""
            ).fetchall()
        removed = 0
        for row in rows:
            try:
                mapping = json.loads(row["scope_json"])
                if mapping.get("kind") == "classic":
                    from gateway.classic_output_exports import ClassicExportScope
                    scope = ClassicExportScope(mapping["export_id"], mapping["execution_generation"])
                else:
                    scope = RoomArtifactScope.from_mapping(mapping)
                removed += self.discard(scope)
            except Exception:
                continue
        return removed

    def discard(self, scope: RoomArtifactScope) -> int:
        """Purge every unreclaimed private blob for one retired attempt."""

        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            self._retire_generation(conn, scope)
            rows = conn.execute(
                """SELECT artifact_id, blob_name FROM hosted_room_output_artifacts
                   WHERE scope_key=? AND blob_reclaimed_at IS NULL""",
                (scope.key,),
            ).fetchall()
            for row in rows:
                (self.blob_root / str(row["blob_name"])).unlink(missing_ok=True)
            conn.execute(
                "DELETE FROM hosted_room_output_artifacts WHERE scope_key=?",
                (scope.key,),
            )
            conn.commit()
        return len(rows)

    def discard_claims(self, claims: Mapping[str, Any]) -> int:
        """Purge output bytes covered by one revoked room-member grant."""

        expected = {
            key: claims.get(key)
            for key in (
                "room_id",
                "home_install_id",
                "authority_gateway_id",
                "authority_epoch",
                "member_id",
                "target_install_id",
                "target_profile",
            )
        }
        scopes_to_discard: dict[str, RoomArtifactScope] = {}
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL"
            ).fetchall()
            for row in rows:
                try:
                    scope = RoomArtifactScope.from_mapping(json.loads(row["scope_json"]))
                except Exception:
                    continue
                mapping = scope.as_mapping()
                if all(mapping.get(key) == value for key, value in expected.items()):
                    scopes_to_discard[scope.key] = scope
            if scopes_to_discard:
                for scope in scopes_to_discard.values():
                    self._retire_generation(conn, scope)
                conn.executemany(
                    """UPDATE hosted_room_output_artifacts
                          SET cleanup_required_at=?
                        WHERE scope_key=? AND acknowledged_at IS NULL""",
                    (
                        (time.time(), scope.key)
                        for scope in scopes_to_discard.values()
                    ),
                )
                conn.commit()
        removed = 0
        for scope in scopes_to_discard.values():
            removed += self.discard(scope)
        return removed

    def discard_superseded(self, scope: RoomArtifactScope) -> int:
        """Purge older execution generations for the same logical task."""

        if scope.as_mapping().get("kind") == "classic":
            return 0  # Published classic versions belong to the owner until retirement.

        scopes_to_discard: dict[str, RoomArtifactScope] = {}
        with self._lock, self._connect() as conn:
            self._initialize(conn)
            conn.execute("BEGIN IMMEDIATE")
            self._admit_generation(conn, scope)
            rows = conn.execute(
                "SELECT * FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL"
            ).fetchall()
            current = scope.as_mapping()
            for row in rows:
                try:
                    candidate = RoomArtifactScope.from_mapping(
                        json.loads(row["scope_json"])
                    )
                except Exception:
                    continue
                mapping = candidate.as_mapping()
                if (
                    candidate.execution_generation < scope.execution_generation
                    and all(
                        mapping[key] == current[key]
                        for key in current
                        if key != "execution_generation"
                    )
                ):
                    scopes_to_discard[candidate.key] = candidate
            if scopes_to_discard:
                conn.executemany(
                    """UPDATE hosted_room_output_artifacts
                          SET cleanup_required_at=?
                        WHERE scope_key=? AND acknowledged_at IS NULL""",
                    (
                        (time.time(), candidate.key)
                        for candidate in scopes_to_discard.values()
                    ),
                )
            conn.commit()
        removed = 0
        for candidate in scopes_to_discard.values():
            removed += self.discard(candidate)
        return removed


def terminal_artifact_manifest(db_path: Path | str, scope: RoomArtifactScope) -> dict[str, Any] | None:
    items = RoomArtifactOutbox(db_path).list(scope)
    if not items:
        return None
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"version": 1, "manifest_digest": digest, "items": items}


def validate_terminal_artifact_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate the bounded metadata returned by one room member turn."""

    if value is None:
        return []
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "manifest_digest",
        "items",
    }:
        raise RoomArtifactError("terminal artifact manifest fields are invalid")
    if value.get("version") != 1 or not isinstance(value.get("items"), list):
        raise RoomArtifactError("terminal artifact manifest version is unsupported")
    items = value["items"]
    if not 0 < len(items) <= MAX_ATTACHMENTS_PER_MESSAGE:
        raise RoomArtifactError("terminal artifact manifest count is invalid")
    normalized: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, Mapping) or set(raw) != {
            "artifact_id",
            "kind",
            "name",
            "size",
            "mime",
            "sha256",
        }:
            raise RoomArtifactError("terminal artifact item fields are invalid")
        artifact_id = str(raw["artifact_id"])
        sha256 = str(raw["sha256"])
        kind = str(raw["kind"])
        mime = str(raw["mime"])
        size = raw["size"]
        if (
            not artifact_id.startswith("rart_")
            or len(artifact_id) != 37
            or not all(char in "0123456789abcdef" for char in artifact_id[5:])
            or len(sha256) != 64
            or not all(char in "0123456789abcdef" for char in sha256)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= MAX_ATTACHMENT_BYTES
            or kind not in {"image", "pdf", "file"}
            or "/" not in mime
            or len(mime) > 127
            or (kind == "image" and not mime.startswith("image/"))
            or (kind == "pdf" and mime != "application/pdf")
        ):
            raise RoomArtifactError("terminal artifact item metadata is invalid")
        normalized.append({
            "artifact_id": artifact_id,
            "kind": kind,
            "name": RoomArtifactOutbox._safe_name(str(raw["name"])),
            "size": size,
            "mime": mime,
            "sha256": sha256,
        })
    if sum(item["size"] for item in normalized) > MAX_MESSAGE_ATTACHMENT_BYTES:
        raise RoomArtifactError("terminal artifact manifest byte size is invalid")
    expected = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if str(value.get("manifest_digest")) != expected:
        raise RoomArtifactError("terminal artifact manifest digest changed")
    return normalized


__all__ = [
    "ACKNOWLEDGED_ARTIFACT_RETENTION_SECONDS",
    "RoomArtifactError",
    "RoomArtifactOutbox",
    "RoomArtifactScope",
    "bind_room_artifact_scope",
    "current_room_artifact_scope",
    "reset_room_artifact_scope",
    "terminal_artifact_manifest",
    "validate_terminal_artifact_manifest",
]
