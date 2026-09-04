"""Scoped, durable RoomLink attachment staging for peer run admission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.hosted_room_peer import (
    MAX_ROOM_LINK_ATTACHMENT_BYTES,
    HostedMemberDispatch,
    HostedRoomGrantError,
    attachment_manifest_digest,
    canonical_attachment_manifest,
    verify_room_grant,
)
from gateway.platforms.api_server_room_grants import (
    RoomGrantReauthorizationRequired,
    _effective_room_profile,
)


logger = logging.getLogger(__name__)


SPOOL_TTL_SECONDS = 24 * 60 * 60
MAX_SPOOL_BYTES = 512 * 1024 * 1024
MAX_ROOM_SPOOL_BYTES = 128 * 1024 * 1024
MAX_MEMBER_SPOOL_BYTES = 64 * 1024 * 1024
MAX_SPOOL_BATCHES = 1024
MAX_ROOM_SPOOL_BATCHES = 256
MAX_MEMBER_SPOOL_BATCHES = 128
MAX_SPOOL_FILES = 4096
MAX_ROOM_SPOOL_FILES = 1024
MAX_MEMBER_SPOOL_FILES = 512
_READ_CHUNK_BYTES = 64 * 1024


class RoomAttachmentSpoolError(ValueError):
    """Base error for a rejected target-side attachment batch."""


class RoomAttachmentSpoolConflict(RoomAttachmentSpoolError):
    """A durable identity was reused with different content."""


class RoomAttachmentSpoolIncomplete(RoomAttachmentSpoolError):
    """A run was presented before its complete attachment batch."""


def roomlink_attachments_available() -> bool:
    """Advertise the byte transport independently of optional PDF rendering."""

    return web is not None


def _batch_key(dispatch: HostedMemberDispatch) -> str:
    identity = "\0".join(
        (
            dispatch.home_install_id,
            dispatch.room_id,
            dispatch.authority_gateway_id,
            str(dispatch.authority_epoch),
            dispatch.member_id,
            dispatch.target_install_id,
            dispatch.target_profile,
            dispatch.task_id,
            str(dispatch.execution_generation),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _attempt_scope(dispatch: HostedMemberDispatch) -> tuple[Any, ...]:
    return (
        dispatch.room_id,
        dispatch.home_install_id,
        dispatch.authority_gateway_id,
        dispatch.authority_epoch,
        dispatch.member_id,
        dispatch.target_install_id,
        dispatch.target_profile,
        dispatch.task_id,
    )


class RoomAttachmentSpool:
    """Private atomic target spool keyed to one exact peer-run attempt."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        root: Path | str | None = None,
        clock=time.time,
    ) -> None:
        self.db_path = Path(db_path)
        self.root = Path(root or self.db_path.parent / "roomlink-attachment-spool")
        self.clock = clock
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        with self._transaction(immediate=True):
            pass
        self.prune()

    def _connect(self) -> sqlite3.Connection:
        from hermes_state_wal import apply_wal_with_fallback

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(conn, db_label="state.db (RoomLink attachment spool)")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS roomlink_attachment_batches (
                batch_key TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                home_install_id TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                member_id TEXT NOT NULL,
                target_install_id TEXT NOT NULL,
                target_profile TEXT NOT NULL,
                task_id TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                manifest_digest TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0, 1)),
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS roomlink_attachment_files (
                batch_key TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                stored INTEGER NOT NULL DEFAULT 0 CHECK (stored IN (0, 1)),
                PRIMARY KEY (batch_key, attachment_id),
                FOREIGN KEY (batch_key) REFERENCES roomlink_attachment_batches(batch_key)
                    ON DELETE CASCADE
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS roomlink_attachment_attempt_fences (
                room_id TEXT NOT NULL,
                home_install_id TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                member_id TEXT NOT NULL,
                target_install_id TEXT NOT NULL,
                target_profile TEXT NOT NULL,
                task_id TEXT NOT NULL,
                max_generation INTEGER NOT NULL CHECK (max_generation >= 1),
                expires_at REAL NOT NULL,
                PRIMARY KEY (
                    room_id, home_install_id, authority_gateway_id,
                    authority_epoch, member_id, target_install_id,
                    target_profile, task_id
                )
            )"""
        )
        batch_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(roomlink_attachment_batches)")
        }
        if "authority_gateway_id" not in batch_columns:
            conn.execute(
                """ALTER TABLE roomlink_attachment_batches
                   ADD COLUMN authority_gateway_id TEXT NOT NULL DEFAULT 'legacy'"""
            )
        if "authority_epoch" not in batch_columns:
            conn.execute(
                """ALTER TABLE roomlink_attachment_batches
                   ADD COLUMN authority_epoch INTEGER NOT NULL DEFAULT 0"""
            )
        if "complete" not in batch_columns:
            conn.execute(
                """ALTER TABLE roomlink_attachment_batches
                   ADD COLUMN complete INTEGER NOT NULL DEFAULT 0
                   CHECK (complete IN (0, 1))"""
            )
            conn.execute(
                """UPDATE roomlink_attachment_batches AS batch
                      SET complete=1
                    WHERE NOT EXISTS (
                        SELECT 1 FROM roomlink_attachment_files AS file
                         WHERE file.batch_key=batch.batch_key AND file.stored=0
                )"""
            )
        conn.execute(
            """INSERT INTO roomlink_attachment_attempt_fences(
                   room_id, home_install_id, authority_gateway_id,
                   authority_epoch, member_id, target_install_id,
                   target_profile, task_id, max_generation, expires_at
               )
               SELECT room_id, home_install_id, authority_gateway_id,
                      authority_epoch, member_id, target_install_id,
                      target_profile, task_id, MAX(execution_generation),
                      MAX(expires_at)
                 FROM roomlink_attachment_batches
                GROUP BY room_id, home_install_id, authority_gateway_id,
                         authority_epoch, member_id, target_install_id,
                         target_profile, task_id
               ON CONFLICT(
                   room_id, home_install_id, authority_gateway_id,
                   authority_epoch, member_id, target_install_id,
                   target_profile, task_id
               ) DO UPDATE SET
                   max_generation=MAX(
                       roomlink_attachment_attempt_fences.max_generation,
                       excluded.max_generation
                   ),
                   expires_at=MAX(
                       roomlink_attachment_attempt_fences.expires_at,
                       excluded.expires_at
                   )"""
        )
        conn.commit()
        return conn

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
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

    def _file_path(self, batch_key: str, attachment_id: str) -> Path:
        token = hashlib.sha256(f"{batch_key}\0{attachment_id}".encode()).hexdigest()
        return self.root / token

    @staticmethod
    def _read_verified(path: Path, *, size: int, digest: str) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RoomAttachmentSpoolIncomplete(
                "staged attachment bytes are unavailable"
            ) from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise RoomAttachmentSpoolIncomplete(
                    "staged attachment size no longer matches"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(MAX_ROOM_LINK_ATTACHMENT_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise RoomAttachmentSpoolIncomplete(
                "staged attachment failed SHA-256 validation"
            )
        return data

    def _write_atomic(self, target: Path, data: bytes) -> None:
        temp = self.root / f".tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
        descriptor = None
        try:
            descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
            os.chmod(target, 0o600)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temp.unlink(missing_ok=True)

    @staticmethod
    def _count_rows(
        conn: sqlite3.Connection,
        *,
        table: str,
        room_id: str | None = None,
        member_id: str | None = None,
    ) -> int:
        joins = ""
        where: list[str] = []
        params: list[str] = []
        if table == "roomlink_attachment_files":
            joins = " JOIN roomlink_attachment_batches AS batch USING(batch_key)"
        if room_id is not None:
            where.append("batch.room_id=?" if joins else "room_id=?")
            params.append(room_id)
        if member_id is not None:
            where.append("batch.member_id=?" if joins else "member_id=?")
            params.append(member_id)
        predicate = f" WHERE {' AND '.join(where)}" if where else ""
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM {table}{joins}{predicate}",
                tuple(params),
            ).fetchone()[0]
        )

    def _enforce_registration_quotas(
        self,
        conn: sqlite3.Connection,
        dispatch: HostedMemberDispatch,
        *,
        incoming_files: int,
    ) -> None:
        axes = (
            (None, None, MAX_SPOOL_BATCHES, MAX_SPOOL_FILES, "gateway"),
            (
                dispatch.room_id,
                None,
                MAX_ROOM_SPOOL_BATCHES,
                MAX_ROOM_SPOOL_FILES,
                "room",
            ),
            (
                dispatch.room_id,
                dispatch.member_id,
                MAX_MEMBER_SPOOL_BATCHES,
                MAX_MEMBER_SPOOL_FILES,
                "member",
            ),
        )
        for room_id, member_id, max_batches, max_files, label in axes:
            batches = self._count_rows(
                conn,
                table="roomlink_attachment_batches",
                room_id=room_id,
                member_id=member_id,
            )
            files = self._count_rows(
                conn,
                table="roomlink_attachment_files",
                room_id=room_id,
                member_id=member_id,
            )
            if batches + 1 > max_batches or files + incoming_files > max_files:
                raise RoomAttachmentSpoolError(
                    f"RoomLink {label} attachment registration quota is full"
                )

    def prepare(
        self,
        dispatch: HostedMemberDispatch,
        manifest_value: Any,
    ) -> dict[str, Any]:
        manifest = canonical_attachment_manifest(manifest_value)
        digest = attachment_manifest_digest(manifest)
        if dispatch.attachment_manifest_digest is None:
            raise RoomAttachmentSpoolError(
                "attachment manifest is not bound to this dispatch"
            )
        if digest != dispatch.attachment_manifest_digest:
            raise RoomAttachmentSpoolConflict(
                "attachment manifest does not match the dispatch"
            )
        key = _batch_key(dispatch)
        encoded = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        now = float(self.clock())
        self.prune(now=now)
        retired: list[tuple[str, str]] = []
        with self._lock, self._transaction(immediate=True) as conn:
            scope = _attempt_scope(dispatch)
            existing = conn.execute(
                "SELECT * FROM roomlink_attachment_batches WHERE batch_key=?",
                (key,),
            ).fetchone()
            fence = conn.execute(
                """SELECT max_generation FROM roomlink_attachment_attempt_fences
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?
                      AND task_id=?""",
                scope,
            ).fetchone()
            if fence is not None:
                highest = int(fence["max_generation"])
                if dispatch.execution_generation < highest:
                    raise RoomAttachmentSpoolConflict(
                        "attachment attempt was superseded by a later generation"
                    )
                if dispatch.execution_generation == highest and existing is None:
                    raise RoomAttachmentSpoolConflict(
                        "attachment attempt was already retired"
                    )
            conn.execute(
                """INSERT INTO roomlink_attachment_attempt_fences(
                       room_id, home_install_id, authority_gateway_id,
                       authority_epoch, member_id, target_install_id,
                       target_profile, task_id, max_generation, expires_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(
                       room_id, home_install_id, authority_gateway_id,
                       authority_epoch, member_id, target_install_id,
                       target_profile, task_id
                   ) DO UPDATE SET
                       max_generation=MAX(
                           roomlink_attachment_attempt_fences.max_generation,
                           excluded.max_generation
                       ),
                       expires_at=MAX(
                           roomlink_attachment_attempt_fences.expires_at,
                           excluded.expires_at
                       )""",
                (*scope, dispatch.execution_generation, now + SPOOL_TTL_SECONDS),
            )
            superseded = conn.execute(
                """SELECT batch_key FROM roomlink_attachment_batches
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?
                      AND task_id=? AND execution_generation<?""",
                (
                    dispatch.room_id,
                    dispatch.home_install_id,
                    dispatch.authority_gateway_id,
                    dispatch.authority_epoch,
                    dispatch.member_id,
                    dispatch.target_install_id,
                    dispatch.target_profile,
                    dispatch.task_id,
                    dispatch.execution_generation,
                ),
            ).fetchall()
            for row in superseded:
                old_key = str(row["batch_key"])
                retired.extend(
                    (old_key, str(file_row["attachment_id"]))
                    for file_row in conn.execute(
                        """SELECT attachment_id FROM roomlink_attachment_files
                            WHERE batch_key=?""",
                        (old_key,),
                    ).fetchall()
                )
                conn.execute(
                    "DELETE FROM roomlink_attachment_batches WHERE batch_key=?",
                    (old_key,),
                )
            if existing is not None:
                if (
                    existing["manifest_digest"] != digest
                    or existing["manifest_json"] != encoded
                    or existing["authority_gateway_id"]
                    != dispatch.authority_gateway_id
                    or int(existing["authority_epoch"])
                    != dispatch.authority_epoch
                ):
                    raise RoomAttachmentSpoolConflict(
                        "attachment batch identity was reused with a different manifest"
                    )
                result = {
                    "batch_key": key,
                    "manifest_digest": digest,
                    "complete": self._is_complete(conn, key),
                    "idempotent": True,
                }
            else:
                self._enforce_registration_quotas(
                    conn,
                    dispatch,
                    incoming_files=len(manifest),
                )
                conn.execute(
                    """INSERT INTO roomlink_attachment_batches(
                           batch_key, room_id, home_install_id,
                           authority_gateway_id, authority_epoch, member_id,
                           target_install_id, target_profile, task_id,
                           execution_generation, manifest_digest, manifest_json,
                           created_at, expires_at
                       ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        dispatch.room_id,
                        dispatch.home_install_id,
                        dispatch.authority_gateway_id,
                        dispatch.authority_epoch,
                        dispatch.member_id,
                        dispatch.target_install_id,
                        dispatch.target_profile,
                        dispatch.task_id,
                        dispatch.execution_generation,
                        digest,
                        encoded,
                        now,
                        now + SPOOL_TTL_SECONDS,
                    ),
                )
                conn.executemany(
                    """INSERT INTO roomlink_attachment_files(
                           batch_key, attachment_id, sha256, size, stored
                       ) VALUES(?, ?, ?, ?, 0)""",
                    [
                        (key, item["attachment_id"], item["sha256"], item["size"])
                        for item in manifest
                    ],
                )
                result = {
                    "batch_key": key,
                    "manifest_digest": digest,
                    "complete": False,
                    "idempotent": False,
                }
        for old_key, attachment_id in retired:
            self._file_path(old_key, attachment_id).unlink(missing_ok=True)
        return result

    def put(
        self,
        *,
        claims: Mapping[str, Any],
        task_id: str,
        execution_generation: int,
        attachment_id: str,
        data: bytes,
    ) -> dict[str, Any]:
        if not data or len(data) > MAX_ROOM_LINK_ATTACHMENT_BYTES:
            raise RoomAttachmentSpoolError(
                "attachment bytes are outside the RoomLink limit"
            )
        now = float(self.clock())
        self.prune(now=now)
        with self._lock, self._transaction(immediate=True) as conn:
            batch = conn.execute(
                """SELECT * FROM roomlink_attachment_batches
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?
                      AND task_id=? AND execution_generation=?""",
                (
                    claims["room_id"],
                    claims["home_install_id"],
                    claims["authority_gateway_id"],
                    claims["authority_epoch"],
                    claims["member_id"],
                    claims["target_install_id"],
                    claims["target_profile"],
                    task_id,
                    execution_generation,
                ),
            ).fetchone()
            if batch is None or float(batch["expires_at"]) <= now:
                raise RoomAttachmentSpoolError("attachment batch is unavailable")
            entry = conn.execute(
                """SELECT * FROM roomlink_attachment_files
                    WHERE batch_key=? AND attachment_id=?""",
                (batch["batch_key"], attachment_id),
            ).fetchone()
            if entry is None:
                raise RoomAttachmentSpoolError(
                    "attachment is not part of the registered manifest"
                )
            digest = hashlib.sha256(data).hexdigest()
            if len(data) != int(entry["size"]) or digest != entry["sha256"]:
                raise RoomAttachmentSpoolConflict(
                    "attachment bytes do not match the registered manifest"
                )
            path = self._file_path(str(batch["batch_key"]), attachment_id)
            if int(entry["stored"]):
                repaired = False
                try:
                    self._read_verified(path, size=len(data), digest=digest)
                except RoomAttachmentSpoolIncomplete:
                    # SQLite may have committed immediately before a crash or
                    # the private spool file may have been lost. The caller
                    # just retransmitted bytes matching the durable manifest,
                    # so repair atomically instead of poisoning the idempotent
                    # batch until expiry.
                    self._write_atomic(path, data)
                    repaired = True
                complete = self._mark_complete_if_ready(
                    conn, str(batch["batch_key"])
                )
                return {
                    "complete": complete,
                    "idempotent": True,
                    **({"repaired": True} if repaired else {}),
                }
            used = int(
                conn.execute(
                    """SELECT COALESCE(SUM(size), 0) AS bytes
                         FROM roomlink_attachment_files WHERE stored=1"""
                ).fetchone()["bytes"]
            )
            if used + len(data) > MAX_SPOOL_BYTES:
                raise RoomAttachmentSpoolError("RoomLink attachment spool is full")
            room_used = int(
                conn.execute(
                    """SELECT COALESCE(SUM(file.size), 0) AS bytes
                         FROM roomlink_attachment_files AS file
                         JOIN roomlink_attachment_batches AS batch USING(batch_key)
                        WHERE file.stored=1 AND batch.room_id=?""",
                    (claims["room_id"],),
                ).fetchone()["bytes"]
            )
            if room_used + len(data) > MAX_ROOM_SPOOL_BYTES:
                raise RoomAttachmentSpoolError("RoomLink room attachment quota is full")
            member_used = int(
                conn.execute(
                    """SELECT COALESCE(SUM(file.size), 0) AS bytes
                         FROM roomlink_attachment_files AS file
                         JOIN roomlink_attachment_batches AS batch USING(batch_key)
                        WHERE file.stored=1 AND batch.room_id=? AND batch.member_id=?""",
                    (claims["room_id"], claims["member_id"]),
                ).fetchone()["bytes"]
            )
            if member_used + len(data) > MAX_MEMBER_SPOOL_BYTES:
                raise RoomAttachmentSpoolError("RoomLink member attachment quota is full")
            if path.exists():
                self._read_verified(path, size=len(data), digest=digest)
            else:
                self._write_atomic(path, data)
            conn.execute(
                """UPDATE roomlink_attachment_files SET stored=1
                    WHERE batch_key=? AND attachment_id=?""",
                (batch["batch_key"], attachment_id),
            )
            complete = self._mark_complete_if_ready(
                conn, str(batch["batch_key"])
            )
        return {"complete": complete, "idempotent": False}

    @staticmethod
    def _is_complete(conn: sqlite3.Connection, key: str) -> bool:
        row = conn.execute(
            "SELECT complete FROM roomlink_attachment_batches WHERE batch_key=?",
            (key,),
        ).fetchone()
        return row is not None and bool(row["complete"])

    def _mark_complete_if_ready(
        self, conn: sqlite3.Connection, key: str
    ) -> bool:
        if self._is_complete(conn, key):
            return True
        missing = conn.execute(
            """SELECT 1 FROM roomlink_attachment_files
                WHERE batch_key=? AND stored=0 LIMIT 1""",
            (key,),
        ).fetchone()
        if missing is not None:
            return False
        changed = conn.execute(
            """UPDATE roomlink_attachment_batches SET complete=1
                WHERE batch_key=? AND complete=0""",
            (key,),
        )
        return changed.rowcount == 1 or self._is_complete(conn, key)

    def _complete_manifest(
        self,
        dispatch: HostedMemberDispatch,
        *,
        verify_bytes: bool,
    ) -> list[dict[str, Any]]:
        if dispatch.attachment_manifest_digest is None:
            return []
        key = _batch_key(dispatch)
        now = float(self.clock())
        self.prune(now=now)
        with self._transaction() as conn:
            batch = conn.execute(
                "SELECT * FROM roomlink_attachment_batches WHERE batch_key=?",
                (key,),
            ).fetchone()
            if (
                batch is None
                or float(batch["expires_at"]) <= now
                or batch["manifest_digest"] != dispatch.attachment_manifest_digest
                or batch["authority_gateway_id"]
                != dispatch.authority_gateway_id
                or int(batch["authority_epoch"]) != dispatch.authority_epoch
            ):
                raise RoomAttachmentSpoolIncomplete(
                    "complete RoomLink attachments have not reached this gateway"
                )
            manifest = canonical_attachment_manifest(
                json.loads(str(batch["manifest_json"]))
            )
            if not self._is_complete(conn, key):
                raise RoomAttachmentSpoolIncomplete(
                    "complete RoomLink attachments have not reached this gateway"
                )
        if verify_bytes:
            for item in manifest:
                self._read_verified(
                    self._file_path(key, item["attachment_id"]),
                    size=item["size"],
                    digest=item["sha256"],
                )
        return manifest

    def require_complete(self, dispatch: HostedMemberDispatch) -> list[dict[str, Any]]:
        """Require a complete batch and verify every committed byte once."""

        return self._complete_manifest(dispatch, verify_bytes=True)

    def materialize(self, dispatch: HostedMemberDispatch) -> list[dict[str, Any]]:
        """Return verified target-local paths in the manifest's original order."""

        manifest = self._complete_manifest(dispatch, verify_bytes=False)
        if not manifest:
            return []
        key = _batch_key(dispatch)
        materialized = []
        for item in manifest:
            path = self._file_path(key, item["attachment_id"])
            self._read_verified(
                path,
                size=item["size"],
                digest=item["sha256"],
            )
            materialized.append({**item, "path": str(path)})
        return materialized

    def discard_scope(self, claims: Mapping[str, Any]) -> int:
        """Remove every staged batch owned by one revoked room-member grant."""

        removed: list[tuple[str, str]] = []
        with self._lock, self._transaction(immediate=True) as conn:
            rows = conn.execute(
                """SELECT batch_key FROM roomlink_attachment_batches
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?""",
                (
                    claims["room_id"],
                    claims["home_install_id"],
                    claims["authority_gateway_id"],
                    claims["authority_epoch"],
                    claims["member_id"],
                    claims["target_install_id"],
                    claims["target_profile"],
                ),
            ).fetchall()
            keys = [str(row["batch_key"]) for row in rows]
            for key in keys:
                removed.extend(
                    (key, str(row["attachment_id"]))
                    for row in conn.execute(
                        "SELECT attachment_id FROM roomlink_attachment_files WHERE batch_key=?",
                        (key,),
                    ).fetchall()
                )
                conn.execute(
                    "DELETE FROM roomlink_attachment_batches WHERE batch_key=?",
                    (key,),
                )
            conn.execute(
                """DELETE FROM roomlink_attachment_attempt_fences
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?""",
                (
                    claims["room_id"],
                    claims["home_install_id"],
                    claims["authority_gateway_id"],
                    claims["authority_epoch"],
                    claims["member_id"],
                    claims["target_install_id"],
                    claims["target_profile"],
                ),
            )
        for key, attachment_id in removed:
            self._file_path(key, attachment_id).unlink(missing_ok=True)
        return len(keys)

    def discard_attempt(
        self,
        *,
        claims: Mapping[str, Any],
        task_id: str,
        execution_generation: int,
    ) -> int:
        """Delete one exact terminal run's private attachment batch idempotently."""

        removed: list[tuple[str, str]] = []
        with self._lock, self._transaction(immediate=True) as conn:
            rows = conn.execute(
                """SELECT batch_key FROM roomlink_attachment_batches
                    WHERE room_id=? AND home_install_id=?
                      AND authority_gateway_id=? AND authority_epoch=?
                      AND member_id=?
                      AND target_install_id=? AND target_profile=?
                      AND task_id=? AND execution_generation=?""",
                (
                    claims["room_id"],
                    claims["home_install_id"],
                    claims["authority_gateway_id"],
                    claims["authority_epoch"],
                    claims["member_id"],
                    claims["target_install_id"],
                    claims["target_profile"],
                    str(task_id),
                    int(execution_generation),
                ),
            ).fetchall()
            keys = [str(row["batch_key"]) for row in rows]
            for key in keys:
                removed.extend(
                    (key, str(row["attachment_id"]))
                    for row in conn.execute(
                        "SELECT attachment_id FROM roomlink_attachment_files WHERE batch_key=?",
                        (key,),
                    ).fetchall()
                )
                conn.execute(
                    "DELETE FROM roomlink_attachment_batches WHERE batch_key=?",
                    (key,),
                )
        for key, attachment_id in removed:
            self._file_path(key, attachment_id).unlink(missing_ok=True)
        return len(keys)

    def prune(self, *, now: float | None = None) -> int:
        checked_now = float(self.clock()) if now is None else float(now)
        quarantined: list[Path] = []
        with self._lock:
            try:
                candidates = tuple(self.root.iterdir())
            except OSError:
                candidates = ()
            old_temps: set[Path] = set()
            regular_files: set[Path] = set()
            for candidate in candidates:
                try:
                    if candidate.is_file():
                        regular_files.add(candidate)
                    if (
                        candidate.name.startswith(".tmp-")
                        and candidate.stat().st_mtime <= checked_now - 60
                    ):
                        old_temps.add(candidate)
                except OSError:
                    continue

            with self._transaction(immediate=True) as conn:
                changed = conn.execute(
                    "DELETE FROM roomlink_attachment_batches WHERE expires_at<=?",
                    (checked_now,),
                ).rowcount
                conn.execute(
                    """DELETE FROM roomlink_attachment_attempt_fences
                        WHERE expires_at<=?""",
                    (checked_now,),
                )
                retained_paths = {
                    self._file_path(
                        str(row["batch_key"]),
                        str(row["attachment_id"]),
                    )
                    for row in conn.execute(
                        "SELECT batch_key, attachment_id FROM roomlink_attachment_files"
                    ).fetchall()
                }
                for index, candidate in enumerate(candidates):
                    is_orphan = (
                        candidate.name.startswith(".prune-")
                        or candidate in old_temps
                        or (
                            candidate in regular_files
                            and not candidate.name.startswith(".tmp-")
                            and candidate not in retained_paths
                        )
                    )
                    if not is_orphan:
                        continue
                    quarantine = self.root / (
                        f".prune-{os.getpid()}-{threading.get_ident()}-"
                        f"{time.time_ns()}-{index}"
                    )
                    try:
                        os.replace(candidate, quarantine)
                    except OSError:
                        continue
                    quarantined.append(quarantine)
            # Atomic renames above fence cross-process writers with SQLite.
            # Potentially slow unlink work happens only after the transaction.
            for candidate in quarantined:
                candidate.unlink(missing_ok=True)
        return int(changed)


@lru_cache(maxsize=16)
def _spool(db_path: str) -> RoomAttachmentSpool:
    return RoomAttachmentSpool(Path(db_path))


def _default_spool() -> RoomAttachmentSpool:
    from gateway import hosted_rooms

    return _spool(str(hosted_rooms.default_db_path()))


def _http_routes(self) -> list[tuple[str, str, Any]]:
    async def manifest(request):
        from gateway.platforms import api_server

        return await _handle_room_attachment_manifest(
            self,
            request,
            _openai_error=api_server._openai_error,
            _api_request_profile=api_server._api_request_profile,
        )

    async def upload(request):
        from gateway.platforms import api_server

        return await _handle_room_attachment_upload(
            self,
            request,
            _openai_error=api_server._openai_error,
            _api_request_profile=api_server._api_request_profile,
        )

    async def discard(request):
        from gateway.platforms import api_server

        return await _handle_room_attachment_discard(
            self,
            request,
            _openai_error=api_server._openai_error,
            _api_request_profile=api_server._api_request_profile,
        )

    return [
        (
            "POST",
            "/v1/room-members/attachments",
            manifest,
        ),
        (
            "PUT",
            "/v1/room-members/attachments/{task_id}/{execution_generation}/{attachment_id}",
            upload,
        ),
        (
            "DELETE",
            "/v1/room-members/attachments/{task_id}/{execution_generation}",
            discard,
        ),
    ]


def _validate_target_scope(claims: Mapping[str, Any], profile: str) -> None:
    from gateway import hosted_rooms

    if (
        claims["target_profile"] != profile
        or claims["target_install_id"] != hosted_rooms.local_authority_gateway_id()
    ):
        raise RoomAttachmentSpoolError(
            "room grant target does not match this profile"
        )


async def _handle_room_attachment_manifest(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    body, error = await self._read_json_body(request)
    if error:
        return error
    if not isinstance(body, Mapping) or set(body) != {
        "hosted_room_dispatch",
        "attachments",
    }:
        return web.json_response(
            _openai_error(
                "Attachment staging requires a dispatch and manifest.",
                code="invalid_room_attachments",
            ),
            status=400,
        )
    try:
        dispatch = HostedMemberDispatch.from_mapping(body["hosted_room_dispatch"])
        claims = self._room_grant_claims(
            request,
            permission="attachment.stage",
        )
        verified = verify_room_grant(
            self._room_grant_secret(),
            self._room_grant_token(request),
            dispatch,
            permission="attachment.stage",
        )
        if verified["grant_id"] != claims["grant_id"]:
            raise RoomAttachmentSpoolError("room grant verification changed")
        _validate_target_scope(claims, _effective_room_profile(_api_request_profile))
        manifest = canonical_attachment_manifest(body["attachments"])
        if (
            any(item["kind"] == "pdf" for item in manifest)
            and shutil.which("pdftoppm") is None
        ):
            raise RoomAttachmentSpoolError(
                "This gateway cannot receive PDFs until Poppler is installed."
            )
        result = await asyncio.to_thread(
            _default_spool().prepare,
            dispatch,
            manifest,
        )
    except RoomAttachmentSpoolConflict as exc:
        return web.json_response(
            _openai_error(str(exc), code="room_attachment_conflict"),
            status=409,
        )
    except (HostedRoomGrantError, RoomGrantReauthorizationRequired) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_grant"),
            status=401,
        )
    except (RoomAttachmentSpoolError, ValueError) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_attachments"),
            status=400,
        )
    except Exception:
        logger.exception("RoomLink attachment manifest staging failed")
        return web.json_response(
            _openai_error(
                "The Group Chat files could not be staged on this gateway.",
                code="room_attachments_unavailable",
            ),
            status=500,
        )
    return web.json_response(
        {"object": "hermes.room_attachment_batch", **result},
        status=200 if result["idempotent"] else 201,
    )


async def _handle_room_attachment_upload(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    try:
        claims = self._room_grant_claims(request, permission="attachment.stage")
        _validate_target_scope(claims, _effective_room_profile(_api_request_profile))
        task_id = str(request.match_info["task_id"])
        generation = int(request.match_info["execution_generation"])
        if generation < 1:
            raise RoomAttachmentSpoolError("execution_generation is invalid")
        attachment_id = str(request.match_info["attachment_id"])
        content_length = request.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_ROOM_LINK_ATTACHMENT_BYTES:
            return web.json_response(
                _openai_error(
                    "Attachment exceeds the RoomLink transfer limit.",
                    code="room_attachment_too_large",
                ),
                status=413,
            )
        data = bytearray()
        async for chunk in request.content.iter_chunked(_READ_CHUNK_BYTES):
            data.extend(chunk)
            if len(data) > MAX_ROOM_LINK_ATTACHMENT_BYTES:
                return web.json_response(
                    _openai_error(
                        "Attachment exceeds the RoomLink transfer limit.",
                        code="room_attachment_too_large",
                    ),
                    status=413,
                )
        result = await asyncio.to_thread(
            _default_spool().put,
            claims=claims,
            task_id=task_id,
            execution_generation=generation,
            attachment_id=attachment_id,
            data=bytes(data),
        )
    except RoomAttachmentSpoolConflict as exc:
        return web.json_response(
            _openai_error(str(exc), code="room_attachment_conflict"),
            status=409,
        )
    except (HostedRoomGrantError, RoomGrantReauthorizationRequired) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_grant"),
            status=401,
        )
    except (RoomAttachmentSpoolError, ValueError) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_attachments"),
            status=400,
        )
    except Exception:
        logger.exception("RoomLink attachment upload failed")
        return web.json_response(
            _openai_error(
                "The Group Chat file could not be stored on this gateway.",
                code="room_attachments_unavailable",
            ),
            status=500,
        )
    return web.json_response(
        {"object": "hermes.room_attachment", **result},
        status=200 if result["idempotent"] else 201,
    )


async def _handle_room_attachment_discard(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Retire one exact terminal run's private target-side bytes."""

    try:
        # Status authorization intentionally outlives dispatch/staging so long
        # runs can retire their exact private batch after the short write grant
        # expires.
        claims = self._room_grant_claims(request, permission="status")
        _validate_target_scope(claims, _effective_room_profile(_api_request_profile))
        generation = int(request.match_info["execution_generation"])
        if generation < 1:
            raise RoomAttachmentSpoolError("execution_generation is invalid")
        removed = await asyncio.to_thread(
            _default_spool().discard_attempt,
            claims=claims,
            task_id=str(request.match_info["task_id"]),
            execution_generation=generation,
        )
    except (HostedRoomGrantError, RoomGrantReauthorizationRequired) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_grant"),
            status=401,
        )
    except (RoomAttachmentSpoolError, ValueError) as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_attachments"),
            status=400,
        )
    except Exception:
        logger.exception("RoomLink attachment cleanup failed")
        return web.json_response(
            _openai_error(
                "The Group Chat files could not be retired on this gateway.",
                code="room_attachments_unavailable",
            ),
            status=500,
        )
    return web.json_response(
        {"object": "hermes.room_attachment_retirement", "removed": removed},
        status=200,
    )


async def _validate_dispatch_attachments(
    normalized: Any,
    *,
    _openai_error,
) -> tuple[Any, "web.Response | None"]:
    if not isinstance(normalized, Mapping):
        return normalized, None
    raw_dispatch = normalized.get("hosted_room_dispatch")
    if not isinstance(raw_dispatch, Mapping):
        return normalized, None
    try:
        dispatch = HostedMemberDispatch.from_mapping(raw_dispatch)
        if dispatch.attachment_manifest_digest is not None:
            attachments = await asyncio.to_thread(
                _default_spool().materialize,
                dispatch,
            )
            image_paths = [
                item["path"] for item in attachments if item["kind"] == "image"
            ]
            file_rows = [
                item for item in attachments if item["kind"] in {"pdf", "file"}
            ]
            prompt = dispatch.prompt
            if attachments:
                labels = ", ".join(item["name"] for item in attachments)
                prompt = f"{prompt}\n\nAttached to this Group Chat message: {labels}."
                normalized = {
                    **normalized,
                    "_room_persist_user_message": (
                        f"{dispatch.prompt}\n\n[Group Chat files: {labels}]"
                    ),
                }
            if file_rows:
                prompt += (
                    "\nUse the file tools to inspect these target-local files:\n"
                    + "\n".join(
                        f"- {item['name']}: {item['path']}" for item in file_rows
                    )
                )
            if image_paths:
                from agent.image_routing import build_native_content_parts

                content, skipped = await asyncio.to_thread(
                    build_native_content_parts,
                    prompt,
                    image_paths,
                )
                if skipped:
                    raise RoomAttachmentSpoolIncomplete(
                        "a staged RoomLink image is no longer readable"
                    )
                normalized = {
                    **normalized,
                    "input": [{"role": "user", "content": content}],
                }
            else:
                normalized = {**normalized, "input": prompt}
    except RoomAttachmentSpoolIncomplete as exc:
        return normalized, web.json_response(
            _openai_error(str(exc), code="room_attachments_incomplete"),
            status=409,
        )
    except Exception:
        return normalized, web.json_response(
            _openai_error(
                "The Group Chat files could not be prepared on this gateway.",
                code="room_attachments_unavailable",
            ),
            status=409,
        )
    return normalized, None
