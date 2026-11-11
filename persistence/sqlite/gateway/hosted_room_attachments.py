"""``HostedRoomAttachmentRepository`` on SQLite: room attachment rows and their content-deduplicated blob files
in the hosted-room tables of either file.

Blob bytes are files under ``<dir of the file>/hosted-room-attachments/blobs/`` (``HostedFiles.attachment_root``). A blob is written or verified inside the write transaction that records it, so a committed row always names a verified file; released blobs are unlinked after the commit, and the orphan sweep renames ``.tmp-`` staging files, ``.prune-`` quarantine files and unreferenced ``blob_*`` files inside its own write and unlinks them after it. The first attachment verb on a file in a process first runs the file's setup: the attachment tables in their own ``BEGIN IMMEDIATE``, the reconciliation of commitments their durable events name, then the prune with its sweep. ``put`` prunes again before it stores, so an expired upload never holds quota. On a read-only handle the read verbs skip that setup and only read: a file without the attachment tables holds no attachment, a missing blob reads as on a writable handle, a commitment its durable event names is never read as expired (the reconciliation's rule, applied on the read), and tables older than this version are refused, never migrated. The quotas are the ``session_rules.hosted_attachments`` constants.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from hermes_dir_acl import make_user_dir
from persistence import JsonValue, PermanentStoreError
from persistence.gateway.hosted_rooms import (
    AttachmentConflictError,
    AttachmentData,
    AttachmentError,
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    EventAttachmentConflictError,
    HostedPlacement,
)
from session_rules.hosted_attachments import (
    DISBANDED_GRACE_SECONDS, MAX_ATTACHMENT_BYTES, MAX_GATEWAY_ATTACHMENT_COUNT, MAX_GATEWAY_BLOB_BYTES,
    MAX_ROOM_ATTACHMENT_BYTES, MAX_ROOM_ATTACHMENT_COUNT, MAX_ROOM_UNCOMMITTED_BYTES, MAX_ROOM_UNCOMMITTED_COUNT,
    UNCOMMITTED_TTL_SECONDS, _BLOB_ID_RE, fold_catalog_text, validate_manifest,
)
from session_rules.hosted_rooms import clock

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_common import table_columns, table_exists
from ._hosted_store import HostedFiles

__all__ = ["SQLiteHostedRoomAttachmentRepository", "retain_message_attachments"]

_T = TypeVar("_T")

# (state.db, blob root) file identities whose reconciliation and prune already ran in this process.
_STARTED: set[tuple[str, str, int, int] | None] = set()
_STARTED_LOCK = threading.Lock()


# ── schema ────────────────────────────────────────────────────────────────────


# The columns ``_initialize`` adds to older attachment tables.
_CURRENT_COLUMNS = frozenset({"viewer_access", "catalog_name"})


# The attachment and blob tables and their indexes; older tables gain ``viewer_access`` and a filled ``catalog_name``.
def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_attachment_blobs (
                blob_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL CHECK (size > 0),
                ref_count INTEGER NOT NULL CHECK (ref_count > 0),
                created_at REAL NOT NULL
            )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_attachments (
                attachment_id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                event_id TEXT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                size INTEGER NOT NULL CHECK (size > 0),
                mime TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                blob_id TEXT NOT NULL,
                recipient_member_ids_json TEXT NOT NULL DEFAULT '[]',
                viewer_access INTEGER NOT NULL DEFAULT 0 CHECK (viewer_access IN (0, 1)),
                state TEXT NOT NULL CHECK (state IN ('uploaded', 'committed', 'disbanded')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                UNIQUE (room_id, upload_id),
                FOREIGN KEY (blob_id) REFERENCES hosted_room_attachment_blobs(blob_id)
            )"""
    )
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(hosted_room_attachments)")
    }
    if "viewer_access" not in columns:
        conn.execute(
            """ALTER TABLE hosted_room_attachments
                   ADD COLUMN viewer_access INTEGER NOT NULL DEFAULT 0"""
        )
    if "catalog_name" not in columns:
        conn.execute("ALTER TABLE hosted_room_attachments ADD COLUMN catalog_name TEXT")
        reader = conn.execute("SELECT attachment_id, name FROM hosted_room_attachments")
        while rows := reader.fetchmany(512):
            conn.executemany(
                "UPDATE hosted_room_attachments SET catalog_name=? WHERE attachment_id=?",
                ((fold_catalog_text(str(row["name"])), row["attachment_id"]) for row in rows),
            )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_room_state
               ON hosted_room_attachments(room_id, state, created_at)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_expiry
               ON hosted_room_attachments(expires_at)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_event
               ON hosted_room_attachments(room_id, event_id)"""
    )


# ── blob files ────────────────────────────────────────────────────────────────


def _blob_path(blob_root: Path, blob_id: str) -> Path:
    if _BLOB_ID_RE.fullmatch(blob_id) is None:
        raise AttachmentIntegrityError("stored blob id is invalid")
    return blob_root / blob_id


def _write_blob(target: Path, data: bytes) -> None:
    from utils import atomic_write_bytes
    # ``.tmp-`` staging names are what the orphan sweep reclaims after a crash.
    atomic_write_bytes(target, data, tmp_prefix=".tmp-", mode=0o600, fsync_dir=True)


def _open_blob(blob_root: Path, *, blob_id: str, size: int) -> int:
    """Open the regular blob file and fence its size against the durable row."""
    path = _blob_path(blob_root, blob_id)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AttachmentIntegrityError("canonical attachment blob is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise AttachmentIntegrityError("canonical attachment blob size changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_blob(blob_root: Path, *, blob_id: str, size: int, sha256: str) -> bytes:
    descriptor = _open_blob(blob_root, blob_id=blob_id, size=size)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(MAX_ATTACHMENT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise AttachmentIntegrityError("canonical attachment blob failed SHA-256 validation")
    return data


def _read_blob_range(blob_root: Path, *, blob_id: str, size: int, offset: int, length: int) -> bytes:
    """One slice of the blob; the row's SHA-256 is the caller's whole-file digest and the
    receiver verifies the reassembled bytes against it, so no slice is hashed here."""
    descriptor = _open_blob(blob_root, blob_id=blob_id, size=size)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            handle.seek(offset)
            data = handle.read(length)
    finally:
        os.close(descriptor)
    if len(data) != min(length, size - offset):
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


# ── row steps ─────────────────────────────────────────────────────────────────


# The metadata dict of an attachment row: its identity, content facts and state, ``event_id`` only once one is bound.
def _metadata(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    result = {
        "attachment_id": str(row["attachment_id"]),
        "kind": str(row["kind"]),
        "name": str(row["name"]),
        "size": int(row["size"]),
        "mime": str(row["mime"]),
        "sha256": str(row["sha256"]),
        "state": str(row["state"]),
        "created_at": float(row["created_at"]),
        "idempotent": idempotent,
    }
    if row["event_id"] is not None:
        result["event_id"] = str(row["event_id"])
    return result


# Refuses a viewer read unless the room is live under the caller's authority and neither quarantined nor fenced.
def _require_viewer_room(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
) -> None:
    if not table_exists(conn, "hosted_rooms"):
        raise AttachmentNotFoundError("Group Chat is unavailable to viewers")
    room = conn.execute(
        """SELECT authority_gateway_id, authority_epoch, disbanded_at
           FROM hosted_rooms WHERE room_id=?""",
        (room_id,),
    ).fetchone()
    if (
        room is None
        or room["disbanded_at"] is not None
        or room["authority_gateway_id"] != authority_gateway_id
        or room["authority_epoch"] != authority_epoch
    ):
        raise AttachmentNotFoundError("Group Chat viewer authority changed")
    # Files-only sources need not install the optional safety/retirement schema.
    for table in ("hosted_room_quarantine", "hosted_room_disband_fences"):
        if table_exists(conn, table) and conn.execute(
            f"SELECT 1 FROM {table} WHERE room_id=?", (room_id,),
        ).fetchone() is not None:
            raise AttachmentNotFoundError("Group Chat is unavailable to viewers")


# The committed, unexpired row this recipient or viewer may read; a room-visible file must be published in its event.
def _read_committed_row(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    attachment_id: str,
    recipient_member_id: str,
    normalized_event: str | None,
    viewer: bool,
    now: float,
    reconciled: bool = True,
) -> sqlite3.Row:
    # A read-only handle never creates the tables, so a file without them holds no attachment.
    row = conn.execute(
        "SELECT * FROM hosted_room_attachments WHERE attachment_id=? AND room_id=?",
        (attachment_id, room_id),
    ).fetchone() if table_exists(conn, "hosted_room_attachments") else None
    if row is None or str(row["state"]) != "committed":
        raise AttachmentNotFoundError("attachment is not committed for this room")
    # A read-only handle never reconciles, so it applies the reconciliation's rule here.
    if row["expires_at"] is not None and float(row["expires_at"]) <= now and (reconciled or not _named_by_owner_event(conn, row)):
        raise AttachmentNotFoundError("attachment has expired")
    if normalized_event is not None and str(row["event_id"] or "") != normalized_event:
        raise AttachmentNotFoundError("attachment is not owned by this room event")
    if int(row["viewer_access"] or 0):
        # Room-visible files need a published owner even for member reads.
        # A pre-event commitment is staging, not authority to expose bytes.
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_events'"
            ).fetchone()
            is None
        ):
            raise AttachmentNotFoundError(
                "attachment is not published in this room"
            )
        owner = conn.execute(
            """SELECT payload_json FROM hosted_room_events
                   WHERE room_id=? AND event_id=? AND kind IN ('message.user', 'message.member')""",
            (room_id, str(row["event_id"] or "")),
        ).fetchone()
        manifest = {
            key: row[key]
            for key in ("attachment_id", "kind", "name", "size", "mime")
        }
        payload = (
            json.loads(owner["payload_json"]) if owner is not None else None
        )
        published = (
            payload.get("attachments") if isinstance(payload, Mapping) else None
        )
        if not isinstance(published, list) or manifest not in published:
            raise AttachmentNotFoundError(
                "attachment is not published in this room"
            )
    recipients = json.loads(str(row["recipient_member_ids_json"]))
    if viewer:
        if int(row["viewer_access"] or 0) != 1:
            raise AttachmentNotFoundError(
                "attachment is unavailable to Group Chat viewers"
            )
    elif recipient_member_id not in recipients:
        raise AttachmentNotFoundError("attachment is unavailable to this recipient")
    return row


# The event that owns an attachment, or None when the file has no room log or no such event.
def _owner_event(
    conn: sqlite3.Connection, room_id: str, event_id: str
) -> sqlite3.Row | None:
    # The standalone private-transport store need not have a canonical room log.
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_events'"
        ).fetchone()
        is None
    ):
        return None
    return conn.execute(
        "SELECT payload_json FROM hosted_room_events WHERE room_id=? AND event_id=?",
        (room_id, event_id),
    ).fetchone()


# ``_reconcile_room_events``'s rule for one row: its durable event payload names it, so the commitment is retained.
def _named_by_owner_event(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    owner = _owner_event(conn, str(row["room_id"]), str(row["event_id"] or ""))
    if owner is None:
        return False
    try:
        payload = json.loads(str(owner["payload_json"]))
        return str(row["attachment_id"]) in {
            str(item.get("attachment_id") or "") for item in payload.get("attachments", []) if isinstance(item, Mapping)
        }
    # Malformed row: the reconciliation skips it too.
    except (ValueError, TypeError, AttributeError, RecursionError):
        return False


def retain_message_attachments(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    event_id: str,
    manifest: Any,
    now: float,
) -> None:
    """Validate and retain exact commitments inside the canonical append transaction."""
    normalized = validate_manifest(manifest)
    if not normalized:
        return
    if normalized != manifest:
        raise AttachmentError("published attachment metadata must be canonical")
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_attachments'"
        ).fetchone()
        is None
    ):
        raise EventAttachmentConflictError("message attachments are not committed")
    for entry in normalized:
        row = conn.execute(
            "SELECT * FROM hosted_room_attachments WHERE attachment_id=?",
            (entry["attachment_id"],),
        ).fetchone()
        if (
            row is None
            or row["room_id"] != room_id
            or row["event_id"] != event_id
            or row["state"] != "committed"
            or row["viewer_access"] != 1
            or (row["expires_at"] is not None and float(row["expires_at"]) <= now)
            or any(row[key] != value for key, value in entry.items())
        ):
            raise EventAttachmentConflictError(
                "attachment commitment changed before event publication"
            )
    # Rollback and expiry use the same SQL write lock. No byte or peer I/O is
    # needed here; import verified the bytes before staging these commitments.
    conn.executemany(
        """UPDATE hosted_room_attachments SET expires_at=NULL, updated_at=?
           WHERE attachment_id=?""",
        ((now, entry["attachment_id"]) for entry in normalized),
    )


class SQLiteHostedRoomAttachmentRepository:
    """Attachment rows and blob files on either hosted-room file."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._files = HostedFiles(store)

    def _blob_root(self, placement: HostedPlacement) -> Path:
        return self._files.attachment_root(placement) / "blobs"

    def _read_root(self, placement: HostedPlacement, now: float) -> Path:
        """The blob directory for a read verb: a writable handle runs ``_open`` first; a read-only handle only reads, so it creates, chmods, prunes and migrates nothing."""
        return self._blob_root(placement) if self._store.read_only else self._open(placement, now)

    def _read_rows(self, placement: HostedPlacement, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """``fn`` on a read connection of the placement's file; a read-only handle refuses attachment tables older than this version instead of migrating them."""
        if not self._store.read_only:
            return self._files._hosted_read(placement, fn)

        def _current(conn: sqlite3.Connection) -> _T:
            if table_exists(conn, "hosted_room_attachments") and not _CURRENT_COLUMNS <= table_columns(conn, "hosted_room_attachments"):
                raise PermanentStoreError(f"{self._files.path(placement)} is not set up for this version; a read-only store handle never migrates it")
            return fn(conn)
        return self._files._hosted_read(placement, _current)

    def _open(self, placement: HostedPlacement, now: float) -> Path:
        """Prepare the private root, run the first-touch schema, reconciliation and prune of this file once per process, and return the blob directory."""
        self._files.refuse_read_only()
        root = self._files.attachment_root(placement)
        blob_root = root / "blobs"
        make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
        make_user_dir(blob_root, parents=True, exist_ok=True, mode=0o700)
        for path in (root, blob_root):
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
        with _STARTED_LOCK:
            if self._startup_key(placement, root) in _STARTED:
                return blob_root
            self._files._hosted_write(placement, _initialize, immediate=True)
            self._reconcile_room_events(placement, now)
            self._prune(placement, now)
            _STARTED.add(self._startup_key(placement, root))
        return blob_root

    def _startup_key(self, placement: HostedPlacement, root: Path) -> tuple[str, str, int, int] | None:
        """Identity of the database FILE, so a recreated file is set up again."""
        db_path = self._files.path(placement)
        try:
            info = db_path.stat()
        except OSError:
            return None
        return str(db_path.resolve()), str(root.resolve()), info.st_dev, info.st_ino

    def _reconcile_room_events(self, placement: HostedPlacement, now: float) -> int:
        """Recover only attachment commitments named by the durable event payload."""
        def _txn(conn: sqlite3.Connection) -> int:
            changed = 0
            has_events = conn.execute(
                """SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='hosted_room_events'"""
            ).fetchone()
            if has_events is None:
                return 0
            rows = conn.execute(
                """SELECT attachment.attachment_id, attachment.event_id,
                          event.payload_json
                     FROM hosted_room_attachments AS attachment
                     JOIN hosted_room_events AS event
                       ON event.room_id=attachment.room_id
                      AND event.event_id=attachment.event_id
                    WHERE attachment.state='committed'
                      AND attachment.expires_at IS NOT NULL"""
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(str(row["payload_json"]))
                    ids = {
                        str(item.get("attachment_id") or "")
                        for item in payload.get("attachments", [])
                        if isinstance(item, Mapping)
                    }
                # Malformed row: bad/over-nested JSON, a non-object payload, or non-iterable attachments.
                except (ValueError, TypeError, AttributeError, RecursionError):
                    continue
                if str(row["attachment_id"]) not in ids:
                    continue
                changed += conn.execute(
                    """UPDATE hosted_room_attachments
                          SET expires_at=NULL, updated_at=?
                        WHERE attachment_id=? AND state='committed'""",
                    (now, str(row["attachment_id"])),
                ).rowcount
            return int(changed)
        return self._files._hosted_write(placement, _txn, immediate=True)

    def _prune(self, placement: HostedPlacement, now: float) -> int:
        """Internal step ``prune_expired_attachments``: mark disbanded rooms' rows, delete expired rows and their last
        blob references, then unlink the released blobs after the commit and sweep orphans."""
        def _txn(conn: sqlite3.Connection) -> tuple[int, list[str]]:
            removed_blob_ids: list[str] = []
            removed = 0
            has_rooms = conn.execute(
                """SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='hosted_rooms'"""
            ).fetchone()
            if has_rooms is not None:
                conn.execute(
                    """UPDATE hosted_room_attachments
                          SET state='disbanded',
                              expires_at=(
                                SELECT room.disbanded_at + ? FROM hosted_rooms AS room
                                 WHERE room.room_id=hosted_room_attachments.room_id
                              ),
                              updated_at=?
                        WHERE state='committed' AND EXISTS (
                            SELECT 1 FROM hosted_rooms AS room
                             WHERE room.room_id=hosted_room_attachments.room_id
                               AND room.disbanded_at IS NOT NULL
                        )""",
                    (DISBANDED_GRACE_SECONDS, now),
                )
            rows = conn.execute(
                """SELECT attachment_id, blob_id FROM hosted_room_attachments
                    WHERE expires_at IS NOT NULL AND expires_at <= ?""",
                (now,),
            ).fetchall()
            released = Counter(str(row["blob_id"]) for row in rows)
            for row in rows:
                conn.execute(
                    "DELETE FROM hosted_room_attachments WHERE attachment_id=?",
                    (str(row["attachment_id"]),),
                )
                removed += 1
            for blob_id, count in released.items():
                blob = conn.execute(
                    "SELECT ref_count FROM hosted_room_attachment_blobs WHERE blob_id=?",
                    (blob_id,),
                ).fetchone()
                if blob is None:
                    continue
                if int(blob["ref_count"]) <= count:
                    removed_blob_ids.append(blob_id)
                else:
                    conn.execute(
                        """UPDATE hosted_room_attachment_blobs
                              SET ref_count=ref_count-?
                            WHERE blob_id=?""",
                        (count, blob_id),
                    )
            if removed_blob_ids:
                conn.executemany(
                    "DELETE FROM hosted_room_attachment_blobs WHERE blob_id=?",
                    ((blob_id,) for blob_id in removed_blob_ids),
                )
            return removed, removed_blob_ids
        removed, removed_blob_ids = self._files._hosted_write(placement, _txn, immediate=True)
        blob_root = self._blob_root(placement)
        for blob_id in removed_blob_ids:
            _blob_path(blob_root, blob_id).unlink(missing_ok=True)
        self._sweep_orphans(placement)
        return removed

    # Moves stray staging, quarantine and unreferenced blob files aside inside a write, then unlinks them.
    def _sweep_orphans(self, placement: HostedPlacement) -> None:
        """Internal step ``sweep_orphan_blobs``."""
        blob_root = self._blob_root(placement)

        # The listing runs inside the write transaction, so a concurrent put's blob and row land
        # wholly before or wholly after it.
        def _txn(conn: sqlite3.Connection) -> list[Path]:
            try:
                candidates = tuple(blob_root.iterdir())
            except OSError:
                return []
            quarantined: list[Path] = []
            live = {
                str(row["blob_id"])
                for row in conn.execute(
                    "SELECT blob_id FROM hosted_room_attachment_blobs"
                )
            }
            for index, path in enumerate(candidates):
                if not (
                    path.name.startswith((".tmp-", ".prune-"))
                    or (
                        _BLOB_ID_RE.fullmatch(path.name) is not None
                        and path.name not in live
                    )
                ):
                    continue
                quarantine = blob_root / (
                    f".prune-{os.getpid()}-{threading.get_ident()}-"
                    f"{time.time_ns()}-{index}"
                )
                try:
                    os.replace(path, quarantine)
                except OSError:
                    continue
                quarantined.append(quarantine)
            return quarantined
        # Renaming while the write transaction is active is the atomic
        # reference fence; unlinking the private quarantine can be slow and
        # therefore happens only after SQLite has committed.
        for path in self._files._hosted_write(placement, _txn, immediate=True):
            path.unlink(missing_ok=True)

    # The file's first-touch setup on its own, for a caller that runs it before any other verb.
    def prepare(self, *, placement: HostedPlacement, now: float | None = None) -> None:
        with translated():
            self._open(placement, clock(now))

    # ── writes ──────────────────────────────────────────────────────────────

    # Stores an upload under the room and gateway quotas, sharing a blob with the same SHA-256; a repeated
    # ``upload_id`` with the same content returns the stored row.
    def put(self, *, placement: HostedPlacement, room_id: str, upload_id: str, kind: str, name: str, mime: str,
            data: bytes, now: float | None = None) -> Mapping[str, JsonValue]:
        digest = hashlib.sha256(data).hexdigest()
        now = clock(now)
        with translated():
            blob_root = self._open(placement, now)
            # Long-lived gateways keep one store instance. Reclaim abandoned
            # uploads before applying quotas so an expired failed send cannot
            # permanently exhaust the room without a process restart.
            self._prune(placement, now)

            def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
                existing = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE room_id=? AND upload_id=?",
                    (room_id, upload_id),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["kind"]) != kind
                        or str(existing["name"]) != name
                        or str(existing["mime"]) != mime
                        or int(existing["size"]) != len(data)
                        or str(existing["sha256"]) != digest
                    ):
                        raise AttachmentConflictError(
                            "upload_id was already used for different attachment content"
                        )
                    _read_blob(
                        blob_root,
                        blob_id=str(existing["blob_id"]),
                        size=int(existing["size"]),
                        sha256=str(existing["sha256"]),
                    )
                    return _metadata(existing, idempotent=True)

                room_totals = conn.execute(
                    """SELECT COALESCE(SUM(size), 0) AS bytes, COUNT(*) AS count
                         FROM hosted_room_attachments
                        WHERE room_id=? AND state!='disbanded'""",
                    (room_id,),
                ).fetchone()
                uncommitted = conn.execute(
                    """SELECT COALESCE(SUM(size), 0) AS bytes, COUNT(*) AS count
                         FROM hosted_room_attachments
                        WHERE room_id=? AND state='uploaded'""",
                    (room_id,),
                ).fetchone()
                if int(room_totals["bytes"]) + len(data) > MAX_ROOM_ATTACHMENT_BYTES:
                    raise AttachmentQuotaError("room attachment quota exceeded")
                if int(room_totals["count"]) >= MAX_ROOM_ATTACHMENT_COUNT:
                    raise AttachmentQuotaError("room attachment count quota exceeded")
                total_count = int(conn.execute("SELECT COUNT(*) FROM hosted_room_attachments").fetchone()[0])
                if total_count >= MAX_GATEWAY_ATTACHMENT_COUNT:
                    raise AttachmentQuotaError("gateway attachment count quota exceeded")
                if (
                    int(uncommitted["bytes"]) + len(data) > MAX_ROOM_UNCOMMITTED_BYTES
                    or int(uncommitted["count"]) + 1 > MAX_ROOM_UNCOMMITTED_COUNT
                ):
                    raise AttachmentQuotaError("room uncommitted upload quota exceeded")

                blob = conn.execute(
                    "SELECT * FROM hosted_room_attachment_blobs WHERE sha256=?",
                    (digest,),
                ).fetchone()
                if blob is None:
                    physical = int(
                        conn.execute(
                            "SELECT COALESCE(SUM(size), 0) FROM hosted_room_attachment_blobs"
                        ).fetchone()[0]
                    )
                    if physical + len(data) > MAX_GATEWAY_BLOB_BYTES:
                        raise AttachmentQuotaError("gateway attachment quota exceeded")
                    blob_id = f"blob_{secrets.token_hex(16)}"
                    target = _blob_path(blob_root, blob_id)
                    _write_blob(target, data)
                    conn.execute(
                        """INSERT INTO hosted_room_attachment_blobs
                           (blob_id, sha256, size, ref_count, created_at)
                           VALUES (?, ?, ?, 1, ?)""",
                        (blob_id, digest, len(data), now),
                    )
                else:
                    blob_id = str(blob["blob_id"])
                    _read_blob(blob_root, blob_id=blob_id, size=int(blob["size"]), sha256=digest)
                    conn.execute(
                        "UPDATE hosted_room_attachment_blobs SET ref_count=ref_count+1 WHERE blob_id=?",
                        (blob_id,),
                    )

                attachment_id = f"att_{secrets.token_hex(16)}"
                conn.execute(
                    """INSERT INTO hosted_room_attachments
                       (attachment_id, upload_id, room_id, event_id, kind, name, catalog_name, size,
                        mime, sha256, blob_id, recipient_member_ids_json, state,
                        created_at, updated_at, expires_at)
                       VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, '[]', 'uploaded', ?, ?, ?)""",
                    (
                        attachment_id,
                        upload_id,
                        room_id,
                        kind,
                        name,
                        fold_catalog_text(name),
                        len(data),
                        mime,
                        digest,
                        blob_id,
                        now,
                        now,
                        now + UNCOMMITTED_TTL_SECONDS,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE attachment_id=?",
                    (attachment_id,),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by insert
                    raise RuntimeError("stored attachment could not be reloaded")
                return _metadata(row)
            return self._files._hosted_write(placement, _txn, immediate=True)

    def commit_message_with_receipt(
        self, *, placement: HostedPlacement, room_id: str, event_id: str, manifest: Sequence[Mapping[str, JsonValue]],
        recipient_member_ids: Sequence[str], viewer_access: bool = False, retention_seconds: float | None = None,
        hold_until_event: bool = False, now: float | None = None,
    ) -> tuple[list[Mapping[str, JsonValue]], tuple[str, ...]]:
        """Commit a manifest and report only rows transitioned by this call."""
        normalized = [dict(entry) for entry in manifest]
        recipients_json = json.dumps(sorted(recipient_member_ids), separators=(",", ":"))
        now = clock(now)
        expires_at = (
            now + UNCOMMITTED_TTL_SECONDS
            if hold_until_event
            else None
            if retention_seconds is None
            else now + float(retention_seconds)
        )

        def _txn(conn: sqlite3.Connection) -> tuple[str, ...]:
            transitioned: list[str] = []
            for entry in normalized:
                row = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE attachment_id=?",
                    (entry["attachment_id"],),
                ).fetchone()
                if row is None or str(row["room_id"]) != room_id:
                    raise AttachmentNotFoundError("attachment is not uploaded for this room")
                durable = {
                    "attachment_id": str(row["attachment_id"]),
                    "kind": str(row["kind"]),
                    "name": str(row["name"]),
                    "size": int(row["size"]),
                    "mime": str(row["mime"]),
                }
                if durable != entry:
                    raise AttachmentConflictError(
                        "attachment manifest metadata does not match the uploaded bytes"
                    )
                if row["expires_at"] is not None and float(row["expires_at"]) <= now:
                    raise AttachmentNotFoundError("attachment has expired")
                state = str(row["state"])
                if state == "disbanded":
                    raise AttachmentNotFoundError("attachment belongs to a disbanded room")
                if state == "committed" and (
                    str(row["event_id"] or "") != event_id
                    or str(row["recipient_member_ids_json"]) != recipients_json
                    or bool(row["viewer_access"]) != bool(viewer_access)
                ):
                    raise AttachmentConflictError(
                        "attachment is already owned by a different room event"
                    )
                if state == "uploaded":
                    transitioned.append(entry["attachment_id"])
                _read_blob(
                    blob_root,
                    blob_id=str(row["blob_id"]),
                    size=int(row["size"]),
                    sha256=str(row["sha256"]),
                )
            for entry in normalized:
                if entry["attachment_id"] not in transitioned:
                    continue
                updated = conn.execute(
                    """UPDATE hosted_room_attachments
                          SET event_id=?, recipient_member_ids_json=?, viewer_access=?, state='committed',
                              updated_at=?, expires_at=?
                        WHERE attachment_id=? AND state='uploaded'""",
                    (
                        event_id,
                        recipients_json,
                        1 if viewer_access else 0,
                        now,
                        expires_at,
                        entry["attachment_id"],
                    ),
                )
                if updated.rowcount != 1:
                    raise AttachmentConflictError(
                        "attachment changed during message commitment"
                    )
            return tuple(transitioned)
        with translated():
            blob_root = self._open(placement, now)
            return normalized, self._files._hosted_write(placement, _txn, immediate=True)

    def abort_message_commit(self, *, placement: HostedPlacement, room_id: str, event_id: str,
                             attachment_ids: Sequence[str], now: float | None = None) -> int:
        """Return a failed pre-event commitment to its bounded upload state."""
        attachment_ids = list(attachment_ids)
        if not attachment_ids:
            return 0
        now = clock(now)

        def _txn(conn: sqlite3.Connection) -> int:
            ids = attachment_ids
            owner = _owner_event(conn, room_id, event_id)
            if owner is not None:
                published_ids = {
                    item.get("attachment_id")
                    for item in json.loads(owner["payload_json"]).get("attachments", [])
                    if isinstance(item, Mapping)
                }
                ids = [
                    value for value in ids if value not in published_ids
                ]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            changed = conn.execute(
                f"""UPDATE hosted_room_attachments
                        SET event_id=NULL, recipient_member_ids_json='[]', viewer_access=0,
                            state='uploaded', updated_at=?, expires_at=?
                      WHERE room_id=? AND event_id=? AND state='committed'
                        AND attachment_id IN ({placeholders})""",
                (
                    now,
                    now + UNCOMMITTED_TTL_SECONDS,
                    room_id,
                    event_id,
                    *ids,
                ),
            )
            return int(changed.rowcount)
        with translated():
            self._open(placement, now)
            return self._files._hosted_write(placement, _txn, immediate=True)

    # ── reads ───────────────────────────────────────────────────────────────

    # A committed attachment's metadata and verified bytes, for one recipient or for a viewer.
    def read(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str | None,
             event_id: str | None = None, viewer: bool = False, now: float | None = None) -> AttachmentData:
        now = clock(now)

        def _read(conn: sqlite3.Connection) -> sqlite3.Row:
            return _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id or "",
                normalized_event=event_id, viewer=viewer, now=now, reconciled=not self._store.read_only,
            )
        with translated():
            blob_root = self._read_root(placement, now)
            row = self._read_rows(placement, _read)
            data = _read_blob(
                blob_root,
                blob_id=str(row["blob_id"]),
                size=int(row["size"]),
                sha256=str(row["sha256"]),
            )
            return AttachmentData(_metadata(row), data)

    def describe(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str,
                 event_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """``read`` without bytes: the metadata (including the upload-verified SHA-256) of a
        committed attachment this recipient may read for this event."""
        now = clock(now)

        def _read(conn: sqlite3.Connection) -> dict[str, Any]:
            row = _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id,
                normalized_event=event_id, viewer=False, now=now, reconciled=not self._store.read_only,
            )
            return _metadata(row)
        with translated():
            self._read_root(placement, now)
            return self._read_rows(placement, _read)

    def read_range(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str,
                   event_id: str, offset: int, length: int, now: float | None = None) -> AttachmentData:
        """``read`` for one slice: same authorization, metadata carries the stored whole-file
        SHA-256 and the receiver verifies the reassembled bytes against it."""
        now = clock(now)

        def _read(conn: sqlite3.Connection) -> sqlite3.Row:
            row = _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id,
                normalized_event=event_id, viewer=False, now=now, reconciled=not self._store.read_only,
            )
            if offset >= int(row["size"]):
                raise AttachmentError("attachment range starts past the end")
            return row
        with translated():
            blob_root = self._read_root(placement, now)
            row = self._read_rows(placement, _read)
            data = _read_blob_range(
                blob_root, blob_id=str(row["blob_id"]), size=int(row["size"]), offset=offset, length=length)
            return AttachmentData(_metadata(row), data)

    def read_viewer(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, event_id: str | None = None,
                    recipient_member_id: str | None = None, authority_gateway_id: str, authority_epoch: int,
                    now: float | None = None) -> AttachmentData:
        """Read for a live hosted viewer, fencing revocation across byte I/O."""
        scope = {
            "room_id": room_id,
            "authority_gateway_id": authority_gateway_id,
            "authority_epoch": authority_epoch,
        }
        selection = {
            "room_id": room_id,
            "attachment_id": attachment_id,
            "recipient_member_id": recipient_member_id or "",
            "normalized_event": event_id,
            "viewer": True,
            "reconciled": not self._store.read_only,
        }

        # Each viewer read is one deferred read transaction: the room check and the row read see one snapshot.
        def _first(conn: sqlite3.Connection) -> sqlite3.Row:
            conn.execute("BEGIN")
            try:
                _require_viewer_room(conn, **scope)
                return _read_committed_row(conn, **selection, now=clock(now))
            finally:
                conn.rollback()

        def _second(conn: sqlite3.Connection) -> sqlite3.Row:
            conn.execute("BEGIN")
            try:
                _require_viewer_room(conn, **scope)
                current = _read_committed_row(conn, **selection, now=clock(now))
            finally:
                conn.rollback()
            if current["blob_id"] != row["blob_id"] or _metadata(current) != _metadata(row):
                raise AttachmentNotFoundError("attachment changed during viewer read")
            return current
        with translated():
            blob_root = self._read_root(placement, clock(now))
            row = self._read_rows(placement, _first)
            # No execution lock or SQLite snapshot spans filesystem/hash work.
            data = _read_blob(
                blob_root, blob_id=str(row["blob_id"]), size=int(row["size"]), sha256=str(row["sha256"]),
            )
            current = self._read_rows(placement, _second)
            return AttachmentData(_metadata(current), data)


register_repository("hosted_room_attachments", SQLiteHostedRoomAttachmentRepository)
