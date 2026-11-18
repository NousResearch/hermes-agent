"""Bounded, read-only snapshots of published hosted-room versions in SQLite."""

from contextlib import closing
from functools import partial
from pathlib import Path
import json
import sqlite3
import stat
import time
from typing import Any, Mapping

from persistence.core.errors import StoreUnavailable
from persistence.gateway.hosted_rooms import AttachmentError
from session_rules import hosted_attachment_catalog as catalog
from session_rules.hosted_attachments import _BLOB_ID_RE
from ._hosted_rooms_common import table_exists
from .hosted_room_attachments import _read_committed_row, _require_viewer_room

SNAPSHOT_SECONDS = 2.0


def _snapshot(db_path: Path) -> sqlite3.Connection:
    """A read-only, time-bounded snapshot: no write reservation, no schema work."""
    conn = sqlite3.connect(
        Path(db_path).resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=SNAPSHOT_SECONDS,
    )
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + SNAPSHOT_SECONDS
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    conn.execute("BEGIN")
    return conn


def _servable(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    event_id: str,
    entry: Mapping[str, Any],
    now: float,
    blob_root: Path,
) -> bool:
    """The download's viewer check plus blob metadata and file presence, without reading bytes."""
    try:
        row = _read_committed_row(
            conn,
            room_id=room_id,
            attachment_id=entry["attachment_id"],
            recipient_member_id="",
            normalized_event=event_id,
            viewer=True,
            reconciled=False,
            now=now,
        )
    except (AttachmentError, TypeError, ValueError):
        return False
    blob = conn.execute(
        "SELECT size, sha256 FROM hosted_room_attachment_blobs WHERE blob_id=?",
        (row["blob_id"],),
    ).fetchone()
    if (
        blob is None
        or blob["size"] != row["size"]
        or blob["sha256"] != row["sha256"]
        or any(row[field] != entry[field] for field in catalog.MANIFEST_FIELDS)
        or _BLOB_ID_RE.fullmatch(str(row["blob_id"])) is None
    ):
        return False
    try:
        info = (blob_root / row["blob_id"]).lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == row["size"]


def _availability(conn, *, room_id, event_id, entry, now, blob_root) -> bool | None:
    """False is known published metadata without local bytes; None is conflicting/private state."""
    if not table_exists(conn, "hosted_room_attachments"):
        return False
    row = conn.execute(
        "SELECT * FROM hosted_room_attachments WHERE attachment_id=?",
        (entry["attachment_id"],),
    ).fetchone()
    if row is None:
        return False
    if not catalog.public_row(row, room_id=room_id, event_id=event_id, entry=entry):
        return None
    return _servable(
        conn,
        room_id=room_id,
        event_id=event_id,
        entry=entry,
        now=now,
        blob_root=blob_root,
    )


def _file_events(conn, *, room_id, snapshot, position):
    """Seek one bounded page of log rows; sparse searches continue on the next page too."""
    bound = (
        ("<=", snapshot)
        if position is None
        else ("<=" if position["index"] is not None else "<", position["seq"])
    )
    return conn.execute(
        f"""SELECT seq, event_id, kind, actor_json, created_at,
                   CASE WHEN json_valid(payload_json) AND json_type(payload_json, '$.attachments')='array'
                        THEN json_extract(payload_json, '$.attachments') END AS manifest_json
              FROM hosted_room_events
             WHERE room_id=? AND seq {bound[0]} ?
             ORDER BY seq DESC LIMIT ?""",
        (room_id, bound[1], catalog.EVENT_SCAN_LIMIT + 1),
    ).fetchall()


def list_published(
    files, placement, *, room_id, authority_gateway_id, authority_epoch, **filters
):
    scope = dict(
        room_id=room_id,
        authority_gateway_id=authority_gateway_id,
        authority_epoch=authority_epoch,
    )
    with closing(_snapshot(files.path(placement))) as conn:
        _require_viewer_room(conn, **scope)
        room = conn.execute(
            "SELECT next_seq, members_json FROM hosted_rooms WHERE room_id=?",
            (room_id,),
        ).fetchone()
        return catalog.list_published(
            room,
            partial(_file_events, conn, room_id=room_id),
            partial(
                _availability,
                conn,
                room_id=room_id,
                now=time.time(),
                blob_root=files.attachment_root(placement) / "blobs",
            ),
            **scope,
            **filters,
        )


def published_reference(
    files,
    placement,
    *,
    room_id,
    event_id,
    attachment_id,
    authority_gateway_id,
    authority_epoch,
):
    with closing(_snapshot(files.path(placement))) as conn:
        _require_viewer_room(
            conn,
            room_id=room_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
        )
        event = conn.execute(
            "SELECT kind, actor_json, payload_json FROM hosted_room_events WHERE room_id=? AND event_id=?",
            (room_id, event_id),
        ).fetchone()
        availability = partial(
            _availability,
            conn,
            room_id=room_id,
            event_id=event_id,
            now=time.time(),
            blob_root=files.attachment_root(placement) / "blobs",
        )
        published = catalog.published_reference(event, attachment_id, availability)
        if published and not table_exists(conn, "hosted_room_attachments"):
            # The room log proves visibility, but a missing store cannot attest a permanent byte loss.
            raise StoreUnavailable("room attachment storage is unavailable")
        return published
