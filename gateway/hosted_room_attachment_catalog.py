"""Read-only Files catalog for a gateway-owned Group Chat.

Lists the file versions a viewer can fetch with ``groups.attachment.download``,
newest share first, from one read-only SQLite snapshot. Every listed version
passes the store's own viewer check (committed, published by its message,
viewer-visible, unexpired) in that snapshot, so the catalog never offers a file
the download would refuse. Nothing is written and blob bytes are never read.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from gateway.hosted_room_attachments import (
    AttachmentError,
    HostedRoomAttachmentStore,
    fold_catalog_text,
    validate_manifest,
)
from gateway.hosted_rooms import HostedRoomError
from gateway.hosted_rooms_common import identifier, table_exists

DEFAULT_LIMIT = 8
MAX_LIMIT = 32
MAX_QUERY_CHARS = 255
MAX_CURSOR_CHARS = 4096
# File-bearing messages examined per page; a sparse page returns a cursor to continue.
EVENT_SCAN_LIMIT = 256
SNAPSHOT_SECONDS = 2.0
_SHARER_KINDS = {"message.user": "user", "message.member": "member"}
_MANIFEST_FIELDS = ("attachment_id", "kind", "name", "size", "mime")
_CURSOR_FIELDS = frozenset({"v", "scope", "snapshot", "seq", "index"})


class CatalogCursorError(HostedRoomError):
    """The cursor does not continue this listing; restart from the newest files."""

    reason = "attachment_cursor_invalid"


def _limit(value: Any) -> int:
    if value is None:
        return DEFAULT_LIMIT
    if type(value) is not int or not 1 <= value <= MAX_LIMIT:
        raise AttachmentError(f"attachment list limit must be an integer from 1 to {MAX_LIMIT}")
    return value


def _query(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_QUERY_CHARS:
        raise AttachmentError(f"attachment query must be a string of at most {MAX_QUERY_CHARS} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AttachmentError("attachment query must be valid Unicode") from None
    return fold_catalog_text(value.strip())


def _producer_filter(value: Any) -> str:
    if value is None or value == "":
        return ""
    return identifier(value, label="producer_member_id", error=AttachmentError)


def _snapshot(db_path: Path) -> sqlite3.Connection:
    """A read-only, time-bounded snapshot: no write reservation, no schema work."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=SNAPSHOT_SECONDS)
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + SNAPSHOT_SECONDS
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    conn.execute("BEGIN")
    return conn


def _roster_labels(members_json: Any) -> dict[str, str]:
    try:
        members = json.loads(members_json)
    except (TypeError, ValueError):
        return {}
    labels = {}
    for member in members if isinstance(members, list) else ():
        member_id = member.get("member_id") if isinstance(member, Mapping) else None
        if isinstance(member_id, str):
            label = member.get("display_name") or member.get("handle") or member.get("profile")
            labels[member_id] = label.strip() if isinstance(label, str) and label.strip() else member_id
    return labels


def _sharer(event_kind: Any, actor_json: Any, roster: Mapping[str, str]) -> dict[str, str] | None:
    """Who shared a message's files, as the room log records it; None if unattributable."""
    kind = _SHARER_KINDS.get(event_kind)
    try:
        actor = json.loads(actor_json)
    except (TypeError, ValueError):
        return None
    if kind is None or not isinstance(actor, Mapping) or actor.get("kind") != kind:
        return None
    sharer_id = actor.get("id")
    if not isinstance(sharer_id, str) or not sharer_id:
        return None
    label = actor.get("display_name")
    label = label.strip() if isinstance(label, str) else ""
    return {"kind": kind, "id": sharer_id,
            "label": label or ("You" if kind == "user" else roster.get(sharer_id, sharer_id))}


def _matches(name: str, sharer: Mapping[str, str], query: str, producer: str) -> bool:
    if producer and sharer["id"] != producer:
        return False
    return not query or query in fold_catalog_text(name) or query in fold_catalog_text(sharer["label"])


def _scope(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _encode_cursor(**fields: Any) -> str:
    raw = json.dumps({"v": 1, **fields}, separators=(",", ":"), sort_keys=True).encode("ascii")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_cursor(value: Any, *, scope: str, latest_seq: int) -> dict[str, Any]:
    if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_CHARS:
        raise CatalogCursorError("attachment list cursor is invalid")
    try:
        cursor = json.loads(base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True))
    except (binascii.Error, ValueError):
        raise CatalogCursorError("attachment list cursor is invalid") from None
    if (not isinstance(cursor, dict) or set(cursor) != _CURSOR_FIELDS or cursor["v"] != 1
            or cursor["scope"] != scope
            or type(cursor["snapshot"]) is not int or not 0 <= cursor["snapshot"] <= latest_seq
            or type(cursor["seq"]) is not int or not 1 <= cursor["seq"] <= cursor["snapshot"]
            or (cursor["index"] is not None and (type(cursor["index"]) is not int or cursor["index"] < 0))):
        raise CatalogCursorError("attachment list cursor does not match this request")
    return cursor


def _servable(conn: sqlite3.Connection, *, room_id: str, event_id: str, entry: Mapping[str, Any],
              now: float) -> bool:
    """The download's own viewer check, plus intact blob metadata, in this snapshot."""
    try:
        row = HostedRoomAttachmentStore._read_committed_row(
            conn, room_id=room_id, attachment_id=entry["attachment_id"], recipient_member_id="",
            normalized_event=event_id, viewer=True, now=now)
    except (AttachmentError, TypeError, ValueError):
        return False
    blob = conn.execute("SELECT size, sha256 FROM hosted_room_attachment_blobs WHERE blob_id=?",
                        (row["blob_id"],)).fetchone()
    return (blob is not None and blob["size"] == row["size"] and blob["sha256"] == row["sha256"]
            and all(row[field] == entry[field] for field in _MANIFEST_FIELDS))


def list_published(
    db_path: Path | str,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    cursor: Any = None,
    limit: Any = None,
    query: Any = None,
    producer_member_id: Any = None,
) -> dict[str, Any]:
    """One page of downloadable file versions, newest share first.

    ``query`` matches file names and sharer labels, ignoring case and accents;
    ``producer_member_id`` keeps one sharer's files. A cursor continues the
    same listing from its first page's snapshot, so files shared meanwhile never
    shift pages, and it is refused for any other room, authority or filter.
    """
    limit, query, producer = _limit(limit), _query(query), _producer_filter(producer_member_id)
    scope = _scope(room_id, authority_gateway_id, authority_epoch, query, producer)
    now = time.time()
    with closing(_snapshot(Path(db_path))) as conn:
        HostedRoomAttachmentStore._require_viewer_room(
            conn, room_id=room_id, authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch)
        room = conn.execute("SELECT next_seq, members_json FROM hosted_rooms WHERE room_id=?",
                            (room_id,)).fetchone()
        latest = int(room["next_seq"]) - 1
        position = None if cursor is None else _decode_cursor(cursor, scope=scope, latest_seq=latest)
        snapshot = latest if position is None else position["snapshot"]

        def page(items: list[dict[str, Any]], seq: int | None = None, index: int | None = None) -> dict[str, Any]:
            next_cursor = None if seq is None else _encode_cursor(
                scope=scope, snapshot=snapshot, seq=seq, index=index)
            return {"room_id": room_id, "authority": {"gateway_id": authority_gateway_id, "epoch": authority_epoch},
                    "snapshot_seq": snapshot, "items": items, "next_cursor": next_cursor,
                    "has_more": next_cursor is not None}

        if not table_exists(conn, "hosted_room_attachments"):
            return page([])
        roster = _roster_labels(room["members_json"])
        # Search and the sharer filter run in SQL so a rare match deep in history needs one page.
        # CROSS JOIN keeps the room's published files as the outer loop: the work is bounded by
        # the files in the room, not by the length of its message log.
        filters = ""
        if query or producer:
            sharers: dict[tuple[Any, Any], dict[str, str] | None] = {}

            def match(name: Any, kind: Any, actor_json: Any) -> int:
                if (kind, actor_json) not in sharers:
                    sharers[kind, actor_json] = _sharer(kind, actor_json, roster)
                sharer = sharers[kind, actor_json]
                return int(sharer is not None and isinstance(name, str) and _matches(name, sharer, query, producer))

            conn.create_function("catalog_match", 3, match, deterministic=True)
            filters = "AND catalog_match(attachment.name, event.kind, event.actor_json)"
        if position is None:
            bound, after = ("<=", snapshot), None
        else:
            bound = ("<=" if position["index"] is not None else "<", position["seq"])
            after = (position["seq"], position["index"])
        events = conn.execute(
            f"""WITH shares AS (
                    SELECT DISTINCT event.seq AS seq
                      FROM hosted_room_attachments AS attachment
                     CROSS JOIN hosted_room_events AS event
                        ON event.room_id=attachment.room_id AND event.event_id=attachment.event_id
                     WHERE attachment.room_id=? AND attachment.state='committed' AND attachment.viewer_access=1
                       AND (attachment.expires_at IS NULL OR attachment.expires_at>?)
                       AND event.kind IN ('message.user', 'message.member') AND event.seq {bound[0]} ? {filters}
                     ORDER BY event.seq DESC LIMIT ?)
                SELECT event.seq, event.event_id, event.kind, event.actor_json, event.created_at,
                       CASE WHEN json_valid(event.payload_json)
                             AND json_type(event.payload_json, '$.attachments')='array'
                            THEN json_extract(event.payload_json, '$.attachments') END AS manifest_json
                  FROM shares CROSS JOIN hosted_room_events AS event ON event.room_id=? AND event.seq=shares.seq
                 ORDER BY event.seq DESC""",
            (room_id, now, bound[1], EVENT_SCAN_LIMIT + 1, room_id)).fetchall()

        items: list[dict[str, Any]] = []
        for event in events[:EVENT_SCAN_LIMIT]:
            seq, event_id = int(event["seq"]), str(event["event_id"])
            sharer = _sharer(event["kind"], event["actor_json"], roster)
            try:
                raw = json.loads(event["manifest_json"])
                manifest = validate_manifest(raw)
            except (TypeError, ValueError):
                continue
            if sharer is None or manifest != raw:
                continue
            for index, entry in enumerate(manifest):
                if after is not None and seq == after[0] and after[1] is not None and index <= after[1]:
                    continue
                if not _matches(entry["name"], sharer, query, producer):
                    continue
                if not _servable(conn, room_id=room_id, event_id=event_id, entry=entry, now=now):
                    continue
                if len(items) == limit:
                    return page(items, items[-1]["seq"], items[-1]["manifest_index"])
                items.append({**{field: entry[field] for field in _MANIFEST_FIELDS},
                              "event_id": event_id, "seq": seq, "manifest_index": index,
                              "producer": sharer, "shared_at": float(event["created_at"])})
        if len(events) > EVENT_SCAN_LIMIT:
            return page(items, int(events[EVENT_SCAN_LIMIT - 1]["seq"]))
        return page(items)
