"""Metadata-only discovery over the canonical Group Chat attachment store."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from gateway.hosted_rooms import (
    HostedRoomError,
    MAX_ACTOR_LABEL_CHARS,
    MAX_EVENT_JSON_BYTES,
    MAX_MEMBERS,
    _validate_actor,
)
from gateway.hosted_room_attachments import (
    ATTACHMENT_LIST_EVENT_SCAN_LIMIT,
    MAX_ATTACHMENT_LIST_CURSOR_BYTES,
    MAX_ATTACHMENT_LIST_RESPONSE_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
    MAX_IDENTIFIER_CHARS,
    AttachmentCursorError,
    AttachmentError,
    AttachmentNotFoundError,
    _ATTACHMENT_ID_RE,
    _BLOB_ID_RE,
    _CURSOR_FIELDS,
    _CURSOR_RESET_MESSAGE,
    _SHA256_RE,
    _catalog_limit,
    _catalog_query,
    _identifier,
    fold_catalog_text,
    validate_manifest,
)


@contextmanager
def _catalog_snapshot(store):
    conn = sqlite3.connect(
        store.db_path.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=2.0,
    )
    conn.row_factory = sqlite3.Row

    @lru_cache(maxsize=256)
    def cached_fold(value):
        return fold_catalog_text(value)

    def catalog_fold(value):
        text = str(value or "")
        return cached_fold(text) if len(text) <= 512 else fold_catalog_text(text)

    conn.create_function(
        "catalog_fold",
        1,
        catalog_fold,
        deterministic=True,
    )
    deadline = time.monotonic() + 2.0
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _encode_list_cursor(self, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _decode_list_cursor(self, value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise AttachmentCursorError("attachment list cursor must be a non-empty string")
    if len(value) > MAX_ATTACHMENT_LIST_CURSOR_BYTES:
        raise AttachmentCursorError("attachment list cursor is too large")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise AttachmentCursorError(_CURSOR_RESET_MESSAGE) from None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AttachmentCursorError(_CURSOR_RESET_MESSAGE) from None
    if not isinstance(payload, dict) or frozenset(payload) != _CURSOR_FIELDS:
        raise AttachmentCursorError(_CURSOR_RESET_MESSAGE)
    return payload


def _event_epoch_is_visible(value: Any, authority_epoch: int) -> bool:
    return value is None or (type(value) is int and 1 <= value <= authority_epoch)


def _producer_label(
    kind: Any, producer_id: Any, display_name: Any, room_members: Mapping[str, str]
) -> str:
    label = display_name.strip() if isinstance(display_name, str) else ""
    identity = producer_id if isinstance(producer_id, str) else ""
    return label or (
        "You" if kind == "user" else room_members.get(identity, identity)
    )


def _matches_filters(
    item: Mapping[str, Any], folded_query: str, producer_member_id: str
) -> bool:
    return (
        not producer_member_id or item["producer"]["id"] == producer_member_id
    ) and (
        not folded_query or any(
            folded_query in fold_catalog_text(value)
            for value in (item["name"], item["producer"]["label"])
        )
    )


def _published_event_attachments(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    event: sqlite3.Row,
    room_members: Mapping[str, str],
    now: float,
    recipient_member_id: str = "",
) -> list[dict[str, Any]]:
    if str(event["kind"]) not in {"message.user", "message.member"}:
        return []
    candidate = conn.execute(
        """SELECT 1 FROM hosted_room_attachments
            WHERE room_id=? AND event_id=? LIMIT 1""",
        (room_id, str(event["event_id"])),
    ).fetchone()
    if candidate is None:
        return []
    owner = conn.execute(
        """SELECT actor_json,
                  json_extract(CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,
                               '$.attachments') AS attachments_json,
                  json_type(CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,
                            '$.attachments') AS attachments_type,
                  json_extract(CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,
                               '$.member_id') AS payload_member_id,
                  json_type(CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,
                            '$.member_id') AS payload_member_type
             FROM hosted_room_events
            WHERE room_id=? AND seq=? AND event_id=?
              AND length(CAST(payload_json AS BLOB))<=?
              AND length(CAST(actor_json AS BLOB))<=4096
              AND NOT EXISTS (
                  SELECT 1 FROM json_each(
                      CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END
                  ) GROUP BY key HAVING COUNT(*)>1
              )""",
        (room_id, int(event["seq"]), str(event["event_id"]), MAX_EVENT_JSON_BYTES),
    ).fetchone()
    if owner is None or owner["attachments_type"] != "array":
        return []
    try:
        actor = json.loads(str(owner["actor_json"]))
        if not isinstance(actor, Mapping):
            return []
        event_kind = str(event["kind"])
        _identifier(event["event_id"], label="event_id")
        normalized_actor, _encoded_actor = _validate_actor(actor, kind=event_kind)
        if normalized_actor != actor:
            return []
        actor_kind = str(actor.get("kind") or "")
        producer_id = _identifier(actor.get("id"), label="producer id")
        if event_kind == "message.user":
            if actor_kind != "user":
                return []
        elif event_kind == "message.member":
            if actor_kind != "member" or (
                owner["payload_member_type"] is not None
                and owner["payload_member_id"] != producer_id
            ):
                return []
        else:
            return []
        raw_manifest = json.loads(str(owner["attachments_json"]))
        manifest = validate_manifest(raw_manifest)
        if manifest != raw_manifest or not manifest:
            return []
        shared_at = float(event["created_at"])
        if not math.isfinite(shared_at):
            return []
    except (HostedRoomError, AttachmentError, TypeError, ValueError):
        return []

    # The manifest bounds this join even if abandoned commitments share the
    # event id. Eligibility filters must not turn LIMIT into an unbounded scan.
    placeholders = ",".join("?" for _ in manifest)
    rows = conn.execute(
        f"""SELECT attachment.*, blob.size AS blob_size,
                   blob.sha256 AS blob_sha256, owner.seq AS owner_seq
              FROM hosted_room_attachments AS attachment
              LEFT JOIN hosted_room_events AS owner
                ON owner.room_id=attachment.room_id
               AND owner.event_id=attachment.event_id
               AND owner.seq=?
              LEFT JOIN hosted_room_attachment_blobs AS blob
                ON blob.blob_id=attachment.blob_id
             WHERE attachment.attachment_id IN ({placeholders})""",
        (int(event["seq"]), *(item["attachment_id"] for item in manifest)),
    ).fetchall()
    by_id = {str(row["attachment_id"]): row for row in rows}
    label = _producer_label(
        actor_kind, producer_id, actor.get("display_name"), room_members
    )
    result: list[dict[str, Any]] = []
    for manifest_index, entry in enumerate(manifest):
        row = by_id.get(entry["attachment_id"])
        if row is None:
            continue
        durable = {
            key: row[key] for key in ("attachment_id", "kind", "name", "size", "mime")
        }
        try:
            recipients = json.loads(str(row["recipient_member_ids_json"]))
            valid_recipients = (
                isinstance(recipients, list)
                and 0 < len(recipients) <= MAX_MEMBERS
                and len(set(recipients)) == len(recipients)
                and all(
                    _identifier(value, label="recipient_member_id") == value
                    for value in recipients
                )
            )
            expires_at = (
                float(row["expires_at"]) if row["expires_at"] is not None else None
            )
            eligible = (
                durable == entry
                and valid_recipients
                and (not recipient_member_id or recipient_member_id in recipients)
                and row["room_id"] == room_id
                and row["event_id"] == event["event_id"]
                and row["owner_seq"] == event["seq"]
                and str(row["state"]) == "committed"
                and int(row["viewer_access"] or 0) == 1
                and (
                    expires_at is None
                    or (math.isfinite(expires_at) and expires_at > now)
                )
                and row["blob_size"] is not None
                and int(row["blob_size"]) == int(row["size"])
                and str(row["blob_sha256"] or "") == str(row["sha256"])
                and _BLOB_ID_RE.fullmatch(str(row["blob_id"])) is not None
                and _SHA256_RE.fullmatch(str(row["sha256"])) is not None
            )
        except (TypeError, ValueError):
            eligible = False
        if not eligible:
            continue
        result.append({
            "attachment_id": entry["attachment_id"],
            "event_id": str(event["event_id"]),
            "seq": int(event["seq"]),
            "manifest_index": manifest_index,
            "kind": entry["kind"],
            "name": entry["name"],
            "mime": entry["mime"],
            "size": entry["size"],
            "producer": {
                "kind": actor_kind,
                "id": producer_id,
                "label": label,
            },
            "shared_at": shared_at,
        })
    return result


def list_published(
    self,
    *,
    room_id: Any,
    authority_gateway_id: Any,
    authority_epoch: Any,
    cursor: Any = None,
    limit: Any = None,
    query: Any = None,
    producer_member_id: Any = None,
    recipient_member_id: Any = None,
) -> dict[str, Any]:
    """List canonical viewer-visible metadata within one read snapshot.

    Sparse history/search can return an empty page with a continuation.
    Cursors bind immutable share order and survive store/worker reopen.
    """

    room_id = _identifier(room_id, label="room_id")
    authority_gateway_id = _identifier(
        authority_gateway_id, label="authority_gateway_id"
    )
    if (
        isinstance(authority_epoch, bool)
        or not isinstance(authority_epoch, int)
        or authority_epoch < 1
    ):
        raise AttachmentError("authority_epoch must be a positive integer")
    limit = _catalog_limit(limit)
    folded_query = _catalog_query(query)
    if producer_member_id is None:
        producer_member_id = ""
    if not isinstance(producer_member_id, str):
        raise AttachmentError("producer_member_id must be a string")
    if len(producer_member_id) > MAX_IDENTIFIER_CHARS:
        raise AttachmentError("invalid producer_member_id")
    producer_member_id = producer_member_id.strip()
    if producer_member_id:
        producer_member_id = _identifier(producer_member_id, label="producer_member_id")
    recipient_member_id = (
        _identifier(recipient_member_id, label="recipient_member_id")
        if recipient_member_id is not None
        else ""
    )
    decoded_cursor = _decode_list_cursor(self, cursor) if cursor is not None else None
    now = float(self.clock())

    with _catalog_snapshot(self) as conn:
        self._require_viewer_room(
            conn,
            room_id=room_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
        )
        room = conn.execute(
            """SELECT room.authority_gateway_id, room.authority_epoch,
                      room.next_seq, room.members_json
                 FROM hosted_rooms AS room
                WHERE room.room_id=?""",
            (room_id,),
        ).fetchone()
        if room is None:
            raise AttachmentNotFoundError(
                "attachment catalogue is unavailable for this room authority"
            )
        snapshot_seq = int(room["next_seq"]) - 1
        try:
            raw_members = json.loads(str(room["members_json"]))
            if not isinstance(raw_members, list) or len(raw_members) > MAX_MEMBERS:
                raise ValueError
            room_members = {}
            for member in raw_members:
                if not isinstance(member, Mapping):
                    raise ValueError
                member_id = _identifier(member.get("member_id"), label="member_id")
                member_label = (
                    member.get("display_name")
                    or member.get("handle")
                    or member.get("profile")
                    or member_id
                )
                if (
                    not isinstance(member_label, str)
                    or len(member_label) > MAX_ACTOR_LABEL_CHARS
                ):
                    raise ValueError
                room_members[member_id] = member_label.strip()
        except (AttachmentError, TypeError, ValueError, json.JSONDecodeError):
            raise AttachmentNotFoundError(
                "attachment catalogue room metadata is invalid"
            ) from None

        last_seq: int | None = None
        last_attachment_id: str | None = None
        last_manifest_index: int | None = None
        if decoded_cursor is not None:
            if (
                decoded_cursor.get("version") != 2
                or decoded_cursor.get("room_id") != room_id
                or decoded_cursor.get("authority_gateway_id") != authority_gateway_id
                or type(decoded_cursor.get("authority_epoch")) is not int
                or decoded_cursor.get("authority_epoch") != authority_epoch
                or decoded_cursor.get("query_digest")
                != hashlib.sha256(folded_query.encode("utf-8")).hexdigest()
                or decoded_cursor.get("producer_member_id") != producer_member_id
                or decoded_cursor.get("recipient_member_id") != recipient_member_id
            ):
                raise AttachmentCursorError(
                    "attachment list cursor does not match this request"
                )
            cursor_snapshot = decoded_cursor.get("snapshot_seq")
            last_seq = decoded_cursor.get("last_seq")
            last_attachment_id = decoded_cursor.get("last_attachment_id")
            last_manifest_index = decoded_cursor.get("last_manifest_index")
            if (
                isinstance(cursor_snapshot, bool)
                or not isinstance(cursor_snapshot, int)
                or cursor_snapshot < 0
                or cursor_snapshot > snapshot_seq
                or isinstance(last_seq, bool)
                or not isinstance(last_seq, int)
                or not 1 <= last_seq <= cursor_snapshot
                or (
                    last_attachment_id is not None
                    and (
                        not isinstance(last_attachment_id, str)
                        or _ATTACHMENT_ID_RE.fullmatch(last_attachment_id) is None
                    )
                )
                or (last_attachment_id is None) != (last_manifest_index is None)
                or (
                    last_manifest_index is not None
                    and (
                        type(last_manifest_index) is not int
                        or not 0 <= last_manifest_index < MAX_ATTACHMENTS_PER_MESSAGE
                    )
                )
            ):
                raise AttachmentCursorError(_CURSOR_RESET_MESSAGE)
            snapshot_seq = cursor_snapshot

        conn.create_function(
            "catalog_sharer",
            3,
            lambda kind, identity, label: _producer_label(
                kind, identity, label, room_members
            ),
            deterministic=True,
        )
        actor_expr = (
            "CASE WHEN json_valid(event.actor_json) THEN event.actor_json ELSE '{}' END"
        )
        # Older folded caches can be lossy; the canonical name remains authoritative.
        search_where = f"""(?='' OR instr(catalog_fold(attachment.name), ?)>0
            OR instr(catalog_fold(catalog_sharer(
                json_extract({actor_expr}, '$.kind'),
                json_extract({actor_expr}, '$.id'),
                json_extract({actor_expr}, '$.display_name'))), ?)>0)
            AND (?='' OR json_extract({actor_expr}, '$.id')=?)"""
        search_params = (
            folded_query,
            folded_query,
            folded_query,
            producer_member_id,
            producer_member_id,
        )
        candidates = f"""WITH matching AS MATERIALIZED (
            SELECT attachment.event_id
              FROM hosted_room_attachments AS attachment
              JOIN hosted_room_events AS event
                ON event.room_id=attachment.room_id AND event.event_id=attachment.event_id
             WHERE attachment.room_id=? AND attachment.state='committed'
               AND attachment.viewer_access=1
               AND (attachment.expires_at IS NULL OR attachment.expires_at>?)
               AND (?='' OR EXISTS (
                   SELECT 1 FROM json_each(CASE WHEN json_valid(attachment.recipient_member_ids_json)
                       THEN attachment.recipient_member_ids_json ELSE '[]' END) AS recipient
                    WHERE recipient.value=?
               ))
               AND event.kind IN ('message.user','message.member')
               AND {search_where}
             GROUP BY attachment.event_id
        ) SELECT event.seq, event.event_id, event.kind, event.authority_epoch, event.created_at
            FROM matching
            JOIN hosted_room_events AS event
              ON event.room_id=? AND event.event_id=matching.event_id
        """
        common_params = (
            room_id,
            now,
            recipient_member_id,
            recipient_member_id,
            *search_params,
            room_id,
        )
        latest_events = conn.execute(
            candidates + " ORDER BY event.seq DESC LIMIT ?",
            (*common_params, ATTACHMENT_LIST_EVENT_SCAN_LIMIT),
        ).fetchall()
        latest_seq = 0
        for latest_event in latest_events:
            if not _event_epoch_is_visible(
                latest_event["authority_epoch"], authority_epoch
            ):
                continue
            eligible = _published_event_attachments(
                conn,
                room_id=room_id,
                event=latest_event,
                room_members=room_members,
                now=now,
                recipient_member_id=recipient_member_id,
            )
            if any(
                _matches_filters(item, folded_query, producer_member_id)
                for item in eligible
            ):
                latest_seq = int(latest_event["seq"])
                break

        event_params: tuple[Any, ...]
        if last_seq is None:
            event_where = "event.seq<=?"
            event_params = (snapshot_seq,)
        elif last_attachment_id is None:
            event_where = "event.seq<?"
            event_params = (last_seq,)
        else:
            event_where = "event.seq<=?"
            event_params = (last_seq,)
        events = conn.execute(
            candidates + f" WHERE {event_where} ORDER BY event.seq DESC LIMIT ?",
            (*common_params, *event_params, ATTACHMENT_LIST_EVENT_SCAN_LIMIT + 1),
        ).fetchall()
        more_events = len(events) > ATTACHMENT_LIST_EVENT_SCAN_LIMIT
        events = events[:ATTACHMENT_LIST_EVENT_SCAN_LIMIT]

        items: list[dict[str, Any]] = []
        last_returned: tuple[int, int, str] | None = None
        last_scanned_seq: int | None = None
        for event in events:
            event_seq = int(event["seq"])
            last_scanned_seq = event_seq
            if not _event_epoch_is_visible(event["authority_epoch"], authority_epoch):
                continue
            published = _published_event_attachments(
                conn,
                room_id=room_id,
                event=event,
                room_members=room_members,
                now=now,
                recipient_member_id=recipient_member_id,
            )
            for item in published:
                if (
                    last_seq == event_seq
                    and last_attachment_id is not None
                    and (item["manifest_index"], item["attachment_id"])
                    <= (last_manifest_index, last_attachment_id)
                ):
                    continue
                if not _matches_filters(item, folded_query, producer_member_id):
                    continue
                if len(items) >= limit:
                    if last_returned is None:  # pragma: no cover - guarded by limit
                        raise RuntimeError("attachment page cursor is unavailable")
                    return _published_attachment_page(
                        self,
                        items=items,
                        room_id=room_id,
                        authority_gateway_id=authority_gateway_id,
                        authority_epoch=authority_epoch,
                        snapshot_seq=snapshot_seq,
                        latest_seq=latest_seq,
                        folded_query=folded_query,
                        producer_member_id=producer_member_id,
                        recipient_member_id=recipient_member_id,
                        position=last_returned,
                    )
                items.append(item)
                last_returned = (
                    event_seq,
                    item["manifest_index"],
                    item["attachment_id"],
                )

        if more_events and last_scanned_seq is not None:
            return _published_attachment_page(
                self,
                items=items,
                room_id=room_id,
                authority_gateway_id=authority_gateway_id,
                authority_epoch=authority_epoch,
                snapshot_seq=snapshot_seq,
                latest_seq=latest_seq,
                folded_query=folded_query,
                producer_member_id=producer_member_id,
                recipient_member_id=recipient_member_id,
                position=(last_scanned_seq, None, None),
            )
        return _published_attachment_page(
            self,
            items=items,
            room_id=room_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
            snapshot_seq=snapshot_seq,
            latest_seq=latest_seq,
            folded_query=folded_query,
            producer_member_id=producer_member_id,
            recipient_member_id=recipient_member_id,
            position=None,
        )


def _published_attachment_page(
    self,
    *,
    items: list[dict[str, Any]],
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    snapshot_seq: int,
    latest_seq: int,
    folded_query: str,
    producer_member_id: str,
    recipient_member_id: str,
    position: tuple[int, int | None, str | None] | None,
) -> dict[str, Any]:
    cursor = None
    if position is not None:
        last_seq, last_manifest_index, last_attachment_id = position
        cursor = _encode_list_cursor(
            self,
            {
                "version": 2,
                "room_id": room_id,
                "authority_gateway_id": authority_gateway_id,
                "authority_epoch": authority_epoch,
                "snapshot_seq": snapshot_seq,
                "query_digest": hashlib.sha256(
                    folded_query.encode("utf-8")
                ).hexdigest(),
                "producer_member_id": producer_member_id,
                "recipient_member_id": recipient_member_id,
                "last_seq": last_seq,
                "last_attachment_id": last_attachment_id,
                "last_manifest_index": last_manifest_index,
            },
        )
    latest_seq = max(latest_seq, max((item["seq"] for item in items), default=0))
    page = {
        "items": items,
        "next_cursor": cursor,
        "has_more": cursor is not None,
        "snapshot_seq": snapshot_seq,
        "latest_seq": latest_seq,
        "authority": {
            "gateway_id": authority_gateway_id,
            "epoch": authority_epoch,
        },
    }
    encoded_size = len(json.dumps(page, ensure_ascii=False).encode("utf-8"))
    if encoded_size > MAX_ATTACHMENT_LIST_RESPONSE_BYTES:
        raise AttachmentError("attachment catalogue response exceeds its byte limit")
    return page
