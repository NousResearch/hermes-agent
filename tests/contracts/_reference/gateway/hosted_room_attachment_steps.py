"""The internal steps of the in-memory ``HostedRoomAttachmentRepository`` over a placement's hosted-room tables.

``hosted_room_attachments[room_id][attachment_id]`` holds the attachment rows and ``hosted_room_attachment_blobs[blob_id]`` the blob rows. ``retain_message_attachments`` runs inside ``HostedRoomRepository.append_event``'s write; the other steps serve the attachment verbs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from persistence.gateway.hosted_rooms import (
    AttachmentError,
    AttachmentNotFoundError,
    EventAttachmentConflictError,
    HostedPlacement,
)

from . import hosted_room_attachment_rules as rules
from .hosted_room_attachment_files import blob_files, sweep_orphan_blobs, unlink_blobs
from .hosted_room_tables import HostedTables, find_event, write_tables

__all__ = ["owner_event_ids", "prune_expired_attachments", "read_committed_row", "require_viewer_room", "retain_message_attachments"]

_MESSAGE_KINDS = frozenset({"message.user", "message.member"})


def _expired(row: Mapping[str, Any], now: float) -> bool:
    return row["expires_at"] is not None and float(row["expires_at"]) <= now


def _mark_disbanded(tables: HostedTables, now: float) -> None:
    """Committed rows of a disbanded room become ``disbanded`` and expire after the grace."""
    rooms = tables["hosted_rooms"]
    for room_id, rows in tables["hosted_room_attachments"].items():
        room = rooms.get(room_id)
        if room is None or room["disbanded_at"] is None:
            continue
        for row in rows.values():
            if row["state"] == "committed":
                row.update(state="disbanded", expires_at=float(room["disbanded_at"]) + rules.DISBANDED_GRACE_SECONDS, updated_at=now)


def _release_blobs(tables: HostedTables, released: Counter[str]) -> list[str]:
    """Drop each released reference; a blob whose last reference went loses its row."""
    blobs, removed = tables["hosted_room_attachment_blobs"], []
    for blob_id, count in released.items():
        blob = blobs.get(blob_id)
        if blob is None:
            continue
        if int(blob["ref_count"]) <= count:
            removed.append(blob_id)
        else:
            blob["ref_count"] = int(blob["ref_count"]) - count
    for blob_id in removed:
        del blobs[blob_id]
    return removed


def prune_expired_attachments(store: Any, placement: HostedPlacement, now: float) -> int:
    """Internal step ``prune_expired_attachments``: one write marks disbanded rooms' rows and deletes expired rows and their last blob references; after it commits, the released blob files are unlinked and ``sweep_orphan_blobs`` runs. Returns the deleted row count."""
    with write_tables(store, placement) as tables:
        _mark_disbanded(tables, now)
        released: Counter[str] = Counter()
        for rows in tables["hosted_room_attachments"].values():
            for attachment_id in [key for key, row in rows.items() if _expired(row, now)]:
                released[str(rows.pop(attachment_id)["blob_id"])] += 1
        removed_blob_ids = _release_blobs(tables, released)
    unlink_blobs(blob_files(store, placement), removed_blob_ids)
    sweep_orphan_blobs(store, placement)
    return sum(released.values())


def retain_message_attachments(tables: HostedTables, *, room_id: str, event_id: str, manifest: Any, now: float) -> None:
    """Internal step ``retain_message_attachments`` inside ``append_event``: the manifest is validated (``AttachmentError``) and must already be canonical; each entry must name a committed, viewer-visible, unexpired row of this room and event with the entry's metadata, else ``EventAttachmentConflictError``; the rows then lose their expiry. The attachment table always exists here (the backend's open creates it), so the ``'message attachments are not committed'`` refusal for a file without it does not arise."""
    normalized = rules.validate_manifest(manifest)
    if not normalized:
        return
    if normalized != manifest:
        raise AttachmentError("published attachment metadata must be canonical")
    rows = tables["hosted_room_attachments"].get(room_id, {})
    for entry in normalized:
        row = rows.get(entry["attachment_id"])
        if (row is None or row["event_id"] != event_id or row["state"] != "committed" or row["viewer_access"] != 1
                or _expired(row, now) or any(row[key] != value for key, value in entry.items())):
            raise EventAttachmentConflictError("attachment commitment changed before event publication")
    for entry in normalized:
        rows[entry["attachment_id"]].update(expires_at=None, updated_at=now)


def owner_event_ids(tables: HostedTables, room_id: str, event_id: str) -> set[Any] | None:
    """Internal step ``owner_event_ids``: the attachment ids the published owner event names, or None when the event does not exist; a malformed payload raises ``json.loads``'s ``ValueError``."""
    owner = find_event(tables, room_id, event_id)
    if owner is None:
        return None
    return {item.get("attachment_id") for item in json.loads(owner["payload_json"]).get("attachments", []) if isinstance(item, Mapping)}


def _published(tables: HostedTables, room_id: str, row: Mapping[str, Any]) -> bool:
    """A room-visible row needs a published ``message.*`` owner whose manifest carries the row's metadata."""
    owner = find_event(tables, room_id, str(row["event_id"] or ""))
    if owner is None or owner["kind"] not in _MESSAGE_KINDS:
        return False
    payload = json.loads(owner["payload_json"])
    published = payload.get("attachments") if isinstance(payload, Mapping) else None
    return isinstance(published, list) and {key: row[key] for key in rules.MANIFEST_FIELDS} in published


def read_committed_row(tables: HostedTables, *, room_id: str, attachment_id: str, recipient_member_id: str, event_id: str | None, viewer: bool, now: float) -> dict[str, Any]:
    """Internal step ``read_committed_row``, each refusal ``AttachmentNotFoundError`` with its fixed message, in a fixed order."""
    row = tables["hosted_room_attachments"].get(room_id, {}).get(attachment_id)
    if row is None or row["state"] != "committed":
        raise AttachmentNotFoundError("attachment is not committed for this room")
    if _expired(row, now):
        raise AttachmentNotFoundError("attachment has expired")
    if event_id is not None and str(row["event_id"] or "") != event_id:
        raise AttachmentNotFoundError("attachment is not owned by this room event")
    if int(row["viewer_access"] or 0) and not _published(tables, room_id, row):
        raise AttachmentNotFoundError("attachment is not published in this room")
    if viewer:
        if int(row["viewer_access"] or 0) != 1:
            raise AttachmentNotFoundError("attachment is unavailable to Group Chat viewers")
    elif recipient_member_id not in json.loads(str(row["recipient_member_ids_json"])):
        raise AttachmentNotFoundError("attachment is unavailable to this recipient")
    return row


def require_viewer_room(tables: HostedTables, *, room_id: str, authority_gateway_id: str, authority_epoch: int) -> None:
    """Internal step ``require_viewer_room``: the live room at the passed authority. No ``hosted_room_quarantine`` or ``hosted_room_disband_fences`` table is ever created, so neither is read."""
    room = tables["hosted_rooms"].get(room_id)
    if (room is None or room["disbanded_at"] is not None or room["authority_gateway_id"] != authority_gateway_id
            or room["authority_epoch"] != authority_epoch):
        raise AttachmentNotFoundError("Group Chat viewer authority changed")
