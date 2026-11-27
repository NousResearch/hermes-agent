"""In-memory reference ``HostedRoomAttachmentRepository`` over the placement's hosted-room tables (``hosted_room_tables``) and blob files (``hosted_room_attachment_files``).

Each verb leaves the caller-side validation to the caller: ``put`` prunes in its own write and then stores the upload in a second write that writes or verifies the blob file before its rows; ``commit_message_with_receipt`` verifies every named blob inside its write; the reads test the row's frozen recipient set and the viewer authority inside their read, and ``read_viewer`` repeats its checks after the byte read. The internal steps live in ``hosted_room_attachment_steps``; ``HostedRoomRepository.append_event`` runs ``retain_message_attachments`` from there.

Reference limits: the open-time steps of each file (schema and migrations, ``reconcile_committed_attachments`` and the startup prune) have nothing to act on, since a reference file starts empty and is never reopened (``reopen_durability`` is False); a crash between a blob write and its commit is not modelled, so the ``.tmp-`` and unreferenced files the orphan sweep reclaims only arise when placed directly into the file map; the backend-wide lock makes every verb one atomic step, so a write can only land between ``read_viewer``'s two read contexts through the file map's ``before_read`` seam.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    AttachmentConflictError,
    AttachmentData,
    AttachmentError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    HostedPlacement,
)

from ..backend import register_repository
from . import hosted_room_attachment_rules as rules
from .hosted_room_attachment_files import BlobFiles, blob_files, read_blob_slice, read_verified_blob, write_blob_file
from .hosted_room_attachment_steps import (
    owner_event_ids,
    prune_expired_attachments,
    read_committed_row,
    require_viewer_room,
)
from .hosted_room_rules import clock
from .hosted_room_tables import HostedTables, read_tables, write_tables

__all__ = ["InMemoryHostedRoomAttachmentRepository"]


def _replay_upload(files: BlobFiles, existing: Mapping[str, Any], *, kind: str, name: str, mime: str, data: bytes, digest: str) -> dict[str, Any]:
    """The same upload replays after its blob is verified; other content refuses."""
    if (existing["kind"], existing["name"], existing["mime"], int(existing["size"]), existing["sha256"]) != (kind, name, mime, len(data), digest):
        raise AttachmentConflictError("upload_id was already used for different attachment content")
    read_verified_blob(files, blob_id=str(existing["blob_id"]), size=int(existing["size"]), sha256=str(existing["sha256"]))
    return rules.metadata(existing, idempotent=True)


def _check_quotas(tables: HostedTables, room_id: str, size: int) -> None:
    """The upload quotas, in order: room bytes, room count, gateway count, then the room's uncommitted uploads."""
    rows = list(tables["hosted_room_attachments"].get(room_id, {}).values())
    room = [row for row in rows if row["state"] != "disbanded"]
    uploaded = [row for row in rows if row["state"] == "uploaded"]
    if sum(int(row["size"]) for row in room) + size > rules.MAX_ROOM_ATTACHMENT_BYTES:
        raise AttachmentQuotaError("room attachment quota exceeded")
    if len(room) >= rules.MAX_ROOM_ATTACHMENT_COUNT:
        raise AttachmentQuotaError("room attachment count quota exceeded")
    if sum(len(room_rows) for room_rows in tables["hosted_room_attachments"].values()) >= rules.MAX_GATEWAY_ATTACHMENT_COUNT:
        raise AttachmentQuotaError("gateway attachment count quota exceeded")
    if sum(int(row["size"]) for row in uploaded) + size > rules.MAX_ROOM_UNCOMMITTED_BYTES or len(uploaded) + 1 > rules.MAX_ROOM_UNCOMMITTED_COUNT:
        raise AttachmentQuotaError("room uncommitted upload quota exceeded")


def _hold_blob(tables: HostedTables, files: BlobFiles, digest: str, data: bytes, now: float) -> str:
    """New content passes the physical quota, is written, then gets its row; known content is verified and gains a reference."""
    blobs = tables["hosted_room_attachment_blobs"]
    blob = next((row for row in blobs.values() if row["sha256"] == digest), None)
    if blob is None:
        if sum(int(row["size"]) for row in blobs.values()) + len(data) > rules.MAX_GATEWAY_BLOB_BYTES:
            raise AttachmentQuotaError("gateway attachment quota exceeded")
        blob_id = f"blob_{secrets.token_hex(16)}"
        write_blob_file(files, blob_id, data)
        blobs[blob_id] = {"blob_id": blob_id, "sha256": digest, "size": len(data), "ref_count": 1, "created_at": now}
        return blob_id
    blob_id = str(blob["blob_id"])
    read_verified_blob(files, blob_id=blob_id, size=int(blob["size"]), sha256=digest)
    blob["ref_count"] = int(blob["ref_count"]) + 1
    return blob_id


def _commit_check(row: Mapping[str, Any] | None, entry: Mapping[str, Any], *, event_id: str, recipients_json: str, viewer_access: bool, now: float) -> None:
    """Pass 1's refusals for one manifest entry, in a fixed order."""
    if row is None:
        raise AttachmentNotFoundError("attachment is not uploaded for this room")
    if {key: row[key] for key in rules.MANIFEST_FIELDS} != dict(entry):
        raise AttachmentConflictError("attachment manifest metadata does not match the uploaded bytes")
    if row["expires_at"] is not None and float(row["expires_at"]) <= now:
        raise AttachmentNotFoundError("attachment has expired")
    if row["state"] == "disbanded":
        raise AttachmentNotFoundError("attachment belongs to a disbanded room")
    if row["state"] == "committed" and (
            str(row["event_id"] or "") != event_id or row["recipient_member_ids_json"] != recipients_json
            or bool(row["viewer_access"]) != bool(viewer_access)):
        raise AttachmentConflictError("attachment is already owned by a different room event")


class InMemoryHostedRoomAttachmentRepository:
    """The attachment verbs on the placement's tables and blob files; every verb runs under the backend lock."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def prepare(self, *, placement: HostedPlacement, now: float | None = None) -> None:
        # A reference file's open-time steps have nothing to act on (see the reference limits above).
        return None

    def put(self, *, placement: HostedPlacement, room_id: str, upload_id: str, kind: str, name: str, mime: str, data: bytes, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = clock(now)
        digest = hashlib.sha256(data).hexdigest()
        prune_expired_attachments(self._store, placement, timestamp)
        files = blob_files(self._store, placement)
        with write_tables(self._store, placement) as tables:
            rows = tables.rows("hosted_room_attachments", room_id)
            existing = next((row for row in rows.values() if row["upload_id"] == upload_id), None)
            if existing is not None:
                return _replay_upload(files, existing, kind=kind, name=name, mime=mime, data=data, digest=digest)
            _check_quotas(tables, room_id, len(data))
            blob_id = _hold_blob(tables, files, digest, data, timestamp)
            attachment_id = f"att_{secrets.token_hex(16)}"
            rows[attachment_id] = row = {
                "attachment_id": attachment_id, "upload_id": upload_id, "room_id": room_id, "event_id": None, "kind": kind,
                "name": name, "catalog_name": rules.fold_catalog_text(name), "size": len(data), "mime": mime, "sha256": digest,
                "blob_id": blob_id, "recipient_member_ids_json": "[]", "viewer_access": 0, "state": "uploaded",
                "created_at": timestamp, "updated_at": timestamp, "expires_at": timestamp + rules.UNCOMMITTED_TTL_SECONDS}
            return rules.metadata(row)

    def commit_message_with_receipt(self, *, placement: HostedPlacement, room_id: str, event_id: str, manifest: Sequence[Mapping[str, JsonValue]], recipient_member_ids: Sequence[str], viewer_access: bool = False, retention_seconds: float | None = None, hold_until_event: bool = False, now: float | None = None) -> tuple[list[Mapping[str, JsonValue]], tuple[str, ...]]:
        timestamp = clock(now)
        recipients_json = json.dumps(sorted(recipient_member_ids), separators=(",", ":"))
        expires_at = (timestamp + rules.UNCOMMITTED_TTL_SECONDS if hold_until_event
                      else None if retention_seconds is None else timestamp + float(retention_seconds))
        normalized: list[Mapping[str, JsonValue]] = [dict(entry) for entry in manifest]
        files = blob_files(self._store, placement)
        transitioned: list[str] = []
        with write_tables(self._store, placement) as tables:
            rows = tables["hosted_room_attachments"].get(room_id, {})
            for entry in normalized:
                row = rows.get(str(entry["attachment_id"]))
                _commit_check(row, entry, event_id=event_id, recipients_json=recipients_json, viewer_access=viewer_access, now=timestamp)
                assert row is not None
                if row["state"] == "uploaded":
                    transitioned.append(str(entry["attachment_id"]))
                read_verified_blob(files, blob_id=str(row["blob_id"]), size=int(row["size"]), sha256=str(row["sha256"]))
            for entry in normalized:
                if entry["attachment_id"] not in transitioned:
                    continue
                row = rows[str(entry["attachment_id"])]
                if row["state"] != "uploaded":
                    raise AttachmentConflictError("attachment changed during message commitment")
                row.update(event_id=event_id, recipient_member_ids_json=recipients_json, viewer_access=1 if viewer_access else 0,
                           state="committed", updated_at=timestamp, expires_at=expires_at)
        return normalized, tuple(transitioned)

    def abort_message_commit(self, *, placement: HostedPlacement, room_id: str, event_id: str, attachment_ids: Sequence[str], now: float | None = None) -> int:
        if not attachment_ids:
            return 0
        timestamp = clock(now)
        with write_tables(self._store, placement) as tables:
            published = owner_event_ids(tables, room_id, event_id)
            remaining = {value for value in attachment_ids if published is None or value not in published}
            rows = tables["hosted_room_attachments"].get(room_id, {})
            changed = [rows[value] for value in sorted(remaining)
                       if value in rows and rows[value]["event_id"] == event_id and rows[value]["state"] == "committed"]
            for row in changed:
                row.update(event_id=None, recipient_member_ids_json="[]", viewer_access=0, state="uploaded",
                           updated_at=timestamp, expires_at=timestamp + rules.UNCOMMITTED_TTL_SECONDS)
        return len(changed)

    def read(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str | None, event_id: str | None = None, viewer: bool = False, now: float | None = None) -> AttachmentData:
        row = read_committed_row(
            read_tables(self._store, placement), room_id=room_id, attachment_id=attachment_id,
            recipient_member_id=recipient_member_id or "", event_id=event_id, viewer=viewer, now=clock(now))
        data = read_verified_blob(blob_files(self._store, placement), blob_id=str(row["blob_id"]), size=int(row["size"]), sha256=str(row["sha256"]))
        return AttachmentData(rules.metadata(row), data)

    def describe(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str, event_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        row = read_committed_row(
            read_tables(self._store, placement), room_id=room_id, attachment_id=attachment_id,
            recipient_member_id=recipient_member_id, event_id=event_id, viewer=False, now=clock(now))
        return rules.metadata(row)

    def read_range(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str, event_id: str, offset: int, length: int, now: float | None = None) -> AttachmentData:
        row = read_committed_row(
            read_tables(self._store, placement), room_id=room_id, attachment_id=attachment_id,
            recipient_member_id=recipient_member_id, event_id=event_id, viewer=False, now=clock(now))
        if offset >= int(row["size"]):
            raise AttachmentError("attachment range starts past the end")
        data = read_blob_slice(blob_files(self._store, placement), blob_id=str(row["blob_id"]), size=int(row["size"]), offset=offset, length=length)
        return AttachmentData(rules.metadata(row), data)

    def _viewer_row(self, placement: HostedPlacement, room_id: str, attachment_id: str, event_id: str | None, recipient_member_id: str, authority: tuple[str, int], now: float | None) -> dict[str, Any]:
        """One of ``read_viewer``'s read contexts: the viewer room, then the committed row."""
        tables = read_tables(self._store, placement)
        require_viewer_room(tables, room_id=room_id, authority_gateway_id=authority[0], authority_epoch=authority[1])
        return read_committed_row(
            tables, room_id=room_id, attachment_id=attachment_id, recipient_member_id=recipient_member_id,
            event_id=event_id, viewer=True, now=clock(now))

    def read_viewer(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, event_id: str | None = None, recipient_member_id: str | None = None, authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> AttachmentData:
        selection = (placement, room_id, attachment_id, event_id, recipient_member_id or "", (authority_gateway_id, authority_epoch), now)
        row = self._viewer_row(*selection)
        data = read_verified_blob(blob_files(self._store, placement), blob_id=str(row["blob_id"]), size=int(row["size"]), sha256=str(row["sha256"]))
        current = self._viewer_row(*selection)
        if current["blob_id"] != row["blob_id"] or rules.metadata(current) != rules.metadata(row):
            raise AttachmentNotFoundError("attachment changed during viewer read")
        return AttachmentData(rules.metadata(current), data)


register_repository("hosted_room_attachments", InMemoryHostedRoomAttachmentRepository)
