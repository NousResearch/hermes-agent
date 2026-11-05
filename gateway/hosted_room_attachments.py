"""Durable, private attachment storage for gateway-owned Group Chats.

The room log carries only server-minted attachment ids and bounded metadata.
Canonical bytes live in a private directory beside the room file and are shared
by SHA-256 through an internal blob table.  Upload ids make client retries
idempotent; send-time commitment binds each upload to one room event and the
room's frozen recipient roster.  The rows and bytes are the storage interface's
``hosted_room_attachments`` repository; this module checks and normalizes what
callers send before each call.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any, Callable, Sequence

from persistence.gateway.hosted_rooms import (
    AttachmentConflictError,
    AttachmentData,
    AttachmentError,
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    HostedPlacement,
)
from session_rules.hosted_attachments import (
    CLASSIC_ATTACHMENT_TTL_SECONDS,
    DISBANDED_GRACE_SECONDS,
    MAX_ATTACHMENTS_PER_MESSAGE,
    MAX_ATTACHMENT_BYTES,
    MAX_IDENTIFIER_CHARS,
    MAX_MESSAGE_ATTACHMENT_BYTES,
    MAX_TASK_ATTACHMENT_BYTES,
    MAX_TASK_ATTACHMENTS,
    UNCOMMITTED_TTL_SECONDS,
    _IDENTIFIER_RE,
    _SHA256_RE,
    _attachment_id,
    _kind,
    _mime,
    _name,
    validate_manifest,
    validate_task_manifest,
)


def default_attachment_root(db_path: Path | str) -> Path:
    """Return the private byte-store root beside the room state database."""

    return Path(db_path).parent / "hosted-room-attachments"


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise AttachmentError(f"{label} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > MAX_IDENTIFIER_CHARS
        or _IDENTIFIER_RE.fullmatch(normalized) is None
    ):
        raise AttachmentError(f"invalid {label}")
    return normalized


def _sniff_known_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    if data.startswith(b"\x1f\x8b"):
        return "application/gzip"
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return "text/plain"
    return "application/octet-stream"


def _validate_mime_and_kind(data: bytes, *, kind: str, mime: str) -> None:
    sniffed = _sniff_known_mime(data)
    if kind == "image":
        if not mime.startswith("image/") or sniffed != mime:
            raise AttachmentError("image bytes do not match the declared MIME type")
        return
    if kind == "pdf":
        if mime != "application/pdf" or sniffed != "application/pdf":
            raise AttachmentError("PDF bytes do not match application/pdf")
        return
    if mime.startswith("image/") or mime == "application/pdf":
        raise AttachmentError("image and PDF MIME types require their matching kind")
    if mime.startswith("text/") and sniffed != "text/plain":
        raise AttachmentError("text bytes do not match the declared MIME type")
    # Generic files are opaque bytes. ZIP and gzip are container formats used
    # by many non-archive MIME types (DOCX, XLSX, PPTX, EPUB, JAR, APK, etc.).
    # Only image and PDF payloads are security-sensitive enough to require an
    # exact magic/MIME match here; text remains strict to avoid binary prompts.


def decode_content_base64(value: Any) -> bytes:
    """Strictly decode one bounded RPC upload body without accepting data URLs."""

    if not isinstance(value, str) or not value:
        raise AttachmentError("content_base64 is required")
    if value.lstrip().lower().startswith("data:"):
        raise AttachmentError("data URLs are not accepted")
    maximum_encoded = ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4
    if len(value) > maximum_encoded + 4:
        raise AttachmentError("attachment exceeds the per-file size limit")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError("content_base64 is not valid base64") from exc
    if not data:
        raise AttachmentError("attachment must not be empty")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise AttachmentError("attachment exceeds the per-file size limit")
    return data


def encode_content_base64(data: bytes) -> str:
    """Encode verified bytes for the bounded attachment read RPC."""

    return base64.b64encode(data).decode("ascii")


class HostedRoomAttachmentStore:
    """The attachment verbs of one room file: each checks and normalizes its arguments, then calls the
    storage interface's ``hosted_room_attachments`` on ``placement``.

    ``clock`` fixes ``now`` for every call; without one the store reads the wall clock where each verb
    needs it.
    """

    def __init__(
        self,
        persistence: Any,
        placement: HostedPlacement = "profile",
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._repository = persistence.hosted_room_attachments
        self.placement = placement
        self.clock = clock

    def _now(self) -> float | None:
        return None if self.clock is None else float(self.clock())

    def prepare(self) -> None:
        """Run the file's attachment setup (tables, reconciliation, prune) unless this process already has."""
        self._repository.prepare(placement=self.placement, now=self._now())

    def put(
        self,
        *,
        room_id: Any,
        upload_id: Any,
        kind: Any,
        name: Any,
        mime: Any,
        data: bytes,
    ) -> dict[str, Any]:
        room_id = _identifier(room_id, label="room_id")
        upload_id = _identifier(upload_id, label="upload_id")
        kind = _kind(kind)
        name = _name(name)
        mime = _mime(mime)
        if not isinstance(data, bytes) or not data:
            raise AttachmentError("attachment bytes must not be empty")
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise AttachmentError("attachment exceeds the per-file size limit")
        _validate_mime_and_kind(data, kind=kind, mime=mime)
        return self._repository.put(
            placement=self.placement, room_id=room_id, upload_id=upload_id, kind=kind, name=name,
            mime=mime, data=data, now=self._now())

    def commit_message(
        self,
        *,
        room_id: Any,
        event_id: Any,
        manifest: Any,
        recipient_member_ids: Sequence[str],
        viewer_access: bool = False,
        retention_seconds: float | None = None,
        hold_until_event: bool = False,
    ) -> list[dict[str, Any]]:
        normalized, _transitioned = self.commit_message_with_receipt(
            room_id=room_id,
            event_id=event_id,
            manifest=manifest,
            recipient_member_ids=recipient_member_ids,
            viewer_access=viewer_access,
            retention_seconds=retention_seconds,
            hold_until_event=hold_until_event,
        )
        return normalized

    def commit_message_with_receipt(
        self,
        *,
        room_id: Any,
        event_id: Any,
        manifest: Any,
        recipient_member_ids: Sequence[str],
        viewer_access: bool = False,
        retention_seconds: float | None = None,
        hold_until_event: bool = False,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        """Commit a manifest and report only rows transitioned by this call."""

        room_id = _identifier(room_id, label="room_id")
        event_id = _identifier(event_id, label="event_id")
        normalized = validate_manifest(manifest)
        recipients = tuple(
            _identifier(member_id, label="recipient_member_id")
            for member_id in recipient_member_ids
        )
        if not recipients:
            raise AttachmentError("attachment commitment requires recipient ids")
        if len(set(recipients)) != len(recipients):
            raise AttachmentError("recipient ids must be unique")
        return self._repository.commit_message_with_receipt(
            placement=self.placement, room_id=room_id, event_id=event_id, manifest=normalized,
            recipient_member_ids=recipients, viewer_access=viewer_access,
            retention_seconds=retention_seconds, hold_until_event=hold_until_event, now=self._now())

    def abort_message_commit(
        self,
        *,
        room_id: Any,
        event_id: Any,
        attachment_ids: Sequence[str],
    ) -> int:
        """Return a failed pre-event commitment to its bounded upload state."""

        room_id = _identifier(room_id, label="room_id")
        event_id = _identifier(event_id, label="event_id")
        attachment_ids = [_attachment_id(value) for value in attachment_ids]
        return self._repository.abort_message_commit(
            placement=self.placement, room_id=room_id, event_id=event_id,
            attachment_ids=attachment_ids, now=self._now())

    def read(
        self,
        *,
        room_id: Any,
        attachment_id: Any,
        recipient_member_id: Any,
        event_id: Any | None = None,
        viewer: bool = False,
    ) -> AttachmentData:
        room_id = _identifier(room_id, label="room_id")
        attachment_id = _attachment_id(attachment_id)
        recipient_member_id = (
            _identifier(recipient_member_id, label="recipient_member_id")
            if recipient_member_id is not None
            else ""
        )
        normalized_event = _identifier(event_id, label="event_id") if event_id is not None else None
        return self._repository.read(
            placement=self.placement, room_id=room_id, attachment_id=attachment_id,
            recipient_member_id=recipient_member_id, event_id=normalized_event, viewer=viewer,
            now=self._now())

    def describe(
        self,
        *,
        room_id: Any,
        attachment_id: Any,
        recipient_member_id: Any,
        event_id: Any,
    ) -> dict[str, Any]:
        """``read`` without bytes: the metadata (including the upload-verified SHA-256) of a
        committed attachment this recipient may read for this event."""
        return self._repository.describe(
            placement=self.placement, room_id=_identifier(room_id, label="room_id"),
            attachment_id=_attachment_id(attachment_id),
            recipient_member_id=_identifier(recipient_member_id, label="recipient_member_id"),
            event_id=_identifier(event_id, label="event_id"), now=self._now())

    def read_range(
        self,
        *,
        room_id: Any,
        attachment_id: Any,
        recipient_member_id: Any,
        event_id: Any,
        offset: int,
        length: int,
    ) -> AttachmentData:
        """``read`` for one slice: same authorization, metadata carries the stored whole-file
        SHA-256 and the receiver verifies the reassembled bytes against it."""
        room_id = _identifier(room_id, label="room_id")
        attachment_id = _attachment_id(attachment_id)
        recipient_member_id = _identifier(recipient_member_id, label="recipient_member_id")
        normalized_event = _identifier(event_id, label="event_id")
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
            raise AttachmentError("attachment range must be non-negative integers")
        return self._repository.read_range(
            placement=self.placement, room_id=room_id, attachment_id=attachment_id,
            recipient_member_id=recipient_member_id, event_id=normalized_event, offset=offset,
            length=length, now=self._now())

    def read_viewer(
        self,
        *,
        room_id: Any,
        attachment_id: Any,
        event_id: Any | None = None,
        recipient_member_id: Any = None,
        authority_gateway_id: Any,
        authority_epoch: Any,
    ) -> AttachmentData:
        """Read for a live hosted viewer, fencing revocation across byte I/O."""

        room_id = _identifier(room_id, label="room_id")
        attachment_id = _attachment_id(attachment_id)
        normalized_event = _identifier(event_id, label="event_id") if event_id is not None else None
        recipient_member_id = (
            _identifier(recipient_member_id, label="recipient_member_id")
            if recipient_member_id is not None else ""
        )
        authority_gateway_id = _identifier(authority_gateway_id, label="authority_gateway_id")
        if isinstance(authority_epoch, bool) or not isinstance(authority_epoch, int) or authority_epoch < 1:
            raise AttachmentError("authority_epoch must be a positive integer")
        return self._repository.read_viewer(
            placement=self.placement, room_id=room_id, attachment_id=attachment_id,
            event_id=normalized_event, recipient_member_id=recipient_member_id,
            authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch,
            now=self._now())


__all__ = [
    "CLASSIC_ATTACHMENT_TTL_SECONDS",
    "DISBANDED_GRACE_SECONDS",
    "MAX_ATTACHMENTS_PER_MESSAGE",
    "MAX_ATTACHMENT_BYTES",
    "MAX_MESSAGE_ATTACHMENT_BYTES",
    "MAX_TASK_ATTACHMENTS",
    "MAX_TASK_ATTACHMENT_BYTES",
    "UNCOMMITTED_TTL_SECONDS",
    "AttachmentConflictError",
    "AttachmentData",
    "AttachmentError",
    "AttachmentIntegrityError",
    "AttachmentNotFoundError",
    "AttachmentQuotaError",
    "HostedRoomAttachmentStore",
    "decode_content_base64",
    "default_attachment_root",
    "encode_content_base64",
    "validate_manifest",
    "validate_task_manifest",
]
