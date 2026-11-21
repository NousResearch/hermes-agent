"""The pure attachment rules the store applies inside its attachment and room-event transactions, shared by the gateway and every backend.

The attachment limits, TTLs and patterns, the manifest validators and the catalog text fold. ``AttachmentError`` is the persistence contract's class. Standard library and ``persistence.gateway.hosted_rooms`` only.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from persistence.gateway.hosted_rooms import AttachmentError


MAX_ATTACHMENTS_PER_MESSAGE = 8
MAX_ATTACHMENT_BYTES = 15_000_000
MAX_MESSAGE_ATTACHMENT_BYTES = 25_000_000
MAX_ROOM_ATTACHMENT_BYTES = 512 * 1024 * 1024
MAX_GATEWAY_BLOB_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROOM_ATTACHMENT_COUNT = 50_000
MAX_GATEWAY_ATTACHMENT_COUNT = 200_000
MAX_ROOM_UNCOMMITTED_BYTES = 100 * 1024 * 1024
MAX_ROOM_UNCOMMITTED_COUNT = 64
MAX_TASK_ATTACHMENTS = 16
MAX_TASK_ATTACHMENT_BYTES = 50_000_000
UNCOMMITTED_TTL_SECONDS = 60 * 60
DISBANDED_GRACE_SECONDS = 15 * 60
CLASSIC_ATTACHMENT_TTL_SECONDS = 7 * 24 * 60 * 60

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MAX_ATTACHMENT_NAME_CHARS = 255
MAX_ATTACHMENT_MIME_CHARS = 127
MAX_IDENTIFIER_CHARS = 128

_ATTACHMENT_FIELDS = frozenset({"attachment_id", "kind", "name", "size", "mime"})
_ATTACHMENT_KINDS = frozenset({"image", "pdf", "file"})
_ATTACHMENT_ID_RE = re.compile(r"^att_[0-9a-f]{32}$")
_BLOB_ID_RE = re.compile(r"^blob_[0-9a-f]{32}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MIME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)


def _attachment_id(value: Any) -> str:
    if not isinstance(value, str) or _ATTACHMENT_ID_RE.fullmatch(value) is None:
        raise AttachmentError("invalid attachment_id")
    return value


def _name(value: Any) -> str:
    if not isinstance(value, str):
        raise AttachmentError("attachment name must be a string")
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeError:
        raise AttachmentError("attachment name must contain valid Unicode") from None
    if (
        not normalized
        or len(normalized) > MAX_ATTACHMENT_NAME_CHARS
        or normalized in {".", ".."}
        or any(token in normalized for token in ("/", "\\", "\x00", "\n", "\r"))
    ):
        raise AttachmentError("attachment name must be a bounded basename")
    return normalized


def _mime(value: Any) -> str:
    if not isinstance(value, str):
        raise AttachmentError("attachment mime must be a string")
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > MAX_ATTACHMENT_MIME_CHARS
        or _MIME_RE.fullmatch(normalized) is None
    ):
        raise AttachmentError("attachment mime is invalid")
    return normalized


def _kind(value: Any) -> str:
    if not isinstance(value, str) or value not in _ATTACHMENT_KINDS:
        raise AttachmentError("attachment kind must be image, pdf, or file")
    return value


def _safe_manifest_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AttachmentError("each attachment manifest entry must be an object")
    fields = frozenset(value)
    if fields != _ATTACHMENT_FIELDS:
        unknown = fields - _ATTACHMENT_FIELDS
        missing = _ATTACHMENT_FIELDS - fields
        if unknown:
            raise AttachmentError(
                f"attachment manifest has unknown fields: {', '.join(sorted(unknown))}"
            )
        raise AttachmentError(
            f"attachment manifest is missing fields: {', '.join(sorted(missing))}"
        )
    size = value["size"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ATTACHMENT_BYTES:
        raise AttachmentError("attachment size is outside the per-file limit")
    kind = _kind(value["kind"])
    mime = _mime(value["mime"])
    if kind == "image" and not mime.startswith("image/"):
        raise AttachmentError("image attachments require an image MIME type")
    if kind == "pdf" and mime != "application/pdf":
        raise AttachmentError("PDF attachments require application/pdf")
    return {
        "attachment_id": _attachment_id(value["attachment_id"]),
        "kind": kind,
        "name": _name(value["name"]),
        "size": size,
        "mime": mime,
    }


def validate_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AttachmentError("attachments must be a list")
    if len(value) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise AttachmentError(
            f"attachments must contain at most {MAX_ATTACHMENTS_PER_MESSAGE} entries"
        )
    normalized = [_safe_manifest_entry(entry) for entry in value]
    if len({entry["attachment_id"] for entry in normalized}) != len(normalized):
        raise AttachmentError("attachment ids must be unique within one message")
    if sum(entry["size"] for entry in normalized) > MAX_MESSAGE_ATTACHMENT_BYTES:
        raise AttachmentError("attachments exceed the per-message total size limit")
    return normalized


def validate_task_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate a bounded aggregate of several already-valid message manifests."""

    if not isinstance(value, list):
        raise AttachmentError("task attachments must be a list")
    if len(value) > MAX_TASK_ATTACHMENTS:
        raise AttachmentError(
            f"task attachments must contain at most {MAX_TASK_ATTACHMENTS} entries"
        )
    normalized = [_safe_manifest_entry(entry) for entry in value]
    if len({entry["attachment_id"] for entry in normalized}) != len(normalized):
        raise AttachmentError("task attachment ids must be unique")
    if sum(entry["size"] for entry in normalized) > MAX_TASK_ATTACHMENT_BYTES:
        raise AttachmentError("task attachments exceed the aggregate byte limit")
    return normalized


def fold_catalog_text(value: str) -> str:
    if value.isascii():
        return value.lower()
    return "".join(
        char for char in unicodedata.normalize("NFKD", value).casefold()
        if not unicodedata.combining(char)
    )
