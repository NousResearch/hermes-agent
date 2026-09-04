"""Bounded wire values for owner-messaging access to published room files."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

MAX_CATALOG_BYTES = 128 * 1024
MAX_FILE_BYTES = 15_000_000
MAX_QUERY_BYTES = 16 * 1024
MAX_RECEIPT_CHARS = 8 * 1024
MAX_ERROR_BYTES = 4096
FILE_TIMEOUT_SECONDS = 15.0
MANIFEST_FIELDS = frozenset({"attachment_id", "kind", "name", "mime", "size"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ATTACHMENT_ID = re.compile(r"^att_[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MESSAGES = {
    "file_code_ambiguous": "That code matches several files. Use a longer code from Files.",
    "file_lookup_limit": "This lookup could not be completed safely. Open Files on Desktop.",
    "classic_files_on_desktop": "Files for this Group Chat are on your Desktop.",
    "file_access_unsupported": "File access is not available for this Group Chat yet.",
    "file_access_denied": "File access is unavailable or no longer authorized.",
    "file_unavailable": "This file is no longer available.",
    "file_invalid_request": "The file request is invalid.",
    "file_invalid_response": "The Group Chat host returned invalid file data.",
    "file_integrity_failed": "This file could not be verified.",
    "file_too_large": "This file exceeds the allowed size.",
    "file_timeout": "The file request exceeded its time limit.",
    "file_host_unavailable": "The Group Chat host is temporarily unavailable.",
    "attachment_cursor_reset_required": "Return to Latest to restart file browsing.",
}


class FileAccessError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(_MESSAGES[code])


class FileCodeAmbiguous(FileAccessError):
    def __init__(self, matches):
        super().__init__("file_code_ambiguous")
        self.matches = tuple(dict(item) for item in matches)


def identifier(value: Any) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise FileAccessError("file_invalid_request")
    return value


def attachment_id(value: Any) -> str:
    if not isinstance(value, str) or _ATTACHMENT_ID.fullmatch(value) is None:
        raise FileAccessError("file_invalid_request")
    return value


def byte_limit(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_FILE_BYTES
    ):
        raise FileAccessError("file_invalid_request")
    return value


def catalog_options(options: Mapping[str, Any]) -> dict[str, Any]:
    if set(options) - {"cursor", "limit", "query", "producer_member_id"}:
        raise FileAccessError("file_invalid_request")
    result = {key: value for key, value in options.items() if value is not None}
    limit = result.get("limit", 8)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise FileAccessError("file_invalid_request")
    result["limit"] = limit
    for key, maximum in (("cursor", 4096), ("query", 255), ("producer_member_id", 128)):
        value = result.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > maximum):
            raise FileAccessError("file_invalid_request")
    return result


def scope(
    room: Mapping[str, Any], member_id: str, target_profile: str
) -> dict[str, Any]:
    epoch = room.get("authority_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise FileAccessError("file_access_denied")
    return {
        "room_id": identifier(room.get("room_id")),
        "member_id": identifier(member_id),
        "target_profile": identifier(target_profile),
        "authority_gateway_id": identifier(room.get("authority_gateway_id")),
        "authority_epoch": epoch,
    }


def manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    from gateway.hosted_room_attachments import validate_manifest

    try:
        entry = {key: value[key] for key in MANIFEST_FIELDS}
        normalized = validate_manifest([entry])[0]
        if entry != normalized:
            raise ValueError
        return normalized
    except (KeyError, TypeError, ValueError):
        raise FileAccessError("file_invalid_response") from None


def validate_scope(value: Any, expected: Mapping[str, Any]) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != set(expected)
        or any(
            type(value[key]) is not type(expected[key]) or value[key] != expected[key]
            for key in expected
        )
    ):
        raise FileAccessError("file_invalid_response")


def file_metadata(value: Any, *, event_id: str, selected_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS | {
        "event_id",
        "sha256",
    }:
        raise FileAccessError("file_invalid_response")
    entry = manifest(value)
    if value["event_id"] != event_id or entry["attachment_id"] != selected_id:
        raise FileAccessError("file_invalid_response")
    if (
        not isinstance(value["sha256"], str)
        or _DIGEST.fullmatch(value["sha256"]) is None
    ):
        raise FileAccessError("file_invalid_response")
    return dict(value)


def catalog_page(
    value: Any, *, expected_scope: dict[str, Any], limit: int
) -> dict[str, Any]:
    fields = {"items", "next_cursor", "has_more", "snapshot_seq", "authority", "scope"}
    if (
        not isinstance(value, dict)
        or not fields <= set(value) <= fields | {"latest_seq"}
        or value["scope"] != expected_scope
    ):
        raise FileAccessError("file_invalid_response")
    validate_scope(value["scope"], expected_scope)
    if "latest_seq" in value and (
        type(value["latest_seq"]) is not int or value["latest_seq"] < 0
    ):
        raise FileAccessError("file_invalid_response")
    validate_scope(
        value["authority"],
        {
            "gateway_id": expected_scope["authority_gateway_id"],
            "epoch": expected_scope["authority_epoch"],
        },
    )
    snapshot = value["snapshot_seq"]
    if isinstance(snapshot, bool) or not isinstance(snapshot, int) or snapshot < 0:
        raise FileAccessError("file_invalid_response")
    cursor = value["next_cursor"]
    if (
        type(value["has_more"]) is not bool
        or (
            value["has_more"]
            and (not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096)
        )
        or (not value["has_more"] and cursor is not None)
    ):
        raise FileAccessError("file_invalid_response")
    items = value["items"]
    if not isinstance(items, list) or len(items) > limit:
        raise FileAccessError("file_invalid_response")
    previous = None
    seen = set()
    item_fields = MANIFEST_FIELDS | {"event_id", "seq", "producer", "shared_at"}
    for item in items:
        if not isinstance(item, dict) or not item_fields <= set(item) <= item_fields | {
            "manifest_index"
        }:
            raise FileAccessError("file_invalid_response")
        manifest(item)
        try:
            identifier(item["event_id"])
            seq = item["seq"]
            producer = item["producer"]
            if (
                isinstance(seq, bool)
                or not isinstance(seq, int)
                or not 1 <= seq <= snapshot
            ):
                raise ValueError
            if "manifest_index" in item and (
                type(item["manifest_index"]) is not int
                or not 0 <= item["manifest_index"] <= 7
            ):
                raise ValueError
            if not isinstance(producer, dict) or set(producer) != {
                "kind",
                "id",
                "label",
            }:
                raise ValueError
            identifier(producer["id"])
            if (
                producer["kind"] not in {"user", "member"}
                or not isinstance(producer["label"], str)
                or len(producer["label"]) > 200
            ):
                raise ValueError
            if (
                isinstance(item["shared_at"], bool)
                or not isinstance(item["shared_at"], (int, float))
                or not math.isfinite(item["shared_at"])
            ):
                raise ValueError
            identity = (item["event_id"], item["attachment_id"])
            if (previous is not None and seq > previous) or identity in seen:
                raise ValueError
        except (ValueError, TypeError, FileAccessError):
            raise FileAccessError("file_invalid_response") from None
        previous = seq
        seen.add(identity)
    return value
