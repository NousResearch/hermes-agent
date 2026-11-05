"""The attachment rules the reference reads, under one module its quota cases can patch: the limits, TTLs, patterns, manifest validators and ``fold_catalog_text`` of ``session_rules.hosted_attachments``, and the metadata dict the store returns for an attachment row."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from session_rules.hosted_attachments import _BLOB_ID_RE as BLOB_ID_RE
from session_rules.hosted_attachments import (
    CLASSIC_ATTACHMENT_TTL_SECONDS,
    DISBANDED_GRACE_SECONDS,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
    MAX_GATEWAY_ATTACHMENT_COUNT,
    MAX_GATEWAY_BLOB_BYTES,
    MAX_MESSAGE_ATTACHMENT_BYTES,
    MAX_ROOM_ATTACHMENT_BYTES,
    MAX_ROOM_ATTACHMENT_COUNT,
    MAX_ROOM_UNCOMMITTED_BYTES,
    MAX_ROOM_UNCOMMITTED_COUNT,
    MAX_TASK_ATTACHMENT_BYTES,
    MAX_TASK_ATTACHMENTS,
    UNCOMMITTED_TTL_SECONDS,
    fold_catalog_text,
    validate_manifest,
    validate_task_manifest,
)

__all__ = [
    "BLOB_ID_RE",
    "CLASSIC_ATTACHMENT_TTL_SECONDS",
    "DISBANDED_GRACE_SECONDS",
    "MANIFEST_FIELDS",
    "MAX_ATTACHMENTS_PER_MESSAGE",
    "MAX_ATTACHMENT_BYTES",
    "MAX_GATEWAY_ATTACHMENT_COUNT",
    "MAX_GATEWAY_BLOB_BYTES",
    "MAX_MESSAGE_ATTACHMENT_BYTES",
    "MAX_ROOM_ATTACHMENT_BYTES",
    "MAX_ROOM_ATTACHMENT_COUNT",
    "MAX_ROOM_UNCOMMITTED_BYTES",
    "MAX_ROOM_UNCOMMITTED_COUNT",
    "MAX_TASK_ATTACHMENTS",
    "MAX_TASK_ATTACHMENT_BYTES",
    "UNCOMMITTED_TTL_SECONDS",
    "fold_catalog_text",
    "metadata",
    "validate_manifest",
    "validate_task_manifest",
]

# The key order of a normalized manifest entry.
MANIFEST_FIELDS = ("attachment_id", "kind", "name", "size", "mime")


def metadata(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    """The metadata dict of an attachment row: its identity, content facts and state, ``event_id`` only once one is bound."""
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
