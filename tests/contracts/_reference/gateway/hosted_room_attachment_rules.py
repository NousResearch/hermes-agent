"""The attachment rules the reference reads, under one module its quota cases can patch: the limits, TTLs, patterns, manifest validators, ``fold_catalog_text`` and metadata projection of ``session_rules.hosted_attachments``."""

from __future__ import annotations

from session_rules.hosted_attachments import _BLOB_ID_RE as BLOB_ID_RE
from session_rules.hosted_attachments import _metadata as metadata
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
