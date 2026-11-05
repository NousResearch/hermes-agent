"""Flush marker sync for transcript batch appends: after a batch commits, each live message dict takes the persist
marker, its row's identity, timestamp and ownership version, and what the store's canonical row hands it."""

from __future__ import annotations

from typing import Any

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_metadata import CANONICAL_ROW, DB_ROW_SNAPSHOT, copy_identity_fields


# The durable payload fields a canonical row hands a live dict, ``role`` first: every column a transcript write stores
# except row identity, the active flag and the ones the display index, timestamp and session linkage own.
_SYNC_FIELDS = (
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "effect_disposition",
    "token_count",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "platform_message_id",
    "observed",
    "_compressed_summary",
    "api_content",
    "display_kind",
    "display_metadata",
    "absorbed_message_uids",
    "tool_call_uids",
)
# Presentation-only metadata a matched (ours) row may hand to a live dict that lacks it.
_LIVE_MISSING_METADATA = ("display_kind", "display_metadata")
# Canonical-row markers: a matched row hands over only missing presentation metadata; a legacy (no-digest)
# dict over a filled assistant row adopts only its content.
_METADATA_ONLY = "_metadata_only"
_CONTENT_ONLY = "_content_only"


def sync_flushed_message_markers(batch_msgs: list[dict[str, Any]], batch_rows: list[dict[str, Any]]) -> None:
    """Stamp persistence markers and sync canonical durable fields onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        copy_identity_fields(row, written)
        if isinstance(row.get("timestamp"), (int, float)):
            written["timestamp"] = row["timestamp"]
        if isinstance(row.get(DB_ROW_SNAPSHOT), str):
            written[DB_ROW_SNAPSHOT] = row[DB_ROW_SNAPSHOT]
        canonical = row.get(CANONICAL_ROW)
        if isinstance(canonical, dict) and canonical.get(_METADATA_ONLY):
            # Our own row: only hand over presentation metadata the live dict lacks, never payload.
            for key in _LIVE_MISSING_METADATA:
                if canonical.get(key) is not None and written.get(key) is None:
                    written[key] = canonical[key]
        elif isinstance(canonical, dict) and canonical.get(_CONTENT_ONLY):
            written["content"] = canonical["content"]
        elif isinstance(canonical, dict):
            for key in _SYNC_FIELDS:
                if key in canonical and canonical[key] is not None:
                    written[key] = canonical[key]
                elif key not in ("role", "content"):
                    written.pop(key, None)
