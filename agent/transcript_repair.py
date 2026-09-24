"""Transcript repair for SessionDB batch appends: reconcile in-memory rows that already own a committed SQLite
row (in-place rewrite, assistant concurrent-winner adoption, watermark-compaction clone lookup) and sync
markers after commit."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Dict, List

from agent.context_compressor import _DB_PERSISTED_MARKER


def is_content_blank(content: Any) -> bool:
    """True when decoded message content is None, whitespace-only, or has no visible text parts."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return False


def resolve_and_repair_transcript_batch(
    conn: sqlite3.Connection,
    session_id: str,
    messages: List[Dict[str, Any]],
    encode_content_fn: Callable[[Any], Any],
    decode_content_fn: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    """Partition a message batch within an active write transaction. A message carrying an existing integer
    ``_row_id`` targets its active SQLite row of the same role (or the active clone a watermark compaction
    made of it) and is never inserted again. A non-assistant row is rewritten in place; an assistant row is
    updated only when blank, otherwise (concurrent winner) its canonical content is adopted without
    overwrite. Returns the messages that must be inserted as fresh rows."""
    inserted_rows: List[Dict[str, Any]] = []
    for msg in messages:
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        target_row = None
        if isinstance(existing_row_id, int):
            target_row = _active_row(conn, session_id, existing_row_id, msg.get("role", "unknown"))
        if target_row is None:
            inserted_rows.append(msg)
            continue
        target_id = int(target_row["id"])
        msg["_row_id"] = target_id
        if msg.get("role") != "assistant":
            # A dict that owns a live row reaches the flush only because an in-place repair popped
            # its marker to get the change written (_DB_PERSISTED_MARKER contract): rewrite that
            # row, since appending would save the message twice.
            conn.execute(
                "UPDATE messages SET content = ?, api_content = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (encode_content_fn(msg.get("content")), msg.get("api_content"), target_id, session_id),
            )
            continue
        decoded = decode_content_fn(target_row["content"])
        if is_content_blank(decoded):
            conn.execute(
                "UPDATE messages SET content = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (encode_content_fn(msg.get("content")), target_id, session_id),
            )
        else:
            msg["_canonical_content"] = decoded  # concurrent winner: adopt, don't overwrite
    return inserted_rows


def _active_row(conn: sqlite3.Connection, session_id: str, row_id: int, role: str):
    """The active ``role`` row for ``row_id``, or the active clone a watermark compaction made of it."""
    row = conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE id = ? AND session_id = ?",
        (row_id, session_id),
    ).fetchone()
    if row is None or row["role"] != role:
        return None
    if int(row["active"] or 0) == 1:
        return row
    # Watermark compaction soft-archived the concurrent tail and cloned it.
    return conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE session_id = ? AND active = 1 AND role = ? "
        "AND timestamp IS ? AND id != ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id, role, row["timestamp"], row["id"]),
    ).fetchone()


def sync_flushed_message_markers(batch_msgs: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> None:
    """Stamp _DB_PERSISTED_MARKER and sync canonical durable fields onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        if isinstance(row.get("timestamp"), (int, float)):
            written["timestamp"] = row["timestamp"]
        if "_canonical_content" in row:
            written["content"] = row["_canonical_content"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
