"""Transcript repair for SessionDB batch appends: reconcile in-memory assistant rows with committed SQLite
rows (blank-row in-place update, concurrent-winner adoption, watermark-compaction clone lookup) and sync
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
    """Partition a message batch within an active write transaction. An assistant message carrying an
    existing integer ``_row_id`` targets that SQLite row, or the active clone a watermark compaction made
    of it. An inactive row without a clone is repaired in place without changing its archive/rewind state;
    an active blank row is filled, while an active non-blank row (concurrent winner) has its canonical
    content adopted without overwrite. Returns the messages that must be inserted as fresh rows."""
    inserted_rows: List[Dict[str, Any]] = []
    for msg in messages:
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        target_row = None
        if isinstance(existing_row_id, int) and msg.get("role", "unknown") == "assistant":
            target_row = _active_assistant_row(conn, session_id, existing_row_id)
        if target_row is None:
            inserted_rows.append(msg)
            continue
        target_id = int(target_row["id"])
        decoded = decode_content_fn(target_row["content"])
        msg["_row_id"] = target_id
        if int(target_row["active"] or 0) == 0:
            # The row identity still belongs to this session, but compaction/rewind removed it from the
            # model projection. Persist an in-place sanitizer rewrite without resurrecting the row or
            # appending a second display identity.
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ? AND session_id = ? AND active = 0",
                (encode_content_fn(msg.get("content")), target_id, session_id),
            )
        elif is_content_blank(decoded):
            conn.execute(
                "UPDATE messages SET content = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (encode_content_fn(msg.get("content")), target_id, session_id),
            )
        else:
            msg["_canonical_content"] = decoded  # concurrent winner: adopt, don't overwrite
    return inserted_rows


def _active_assistant_row(conn: sqlite3.Connection, session_id: str, row_id: int):
    """The active clone for ``row_id``, or the addressed inactive assistant when no clone exists."""
    row = conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE id = ? AND session_id = ?",
        (row_id, session_id),
    ).fetchone()
    if row is None or row["role"] != "assistant":
        return None
    if int(row["active"] or 0) == 1:
        return row
    # Watermark compaction soft-archived the concurrent tail and cloned it. Prefer that live identity;
    # otherwise keep the addressed inactive row so a later sanitizer pass cannot append it as new.
    clone = conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE session_id = ? AND active = 1 AND role = 'assistant' "
        "AND timestamp IS ? AND id != ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id, row["timestamp"], row["id"]),
    ).fetchone()
    return clone if clone is not None else row


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
