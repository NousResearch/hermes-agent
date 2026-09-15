"""Transcript repair for SessionDB batch appends: reconcile in-memory assistant rows with committed SQLite
rows (blank-row in-place update, concurrent-winner adoption, watermark-compaction clone lookup) and sync
markers after commit.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Dict, List, Tuple

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


def _is_row_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _logical_tool_id(msg: Dict[str, Any]) -> str | None:
    """Durable tool-call identity: the tool result id, else the first assistant tool_call id."""
    tool_id = msg.get("tool_call_id")
    if isinstance(tool_id, str) and tool_id:
        return tool_id
    calls = msg.get("tool_calls")
    if isinstance(calls, list) and calls:
        first = calls[0]
        if isinstance(first, dict):
            cid = first.get("id")
            if isinstance(cid, str) and cid:
                return cid
    return None


def _adoption_key(msg: Dict[str, Any], encode_content_fn: Callable[[Any], Any]) -> Tuple[Any, ...] | None:
    """Identity safe for same-batch collapse; absent timestamps are not durable identity."""
    timestamp = msg.get("timestamp")
    if timestamp is None:
        return None
    return (
        "body",
        msg.get("role", "unknown"),
        timestamp,
        encode_content_fn(msg.get("content")),
        _logical_tool_id(msg),
    )


def _encode_tool_calls_column(tool_calls: Any) -> Optional[str]:
    if not tool_calls:
        return None
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except (json.JSONDecodeError, TypeError):
            return None
    return json.dumps(tool_calls) if tool_calls else None


def resolve_and_repair_transcript_batch(
    conn: sqlite3.Connection,
    session_id: str,
    messages: List[Dict[str, Any]],
    encode_content_fn: Callable[[Any], Any],
    decode_content_fn: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    """Partition a message batch within an active write transaction.

    An assistant message carrying an existing integer ``_row_id`` targets its active SQLite row
    (or the active clone a watermark compaction made of it): a blank row is updated in place; a
    non-blank one (concurrent winner) has its canonical content adopted without overwrite unless
    the live dict is a repair mutation.

    Rematerialized copies (compaction/repair rebuilt dicts with no usable ``_row_id``) are adopted
    onto the already-ACTIVE row with the same logical identity instead of being INSERTed again.
    Returns the messages that must be inserted as fresh rows.
    """
    inserted_rows: List[Dict[str, Any]] = []
    seen_row_ids: set[int] = set()
    seen_keys: set[Tuple[Any, ...]] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            inserted_rows.append(msg)
            continue
        target_row = _lookup_active_target(conn, session_id, msg, encode_content_fn)
        key = _adoption_key(msg, encode_content_fn)
        if target_row is not None and int(target_row["id"]) in seen_row_ids:
            continue
        if target_row is None and key is not None and key in seen_keys:
            continue
        if target_row is None:
            inserted_rows.append(msg)
            if key is not None:
                seen_keys.add(key)
            continue
        target_id = int(target_row["id"])
        seen_row_ids.add(target_id)
        if key is not None:
            seen_keys.add(key)
        msg["_row_id"] = target_id
        decoded = decode_content_fn(target_row["content"])
        if msg.get("_repair_mutated"):
            conn.execute(
                "UPDATE messages SET content = ?, tool_calls = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (
                    encode_content_fn(msg.get("content")),
                    _encode_tool_calls_column(msg.get("tool_calls")),
                    target_id,
                    session_id,
                ),
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


def _lookup_active_target(
    conn: sqlite3.Connection,
    session_id: str,
    msg: Dict[str, Any],
    encode_content_fn: Callable[[Any], Any],
):
    """Active row this live dict should reuse, or None if it is a genuine new message."""
    existing_row_id = msg.get("_row_id")
    role = msg.get("role", "unknown")
    if _is_row_id(existing_row_id):
        if role == "assistant":
            row = _active_assistant_row(conn, session_id, existing_row_id)
            if row is not None:
                return row
        else:
            row = conn.execute(
                "SELECT id, role, active, timestamp, content, tool_calls, tool_call_id FROM messages "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (existing_row_id, session_id),
            ).fetchone()
            if row is not None:
                return row
    timestamp = msg.get("timestamp")
    if timestamp is None:
        return None
    tool_id = _logical_tool_id(msg)
    if tool_id is not None:
        if role == "assistant":
            identity = "(json_extract(tool_calls, '$[0].id') = ? OR tool_call_id = ?)"
            params = (session_id, role, timestamp, encode_content_fn(msg.get("content")), tool_id, tool_id)
        else:
            identity = "tool_call_id = ?"
            params = (session_id, role, timestamp, encode_content_fn(msg.get("content")), tool_id)
        return conn.execute(
            "SELECT id, role, active, timestamp, content, tool_calls, tool_call_id FROM messages "
            f"WHERE session_id = ? AND active = 1 AND role = ? AND timestamp IS ? AND content IS ? AND {identity} "
            "ORDER BY id LIMIT 1", params,
        ).fetchone()
    return conn.execute(
        "SELECT id, role, active, timestamp, content, tool_calls, tool_call_id FROM messages "
        "WHERE session_id = ? AND active = 1 AND role = ? AND timestamp IS ? AND content IS ? "
        "ORDER BY id LIMIT 1",
        (session_id, role, msg.get("timestamp"), encode_content_fn(msg.get("content"))),
    ).fetchone()


def _active_assistant_row(conn: sqlite3.Connection, session_id: str, row_id: int):
    """The active assistant row for ``row_id``, or the active clone a watermark compaction made of it."""
    row = conn.execute(
        "SELECT id, role, active, timestamp, content, tool_calls, tool_call_id FROM messages "
        "WHERE id = ? AND session_id = ?",
        (row_id, session_id),
    ).fetchone()
    if row is None or row["role"] != "assistant":
        return None
    if int(row["active"] or 0) == 1:
        return row
    # Watermark compaction soft-archived the concurrent tail and cloned it.
    return conn.execute(
        "SELECT id, role, active, timestamp, content, tool_calls, tool_call_id FROM messages "
        "WHERE session_id = ? AND active = 1 AND role = 'assistant' "
        "AND timestamp IS ? AND id != ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id, row["timestamp"], row["id"]),
    ).fetchone()


def sync_flushed_message_markers(batch_msgs: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> None:
    """Stamp _DB_PERSISTED_MARKER and sync canonical row ID / content onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        if "_canonical_content" in row:
            written["content"] = row["_canonical_content"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
