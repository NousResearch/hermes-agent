"""Transcript repair for SessionDB batch appends: reconcile in-memory rows with committed SQLite rows
(in-place sanitizer rewrites, assistant blank-row repair, concurrent-winner adoption, watermark-compaction
clone lookup) and sync markers after commit."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Dict, List, Mapping

from agent.context_compressor import _DB_PERSISTED_MARKER


_DB_ROW_SNAPSHOT = "_db_row_snapshot"
_CANONICAL_ROW = "_canonical_row"
_REPAIR_COLUMNS = (
    "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition", "token_count",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "platform_message_id", "observed", "_compressed_summary", "api_content",
    "display_kind", "display_metadata",
)
_SYNC_FIELDS = (
    "role", "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition", "token_count",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "message_id", "platform_message_id", "observed",
    "_compressed_summary", "api_content", "display_kind", "display_metadata",
)


def transcript_row_snapshot(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Serialized durable values used as the CAS version, or ``None`` for a partial SELECT."""
    keys = set(row.keys()) if hasattr(row, "keys") else set(row)
    if not set(_REPAIR_COLUMNS) <= keys:
        return None
    return {column: row[column] for column in _REPAIR_COLUMNS}


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
    serialize_message_fn: Callable[[Dict[str, Any], float], Mapping[str, Any]],
    decode_row_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve row-addressed rewrites without appending duplicates or replacing concurrent winners.

    A durable row snapshot is a compare-and-swap version for sanitizer rewrites. Watermark-compaction clones
    are matched by their copied payload identity, not timestamp alone. Legacy blank assistant rows retain the
    narrow interrupted-stream content repair. Returns only rows that need fresh inserts.
    """
    inserted_rows: List[Dict[str, Any]] = []
    for msg in messages:
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        target_row = None
        if isinstance(existing_row_id, int):
            target_row = _active_message_row(conn, session_id, existing_row_id, role)
        if target_row is None:
            inserted_rows.append(msg)
            continue

        target_id = int(target_row["id"])
        msg["_row_id"] = target_id
        serialized = serialize_message_fn(msg, float(target_row["timestamp"]))
        expected = msg.get(_DB_ROW_SNAPSHOT)
        has_snapshot = isinstance(expected, dict) and all(column in expected for column in _REPAIR_COLUMNS)
        repaired = _compare_and_swap_row(conn, session_id, target_row, serialized, expected) if has_snapshot else False
        if not has_snapshot and role == "assistant" and is_content_blank(decode_content_fn(target_row["content"])):
            # Blank assistant rows are the pre-existing interrupted-stream repair path. Keep its narrow
            # content-only CAS for live dicts that predate durable row snapshots.
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ? AND session_id = ? AND content IS ?",
                (encode_content_fn(msg.get("content")), target_id, session_id, target_row["content"]),
            )

        final_row = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND session_id = ?", (target_id, session_id)
        ).fetchone()
        msg["timestamp"] = final_row["timestamp"]
        msg[_DB_ROW_SNAPSHOT] = transcript_row_snapshot(final_row)
        msg[_CANONICAL_ROW] = decode_row_fn(final_row)
    return inserted_rows


def _compare_and_swap_row(
    conn: sqlite3.Connection,
    session_id: str,
    target_row: Mapping[str, Any],
    serialized: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Rewrite one durable payload only while it still equals the live dict's last committed snapshot."""
    old_identity = target_row["display_identity"]
    old_peer_ids = [
        int(row["id"])
        for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND id != ? "
            "AND (active = 1 OR compacted = 1) AND display_identity IS ?",
            (session_id, target_row["id"], old_identity),
        ).fetchall()
    ] if old_identity is not None else []

    assignments = ", ".join(f"{column} = ?" for column in _REPAIR_COLUMNS)
    predicates = " AND ".join(f"{column} IS ?" for column in _REPAIR_COLUMNS)
    params = [serialized[column] for column in _REPAIR_COLUMNS]
    params += [int(target_row["id"]), session_id]
    params += [expected[column] for column in _REPAIR_COLUMNS]
    cur = conn.execute(
        f"UPDATE messages SET {assignments} WHERE id = ? AND session_id = ? AND {predicates}", params,
    )
    if cur.rowcount != 1:
        return False
    _restore_display_index(conn, session_id, target_row, serialized, old_identity, old_peer_ids)
    return True


def _restore_display_index(
    conn: sqlite3.Connection,
    session_id: str,
    target_row: Mapping[str, Any],
    serialized: Mapping[str, Any],
    old_identity: Any,
    old_peer_ids: List[int],
) -> None:
    """Restore display identities/orders invalidated by the payload-update trigger."""
    if old_peer_ids:
        placeholders = ", ".join("?" for _ in old_peer_ids)
        old_order = min(old_peer_ids)
        conn.execute(
            f"UPDATE messages SET display_identity = ?, display_order = ? "
            f"WHERE session_id = ? AND id IN ({placeholders})",
            (old_identity, old_order, session_id, *old_peer_ids),
        )

    target_id = int(target_row["id"])
    new_identity = serialized["display_identity"]
    visible = bool(target_row["active"] or target_row["compacted"])
    if not visible:
        conn.execute(
            "UPDATE messages SET display_identity = ?, display_order = ? WHERE id = ? AND session_id = ?",
            (new_identity, target_id, target_id, session_id),
        )
        return

    peers = conn.execute(
        "SELECT id, display_order FROM messages WHERE session_id = ? AND id != ? "
        "AND (active = 1 OR compacted = 1) AND display_identity IS ?",
        (session_id, target_id, new_identity),
    ).fetchall()
    new_order = min(
        [target_id]
        + [int(peer["display_order"] if peer["display_order"] is not None else peer["id"]) for peer in peers]
    )
    conn.execute(
        "UPDATE messages SET display_identity = ?, display_order = ? WHERE id = ? AND session_id = ?",
        (new_identity, new_order, target_id, session_id),
    )
    if peers:
        conn.executemany(
            "UPDATE messages SET display_order = ? WHERE id = ? AND session_id = ?",
            [(new_order, int(peer["id"]), session_id) for peer in peers],
        )


def _active_message_row(conn: sqlite3.Connection, session_id: str, row_id: int, role: str):
    """The same-role active clone for ``row_id``, or the addressed inactive row when no clone exists."""
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ? AND session_id = ?", (row_id, session_id)
    ).fetchone()
    if row is None or row["role"] != role:
        return None
    if int(row["active"] or 0) == 1:
        return row
    # Watermark compaction copies the complete durable row payload and display identity byte-for-byte.
    # Timestamp alone is not an identity: externally supplied event timestamps may collide across unrelated
    # messages. Legacy rows without an indexed identity cannot be resolved safely, so retain the addressed row.
    if row["display_identity"] is None:
        return row
    payload_predicates = " AND ".join(f"{column} IS ?" for column in _REPAIR_COLUMNS)
    clones = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? AND active = 1 AND role = ? "
        "AND display_identity IS ? AND timestamp IS ? AND id != ? AND "
        f"{payload_predicates} ORDER BY id DESC LIMIT 2",
        (
            session_id,
            role,
            row["display_identity"],
            row["timestamp"],
            row["id"],
            *(row[column] for column in _REPAIR_COLUMNS),
        ),
    ).fetchall()
    return clones[0] if len(clones) == 1 else row


def sync_flushed_message_markers(batch_msgs: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> None:
    """Stamp persistence markers and sync canonical durable fields onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        if isinstance(row.get("timestamp"), (int, float)):
            written["timestamp"] = row["timestamp"]
        if isinstance(row.get(_DB_ROW_SNAPSHOT), dict):
            written[_DB_ROW_SNAPSHOT] = dict(row[_DB_ROW_SNAPSHOT])
        canonical = row.get(_CANONICAL_ROW)
        if isinstance(canonical, dict):
            for key in _SYNC_FIELDS:
                if key in canonical and canonical[key] is not None:
                    written[key] = canonical[key]
                elif key not in ("role", "content"):
                    written.pop(key, None)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
