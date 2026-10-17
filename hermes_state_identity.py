"""Durable message-identity columns for SessionDB (schema v31): the codec between the live identity keys
(``agent.message_metadata``) and the ``message_uid`` / ``absorbed_message_uids`` / ``tool_call_uids`` /
``tool_call_uid`` columns, shared by the transcript writers, restore, and transcript repair."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MESSAGE_UID, TOOL_CALL_UID, TOOL_CALL_UIDS, uid_list
from agent.message_sanitization import coalesce_tool_call_id
from hermes_state_common import _json_or


USER_DISPLAY_VERSION_KEY = "display_user_occurrence_version"
USER_DISPLAY_VERSION = "1"


def authored_user_occurrence(db, row):
    """UID identity only for visible authored users, using the canonical carrier classifier."""
    from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY, user_originated_turn_view

    if row["role"] != "user" or "message_uid" not in row.keys() or not row["message_uid"]:
        return None
    meta = db._decode_display_metadata(row["display_metadata"]) or {}
    if meta.get(MODEL_ONLY_DISPLAY_METADATA_KEY):
        return None
    message = {"role": "user", "content": db._decode_content(row["content"]),
               "display_kind": row["display_kind"], "display_metadata": meta,
               "_compressed_summary": bool(row["_compressed_summary"]) if "_compressed_summary" in row.keys() else False}
    if user_originated_turn_view(message) is None:
        return None
    return ("user-occurrence-v1", row["message_uid"],
            "steer" if row["display_kind"] == "steer" else "user")


def user_display_index_current(conn):
    try:
        return conn.execute("SELECT 1 FROM state_meta WHERE key = ? AND value = ?",
                            (USER_DISPLAY_VERSION_KEY, USER_DISPLAY_VERSION)).fetchone() is not None
    except sqlite3.OperationalError as exc:
        if "no such table: state_meta" not in str(exc):
            raise
        return False


def repair_user_display_slice(db, conn, lower, upper):
    """Repair derived USER index columns only; seek first compatible peer by the UID index."""
    fields = ("id, role, content, message_uid, display_kind, display_metadata, "
              "_compressed_summary, display_order, display_identity")
    rows = conn.execute(f"SELECT {fields} FROM messages WHERE id > ? AND id <= ? "
                        "AND role = 'user' AND message_uid IS NOT NULL "
                        "AND (active = 1 OR compacted = 1) ORDER BY id", (lower, upper)).fetchall()
    for row in rows:
        key = authored_user_occurrence(db, row)
        if key is None:
            continue
        first = conn.execute(f"SELECT {fields} FROM messages INDEXED BY idx_messages_display_uid "
                             "WHERE session_id = (SELECT session_id FROM messages WHERE id = ?) "
                             "AND message_uid = ? AND role = 'user' "
                             "AND (active = 1 OR compacted = 1) ORDER BY id",
                             (row["id"], row["message_uid"]))
        try:
            order = next(peer["id"] for peer in first if authored_user_occurrence(db, peer) == key)
        finally:
            first.close()
        identity = db._display_identity(key)
        if (row["display_order"], row["display_identity"]) != (order, identity):
            conn.execute("UPDATE messages SET display_order = ?, display_identity = ? WHERE id = ?",
                         (order, identity, row["id"]))


def legacy_user_display_group(db, row, indexed):
    """Correct authored users only; populated legacy non-user slots are authoritative."""
    if indexed and authored_user_occurrence(db, row) is None:
        return ("display-slot", row["display_order"]), row["display_order"]
    return db._display_identity(db._display_dedupe_key(row)), row["id"]


def _live_or_column(msg: Mapping[str, Any], live_key: str, column: str) -> Any:
    """Flushed dicts and batch rows carry the live key; import payloads carry the column name."""
    return msg.get(live_key) if live_key in msg else msg.get(column)


def _uid_list(value: Any) -> list[str]:
    """Normalize a uid list (a live list, or the JSON text an export/import carries) to unique non-empty
    strings in order; anything else is ``[]``."""
    if isinstance(value, str):
        value = _json_or(value, [], "Failed to deserialize a message uid list, falling back to []")
    return uid_list(value)


def _uid_map(value: Any) -> dict[str, Any]:
    """Normalize a ``{tool call id: uid}`` map (a live dict, or the JSON text an export/import carries): a
    non-empty string uid, or a list of them for a provider id repeated inside one row (one per occurrence,
    see ``merge_tool_call_uids``); anything else is dropped, and a non-map is ``{}``."""
    if isinstance(value, str):
        value = _json_or(value, {}, "Failed to deserialize a tool-call uid map, falling back to {}")
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and k and (
        (isinstance(v, str) and v) or (isinstance(v, list) and v and all(isinstance(u, str) and u for u in v)))}


def _tool_call_uid_map(msg: Mapping[str, Any]) -> dict[str, Any]:
    return _uid_map(_live_or_column(msg, TOOL_CALL_UIDS, "tool_call_uids"))


def _absorbed_uids_json(msg: Mapping[str, Any]) -> Optional[str]:
    uids = _uid_list(_live_or_column(msg, ABSORBED_MESSAGE_UIDS, "absorbed_message_uids"))
    return json.dumps(uids) if uids else None


def _tool_call_uids_json(msg: Mapping[str, Any]) -> Optional[str]:
    uids = _tool_call_uid_map(msg)
    return json.dumps(uids, sort_keys=True) if uids else None


def _tool_call_uid_or_none(msg: Mapping[str, Any]) -> Optional[str]:
    uid = _live_or_column(msg, TOOL_CALL_UID, "tool_call_uid")
    return uid if isinstance(uid, str) and uid else None


def _live_extends(live_uid: Any, stored_uid: Any) -> bool:
    """A live per-occurrence list that starts with every stored occurrence: a fold's union not yet written."""
    stored = stored_uid if isinstance(stored_uid, list) else [stored_uid]
    return isinstance(live_uid, list) and live_uid[:len(stored)] == stored


def _fill_missing_tool_call_uids(row: Any, msg: MutableMapping[str, Any]) -> None:
    """Before a row-addressed rewrite serializes *msg*: stored uids for calls it still names but whose uid it
    lost (a clone), so the rewrite does not null them. A call the live dict dropped keeps no uid."""
    if not row["tool_call_uids"] or not isinstance(calls := msg.get("tool_calls"), list):
        return
    live = msg.get(TOOL_CALL_UIDS)
    live = live if isinstance(live, dict) else {}
    named = {coalesce_tool_call_id(tc) for tc in calls}
    if missing := {k: v for k, v in _uid_map(row["tool_call_uids"]).items() if k in named and k not in live}:
        msg[TOOL_CALL_UIDS] = {**live, **missing}


def _restore_row_identity(row: Any, msg: MutableMapping[str, Any]) -> None:
    """A stored row's uid and tool-call uids onto its live dict (the stored value wins). A live per-occurrence
    list that already holds the stored uid is kept: it is a fold's union the row has not been rewritten
    with yet, and replacing it with the stored single uid would strip the later occurrence."""
    if row[MESSAGE_UID]:
        msg[MESSAGE_UID] = row[MESSAGE_UID]
    if row["tool_call_uids"] and (stored := _uid_map(row["tool_call_uids"])):
        live = msg.get(TOOL_CALL_UIDS)
        live = live if isinstance(live, dict) else {}
        msg[TOOL_CALL_UIDS] = {**live, **{call_id: uid for call_id, uid in stored.items()
                                          if not _live_extends(live.get(call_id), uid)}}
    if row["tool_call_uid"]:
        msg[TOOL_CALL_UID] = row["tool_call_uid"]


def _restore_identity_columns(row: Any, msg: MutableMapping[str, Any]) -> None:
    """The stored identity columns onto a restored dict under their live keys (NULL/empty add nothing)."""
    _restore_row_identity(row, msg)
    if row["absorbed_message_uids"] and (absorbed := _uid_list(row["absorbed_message_uids"])):
        msg[ABSORBED_MESSAGE_UIDS] = absorbed


def _stable_tool_key(row: Any) -> Optional[tuple[Any, ...]]:
    """Display-dedupe key of a tool-calling assistant row built from its stable call ids instead of the arguments a
    prune rewrites (#117750: a pruned carried-forward copy must collapse with its durable original). ``None`` for
    every other row and for an incomplete id set, so the caller keeps the full content key: a tool RESULT row keeps
    its payload in the key, because folding the archived full output into its pruned stub would drop the original
    from compacted history and transcript exports, and distinct id-less calls never merge."""
    if row["role"] != "assistant":
        return None
    calls = _json_or(row["tool_calls"] or "[]", [], "Failed to deserialize tool_calls, falling back to []")
    call_ids = tuple(coalesce_tool_call_id(tc) for tc in calls or ())
    if call_ids and all(call_ids):
        return ("assistant", None, row["timestamp"], row["tool_call_id"], call_ids, row["tool_name"])
    return None
