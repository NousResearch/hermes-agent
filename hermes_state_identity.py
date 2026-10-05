"""Durable message-identity columns for SessionDB (schema v31): the codec between the live identity keys
(``agent.message_metadata``) and the ``message_uid`` / ``absorbed_message_uids`` / ``tool_call_uids`` /
``tool_call_uid`` columns, shared by the transcript writers, restore, and transcript repair."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MESSAGE_UID, TOOL_CALL_UID, TOOL_CALL_UIDS, uid_list
from agent.message_sanitization import coalesce_tool_call_id
from hermes_state_common import _json_or


def _live_or_column(msg: Mapping[str, Any], live_key: str, column: str) -> Any:
    """Flushed dicts and batch rows carry the live key; import payloads carry the column name."""
    return msg.get(live_key) if live_key in msg else msg.get(column)


def _uid_list(value: Any) -> List[str]:
    """Normalize a uid list (a live list, or the JSON text an export/import carries) to unique non-empty
    strings in order; anything else is ``[]``."""
    if isinstance(value, str):
        value = _json_or(value, [], "Failed to deserialize a message uid list, falling back to []")
    return uid_list(value)


def _uid_map(value: Any) -> Dict[str, Any]:
    """Normalize a ``{tool call id: uid}`` map (a live dict, or the JSON text an export/import carries): a
    non-empty string uid, or a list of them for a provider id repeated inside one row (one per occurrence,
    see ``merge_tool_call_uids``); anything else is dropped, and a non-map is ``{}``."""
    if isinstance(value, str):
        value = _json_or(value, {}, "Failed to deserialize a tool-call uid map, falling back to {}")
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and k and (
        (isinstance(v, str) and v) or (isinstance(v, list) and v and all(isinstance(u, str) and u for u in v)))}


def _tool_call_uid_map(msg: Mapping[str, Any]) -> Dict[str, Any]:
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


def _stable_tool_key(row: Any) -> Optional[Tuple[Any, ...]]:
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
