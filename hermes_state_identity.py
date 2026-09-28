"""Durable message-identity columns for SessionDB (schema v31): the codec between the live identity keys
(``agent.message_metadata``) and the ``message_uid`` / ``absorbed_message_uids`` / ``tool_call_uids`` /
``tool_call_uid`` columns, shared by the transcript writers, restore, and transcript repair."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MESSAGE_UID, TOOL_CALL_UID, TOOL_CALL_UIDS
from hermes_state_common import _json_or


def _uid_list(value: Any) -> List[str]:
    """Normalize a uid list (a live list, or the JSON text an export/import carries) to unique non-empty
    strings in order; anything else is ``[]``."""
    if isinstance(value, str):
        value = _json_or(value, [], "Failed to deserialize a message uid list, falling back to []")
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _uid_list_json(msg: Dict[str, Any], live_key: str, column: str) -> Optional[str]:
    """JSON text for a uid-list column, read from the live key first (flushed dicts, batch rows) and the
    column name second (import payloads); ``None`` when empty."""
    uids = _uid_list(msg.get(live_key) if live_key in msg else msg.get(column))
    return json.dumps(uids) if uids else None


def _uid_map(value: Any) -> Dict[str, str]:
    """Normalize a ``{tool call id: uid}`` map (a live dict, or the JSON text an export/import carries): a
    non-empty string uid, or a list of them for a provider id repeated inside one row (one per occurrence,
    see ``merge_tool_call_uids``); anything else is dropped, and a non-map is ``{}``."""
    if isinstance(value, str):
        value = _json_or(value, {}, "Failed to deserialize a tool-call uid map, falling back to {}")
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and k and (
        (isinstance(v, str) and v) or (isinstance(v, list) and v and all(isinstance(u, str) and u for u in v)))}


def _restore_identity_columns(row: Any, msg: Dict[str, Any]) -> None:
    """The stored identity columns onto a restored dict under their live keys (NULL/empty add nothing)."""
    if row[MESSAGE_UID]:
        msg[MESSAGE_UID] = row[MESSAGE_UID]
    if absorbed := _uid_list(row["absorbed_message_uids"]):
        msg[ABSORBED_MESSAGE_UIDS] = absorbed
    if tool_uids := _uid_map(row["tool_call_uids"]):
        msg[TOOL_CALL_UIDS] = tool_uids
    if row["tool_call_uid"]:
        msg[TOOL_CALL_UID] = row["tool_call_uid"]


def _tool_call_uids_json(msg: Dict[str, Any]) -> Optional[str]:
    uids = _uid_map(msg.get(TOOL_CALL_UIDS) if TOOL_CALL_UIDS in msg else msg.get("tool_call_uids"))
    return json.dumps(uids, sort_keys=True) if uids else None


def _tool_call_uid_or_none(msg: Dict[str, Any]) -> Optional[str]:
    uid = msg.get(TOOL_CALL_UID) if TOOL_CALL_UID in msg else msg.get("tool_call_uid")
    return uid if isinstance(uid, str) and uid else None
