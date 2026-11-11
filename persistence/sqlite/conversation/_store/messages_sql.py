"""The transcript statements and the row helpers the message functions share."""


from __future__ import annotations
import json
import logging
import re
from typing import Any
from session_rules.turns import (
    MODEL_ONLY_DISPLAY_METADATA_KEY,
)
from persistence.sqlite._schema_ddl import _sql_json_extract
from persistence.sqlite.conversation._store.rules import (
    _sanitize_surrogates, coerce_epoch,
)


logger = logging.getLogger("hermes_state")  # caplog tests pin the origin module's name


_INSERT_MESSAGE_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind,
                   display_metadata, display_identity, message_uid, absorbed_message_uids, tool_call_uids,
                   tool_call_uid)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


_MESSAGE_WRITE_COLUMNS = tuple(
    re.findall(r"\w+", _INSERT_MESSAGE_SQL.split("(", 1)[1].split(")", 1)[0])
)


_MESSAGE_SCHEMA_KEYS = frozenset(_MESSAGE_WRITE_COLUMNS) | {"id", "compacted", "display_order"}


_BUMP_GENERATION_SQL = """
            INSERT INTO conversation_generations (source, session_key, generation)
            VALUES (?, ?, 1)
            ON CONFLICT(source, session_key) DO UPDATE
                SET generation = conversation_generations.generation + 1
            """


_TURN_LEASE_ROW_SQL = "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?"


_DELETE_COMPRESSION_LOCK_SQL = "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?"


_DISPLAY_ACTIVE_CLAUSE = " AND (active = 1 OR compacted = 1)"


DISPLAY_VISIBLE_SQL = (
    f" AND COALESCE({_sql_json_extract('display_metadata', '$.' + MODEL_ONLY_DISPLAY_METADATA_KEY)}, 0) = 0")


_DISPLAY_META_ROW_SQL = "SELECT display_metadata FROM messages WHERE id = ? AND session_id IN ({ids})" + _DISPLAY_ACTIVE_CLAUSE


_DISPLAY_INDEX_MISSING_SQL = ("SELECT 1 FROM messages WHERE session_id = ?" + _DISPLAY_ACTIVE_CLAUSE
                              + " AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1")


_ACTIVE_IDS_SQL = "SELECT id FROM messages WHERE session_id = ? AND active = 1 ORDER BY id"


_LIVE_IDENTITY_SQL = ("SELECT id, role, content, tool_call_id, tool_calls, message_uid FROM messages "
                      "WHERE session_id = ? AND active = 1 ORDER BY id LIMIT ?")


_SET_COUNTERS_SQL = "UPDATE sessions SET message_count = ?, tool_call_count = ?"


_RESET_COUNTERS_SQL = "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?"


_SET_DISPLAY_META_SQL = "UPDATE messages SET display_metadata = ? WHERE id = ?"


_ARCHIVE_ACTIVE_SQL = "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ? AND active = 1"


_SHADOWED_CHECKPOINT_ROWS_SQL = ("SELECT id, codex_reasoning_items FROM messages WHERE session_id = ? AND active = 1 "
    "AND role = 'assistant' AND id < ? AND codex_reasoning_items LIKE '%\"compaction\"%'")


_SET_CODEX_REASONING_SQL = "UPDATE messages SET codex_reasoning_items = ? WHERE id = ?"


_INVALID = object()  # _json_or sentinel where the fallback must be distinguishable from JSON null


def _coerce_timestamp(value: Any, default: float) -> float:
    """Explicit message timestamp (datetime or number) or *default* when invalid or outside the sane
    epoch window — the write-side twin of the readers' ``coerce_epoch``: a bad row is never persisted."""
    result = coerce_epoch(value, field="message timestamp")
    if result is None:
        return default
    # SQLite reads both signed zero spellings back as 0.0. Hash the value
    # in that stored form so -0.0 and 0.0 retain the historical SQL/Python
    # identity equality.
    return 0.0 if result == 0.0 else result


def _parse_tool_calls(tool_calls: Any) -> Any:
    """tool_calls is a list (live agent) or JSON string (import/export); parse so json.dumps never double-encodes."""
    if not isinstance(tool_calls, str):
        return tool_calls
    try:
        return json.loads(tool_calls)
    except (json.JSONDecodeError, TypeError):
        return []


def _tool_calls_count(tool_calls: Any) -> int:
    return 0 if tool_calls is None else (len(tool_calls) if isinstance(tool_calls, list) else 1)


def _tool_calls_len(raw: Any, scalar: int = 0) -> int:
    """Count of a stored ``tool_calls`` column: list length, *scalar* for a truthy non-list, else 0."""
    parsed = _parse_tool_calls(raw)
    return len(parsed) if isinstance(parsed, list) else (scalar if parsed else 0)


def _scrub_surrogates(value: Any) -> Any:
    """Lone surrogates make sqlite3 raise UnicodeEncodeError and abort the whole write."""
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def _stale_holder(row, now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process."""
    from persistence.sqlite.conversation._store.core import _compression_lock_holder_process_is_dead
    return float(row["expires_at"]) <= now or _compression_lock_holder_process_is_dead(row["holder"])
