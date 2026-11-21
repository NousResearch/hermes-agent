"""The turn path's transcript flush, as the probes drive it: rows shaped as ``_db_flush_row`` emits them, written with
``agent.session_persistence._db_flush_write`` for an agent that carries its store handle (``agent.persistence``)."""

from __future__ import annotations

from typing import Any

from agent.session_persistence import _ROW_REASONING_KEYS, _db_flush_write

_COLUMNS = ("tool_calls", "tool_name", "tool_call_id", "display_kind", "_compressed_summary", "api_content",
            "display_metadata", "platform_message_id", "observed", "effect_disposition", "finish_reason", "token_count",
            *_ROW_REASONING_KEYS)


def flush_row(content: str, role: str = "assistant") -> dict[str, Any]:
    """One flush row with ``content``; every other column empty."""
    return {"role": role, "content": content, "timestamp": 1_800_000_000.0, **dict.fromkeys(_COLUMNS)}


def flush(agent: Any, *contents: str) -> None:
    """Flush ``contents`` as the turn's new rows, in one write (``_db_flush_write``)."""
    rows = [flush_row(content) for content in contents]
    messages = [{"role": row["role"], "content": row["content"]} for row in rows]
    _db_flush_write(agent, rows, messages, messages)
