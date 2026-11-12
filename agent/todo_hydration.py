"""Shared decoding and paired-history lookup for todo hydration."""

import json
import logging
from typing import Any

from agent.message_sanitization import coalesce_tool_call_id
from tools.todo_tool import MAX_TODO_RESULT_CHARS, is_todo_tool_call

logger = logging.getLogger("run_agent")


def decode_todo_result(content: Any, *, session_id: str | None = None) -> tuple | None:
    """Decode a size-bounded result whose Todo-tool pairing the caller has verified.

    Item validation and revision coercion remain the responsibility of TodoStore.restore.
    """
    if not isinstance(content, str):
        return None
    if len(content) > MAX_TODO_RESULT_CHARS:
        logger.warning("Skipping oversized todo tool response during hydration: "
                       "session=%s chars=%d", session_id or "none", len(content))
        return None
    if '"todos"' not in content:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if "todos" in data and isinstance(data["todos"], list):
        return data["todos"], data.get("revision", 1)
    return None


def assistant_has_todo_tool_call(assistant_msg: dict[str, Any], tool_call_id: str) -> bool:
    """Match canonical/legacy Todo calls, including the tool_call bridge."""
    tool_calls = assistant_msg.get("tool_calls")
    return isinstance(tool_calls, list) and any(
        coalesce_tool_call_id(tc) == tool_call_id and is_todo_tool_call(tc) for tc in tool_calls
    )


def tool_response_matches_todo_call(history: list[dict[str, Any]], tool_index: int) -> bool:
    """Require a matching call in the nearest assistant, without crossing user/system boundaries."""
    tool_call_id = history[tool_index].get("tool_call_id") if 0 <= tool_index < len(history) else None
    if not tool_call_id:
        return False
    for prior in reversed(history[:tool_index]):
        role = prior.get("role")
        if role == "assistant":
            return assistant_has_todo_tool_call(prior, tool_call_id)
        if role in {"user", "system"}:
            return False
    return False


def latest_todo_response(history: list[dict[str, Any]], *, session_id: str | None = None) -> tuple | None:
    """Read the newest paired, decodable result; an empty list is an authoritative clear."""
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") != "tool" or not tool_response_matches_todo_call(history, idx):
            continue
        found = decode_todo_result(history[idx].get("content", ""), session_id=session_id)
        if found is not None:
            return found
    return None
