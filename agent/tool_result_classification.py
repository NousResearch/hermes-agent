"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})
TOOL_RESULT_ENVELOPE_KEYS = frozenset({"result", "content", "structuredContent"})


def unwrap_tool_result_envelope(result: Any, *, max_depth: int = 3) -> Any:
    """Decode bounded MCP-style result envelopes without scanning prose.

    MCP adapters commonly return application payloads as
    ``{"result": "{...}"}``, optionally alongside ``structuredContent``.
    Failure classification must inspect that payload rather than treating the
    transport-level wrapper as a successful result.
    """

    value = result
    for _ in range(max_depth + 1):
        if isinstance(value, str):
            try:
                value = json.loads(value.strip())
            except Exception:
                return value
        if not isinstance(value, dict):
            return value
        if any(key in value for key in ("ok", "success", "error")):
            return value
        if not value or not set(value).issubset(TOOL_RESULT_ENVELOPE_KEYS):
            return value
        if "structuredContent" in value and value["structuredContent"] is not None:
            value = value["structuredContent"]
        elif "result" in value:
            value = value["result"]
        else:
            value = value.get("content")
    return value


def structured_tool_failure_message(result: Any) -> str | None:
    """Return a structured failure message, including nested MCP payloads."""

    data = unwrap_tool_result_envelope(result)
    if not isinstance(data, dict):
        return None

    error = data.get("error")
    failed = data.get("ok") is False or data.get("success") is False
    # A concrete error outranks an explicit success marker, while JSON-falsey
    # placeholders such as ``null`` and ``false`` are not failures by
    # themselves. This keeps the structured classifier value-aware without
    # changing the separate generic string heuristic used by display callers.
    if error:
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            return str(message or "tool returned a structured error")
        if error is True:
            return "tool returned a structured error"
        return str(error)
    if failed:
        message = data.get("message")
        return str(message or "tool returned an unsuccessful result")
    return None


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hermes session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
