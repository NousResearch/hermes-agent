"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


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


def is_skill_view_dedup_result(tool_name: str, result: Any) -> bool:
    """Return True for skill_view's model-facing repeat-read control result.

    Repeat reads deliberately use ``success: false`` plus an ``error`` message
    to make the model stop re-reading and act on content already in context.
    Operational failure classifiers must not count that benign cache hit as a
    failed execution, however, so recognize the complete typed shape here.
    """
    if tool_name != "skill_view":
        return False
    if isinstance(result, str):
        try:
            result = json.loads(result.strip())
        except Exception:
            return False
    return (
        isinstance(result, dict)
        and result.get("success") is False
        and result.get("status") == "deduplicated"
        and result.get("dedup") is True
        and result.get("content_returned") is False
        and bool(result.get("error"))
    )


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
