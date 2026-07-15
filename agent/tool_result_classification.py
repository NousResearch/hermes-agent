"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any, Mapping


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


def terminal_exit_code_failure_adapter(result: str | None) -> tuple[bool, str]:
    """Classify only the terminal tool's owned integer ``exit_code`` field."""

    data = _json_mapping(result)
    if data is None:
        return False, ""
    exit_code = data.get("exit_code")
    if type(exit_code) is int and exit_code != 0:
        return True, f" [exit {exit_code}]"
    return False, ""


def file_operation_failure_adapter(result: str | None) -> tuple[bool, str]:
    """Classify the exact built-in file-operation result envelope."""

    data = _json_mapping(result)
    if data is None:
        return False, ""
    error = data.get("error")
    if isinstance(error, str) and error:
        return True, " [error]"
    return False, ""


def classify_registered_tool_result_failure(
    tool_name: str,
    result: str | None,
) -> tuple[bool, str]:
    """Use only the currently registered implementation's exact adapter.

    Arbitrary top-level ``success``, ``ok``, ``status`` and ``error`` fields
    are model-visible business data, not runtime authority.  A tool result may
    affect loop accounting only when that exact registry entry opted into a
    mechanical adapter for its owned envelope.  Replacing/deregistering the
    implementation also replaces/removes that authority.
    """

    if result is None:
        return False, ""
    try:
        from tools.registry import registry

        entry = registry.get_entry(str(tool_name or ""))
        adapter = getattr(entry, "result_failure_adapter", None)
    except Exception:
        adapter = None
    if not callable(adapter):
        return False, ""
    try:
        classified = adapter(result)
    except Exception:
        return False, ""
    if (
        not isinstance(classified, tuple)
        or len(classified) != 2
        or type(classified[0]) is not bool
        or not isinstance(classified[1], str)
    ):
        return False, ""
    return classified


def _json_mapping(result: str | None) -> Mapping[str, Any] | None:
    if not isinstance(result, str):
        return None
    try:
        value = json.loads(result.strip())
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None
