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


def classify_memory_result(data: Any) -> tuple[bool, str] | None:
    """Classify a ``memory`` tool payload as ``(is_failure, suffix)``.

    Shared by ``agent.display._detect_tool_failure`` and
    ``agent.tool_guardrails.classify_tool_failure`` so the CLI tag and the
    guardrail counter can never disagree about a memory result.

    ``done=True`` marks a payload the tool considers SETTLED — both the
    ordinary success and the terminal graceful degradation returned once
    consolidation has failed too often in one turn (#42405). Neither is a
    failure the model should react to, and counting one feeds the same-tool
    halt counter, which aborts the turn and suppresses the user-facing reply:
    the exact outcome #42405 exists to prevent.

    The ``done`` check comes FIRST and that order is load-bearing. An
    at-capacity refusal is routed through the same consolidation-failure
    budget, so the Nth consecutive one comes back as the terminal payload —
    at which point the model has been told to stop retrying and a ``[full]``
    tag is no longer actionable. (The two cannot both match in any case: the
    terminal payload carries a fixed error text that never mentions a limit.)

    ``None`` means "no memory verdict" — the caller keeps applying its own
    generic rules, exactly as each did before this was factored out. That is
    why the return is optional rather than ``(False, "")``.
    """
    if not isinstance(data, dict):
        return None
    if data.get("done") is True:
        return False, ""
    if data.get("success") is False and "exceed the limit" in data.get("error", ""):
        return True, " [full]"
    return None


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
