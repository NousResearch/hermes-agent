"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# Back-reference placeholder written by
# ``agent.context_compressor._prune_old_tool_results`` when an older tool
# result is byte-identical to a newer one. Defined here (rather than in the
# compressor) because both the compressor that WRITES it and the loop
# guardrails that READ it need the same string, and this module is the
# dependency-free leaf both already import.
DUPLICATE_OUTPUT_MARKER = "[Duplicate tool output — same content as a more recent call]"

# Prefix used for recognition, so the marker can gain a suffix later without
# silently breaking every consumer that matches on it.
DUPLICATE_OUTPUT_MARKER_PREFIX = "[Duplicate tool output"


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
        # ``bytes_written`` is the historical proof-of-write field, but the
        # tool may instead report a verified content hash. Either is
        # sufficient evidence that the write reached disk; requiring only the
        # former silently misses real mutations and leaves stale no-progress
        # streaks in place, blocking the legitimate read-back that follows.
        return "bytes_written" in data or data.get("verified") is True
    if tool_name == "patch":
        return data.get("success") is True
    return False
