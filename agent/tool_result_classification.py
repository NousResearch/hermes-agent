"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})
WORKSPACE_MUTATION_KEY = "workspace_mutation"
WORKSPACE_MUTATION_OPERATIONS = frozenset({"write", "patch", "delete", "rename", "install", "unknown"})


def _as_dict(result: Any) -> dict[str, Any] | None:
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result.strip())
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def extract_workspace_mutation(result: Any, *, fallback_workspace: str | None = None) -> dict[str, Any] | None:
    """Validate and normalize optional common mutation metadata from a tool result."""
    data = _as_dict(result)
    raw = data.get(WORKSPACE_MUTATION_KEY) if data else None
    if not isinstance(raw, dict):
        return None

    workspace = raw.get("workspace") or fallback_workspace
    if not isinstance(workspace, str) or not workspace.strip():
        return None
    root = Path(workspace).expanduser()
    if not root.is_absolute():
        return None
    root = root.resolve()

    operation = str(raw.get("operation") or "unknown").strip().lower()
    if operation not in WORKSPACE_MUTATION_OPERATIONS:
        operation = "unknown"

    raw_paths = raw.get("paths")
    if raw_paths is None:
        raw_paths = []
    if not isinstance(raw_paths, list) or any(not isinstance(path, str) for path in raw_paths):
        return None
    paths = []
    for value in raw_paths:
        if not value.strip():
            continue
        path = Path(value).expanduser()
        paths.append(str((path if path.is_absolute() else root / path).resolve()))
    return {"workspace": str(root), "operation": operation, "paths": sorted(set(paths))}


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


# Set by a tool that REFUSED a call the harness judged redundant (repeated identical
# read/search). The body still carries ``"error"`` so the model reads it as a stop
# signal, but nothing failed: failure classifiers must not count it, or the cheap
# refusal feeds the streak that fires ``repeated_exact_failure_block``.
GUARDRAIL_REFUSAL_KEY = "guardrail_refusal"


def is_guardrail_refusal(result: Any) -> bool:
    """Return True when ``result`` (JSON string or parsed dict) is a harness refusal."""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result.strip())
        except Exception:
            return False
    return isinstance(data, dict) and data.get(GUARDRAIL_REFUSAL_KEY) is True


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
