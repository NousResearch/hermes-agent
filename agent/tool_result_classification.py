"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch", "execute_code"})


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
    if tool_name == "execute_code":
        # execute_code returns {"status": "success", "output": "..."} on success.
        # The code may or may not have written files — landed means the script
        # itself ran without error, not that a specific file was written.
        # Whether a file was actually written is determined by path extraction
        # in _extract_file_mutation_targets; if no paths are extracted, the
        # mutation set stays empty (correct behavior).
        return data.get("status") == "success"
    return False
