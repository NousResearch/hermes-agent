"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import re
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


def file_mutation_targets(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Return file paths targeted by a dedicated file-mutation tool call."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES:
        return []
    if not isinstance(args, dict):
        return []
    if tool_name == "write_file":
        path = args.get("path")
        return [str(path)] if path else []

    # tool_name == "patch"
    mode = args.get("mode") or "replace"
    if mode == "replace":
        path = args.get("path")
        return [str(path)] if path else []
    if mode == "patch":
        body = args.get("patch") or ""
        if not isinstance(body, str) or not body:
            return []
        paths: list[str] = []
        for match in re.finditer(
            r'^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$',
            body,
            re.MULTILINE,
        ):
            path = match.group(1).strip()
            if path:
                paths.append(path)
        return paths
    return []


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
