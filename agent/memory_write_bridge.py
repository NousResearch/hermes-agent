"""Helpers for mirroring built-in memory writes to external providers."""

from __future__ import annotations

import json
from typing import Any, Dict, List

_MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}


def _memory_tool_result_succeeded(result: Any) -> bool:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return False

    if isinstance(result, dict):
        if result.get("success") is False:
            return False
        if result.get("staged") is True:
            return False
        if "error" in result and result.get("success") is not True:
            return False

    return True


def collect_memory_write_notifications(
    tool_result: Any,
    tool_args: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Return provider notifications for a successful built-in memory write."""
    if not _memory_tool_result_succeeded(tool_result):
        return []

    target = str(tool_args.get("target") or "memory")
    operations = tool_args.get("operations")
    if isinstance(operations, list) and operations:
        raw_operations = operations
    else:
        raw_operations = [{
            "action": tool_args.get("action"),
            "content": tool_args.get("content"),
            "old_text": tool_args.get("old_text"),
        }]

    notifications: List[Dict[str, str]] = []
    for op in raw_operations:
        if not isinstance(op, dict):
            continue
        action = str(op.get("action") or "")
        if action not in _MIRRORED_MEMORY_ACTIONS:
            continue
        notifications.append({
            "action": action,
            "target": target,
            "content": str(op.get("content") or ""),
            "old_text": str(op.get("old_text") or ""),
        })
    return notifications
