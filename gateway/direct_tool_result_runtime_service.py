"""Shared result validation helpers for direct tool-backed gateway shortcuts."""

from __future__ import annotations

from typing import Any

from tool_result_validation import tool_result_failure_text


def shortcut_tool_failure_text(result: Any, *, failure_prefix: str) -> str | None:
    """Return a user-facing failure string unless the tool result is explicit success."""
    return tool_result_failure_text(result, failure_prefix=failure_prefix)
