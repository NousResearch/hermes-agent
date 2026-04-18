"""Shared result validation helpers for direct tool-backed gateway shortcuts."""

from __future__ import annotations

from typing import Any


def shortcut_tool_failure_text(result: Any, *, failure_prefix: str) -> str | None:
    """Return a user-facing failure string unless the tool result is explicit success."""

    if not isinstance(result, dict):
        return f"{failure_prefix}：工具未返回有效结果"

    error = str(result.get("error") or "").strip()
    if error:
        return error

    if result.get("success") is True:
        return None

    detail = str(
        result.get("message")
        or result.get("detail")
        or result.get("status")
        or ""
    ).strip()
    if detail:
        return f"{failure_prefix}：{detail}"
    return f"{failure_prefix}：工具未返回成功结果"
