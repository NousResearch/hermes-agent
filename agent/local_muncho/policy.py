"""Policy classification for Local Muncho guard hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


READ_ONLY_TOOLS = frozenset(
    {
        "browser_snapshot",
        "read_file",
        "read_terminal",
        "session_search",
        "tool_describe",
        "tool_search",
        "web_search",
    }
)


@dataclass(frozen=True)
class ToolPolicy:
    action: str
    requires_lease: bool
    approval_class: str | None = None


def classify_tool_action(tool_name: str, args: Mapping[str, Any] | None) -> ToolPolicy:
    args = args or {}
    if tool_name == "send_message":
        action = str(args.get("action") or "send").lower().strip()
        if action == "list":
            return ToolPolicy("tool:send_message:list", requires_lease=False)
        return ToolPolicy(
            f"tool:send_message:{action}",
            requires_lease=True,
            approval_class="visible_send",
        )
    if tool_name == "delegate_task":
        return ToolPolicy("tool:delegate_task", requires_lease=True, approval_class="worker_spawn")
    if tool_name.startswith("kanban_"):
        return ToolPolicy(f"tool:{tool_name}", requires_lease=True, approval_class="kanban")
    if tool_name in {"approval", "slash_confirm"}:
        return ToolPolicy(f"tool:{tool_name}", requires_lease=True, approval_class="approval")
    if tool_name in READ_ONLY_TOOLS:
        return ToolPolicy(f"tool:{tool_name}", requires_lease=False)
    return ToolPolicy(f"tool:{tool_name}", requires_lease=True)


def visible_send_action(kind: str) -> str:
    return f"visible_send:{str(kind or 'final').lower()}"
