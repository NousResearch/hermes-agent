"""Official SDK PreToolUse guard for workspace-confined Claude tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


def _output(decision: str, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def _workspace_path(workspace: Path, raw: Any) -> Path | None:
    value = str(raw or ".")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=False)
        if resolved == workspace or resolved.is_relative_to(workspace):
            return resolved
    except (OSError, RuntimeError):
        return None
    return None


def _safe_glob_pattern(raw: Any) -> bool:
    pattern = str(raw or "")
    if not pattern or Path(pattern).is_absolute():
        return False
    return ".." not in Path(pattern).parts


def create_workspace_pre_tool_hook(
    workspace: str | Path,
    *,
    allowed_mcp_tools: Iterable[str] | None = None,
) -> Callable[[Mapping[str, Any], str | None, Any], Awaitable[dict[str, Any]]]:
    """Return a hook that denies every capability outside the worker root."""

    root = Path(workspace).expanduser().resolve()
    exact_mcp_tools = (
        {str(name) for name in allowed_mcp_tools}
        if allowed_mcp_tools is not None
        else None
    )

    async def _guard(
        hook_input: Mapping[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        del tool_use_id, context
        tool_name = str(hook_input.get("tool_name") or "")
        tool_input = hook_input.get("tool_input") or {}
        if not isinstance(tool_input, Mapping):
            return _output("deny", "Tool input is not an object")
        default_worker_mcp = tool_name.startswith("mcp__hermes__kanban_") or tool_name in {
            "mcp__hermes__terminal",
            "mcp__hermes__process",
            "mcp__hermes__read_file",
            "mcp__hermes__write_file",
        }
        if (
            tool_name in exact_mcp_tools
            if exact_mcp_tools is not None
            else default_worker_mcp
        ):
            return _output("allow", "Hermes in-process Kanban tool")
        if tool_name == "Bash":
            return _output("deny", "Bash is disabled for subscription workers")
        path_fields = {
            "Read": "file_path",
            "Edit": "file_path",
            "Write": "file_path",
            "Glob": "path",
            "Grep": "path",
        }
        field = path_fields.get(tool_name)
        if field is None:
            return _output("deny", f"Tool {tool_name or '<unknown>'} is not allowed")
        if _workspace_path(root, tool_input.get(field, ".")) is None:
            return _output("deny", f"{tool_name} path is outside the worker workspace")
        if tool_name == "Glob" and not _safe_glob_pattern(tool_input.get("pattern")):
            return _output("deny", "Glob pattern may not escape the worker workspace")
        return _output("allow", "Path is confined to the worker workspace")

    return _guard


__all__ = ["create_workspace_pre_tool_hook"]
