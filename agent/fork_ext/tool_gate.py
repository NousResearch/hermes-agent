"""Fork-owned tool-call gate helpers extracted from high-churn dispatchers."""

from __future__ import annotations

import json
from typing import Any

from agent.budget_grace_gate import grace_block_message, is_readonly_grace_tool


def tool_search_scoped_names(agent) -> frozenset:
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:  # pragma: no cover - import failure is a fail-closed safety net
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    try:
        scoped_defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        names = _ts.scoped_deferrable_names(scoped_defs)
    except Exception:  # pragma: no cover - registry rebuild failure fails closed
        names = frozenset()
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:  # pragma: no cover - cache write failure must not block dispatch
        pass
    return names


def tool_scope_block_message(tool_name: str) -> str:
    return (
        f"'{tool_name}' is not available in this session. "
        "Use tool_search to find tools you can call."
    )


def tool_scope_block_result(tool_name: str) -> str:
    return json.dumps({"error": tool_scope_block_message(tool_name)}, ensure_ascii=False)


def resolve_tool_search_unwrap(agent, function_name: str, function_args: dict[str, Any]):
    try:
        from tools import tool_search as _ts

        if function_name == _ts.TOOL_CALL_NAME:
            underlying, underlying_args, err = _ts.resolve_underlying_call(function_args)
            if not err and underlying:
                if underlying in tool_search_scoped_names(agent):
                    return underlying, underlying_args, None, None
                block_message = tool_scope_block_message(underlying)
                return function_name, function_args, block_message, tool_scope_block_result(underlying)
    except Exception:  # pragma: no cover - unwrap failures preserve the original call
        pass
    return function_name, function_args, None, None


def pre_tool_block_from_builtin_gate(agent, function_name: str, tool_scope_block: str | None) -> dict[str, str] | None:
    if getattr(agent, "_in_budget_grace", False) and not is_readonly_grace_tool(function_name):
        return {
            "message": grace_block_message(function_name),
            "error_type": "budget_grace_block",
        }
    if tool_scope_block is not None:
        return {
            "message": tool_scope_block,
            "error_type": "tool_scope_block",
        }
    return None


_DEFAULT_DELEGATE_BLOCKED_TOOLS = frozenset(
    {"delegate_task", "clarify", "memory", "send_message", "execute_code", "cronjob"}
)
_TOOLSET_STRIP_EXEMPT = frozenset({"code_execution"})


def strip_blocked_delegate_toolsets(
    toolsets: list[str],
    *,
    toolset_definitions: dict[str, dict[str, Any]] | None = None,
    delegate_blocked_tools=frozenset(_DEFAULT_DELEGATE_BLOCKED_TOOLS),
) -> list[str]:
    if toolset_definitions is None:
        from toolsets import TOOLSETS

        toolset_definitions = TOOLSETS
    blocked_toolset_names = {"delegation"}
    for name, definition in toolset_definitions.items():
        if name in _TOOLSET_STRIP_EXEMPT:
            continue
        tools = definition.get("tools", [])
        if tools and all(tool in delegate_blocked_tools for tool in tools):
            blocked_toolset_names.add(name)
    return [toolset for toolset in toolsets if toolset not in blocked_toolset_names]
