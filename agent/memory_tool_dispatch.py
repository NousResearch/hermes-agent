"""Shared built-in memory tool dispatch.

Both direct agent invocation and sequential tool execution call this operation.
Path-specific middleware, result formatting, display, and persistence stay with
their respective callers.
"""

from __future__ import annotations

from typing import Any, Optional


def dispatch_memory_tool(
    agent: Any,
    function_args: dict[str, Any],
    *,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Any:
    """Execute one built-in memory call and notify external providers.

    ``MemoryManager.notify_memory_tool_write`` remains the sole authority for
    committed-write gating, batch expansion, and add/replace/remove mirroring.
    """
    from tools.memory_tool import memory_tool

    result = memory_tool(
        action=function_args.get("action"),
        target=function_args.get("target", "memory"),
        content=function_args.get("content"),
        old_text=function_args.get("old_text"),
        operations=function_args.get("operations"),
        store=agent._memory_store,
    )
    if agent._memory_manager:
        agent._memory_manager.notify_memory_tool_write(
            result,
            function_args,
            build_metadata=lambda: agent._build_memory_write_metadata(
                task_id=task_id,
                tool_call_id=tool_call_id,
            ),
        )
    return result
