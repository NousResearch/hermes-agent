"""Agent-level ("inline") tool executors shared by the sequential and concurrent tool paths.

These tools need live ``AIAgent`` state (stores, callbacks, session DB) and therefore
bypass the tool registry. Each executor is ``fn(agent, args, ctx) -> result``; the
table replaces two hand-maintained if/elif chains (``invoke_tool`` and
``execute_tool_calls_sequential``) that had drifted apart. Tool modules are imported
lazily at call time so ``patch("tools.x.y")`` in tests keeps working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple


def tool_hook_ids(agent, effective_task_id: str, tool_call_id: Optional[str]) -> Dict[str, str]:
    """Identity kwargs every tool hook/middleware call carries (all coerced to ``""``)."""
    return {
        "task_id": effective_task_id or "",
        "session_id": getattr(agent, "session_id", "") or "",
        "tool_call_id": tool_call_id or "",
        "turn_id": getattr(agent, "_current_turn_id", "") or "",
        "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
    }


def emit_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    result: Any,
    effective_task_id: str,
    tool_call_id: Optional[str],
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[list] = None,
) -> None:
    """Emit the one terminal ``post_tool_call`` hook for a tool_call_id (best-effort)."""
    try:
        from model_tools import _emit_post_tool_call_hook
        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception:
        pass


@dataclass
class InlineToolContext:
    """Per-call state an inline executor may need beyond its arguments."""

    effective_task_id: str
    tool_call_id: Optional[str] = None
    messages: Optional[list] = None


InlineToolExecutor = Callable[[Any, dict, InlineToolContext], Any]

# ``(kwarg, args_key)`` → ``args.get(key)``; ``(kwarg, args_key, default)`` → ``args.get(key, default)``.
_ArgSpec = Tuple[Any, ...]


def _call_tool(module: str, func: str, args: dict, arg_specs: Tuple[_ArgSpec, ...], **fixed: Any) -> Any:
    """Import ``module.func`` lazily and call it with args mapped per ``arg_specs`` plus ``fixed``."""
    fn = getattr(import_module(module), func)
    return fn(**{spec[0]: args.get(*spec[1:]) for spec in arg_specs}, **fixed)


def _tool(
    module: str, func: str, *arg_specs: _ArgSpec, **fixed: Callable[[Any, InlineToolContext], Any],
) -> InlineToolExecutor:
    """Executor calling ``module.func`` with mapped args plus ``fixed`` kwargs computed from ``(agent, ctx)``."""
    def _exec(agent, args: dict, ctx: InlineToolContext) -> Any:
        return _call_tool(module, func, args, arg_specs, **{k: f(agent, ctx) for k, f in fixed.items()})
    return _exec


def _callback_tool(module: str, func: str, callback_attr: str, *arg_specs: _ArgSpec) -> InlineToolExecutor:
    """Executor for a GUI-callback tool: mapped args plus ``callback=getattr(agent, callback_attr, None)``."""
    return _tool(module, func, *arg_specs, callback=lambda agent, ctx: getattr(agent, callback_attr, None))


def _session_search(agent, args: dict, ctx: InlineToolContext) -> Any:
    session_db = agent._get_session_db_for_recall()
    if not session_db:
        from hermes_state import format_session_db_unavailable

        return json.dumps({"success": False, "error": format_session_db_unavailable()})
    return _call_tool(
        "tools.session_search_tool", "session_search", args,
        (
            ("query", "query", ""), ("role_filter", "role_filter"), ("limit", "limit", 3),
            ("session_id", "session_id"), ("around_message_id", "around_message_id"),
            ("window", "window", 5), ("sort", "sort"), ("detail", "detail", "adaptive"),
        ),
        db=session_db, current_session_id=agent.session_id,
    )


def _memory(agent, args: dict, ctx: InlineToolContext) -> Any:
    result = _call_tool(
        "tools.memory_tool", "memory_tool", args,
        (
            ("action", "action"), ("target", "target", "memory"), ("content", "content"),
            ("old_text", "old_text"), ("operations", "operations"),
        ),
        store=agent._memory_store,
    )
    # Mirror built-in memory writes to external providers; gating lives in
    # MemoryManager.notify_memory_tool_write.
    if agent._memory_manager:
        agent._memory_manager.notify_memory_tool_write(
            result,
            args,
            build_metadata=lambda: agent._build_memory_write_metadata(
                task_id=ctx.effective_task_id,
                tool_call_id=ctx.tool_call_id,
            ),
        )
    return result


_read_preview = _callback_tool(
    "tools.read_preview_tool", "read_preview_tool", "read_preview_callback",
    ("start", "start"), ("count", "count"),
)


def _desktop_preview(agent, args: dict, ctx: InlineToolContext) -> Any:
    # action=read needs the GUI callback (agent-level); open/close go through the
    # registry handler like any other tool.
    if (args.get("action") or "").strip() == "read":
        return _read_preview(agent, args, ctx)
    from tools.preview_tool import _handle_preview

    return _handle_preview(args)


# Order is the historical if/elif order of ``execute_tool_calls_sequential``.
INLINE_TOOL_EXECUTORS: Dict[str, InlineToolExecutor] = {
    "todo_list": _tool(
        "tools.todo_tool", "todo_tool", ("todos", "todos"), ("merge", "merge", False),
        store=lambda agent, ctx: agent._todo_store,
    ),
    # Bot Mode teammate DM is injected, not registered: only a canonical Bot
    # Chat session carries the schema, and the tool re-gates on the title.
    "message_agent": _tool(
        "tools.bot_mode_dm", "message_agent_tool", ("target", "target", ""), ("message", "message", ""),
        task_id=lambda agent, ctx: ctx.effective_task_id, agent=lambda agent, ctx: agent,
    ),
    "session_search": _session_search,
    "memory": _memory,
    "clarify": _tool(
        "tools.clarify_tool", "clarify_tool",
        ("question", "question", ""), ("choices", "choices"), ("multi_select", "multi_select", False),
        ("questions", "questions"),
        callback=lambda agent, ctx: agent.clarify_callback,
    ),
    "read_terminal": _callback_tool(
        "tools.read_terminal_tool", "read_terminal_tool", "read_terminal_callback",
        ("start_line", "start_line"), ("count", "count"),
    ),
    "desktop_preview": _desktop_preview,
    "drive_preview": _callback_tool(
        "tools.drive_preview_tool", "drive_preview_tool", "drive_preview_callback",
        ("action", "action", ""), ("ref", "ref"), ("selector", "selector"), ("text", "text"),
        ("key", "key"), ("submit", "submit"), ("amount", "amount"), ("to", "to"), ("limit", "max"),
    ),
    "annotate_preview": _callback_tool(
        "tools.annotate_preview_tool", "annotate_preview_tool", "drive_preview_callback",
        ("action", "action", "add"), ("ref", "ref"), ("selector", "selector"), ("label", "label"),
    ),
    "read_window_below": _callback_tool(
        "tools.read_window_tool", "read_window_below_tool", "read_window_below_callback",
    ),
    "gui_tour": _callback_tool(
        "tools.tour_tool", "tour_tool", "tour_callback",
        ("action", "action", ""), ("surface", "surface"), ("selector", "selector"), ("title", "title"),
        ("text", "text"), ("side", "side"), ("steps", "steps"), ("step_index", "step_index"),
    ),
    "setup_mcp": _callback_tool(
        "tools.setup_mcp_tool", "setup_mcp_tool", "setup_mcp_callback",
        ("server", "server", ""), ("action", "action", "install"), ("reason", "reason", ""),
    ),
    "workflow": _callback_tool(
        "tools.workflow_tools", "workflow_tool", "workflow_callback",
        ("action", "action", ""), ("ops", "ops"), ("workflow", "workflow"), ("scenario", "scenario"),
        ("payload", "payload"),
    ),
    "delegate_task": lambda agent, args, ctx: agent._dispatch_delegate_task(args),
}

# ``invoke_tool`` (concurrent path) consults the memory manager right after these three
# names and before the remaining inline tools; ``message_agent`` falls through to the
# registry there (Bot Mode DM is only injected into the sequential path's schema).
INVOKE_TOOL_PRE_MEMORY_MANAGER_NAMES = frozenset({"todo_list", "session_search", "memory"})


def resolve_invoke_tool_executor(agent, function_name: str) -> Optional[InlineToolExecutor]:
    """Inline executor for ``invoke_tool`` (concurrent path), or None for registry dispatch.

    Precedence: todo_list/session_search/memory, then memory-manager tools, then the
    remaining inline tools (``message_agent`` excluded).
    """
    if function_name in INVOKE_TOOL_PRE_MEMORY_MANAGER_NAMES:
        return INLINE_TOOL_EXECUTORS[function_name]
    memory_manager = agent._memory_manager
    if memory_manager and memory_manager.has_tool(function_name):
        return lambda agent, args, ctx: agent._memory_manager.handle_tool_call(function_name, args)
    if function_name == "message_agent":
        return None
    return INLINE_TOOL_EXECUTORS.get(function_name)
