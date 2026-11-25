"""Invoke agent and registry tools with their approval and middleware hooks."""

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pre_tool_block_message(agent, function_name, function_args, effective_task_id, tool_call_id, middleware_trace):
    """Plugin pre-tool-call verdict; approve is deferred until execution middleware settles args."""
    try:
        from hermes_cli.plugins import _prepare_pre_tool_call_hooks_for_dispatch
        block_message, modified_args, pending_approval = _prepare_pre_tool_call_hooks_for_dispatch(
            function_name, function_args, task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "", tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            middleware_trace=list(middleware_trace),
        )
        return block_message, (modified_args if modified_args is not None else function_args), pending_approval
    except Exception:
        return None, function_args, None


def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list | None = None,
                 pre_tool_block_checked: bool = False,
                 skip_tool_request_middleware: bool = False,
                 tool_request_middleware_trace: Optional[list[dict[str, Any]]] = None,
                 skip_tool_execution_middleware: bool = False) -> str:
    """Invoke a single tool (agent-level or registry-dispatched) and return the result string;
    no display logic. Used by the concurrent path; the sequential path keeps its own inline
    invocation for display."""
    from agent.inline_tool_executors import (
        InlineToolContext, apply_transform_tool_result, emit_terminal_post_tool_call,
        resolve_invoke_tool_executor, tool_hook_ids
    )
    if not isinstance(function_args, dict):
        function_args = {}
    hook_ids = tool_hook_ids(agent, effective_task_id, tool_call_id)
    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    try:
        from hermes_cli.middleware import apply_tool_request_middleware
        if not skip_tool_request_middleware:
            _tool_request_mw = apply_tool_request_middleware(function_name, function_args, **hook_ids)
            function_args = _tool_request_mw.payload
            _tool_middleware_trace = _tool_request_mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)
    block_message: Optional[str] = None
    pending_approval = None
    if not pre_tool_block_checked:
        block_message, function_args, pending_approval = _pre_tool_block_message(
            agent, function_name, function_args, effective_task_id, tool_call_id, _tool_middleware_trace
        )
    if block_message is not None:
        result = json.dumps({"error": block_message}, ensure_ascii=False)
        emit_terminal_post_tool_call(
            agent, function_name=function_name, function_args=function_args, result=result,
            effective_task_id=effective_task_id, tool_call_id=tool_call_id, status="blocked",
            error_type="plugin_block", error_message=block_message,
            middleware_trace=_tool_middleware_trace,
        )
        return result
    tool_start_time = time.monotonic()

    def _deferred_approval_block(call_args: dict) -> Optional[str]:
        if pending_approval is None:
            return None
        from hermes_cli.plugins import _resolve_pending_pre_tool_approval
        return _resolve_pending_pre_tool_approval(
            pending_approval,
            function_name,
            call_args,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            middleware_trace=list(_tool_middleware_trace),
        )

    def _blocked_by_deferred_approval(call_args: dict, block: str) -> str:
        result = json.dumps({"error": block}, ensure_ascii=False)
        emit_terminal_post_tool_call(
            agent, function_name=function_name, function_args=call_args, result=result,
            effective_task_id=effective_task_id, tool_call_id=tool_call_id, status="blocked",
            error_type="plugin_block", error_message=block,
            middleware_trace=_tool_middleware_trace,
        )
        return result

    def _approved_args(call_args: dict):
        if pending_approval is None:
            return None, call_args
        try:
            from tools.arg_coercion import ApprovedToolArgs, coerce_tool_args
            snapshot = json.dumps(call_args, allow_nan=False)
            call_args = coerce_tool_args(function_name, json.loads(snapshot))
            snapshot = json.dumps(call_args, allow_nan=False)
            block = _deferred_approval_block(call_args)
            return block, ApprovedToolArgs(function_name, snapshot) if block is None else call_args
        except Exception:  # health: allow BLE001 -- fail closed without logging sensitive payloads
            return f"BLOCKED: plugin approval gate failed for {function_name}", call_args

    inline_executor = resolve_invoke_tool_executor(agent, function_name)
    if inline_executor is not None:
        inline_ctx = InlineToolContext(
            effective_task_id=effective_task_id, tool_call_id=tool_call_id, messages=messages
        )

        def _execute(next_args: dict) -> Any:
            call_args = next_args if isinstance(next_args, dict) else function_args
            block, call_args = _approved_args(call_args)
            if block is not None:
                return _blocked_by_deferred_approval(call_args, block)
            next_args = call_args
            result = inline_executor(agent, next_args, inline_ctx)
            duration_ms = int((time.monotonic() - tool_start_time) * 1000)
            emit_terminal_post_tool_call(
                agent, function_name=function_name, function_args=call_args,
                result=result, effective_task_id=effective_task_id, tool_call_id=tool_call_id,
                duration_ms=duration_ms, middleware_trace=_tool_middleware_trace,
            )
            return apply_transform_tool_result(
                agent, function_name=function_name, function_args=call_args, result=result,
                effective_task_id=effective_task_id, tool_call_id=tool_call_id, duration_ms=duration_ms,
            )
    else:
        def _execute(next_args: dict) -> Any:
            call_args = next_args if isinstance(next_args, dict) else function_args
            block, call_args = _approved_args(call_args)
            if block is not None:
                return _blocked_by_deferred_approval(call_args, block)
            next_args = call_args
            dispatch_kwargs = dict(
                tool_call_id=tool_call_id, session_id=agent.session_id or "", conversation_id=hook_ids["conversation_id"],
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                tool_request_middleware_trace=list(_tool_middleware_trace),
            )
            # This helper or its managed caller owns middleware; approval must stay final.
            dispatch_kwargs["skip_tool_execution_middleware"] = True
            import model_tools
            return model_tools.handle_function_call(function_name, next_args, effective_task_id, **dispatch_kwargs)
    if skip_tool_execution_middleware:
        return _execute(function_args)
    from hermes_cli.middleware import run_tool_execution_middleware
    return run_tool_execution_middleware(
        function_name, function_args,
        lambda next_args: _execute(next_args if isinstance(next_args, dict) else function_args),
        original_args=function_args, **hook_ids,
    )


