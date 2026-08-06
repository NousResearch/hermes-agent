"""Tool-call execution — sequential and concurrent dispatch.

Both AIAgent methods (``_execute_tool_calls_sequential`` and
``_execute_tool_calls_concurrent``) live here as module-level
functions that take the parent ``AIAgent`` as their first argument.

``run_agent`` keeps thin wrappers so existing call sites work; tests
that patch ``run_agent._set_interrupt`` are honored because the
extracted functions reach back through the ``run_agent`` module via
``_ra()`` for that symbol.

The per-tool lifecycle both dispatchers share lives in
:func:`execute_one_tool`.  Any runtime that owns its own agent loop (the
``claude_agent_sdk`` bridge in ``agent/transports/claude_tool_bridge.py``)
must route its tool calls through that function rather than reaching
``model_tools.handle_function_call`` directly — approvals, guardrails,
plugin hooks, checkpoints, and environment routing all live in the
wrappers it applies.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    build_tool_label as _build_tool_label,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.tool_dispatch_helpers import (
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    _plan_tool_batch_segments,
    make_tool_result_message,
)
from tools.terminal_tool import (
    get_active_env,
)
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)


def _ensure_file_checkpoint(
    agent,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
) -> None:
    """Checkpoint the same workspace path that the file tool will mutate."""
    file_path = function_args.get("path", "")
    if not file_path:
        return

    # File tools resolve relative paths against the task's live/session cwd,
    # which can differ from the Hermes process cwd (notably in Docker).  Resolve
    # through that same path pipeline before asking the checkpoint manager to
    # discover the project root.
    from tools.file_tools import _resolve_path_for_task

    resolved_path = _resolve_path_for_task(file_path, effective_task_id or "default")
    work_dir = agent._checkpoint_mgr.get_working_dir_for_path(str(resolved_path))
    agent._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")


def _budget_for_agent(agent) -> BudgetConfig:
    """Resolve a tool-result BudgetConfig scaled to the agent's context window.

    Large-context models keep the historical 100K/200K char defaults; small
    models (e.g. a 65K-token local model switched into mid-session) get a budget
    proportional to their window so a single large tool result can't push the
    request past the model's limit (#23767). Falls back to the default budget
    when the context length isn't resolvable.
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        return budget_for_context_window(int(ctx)) if ctx else DEFAULT_BUDGET
    except Exception:
        return DEFAULT_BUDGET

# Maximum number of concurrent worker threads for parallel tool execution.
# Mirrors the constant in ``run_agent`` for tests/imports that look here.
_MAX_TOOL_WORKERS = 8
# Keep this above the stock auxiliary.web_extract timeout (360s) so the batch
# guard does not preempt a slow-but-valid summarization attempt.
_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """Parse model-emitted arguments without repairing or coercing them."""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {
            "error": "Invalid tool arguments",
            "message": (
                "Tool arguments must be a valid JSON object; tool was not executed."
            ),
        },
        ensure_ascii=False,
    )


def _resolve_concurrent_tool_timeout() -> float | None:
    raw = os.getenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "").strip()
    if not raw:
        return _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "invalid HERMES_CONCURRENT_TOOL_TIMEOUT_S=%r; using %.0fs",
            raw,
            _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
        )
        return _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S
    if value <= 0:
        return None
    return value


def _flush_session_db_after_tool_progress(
    agent,
    messages: list,
    *,
    stage: str,
) -> bool:
    """Flush tool-call progress before projecting it to any UI surface.

    Tool execution can perform side effects that terminate or restart the
    current Hermes process before the normal turn-end persistence path runs.
    Flush the already-appended assistant/tool messages immediately so the
    transcript survives destructive-but-valid tool calls.
    """
    try:
        persisted = agent._flush_messages_to_session_db(messages) is not False
        if not persisted:
            agent._incremental_persistence_failed = True
        return persisted
    except Exception as exc:
        agent._incremental_persistence_failed = True
        logger.warning("Incremental tool-call persistence failed after %s: %s", stage, exc)
        return False


def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    return run_agent


def _is_interpreter_shutdown_submit_error(exc: RuntimeError) -> bool:
    return "cannot schedule new futures after interpreter shutdown" in str(exc)


def _emit_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    result: Any,
    effective_task_id: str,
    tool_call_id: str,
    duration_ms: int = 0,
    status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    middleware_trace: Optional[list[dict[str, Any]]] = None,
) -> None:
    try:
        from model_tools import _emit_post_tool_call_hook
        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception:
        pass


def _cancelled_tool_result(reason: str = "user interrupt") -> str:
    return json.dumps(
        {
            "error": f"Tool execution cancelled by {reason}",
            "status": "cancelled",
        },
        ensure_ascii=False,
    )


def _emit_cancelled_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    start_time: float,
    reason: str = "user interrupt",
    error_type: str = "keyboard_interrupt",
    middleware_trace: Optional[list[dict[str, Any]]] = None,
) -> str:
    result = _cancelled_tool_result(reason)
    _emit_terminal_post_tool_call(
        agent,
        function_name=function_name,
        function_args=function_args,
        result=result,
        effective_task_id=effective_task_id,
        tool_call_id=tool_call_id,
        duration_ms=int((time.time() - start_time) * 1000),
        status="cancelled",
        error_type=error_type,
        error_message=f"Tool execution cancelled by {reason}",
        middleware_trace=list(middleware_trace or []),
    )
    return result


def _tool_search_scoped_names(agent) -> frozenset:
    """Return the deferrable tool names the session may invoke via tool_call.

    The Tool Search unwrap dispatches the underlying tool directly, bypassing
    the bridge branch (and its scope check) in
    ``model_tools.handle_function_call``. To keep a restricted-toolset session
    (subagent, kanban worker, curated gateway session) from reaching tools it
    was never granted, the unwrap validates the underlying name against this
    set: the deferrable subset of the session's own enabled/disabled toolset
    scope.

    Result is cached on the agent and refreshed when the tool registry's
    generation changes (e.g. an MCP server reconnects), so the common case is
    a dict lookup, not a full tool-defs rebuild on every tool call.
    """
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
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
    except Exception:
        names = frozenset()
    try:
        agent._tool_search_scope_cache = (cache_key, names)
    except Exception:
        pass
    return names


@dataclass
class _ManagedToolResult:
    result: Any
    args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    blocked: bool


class _ConcurrentToolAuthorizationGate:
    """Serialize policy prompts and exclude their queue from batch deadlines."""

    def __init__(self) -> None:
        self._serialization_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending = 0
        self._window_started: float | None = None
        self._excluded_seconds = 0.0

    def run(self, callback):
        now = time.monotonic()
        with self._state_lock:
            if self._pending == 0:
                self._window_started = now
            self._pending += 1
        try:
            with self._serialization_lock:
                return callback()
        finally:
            now = time.monotonic()
            with self._state_lock:
                self._pending -= 1
                if self._pending == 0:
                    if self._window_started is not None:
                        self._excluded_seconds += max(
                            0.0, now - self._window_started
                        )
                    self._window_started = None

    def excluded_seconds(self) -> float:
        """Return completed plus currently active authorization wait time."""
        now = time.monotonic()
        with self._state_lock:
            excluded = self._excluded_seconds
            if self._window_started is not None:
                excluded += max(0.0, now - self._window_started)
            return excluded


def _managed_values(
    outcome: _ManagedToolResult,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], bool]:
    return (
        outcome.result,
        outcome.args,
        outcome.middleware_trace,
        outcome.blocked,
    )


def _run_agent_tool_execution_middleware(
    agent,
    *,
    function_name: str,
    function_args: dict,
    effective_task_id: str,
    tool_call_id: str,
    execute,
    scope_block: str | None = None,
    display_index: int | None = None,
    middleware_trace: list[dict[str, Any]] | None = None,
    begin_execution=None,
    authorization_gate: _ConcurrentToolAuthorizationGate | None = None,
) -> _ManagedToolResult:
    """Run Relay rewrites before Hermes policy and dispatch exactly once."""
    from agent import relay_tools
    from hermes_cli.middleware import (
        apply_tool_request_middleware,
        run_tool_execution_middleware,
    )

    trace = middleware_trace if middleware_trace is not None else []
    state = {
        "args": function_args,
        "middleware_trace": trace,
        "blocked": False,
        "dispatched": False,
    }
    dispatch_lock = threading.Lock()

    def _authorized_dispatch(final_args: dict[str, Any]) -> Any:
        with dispatch_lock:
            if state["dispatched"]:
                raise RuntimeError(
                    "Hermes tool execution callback invoked more than once"
                )
            state["dispatched"] = True
            state["blocked"] = False
            state["args"] = final_args

        def _begin() -> None:
            _begin_tool_execution(
                agent,
                function_name=function_name,
                function_args=final_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                display_index=display_index,
            )

        def _advance_start_order(callback=None) -> None:
            if begin_execution is None:
                if callback is not None:
                    callback()
                return
            begin_execution(callback)

        block_message = scope_block
        block_error_type = "tool_scope_block"
        if block_message is None:
            block_error_type = "plugin_block"

            def _resolve_pre_tool_block():
                try:
                    from hermes_cli.plugins import resolve_pre_tool_block

                    return resolve_pre_tool_block(
                        function_name,
                        final_args,
                        task_id=effective_task_id or "",
                        session_id=getattr(agent, "session_id", "") or "",
                        tool_call_id=tool_call_id or "",
                        turn_id=getattr(agent, "_current_turn_id", "") or "",
                        api_request_id=getattr(agent, "_current_api_request_id", "")
                        or "",
                        middleware_trace=list(state["middleware_trace"]),
                    )
                except Exception:
                    return None

            block_message = (
                _resolve_pre_tool_block()
                if authorization_gate is None
                else authorization_gate.run(_resolve_pre_tool_block)
            )

        guardrail_decision = None
        if block_message is None:
            guardrail_decision = agent._tool_guardrails.before_call(
                function_name, final_args
            )
            if guardrail_decision.allows_execution:
                guardrail_decision = None

        if block_message is not None or guardrail_decision is not None:
            _advance_start_order()
            state["blocked"] = True
            if block_message is not None:
                result = json.dumps({"error": block_message}, ensure_ascii=False)
                error_type = block_error_type
                error_message = block_message
            else:
                result = agent._guardrail_block_result(guardrail_decision)
                error_type = "guardrail_block"
                error_message = (
                    getattr(guardrail_decision, "message", None)
                    or "Tool blocked by guardrail policy"
                )
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=final_args,
                result=result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="blocked",
                error_type=error_type,
                error_message=error_message,
                middleware_trace=list(state["middleware_trace"]),
            )
            return result

        if function_name == "memory":
            agent._turns_since_memory = 0
        elif function_name == "skill_manage":
            agent._iters_since_skill = 0

        _advance_start_order(_begin)
        return execute(final_args)

    def _hermes_pipeline(relay_args: dict[str, Any]) -> Any:
        request_result = apply_tool_request_middleware(
            function_name,
            relay_args,
            skip_relay=True,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
        )
        request_args = (
            request_result.payload
            if isinstance(request_result.payload, dict)
            else relay_args
        )
        trace.clear()
        trace.extend(request_result.trace)
        return run_tool_execution_middleware(
            function_name,
            request_args,
            lambda next_args: _authorized_dispatch(
                next_args if isinstance(next_args, dict) else request_args
            ),
            original_args=function_args,
            task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "",
            tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
        )

    result, _relay_args = relay_tools.execute(
        function_name,
        function_args,
        _hermes_pipeline,
        session_id=str(getattr(agent, "session_id", "") or ""),
        metadata={
            "task_id": effective_task_id or "",
            "turn_id": getattr(agent, "_current_turn_id", "") or "",
            "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
            "tool_call_id": tool_call_id or "",
        },
    )
    return _ManagedToolResult(
        result=result,
        args=state["args"],
        middleware_trace=state["middleware_trace"],
        blocked=bool(state["blocked"]),
    )


def _begin_tool_execution(
    agent,
    *,
    function_name: str,
    function_args: dict[str, Any],
    effective_task_id: str,
    tool_call_id: str,
    display_index: int | None,
) -> None:
    """Run user-visible and checkpoint preflight on final tool arguments."""
    if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
        display_args = (
            _redact_tool_args_for_display(function_name, function_args) or function_args
        )
        args_str = json.dumps(display_args, ensure_ascii=False)
        prefix = f"Tool {display_index}" if display_index is not None else "Tool"
        if agent.verbose_logging:
            print(f"  📞 {prefix}: {function_name}({list(display_args.keys())})")
            print(
                agent._wrap_verbose(
                    "Args: ", json.dumps(display_args, indent=2, ensure_ascii=False)
                )
            )
        else:
            args_preview = (
                args_str[: agent.log_prefix_chars] + "..."
                if len(args_str) > agent.log_prefix_chars
                else args_str
            )
            print(
                f"  📞 {prefix}: {function_name}({list(function_args.keys())}) - "
                f"{args_preview}"
            )

    agent._current_tool = function_name
    agent._touch_activity(f"executing tool: {function_name}")
    try:
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)
    except Exception:
        pass

    if agent.tool_progress_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            preview = _build_tool_preview(function_name, display_args)
            agent.tool_progress_callback(
                "tool.started", function_name, preview, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool progress callback error: %s", callback_error)

    if agent.tool_start_callback:
        try:
            display_args = (
                _redact_tool_args_for_display(function_name, function_args)
                or function_args
            )
            agent.tool_start_callback(
                tool_call_id, function_name, display_args
            )
        except Exception as callback_error:
            logging.debug("Tool start callback error: %s", callback_error)

    if function_name in {"write_file", "patch"} and agent._checkpoint_mgr.enabled:
        try:
            _ensure_file_checkpoint(
                agent,
                function_name,
                function_args,
                effective_task_id,
            )
        except Exception:
            pass

    if function_name == "terminal" and agent._checkpoint_mgr.enabled:
        try:
            command = function_args.get("command", "")
            if _is_destructive_command(command):
                cwd = function_args.get("workdir") or os.getenv(
                    "TERMINAL_CWD", os.getcwd()
                )
                agent._checkpoint_mgr.ensure_checkpoint(
                    cwd, f"before terminal: {command[:60]}"
                )
        except Exception:
            pass


@dataclass
class PreparedToolCall:
    """One model-emitted tool call resolved to the tool Hermes will run.

    Separate from :func:`execute_one_tool` because both dispatchers need the
    resolved name before execution starts — the concurrent path to plan its
    batch and log it, the sequential path to pick the tool's display mode.
    """

    tool_call: Any
    tool_call_id: str
    function_name: str
    function_args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    scope_block: Optional[str] = None
    malformed_result: Optional[str] = None


@dataclass
class ToolExecutionOutcome:
    """One tool call's result after every Hermes wrapper has run."""

    tool_call: Any
    tool_call_id: str
    function_name: str
    function_args: dict[str, Any]
    result: Any
    duration: float
    is_error: bool
    blocked: bool
    cancelled: bool
    malformed: bool
    middleware_trace: list[dict[str, Any]]


def _flatten_probe_schema_error(probe_error: str) -> str:
    """Flatten a deferred-call probe payload into one plain string.

    A ``scope_block`` is re-wrapped as ``{"error": ...}`` downstream, so the
    probe's own JSON payload cannot be passed through verbatim without
    nesting one JSON document inside another.
    """
    try:
        probe = json.loads(probe_error)
        return (
            f"{probe.get('error', '')} Parameters schema: "
            f"{json.dumps(probe.get('parameters', {}), ensure_ascii=False)}. "
            f"{probe.get('hint', '')}"
        ).strip()
    except Exception:
        return probe_error


def prepare_tool_call(agent, tool_call) -> PreparedToolCall:
    """Parse a tool call's arguments and resolve the tool it really targets."""
    function_name = tool_call.function.name
    function_args, malformed_args_result = _parse_tool_arguments(
        tool_call.function.arguments
    )
    prepared = PreparedToolCall(
        tool_call=tool_call,
        tool_call_id=getattr(tool_call, "id", "") or "",
        function_name=function_name,
        function_args=function_args,
        middleware_trace=[],
    )
    if malformed_args_result is not None:
        prepared.malformed_result = malformed_args_result
        return prepared

    # ── Tool Search unwrap ────────────────────────────────────────────
    # When the model invokes the tool_call bridge, peel it open so every
    # downstream check (checkpointing, guardrails, plugin pre-tool-call
    # hooks, the display/activity feed, the post-call callback) sees the
    # underlying tool — not the bridge. This is the OpenClaw lesson: hooks
    # must observe the real tool name.
    #
    # The original tool_call entry on ``tool_call.function`` is left
    # untouched so the conversation transcript and the matching
    # tool_call_id are preserved exactly as the model emitted them.
    #
    # Scope gate: the unwrap dispatches the underlying tool directly
    # (bypassing the bridge branch in handle_function_call and its scope
    # check), so we enforce session toolset scope HERE. A tool the session
    # was not granted is rejected before any checkpoint, hook, or dispatch
    # fires.
    try:
        from tools import tool_search as _ts
        if function_name == _ts.TOOL_CALL_NAME:
            _underlying, _underlying_args, _err = _ts.resolve_underlying_call(function_args)
            if not _err and _underlying:
                if _underlying in _tool_search_scoped_names(agent):
                    # Probe-validate before unwrapping (ironclaw#5149):
                    # missing required args return the parameter schema
                    # instead of dispatching into an opaque failure.
                    _probe_err = _ts.validate_deferred_call_args(_underlying, _underlying_args)
                    if _probe_err is not None:
                        prepared.scope_block = _flatten_probe_schema_error(_probe_err)
                    else:
                        prepared.function_name = _underlying
                        prepared.function_args = _underlying_args
                else:
                    prepared.scope_block = (
                        f"'{_underlying}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    )
    except Exception:
        pass

    return prepared


def _tool_dispatcher(agent, prepared: PreparedToolCall, effective_task_id: str,
                     messages: Optional[list], state: dict):
    """Build the single dispatch callback for one prepared tool call.

    Everything except the context-engine tools routes through
    ``AIAgent._invoke_tool``, which owns the one agent-level/registry
    dispatch ladder and fires the terminal ``post_tool_call`` hook for the
    branches ``handle_function_call`` does not cover.  Context-engine tools
    are not in the registry and need the live message list, so they are
    dispatched here and their hook is emitted by :func:`execute_one_tool`
    (``state["inline_post_hook"]``).
    """
    function_name = prepared.function_name

    context_engine_names = getattr(agent, "_context_engine_tool_names", None)
    if context_engine_names and function_name in context_engine_names:
        state["inline_post_hook"] = True

        def _execute_context_engine(next_args: dict) -> Any:
            try:
                return agent.context_compressor.handle_tool_call(
                    function_name, next_args, messages=messages,
                )
            except Exception as tool_error:
                logger.error(
                    "context_engine.handle_tool_call raised for %s: %s",
                    function_name, tool_error, exc_info=True,
                )
                return json.dumps(
                    {"error": f"Context engine tool '{function_name}' failed: {tool_error}"}
                )

        return _execute_context_engine

    def _execute(next_args: dict) -> Any:
        try:
            return agent._invoke_tool(
                function_name,
                next_args,
                effective_task_id,
                prepared.tool_call_id,
                messages=messages,
                pre_tool_block_checked=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
                tool_request_middleware_trace=list(prepared.middleware_trace),
            )
        except Exception as tool_error:
            memory_manager = getattr(agent, "_memory_manager", None)
            if memory_manager is not None and memory_manager.has_tool(function_name):
                # Memory-provider tools have no registry error wrapper; the
                # executor is the only layer that can stop a provider fault
                # from aborting the whole turn.
                logger.error(
                    "memory_manager.handle_tool_call raised for %s: %s",
                    function_name, tool_error, exc_info=True,
                )
                state["inline_post_hook"] = True
                return json.dumps(
                    {"error": f"Memory tool '{function_name}' failed: {tool_error}"}
                )
            from agent.agent_runtime_helpers import AGENT_RUNTIME_POST_HOOK_TOOL_NAMES
            if function_name in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES:
                raise
            logger.error(
                "handle_function_call raised for %s: %s",
                function_name, tool_error, exc_info=True,
            )
            return f"Error executing tool '{function_name}': {tool_error}"

    return _execute


def execute_one_tool(
    agent,
    tool_call,
    effective_task_id: str,
    *,
    messages: Optional[list] = None,
    prepared: Optional[PreparedToolCall] = None,
    display_index: Optional[int] = None,
    begin_execution=None,
    authorization_gate: Optional[_ConcurrentToolAuthorizationGate] = None,
) -> ToolExecutionOutcome:
    """Run one tool call through the whole Hermes tool lifecycle.

    This is the only sanctioned entry point for executing a model tool call.
    Calling ``model_tools.handle_function_call`` directly skips every wrapper
    applied here: argument parsing, Tool Search unwrap and session-scope
    gating, Relay rewrites, the tool request/execution middleware chain,
    plugin ``pre_tool_call`` blocks, user approvals, before-call guardrails,
    file checkpoints, tool-start display and progress callbacks, environment
    routing through ``effective_task_id`` (local / Docker / SSH / Modal /
    Daytona / Singularity), dispatch, and the terminal ``post_tool_call``
    hook.

    Callers keep scheduling, display, transcript mutation, output budgets,
    and the after-call tail (:func:`finalize_tool_outcome`) — those are
    per-path concerns that cannot be shared without changing behavior.

    ``begin_execution`` and ``authorization_gate`` are the concurrent path's
    start-ordering and approval-serialization hooks; both are optional.
    """
    prep = prepared if prepared is not None else prepare_tool_call(agent, tool_call)

    if prep.malformed_result is not None:
        return ToolExecutionOutcome(
            tool_call=prep.tool_call,
            tool_call_id=prep.tool_call_id,
            function_name=prep.function_name,
            function_args=prep.function_args,
            result=prep.malformed_result,
            duration=0.0,
            is_error=True,
            blocked=True,
            cancelled=False,
            malformed=True,
            middleware_trace=prep.middleware_trace,
        )

    state = {"inline_post_hook": False}
    execute = _tool_dispatcher(agent, prep, effective_task_id, messages, state)
    start = time.time()

    try:
        managed = _run_agent_tool_execution_middleware(
            agent,
            function_name=prep.function_name,
            function_args=prep.function_args,
            effective_task_id=effective_task_id,
            tool_call_id=prep.tool_call_id,
            execute=execute,
            scope_block=prep.scope_block,
            display_index=display_index,
            middleware_trace=prep.middleware_trace,
            begin_execution=begin_execution,
            authorization_gate=authorization_gate,
        )
    except KeyboardInterrupt:
        result = _emit_cancelled_terminal_post_tool_call(
            agent,
            function_name=prep.function_name,
            function_args=prep.function_args,
            effective_task_id=effective_task_id,
            tool_call_id=prep.tool_call_id,
            start_time=start,
            middleware_trace=list(prep.middleware_trace),
        )
        try:
            agent.interrupt("keyboard interrupt")
        except Exception:
            pass
        return ToolExecutionOutcome(
            tool_call=prep.tool_call,
            tool_call_id=prep.tool_call_id,
            function_name=prep.function_name,
            function_args=prep.function_args,
            result=result,
            duration=time.time() - start,
            is_error=True,
            blocked=False,
            cancelled=True,
            malformed=False,
            middleware_trace=prep.middleware_trace,
        )

    result, function_args, middleware_trace, blocked = _managed_values(managed)
    duration = time.time() - start
    is_error, _ = _detect_tool_failure(prep.function_name, result)

    if state["inline_post_hook"] and not blocked:
        _emit_terminal_post_tool_call(
            agent,
            function_name=prep.function_name,
            function_args=function_args,
            result=result,
            effective_task_id=effective_task_id,
            tool_call_id=prep.tool_call_id,
            duration_ms=int(duration * 1000),
            middleware_trace=list(middleware_trace),
        )

    return ToolExecutionOutcome(
        tool_call=prep.tool_call,
        tool_call_id=prep.tool_call_id,
        function_name=prep.function_name,
        function_args=function_args,
        result=result,
        duration=duration,
        is_error=is_error,
        blocked=blocked,
        cancelled=False,
        malformed=False,
        middleware_trace=middleware_trace,
    )


def finalize_tool_outcome(agent, outcome: ToolExecutionOutcome) -> Any:
    """Apply the after-call guardrail observation and file-mutation record.

    Kept out of :func:`execute_one_tool` because the concurrent path runs it
    on the collection thread in tool-call order — ``ToolGuardrails.after_call``
    and the turn-end file-mutation verifier are single-threaded state that a
    worker pool must not touch. Blocked calls never ran, so a guardrail block
    counts as neither a failure nor a success.
    """
    if outcome.blocked:
        return outcome.result

    result = agent._append_guardrail_observation(
        outcome.function_name,
        outcome.function_args,
        outcome.result,
        failed=outcome.is_error,
    )
    try:
        agent._record_file_mutation_result(
            outcome.function_name, outcome.function_args, result, outcome.is_error,
        )
    except Exception as _ver_err:
        logging.debug("file-mutation verifier record failed: %s", _ver_err)
    return result


def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute multiple tool calls concurrently using a thread pool.

    Results are collected in the original tool-call order and appended to
    messages so the API sees them in the expected sequence.

    ``finalize=False`` skips the end-of-batch aggregate budget enforcement
    and /steer injection — used when this call is one segment of a larger
    mixed batch and the segmented dispatcher owns the turn-end work.
    """
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)

    # Resolve the context-scaled tool-output budget once per turn (cheap, but
    # avoids rebuilding it per result inside the loop below).
    _tool_budget = _budget_for_agent(agent)

    # ── Pre-flight: interrupt check ──────────────────────────────────
    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        for tc in tool_calls:
            messages.append(make_tool_result_message(
                tc.function.name,
                f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                tc.id,
                effect_disposition="none",
            ))
            _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"cancelled tool result {tc.function.name}",
            )
        return

    # ── Parse args + pre-execution bookkeeping ───────────────────────
    parsed_calls = [prepare_tool_call(agent, tool_call) for tool_call in tool_calls]

    # ── Logging / callbacks ──────────────────────────────────────────
    tool_names_str = ", ".join(prep.function_name for prep in parsed_calls)
    if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")

    # ── Concurrent execution ─────────────────────────────────────────
    # Each slot holds the call's ``ToolExecutionOutcome`` once it lands.
    results: list = [None] * num_tools
    for i, prep in enumerate(parsed_calls):
        if prep.malformed_result is not None:
            results[i] = ToolExecutionOutcome(
                tool_call=prep.tool_call,
                tool_call_id=prep.tool_call_id,
                function_name=prep.function_name,
                function_args=prep.function_args,
                result=prep.malformed_result,
                duration=0.0,
                is_error=True,
                blocked=True,
                cancelled=False,
                malformed=True,
                middleware_trace=prep.middleware_trace,
            )

    start_condition = threading.Condition()
    next_start_order = 0
    authorization_gate = _ConcurrentToolAuthorizationGate()

    def _begin_in_order(order: int, callback=None) -> None:
        nonlocal next_start_order
        with start_condition:
            start_condition.wait_for(lambda: order == next_start_order)
            try:
                if callback is not None:
                    callback()
            finally:
                next_start_order += 1
                start_condition.notify_all()

    # Touch activity before launching workers so the gateway knows
    # we're executing tools (not stuck).
    agent._current_tool = tool_names_str
    agent._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

    def _run_tool(index, prep, start_order):
        """Worker function executed in a thread."""
        # Register this worker tid so the agent can fan out an interrupt
        # to it — see AIAgent.interrupt().  Must happen first thing, and
        # must be paired with discard + clear in the finally block.
        _worker_tid = threading.current_thread().ident
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(_worker_tid)
        # Race: if the agent was interrupted between fan-out (which
        # snapshotted an empty/earlier set) and our registration, apply
        # the interrupt to our own tid now so is_interrupted() inside
        # the tool returns True on the next poll.
        if agent._interrupt_requested:
            try:
                _ra()._set_interrupt(True, _worker_tid)
            except Exception:
                pass
        # Set the activity callback on THIS worker thread so
        # _wait_for_process (terminal commands) can fire heartbeats.
        # The callback is thread-local; the main thread's callback
        # is invisible to worker threads.
        try:
            from tools.environments.base import set_activity_callback
            set_activity_callback(agent._touch_activity)
        except Exception:
            pass
        # Approval/sudo callbacks (thread-local) and the agent turn's
        # ContextVars are propagated by propagate_context_to_thread() at the
        # submit site below (GHSA-qg5c-hvr5-hjgr, #13617).
        function_name = prep.function_name
        start = time.time()
        start_advanced = False

        def _advance_start(callback=None) -> None:
            nonlocal start_advanced
            if start_advanced:
                return
            try:
                _begin_in_order(start_order, callback)
            finally:
                start_advanced = True

        try:
            try:
                outcome = execute_one_tool(
                    agent,
                    prep.tool_call,
                    effective_task_id,
                    messages=messages,
                    prepared=prep,
                    display_index=index + 1,
                    begin_execution=_advance_start,
                    authorization_gate=authorization_gate,
                )
            except Exception as tool_error:
                logger.error("execute_one_tool raised for %s: %s", function_name, tool_error, exc_info=True)
                results[index] = ToolExecutionOutcome(
                    tool_call=prep.tool_call,
                    tool_call_id=prep.tool_call_id,
                    function_name=function_name,
                    function_args=prep.function_args,
                    result=f"Error executing tool '{function_name}': {tool_error}",
                    duration=time.time() - start,
                    is_error=True,
                    blocked=False,
                    cancelled=False,
                    malformed=False,
                    middleware_trace=prep.middleware_trace,
                )
                return
            if outcome.cancelled:
                logger.info("tool %s cancelled (%.2fs)", function_name, outcome.duration)
            elif outcome.is_error:
                logger.info(
                    "tool %s failed (%.2fs): %s",
                    function_name, outcome.duration, str(outcome.result)[:200],
                )
            else:
                logger.info(
                    "tool %s completed (%.2fs, %d chars)",
                    function_name, outcome.duration, len(outcome.result),
                )
            results[index] = outcome
        finally:
            _advance_start()
            # Tear down worker-tid tracking.  Clear any interrupt bit we may
            # have set so the next task scheduled onto this recycled tid
            # starts with a clean slate.  This MUST be in a finally block
            # because BaseException subclasses (CancelledError, KeyboardInterrupt)
            # bypass ``except Exception`` and would otherwise leak the tid
            # into _interrupted_threads, poisoning the recycled thread.
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(_worker_tid)
            try:
                _ra()._set_interrupt(False, _worker_tid)
            except Exception:
                pass

    # Start spinner for CLI mode (skip when TUI handles tool progress)
    spinner = None
    if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
        face = random.choice(KawaiiSpinner.get_waiting_faces())
        spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=agent._print_fn)
        spinner.start()

    try:
        runnable_calls = [
            (i, prep)
            for i, prep in enumerate(parsed_calls)
            if prep.malformed_result is None
        ]
        futures = []
        future_to_index = {}
        timed_out_indices: set[int] = set()
        timeout_s = _resolve_concurrent_tool_timeout()
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        if runnable_calls:
            max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)
            # Daemon workers: an interrupted/timed-out batch is abandoned with
            # shutdown(wait=False), but stdlib ThreadPoolExecutor workers are
            # non-daemon and registered in concurrent.futures' atexit hook,
            # which joins them unconditionally — so one wedged tool thread
            # would block interpreter exit forever (multi-minute CLI exits).
            from tools.daemon_pool import DaemonThreadPoolExecutor
            executor = DaemonThreadPoolExecutor(max_workers=max_workers)
            abandon_executor = False
            try:
                for submit_index, (i, prep) in enumerate(runnable_calls):
                    # Propagate the agent turn's ContextVars (e.g.
                    # _approval_session_key) AND thread-local approval/sudo
                    # callbacks into the worker thread; clears callbacks on exit.
                    try:
                        f = executor.submit(
                            propagate_context_to_thread(_run_tool),
                            i,
                            prep,
                            submit_index,
                        )
                    except RuntimeError as submit_error:
                        if not _is_interpreter_shutdown_submit_error(submit_error):
                            raise
                        skipped_calls = runnable_calls[submit_index:]
                        logger.warning(
                            "interpreter shutdown while scheduling concurrent tools; "
                            "skipping %d unsubmitted tool(s)",
                            len(skipped_calls),
                        )
                        for skipped_i, skipped_prep in skipped_calls:
                            if results[skipped_i] is None:
                                results[skipped_i] = ToolExecutionOutcome(
                                    tool_call=skipped_prep.tool_call,
                                    tool_call_id=skipped_prep.tool_call_id,
                                    function_name=skipped_prep.function_name,
                                    function_args=skipped_prep.function_args,
                                    result=(
                                        f"Error executing tool '{skipped_prep.function_name}': "
                                        "Python interpreter is shutting down; tool was not started"
                                    ),
                                    duration=0.0,
                                    is_error=True,
                                    blocked=False,
                                    cancelled=False,
                                    malformed=False,
                                    middleware_trace=skipped_prep.middleware_trace,
                                )
                        break
                    futures.append(f)
                    future_to_index[f] = i

                # Wait for all to complete with periodic heartbeats so the
                # gateway's inactivity monitor doesn't kill us during long
                # concurrent tool batches. Also check for user interrupts
                # so we don't block indefinitely when the user sends /stop
                # or a new message during concurrent tool execution.
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    wait_timeout = 5.0
                    if deadline is not None:
                        effective_deadline = (
                            deadline + authorization_gate.excluded_seconds()
                        )
                        remaining = effective_deadline - time.monotonic()
                        if remaining <= 0:
                            done, not_done = set(), {
                                f for f in futures if not f.done()
                            }
                        else:
                            wait_timeout = min(wait_timeout, remaining)
                            done, not_done = concurrent.futures.wait(
                                futures, timeout=wait_timeout,
                            )
                    else:
                        done, not_done = concurrent.futures.wait(
                            futures, timeout=wait_timeout,
                        )
                    if not not_done:
                        break

                    if (
                        deadline is not None
                        and time.monotonic()
                        >= deadline + authorization_gate.excluded_seconds()
                    ):
                        abandon_executor = True
                        timed_out_indices = {
                            future_to_index[f]
                            for f in not_done
                            if f in future_to_index
                        }
                        _still_running = [
                            parsed_calls[i].function_name
                            for i in timed_out_indices
                        ]
                        logger.warning(
                            "concurrent tool batch timed out after %.1fs; "
                            "%d tool(s) still running: %s",
                            timeout_s,
                            len(timed_out_indices),
                            ", ".join(_still_running[:5]),
                        )
                        for f in not_done:
                            f.cancel()
                        with agent._tool_worker_threads_lock:
                            worker_tids = list(agent._tool_worker_threads)
                        for tid in worker_tids:
                            try:
                                _ra()._set_interrupt(True, tid)
                            except Exception:
                                pass
                        break

                    # Check for interrupt — the per-thread interrupt signal
                    # already causes individual tools (terminal, execute_code)
                    # to abort, but tools without interrupt checks (web_search,
                    # read_file) will run to completion. Cancel any futures
                    # that haven't started yet so we don't block on them.
                    if agent._interrupt_requested:
                        abandon_executor = True
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            agent._vprint(
                                f"{agent.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # Give already-running tools a moment to notice the
                        # per-thread interrupt signal and exit gracefully.
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    _conc_elapsed = int(time.time() - _conc_start)
                    # Heartbeat every ~30s (6 × 5s poll intervals)
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[future_to_index[f]].function_name
                            for f in not_done
                            if f in future_to_index
                        ]
                        agent._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
            finally:
                # On abandon (interrupt or deadline) we intentionally do NOT
                # join hung workers: wait=False returns immediately and
                # cancel_futures drops queued-but-unstarted work. A wedged tool
                # thread is left running detached — the deliberate tradeoff vs.
                # deadlocking the whole batch. Normal completion joins (wait=True).
                executor.shutdown(
                    wait=not abandon_executor,
                    cancel_futures=abandon_executor,
                )
    finally:
        if spinner:
            # Build a summary message for the spinner stop
            completed = sum(1 for r in results if r is not None)
            total_dur = sum(r.duration for r in results if r is not None)
            spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

    # ── Post-execution: display per-tool results ─────────────────────
    for i, prep in enumerate(parsed_calls):
        tc = prep.tool_call
        name = prep.function_name
        args = prep.function_args
        middleware_trace = prep.middleware_trace
        r = results[i]
        blocked = False
        is_error = True
        progress_function_name = name
        # A worker can finish and write results[i] in the window between the
        # deadline snapshot (timed_out_indices, taken from not_done) and this
        # loop. Prefer that real result over a fabricated timeout message — the
        # tool genuinely succeeded, just slightly late.
        effect_disposition = None
        if i in timed_out_indices and r is None:
            suffix = f"{timeout_s:.1f}s" if timeout_s is not None else "the configured timeout"
            function_result = f"Error executing tool '{name}': timed out after {suffix}"
            effect_disposition = "unknown"
            _emit_terminal_post_tool_call(
                agent,
                function_name=name,
                function_args=args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tc, "id", "") or "",
                status="timeout",
                error_type="tool_timeout",
                error_message=function_result,
                middleware_trace=list(middleware_trace),
            )
            tool_duration = float(timeout_s or 0.0)
        elif r is None:
            # Tool was cancelled (interrupt) or thread didn't return
            if agent._interrupt_requested:
                function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tc, "id", "") or "",
                    status="cancelled",
                    error_type="keyboard_interrupt",
                    error_message="Tool execution cancelled by user interrupt",
                    middleware_trace=list(middleware_trace),
                )
            else:
                function_result = f"Error executing tool '{name}': thread did not return a result"
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=name,
                    function_args=args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(tc, "id", "") or "",
                    status="error",
                    error_type="thread_missing_result",
                    error_message=function_result,
                    middleware_trace=list(middleware_trace),
                )
            tool_duration = 0.0
        else:
            function_name = r.function_name
            function_args = r.function_args
            tool_duration = r.duration
            is_error = r.is_error
            blocked = r.blocked
            middleware_trace = r.middleware_trace
            name = function_name
            args = function_args
            progress_function_name = function_name
            if blocked:
                effect_disposition = "none"

            function_result = finalize_tool_outcome(agent, r)

            if is_error:
                _err_text = _multimodal_text_summary(function_result)
                result_preview = _err_text[:200] if len(_err_text) > 200 else _err_text
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

            if agent.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

        agent._current_tool = None
        _status_suffix = " (error)" if is_error else ""
        agent._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s){_status_suffix}")

        display_function_result = function_result
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=name,
            tool_use_id=tc.id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(function_result) else function_result

        subdir_hints = agent._subdirectory_hints.check_tool_call(name, args)
        if subdir_hints:
            if _is_multimodal_tool_result(function_result):
                # Append the hint to the text summary part so the model
                # still sees it; don't touch the image blocks.
                _append_subdir_hint_to_multimodal(function_result, subdir_hints)
            else:
                function_result += subdir_hints

        # Unwrap _multimodal dicts to an OpenAI-style content list so any
        # vision-capable provider receives [{type:text},{type:image_url}]
        # rather than a raw Python dict.  The Anthropic adapter already
        # accepts content lists; vision-capable OpenAI-compatible servers
        # (mlx-vlm, GPT-4o, …) accept image_url in tool messages natively.
        # Text-only servers get a string-safe fallback here so a rejected
        # image tool result never poisons canonical session history.
        # String results pass through unchanged.
        _tool_content = agent._tool_result_content_for_active_model(name, function_result)
        tool_message = make_tool_result_message(
            name,
            _tool_content,
            tc.id,
            effect_disposition=effect_disposition,
        )
        messages.append(tool_message)
        risk_metadata = tool_message.get("_tool_output_risk")
        if not _flush_session_db_after_tool_progress(
            agent,
            messages,
            stage=f"tool result {name}",
        ):
            return

        # Every completion surface is downstream of the canonical append. If
        # the UI bridge or process dies while projecting one of these events,
        # resume can reconstruct the tool result that was already visible.
        if not blocked and agent.tool_progress_callback:
            try:
                agent.tool_progress_callback(
                    "tool.completed", progress_function_name, None, None,
                    duration=tool_duration, is_error=is_error,
                    result=display_function_result,
                )
            except Exception as cb_err:
                logging.debug(f"Tool progress callback error: {cb_err}")

        # Print cute message per tool
        if agent._should_emit_quiet_tool_messages():
            cute_msg = _get_cute_tool_message_impl(
                name, args, tool_duration, result=display_function_result,
            )
            agent._safe_print(f"  {cute_msg}")
        elif not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            _preview_str = _multimodal_text_summary(display_function_result)
            if agent.verbose_logging:
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", _preview_str))
            else:
                response_preview = _preview_str[:agent.log_prefix_chars] + "..." if len(_preview_str) > agent.log_prefix_chars else _preview_str
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

        if not blocked and agent.tool_complete_callback:
            try:
                display_args = _redact_tool_args_for_display(name, args) or args
                agent.tool_complete_callback(
                    tc.id, name, display_args, display_function_result,
                )
            except Exception as cb_err:
                logging.debug(f"Tool complete callback error: {cb_err}")

        if (
            risk_metadata is not None
            and risk_metadata.get("risk") != "low"
            and agent.tool_progress_callback
        ):
            try:
                agent.tool_progress_callback(
                    "tool.output_risk",
                    name,
                    None,
                    None,
                    tool_call_id=tc.id,
                    risk_metadata=risk_metadata,
                )
            except Exception as cb_err:
                logging.debug("Tool output risk callback error: %s", cb_err)

        # ── Per-tool /steer drain ───────────────────────────────────
        # Same as the sequential path: drain between each collected
        # result so the steer lands as early as possible.
        agent._apply_pending_steer_to_tool_results(messages, 1)

    # ── Per-turn aggregate budget enforcement ─────────────────────────
    num_tools = len(parsed_calls)
    if finalize and num_tools > 0:
        turn_tool_msgs = messages[-num_tools:]
        enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ──────────────────────────────────────────────
    # Append any pending user steer text to the last tool result so the
    # agent sees it on its next iteration. Runs AFTER budget enforcement
    # so the steer marker is never truncated. See steer() for details.
    if finalize and num_tools > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools)



# Agent-level tools whose sequential display is a single cute line after the
# call — they return fast enough that a spinner would flash and vanish.
_SEQUENTIAL_MESSAGE_TOOLS = frozenset(
    {"todo", "session_search", "memory", "clarify", "read_terminal"}
)


def _start_sequential_tool_display(agent, function_name: str, function_args: dict):
    """Pick the sequential path's display mode for one tool and start its spinner.

    Returns ``(mode, spinner)`` where mode is ``"message"`` (cute line after
    the call), ``"spinner"`` (live spinner, replaced by the cute line), or
    ``"silent"`` (the verbose per-tool tail prints instead).
    """
    if function_name in _SEQUENTIAL_MESSAGE_TOOLS:
        return "message", None

    context_engine_names = getattr(agent, "_context_engine_tool_names", None)
    is_context_engine = bool(
        context_engine_names and function_name in context_engine_names
    )
    memory_manager = getattr(agent, "_memory_manager", None)
    is_memory_manager = bool(
        memory_manager is not None and memory_manager.has_tool(function_name)
    )
    is_delegate = function_name == "delegate_task"

    if not (is_delegate or is_context_engine or is_memory_manager or agent.quiet_mode):
        return "silent", None

    if is_delegate:
        tasks_arg = function_args.get("tasks")
        if tasks_arg and isinstance(tasks_arg, list):
            label = f"🔀 delegating {len(tasks_arg)} tasks · (/agents to monitor)"
        else:
            goal_preview = (function_args.get("goal") or "")[:30]
            label = (
                f"🔀 {goal_preview} · (/agents to monitor)"
                if goal_preview
                else "🔀 delegating · (/agents to monitor)"
            )
        allowed = (
            agent._should_emit_quiet_tool_messages()
            and agent._should_start_quiet_spinner()
        )
    else:
        emoji = _get_tool_emoji(function_name)
        display_args = (
            _redact_tool_args_for_display(function_name, function_args) or function_args
        )
        preview = _build_tool_label(function_name, display_args) or function_name
        label = f"{emoji} {preview}"
        # Context-engine tools spin even when the quiet-spinner gate is
        # closed: they hold the loop for seconds with no other output.
        allowed = agent._should_emit_quiet_tool_messages() and (
            is_context_engine or agent._should_start_quiet_spinner()
        )

    spinner = None
    if allowed:
        face = random.choice(KawaiiSpinner.get_waiting_faces())
        spinner = KawaiiSpinner(
            f"{face} {label}", spinner_type='dots', print_fn=agent._print_fn,
        )
        spinner.start()
    return "spinner", spinner


def _finish_sequential_tool_display(
    agent, mode: str, spinner, function_name: str, function_args: dict,
    duration: float, result: Any,
) -> None:
    """Close out the display started by :func:`_start_sequential_tool_display`."""
    if mode == "silent":
        return
    if mode == "message":
        if agent._should_emit_quiet_tool_messages():
            agent._vprint(
                f"  {_get_cute_tool_message_impl(function_name, function_args, duration, result=result)}"
            )
        return
    cute_msg = _get_cute_tool_message_impl(
        function_name, function_args, duration, result=result,
    )
    if spinner:
        spinner.stop(cute_msg)
    elif agent._should_emit_quiet_tool_messages():
        agent._vprint(f"  {cute_msg}")


def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools.

    ``finalize=False`` skips the end-of-batch aggregate budget enforcement
    and /steer injection — used when this call is one segment of a larger
    mixed batch and the segmented dispatcher owns the turn-end work.
    """
    # Resolve the context-scaled tool-output budget once per turn.
    _tool_budget = _budget_for_agent(agent)
    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        # SAFETY: check interrupt BEFORE starting each tool.
        # If the user sent "stop" during a previous tool's execution,
        # do NOT start any more tools -- skip them all immediately.
        if agent._interrupt_requested:
            remaining_calls = assistant_message.tool_calls[i-1:]
            if remaining_calls:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
            for skipped_tc in remaining_calls:
                skipped_name = skipped_tc.function.name
                messages.append(make_tool_result_message(
                    skipped_name,
                    f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                    skipped_tc.id,
                    effect_disposition="none",
                ))
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"cancelled tool result {skipped_name}",
                ):
                    return
            break

        prep = prepare_tool_call(agent, tool_call)
        function_name = prep.function_name

        if prep.malformed_result is not None:
            messages.append(
                make_tool_result_message(
                    function_name,
                    prep.malformed_result,
                    tool_call.id,
                )
            )
            if not _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"invalid tool arguments {function_name}",
            ):
                return
            agent._apply_pending_steer_to_tool_results(messages, 1)
            continue

        tool_start_time = time.time()
        display_mode, spinner = _start_sequential_tool_display(
            agent, function_name, prep.function_args
        )
        if function_name == "delegate_task":
            agent._delegate_spinner = spinner
        outcome = None
        try:
            outcome = execute_one_tool(
                agent,
                tool_call,
                effective_task_id,
                messages=messages,
                prepared=prep,
                display_index=i,
            )
        finally:
            if function_name == "delegate_task":
                agent._delegate_spinner = None
            # A tool that raised past execute_one_tool leaves no outcome; only
            # the spinner modes have to be closed out in that case.
            if outcome is not None or display_mode == "spinner":
                _finish_sequential_tool_display(
                    agent,
                    display_mode,
                    spinner,
                    function_name,
                    outcome.function_args if outcome is not None else prep.function_args,
                    outcome.duration if outcome is not None else time.time() - tool_start_time,
                    outcome.result if outcome is not None else None,
                )

        if outcome.cancelled:
            raise KeyboardInterrupt

        function_args = outcome.function_args
        function_result = outcome.result
        tool_duration = outcome.duration
        _execution_blocked = outcome.blocked
        _is_error_result = outcome.is_error


        if isinstance(function_result, str):
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
            _result_len = len(function_result)
        else:
            # Multimodal dict result (_multimodal=True) — not sliceable as string
            result_preview = function_result
            _result_len = len(str(function_result))

        # Log tool errors to the persistent error log so [error] tags
        # in the UI always have a corresponding detailed entry on disk.
        if not _execution_blocked:
            function_result = finalize_tool_outcome(agent, outcome)
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
        if _is_error_result:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

        agent._current_tool = None
        _status_suffix = " (error)" if _is_error_result else ""
        agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s){_status_suffix}")

        if agent.verbose_logging:
            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
            _log_result = _multimodal_text_summary(function_result)
            logging.debug(f"Tool result ({len(_log_result)} chars): {_log_result}")

        display_function_result = function_result
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=function_name,
            tool_use_id=tool_call.id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(function_result) else function_result

        # Discover subdirectory context files from tool arguments
        subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
        if subdir_hints:
            if _is_multimodal_tool_result(function_result):
                _append_subdir_hint_to_multimodal(function_result, subdir_hints)
            else:
                function_result += subdir_hints

        # Unwrap _multimodal dicts to an OpenAI-style content list
        # (see parallel path for rationale). String results pass through.
        _tool_content = agent._tool_result_content_for_active_model(function_name, function_result)
        tool_message = make_tool_result_message(function_name, _tool_content, tool_call.id)
        messages.append(tool_message)
        risk_metadata = tool_message.get("_tool_output_risk")
        if not _flush_session_db_after_tool_progress(
            agent,
            messages,
            stage=f"tool result {function_name}",
        ):
            return

        # UI completion/progress events are projections of the canonical tool
        # row, never a competing in-memory authority.
        if not _execution_blocked and agent.tool_progress_callback:
            try:
                agent.tool_progress_callback(
                    "tool.completed", function_name, None, None,
                    duration=tool_duration, is_error=_is_error_result,
                    result=display_function_result,
                )
            except Exception as cb_err:
                logging.debug(f"Tool progress callback error: {cb_err}")

        if not _execution_blocked and agent.tool_complete_callback:
            try:
                display_args = (
                    _redact_tool_args_for_display(function_name, function_args)
                    or function_args
                )
                agent.tool_complete_callback(
                    tool_call.id,
                    function_name,
                    display_args,
                    display_function_result,
                )
            except Exception as cb_err:
                logging.debug(f"Tool complete callback error: {cb_err}")

        if (
            risk_metadata is not None
            and risk_metadata.get("risk") != "low"
            and agent.tool_progress_callback
        ):
            try:
                agent.tool_progress_callback(
                    "tool.output_risk",
                    function_name,
                    None,
                    None,
                    tool_call_id=tool_call.id,
                    risk_metadata=risk_metadata,
                )
            except Exception as cb_err:
                logging.debug("Tool output risk callback error: %s", cb_err)

        # ── Per-tool /steer drain ───────────────────────────────────
        # Drain pending steer BETWEEN individual tool calls so the
        # injection lands as soon as a tool finishes — not after the
        # entire batch.  The model sees it on the next API iteration.
        agent._apply_pending_steer_to_tool_results(messages, 1)

        if not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off":
            if agent.verbose_logging:
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                print(agent._wrap_verbose("Result: ", function_result))
            else:
                _fr_str = function_result if isinstance(function_result, str) else str(function_result)
                response_preview = _fr_str[:agent.log_prefix_chars] + "..." if len(_fr_str) > agent.log_prefix_chars else _fr_str
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

        if agent._interrupt_requested and i < len(assistant_message.tool_calls):
            remaining = len(assistant_message.tool_calls) - i
            agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
            for skipped_tc in assistant_message.tool_calls[i:]:
                skipped_name = skipped_tc.function.name
                messages.append(make_tool_result_message(
                    skipped_name,
                    f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                    skipped_tc.id,
                    effect_disposition="none",
                ))
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"skipped tool result {skipped_name}",
                ):
                    return
            break

    # ── Per-turn aggregate budget enforcement ─────────────────────────
    num_tools_seq = len(assistant_message.tool_calls)
    if finalize and num_tools_seq > 0:
        enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id), config=_tool_budget)

    # ── /steer injection ──────────────────────────────────────────────
    # See _execute_tool_calls_parallel for the rationale. Same hook,
    # applied to sequential execution as well.
    if finalize and num_tools_seq > 0:
        agent._apply_pending_steer_to_tool_results(messages, num_tools_seq)




def execute_tool_calls_segmented(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments=None) -> None:
    """Execute a mixed tool-call batch as ordered parallel/sequential segments.

    ``segments`` is the ``(kind, calls)`` plan from
    ``_plan_tool_batch_segments``: maximal contiguous runs of parallel-safe
    calls execute on the concurrent path, barrier calls on the sequential
    path, strictly in the model's original call order. Because segments are
    contiguous, every tool result is still appended one-per-call in emission
    order and no call ever starts before an earlier barrier finishes —
    identical ordering and side-effect boundaries to fully-sequential
    execution, with I/O parallelism recovered inside the safe runs.

    Turn-end work (aggregate budget enforcement + /steer injection) is done
    once here for the WHOLE batch; the per-segment executor calls run with
    ``finalize=False`` so a multi-segment turn cannot multiply the budget or
    truncate a steer marker.

    Interrupt semantics: each segment executor already checks
    ``agent._interrupt_requested`` up front and appends a cancelled/skipped
    result per call, so an interrupt during segment *k* drains segments
    *k+1..n* without executing them while preserving one result per
    tool_call_id.
    """
    from types import SimpleNamespace

    if segments is None:
        _active_env = get_active_env(effective_task_id)
        _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
        segments = _plan_tool_batch_segments(assistant_message.tool_calls, execution_cwd=_exec_cwd)

    for kind, calls in segments:
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        segment_message = SimpleNamespace(tool_calls=list(calls))
        if kind == "parallel":
            execute_tool_calls_concurrent(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )
        else:
            execute_tool_calls_sequential(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )

        if getattr(agent, "_incremental_persistence_failed", False):
            return

    # ── Whole-turn finalize (budget + /steer) ─────────────────────────
    total_tools = len(assistant_message.tool_calls)
    if total_tools > 0:
        _tool_budget = _budget_for_agent(agent)
        enforce_turn_budget(
            messages[-total_tools:],
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        )
        agent._apply_pending_steer_to_tool_results(messages, total_tools)


__all__ = [
    "PreparedToolCall",
    "ToolExecutionOutcome",
    "prepare_tool_call",
    "execute_one_tool",
    "finalize_tool_outcome",
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
    "execute_tool_calls_segmented",
]
