"""Tool-call execution: sequential and concurrent dispatch, extracted from AIAgent.

Functions take the parent ``AIAgent`` first; ``run_agent`` keeps thin wrappers and is
reached lazily via ``_ra()`` so ``run_agent._set_interrupt`` patches still work. Every
call's identity travels as a ``_ToolCallRef``; both executors end in the same
observe → commit → project pipeline so the tool-result wire shape is produced once.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
from pathlib import Path
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    build_tool_label as _build_tool_label,
    get_cute_tool_message as _get_cute_tool_message_impl,
    get_tool_emoji as _get_tool_emoji,
    redact_tool_args_for_display as _redact_tool_args_for_display,
    _detect_tool_failure,
)
from agent.message_sanitization import coalesce_tool_call_id
from agent.tool_dispatch_helpers import (
    _NEVER_PARALLEL_TOOLS,
    _is_destructive_command,
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _append_subdir_hint_to_multimodal,
    _plan_tool_batch_segments,
    make_tool_result_message,
)
from tools.terminal_tool_lifecycle import get_active_env
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import (
    maybe_persist_tool_result,
    enforce_turn_budget,
    extract_persisted_path,
)
from tools.budget_config import BudgetConfig, DEFAULT_BUDGET, budget_for_context_window

logger = logging.getLogger(__name__)


def _pairing_tool_call_id(tool_call: Any) -> str:
    """Return the canonical id used by the persisted assistant message."""
    return coalesce_tool_call_id(tool_call)


def _record_persisted_path_for_stub(agent, tool_call_id: str, function_result) -> None:
    """Tell the stall guards where a persisted result's full content lives.

    When a large result is spilled to disk (<persisted-output> preview), a
    later result-reference stub pointing at that first occurrence must carry
    the spillover file path so the reference can't dangle. Best-effort: never
    lets bookkeeping break tool execution.
    """
    try:
        if not isinstance(function_result, str):
            return
        path = extract_persisted_path(function_result)
        if path:
            agent._tool_guardrails.record_persisted_result(tool_call_id, path)
    except Exception as exc:
        logger.debug("persisted-path record for result stub failed: %s", exc)


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
    from agent.file_safety import is_nt_namespace_path
    from tools.file_tools_paths import _resolve_path_for_task

    # Resolving an NT-namespace path is itself the NTLM-leak trigger; leave the
    # tool's raw-string guard to refuse it without a checkpoint stat.
    if is_nt_namespace_path(file_path):
        return
    resolved_path = _resolve_path_for_task(file_path, effective_task_id or "default")
    agent._checkpoint_mgr.ensure_checkpoint(
        agent._checkpoint_mgr.get_working_dir_for_path(str(resolved_path)), f"before {function_name}",
    )


def _budget_for_agent(agent) -> BudgetConfig:
    """Tool-result BudgetConfig scaled to the agent's context window. Unknown length goes
    through ``budget_for_context_window(None)`` (not DEFAULT_BUDGET) so the MCP threshold
    override still applies.

    Large-context models keep the historical 100K/200K char defaults; small models (e.g. a 65K-token local
    model switched into mid-session) get a budget proportional to their window so a single large tool result
    can't push the request past the model's limit (#23767). Falls back to the default budget when the
    context length isn't resolvable.
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        # budget_for_context_window(None) (rather than DEFAULT_BUDGET) so the
        # config-driven MCP threshold override still applies when the context
        # length isn't resolvable.
        return budget_for_context_window(int(ctx) if ctx else None)
    except Exception:
        return DEFAULT_BUDGET

_MAX_TOOL_WORKERS = 8  # concurrent worker threads per batch
_DEFAULT_IMAGE_PARALLEL_REQUESTS = 4
# Generous ceiling for slow-but-valid tool work (large page fetches, slow
# remote backends) so the batch guard does not preempt a legitimate attempt.
_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0
# Long enough for an approval round-trip, short enough that one wedged dispatch can't starve the batch.
_START_ORDER_GATE_TIMEOUT_S = 120.0
# Fallback only; the effective bound derives from approvals.timeout (_authorization_gate_lock_timeout).
_AUTHORIZATION_GATE_LOCK_TIMEOUT_S = 360.0


def _authorization_gate_lock_timeout() -> float:
    """Authorization-lock bound = ``tools.approval_human_wait.human_wait_ceiling`` (approval timeout +
    margin, capped so it can't overflow Lock.acquire): never break serialization while a
    prompt is answerable, never let a wedged holder park workers forever. Deliberately NOT
    min()'d with the fallback so the gate never gives up early.

    Delegates to ``tools.approval_human_wait.human_wait_ceiling`` — the same bound that clamps a human-wait window's
    deadline contribution — so the two can't drift. Long enough that serialization is never broken while a
    legitimate approval prompt is still answerable; short enough that a wedged holder (hanging
    ``pre_tool_call`` plugin, dead approval client) cannot park other workers forever (#79719). Resolved
    once per gate (per batch), so a mid-process ``approvals.timeout`` change applies from the next batch.
    """
    try:
        from tools.approval_human_wait import human_wait_ceiling

        # human_wait_ceiling is platform-safety-capped (agent/deadline.py
        # MAX_SAFE_TIMEOUT_S): a huge approvals.timeout can no longer overflow
        # Lock.acquire's time_t on macOS (#83220). Deliberately NOT min()'d
        # with _AUTHORIZATION_GATE_LOCK_TIMEOUT_S — the gate must never give
        # up while a legitimate approval prompt is still answerable (#79719),
        # so a configured approvals.timeout above 360s must extend the gate.
        return human_wait_ceiling()
    except Exception:
        return _AUTHORIZATION_GATE_LOCK_TIMEOUT_S


class _BatchAbandoned(BaseException):
    """Raised inside a worker when the batch was abandoned before dispatch; a BaseException
    so ``except Exception`` handlers in the middleware chain can't swallow it."""


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """Parse model-emitted arguments without repairing or coercing them."""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {"error": "Invalid tool arguments", "message": "Tool arguments must be a valid JSON object; tool was not executed."},
        ensure_ascii=False,
    )


def _resolve_concurrent_tool_timeout() -> float | None:
    """Per-batch concurrent deadline: ``timeouts.tools.concurrent_batch`` wins,
    ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` is the legacy bridge, ``0``/negative disables."""
    from agent.deadline import resolve_timeout

    return resolve_timeout(
        "tools.concurrent_batch",
        default=_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
        env_var="HERMES_CONCURRENT_TOOL_TIMEOUT_S",
    )


def _flush_session_db_after_tool_progress(agent, messages: list, *, stage: str) -> bool:
    """Flush tool-call progress to the session DB before projecting it to any UI: tool side
    effects can kill/restart the process before turn-end persistence runs."""
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning

    # Persist exactly the checkpoint text the next model call will see, before stamping
    # this tool result as durable. Already-written rows must never be rewritten later.
    _maybe_inject_run_budget_wrapup(agent, messages)
    _maybe_inject_iteration_budget_warning(agent, messages)
    try:
        persisted = agent._flush_messages_to_session_db(messages) is not False
        if not persisted:
            agent._incremental_persistence_failed = True
            # The flush recorded any classified cause; default to 'unknown' only if nothing more specific exists.
            if getattr(agent, "_last_persistence_error_cause", None) is None:
                agent._last_persistence_error_cause = "unknown"
        return persisted
    except Exception as exc:
        agent._incremental_persistence_failed = True
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning("Incremental tool-call persistence failed after %s: %s", stage, exc)
        return False


def _image_generate_parallel_limit() -> int:
    """Configured image-generation parallelism cap (conservative: backend bursts hit rate limits)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        value = image_gen.get("max_parallel_requests") if isinstance(image_gen, dict) else None
    except Exception:
        value = None

    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = _DEFAULT_IMAGE_PARALLEL_REQUESTS
    return max(1, min(limit, _MAX_TOOL_WORKERS))


def _max_workers_for_tool_batch(runnable_calls) -> int:
    """Return the worker cap for a concurrent tool batch."""
    if not runnable_calls:
        return 0
    max_workers = _MAX_TOOL_WORKERS
    if any((call[2] if len(call) >= 3 else None) == "image_generate" for call in runnable_calls):
        max_workers = min(max_workers, _image_generate_parallel_limit())
    return min(len(runnable_calls), max_workers)


def _ra():
    """Lazy reference to ``run_agent`` so patches like ``run_agent._set_interrupt`` work."""
    import run_agent
    return run_agent


def _is_interpreter_shutdown_submit_error(exc: RuntimeError) -> bool:
    """Shutdown-race predicate — shared home in ``tools.interpreter_shutdown``.

    Delegates so all sites (cron delivery, conversation-loop retry, tool
    submission) recognize both CPython shutdown-message variants instead of
    each matching its own substring (the bug class behind #55924/#58720).
    """
    from tools.interpreter_shutdown import interpreter_shutting_down

    return interpreter_shutting_down(exc)


_emit_terminal_post_tool_call = emit_terminal_post_tool_call


@dataclass
class _ToolCallRef:
    """Identity of one tool call as every hook / result message sees it: the (possibly
    middleware-rewritten) name and args, the task, the pairing id and the request trace."""

    name: str
    args: dict
    task_id: str
    call_id: str
    trace: list

    def middleware_kwargs(self) -> dict[str, Any]:
        """Keyword form ``_run_agent_tool_execution_middleware`` (and tests patching it) expect."""
        return {
            "function_name": self.name, "function_args": self.args, "effective_task_id": self.task_id,
            "tool_call_id": self.call_id, "middleware_trace": self.trace,
        }

    def emit_post(self, agent, result, *, trace=None, **outcome) -> None:
        """Emit the one terminal ``post_tool_call`` for this call (``outcome`` = status /
        error_type / error_message / duration_ms). Resolved through the module attribute so
        tests patching ``_emit_terminal_post_tool_call`` still intercept."""
        _emit_terminal_post_tool_call(
            agent,
            function_name=self.name,
            function_args=self.args,
            result=result,
            effective_task_id=self.task_id,
            tool_call_id=self.call_id,
            middleware_trace=list(self.trace if trace is None else trace),
            **outcome,
        )

    def emit_cancelled(self, agent, start_time: float) -> str:
        """Synthesize the ``cancelled`` result for a KeyboardInterrupt mid-tool and emit its hook."""
        message = "Tool execution cancelled by user interrupt"
        result = json.dumps({"error": message, "status": "cancelled"}, ensure_ascii=False)
        self.emit_post(
            agent, result, duration_ms=int((time.time() - start_time) * 1000),
            status="cancelled", error_type="keyboard_interrupt", error_message=message,
        )
        return result

    def emit_invalid_arguments(self, agent, result: str) -> None:
        self.emit_post(
            agent, result, trace=[],
            status="error", error_type="invalid_tool_arguments", error_message="Tool arguments must be a valid JSON object",
        )


def _append_skipped_tool_results(
    agent,
    messages: list,
    tool_calls,
    effective_task_id: str,
    *,
    content: str,
    hook_error_type: Optional[str] = None,
    hook_id: Optional[Callable[[Any], str]] = None,
    flush_stage: Optional[str] = None,
    stop_on_flush_failure: bool = True,
) -> bool:
    """Append one ``tool`` result per unstarted call so the assistant tool-call turn never
    lacks matching results (role alternation). ``content`` is formatted with ``{name}``;
    ``hook_error_type`` also emits the terminal ``post_tool_call`` (status=cancelled) per
    call with ``hook_id`` overriding the hook's id; ``flush_stage`` flushes after each
    append and returns False on the first failed flush when ``stop_on_flush_failure``."""
    for tc in tool_calls:
        name = _tc_name(tc)
        result = content.format(name=name)
        messages.append(make_tool_result_message(name, result, _pairing_tool_call_id(tc), effect_disposition="none"))
        if hook_error_type is not None:
            _ToolCallRef(name, {}, effective_task_id, (hook_id or _pairing_tool_call_id)(tc), []).emit_post(
                agent, result,
                status="cancelled", error_type=hook_error_type, error_message="Tool execution skipped due to user interrupt",
            )
        if flush_stage is not None:
            flushed = _flush_session_db_after_tool_progress(agent, messages, stage=f"{flush_stage} {name}")
            if not flushed and stop_on_flush_failure:
                return False
    return True


def _tool_search_scoped_names(agent) -> frozenset:
    """Deferrable tool names the session may invoke via ``tool_call``; the unwrap bypasses
    the bridge's scope check in ``model_tools.handle_function_call``, so restricted sessions
    validate against this set. Cached on the agent, keyed by registry scope/generation."""
    try:
        import model_tools
        from tools import tool_search as _ts
        from tools.registry import registry as _registry
    except Exception:
        return frozenset()

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    cache_key = (
        _registry.current_scope_key(),
        getattr(_registry, "_generation", 0),
        frozenset(enabled) if enabled is not None else None,
        frozenset(disabled) if disabled is not None else None,
    )
    cached = getattr(agent, "_tool_search_scope_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    try:
        names = _ts.scoped_deferrable_names(model_tools.get_tool_definitions(
            enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True, skip_tool_search_assembly=True,
        ) or [])
    except Exception:
        names = frozenset()
    with contextlib.suppress(Exception):
        agent._tool_search_scope_cache = (cache_key, names)
    return names


def _canonical_tool_name(function_name: str) -> str:
    """Map legacy tool-name aliases BEFORE agent-loop dispatch."""
    from model_tools import _LEGACY_TOOL_ALIASES as _lta

    return _lta.get(function_name, function_name)


def _unwrap_tool_search_call(
    agent, function_name: str, function_args: dict, *, flatten_probe: bool = False
) -> tuple[str, dict, Optional[str]]:
    """Peel the ``tool_call`` bridge so downstream hooks (checkpointing, guardrails, plugin
    hooks, activity feed) see the underlying tool; ``tool_call.function`` stays untouched for
    the transcript and tool_call_id pairing.

    The unwrap bypasses handle_function_call's scope check, so session toolset scope is
    enforced HERE. Returns ``(name, args, scope_block)``; ``scope_block`` is the block
    message when the underlying tool is out of scope or its args fail the deferred-schema
    probe (``flatten_probe`` collapses the probe's JSON payload to one plain string for
    callers that wrap the message in ``{"error": ...}``).
    """
    scope_block: Optional[str] = None
    try:
        from tools import tool_search as _ts
        if function_name != _ts.TOOL_CALL_NAME:
            return function_name, function_args, None
        underlying, underlying_args, err = _ts.resolve_underlying_call(function_args)
        if err or not underlying:
            return function_name, function_args, None
        if underlying == _ts.CONNECTOR_BATCH_SENTINEL:
            # Both executors retain the wrapper: scope/probe/hooks run per entry
            # in the batch dispatcher, not against a synthetic registry name.
            return function_name, function_args, None
        if underlying not in _tool_search_scoped_names(agent):
            return function_name, function_args, (
                f"'{underlying}' is not available in this session. Use tool_search to find tools you can call."
            )
        # Validate before unwrapping: the generic bridge hides the concrete
        # parameter schema from provider-native tool-call validation.
        scope_block = _ts.validate_deferred_call_args(underlying, underlying_args)
        if scope_block is None:
            return underlying, underlying_args, None
        if flatten_probe:
            probe = json.loads(scope_block)
            scope_block = (
                f"{probe.get('error', '')} Parameters schema: "
                f"{json.dumps(probe.get('parameters', {}), ensure_ascii=False)}. "
                f"{probe.get('hint', '')}"
            ).strip()
    except Exception:
        pass
    return function_name, function_args, scope_block


@dataclass
class _ParsedCall:
    """One model tool call after alias canonicalization, arg parsing and bridge unwrap."""

    tool_call: Any
    name: str
    args: dict
    middleware_trace: list
    parse_error: Optional[str]
    scope_block: Optional[str]

    def ref(self, task_id: str) -> _ToolCallRef:
        return _ToolCallRef(self.name, self.args, task_id, _pairing_tool_call_id(self.tool_call), self.middleware_trace)


def _parse_tool_call(agent, tool_call, *, flatten_probe: bool = False) -> _ParsedCall:
    name = _canonical_tool_name(tool_call.function.name)
    args, parse_error = _parse_tool_arguments(tool_call.function.arguments)
    scope_block = None
    if parse_error is None:
        name, args, scope_block = _unwrap_tool_search_call(agent, name, args, flatten_probe=flatten_probe)
    return _ParsedCall(tool_call, name, args, [], parse_error, scope_block)


@dataclass
class _ManagedToolResult:
    result: Any
    args: dict[str, Any]
    middleware_trace: list[dict[str, Any]]
    blocked: bool
    dispatched: bool


class _ToolTimeoutResult(str):
    """Marker for a synthesized sequential-tool timeout result."""


class _ToolCancelledResult(str):
    """Marker for a synthesized sequential-tool user-interrupt result; its terminal
    post_tool_call was already emitted, so a late-finishing abandoned worker must not report."""


class _ConcurrentToolAuthorizationGate:
    """Serialize policy prompts and exclude human approval waits from batch deadlines.

    The acquire is BOUNDED: on expiry the worker prompts unserialized rather than starving
    the batch behind a wedged plugin/approval client. Exclusion is measured at the SOURCE
    of the human wait (``tools.approval.human_wait_seconds``), NOT as gate residency —
    residency-based exclusion let a wedged plugin keep the deadline from ever firing.

    Serialization keeps concurrent approval prompts from interleaving on the user's screen. The acquire is
    BOUNDED: a worker wedged inside the gate (a hanging ``pre_tool_call`` plugin, or an approval round-trip
    to a client that went away) must not park every other worker forever. On expiry the worker runs its
    prompt unserialized — worst case is interleaved prompts, strictly better than permanent starvation (same
    tradeoff as the start-order gate, #79705).
    Gate residency is arbitrary code — using it as the exclusion signal let a wedged plugin grow the
    exclusion 1:1 with wall clock, keeping the batch deadline's ``remaining`` constant so it never fired and
    the turn hung forever (#79719). A wedged plugin now contributes nothing to the exclusion and the batch
    times out normally, while a genuine approval wait (which can legitimately exceed any fixed bound) is
    still excluded in full.
    """

    def __init__(self, *, lock_timeout: float | None = None, session_key: str | None = None) -> None:
        self._serialization_lock = threading.Lock()
        self._lock_timeout = _authorization_gate_lock_timeout() if lock_timeout is None else lock_timeout
        self._session_key = session_key
        if self._session_key is None:
            # Snapshot on the SUBMITTING thread: excluded_seconds() is polled from the
            # batch wait loop, whose context may differ from the workers'.
            try:
                from tools.approval_context import get_current_session_key

                self._session_key = get_current_session_key()
            except Exception:
                logger.debug(
                    "authorization gate could not snapshot the session key; "
                    "human-wait exclusion will re-resolve it at poll time",
                    exc_info=True,
                )
        self._baseline_wait_seconds = self._human_wait_seconds()

    def _human_wait_seconds(self) -> float:
        try:
            from tools.approval_human_wait import human_wait_seconds

            return human_wait_seconds(self._session_key)
        except Exception:
            return 0.0

    def run(self, callback):
        if not self._serialization_lock.acquire(timeout=self._lock_timeout):
            # Deterministic failure (bad command, non-MCP URL, 401/403): every retry hits the same wall.
            # Park immediately instead of burning the retry ladder and spamming N identical warnings
            # (#65673). Auth failures park here too rather than returning. Returning ends the run task, and
            # with it the only listener on ``_reconnect_event`` — so a 401 on the very first connect left
            # the server unrevivable for the life of the process, even after the user re-authenticated with
            # ``hermes mcp login``. Parking keeps the task alive so the 300s self-probe (and an explicit
            # /mcp refresh) can pick up fresh tokens.
            logger.warning(
                "authorization gate lock not acquired after %.1fs "
                "(holder wedged in a pre_tool_call plugin or approval "
                "round-trip?); running prompt unserialized",
                self._lock_timeout,
            )
            return callback()
        try:
            return callback()
        finally:
            self._serialization_lock.release()

    def excluded_seconds(self) -> float:
        """Return human-approval wait seconds accrued since the batch started."""
        return max(0.0, self._human_wait_seconds() - self._baseline_wait_seconds)


@contextlib.contextmanager
def _registered_tool_worker(agent):
    """Track this worker tid for interrupt fan-out (``AIAgent.interrupt()``); on ANY exit
    (incl. BaseException) discard it and clear its interrupt bit so a recycled tid starts clean."""
    tid = threading.current_thread().ident
    with agent._tool_worker_threads_lock:
        agent._tool_worker_threads.add(tid)
    try:
        yield tid
    finally:
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.discard(tid)
        with contextlib.suppress(Exception):
            _ra()._set_interrupt(False, tid)


_NO_REASON = object()


def _interrupt_worker_tids(agent, tids, *, reason=_NO_REASON) -> None:
    """Raise the interrupt bit on each worker tid (best-effort, via ``run_agent``)."""
    kwargs = {} if reason is _NO_REASON else {"reason": reason}
    for tid in tids:
        with contextlib.suppress(Exception):
            _ra()._set_interrupt(True, tid, **kwargs)


def _set_worker_activity_callback(agent) -> None:
    """The activity callback is thread-local: bind it on THIS thread so tool-layer heartbeats fire."""
    with contextlib.suppress(Exception):
        from tools.environments.base import set_activity_callback

        set_activity_callback(agent._touch_activity)


# Must stay far below the gateway turn-inactivity timeout (default 1800s) so a silent tool never looks idle.
_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S = 30.0


def _run_tool_activity_heartbeat(
    agent,
    stop_event: threading.Event,
    label: str,
    interval: float = _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S,
) -> None:
    """Daemon thread stamping ``agent._touch_activity`` every ``interval`` seconds until
    ``stop_event`` is set, so the gateway inactivity watchdog never abandons a turn whose
    tool runs silently. Wedged tools stay bounded by the tool layer's own timeouts."""
    try:
        while not stop_event.wait(interval):
            agent._touch_activity(label)
    except Exception:
        pass  # a heartbeat must never break the agent loop


def _run_with_activity_heartbeat(agent, function_name: str, fn):
    """Run ``fn()`` under the activity heartbeat; covers both executor paths."""
    stop = threading.Event()
    thread = threading.Thread(
        # Keep the gateway turn-inactivity watchdog from abandoning a turn whose tool call runs silently for
        # longer than the inactivity timeout (#84491): stamp activity periodically while the tool is in
        # flight, not just at start/completion. Both the sequential and the concurrent paths funnel through
        # here, so a single heartbeat covers every tool.
        target=_run_tool_activity_heartbeat,
        args=(agent, stop, f"tool running: {function_name}"),
        kwargs={"interval": _TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S},
        daemon=True,
        name=f"tool-activity-hb-{function_name[:24]}",
    )
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _blocked_tool_result(agent, ref: _ToolCallRef, *, block_message: Optional[str], block_error_type: str, guardrail_decision) -> str:
    """Synthesize the result for a call blocked by scope/plugin (``block_message``) or by
    guardrail policy (``guardrail_decision``) and emit its terminal post_tool_call."""
    if block_message is not None:
        result, error_type, error_message = json.dumps({"error": block_message}, ensure_ascii=False), block_error_type, block_message
    else:
        result = agent._guardrail_block_result(guardrail_decision)
        error_type = "guardrail_block"
        error_message = getattr(guardrail_decision, "message", None) or "Tool blocked by guardrail policy"
    ref.emit_post(agent, result, status="blocked", error_type=error_type, error_message=error_message)
    return result


def _pre_tool_block(agent, ref: _ToolCallRef):
    """Run ``pre_tool_call`` plugin hooks; returns ``(block_message, final_args)`` with any
    hook-modified args applied. Hook failures never block."""
    try:
        from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

        block_msg, modified_args = _dispatch_pre_tool_call_hooks(
            ref.name,
            ref.args,
            **tool_hook_ids(agent, ref.task_id, ref.call_id),
            middleware_trace=list(ref.trace),
        )
        return block_msg, (ref.args if modified_args is None else modified_args)
    except Exception:
        return None, ref.args


def _dispatch_authorized_once(
    agent,
    state: _ManagedToolResult,
    ref: _ToolCallRef,
    *,
    execute,
    scope_block: str | None,
    display_index: int | None,
    begin_execution,
    authorization_gate: _ConcurrentToolAuthorizationGate | None,
) -> Any:
    """Hermes policy (scope → plugin pre-hooks → guardrails) then the one real dispatch.

    Plugin ``modify`` hooks may rewrite ``ref.args`` (mirrored into ``state.args``).
    ``begin_execution`` (concurrent start-order gate) is advanced exactly once on every
    path so later-ordered workers keep moving; blocked calls advance it without a callback.
    """
    def _advance_start_order(callback=None) -> None:
        if begin_execution is not None:
            begin_execution(callback)
        elif callback is not None:
            callback()

    block_message, block_error_type = scope_block, "tool_scope_block"
    if block_message is None:
        block_error_type = "plugin_block"
        resolve = lambda: _pre_tool_block(agent, ref)  # noqa: E731
        block_message, ref.args = resolve() if authorization_gate is None else authorization_gate.run(resolve)
        state.args = ref.args

    guardrail_decision = None
    if block_message is None:
        guardrail_decision = agent._tool_guardrails.before_call(ref.name, ref.args)
        if guardrail_decision.allows_execution:
            guardrail_decision = None

    if block_message is not None or guardrail_decision is not None:
        _advance_start_order()
        state.blocked = True
        return _blocked_tool_result(
            agent, ref,
            block_message=block_message, block_error_type=block_error_type, guardrail_decision=guardrail_decision,
        )

    if ref.name == "memory":
        agent._turns_since_memory = 0
    elif ref.name == "skill_manage":
        agent._iters_since_skill = 0

    _advance_start_order(lambda: _begin_tool_execution(agent, ref, display_index))
    return _run_with_activity_heartbeat(agent, ref.name, lambda: execute(ref.args))


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
    state = _ManagedToolResult(result=None, args=function_args, middleware_trace=trace, blocked=False, dispatched=False)
    dispatch_lock = threading.Lock()

    def _authorized_dispatch(final_args: dict[str, Any]) -> Any:
        with dispatch_lock:
            if state.dispatched:
                raise RuntimeError("Hermes tool execution callback invoked more than once")
            state.dispatched = True
            state.blocked = False
            state.args = final_args
        return _dispatch_authorized_once(
            agent,
            state,
            _ToolCallRef(function_name, final_args, effective_task_id, tool_call_id, trace),
            execute=execute,
            scope_block=scope_block,
            display_index=display_index,
            begin_execution=begin_execution,
            authorization_gate=authorization_gate,
        )

    def _hermes_pipeline(relay_args: dict[str, Any]) -> Any:
        request_result = apply_tool_request_middleware(
            function_name,
            relay_args,
            skip_relay=True,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )
        request_args = request_result.payload if isinstance(request_result.payload, dict) else relay_args
        trace.clear()
        trace.extend(request_result.trace)
        return run_tool_execution_middleware(
            function_name,
            request_args,
            lambda next_args: _authorized_dispatch(next_args if isinstance(next_args, dict) else request_args),
            original_args=function_args,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
        )

    state.result, _relay_args = relay_tools.execute(
        function_name,
        function_args,
        _hermes_pipeline,
        session_id=str(getattr(agent, "session_id", "") or ""),
        tool_call_id=tool_call_id or None,
        metadata={
            "task_id": effective_task_id or "",
            "turn_id": getattr(agent, "_current_turn_id", "") or "",
            "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
            "tool_call_id": tool_call_id or "",
        },
    )
    return state


# Sequential wait-loop poll cadence: /stop lands within ~1s even if the tool never polls is_interrupted().
_SEQUENTIAL_INTERRUPT_POLL_SECONDS = 1.0


def _resolve_sequential_tool_timeout() -> float | None:
    """Deadline for one sequential call: ``timeouts.tools.sequential_call``, else the
    concurrent batch deadline so the two paths can't drift; ``0``/negative disables.
    Deliberately NOT ``agent.deadline.run_bounded_sync``: both executors extend the
    deadline while an approval prompt is open, which a fixed deadline can't express."""
    from agent.deadline import resolve_timeout

    return resolve_timeout("tools.sequential_call", default=_resolve_concurrent_tool_timeout())


# Tools whose call blocks on a long-running operation that supervises its own liveness: no generic
# sequential deadline. ``delegate_task`` in a nested orchestrator blocks for the whole batch by design
# (children carry heartbeats, the stale monitor, and ``delegation.child_timeout_seconds``); under the
# 420 s deadline every real batch "timed out" while its children ran on as orphans, and the orchestrator
# spent the following hours polling transcripts (measured: 332 timeouts, ~$4k of orchestrator turns in
# one run).
# ``manage_connections`` waits on the connection operation's own deadline; the generic deadline
# would return tool_timeout while its approval card is still open.
_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset({"delegate_task", "manage_connections"})


def _abandoned_sequential_result(agent, ref: _ToolCallRef, message: str, result_cls, **outcome) -> _ManagedToolResult:
    """Emit the terminal post_tool_call for a worker the sequential runner gave up on
    (timeout / interrupt) and wrap ``message`` in its marker ``result_cls``."""
    ref.emit_post(agent, message, **outcome)
    return _ManagedToolResult(result=result_cls(message), args=ref.args, middleware_trace=ref.trace, blocked=False, dispatched=True)


def _poll_sequential_future(agent, future, function_name: str, deadline: float | None, started: float, authorization_gate) -> tuple[str, Any]:
    """Wait for the worker in interrupt-poll slices, extending the deadline by human approval
    wait; returns ``("done", result)``, ``("timeout", None)`` or ``("interrupted", None)``.
    A disabled deadline still polls: this loop is what makes a non-cooperative tool
    interruptible, so no deadline must not mean no interrupt checks."""
    _last_heartbeat = 0
    while True:
        wait_slice = _SEQUENTIAL_INTERRUPT_POLL_SECONDS
        if deadline is not None:
            remaining = deadline + authorization_gate.excluded_seconds() - time.monotonic()
            if remaining <= 0:
                return "timeout", None
            wait_slice = min(wait_slice, remaining)
        try:
            return "done", future.result(timeout=wait_slice)
        except concurrent.futures.TimeoutError:
            if agent._interrupt_requested:
                return "interrupted", None
            elapsed = int(time.monotonic() - started)
            if elapsed - _last_heartbeat >= 30:
                _last_heartbeat = elapsed
                agent._touch_activity(f"sequential tool running ({elapsed}s): {function_name}")


def _run_sequential_tool_execution_middleware(
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
) -> _ManagedToolResult:
    """Run one sequential call on a worker thread under the concurrent executor's deadline.
    Interactive tools (``clarify``) own their wait via ``agent.clarify_timeout``; the
    generic deadline would report ``tool_timeout`` while the prompt is still live."""
    timeout_s = None if function_name in _SEQUENTIAL_DEADLINE_EXEMPT_TOOLS else _resolve_sequential_tool_timeout()
    ref = _ToolCallRef(function_name, function_args, effective_task_id, tool_call_id, middleware_trace)
    kwargs = dict(ref.middleware_kwargs(), execute=execute, scope_block=scope_block, display_index=display_index)
    if function_name in _NEVER_PARALLEL_TOOLS:
        return _run_agent_tool_execution_middleware(agent, **kwargs)

    from tools.daemon_pool import DaemonThreadPoolExecutor

    authorization_gate = _ConcurrentToolAuthorizationGate()
    worker_tid: list[int] = []

    def _run() -> _ManagedToolResult:
        with _registered_tool_worker(agent) as tid:
            worker_tid.append(tid)
            return _run_agent_tool_execution_middleware(agent, authorization_gate=authorization_gate, **kwargs)

    if ref.trace is None:
        ref.trace = []
    executor = DaemonThreadPoolExecutor(max_workers=1)
    future = executor.submit(propagate_context_to_thread(_run))
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    started = time.monotonic()
    abandoned = False
    try:
        while True:
            wait_slice = _SEQUENTIAL_INTERRUPT_POLL_SECONDS
            if deadline is not None:
                remaining = (
                    deadline + authorization_gate.excluded_seconds() - time.monotonic()
                )
                if remaining <= 0:
                    timed_out = True
                    break
                wait_slice = min(wait_slice, remaining)
            try:
                return future.result(timeout=wait_slice)
            except concurrent.futures.TimeoutError:
                if agent._interrupt_requested:
                    interrupted = True
                    break
                elapsed = int(time.monotonic() - started)
                if elapsed - _last_heartbeat >= 30:
                    _last_heartbeat = elapsed
                    agent._touch_activity(
                        f"sequential tool running ({elapsed}s): {function_name}"
                    )

        if interrupted:
            # Belt-and-braces: interrupt() already fans out to tracked worker
            # tids, but the worker may have registered after the fan-out ran.
            for tid in worker_tid:
                try:
                    _ra()._set_interrupt(
                        True,
                        tid,
                        reason=getattr(agent, "_tool_interrupt_reason", None),
                    )
                except Exception:
                    pass
            # Give a cooperative tool a moment to notice its per-thread
            # interrupt bit and return a real result (mirrors the concurrent
            # path's 3s grace).
            concurrent.futures.wait([future], timeout=3.0)
            if future.done() and not future.cancelled():
                return future.result()
            timed_out = True  # reuse the abandon-shutdown path in finally
            future.cancel()
            interrupt_reason = (
                getattr(agent, "_tool_interrupt_reason", None)
                or "interrupt requested"
            )
            message = (
                f"[Tool execution cancelled — {function_name} was abandoned: "
                f"{interrupt_reason}]"
            )
            logger.info(
                "sequential tool %s abandoned due to %s (%.1fs elapsed)",
                function_name, interrupt_reason, time.monotonic() - started,
            )
            trace = middleware_trace if middleware_trace is not None else []
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=message,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int((time.monotonic() - started) * 1000),
                status="cancelled",
                error_type="tool_interrupted",
                error_message=f"Tool execution cancelled: {interrupt_reason}",
                middleware_trace=list(trace),
            )
        else:
            assert timeout_s is not None  # only reachable when a deadline exists
            message = f"Error executing tool '{function_name}': timed out after {timeout_s:.1f}s"
            logger.warning("sequential tool %s timed out after %.1fs", function_name, timeout_s)
            result_cls, outcome = _ToolTimeoutResult, dict(
                duration_ms=int(timeout_s * 1000), status="timeout", error_type="tool_timeout", error_message=message,
            )
        abandoned = True
        future.cancel()
        if state == "timeout":
            _interrupt_worker_tids(agent, worker_tid)
        return _abandoned_sequential_result(agent, ref, message, result_cls, **outcome)
    finally:
        # Never join a wedged worker (daemon pool also keeps it out of the atexit join).
        executor.shutdown(wait=not abandoned, cancel_futures=abandoned)


def _safe_callback(callback, label: str, *args, **kwargs) -> None:
    """Invoke a UI/bridge callback if set; a failing callback is logged, never fatal."""
    if not callback:
        return
    try:
        callback(*args, **kwargs)
    except Exception as callback_error:
        logging.debug("%s callback error: %s", label, callback_error)


def _begin_tool_execution(agent, ref: _ToolCallRef, display_index: int | None) -> None:
    """Run user-visible and checkpoint preflight on final tool arguments."""
    function_name, function_args, effective_task_id, tool_call_id = ref.name, ref.args, ref.task_id, ref.call_id
    display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
    if _tool_progress_enabled(agent):
        prefix = f"Tool {display_index}" if display_index is not None else "Tool"
        if agent.verbose_logging:
            print(f"  📞 {prefix}: {function_name}({list(display_args.keys())})")
            print(agent._wrap_verbose("Args: ", json.dumps(display_args, indent=2, ensure_ascii=False)))
        else:
            print(f"  📞 {prefix}: {function_name}({list(function_args.keys())}) - {_preview(json.dumps(display_args, ensure_ascii=False), agent.log_prefix_chars)}")

    agent._current_tool = function_name
    agent._touch_activity(f"executing tool: {function_name}")
    _set_worker_activity_callback(agent)

    if agent.tool_progress_callback:
        try:
            preview = _build_tool_preview(function_name, display_args)
        except Exception as callback_error:
            logging.debug("Tool progress callback error: %s", callback_error)
        else:
            _safe_callback(agent.tool_progress_callback, "Tool progress", "tool.started", function_name, preview, display_args)
    _safe_callback(agent.tool_start_callback, "Tool start", tool_call_id, function_name, display_args)

    if not agent._checkpoint_mgr.enabled:
        return
    with contextlib.suppress(Exception):
        if function_name in {"write_file", "patch"}:
            _ensure_file_checkpoint(agent, function_name, function_args, effective_task_id)
        elif function_name == "terminal":
            command = function_args.get("command", "")
            if _is_destructive_command(command):
                from agent.runtime_cwd import scope_terminal_cwd
                cwd = function_args.get("workdir") or scope_terminal_cwd() or os.getcwd()
                agent._checkpoint_mgr.ensure_checkpoint(cwd, f"before terminal: {command[:60]}")


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
            cancelled_result = (
                f"[Tool execution cancelled — {tc.function.name} was skipped "
                "due to user interrupt]"
            )
            tool_call_id = _pairing_tool_call_id(tc)
            messages.append(make_tool_result_message(
                tc.function.name,
                cancelled_result,
                tool_call_id,
                effect_disposition="none",
            ))
            _emit_terminal_post_tool_call(
                agent,
                function_name=tc.function.name,
                function_args={},
                result=cancelled_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="cancelled",
                error_type="user_interrupt",
                error_message="Tool execution skipped due to user interrupt",
            )
            _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"cancelled tool result {tc.function.name}",
            )
        return

    # ── Parse args + pre-execution bookkeeping ───────────────────────
    # (tool call, resolved name, parsed args, middleware trace, parse error,
    # tool-search scope block)
    parsed_calls = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name

        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )

        if malformed_args_result is not None:
            parsed_calls.append(
                (
                    tool_call,
                    function_name,
                    function_args,
                    [],
                    malformed_args_result,
                    None,
                )
            )
            continue

        # ── Tool Search unwrap ────────────────────────────────────────
        # When the model invokes the tool_call bridge, peel it open so
        # every downstream check (checkpointing, guardrails, plugin
        # pre-tool-call hooks, the display/activity feed, the post-call
        # callback) sees the underlying tool — not the bridge. This is
        # the OpenClaw lesson: hooks must observe the real tool name.
        #
        # The original tool_call entry on ``tool_call.function`` is left
        # untouched so the conversation transcript and the matching
        # tool_call_id are preserved exactly as the model emitted them.
        #
        # Scope gate: the unwrap dispatches the underlying tool directly
        # (bypassing the bridge branch in handle_function_call and its
        # scope check), so we enforce session toolset scope HERE. A tool
        # the session was not granted is rejected before any checkpoint,
        # hook, or dispatch fires.
        _ts_scope_block = None
        try:
            display_args = _redact_tool_args_for_display(ref.name, ref.args) or ref.args
        except Exception as cb_err:
            logging.debug("Tool complete callback error: %s", cb_err)
        else:
            _safe_callback(agent.tool_complete_callback, "Tool complete", ref.call_id, ref.name, display_args, result)
    if risk_metadata is not None and risk_metadata.get("risk") != "low":
        _safe_callback(
            agent.tool_progress_callback, "Tool output risk",
            "tool.output_risk", ref.name, None, None, tool_call_id=ref.call_id, risk_metadata=risk_metadata,
        )


def _commit_tool_result(
    agent,
    messages: list,
    ref: _ToolCallRef,
    function_result,
    *,
    budget: BudgetConfig,
    tool_duration: float,
    is_error: bool,
    blocked: bool,
    effect_disposition,
    observed: bool = False,
    error_preview: Callable[[Any], Any] = lambda result: result,
    success_log_chars: Optional[int] = None,
    verbose_text: Callable[[Any], Any] = lambda result: result,
):
    """Observe (``observed`` results only) and log the outcome; mark the tool done; persist/
    spill, hint, wrap and append the result; flush the session DB; project ``tool.completed``.

    Blocked calls never ran, so they are neither guardrail-observed nor fed to the file-
    mutation verifier; ``success_log_chars`` (sequential path) also logs the completion line.
    Returns ``(persisted_result, display_result, risk_metadata)`` (``display_result`` =
    pre-persist content for UI previews) or ``None`` when the flush failed (stop the batch).
    """
    function_name, function_args, tool_call_id, effective_task_id = ref.name, ref.args, ref.call_id, ref.task_id
    if observed:
        if not blocked:
            function_result = agent._append_guardrail_observation(
                function_name, function_args, function_result, failed=is_error, tool_call_id=tool_call_id,
            )
        if is_error:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, error_preview(function_result))
        elif success_log_chars is not None:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, success_log_chars)
        if not blocked:
            try:
                agent._record_file_mutation_result(function_name, function_args, function_result, is_error)
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)
        if agent.verbose_logging:
            logging.debug("Tool %s completed in %.2fs", function_name, tool_duration)
            _log_result = verbose_text(function_result)
            logging.debug("Tool result (%d chars): %s", len(_log_result), _log_result)

    agent._current_tool = None
    _status_suffix = " (error)" if is_error else ""
    agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s){_status_suffix}")

    persisted_result = function_result
    if not _is_multimodal_tool_result(persisted_result):
        persisted_result = maybe_persist_tool_result(
            content=persisted_result,
            tool_name=function_name,
            tool_use_id=tool_call_id,
            env=get_active_env(effective_task_id),
            config=budget,
        )
    _record_persisted_path_for_stub(agent, tool_call_id, persisted_result)

    subdir_hints = agent._subdirectory_hints.check_tool_call(function_name, function_args)
    if subdir_hints:
        if _is_multimodal_tool_result(persisted_result):
            # Hint goes on the text summary part so the model still sees it; image blocks untouched.
            _append_subdir_hint_to_multimodal(persisted_result, subdir_hints)
        else:
            persisted_result += subdir_hints

    # Multimodal dicts become an OpenAI-style content list; text-only servers get a
    # string-safe fallback so a rejected image result never poisons history.
    _tool_content = agent._tool_result_content_for_active_model(function_name, persisted_result)
    tool_message = make_tool_result_message(function_name, _tool_content, tool_call_id, effect_disposition=effect_disposition)
    messages.append(tool_message)
    if not _flush_session_db_after_tool_progress(agent, messages, stage=f"tool result {function_name}"):
        return None

    if not blocked:
        # ``tool.completed`` projects AFTER the canonical append + flush so resume can
        # reconstruct the result even if the UI bridge dies mid-projection.
        _safe_callback(
            agent.tool_progress_callback, "Tool progress",
            "tool.completed", function_name, None, None, duration=tool_duration, is_error=is_error, result=function_result,
        )
    return persisted_result, function_result, tool_message.get("_tool_output_risk")


def _finalize_tool_batch(agent, messages: list, effective_task_id: str, num_tools: int, budget: BudgetConfig) -> None:
    """Per-turn aggregate budget enforcement, then /steer injection — in that order, so the
    steer marker is never truncated/discarded when enforcement replaces a result."""
    if num_tools <= 0:
        return
    enforce_turn_budget(messages[-num_tools:], env=get_active_env(effective_task_id), config=budget)
    agent._apply_pending_steer_to_tool_results(messages, num_tools)


def _tool_progress_enabled(agent) -> bool:
    return not agent.quiet_mode and getattr(agent, "tool_progress_mode", "all") != "off"


def _preview(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _print_tool_completed(agent, index: int, tool_duration: float, result) -> None:
    """Non-quiet ``✅ Tool N completed`` line (full result under verbose logging)."""
    if agent.verbose_logging:
        print(f"  ✅ Tool {index} completed in {tool_duration:.2f}s")
        print(agent._wrap_verbose("Result: ", result))
    else:
        print(f"  ✅ Tool {index} completed in {tool_duration:.2f}s - {_preview(result if isinstance(result, str) else str(result), agent.log_prefix_chars)}")


# ── Concurrent batch machinery ──────────────────────────────────────────────


@dataclass
class _ToolOutcome:
    """One finished worker slot of a concurrent batch (``ref`` holds the final name/args/trace)."""

    ref: _ToolCallRef
    result: Any
    duration: float
    is_error: bool
    blocked: bool


def _start_order_gate_timeout(batch_timeout: float | None) -> float:
    """The gate bound must sit UNDER the batch deadline, else parked workers are falsely
    reported timed out without starting. A disabled deadline keeps the stock bound."""
    if batch_timeout is None:
        return _START_ORDER_GATE_TIMEOUT_S
    return min(_START_ORDER_GATE_TIMEOUT_S, batch_timeout / 2)


class _StartOrderGate:
    """Serialize worker dispatch by submit order (prompts appear in call order); ``abandon()``
    releases every parked worker so none dispatches a tool the turn already gave up on."""

    def __init__(self, timeout: float) -> None:
        self._condition = threading.Condition()
        self._next_order = 0
        self._timeout = timeout
        self.abandoned = threading.Event()

    def abandon(self) -> None:
        self.abandoned.set()
        with self._condition:
            self._condition.notify_all()

    def begin_in_order(self, order: int, callback=None, *, tool_name: str = "") -> bool:
        """Wait for ``order``, run ``callback``, advance. Returns False if abandoned."""
        with self._condition:
            # Bounded wait so one wedged dispatch can't starve later-ordered workers; on
            # expiry proceed out of order (interleaved prompts beat starvation). ``>=`` (not
            # ``==``) releases every skipped worker at once; abandoned short-circuits.
            in_order = self._condition.wait_for(
                lambda: self._next_order >= order or self.abandoned.is_set(), timeout=self._timeout,
            )
            if self.abandoned.is_set():
                return False  # the turn already synthesized this result; don't advance
            if not in_order:
                logger.warning(
                    "start-order gate timed out for %s (order=%d next=%d); proceeding out of order",
                    tool_name or "tool", order, self._next_order,
                )
            try:
                if callback is not None:
                    callback()
            finally:
                self._next_order = max(self._next_order, order + 1)
                self._condition.notify_all()
        return True


class _WorkerStartOnce:
    """One worker's handle on the start-order gate: advances at most once, raising
    ``_BatchAbandoned`` (instead of dispatching late) when the batch was abandoned."""

    def __init__(self, gate: _StartOrderGate, order: int, tool_name: str) -> None:
        self._gate, self._order, self._tool_name, self._advanced = gate, order, tool_name, False

    def advance(self, callback=None) -> None:
        if self._advanced:
            return
        self._advanced = True
        if not self._gate.begin_in_order(self._order, callback, tool_name=self._tool_name):
            raise _BatchAbandoned(self._tool_name)


class _ConcurrentBatch:
    """Shared state of one concurrent tool batch: per-slot results, the start-order and
    authorization gates, and the deadline bookkeeping the wait loop needs."""

    def __init__(self, agent, messages: list, effective_task_id: str, parsed_calls: list[_ParsedCall], timeout_s: float | None) -> None:
        self.agent = agent
        self.messages = messages
        self.effective_task_id = effective_task_id
        self.parsed_calls = parsed_calls
        self.timeout_s = timeout_s
        self.results: list[Optional[_ToolOutcome]] = [None] * len(parsed_calls)
        for i, pc in enumerate(parsed_calls):
            if pc.parse_error is not None:
                self.results[i] = _ToolOutcome(pc.ref(effective_task_id), pc.parse_error, 0.0, True, True)
        self.gate = _StartOrderGate(_start_order_gate_timeout(timeout_s))
        self.authorization_gate = _ConcurrentToolAuthorizationGate()
        self.timed_out_indices: set[int] = set()

    def _dispatch_worker(self, index: int, ref: _ToolCallRef, scope_block, start_gate: _WorkerStartOnce) -> Optional[_ToolOutcome]:
        """Run one call through the middleware and synthesize its slot outcome; ``None`` when
        abandoned at the gate (the main thread already wrote this slot; emitting would
        double-report the tool_call_id)."""
        agent = self.agent
        # Approval/sudo callbacks (thread-local) and the agent turn's ContextVars are propagated by
        # propagate_context_to_thread() at the submit site below (GHSA-qg5c-hvr5-hjgr, #13617).
        start = time.time()
        blocked = dispatched = False
        try:
            managed = _run_agent_tool_execution_middleware(
                agent,
                **ref.middleware_kwargs(),
                execute=lambda next_args: agent._invoke_tool(
                    ref.name, next_args, ref.task_id, ref.call_id,
                    messages=self.messages,
                    pre_tool_block_checked=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                    tool_request_middleware_trace=list(ref.trace),
                ),
                scope_block=scope_block,
                display_index=index + 1,
                begin_execution=start_gate.advance,
                authorization_gate=self.authorization_gate,
            )
            result, ref.args, ref.trace = managed.result, managed.args, managed.middleware_trace
            blocked, dispatched = managed.blocked, managed.dispatched
        except _BatchAbandoned:
            logger.info("tool %s abandoned at start-order gate; skipping dispatch", ref.name)
            return None
        except KeyboardInterrupt:
            with contextlib.suppress(Exception):
                agent.interrupt("keyboard interrupt")
            result = ref.emit_cancelled(agent, start)
            duration = time.time() - start
            logger.info("tool %s cancelled (%.2fs)", ref.name, duration)
            return _ToolOutcome(ref, result, duration, True, False)
        except Exception as tool_error:
            result = f"Error executing tool '{ref.name}': {tool_error}"
            logger.error("_invoke_tool raised for %s: %s", ref.name, tool_error, exc_info=True)
        duration = time.time() - start
        if not blocked and not dispatched:
            ref.emit_post(agent, result, duration_ms=int(duration * 1000))
        is_error, _ = _detect_tool_failure(ref.name, result)
        if is_error:
            logger.info("tool %s failed (%.2fs): %s", ref.name, duration, result[:200])
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", ref.name, duration, len(result))
        return _ToolOutcome(ref, result, duration, is_error, blocked)

    def run_worker(self, index: int, start_order: int) -> None:
        """Worker function executed in a thread."""
        agent, pc = self.agent, self.parsed_calls[index]
        with _registered_tool_worker(agent) as _worker_tid:
            # An interrupt may have fanned out before our registration; apply it to our tid.
            if agent._interrupt_requested:
                _interrupt_worker_tids(agent, [_worker_tid], reason=getattr(agent, "_tool_interrupt_reason", None))
            _set_worker_activity_callback(agent)
            start_gate = _WorkerStartOnce(self.gate, start_order, pc.name)
            try:
                outcome = self._dispatch_worker(index, pc.ref(self.effective_task_id), pc.scope_block, start_gate)
                if outcome is not None:
                    self.results[index] = outcome
            finally:
                with contextlib.suppress(_BatchAbandoned):
                    start_gate.advance()  # keep later-ordered workers moving

    def submit_all(self, executor, runnable: list[int]) -> tuple[list, dict]:
        """Submit every runnable slot; on interpreter shutdown, synthesize error results
        for the unsubmitted remainder instead of raising. ``propagate_context_to_thread``
        carries turn ContextVars and thread-local approval/sudo callbacks into the worker."""
        futures = []
        future_to_index = {}
        for submit_index, i in enumerate(runnable):
            try:
                f = executor.submit(propagate_context_to_thread(self.run_worker), i, submit_index)
            except RuntimeError as submit_error:
                if not _is_interpreter_shutdown_submit_error(submit_error):
                    raise
                skipped = runnable[submit_index:]
                logger.warning(
                    "interpreter shutdown while scheduling concurrent tools; skipping %d unsubmitted tool(s)", len(skipped),
                )
                for skipped_i in skipped:
                    ref = self.parsed_calls[skipped_i].ref(self.effective_task_id)
                    if self.results[skipped_i] is None:
                        result = f"Error executing tool '{ref.name}': Python interpreter is shutting down; tool was not started"
                        self.results[skipped_i] = _ToolOutcome(ref, result, 0.0, True, False)
                break
            futures.append(f)
            future_to_index[f] = i
        return futures, future_to_index

    def _running_names(self, not_done, future_to_index) -> list[str]:
        return [self.parsed_calls[future_to_index[f]].name for f in not_done if f in future_to_index]

    def await_completion(self, futures, future_to_index, deadline: float | None) -> bool:
        """Wait with periodic heartbeats and interrupt checks; True when the batch was
        abandoned (deadline or interrupt) and the executor must not join its workers."""
        agent = self.agent
        _conc_start = time.time()
        while True:
            wait_timeout = 5.0
            if deadline is not None:
                remaining = deadline + self.authorization_gate.excluded_seconds() - time.monotonic()
                if remaining <= 0:
                    not_done = {f for f in futures if not f.done()}
                else:
                    wait_timeout = min(wait_timeout, remaining)
            if deadline is None or remaining > 0:
                _done, not_done = concurrent.futures.wait(futures, timeout=wait_timeout)
            if not not_done:
                return False

            timed_out = deadline is not None and time.monotonic() >= deadline + self.authorization_gate.excluded_seconds()
            if timed_out:
                self.timed_out_indices = {future_to_index[f] for f in not_done if f in future_to_index}
                logger.warning(
                    "concurrent tool batch timed out after %.1fs; %d tool(s) still running: %s",
                    self.timeout_s,
                    len(self.timed_out_indices),
                    ", ".join(self._running_names(not_done, future_to_index)[:5]),
                )
            elif agent._interrupt_requested:
                # Tools without interrupt checks (web_search, read_file) run to
                # completion; cancel unstarted futures so we don't block on them.
                agent._vprint(
                    f"{agent.log_prefix}⚡ Interrupt: cancelling {len(not_done)} pending concurrent tool(s)",
                    force=True,
                )
            else:
                _conc_elapsed = int(time.time() - _conc_start)
                # Heartbeat every ~30s (6 × 5s poll intervals)
                if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                    _still_running = self._running_names(not_done, future_to_index)
                    agent._touch_activity(
                        f"concurrent tools running ({_conc_elapsed}s, "
                        f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                    )
                continue
            for f in not_done:
                f.cancel()
            # Release gate-parked workers BEFORE interrupt fan-out so none later
            # dispatches a tool the turn already reported as timed out / interrupted.
            self.gate.abandon()
            if timed_out:
                with agent._tool_worker_threads_lock:
                    worker_tids = list(agent._tool_worker_threads)
                _interrupt_worker_tids(agent, worker_tids)
            else:
                # Give running tools a moment to notice the per-thread interrupt and exit gracefully.
                concurrent.futures.wait(not_done, timeout=3.0)
            return True

    def run(self) -> None:
        """Dispatch the runnable calls on a daemon pool and wait for the batch."""
        runnable = [i for i, pc in enumerate(self.parsed_calls) if pc.parse_error is None]
        if not runnable:
            return
        deadline = time.monotonic() + self.timeout_s if self.timeout_s is not None else None
        max_workers = _max_workers_for_tool_batch([(i, None, self.parsed_calls[i].name) for i in runnable])
        # Daemon workers: the stdlib pool's atexit join would let one wedged tool block exit.
        from tools.daemon_pool import DaemonThreadPoolExecutor
        executor = DaemonThreadPoolExecutor(max_workers=max_workers)
        abandon_executor = False
        try:
            futures, future_to_index = self.submit_all(executor, runnable)
            abandon_executor = self.await_completion(futures, future_to_index, deadline)
        finally:
            # Every abandoning exit releases gate-parked workers and leaves wedged threads
            # detached rather than joining them; normal completion joins.
            if abandon_executor:
                self.gate.abandon()
            executor.shutdown(wait=not abandon_executor, cancel_futures=abandon_executor)


def _unfinished_tool_result(agent, ref: _ToolCallRef, *, timed_out: bool, timeout_s: float | None) -> tuple[str, float, Optional[str]]:
    """Synthesize the result for a slot no worker filled (deadline, interrupt, or a thread
    that never returned), emit its terminal post_tool_call, and return
    ``(function_result, tool_duration, effect_disposition)``."""
    if timed_out:
        suffix = f"{timeout_s:.1f}s" if timeout_s is not None else "the configured timeout"
        function_result = f"Error executing tool '{ref.name}': timed out after {suffix}"
        outcome = dict(duration_ms=int((timeout_s or 0.0) * 1000), status="timeout", error_type="tool_timeout", error_message=function_result)
        tool_duration, effect_disposition = float(timeout_s or 0.0), "unknown"
    elif agent._interrupt_requested:
        function_result = f"[Tool execution cancelled — {ref.name} was skipped due to user interrupt]"
        outcome = dict(status="cancelled", error_type="keyboard_interrupt", error_message="Tool execution cancelled by user interrupt")
        tool_duration, effect_disposition = 0.0, None
    else:
        function_result = f"Error executing tool '{ref.name}': thread did not return a result"
        outcome = dict(status="error", error_type="thread_missing_result", error_message=function_result)
        tool_duration, effect_disposition = 0.0, None
    ref.emit_post(agent, function_result, **outcome)
    return function_result, tool_duration, effect_disposition


def _append_batch_results(agent, messages: list, effective_task_id: str, batch: _ConcurrentBatch, budget: BudgetConfig) -> bool:
    """Append every slot's result in original call order; returns False at the first
    failed flush (the caller must stop the batch)."""
    for i, pc in enumerate(batch.parsed_calls):
        r = batch.results[i]
        # A worker may finish between the deadline snapshot and this loop;
        # prefer its real result over a fabricated timeout.
        if r is None:
            ref, is_error, blocked = pc.ref(effective_task_id), True, False
            function_result, tool_duration, effect_disposition = _unfinished_tool_result(
                agent, ref, timed_out=i in batch.timed_out_indices, timeout_s=batch.timeout_s,
            )
        else:
            ref, function_result, tool_duration, is_error, blocked = r.ref, r.result, r.duration, r.is_error, r.blocked
            effect_disposition = "none" if blocked else None
            if pc.parse_error is not None:
                ref.emit_invalid_arguments(agent, r.result)
        committed = _commit_tool_result(
            agent, messages, ref, function_result,
            budget=budget, tool_duration=tool_duration, is_error=is_error, blocked=blocked,
            effect_disposition=effect_disposition, observed=r is not None,
            error_preview=lambda res: _multimodal_text_summary(res)[:200],
        )
        if committed is None:
            return False
        _persisted, display_function_result, risk_metadata = committed

        if agent._should_emit_quiet_tool_messages():
            cute_msg = _get_cute_tool_message_impl(ref.name, ref.args, tool_duration, result=display_function_result)
            agent._safe_print(f"  {cute_msg}")
        elif _tool_progress_enabled(agent):
            _print_tool_completed(agent, i + 1, tool_duration, _multimodal_text_summary(display_function_result))

        _emit_tool_complete_and_risk(agent, ref, display_function_result, risk_metadata, blocked)
    return True


def execute_tool_calls_concurrent(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls concurrently; results are appended in original call order.
    ``finalize=False`` skips end-of-batch budget enforcement and /steer injection (the
    segmented dispatcher owns turn-end work)."""
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)
    _tool_budget = _budget_for_agent(agent)  # once per turn, not per result

    if agent._interrupt_requested:
        print(f"{agent.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
        _append_skipped_tool_results(
            agent, messages, tool_calls, effective_task_id,
            content="[Tool execution cancelled — {name} was skipped due to user interrupt]",
            hook_error_type="user_interrupt",
            flush_stage="cancelled tool result",
            stop_on_flush_failure=False,
        )
        return

    parsed_calls = [_parse_tool_call(agent, tc) for tc in tool_calls]

    tool_names_str = ", ".join(pc.name for pc in parsed_calls)
    if _tool_progress_enabled(agent):
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")

    # Resolved before the batch is built so the start-order gate can clamp under the deadline.
    timeout_s = _resolve_concurrent_tool_timeout()
    batch = _ConcurrentBatch(agent, messages, effective_task_id, parsed_calls, timeout_s)
    agent._current_tool = tool_names_str
    agent._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

    def _run_tool(
        index,
        tool_call,
        function_name,
        function_args,
        middleware_trace,
        scope_block,
        start_order,
    ):
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
                _ra()._set_interrupt(
                    True,
                    _worker_tid,
                    reason=getattr(agent, "_tool_interrupt_reason", None),
                )
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
        start = time.time()
        tool_call_id = _pairing_tool_call_id(tool_call)
        blocked = False
        dispatched = False
        start_advanced = False

        def _advance_start(callback=None) -> None:
            nonlocal start_advanced
            if start_advanced:
                return
            try:
                proceed = _begin_in_order(
                    start_order,
                    callback,
                    tool_name=function_name,
                    gate_timeout=gate_timeout_s,
                )
            finally:
                start_advanced = True
            if not proceed:
                # Batch already abandoned: the turn synthesized this tool's
                # result and moved on. Abort instead of dispatching late.
                raise _BatchAbandoned(function_name)

        try:
            try:
                def _execute(next_args: dict[str, Any]) -> Any:
                    return agent._invoke_tool(
                        function_name,
                        next_args,
                        effective_task_id,
                        tool_call_id,
                        messages=messages,
                        pre_tool_block_checked=True,
                        skip_tool_request_middleware=True,
                        skip_tool_execution_middleware=True,
                        tool_request_middleware_trace=list(middleware_trace),
                    )

                managed = _run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=scope_block,
                    display_index=index + 1,
                    middleware_trace=middleware_trace,
                    begin_execution=_advance_start,
                    authorization_gate=authorization_gate,
                )
                result = managed.result
                function_args = managed.args
                middleware_trace = managed.middleware_trace
                blocked = managed.blocked
                dispatched = managed.dispatched
            except _BatchAbandoned:
                # The batch was abandoned while we were parked at the start-order
                # gate. The main thread already synthesized this tool's result
                # (timeout/cancelled) and moved on, so write nothing: a late
                # results[index] write, post_tool_call emit, or progress print
                # would double-report a tool_call_id the turn already closed.
                logger.info(
                    "tool %s abandoned at start-order gate; skipping dispatch",
                    function_name,
                )
                return
            except KeyboardInterrupt:
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                result = _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=start,
                    middleware_trace=list(middleware_trace),
                )
                duration = time.time() - start
                logger.info("tool %s cancelled (%.2fs)", function_name, duration)
                results[index] = (
                    function_name,
                    function_args,
                    result,
                    duration,
                    True,
                    False,
                    middleware_trace,
                )
                return
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            if not blocked and not dispatched:
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    result=result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    duration_ms=int(duration * 1000),
                    middleware_trace=list(middleware_trace),
                )
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (
                function_name,
                function_args,
                result,
                duration,
                is_error,
                blocked,
                middleware_trace,
            )
        finally:
            # Teardown advance: keep the counter moving for any later-ordered
            # worker. Never let the abandonment signal escape from here — the
            # worker is already unwinding and the turn owns the result.
            try:
                _advance_start()
            except _BatchAbandoned:
                pass
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
        batch.run()
    finally:
        if spinner:
            finished = [r for r in batch.results if r is not None]
            spinner.stop(f"⚡ {len(finished)}/{num_tools} tools completed in {sum(r.duration for r in finished):.1f}s total")

    # ── Post-execution: display per-tool results ─────────────────────
    for i, (tc, name, args, middleware_trace, _parse_error, _scope_block) in enumerate(
        parsed_calls
    ):
        r = results[i]
        tool_call_id = _pairing_tool_call_id(tc)
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
                tool_call_id=tool_call_id,
                duration_ms=int((timeout_s or 0.0) * 1000),
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
                    tool_call_id=tool_call_id,
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
                    tool_call_id=tool_call_id,
                    status="error",
                    error_type="thread_missing_result",
                    error_message=function_result,
                    middleware_trace=list(middleware_trace),
                )
            tool_duration = 0.0
        else:
            function_name, function_args, function_result, tool_duration, is_error, blocked, middleware_trace = r
            name = function_name
            args = function_args
            progress_function_name = function_name
            if _parse_error is not None:
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    result=function_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    status="error",
                    error_type="invalid_tool_arguments",
                    error_message="Tool arguments must be a valid JSON object",
                    middleware_trace=list(middleware_trace),
                )
            if blocked:
                effect_disposition = "none"

            if not blocked:
                function_result = agent._append_guardrail_observation(
                    function_name,
                    function_args,
                    function_result,
                    failed=is_error,
                    tool_call_id=tool_call_id,
                )

# ── Sequential dispatch ─────────────────────────────────────────────────────


def _start_quiet_tool_spinner(agent, function_name: str, function_args: dict, *, gate: bool = True, label: Optional[str] = None):
    """Start the quiet-mode kawaii spinner for one tool call, or return None; ``gate=False``
    skips ``_should_start_quiet_spinner`` (context-engine tools always spin)."""
    if not agent._should_emit_quiet_tool_messages() or (gate and not agent._should_start_quiet_spinner()):
        return None
    face = random.choice(KawaiiSpinner.get_waiting_faces())
    if label is None:
        display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
        label = f"{_get_tool_emoji(function_name)} {_build_tool_label(function_name, display_args) or function_name}"
    spinner = KawaiiSpinner(f"{face} {label}", spinner_type='dots', print_fn=agent._print_fn)
    spinner.start()
    return spinner


        display_function_result = function_result
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=name,
            tool_use_id=tool_call_id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(function_result) else function_result
        _record_persisted_path_for_stub(agent, tool_call_id, function_result)


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
            tool_call_id,
            effect_disposition=effect_disposition,
        )

    # Registry tools: post hook is owned by this executor (inner observer suppressed).
    def _execute(next_args: dict) -> Any:
        import model_tools

        with model_tools.suppress_post_tool_call_hook():
            return model_tools.handle_function_call(
                function_name,
                next_args,
                effective_task_id,
                tool_call_id=tool_call_id,
                session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
                tool_request_middleware_trace=list(middleware_trace),
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            )

        if not blocked and agent.tool_complete_callback:
            try:
                display_args = _redact_tool_args_for_display(name, args) or args
                agent.tool_complete_callback(
                    tool_call_id, name, display_args, display_function_result,
                )
            except Exception as cb_err:
                logging.debug("Tool complete callback error: %s", cb_err)

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
                    tool_call_id=tool_call_id,
                    risk_metadata=risk_metadata,
                )
            except Exception as cb_err:
                logging.debug("Tool output risk callback error: %s", cb_err)

    # ── Per-turn aggregate budget enforcement ─────────────────────────
    # Keep /steer pending until the final post-budget drain below.  The model
    # cannot observe a partial batch, while an early drain can be discarded
    # when aggregate budget enforcement replaces that tool result.
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


def _skip_remaining_sequential(agent, messages: list, remaining, effective_task_id: str, *, notice: str, **skip_kwargs) -> bool:
    """Announce an interrupt and append one skipped result per unstarted call; False when
    a flush failed (the caller must stop the batch)."""
    agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining)} {notice}", force=True)
    return _append_skipped_tool_results(agent, messages, remaining, effective_task_id, **skip_kwargs)


    Used when a hard interrupt (KeyboardInterrupt / BaseException) aborts the
    sequential executor mid-batch. Without this, the loop re-raises leaving the
    assistant tool-call turn with no matching tool results — a message-role
    alternation violation that malforms the next provider request. Mirrors the
    cooperative-interrupt skip block and the concurrent path, both of which
    already emit a result for every call_id.
    """
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", "") or "tool"
        messages.append(make_tool_result_message(
            name,
            f"[Tool execution cancelled — {name} was skipped due to {reason}]",
            _pairing_tool_call_id(tc),
            effect_disposition="none",
        ))


def execute_tool_calls_sequential(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, *, finalize: bool = True) -> None:
    """Execute tool calls sequentially (single calls or interactive tools). ``finalize=False``
    skips end-of-batch budget enforcement and /steer injection (the segmented dispatcher
    owns turn-end work)."""
    _tool_budget = _budget_for_agent(agent)  # once per turn, not per result
    tool_calls = assistant_message.tool_calls

    ``finalize=False`` skips the end-of-batch aggregate budget enforcement
    and /steer injection — used when this call is one segment of a larger
    mixed batch and the segmented dispatcher owns the turn-end work.
    """
    # Resolve the context-scaled tool-output budget once per turn.
    _tool_budget = _budget_for_agent(agent)

    # Keep every runtime-tool branch on one bounded execution funnel without
    # duplicating timeout policy across the branch-specific callbacks below.
    def _run_agent_tool_execution_middleware(agent, **kwargs):
        return _run_sequential_tool_execution_middleware(agent, **kwargs)

    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
        tool_call_id = _pairing_tool_call_id(tool_call)
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        # Check interrupt BEFORE each tool so a "stop" during the previous one skips the rest.
        if agent._interrupt_requested:
            remaining_calls = assistant_message.tool_calls[i-1:]
            if remaining_calls:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
            for skipped_tc in remaining_calls:
                skipped_name = skipped_tc.function.name
                cancelled_result = (
                    f"[Tool execution cancelled — {skipped_name} was skipped "
                    "due to user interrupt]"
                )
                messages.append(make_tool_result_message(
                    skipped_name,
                    cancelled_result,
                    _pairing_tool_call_id(skipped_tc),
                    effect_disposition="none",
                ))
                _emit_terminal_post_tool_call(
                    agent,
                    function_name=skipped_name,
                    function_args={},
                    result=cancelled_result,
                    effective_task_id=effective_task_id,
                    tool_call_id=getattr(skipped_tc, "id", "") or "",
                    status="cancelled",
                    error_type="user_interrupt",
                    error_message="Tool execution skipped due to user interrupt",
                )
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"cancelled tool result {skipped_name}",
                ):
                    return
            break

        function_name = tool_call.function.name

        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )
        if malformed_args_result is not None:
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=malformed_args_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                status="error",
                error_type="invalid_tool_arguments",
                error_message="Tool arguments must be a valid JSON object",
            )
            messages.append(
                make_tool_result_message(
                    function_name,
                    malformed_args_result,
                    tool_call_id,
                )
            )
            if not _flush_session_db_after_tool_progress(
                agent,
                messages,
                stage=f"invalid tool arguments {function_name}",
            ):
                return
            continue

        tool_start_time = time.time()

        if function_name == "todo":
            def _execute(next_args: dict) -> Any:
                from tools.todo_tool import todo_tool as _todo_tool
                return _todo_tool(
                    todos=next_args.get("todos"),
                    merge=next_args.get("merge", False),
                    store=agent._todo_store,
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
        elif function_name == "message_agent":
            # Bot Mode teammate DM (tools/bot_mode_dm.py) — injected, not
            # registered: only a canonical Bot Chat session carries the
            # schema, and the tool re-gates on the session title itself.
            def _execute(next_args: dict) -> Any:
                from tools.bot_mode_dm import message_agent_tool as _message_agent_tool
                return _message_agent_tool(
                    target=next_args.get("target", ""),
                    message=next_args.get("message", ""),
                    task_id=effective_task_id,
                    agent=agent,
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('message_agent', function_args, tool_duration, result=function_result)}")
        elif function_name == "session_search":
            def _execute(next_args: dict) -> Any:
                session_db = agent._get_session_db_for_recall()
                if not session_db:
                    from hermes_state import format_session_db_unavailable
                    return json.dumps({"success": False, "error": format_session_db_unavailable()})
                from tools.session_search_tool import session_search as _session_search
                return _session_search(
                    query=next_args.get("query", ""),
                    role_filter=next_args.get("role_filter"),
                    limit=next_args.get("limit", 3),
                    session_id=next_args.get("session_id"),
                    around_message_id=next_args.get("around_message_id"),
                    window=next_args.get("window", 5),
                    sort=next_args.get("sort"),
                    detail=next_args.get("detail", "adaptive"),
                    db=session_db,
                    current_session_id=agent.session_id,
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
        elif function_name == "memory":
            def _execute(next_args: dict) -> Any:
                target = next_args.get("target", "memory")
                operations = next_args.get("operations")
                from tools.memory_tool import memory_tool as _memory_tool
                result = _memory_tool(
                    action=next_args.get("action"),
                    target=target,
                    content=next_args.get("content"),
                    old_text=next_args.get("old_text"),
                    operations=operations,
                    store=agent._memory_store,
                )
                # Mirror successful built-in memory writes to external
                # providers. All gating/op-expansion lives behind the manager
                # interface (MemoryManager.notify_memory_tool_write).
                if agent._memory_manager:
                    agent._memory_manager.notify_memory_tool_write(
                        result,
                        next_args,
                        build_metadata=lambda: agent._build_memory_write_metadata(
                            task_id=effective_task_id,
                            tool_call_id=tool_call_id,
                        ),
                    )
                return result
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
        elif function_name == "clarify":
            def _execute(next_args: dict) -> Any:
                from tools.clarify_tool import clarify_tool as _clarify_tool
                return _clarify_tool(
                    question=next_args.get("question", ""),
                    choices=next_args.get("choices"),
                    multi_select=next_args.get("multi_select", False),
                    questions=next_args.get("questions"),
                    callback=agent.clarify_callback,
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
        elif function_name == "read_terminal":
            def _execute(next_args: dict) -> Any:
                from tools.read_terminal_tool import read_terminal_tool as _read_terminal_tool
                return _read_terminal_tool(
                    start_line=next_args.get("start_line"),
                    count=next_args.get("count"),
                    callback=getattr(agent, "read_terminal_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('read_terminal', function_args, tool_duration, result=function_result)}")
        elif function_name == "desktop_preview":
            def _execute(next_args: dict) -> Any:
                if (next_args.get("action") or "").strip() == "read":
                    from tools.read_preview_tool import read_preview_tool as _read_preview_tool
                    return _read_preview_tool(
                        start=next_args.get("start"),
                        count=next_args.get("count"),
                        callback=getattr(agent, "read_preview_callback", None),
                    )
                from tools.preview_tool import _handle_preview
                return _handle_preview(next_args)
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('desktop_preview', function_args, tool_duration, result=function_result)}")
        elif function_name == "drive_preview":
            def _execute(next_args: dict) -> Any:
                from tools.drive_preview_tool import drive_preview_tool as _drive_preview_tool
                return _drive_preview_tool(
                    action=next_args.get("action", ""),
                    ref=next_args.get("ref"),
                    selector=next_args.get("selector"),
                    text=next_args.get("text"),
                    key=next_args.get("key"),
                    submit=next_args.get("submit"),
                    amount=next_args.get("amount"),
                    to=next_args.get("to"),
                    limit=next_args.get("max"),
                    callback=getattr(agent, "drive_preview_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('drive_preview', function_args, tool_duration, result=function_result)}")
        elif function_name == "annotate_preview":
            def _execute(next_args: dict) -> Any:
                from tools.annotate_preview_tool import annotate_preview_tool as _annotate_preview_tool
                return _annotate_preview_tool(
                    action=next_args.get("action", "add"),
                    ref=next_args.get("ref"),
                    selector=next_args.get("selector"),
                    label=next_args.get("label"),
                    callback=getattr(agent, "drive_preview_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('annotate_preview', function_args, tool_duration, result=function_result)}")
        elif function_name == "read_window_below":
            def _execute(next_args: dict) -> Any:
                from tools.read_window_tool import read_window_below_tool as _read_window_below_tool
                return _read_window_below_tool(
                    callback=getattr(agent, "read_window_below_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('read_window_below', function_args, tool_duration, result=function_result)}")
        elif function_name == "tour":
            def _execute(next_args: dict) -> Any:
                from tools.tour_tool import tour_tool as _tour_tool
                return _tour_tool(
                    action=next_args.get("action", ""),
                    surface=next_args.get("surface"),
                    selector=next_args.get("selector"),
                    title=next_args.get("title"),
                    text=next_args.get("text"),
                    side=next_args.get("side"),
                    steps=next_args.get("steps"),
                    step_index=next_args.get("step_index"),
                    callback=getattr(agent, "tour_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('tour', function_args, tool_duration, result=function_result)}")
        elif function_name == "setup_mcp":
            def _execute(next_args: dict) -> Any:
                from tools.setup_mcp_tool import setup_mcp_tool as _setup_mcp_tool
                return _setup_mcp_tool(
                    server=next_args.get("server", ""),
                    action=next_args.get("action", "install"),
                    reason=next_args.get("reason", ""),
                    callback=getattr(agent, "setup_mcp_callback", None),
                )
            function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                agent,
                function_name=function_name,
                function_args=function_args,
                effective_task_id=effective_task_id,
                tool_call_id=getattr(tool_call, "id", "") or "",
                execute=_execute,
                scope_block=_ts_scope_block,
                display_index=i,
            ))
            tool_duration = time.time() - tool_start_time
            if agent._should_emit_quiet_tool_messages():
                agent._vprint(f"  {_get_cute_tool_message_impl('setup_mcp', function_args, tool_duration, result=function_result)}")
        elif function_name == "delegate_task":
            _action_arg = str(function_args.get("action") or "").strip().lower()
            tasks_arg = function_args.get("tasks")
            if _action_arg in ("list", "steer", "stop"):
                spinner_label = f"🔀 subagent {_action_arg}"
            elif tasks_arg and isinstance(tasks_arg, list):
                spinner_label = f"🔀 delegating {len(tasks_arg)} tasks · (/agents to monitor)"
            else:
                goal_preview = (function_args.get("goal") or "")[:30]
                spinner_label = (
                    f"🔀 {goal_preview} · (/agents to monitor)"
                    if goal_preview
                    else "🔀 delegating · (/agents to monitor)"
                )
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            agent._delegate_spinner = spinner
            _delegate_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._dispatch_delegate_task(next_args)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _delegate_result = function_result
            finally:
                agent._delegate_spinner = None
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._context_engine_tool_names and function_name in agent._context_engine_tool_names:
            # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
            spinner = None
            if agent._should_emit_quiet_tool_messages():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
                preview = _build_tool_label(function_name, display_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _ce_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _ce_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
            # Memory provider tools (hindsight_retain, honcho_search, etc.)
            # These are not in the tool registry — route through MemoryManager.
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
                preview = _build_tool_label(function_name, display_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _mem_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    return agent._memory_manager.handle_tool_call(function_name, next_args)
                function_result, function_args, middleware_trace, _execution_blocked, _execution_dispatched = _managed_values(_run_agent_tool_execution_middleware(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    execute=_execute,
                    scope_block=_ts_scope_block,
                    display_index=i,
                ))
                _mem_result = function_result
            except Exception as tool_error:
                function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        elif agent.quiet_mode:
            spinner = None
            if agent._should_emit_quiet_tool_messages() and agent._should_start_quiet_spinner():
                face = random.choice(KawaiiSpinner.get_waiting_faces())
                emoji = _get_tool_emoji(function_name)
                display_args = _redact_tool_args_for_display(function_name, function_args) or function_args
                preview = _build_tool_label(function_name, display_args) or function_name
                spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=agent._print_fn)
                spinner.start()
            _spinner_result = None
            try:
                def _execute(next_args: dict) -> Any:
                    from model_tools import suppress_post_tool_call_hook

                    with suppress_post_tool_call_hook():
                        return _ra().handle_function_call(
                            function_name,
                            next_args,
                            effective_task_id,
                            tool_call_id=tool_call_id,
                            session_id=agent.session_id or "",
                            turn_id=getattr(agent, "_current_turn_id", "") or "",
                            api_request_id=getattr(agent, "_current_api_request_id", "")
                            or "",
                            enabled_tools=(
                                list(agent.valid_tool_names)
                                if agent.valid_tool_names
                                else None
                            ),
                            skip_pre_tool_call_hook=True,
                            skip_tool_request_middleware=True,
                            skip_tool_execution_middleware=True,
                            tool_request_middleware_trace=list(middleware_trace),
                            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                        )

                (
                    function_result,
                    function_args,
                    middleware_trace,
                    _execution_blocked,
                    _execution_dispatched,
                ) = _managed_values(
                    _run_agent_tool_execution_middleware(
                        agent,
                        function_name=function_name,
                        function_args=function_args,
                        effective_task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                        execute=_execute,
                        scope_block=_ts_scope_block,
                        display_index=i,
                        middleware_trace=middleware_trace,
                    )
                )
                _spinner_result = function_result
            except KeyboardInterrupt:
                function_result = _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=tool_start_time,
                    middleware_trace=list(middleware_trace),
                )
                _spinner_result = function_result
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                # Emit a tool result for THIS call and every remaining call in
                # the batch before re-raising, so the assistant tool-call turn
                # is never left without matching tool results (alternation).
                _append_cancelled_tool_results(
                    messages,
                    assistant_message.tool_calls[i - 1:],
                    reason="keyboard interrupt",
                )
                raise
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
            finally:
                tool_duration = time.time() - tool_start_time
                cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                if spinner:
                    spinner.stop(cute_msg)
                elif agent._should_emit_quiet_tool_messages():
                    agent._vprint(f"  {cute_msg}")
        else:
            try:
                def _execute(next_args: dict) -> Any:
                    from model_tools import suppress_post_tool_call_hook

                    with suppress_post_tool_call_hook():
                        return _ra().handle_function_call(
                            function_name,
                            next_args,
                            effective_task_id,
                            tool_call_id=tool_call_id,
                            session_id=agent.session_id or "",
                            turn_id=getattr(agent, "_current_turn_id", "") or "",
                            api_request_id=getattr(agent, "_current_api_request_id", "")
                            or "",
                            enabled_tools=(
                                list(agent.valid_tool_names)
                                if agent.valid_tool_names
                                else None
                            ),
                            skip_pre_tool_call_hook=True,
                            skip_tool_request_middleware=True,
                            skip_tool_execution_middleware=True,
                            tool_request_middleware_trace=list(middleware_trace),
                            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                        )

                (
                    function_result,
                    function_args,
                    middleware_trace,
                    _execution_blocked,
                    _execution_dispatched,
                ) = _managed_values(
                    _run_agent_tool_execution_middleware(
                        agent,
                        function_name=function_name,
                        function_args=function_args,
                        effective_task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                        execute=_execute,
                        scope_block=_ts_scope_block,
                        display_index=i,
                        middleware_trace=middleware_trace,
                    )
                )
            except KeyboardInterrupt:
                _emit_cancelled_terminal_post_tool_call(
                    agent,
                    function_name=function_name,
                    function_args=function_args,
                    effective_task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                    start_time=tool_start_time,
                    middleware_trace=list(middleware_trace),
                )
                try:
                    agent.interrupt("keyboard interrupt")
                except Exception:
                    pass
                # Emit a tool result for THIS call and every remaining call in
                # the batch before re-raising (see interactive branch above).
                _append_cancelled_tool_results(
                    messages,
                    assistant_message.tool_calls[i - 1:],
                    reason="keyboard interrupt",
                )
                raise
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
            tool_duration = time.time() - tool_start_time

        _execution_timed_out = isinstance(
            function_result, (_ToolTimeoutResult, _ToolCancelledResult)
        )
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
        _is_error_result, _ = _detect_tool_failure(function_name, function_result)
        # The agent-runtime tools above (todo, session_search, memory,
        # context-engine, memory-manager, clarify, delegate_task) are
        # dispatched inline — they never reach handle_function_call, so the
        # executor is the one that has to fire post_tool_call. For
        # Every dispatch suppresses the inner handle_function_call observer so
        # the executor owns one terminal event for this tool_call_id. This also
        # prevents an abandoned timeout worker from reporting late success.
        _executor_must_emit_post_hook = (
            not _execution_blocked
            and not _execution_timed_out
        )
        if _executor_must_emit_post_hook:
            _emit_terminal_post_tool_call(
                agent,
                function_name=function_name,
                function_args=function_args,
                result=function_result,
                effective_task_id=effective_task_id,
                tool_call_id=tool_call_id,
                duration_ms=int(tool_duration * 1000),
                middleware_trace=list(middleware_trace),
            )
        if not _execution_blocked:
            function_result = agent._append_guardrail_observation(
                function_name,
                function_args,
                function_result,
                failed=_is_error_result,
                tool_call_id=tool_call_id,
            )
            result_preview = function_result if agent.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )
        if _is_error_result:
            logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
        else:
            logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, _result_len)

        # Track file-mutation outcome for the turn-end verifier.  See
        # the concurrent path for the rationale; both paths must feed
        # the same state so the footer reflects every tool call in the
        # turn, not just the parallel ones.
        if not _execution_blocked:
            try:
                agent._record_file_mutation_result(
                    function_name, function_args, function_result, _is_error_result,
                )
            except Exception as _ver_err:
                logging.debug("file-mutation verifier record failed: %s", _ver_err)

        agent._current_tool = None
        _status_suffix = " (error)" if _is_error_result else ""
        agent._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s){_status_suffix}")

        if agent.verbose_logging:
            logging.debug("Tool %s completed in %.2fs", function_name, tool_duration)
            _log_result = _multimodal_text_summary(function_result)
            logging.debug("Tool result (%d chars): %s", len(_log_result), _log_result)

        display_function_result = function_result
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=function_name,
            tool_use_id=tool_call_id,
            env=get_active_env(effective_task_id),
            config=_tool_budget,
        ) if not _is_multimodal_tool_result(function_result) else function_result
        _record_persisted_path_for_stub(agent, tool_call_id, function_result)

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
        tool_message = make_tool_result_message(
            function_name,
            _tool_content,
            tool_call_id,
            effect_disposition="unknown" if _execution_timed_out else None,
        )
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
                logging.debug("Tool progress callback error: %s", cb_err)

        if not _execution_blocked and agent.tool_complete_callback:
            try:
                display_args = (
                    _redact_tool_args_for_display(function_name, function_args)
                    or function_args
                )
                agent.tool_complete_callback(
                    tool_call_id,
                    function_name,
                    display_args,
                    display_function_result,
                )
            except Exception as cb_err:
                logging.debug("Tool complete callback error: %s", cb_err)

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
                    tool_call_id=tool_call_id,
                    risk_metadata=risk_metadata,
                )
            except Exception as cb_err:
                logging.debug("Tool output risk callback error: %s", cb_err)

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
                    _pairing_tool_call_id(skipped_tc),
                    effect_disposition="none",
                ))
                if not _flush_session_db_after_tool_progress(
                    agent,
                    messages,
                    stage=f"skipped tool result {skipped_name}",
                ):
                    return
            break

    if finalize:
        _finalize_tool_batch(agent, messages, effective_task_id, len(tool_calls), _tool_budget)


def execute_tool_calls_segmented(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments=None) -> None:
    """Execute a mixed batch as ordered parallel/sequential segments (the ``(kind, calls)``
    plan from ``_plan_tool_batch_segments``), preserving per-call result order and barrier
    boundaries exactly as fully-sequential execution. Turn-end work (budget + /steer) runs
    once here (segments run with ``finalize=False``); each segment executor checks the
    interrupt flag up front, so an interrupt drains later segments with one result per call."""
    from types import SimpleNamespace

    if segments is None:
        _active_env = get_active_env(effective_task_id)
        _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
        segments = _plan_tool_batch_segments(assistant_message.tool_calls, execution_cwd=_exec_cwd)

    for kind, calls in segments:
        if getattr(agent, "_incremental_persistence_failed", False):
            return
        segment_message = SimpleNamespace(tool_calls=list(calls))
        run_segment = execute_tool_calls_concurrent if kind == "parallel" else execute_tool_calls_sequential
        run_segment(agent, segment_message, messages, effective_task_id, api_call_count, finalize=False)
        if getattr(agent, "_incremental_persistence_failed", False):
            return

    total_tools = len(assistant_message.tool_calls)
    if total_tools > 0:
        _finalize_tool_batch(agent, messages, effective_task_id, total_tools, _budget_for_agent(agent))


__all__ = [
    "execute_tool_calls_concurrent",
    "execute_tool_calls_sequential",
    "execute_tool_calls_segmented",
]
