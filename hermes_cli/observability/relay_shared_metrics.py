"""Direct NeMo Relay integration for Hermes shared client metrics."""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Callable

from agent import relay_runtime
from hermes_cli import __version__

from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import (
    MODEL_CALL_SCOPE,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    SUBSCRIBER_NAME,
    TASK_SCOPE,
    model_call_fields,
    model_call_outcome,
    task_start_fields,
    task_terminal_fields,
)
from .shared_metrics_subscriber import SharedMetricsSubscriber

logger = logging.getLogger(__name__)

HANDLED_HOOKS = frozenset({
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "pre_api_request",
    "post_tool_call",
    "post_api_request",
    "api_request_error",
    "subagent_stop",
})

_RUNTIME_FAILED = object()
_RUNTIMES: dict[str, _Runtime | object] = {}
_RUNTIME_LOCK = threading.RLock()


def _retry_ordinal(event: dict[str, Any]) -> int | None:
    value = event.get("retry_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


@dataclass
class _ModelCall:
    handle: Any
    task_id: str
    fields: dict[str, str]
    retry_ordinal: int | None = None


@dataclass
class _TaskRun:
    handle: Any
    context: contextvars.Context
    started_ns: int
    start_fields: dict[str, str]
    model_call_ids: set[str] = field(default_factory=set)
    tool_call_ids: set[str] = field(default_factory=set)
    turn_ids: set[str] = field(default_factory=set)
    unidentified_tool_calls: int = 0
    retry_count: int = 0


@dataclass
class _MetricsSession:
    session_id: str
    relay_session: relay_runtime.RelaySession
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    model_calls: dict[str, _ModelCall] = field(default_factory=dict)
    tasks: dict[str, _TaskRun] = field(default_factory=dict)


class _Runtime:
    """Own shared-metrics state layered on the Hermes core Relay host."""

    def __init__(self, host: relay_runtime.RelayRuntime | None = None) -> None:
        resolved_host = host or relay_runtime.get_runtime()
        if resolved_host is None:
            raise RuntimeError("Hermes core Relay runtime is unavailable")
        self.host: relay_runtime.RelayRuntime = resolved_host
        self.relay = self.host.relay
        self._sessions_lock = threading.RLock()
        self._active = True
        self._sessions: dict[str, _MetricsSession] = {}
        self._task_creation_lock = threading.RLock()
        self._task_sessions_lock = threading.RLock()
        self._task_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._turn_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._subscriber_name = f"{SUBSCRIBER_NAME}.{self.host.runtime_id}"
        self.subscriber = SharedMetricsSubscriber(
            SharedMetricsStore(),
            __version__,
            runtime_id=self.host.runtime_id,
        )
        self.relay.subscribers.register(self._subscriber_name, self.subscriber)
        self._registered = True
        atexit.register(self.shutdown)

    def ensure_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return None
        with self._sessions_lock:
            if not self._active:
                return None
            relay_session = self.host.ensure_session(event)
            if relay_session is None:
                return None
            session = self._sessions.get(session_id)
            if session is None:
                session = _MetricsSession(
                    session_id=session_id,
                    relay_session=relay_session,
                )
                self._sessions[session_id] = session
        with session.lock:
            if session.closing:
                return None
        return session

    def _run_in_session(
        self,
        session: _MetricsSession,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.host.run_in_session(
            session.relay_session,
            callback,
            *args,
            **kwargs,
        )

    def start_task(self, event: dict[str, Any]) -> _TaskRun | None:
        """Open one Relay function scope for a Hermes task run."""
        task_key = self._task_key(event)
        if task_key is None:
            return None
        _, task_id = task_key
        with self._task_creation_lock:
            owner = self._task_session(event)
            if owner is not None:
                with owner.lock:
                    if owner.closing:
                        return None
                    task = owner.tasks.get(task_id)
                    if task is not None:
                        self._remember_turn(owner, task, event)
                    return task

            session = self.ensure_session(event)
            if session is None:
                return None
            with session.lock:
                if session.closing or session.relay_session.context is None:
                    return None
                task_context = session.relay_session.context.copy()
                start_fields = task_start_fields(event)
                active_turn = relay_runtime.active_turn(session.session_id)
                parent_handle = session.relay_session.handle
                if (
                    active_turn is not None
                    and active_turn.lease.session_id == session.session_id
                    and active_turn.task_id == task_id
                    and active_turn.handle is not None
                ):
                    parent_handle = active_turn.handle

                def push_task() -> Any:
                    self.relay.get_scope_stack()
                    return self.relay.scope.push(
                        TASK_SCOPE,
                        self.relay.ScopeType.Function,
                        handle=parent_handle,
                        input=start_fields,
                        metadata=self._event_metadata(),
                    )

                handle = task_context.run(push_task)
                task = _TaskRun(
                    handle=handle,
                    context=task_context,
                    started_ns=monotonic_ns(),
                    start_fields=start_fields,
                )
                session.tasks[task_id] = task
                with self._task_sessions_lock:
                    self._task_sessions[task_key] = session
                self._remember_turn(session, task, event)
                return task

    def _run_in_task(
        self,
        task: _TaskRun,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        def invoke() -> Any:
            self.relay.get_scope_stack()
            return callback(*args, **kwargs)

        return task.context.copy().run(invoke)

    def start_model_call(self, event: dict[str, Any]) -> None:
        task_id = str(event.get("task_id") or "")
        session = self._task_session(event, allow_task_id_fallback=True)
        task = session.tasks.get(task_id) if session is not None else None
        if task is None:
            task = self.start_task(event)
            session = self._task_session(event) if task is not None else None
        if session is None:
            session = self.ensure_session(event)
        if session is None:
            return
        request_id = str(event.get("api_request_id") or "")
        if not request_id:
            return
        fields = model_call_fields(event)
        retry_ordinal = _retry_ordinal(event)
        model_family = fields["model_family"]
        with session.lock:
            if session.closing:
                return
            if task is not None:
                self._remember_turn(session, task, event)
            existing = session.model_calls.get(request_id)
            if existing is not None:
                existing.fields = fields
                if task is not None:
                    if retry_ordinal is None or existing.retry_ordinal is None:
                        task.retry_count += 1
                    elif retry_ordinal > existing.retry_ordinal:
                        task.retry_count += retry_ordinal - existing.retry_ordinal
                if retry_ordinal is not None:
                    existing.retry_ordinal = max(
                        existing.retry_ordinal or 0,
                        retry_ordinal,
                    )
                return
            if task is not None:
                task.model_call_ids.add(request_id)
                if retry_ordinal is not None and retry_ordinal > 0:
                    # A real Hermes retry can advance api_request_id while
                    # carrying the retry ordinal. Count that physical attempt.
                    task.retry_count += 1
                handle = self._run_in_task(
                    task,
                    self.relay.llm.call,
                    MODEL_CALL_SCOPE,
                    self.relay.LLMRequest({}, {}),
                    handle=task.handle,
                    metadata=self._event_metadata(),
                    model_name=model_family,
                )
            else:
                handle = self._run_in_session(
                    session,
                    self.relay.llm.call,
                    MODEL_CALL_SCOPE,
                    self.relay.LLMRequest({}, {}),
                    handle=session.relay_session.handle,
                    metadata=self._event_metadata(),
                    model_name=model_family,
                )
            session.model_calls[request_id] = _ModelCall(
                handle=handle,
                task_id=str(event.get("task_id") or ""),
                fields=fields,
                retry_ordinal=retry_ordinal,
            )

    def record_tool_call(self, event: dict[str, Any]) -> None:
        """Count one unique tool invocation under its owning task."""
        task_id = str(event.get("task_id") or "")
        session = self._task_session(event, allow_task_id_fallback=True)
        task = session.tasks.get(task_id) if session is not None else None
        if task is None:
            task = self.start_task(event)
            session = self._task_session(event) if task is not None else None
        if session is None or task is None:
            return
        tool_call_id = str(event.get("tool_call_id") or "")
        with session.lock:
            if session.closing:
                return
            self._remember_turn(session, task, event)
            if tool_call_id:
                task.tool_call_ids.add(tool_call_id)
            else:
                task.unidentified_tool_calls += 1

    def end_model_call(self, event: dict[str, Any], outcome: str | None = None) -> None:
        session = self._task_session(event, allow_task_id_fallback=True)
        if session is None:
            session = self._session(event)
        if session is None:
            return
        request_id = str(event.get("api_request_id") or "")
        with session.lock:
            if session.closing:
                return
            model_call = session.model_calls.get(request_id)
            if model_call is None:
                return
            fields = model_call_fields(event)
            model_call.fields = fields
            self._finish_model_call(
                session,
                request_id,
                outcome or model_call_outcome(event),
            )

    def end_pending_model_calls(self, event: dict[str, Any]) -> None:
        session = self._task_session(event, allow_task_id_fallback=True)
        if session is None:
            session = self._session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            self._end_pending_model_calls(session, event)

    def finish_task(self, event: dict[str, Any]) -> None:
        """Close one task scope exactly once with bounded terminal fields."""
        task_id = str(event.get("task_id") or "")
        session = self._task_session(
            event,
            allow_task_id_fallback=True,
        ) or self._session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            self._finish_task(session, task_id, event)

    def close_session(self, event: dict[str, Any]) -> None:
        session = self._session(event)
        if session is None:
            return
        failures: list[str] = []
        with session.lock:
            if session.closing:
                return
            session.closing = True
            for task_id in list(session.tasks):
                self._finish_task(
                    session,
                    task_id,
                    {
                        **event,
                        "task_id": task_id,
                        "completed": False,
                        "failed": True,
                        "interrupted": False,
                        "turn_exit_reason": "system_aborted",
                    },
                )
            self._end_pending_model_calls(session, event)
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            failures.append(f"subscriber flush failed: {exc}")
        self._export()
        with self._sessions_lock:
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)
        if failures:
            logger.warning(
                "Hermes shared-metrics session %s closed with errors: %s",
                session.session_id,
                "; ".join(failures),
            )

    def shutdown(self) -> None:
        with self._sessions_lock:
            self._active = False
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if not self._registered:
            return
        self._safe(self.relay.subscribers.flush)
        self._export()
        self._safe(self.relay.subscribers.deregister, self._subscriber_name)
        self._registered = False
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass

    def deactivate(self) -> None:
        """Stop collection without exporting locally aggregated metrics."""
        with self._sessions_lock:
            self._active = False
        self.subscriber.deactivate()
        if self._registered:
            self._safe(self.relay.subscribers.deregister, self._subscriber_name)
            self._registered = False
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            with session.lock:
                if session.closing:
                    continue
                session.closing = True
                for task_id in list(session.tasks):
                    self._finish_task(
                        session,
                        task_id,
                        {
                            "session_id": session.session_id,
                            "task_id": task_id,
                            "failed": True,
                            "turn_exit_reason": "system_aborted",
                        },
                    )
                self._end_pending_model_calls(session, {})
        with self._sessions_lock:
            self._sessions.clear()
        with self._task_sessions_lock:
            self._task_sessions.clear()
            self._turn_sessions.clear()
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass

    def _session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = str(event.get("session_id") or "")
        with self._sessions_lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _task_key(event: dict[str, Any]) -> tuple[str, str] | None:
        session_id = str(event.get("session_id") or "")
        task_id = str(event.get("task_id") or "")
        if not session_id or not task_id:
            return None
        return session_id, task_id

    def _task_session(
        self,
        event: dict[str, Any],
        *,
        allow_task_id_fallback: bool = False,
    ) -> _MetricsSession | None:
        task_key = self._task_key(event)
        if task_key is None:
            return None
        turn_key = self._turn_key(event)
        with self._task_sessions_lock:
            if turn_key is not None:
                owner = self._turn_sessions.get(turn_key)
                if owner is not None:
                    return owner
            owner = self._task_sessions.get(task_key)
            if owner is not None or not allow_task_id_fallback:
                return owner
            task_id = task_key[1]
            candidates: list[_MetricsSession] = []
            for (_, candidate_task_id), session in self._task_sessions.items():
                if candidate_task_id != task_id:
                    continue
                if not any(candidate is session for candidate in candidates):
                    candidates.append(session)
            return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _turn_key(event: dict[str, Any]) -> tuple[str, str] | None:
        session_id = str(event.get("session_id") or "")
        turn_id = str(event.get("turn_id") or "")
        if not session_id or not turn_id:
            return None
        return session_id, turn_id

    def _remember_turn(
        self,
        session: _MetricsSession,
        task: _TaskRun,
        event: dict[str, Any],
    ) -> None:
        turn_id = str(event.get("turn_id") or "")
        if not turn_id:
            return
        task.turn_ids.add(turn_id)
        with self._task_sessions_lock:
            self._turn_sessions[(session.session_id, turn_id)] = session

    def _finish_model_call(
        self,
        session: _MetricsSession,
        request_id: str,
        outcome: str,
    ) -> None:
        model_call = session.model_calls.pop(request_id, None)
        if model_call is None:
            return
        try:
            task = session.tasks.get(model_call.task_id)
            if task is not None:
                self._run_in_task(
                    task,
                    self.relay.llm.call_end,
                    model_call.handle,
                    {**model_call.fields, "outcome": outcome},
                    metadata=self._event_metadata(),
                )
            else:
                self._run_in_session(
                    session,
                    self.relay.llm.call_end,
                    model_call.handle,
                    {**model_call.fields, "outcome": outcome},
                    metadata=self._event_metadata(),
                )
        except Exception:
            logger.warning(
                "Hermes shared-metrics model call close failed", exc_info=True
            )

    def _end_pending_model_calls(
        self,
        session: _MetricsSession,
        event: dict[str, Any],
    ) -> None:
        task_id = str(event.get("task_id") or "")
        request_ids = [
            request_id
            for request_id, model_call in session.model_calls.items()
            if not task_id or model_call.task_id == task_id
        ]
        outcome = "cancelled" if event.get("interrupted") else "failed"
        for request_id in request_ids:
            self._finish_model_call(session, request_id, outcome)

    def _finish_task(
        self,
        session: _MetricsSession,
        task_id: str,
        event: dict[str, Any],
    ) -> None:
        task = session.tasks.get(task_id)
        if task is None:
            return
        self._end_pending_model_calls(session, {**event, "task_id": task_id})
        fields = task_terminal_fields(
            {**task.start_fields, **event},
            duration_ms=max(0, (monotonic_ns() - task.started_ns) // 1_000_000),
            model_call_count=len(task.model_call_ids),
            tool_call_count=len(task.tool_call_ids) + task.unidentified_tool_calls,
            retry_count=task.retry_count,
        )
        try:
            self._run_in_task(
                task,
                self.relay.scope.pop,
                task.handle,
                output=fields,
                metadata=self._event_metadata(),
            )
        except Exception:
            logger.warning("Hermes shared-metrics task close failed", exc_info=True)
        finally:
            session.tasks.pop(task_id, None)
            with self._task_sessions_lock:
                task_key = (session.session_id, task_id)
                if self._task_sessions.get(task_key) is session:
                    self._task_sessions.pop(task_key, None)
                for turn_id in task.turn_ids:
                    turn_key = (session.session_id, turn_id)
                    if self._turn_sessions.get(turn_key) is session:
                        self._turn_sessions.pop(turn_key, None)

    def _export(self) -> None:
        self._safe(self.subscriber.store.create_and_export_package)

    def _event_metadata(self) -> dict[str, str]:
        return {
            SCHEMA_KEY: SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: self.host.runtime_id,
        }

    @staticmethod
    def _safe(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning("Hermes shared metrics operation failed", exc_info=True)
            return None


def enabled() -> bool:
    """Return the shared-metrics policy for the active Hermes profile."""
    profile_key = relay_runtime.current_profile_key()
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        logger.debug("Unable to read Hermes shared-metrics policy", exc_info=True)
        value = False
    else:
        telemetry = config.get("telemetry") if isinstance(config, dict) else None
        shared_metrics = (
            telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
        )
        value = (
            isinstance(shared_metrics, dict)
            and shared_metrics.get("enabled") is True
        )
    if value:
        return True
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.pop(profile_key, None)
        if isinstance(runtime, _Runtime):
            runtime.deactivate()
    return False


def handles_hook(hook_name: str) -> bool:
    return hook_name in HANDLED_HOOKS and enabled()


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Project one Hermes lifecycle event into the core Relay integration."""
    if not handles_hook(hook_name):
        return
    runtime = _get_runtime()
    if runtime is None:
        return
    try:
        if hook_name == "on_session_start":
            runtime.ensure_session(kwargs)
        elif hook_name == "pre_llm_call":
            runtime.start_task(kwargs)
        elif hook_name == "pre_api_request":
            runtime.start_model_call(kwargs)
        elif hook_name == "post_tool_call":
            runtime.record_tool_call(kwargs)
        elif hook_name == "post_api_request":
            runtime.end_model_call(kwargs, "success")
        elif hook_name == "api_request_error":
            if kwargs.get("retryable") is False:
                runtime.end_model_call(kwargs, "failed")
        elif hook_name == "on_session_end":
            runtime.finish_task(kwargs)
        elif hook_name == "subagent_stop":
            child_session_id = str(kwargs.get("child_session_id") or "")
            if child_session_id:
                runtime.close_session({"session_id": child_session_id})
        elif hook_name in {"on_session_finalize", "on_session_reset"}:
            runtime.close_session(kwargs)
    except Exception:
        logger.warning(
            "Hermes shared metrics hook failed: %s", hook_name, exc_info=True
        )


def prepare_session_start() -> None:
    """Register the subscriber before any producer opens the session scope."""
    if enabled():
        _get_runtime(retry_failed=True)


def _prepare_core_session(
    host: relay_runtime.RelayRuntime,
    context: dict[str, Any],
) -> None:
    """Prepare the profile subscriber before the coordinator opens a scope."""
    del context
    if host.profile_key == relay_runtime.current_profile_key():
        if enabled():
            _get_runtime(retry_failed=True, host=host)


def start_task_run(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    parent_session_id: str = "",
) -> None:
    """Start task metrics at the outer Hermes execution boundary."""
    if not enabled():
        return
    runtime = _get_runtime(retry_failed=True)
    if runtime is None:
        return
    runtime._safe(
        runtime.start_task,
        {
            "session_id": session_id,
            "task_id": task_id,
            "platform": platform,
            "parent_session_id": parent_session_id,
        },
    )


def finish_task_run(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    result: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Finish task metrics for every return or exception path."""
    if not enabled():
        return
    runtime = _get_runtime()
    if runtime is None:
        return

    terminal = result if isinstance(result, dict) else {}
    interrupted = terminal.get("interrupted") is True
    completed = terminal.get("completed") is True
    failed = terminal.get("failed") is True
    reason = str(
        terminal.get("turn_exit_reason") or terminal.get("failure_reason") or ""
    )
    if error is not None:
        interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError)) or (
            type(error).__name__ == "CancelledError"
        )
        timed_out = isinstance(error, TimeoutError)
        completed = False
        failed = not interrupted
        if interrupted:
            reason = "interrupted_by_user"
        elif timed_out:
            reason = "timed_out"
        else:
            reason = "system_aborted"
    elif not reason:
        reason = "failed" if failed else "unknown"

    runtime._safe(
        runtime.finish_task,
        {
            "session_id": session_id,
            "task_id": task_id,
            "platform": platform,
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted,
            "turn_exit_reason": reason,
        },
    )


def _get_runtime(
    *,
    retry_failed: bool = False,
    host: relay_runtime.RelayRuntime | None = None,
) -> _Runtime | None:
    profile_key = relay_runtime.current_profile_key()
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(profile_key)
        if isinstance(runtime, _Runtime):
            if host is None or runtime.host is host:
                return runtime
            runtime.deactivate()
            _RUNTIMES.pop(profile_key, None)
        if runtime is _RUNTIME_FAILED and not retry_failed:
            return None
        if runtime is _RUNTIME_FAILED:
            _RUNTIMES.pop(profile_key, None)
        try:
            runtime = _Runtime(host=host)
        except Exception:
            logger.warning("Hermes shared metrics initialization failed", exc_info=True)
            _RUNTIMES[profile_key] = _RUNTIME_FAILED
            return None
        _RUNTIMES[profile_key] = runtime
        return runtime


relay_runtime.SESSION_COORDINATOR.register_session_initializer(
    SUBSCRIBER_NAME,
    _prepare_core_session,
)


def _reset_for_tests() -> None:
    """Reset all profile-scoped shared-metrics state for isolated tests."""
    with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        if isinstance(runtime, _Runtime):
            runtime.shutdown()
