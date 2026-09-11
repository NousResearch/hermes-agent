"""Direct NeMo Relay integration for Hermes shared client metrics."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Callable

from agent import relay_runtime
from hermes_cli import __version__

from .shared_metrics import SharedMetricsStore
from . import shared_metrics_contract as contract
from .shared_metrics_contract import MODEL_CALL_SCOPE, SUBSCRIBER_NAME, TASK_SCOPE
from .shared_metrics_subscriber import SharedMetricsSubscriber

logger = logging.getLogger(__name__)

_RUNTIME_FAILED = object()
_RUNTIMES: dict[str, _Runtime | object] = {}
_RUNTIME_LOCK = threading.RLock()

_ABORTED = {"failed": True, "turn_exit_reason": "system_aborted"}


def _text(event: dict[str, Any], key: str) -> str:
    return str(event.get(key) or "")


def _session_pair(event: dict[str, Any], key: str) -> tuple[str, str] | None:
    """(session_id, event[key]) when both are non-empty."""
    session_id, value = _text(event, "session_id"), _text(event, key)
    return (session_id, value) if session_id and value else None


def _retry_ordinal(event: dict[str, Any]) -> int:
    """Hermes's provider-local retry ordinal; 0 when absent or malformed."""
    value = event.get("retry_count")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _forget(index: dict[Any, _MetricsSession], key: Any, owner: _MetricsSession) -> None:
    """Drop ``key`` from ``index`` only while it still points at ``owner``."""
    if index.get(key) is owner:
        index.pop(key, None)


def _task_parent_handle(session: _MetricsSession, task_id: str) -> Any:
    """The active turn's handle when it owns this exact task, else the session handle."""
    active_turn = relay_runtime.active_turn(session.session_id)
    if (
        active_turn is not None
        and active_turn.lease.session_id == session.session_id
        and active_turn.task_id == task_id
        and active_turn.handle is not None
    ):
        return active_turn.handle
    return session.relay_session.handle


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (monotonic_ns() - started_ns) // 1_000_000)


def _scope_handle(session: _MetricsSession, task: _TaskRun | None) -> Any:
    return task.handle if task is not None else session.relay_session.handle


def _sole(items: Any) -> Any:
    """The single distinct element of ``items`` (identity-deduplicated), else None."""
    unique = {id(item): item for item in items}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _identities_compatible(candidate: tuple[str, str, str], observed: tuple[str, str, str]) -> bool:
    """Match partial hook context without crossing known call boundaries."""
    if not observed[2] or candidate[2] != observed[2]:
        return False
    return all(
        not left or not right or left == right
        for left, right in zip(candidate[:2], observed[:2], strict=True)
    )


def _compatible_tool_call_keys(
    session: _MetricsSession, task_id: str, identity: tuple[str, str, str]
) -> list[tuple[str, str, str, str]]:
    return [
        key
        for key in session.tool_calls
        if key[0] == task_id and _identities_compatible(key[1:], identity)
    ]


@dataclass
class _ModelCall:
    handle: Any
    task_id: str
    fields: dict[str, str]


@dataclass
class _ToolCall:
    handle: Any
    category: str
    started_ns: int
    approval_outcome: str = "not_required"


@dataclass
class _TaskRun:
    task_id: str
    handle: Any
    context: contextvars.Context
    started_ns: int
    start_fields: dict[str, str]
    model_call_ids: set[str] = field(default_factory=set)
    tool_call_ids: set[tuple[str, str, str]] = field(default_factory=set)
    turn_ids: set[str] = field(default_factory=set)
    retired_turn_ids: frozenset[str] = field(default_factory=frozenset)
    completed_tool_call_ids: set[tuple[str, str, str]] = field(default_factory=set)
    unidentified_tool_calls: int = 0
    retry_count: int = 0


@dataclass
class _MetricsSession:
    session_id: str
    relay_session: relay_runtime.RelaySession
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    model_calls: dict[tuple[str, str], _ModelCall] = field(default_factory=dict)
    tasks: dict[str, _TaskRun] = field(default_factory=dict)
    tool_calls: dict[tuple[str, str, str, str], _ToolCall] = field(default_factory=dict)
    retired_turn_ids: deque[str] = field(default_factory=lambda: deque(maxlen=256))


class _Runtime:
    """Own shared-metrics state layered on the Hermes core Relay host."""

    def __init__(self, host: relay_runtime.RelayRuntime | None = None) -> None:
        resolved_host = host or relay_runtime.get_runtime()
        if resolved_host is None:
            raise RuntimeError("Hermes core Relay runtime is unavailable")
        self.host: relay_runtime.RelayRuntime = resolved_host
        self.relay = self.host.relay
        self._active = True
        self._sessions: dict[str, _MetricsSession] = {}
        self._task_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._turn_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._sessions_lock = threading.RLock()
        self._task_creation_lock = threading.RLock()
        self._task_sessions_lock = threading.RLock()
        # Guards the opt-in send pass: at most one in flight per process.
        self._send_lock = threading.RLock()
        self._send_thread: threading.Thread | None = None
        self._subscriber_name = f"{SUBSCRIBER_NAME}.{self.host.runtime_id}"
        self.subscriber = SharedMetricsSubscriber(
            SharedMetricsStore(), __version__, runtime_id=self.host.runtime_id
        )
        self.relay.subscribers.register(self._subscriber_name, self.subscriber)
        self.host.retain_managed_execution(self._subscriber_name)
        self._registered = True
        atexit.register(self.shutdown)

    def ensure_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = _text(event, "session_id")
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
                session = _MetricsSession(session_id=session_id, relay_session=relay_session)
                self._sessions[session_id] = session
        with session.lock:
            return None if session.closing else session

    def record_client_active(self, event: dict[str, Any]) -> None:
        """Emit one payload-free activation attempt under the session scope."""
        session = self.ensure_session(event)
        if session is not None:
            self._emit_client_active(session)

    def _emit_client_active(self, session: _MetricsSession) -> None:
        with session.lock:
            if not session.closing:
                self._mark(session, None, contract.CLIENT_ACTIVE_MARK, {})

    def _mark(
        self, session: _MetricsSession, task: _TaskRun | None, name: str, data: dict[str, str]
    ) -> None:
        """Emit one Relay mark under the task scope when given, else the session scope."""
        self._run_scoped(
            session, task, self.relay.scope.event, name,
            handle=_scope_handle(session, task), data=data, metadata=self._event_metadata(),
        )

    def start_task(self, event: dict[str, Any]) -> _TaskRun | None:
        """Open one Relay function scope for a Hermes task run."""
        task_key = _session_pair(event, "task_id")
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
                    if task is not None and not self._admits(owner, task, event):
                        return None
                    return task

            session = self.ensure_session(event)
            if session is None:
                return None
            with session.lock:
                turn_id = _text(event, "turn_id")
                if (
                    session.closing
                    or (turn_id and turn_id in session.retired_turn_ids)
                    or session.relay_session.context is None
                ):
                    return None
                self._emit_client_active(session)
                task_context = session.relay_session.context.copy()
                start_fields = contract.task_start_fields(event)
                handle = task_context.run(
                    self._with_scope_stack, self.relay.scope.push,
                    TASK_SCOPE, self.relay.ScopeType.Function,
                    handle=_task_parent_handle(session, task_id), input=start_fields,
                    metadata=self._event_metadata(),
                )
                task = _TaskRun(
                    task_id=task_id,
                    handle=handle,
                    context=task_context,
                    started_ns=monotonic_ns(),
                    start_fields=start_fields,
                    retired_turn_ids=frozenset(session.retired_turn_ids),
                )
                session.tasks[task_id] = task
                with self._task_sessions_lock:
                    self._task_sessions[task_key] = session
                self._remember_turn(session, task, event)
                return task

    def _run_in_task(
        self, task: _TaskRun, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return task.context.copy().run(self._with_scope_stack, callback, *args, **kwargs)

    def _with_scope_stack(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.relay.get_scope_stack()
        return callback(*args, **kwargs)

    def start_model_call(self, event: dict[str, Any]) -> None:
        task_id = _text(event, "task_id")
        session, task = self._task_pair(event, start=True, allow_task_id_fallback=True)
        if task_id and task is None:
            return
        session = session or self.ensure_session(event)
        if session is None:
            return
        request_id = _text(event, "api_request_id")
        if not request_id:
            return
        model_call_key = (task_id, request_id)
        fields = contract.model_call_fields(event)
        with session.lock:
            if session.closing:
                return
            if task is not None and not self._admits(session, task, event, current=True):
                return
            existing = session.model_calls.get(model_call_key)
            if existing is not None:
                existing.fields = fields
                if task is not None:
                    # Every repeated start for one logical request is another physical
                    # attempt. Provider fallback resets Hermes's provider-local retry
                    # ordinal, so ordinal deltas are not a reliable task-level counter.
                    task.retry_count += 1
                return
            if task is not None:
                task.model_call_ids.add(request_id)
                if _retry_ordinal(event) > 0:
                    # A real Hermes retry can advance api_request_id while carrying the
                    # retry ordinal. Count that physical attempt.
                    task.retry_count += 1
            handle = self._run_scoped(
                session, task, self.relay.llm.call, MODEL_CALL_SCOPE, self.relay.LLMRequest({}, {}),
                handle=_scope_handle(session, task), metadata=self._event_metadata(),
                model_name=contract.MODEL_CALL_PROFILE_MODEL,
            )
            session.model_calls[model_call_key] = _ModelCall(handle, task_id, fields)

    def update_model_call(self, event: dict[str, Any], *, finish: bool) -> None:
        """Refresh the located model call's fields from ``event``; ``finish`` closes it.

        ``api_request_error`` retains the latest attempt error without closing the logical
        call; ``post_api_request`` closes it.
        """
        session = self._any_session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            model_call_key = self._existing_model_call_key(session, event)
            model_call = session.model_calls.get(model_call_key) if model_call_key else None
            if model_call is None:
                return
            model_call.fields = contract.model_call_fields(event)
            if finish:
                self._finish_model_call(session, model_call_key)

    def start_tool_call(self, event: dict[str, Any]) -> None:
        """Open one privacy-safe Relay tool lifecycle under its task."""
        task_id = _text(event, "task_id")
        session, task = self._task_pair(event, start=True, allow_task_id_fallback=True)
        if session is None or task is None or not _text(event, "tool_call_id"):
            return
        identity = self._tool_call_identity(event)
        with session.lock:
            if not self._admits(session, task, event):
                return
            key = (task_id, *identity)
            if identity in task.completed_tool_call_ids or key in session.tool_calls:
                return
            task.tool_call_ids.add(identity)
            session.tool_calls[key] = self._open_tool_call(task, event)

    def record_approval(self, event: dict[str, Any]) -> None:
        """Record one bounded approval result without approval text or commands."""
        session, task = self._approval_task(event)
        if session is None or task is None:
            return
        outcome = contract.tool_approval_outcome(event)
        attribution = "unattributed"
        with session.lock:
            if session.closing or not self._event_matches_task_turn(task, event):
                return
            if _text(event, "tool_call_id"):
                identity = self._tool_call_identity(event)
                tool_call = session.tool_calls.get((task.task_id, *identity))
                if tool_call is None:
                    key = _sole(_compatible_tool_call_keys(session, task.task_id, identity))
                    tool_call = session.tool_calls[key] if key is not None else None
                if tool_call is not None:
                    tool_call.approval_outcome = outcome
                    attribution = "tool_call"
            self._mark(
                session, task, contract.TOOL_APPROVAL_MARK,
                {"attribution": attribution, "outcome": outcome},
            )

    def record_tool_call(self, event: dict[str, Any]) -> None:
        """Close and count one unique privacy-safe tool lifecycle."""
        task_id = _text(event, "task_id")
        session, task = self._task_pair(event, allow_task_id_fallback=True)
        if session is None or task is None:
            return
        with session.lock:
            if not self._admits(session, task, event):
                return
            tool_call = None
            if _text(event, "tool_call_id"):
                observed_identity = self._tool_call_identity(event)
                if observed_identity in task.completed_tool_call_ids:
                    return
                identity = observed_identity
                tool_call = session.tool_calls.pop((task_id, *identity), None)
                if tool_call is None:
                    if any(
                        _identities_compatible(completed, observed_identity)
                        for completed in task.completed_tool_call_ids
                    ):
                        return
                    matching_keys = _compatible_tool_call_keys(session, task_id, observed_identity)
                    if len(matching_keys) > 1:
                        # Partial context cannot safely choose between concurrent calls
                        # that reused the provider-local ID.
                        return
                    if matching_keys:
                        identity = matching_keys[0][1:]
                        tool_call = session.tool_calls.pop(matching_keys[0])
                task.completed_tool_call_ids.update({identity, observed_identity})
                task.tool_call_ids.add(identity)
            else:
                task.unidentified_tool_calls += 1
            if tool_call is None:
                tool_call = self._open_tool_call(task, event)
            self._finish_tool_call(task, tool_call, event)

    def record_skill_lifecycle(self, event: dict[str, Any]) -> None:
        """Emit one allowlisted skill fact without its local identity."""
        if _text(event, "action").strip().lower() == "loaded":
            mark, fields = contract.SKILL_LOAD_MARK, contract.skill_load_fields(event)
        else:
            mark, fields = contract.SKILL_LIFECYCLE_MARK, contract.skill_lifecycle_fields(event)
        if fields is None:
            return

        session_id, task_id = _text(event, "session_id"), _text(event, "task_id")
        session, task = self._task_pair(event, allow_task_id_fallback=not session_id)
        if session is None:
            if not (session_id and task_id):
                # No owning task: a bare process-level mark.
                self._with_scope_stack(
                    self.relay.scope.event, mark, data=fields, metadata=self._event_metadata()
                )
            return
        if task is None:
            return
        with session.lock:
            if (
                not session.closing
                and session.tasks.get(task.task_id) is task
                and self._event_matches_task_turn(task, event)
            ):
                self._mark(session, task, mark, fields)

    def finish_task(self, event: dict[str, Any]) -> None:
        """Close one task scope exactly once with bounded terminal fields."""
        session = self._any_session(event)
        if session is None:
            return
        with session.lock:
            finished = not session.closing and self._finish_task(
                session, _text(event, "task_id"), event
            )
        if finished:
            self._flush_and_export("Hermes shared-metrics task flush failed")

    def close_session(self, event: dict[str, Any]) -> None:
        session = self._session(event)
        if session is None:
            return
        if not self._abort_session(
            session, {**event, **_ABORTED, "completed": False, "interrupted": False}
        ):
            return
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            logger.warning(
                "Hermes shared-metrics session %s closed with errors: subscriber flush failed: %s",
                session.session_id,
                exc,
            )
        else:
            self._export()
        with self._sessions_lock:
            _forget(self._sessions, session.session_id, session)

    def shutdown(self) -> None:
        with self._sessions_lock:
            self._active = False
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if not self._registered:
            return
        self._flush_and_export("Hermes shared-metrics shutdown flush failed")
        self._deregister()
        self._release()

    def _deregister(self) -> None:
        self._safe(self.relay.subscribers.deregister, self._subscriber_name)
        self.host.release_managed_execution(self._subscriber_name)
        self._registered = False

    def deactivate(self) -> None:
        """Stop collection without exporting locally aggregated metrics."""
        with self._sessions_lock:
            self._active = False
        self.subscriber.deactivate()
        if self._registered:
            self._deregister()
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            self._abort_session(session, {"session_id": session.session_id, **_ABORTED})
        with self._sessions_lock:
            self._sessions.clear()
        with self._task_sessions_lock:
            self._task_sessions.clear()
            self._turn_sessions.clear()
        self._release()

    def _release(self) -> None:
        """Let an in-flight send finish briefly, then drop the atexit hook.

        A short-lived CLI process would otherwise exit and kill the daemon send thread
        mid-request — the common case for this feature's one cadence.
        """
        self._join_send_thread()
        with contextlib.suppress(Exception):
            atexit.unregister(self.shutdown)

    def _join_send_thread(self, timeout: float = 2.0) -> None:
        """Bounded on purpose: pending packages stay in SQLite and go out next run, so
        blocking on a slow network is the wrong trade; the daemon thread dies with the process."""
        with self._send_lock:
            thread = self._send_thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout)
            except Exception:
                logger.debug("Shared-metrics send thread join failed", exc_info=True)

    def _session(self, event: dict[str, Any]) -> _MetricsSession | None:
        with self._sessions_lock:
            return self._sessions.get(_text(event, "session_id"))

    def _any_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        """Owner session by task/turn correlation, else by session_id."""
        return self._task_session(event, allow_task_id_fallback=True) or self._session(event)

    def _task_pair(
        self, event: dict[str, Any], *, start: bool = False, **lookup: Any
    ) -> tuple[_MetricsSession | None, _TaskRun | None]:
        """Resolve (session, task) for a task-scoped hook; ``start`` opens a missing task."""
        session = self._task_session(event, **lookup)
        task = session.tasks.get(_text(event, "task_id")) if session is not None else None
        if task is None and start:
            task = self.start_task(event)
            session = self._task_session(event) if task is not None else None
        return session, task

    def _run_scoped(
        self, session: _MetricsSession, task: _TaskRun | None, callback: Callable[..., Any],
        *args: Any, **kwargs: Any,
    ) -> Any:
        """Run under the task context when the call belongs to a task, else the session."""
        if task is not None:
            return self._run_in_task(task, callback, *args, **kwargs)
        return self.host.run_in_session(session.relay_session, callback, *args, **kwargs)

    def _flush_and_export(self, failure_message: str) -> None:
        """Flush the Relay subscriber, then export; a failed flush skips the export."""
        try:
            self.relay.subscribers.flush()
        except Exception:
            logger.warning(failure_message, exc_info=True)
        else:
            self._export()

    def _abort_session(self, session: _MetricsSession, base_event: dict[str, Any]) -> bool:
        """Mark the session closing and system-abort its open tasks; False if already closing."""
        with session.lock:
            if session.closing:
                return False
            session.closing = True
            for task_id in list(session.tasks):
                self._finish_task(session, task_id, {**base_event, "task_id": task_id})
            self._end_pending_model_calls(session, base_event)
        return True

    def _task_session(
        self, event: dict[str, Any], *, allow_task_id_fallback: bool = False
    ) -> _MetricsSession | None:
        """Owner session by (session, turn), then (session, task), then unique task_id."""
        session_id, task_id = _text(event, "session_id"), _text(event, "task_id")
        if not task_id:
            return None
        with self._task_sessions_lock:
            owner = self._turn_sessions.get(_session_pair(event, "turn_id"))
            if owner is None and session_id:
                owner = self._task_sessions.get((session_id, task_id))
            if owner is not None or not allow_task_id_fallback:
                return owner
            return _sole(
                session for (_, tid), session in self._task_sessions.items() if tid == task_id
            )

    def _remember_turn(
        self, session: _MetricsSession, task: _TaskRun, event: dict[str, Any]
    ) -> None:
        turn_id = _text(event, "turn_id")
        if turn_id:
            task.turn_ids.add(turn_id)
            with self._task_sessions_lock:
                self._turn_sessions[(session.session_id, turn_id)] = session

    @staticmethod
    def _tool_call_identity(event: dict[str, Any]) -> tuple[str, str, str]:
        """Identify one provider-local tool call without exporting its IDs."""
        return _text(event, "api_request_id"), _text(event, "turn_id"), _text(event, "tool_call_id")

    @staticmethod
    def _event_matches_task_turn(task: _TaskRun, event: dict[str, Any]) -> bool:
        """Reject delayed hooks from a prior run that reused the task ID."""
        turn_id = _text(event, "turn_id")
        if not turn_id:
            return True
        return turn_id not in task.retired_turn_ids and (
            not task.turn_ids or turn_id in task.turn_ids
        )

    def _admits(
        self,
        session: _MetricsSession,
        task: _TaskRun,
        event: dict[str, Any],
        *,
        current: bool = False,
    ) -> bool:
        """Whether ``event`` may act on ``task`` (caller holds ``session.lock``).

        Rejects closing sessions and stale turns; with ``current`` also requires ``task`` to
        still be the session's live run for its ID. Admitted events have their turn remembered.
        """
        if (
            session.closing
            or not self._event_matches_task_turn(task, event)
            or (current and session.tasks.get(task.task_id) is not task)
        ):
            return False
        self._remember_turn(session, task, event)
        return True

    def _approval_task(
        self, event: dict[str, Any]
    ) -> tuple[_MetricsSession | None, _TaskRun | None]:
        """Resolve approval correlation without guessing across ambiguous turns."""
        active = relay_runtime.active_turn()
        if active is not None:
            session, task = self._task_pair(
                {**event, "session_id": active.lease.session_id, "task_id": active.task_id}
            )
            if task is not None:
                return session, task

        session, task = self._task_pair(event)
        if task is not None:
            return session, task

        turn_id = _text(event, "turn_id")
        session = None
        if turn_id:
            with self._task_sessions_lock:
                session = _sole(
                    candidate
                    for (owner_id, candidate_turn_id), candidate in self._turn_sessions.items()
                    if candidate_turn_id == turn_id and self._sessions.get(owner_id) is candidate
                )
        if session is None:
            return None, None
        task = _sole(task for task in session.tasks.values() if turn_id in task.turn_ids)
        return (None, None) if task is None else (session, task)

    def _open_tool_call(self, task: _TaskRun, event: dict[str, Any]) -> _ToolCall:
        handle = self._run_in_task(
            task, self.relay.tools.call, contract.TOOL_CALL_SCOPE, {},
            handle=task.handle, metadata=self._event_metadata(),
        )
        return _ToolCall(handle, contract.tool_category(event), monotonic_ns())

    def _finish_tool_call(
        self, task: _TaskRun, tool_call: _ToolCall, event: dict[str, Any]
    ) -> None:
        fields = contract.tool_terminal_fields(
            event, category=tool_call.category, approval_outcome=tool_call.approval_outcome,
            fallback_duration_ms=_elapsed_ms(tool_call.started_ns),
        )
        self._guarded(
            "Hermes shared-metrics tool call close failed",
            lambda: self._run_in_task(
                task, self.relay.tools.call_end, tool_call.handle,
                self.relay.ToolExecutionResult(fields),
                metadata=self._event_metadata(),
            ),
        )

    def _end_pending_tool_calls(
        self, session: _MetricsSession, task: _TaskRun, event: dict[str, Any]
    ) -> None:
        task_outcome, _, _ = contract.task_terminal_state(event)
        status = {"cancelled": "cancelled", "timed_out": "timeout"}.get(task_outcome, "error")
        for key in [key for key in session.tool_calls if key[0] == task.task_id]:
            self._finish_tool_call(task, session.tool_calls.pop(key), {**event, "status": status})

    def _finish_model_call(self, session: _MetricsSession, model_call_key: tuple[str, str]) -> None:
        model_call = session.model_calls.pop(model_call_key, None)
        if model_call is None:
            return
        self._guarded(
            "Hermes shared-metrics model call close failed",
            self._run_scoped, session, session.tasks.get(model_call.task_id),
            self.relay.llm.call_end, model_call.handle, model_call.fields,
            metadata=self._event_metadata(),
        )

    def _end_pending_model_calls(self, session: _MetricsSession, event: dict[str, Any]) -> None:
        task_id = _text(event, "task_id")
        pending = [k for k, c in session.model_calls.items() if not task_id or c.task_id == task_id]
        for key in pending:
            self._finish_model_call(session, key)

    @staticmethod
    def _existing_model_call_key(
        session: _MetricsSession, event: dict[str, Any]
    ) -> tuple[str, str] | None:
        """(task_id, request_id) of an open call; a task-less event may match by request alone."""
        request_id = _text(event, "api_request_id")
        if not request_id:
            return None
        key = (_text(event, "task_id"), request_id)
        if key in session.model_calls or key[0]:
            return key if key in session.model_calls else None
        candidates = [candidate for candidate in session.model_calls if candidate[1] == request_id]
        return candidates[0] if len(candidates) == 1 else None

    def _finish_task(self, session: _MetricsSession, task_id: str, event: dict[str, Any]) -> bool:
        task = session.tasks.get(task_id)
        if task is None:
            return False
        self._end_pending_tool_calls(session, task, event)
        self._end_pending_model_calls(session, {**event, "task_id": task_id})
        fields = contract.task_terminal_fields(
            {**task.start_fields, **event},
            duration_ms=_elapsed_ms(task.started_ns),
            model_call_count=len(task.model_call_ids),
            tool_call_count=len(task.tool_call_ids) + task.unidentified_tool_calls,
            retry_count=task.retry_count,
        )
        try:
            self._guarded(
                "Hermes shared-metrics task close failed",
                self._run_in_task, task, relay_runtime.pop_relay_scope, self.relay, task.handle,
                output=fields, metadata=self._event_metadata(),
            )
        finally:
            session.tasks.pop(task_id, None)
            session.retired_turn_ids.extend(task.turn_ids)
            with self._task_sessions_lock:
                _forget(self._task_sessions, (session.session_id, task_id), session)
                for turn_id in task.turn_ids:
                    _forget(self._turn_sessions, (session.session_id, turn_id), session)
        return True

    def _export(self) -> None:
        exported = self._safe(self.subscriber.store.create_and_export_package_if_due)
        # Sending must never delay the caller: _export runs on finish_task, the user's
        # interactive path. The thread is about latency, not correctness.
        if exported is not None:
            self._safe(self._send_exported_packages)

    def _send_exported_packages(self) -> None:
        try:
            resolved = _resolved_send_config()
        except Exception:
            logger.debug("Unable to read shared-metrics send policy", exc_info=True)
            return

        # Observe the consent EDGE before deciding whether to send: the dominant revocation
        # case is "sending turned off while no pass is running", invisible to the send loop.
        # Failures never break the export hook but log at warning (privacy-relevant).
        self._guarded(
            "Unable to record a shared-metrics consent transition",
            _reconcile_store_consent, self.subscriber.store, resolved.send,
        )
        if not resolved.send:
            return

        with self._send_lock:
            # One in-flight pass per process; the next hook fire picks up what is pending.
            if self._send_thread is None or not self._send_thread.is_alive():
                self._send_thread = threading.Thread(
                    target=self._run_send_pass, args=(resolved.endpoint,),
                    name="hermes-shared-metrics-send", daemon=True,
                )
                self._send_thread.start()

    def _run_send_pass(self, endpoint: str) -> None:
        from hermes_cli.observability.shared_metrics_sender import SharedMetricsSender

        def still_consented() -> bool:
            """Re-read consent so revoking `send` stops an in-flight pass."""
            resolved = _resolved_send_config()
            return resolved.send and resolved.endpoint == endpoint

        sender = SharedMetricsSender(self.subscriber.store, endpoint, consent_check=still_consented)
        self._guarded("Shared-metrics send pass failed", sender.send_pending)

    def _event_metadata(self) -> dict[str, str]:
        return {
            contract.SCHEMA_KEY: contract.SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: self.host.runtime_id,
        }

    @staticmethod
    def _guarded(message: str, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``callback``; log-and-swallow any exception, returning None."""
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning(message, exc_info=True)
            return None

    @classmethod
    def _safe(cls, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return cls._guarded("Hermes shared metrics operation failed", callback, *args, **kwargs)


def _raw_config() -> dict[str, Any]:
    """Read-only config snapshot (lazy import: tests patch ``hermes_cli.config``).

    Collection consent is profile-owned: managed overlays cannot opt a profile in or out.
    The read-only path matters because this gate runs 2-3x per agent turn and the mutable
    read_raw_config() paid a full config deepcopy on every call.
    """
    from hermes_cli.config import read_raw_config_readonly

    return read_raw_config_readonly() or {}


def _resolved_send_config():
    """Resolve the opt-in send policy from the read-only config snapshot."""
    from hermes_cli.observability.shared_metrics_send_config import resolve_send_config

    return resolve_send_config(_raw_config())


def _reconcile_store_consent(store: SharedMetricsStore, send_enabled: bool) -> None:
    from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
    from hermes_cli.sqlite_util import write_txn

    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, send_enabled)


def enabled() -> bool:
    """Return the shared-metrics policy for the active Hermes profile."""
    profile_key = relay_runtime.current_profile_key()
    try:
        config: Any = _raw_config()
    except Exception:
        logger.debug("Unable to read Hermes shared-metrics policy", exc_info=True)
        config = None
    for key in ("telemetry", "shared_metrics"):
        config = config.get(key) if isinstance(config, dict) else None
    if isinstance(config, dict) and config.get("enabled") is True:
        return True
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.pop(profile_key, None)
        if isinstance(runtime, _Runtime):
            runtime.deactivate()
    return False


def handles_hook(hook_name: str) -> bool:
    return hook_name in HANDLED_HOOKS and enabled()


_consent_reconcile_done = False


def _reconcile_send_consent_once() -> None:
    """Reconcile consent windows with config, once per process.

    Runs BEFORE and INDEPENDENT of the collection gate, so a user with ``enabled: false``
    still gets send-consent windows reconciled. Skipped only when there is no store on disk
    AND consent is off: nothing to protect, and creating ``~/.hermes/telemetry`` for every
    fully-disabled user would be the wrong behaviour change.
    """
    global _consent_reconcile_done
    if _consent_reconcile_done:
        return
    _consent_reconcile_done = True
    try:
        # Lazy: tests patch ``shared_metrics.SharedMetricsStore`` at its origin.
        from hermes_cli.observability.shared_metrics import SharedMetricsStore
        from hermes_constants import get_hermes_home

        resolved = _resolved_send_config()
        # Probe WITHOUT constructing a store: the constructor creates the directory and
        # schema as a side effect, which would make the skip below dead code.
        default_path = get_hermes_home() / "telemetry" / "shared_metrics" / "metrics.sqlite3"
        if not resolved.send and not default_path.exists():
            return
        _reconcile_store_consent(SharedMetricsStore(), resolved.send)
    except Exception:
        logger.warning("Unable to reconcile shared-metrics send consent", exc_info=True)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Project one Hermes lifecycle event into the core Relay integration."""
    _reconcile_send_consent_once()
    if not handles_hook(hook_name) or not relay_runtime.relay_instrumentation_enabled():
        return
    runtime = _get_runtime()
    if runtime is None:
        return
    try:
        _HOOK_HANDLERS[hook_name](runtime, kwargs)
    except Exception:
        logger.warning("Hermes shared metrics hook failed: %s", hook_name, exc_info=True)


def _with_runtime_toolset(event: dict[str, Any]) -> dict[str, Any]:
    """Attach the toolset already declared by Hermes's runtime registry."""
    tool_name = _text(event, "tool_name")
    if event.get("toolset") or not tool_name:
        return event
    try:
        from model_tools import get_toolset_for_tool

        toolset = get_toolset_for_tool(tool_name)
    except Exception:
        toolset = None
    return {**event, "toolset": toolset or "other"}


def _close_child_session(runtime: _Runtime, kwargs: dict[str, Any]) -> None:
    child_session_id = _text(kwargs, "child_session_id")
    if child_session_id:
        runtime.close_session({"session_id": child_session_id})


_HOOK_HANDLERS: dict[str, Callable[[_Runtime, dict[str, Any]], Any]] = {
    "on_session_start": lambda rt, kw: rt.record_client_active(kw),
    "pre_llm_call": lambda rt, kw: rt.start_task(kw),
    "pre_api_request": lambda rt, kw: rt.start_model_call(kw),
    "pre_tool_call": lambda rt, kw: rt.start_tool_call(_with_runtime_toolset(kw)),
    "post_tool_call": lambda rt, kw: rt.record_tool_call(_with_runtime_toolset(kw)),
    "post_approval_response": lambda rt, kw: rt.record_approval(kw),
    "on_skill_lifecycle": lambda rt, kw: rt.record_skill_lifecycle(kw),
    "post_api_request": lambda rt, kw: rt.update_model_call(kw, finish=True),
    "api_request_error": lambda rt, kw: rt.update_model_call(kw, finish=False),
    "on_session_end": lambda rt, kw: rt.finish_task(kw),
    "subagent_stop": _close_child_session,
    "on_session_finalize": lambda rt, kw: rt.close_session(kw),
    "on_session_reset": lambda rt, kw: rt.close_session(kw),
}
HANDLED_HOOKS = frozenset(_HOOK_HANDLERS)


def _prepare_core_session(host: relay_runtime.RelayRuntime, context: dict[str, Any]) -> None:
    """Prepare the profile subscriber before the coordinator opens a scope."""
    del context
    if host.profile_key == relay_runtime.current_profile_key() and enabled():
        _get_runtime(retry_failed=True, host=host)


def start_task_run(
    *, session_id: str, task_id: str, platform: str, parent_session_id: str = ""
) -> None:
    """Start task metrics at the outer Hermes execution boundary."""
    _run_task_hook(
        "start_task", retry_failed=True, session_id=session_id, task_id=task_id,
        platform=platform, parent_session_id=parent_session_id,
    )


def finish_task_run(
    *, session_id: str, task_id: str, platform: str,
    result: dict[str, Any] | None = None, error: BaseException | None = None,
) -> None:
    """Finish task metrics for every return or exception path."""
    _run_task_hook(
        "finish_task", session_id=session_id, task_id=task_id, platform=platform,
        **_terminal_flags(result, error),
    )


def _run_task_hook(method: str, *, retry_failed: bool = False, **event: Any) -> None:
    if not enabled():
        return
    runtime = _get_runtime(retry_failed=retry_failed)
    if runtime is not None:
        runtime._safe(getattr(runtime, method), event)


def _terminal_flags(result: dict[str, Any] | None, error: BaseException | None) -> dict[str, Any]:
    """Bounded completed/failed/interrupted/turn_exit_reason for a task's return or raise."""
    if error is not None:
        interrupted = (
            isinstance(error, (KeyboardInterrupt, InterruptedError))
            or type(error).__name__ == "CancelledError"
        )
        if interrupted:
            reason = "interrupted_by_user"
        else:
            reason = "timed_out" if isinstance(error, TimeoutError) else "system_aborted"
        return {
            "completed": False, "failed": not interrupted, "interrupted": interrupted,
            "turn_exit_reason": reason,
        }
    terminal = result if isinstance(result, dict) else {}
    failed = terminal.get("failed") is True
    reason = str(terminal.get("turn_exit_reason") or terminal.get("failure_reason") or "")
    return {
        "completed": terminal.get("completed") is True,
        "failed": failed,
        "interrupted": terminal.get("interrupted") is True,
        "turn_exit_reason": reason or ("failed" if failed else "unknown"),
    }


def _get_runtime(
    *, retry_failed: bool = False, host: relay_runtime.RelayRuntime | None = None
) -> _Runtime | None:
    profile_key = relay_runtime.current_profile_key()
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(profile_key)
        if isinstance(runtime, _Runtime):
            if host is None or runtime.host is host:
                return runtime
            runtime.deactivate()
        elif runtime is _RUNTIME_FAILED and not retry_failed:
            return None
        try:
            _RUNTIMES[profile_key] = runtime = _Runtime(host=host)
        except Exception:
            logger.warning("Hermes shared metrics initialization failed", exc_info=True)
            _RUNTIMES[profile_key] = _RUNTIME_FAILED
            return None
        return runtime


relay_runtime.SESSION_COORDINATOR.register_session_initializer(
    SUBSCRIBER_NAME, _prepare_core_session
)


def _reset_for_tests() -> None:
    """Reset all profile-scoped shared-metrics state for isolated tests."""
    with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        if isinstance(runtime, _Runtime):
            runtime.shutdown()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def prepare_session_start() -> None:
    """Register the subscriber before any producer opens the session scope."""
    if enabled():
        _get_runtime(retry_failed=True)


_PLUGIN_COMPAT_LAZY = {
    'CLIENT_ACTIVE_MARK': ('hermes_cli.observability.shared_metrics_contract', 'CLIENT_ACTIVE_MARK'),
    'MODEL_CALL_PROFILE_MODEL': ('hermes_cli.observability.shared_metrics_contract', 'MODEL_CALL_PROFILE_MODEL'),
    'SCHEMA_KEY': ('hermes_cli.observability.shared_metrics_contract', 'SCHEMA_KEY'),
    'SCHEMA_VERSION': ('hermes_cli.observability.shared_metrics_contract', 'SCHEMA_VERSION'),
    'SKILL_LIFECYCLE_MARK': ('hermes_cli.observability.shared_metrics_contract', 'SKILL_LIFECYCLE_MARK'),
    'SKILL_LOAD_MARK': ('hermes_cli.observability.shared_metrics_contract', 'SKILL_LOAD_MARK'),
    'TOOL_APPROVAL_MARK': ('hermes_cli.observability.shared_metrics_contract', 'TOOL_APPROVAL_MARK'),
    'TOOL_CALL_SCOPE': ('hermes_cli.observability.shared_metrics_contract', 'TOOL_CALL_SCOPE'),
    'model_call_fields': ('hermes_cli.observability.shared_metrics_contract', 'model_call_fields'),
    'skill_lifecycle_fields': ('hermes_cli.observability.shared_metrics_contract', 'skill_lifecycle_fields'),
    'skill_load_fields': ('hermes_cli.observability.shared_metrics_contract', 'skill_load_fields'),
    'task_start_fields': ('hermes_cli.observability.shared_metrics_contract', 'task_start_fields'),
    'task_terminal_fields': ('hermes_cli.observability.shared_metrics_contract', 'task_terminal_fields'),
    'task_terminal_state': ('hermes_cli.observability.shared_metrics_contract', 'task_terminal_state'),
    'tool_approval_outcome': ('hermes_cli.observability.shared_metrics_contract', 'tool_approval_outcome'),
    'tool_category': ('hermes_cli.observability.shared_metrics_contract', 'tool_category'),
    'tool_terminal_fields': ('hermes_cli.observability.shared_metrics_contract', 'tool_terminal_fields'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
