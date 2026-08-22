"""Real action handlers for the orchestrator (wire Decision to Kanban).

Each handler is the ONLY place that can mutate Kanban state for a given
action. Handlers:

- Take the DispatchResult from the dispatcher.
- Compute the new state dict for the task.
- Invoke the KanbanAdapter.apply_change() with scope enforcement.
- Return a HandlerResult describing what was done.

Handlers NEVER call HTTP / LLM directly. The RUN_WORKER handler invokes
the worker_runner (subprocess execution with hard timeout + kill fallback)
when a command is provided; otherwise it records the intent only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from agent.orchestrator.dispatcher import DispatchResult  # noqa: E402
from agent.orchestrator.kanban_adapter import (  # noqa: E402
    KanbanAdapter,
    ScopeViolation,
)
from agent.orchestrator.worker_runner import (  # noqa: E402
    active_batch_run_id_scope,
    run_worker_subprocess,
)


@dataclass
class HandlerResult:
    """What a handler did."""
    action: str
    task_id: str
    applied: bool
    scope_violation: bool
    new_state: dict | None
    reason: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "task_id": self.task_id,
            "applied": self.applied,
            "scope_violation": self.scope_violation,
            "new_state": self.new_state,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_handlers(adapter: KanbanAdapter) -> dict:
    """Return a dict of action → handler function.

    Handlers are pure wrt Kanban state: they read the current task,
    compute the new state, and call adapter.apply_change() with scope.

    RUN_WORKER and RETRY are intent-only (no real worker spawn in this
    approval): they record the intent and update the task state to
    RUNNING. Real worker spawn is a separate approval.
    """
    def _read_task(task_id: str) -> dict:
        task = adapter.get_task(task_id)
        if task is None:
            return {}
        return {
            "task_id": task.task_id,
            "state": task.state,
            "last_worker_id": task.last_worker_id,
            "last_worker_status": task.last_worker_status,
            "failure_count": task.failure_count,
            "human_input_required": task.human_input_required,
            "requires_human": task.requires_human,
            "retry_count": task.retry_count,
            "stop_reason": task.stop_reason,
            "board": task.board,
            "updated_at": _utcnow_iso(),
        }

    def handle_run_worker(result: DispatchResult) -> HandlerResult:
        decision = result.decision
        worker_id = decision.selected_worker or "unknown"
        current = _read_task(result.task_id)
        # Generate worker_run_id.
        worker_run_id = str(uuid.uuid4())
        # Look up the command from registry (passed via dispatcher).
        command = result.trace_line.get("command") if result.trace_line else None
        # Step 1: mark task RUNNING before spawning.
        new_state = dict(current)
        new_state["state"] = "RUNNING"
        new_state["last_worker_id"] = worker_id
        new_state["last_worker_status"] = None
        new_state["updated_at"] = _utcnow_iso()
        new_state["worker_run_id"] = worker_run_id
        new_state["intent"] = f"RUN_WORKER:{worker_id}"
        if not command:
            # No command registered → mark FAILED.
            new_state["state"] = "FAILED"
            new_state["stop_reason"] = "worker_command_missing"
            try:
                adapter.apply_change(result.task_id, new_state)
                return HandlerResult(
                    action="RUN_WORKER",
                    task_id=result.task_id,
                    applied=True,
                    scope_violation=False,
                    new_state=new_state,
                    reason="worker_command_missing",
                    timestamp=_utcnow_iso(),
                )
            except ScopeViolation as e:
                return HandlerResult(
                    action="RUN_WORKER",
                    task_id=result.task_id,
                    applied=False,
                    scope_violation=True,
                    new_state=None,
                    reason=str(e),
                    timestamp=_utcnow_iso(),
                )
        try:
            adapter.apply_change(result.task_id, new_state)
        except ScopeViolation as e:
            return HandlerResult(
                action="RUN_WORKER",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )
        # Step 2: spawn the subprocess (REAL execution).
        import agent.orchestrator.worker_runner as _wr
        batch_run_id = result.trace_line.get("batch_run_id") if result.trace_line else None
        with active_batch_run_id_scope(batch_run_id):
            worker_result = run_worker_subprocess(
                worker_id=worker_id,
                task_id=result.task_id,
                command=command,
                timeout_s=60,
                worker_log_root=_wr.WORKER_LOG_ROOT,
                worker_run_log=_wr.WORKER_RUN_LOG,
            )
        # Step 3: update task state based on worker outcome.
        post_state = dict(new_state)
        post_state["updated_at"] = _utcnow_iso()
        post_state["worker_run_id"] = worker_run_id
        post_state["worker_exitcode"] = worker_result.exitcode
        post_state["worker_timed_out"] = worker_result.timed_out
        post_state["worker_killed"] = worker_result.killed
        post_state["worker_latency_ms"] = worker_result.latency_ms
        post_state["worker_stdout_path"] = worker_result.stdout_path
        post_state["worker_stderr_path"] = worker_result.stderr_path
        if worker_result.timed_out or worker_result.killed:
            post_state["state"] = "FAILED"
            post_state["stop_reason"] = "worker_timeout"
            post_state["last_worker_status"] = "timeout"
        elif worker_result.error_type == "worker_command_not_found":
            post_state["state"] = "FAILED"
            post_state["stop_reason"] = "worker_command_not_found"
            post_state["last_worker_status"] = "failure"
        elif worker_result.exitcode == 0:
            post_state["state"] = "DONE"
            post_state["last_worker_status"] = "success"
        else:
            post_state["state"] = "FAILED"
            post_state["stop_reason"] = "worker_failed"
            post_state["last_worker_status"] = "failure"
        try:
            adapter.apply_change(result.task_id, post_state)
            return HandlerResult(
                action="RUN_WORKER",
                task_id=result.task_id,
                applied=True,
                scope_violation=False,
                new_state=post_state,
                reason=(
                    f"worker_exitcode={worker_result.exitcode} "
                    f"timed_out={worker_result.timed_out} killed={worker_result.killed}"
                ),
                timestamp=_utcnow_iso(),
            )
        except ScopeViolation as e:
            return HandlerResult(
                action="RUN_WORKER",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )

    def handle_wait(result: DispatchResult) -> HandlerResult:
        # No-op; just record trace.
        return HandlerResult(
            action="WAIT",
            task_id=result.task_id,
            applied=True,
            scope_violation=False,
            new_state=None,
            reason="no-op; wait one tick",
            timestamp=_utcnow_iso(),
        )

    def handle_ask_human(result: DispatchResult) -> HandlerResult:
        current = _read_task(result.task_id)
        new_state = dict(current)
        new_state["state"] = "BLOCKED"
        new_state["human_input_required"] = True
        new_state["requires_human"] = True
        new_state["updated_at"] = _utcnow_iso()
        try:
            adapter.apply_change(result.task_id, new_state)
            return HandlerResult(
                action="ASK_HUMAN",
                task_id=result.task_id,
                applied=True,
                scope_violation=False,
                new_state=new_state,
                reason="task marked BLOCKED + requires_human=True",
                timestamp=_utcnow_iso(),
            )
        except ScopeViolation as e:
            return HandlerResult(
                action="ASK_HUMAN",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )

    def handle_retry(result: DispatchResult) -> HandlerResult:
        decision = result.decision
        current = _read_task(result.task_id)
        new_state = dict(current)
        new_retry_count = current.get("retry_count", 0) + 1
        # Cap at 3.
        if new_retry_count > 3:
            new_state["state"] = "FAILED"
            new_state["stop_reason"] = "retry_cap_exhausted"
        else:
            new_state["state"] = "READY"
            new_state["retry_count"] = new_retry_count
        new_state["last_worker_status"] = "retry"
        new_state["updated_at"] = _utcnow_iso()
        try:
            adapter.apply_change(result.task_id, new_state)
            return HandlerResult(
                action="RETRY",
                task_id=result.task_id,
                applied=True,
                scope_violation=False,
                new_state=new_state,
                reason=f"retry_count={new_state.get('retry_count')}",
                timestamp=_utcnow_iso(),
            )
        except ScopeViolation as e:
            return HandlerResult(
                action="RETRY",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )

    def handle_finish(result: DispatchResult) -> HandlerResult:
        current = _read_task(result.task_id)
        new_state = dict(current)
        new_state["state"] = "DONE"
        new_state["updated_at"] = _utcnow_iso()
        try:
            adapter.apply_change(result.task_id, new_state)
            return HandlerResult(
                action="FINISH",
                task_id=result.task_id,
                applied=True,
                scope_violation=False,
                new_state=new_state,
                reason="task marked DONE",
                timestamp=_utcnow_iso(),
            )
        except ScopeViolation as e:
            return HandlerResult(
                action="FINISH",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )

    def handle_stop(result: DispatchResult) -> HandlerResult:
        decision = result.decision
        current = _read_task(result.task_id)
        new_state = dict(current)
        new_state["state"] = "FAILED"
        new_state["stop_reason"] = decision.stop_reason or "stopped"
        new_state["updated_at"] = _utcnow_iso()
        try:
            adapter.apply_change(result.task_id, new_state)
            return HandlerResult(
                action="STOP",
                task_id=result.task_id,
                applied=True,
                scope_violation=False,
                new_state=new_state,
                reason=f"stopped: {new_state['stop_reason']}",
                timestamp=_utcnow_iso(),
            )
        except ScopeViolation as e:
            return HandlerResult(
                action="STOP",
                task_id=result.task_id,
                applied=False,
                scope_violation=True,
                new_state=None,
                reason=str(e),
                timestamp=_utcnow_iso(),
            )

    return {
        "RUN_WORKER": handle_run_worker,
        "WAIT": handle_wait,
        "ASK_HUMAN": handle_ask_human,
        "RETRY": handle_retry,
        "FINISH": handle_finish,
        "STOP": handle_stop,
    }