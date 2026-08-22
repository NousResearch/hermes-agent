"""Orchestrator Dispatcher (executor) — wires DecisionEngine to actual state.

Architecture (separation of concerns):
  DecisionEngine = decides (pure planner, no side effects).
  Dispatcher     = executes (this module: converts state → OrchestratorState,
                              converts workers → WorkerCapability, invokes
                              engine.plan(), records the Decision, and only
                              IF the action is RUN_WORKER / RETRY / ASK_HUMAN
                              / FINISH / STOP / WAIT invokes the registered
                              handler).
  Kanban/CLI     = persistence + effects.

This module is the ONLY place that translates raw state into the
OrchestratorState and WorkerCapability types. The DecisionEngine never
sees raw state.

Decisions are appended to a trace log (append-only JSONL). Workers are
NOT actually invoked here — handlers are stubs that record what would
happen. Real handlers should be wired in by callers (CLI / Kanban).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent._orchestrator_decision_engine import (  # noqa: E402
    Decision,
    DecisionEngine,
    OrchestratorState,
    WorkerCapability,
)


_VALID_STATES = {"READY", "RUNNING", "WAITING", "BLOCKED", "FAILED", "DONE"}


@dataclass
class WorkerRegistryEntry:
    """A worker as registered in the orchestrator (raw state)."""
    worker_id: str
    worker_kind: str = "default"
    requires_http: bool = False
    requires_llm: bool = False
    mutates_state: bool = False
    spawns_subproc: bool = False
    is_retryable: bool = False
    recovery_kind: str | None = None
    handles_states: list = field(default_factory=lambda: ["READY"])
    command: list | None = None  # command to run for real worker spawn


@dataclass
class TaskState:
    """A task as registered in the orchestrator (raw state)."""
    task_id: str
    state: str
    last_worker_id: str | None = None
    last_worker_status: str | None = None
    failure_count: int = 0
    human_input_required: bool = False

    def __post_init__(self):
        if self.state not in _VALID_STATES:
            raise ValueError(f"task {self.task_id}: invalid state {self.state!r}")


def to_orchestrator_state(task: TaskState) -> OrchestratorState:
    """Convert raw TaskState → OrchestratorState for the engine."""
    return OrchestratorState(
        state=task.state,
        last_worker_id=task.last_worker_id,
        last_worker_status=task.last_worker_status,
        failure_count=task.failure_count,
        human_input_required=task.human_input_required,
    )


def to_worker_capability(entry: WorkerRegistryEntry) -> WorkerCapability:
    """Convert raw WorkerRegistryEntry → WorkerCapability for the engine."""
    return WorkerCapability(
        worker_id=entry.worker_id,
        handles_states=entry.handles_states,
        requires_http=entry.requires_http,
        requires_llm=entry.requires_llm,
        mutates_state=entry.mutates_state,
        spawns_subproc=entry.spawns_subproc,
        is_retryable=entry.is_retryable,
        recovery_kind=entry.recovery_kind,
    )


@dataclass
class DispatchResult:
    """Result of one dispatch tick."""
    decision: Decision
    task_id: str
    worker_id: str | None
    action_executed: str  # what handler was actually invoked
    timestamp: str
    trace_line: dict

    def to_dict(self) -> dict:
        return {
            "decision": {
                "next_action": self.decision.next_action,
                "selected_worker": self.decision.selected_worker,
                "rationale": self.decision.rationale,
                "confidence": self.decision.confidence,
                "stop_reason": self.decision.stop_reason,
                "discarded_workers": self.decision.discarded_workers,
            },
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "action_executed": self.action_executed,
            "timestamp": self.timestamp,
            "trace_line": self.trace_line,
        }


class Dispatcher:
    """Thin dispatcher that converts state, calls DecisionEngine, records
    the Decision, and invokes the appropriate action handler.

    Action handlers are simple callables that receive the DispatchResult.
    They are NOT workers — they describe what would happen if the
    Decision were actually executed.
    """

    def __init__(
        self,
        *,
        engine: DecisionEngine | None = None,
        trace_log_path: Path | None = None,
        action_handlers: dict | None = None,
    ) -> None:
        self.engine = engine or DecisionEngine()
        self.trace_log_path = Path(trace_log_path) if trace_log_path else None
        self.action_handlers = action_handlers or self._default_handlers()

    @staticmethod
    def _default_handlers() -> dict:
        """Default action handlers — all stubs that record what would happen."""
        def _stub(action_name: str) -> Callable:
            def _handler(result: DispatchResult) -> None:
                # Stub: no actual execution. Just records the decision.
                return None
            _handler.__name__ = f"_stub_{action_name}"
            return _handler
        return {
            "RUN_WORKER": _stub("RUN_WORKER"),
            "WAIT": _stub("WAIT"),
            "ASK_HUMAN": _stub("ASK_HUMAN"),
            "RETRY": _stub("RETRY"),
            "FINISH": _stub("FINISH"),
            "STOP": _stub("STOP"),
        }

    def dispatch(
        self,
        task: TaskState,
        workers: list,
        restrictions: set | None = None,
        *,
        batch_run_id: str | None = None,
    ) -> DispatchResult:
        """Convert state, invoke engine, record decision, execute handler.

        Args:
            task: the current TaskState.
            workers: list of WorkerRegistryEntry.
            restrictions: optional set of restriction strings.
            batch_run_id: optional batch scope propagated to action handlers.

        Returns:
            DispatchResult with the Decision, action executed, and trace line.
        """
        # Convert raw types to engine types.
        ostate = to_orchestrator_state(task)
        capabilities = [to_worker_capability(w) for w in workers]

        # Pure decision.
        decision = self.engine.plan(ostate, capabilities, restrictions)

        # Execute (via handler).
        handler = self.action_handlers.get(decision.next_action)
        if handler is None:
            action_executed = f"NO_HANDLER_FOR_{decision.next_action}"
            # Build trace line for the log even if no handler.
            trace_line = {
                "trace_id": str(uuid.uuid4()),
                "task_id": task.task_id,
                "timestamp": self._utcnow_iso(),
                "input_state": task.state,
                "input_last_worker_id": task.last_worker_id,
                "input_failure_count": task.failure_count,
                "restrictions": sorted(restrictions) if restrictions else [],
                "decision": {
                    "next_action": decision.next_action,
                    "selected_worker": decision.selected_worker,
                    "rationale": decision.rationale,
                    "confidence": decision.confidence,
                    "stop_reason": decision.stop_reason,
                    "discarded_workers": [
                        {"worker_id": w, "reason": r} for w, r in decision.discarded_workers
                    ],
                },
                "action_executed": action_executed,
                "command": self._find_command_for_worker(decision.selected_worker, workers),
                "batch_run_id": batch_run_id,
            }
        else:
            # Build trace line first so the handler can read it.
            trace_line = {
                "trace_id": str(uuid.uuid4()),
                "task_id": task.task_id,
                "timestamp": self._utcnow_iso(),
                "input_state": task.state,
                "input_last_worker_id": task.last_worker_id,
                "input_failure_count": task.failure_count,
                "restrictions": sorted(restrictions) if restrictions else [],
                "decision": {
                    "next_action": decision.next_action,
                    "selected_worker": decision.selected_worker,
                    "rationale": decision.rationale,
                    "confidence": decision.confidence,
                    "stop_reason": decision.stop_reason,
                    "discarded_workers": [
                        {"worker_id": w, "reason": r} for w, r in decision.discarded_workers
                    ],
                },
                "action_executed": decision.next_action,
                "command": self._find_command_for_worker(decision.selected_worker, workers),
                "batch_run_id": batch_run_id,
            }
            stub_result = DispatchResult(
                decision=decision,
                task_id=task.task_id,
                worker_id=decision.selected_worker,
                action_executed=decision.next_action,
                timestamp=trace_line["timestamp"],
                trace_line=trace_line,
            )
            handler(stub_result)
            action_executed = decision.next_action

        # Build trace line.
        result = DispatchResult(
            decision=decision,
            task_id=task.task_id,
            worker_id=decision.selected_worker,
            action_executed=action_executed,
            timestamp=trace_line["timestamp"],
            trace_line=trace_line,
        )

        # Append-only trace.
        if self.trace_log_path is not None:
            self._append_trace(trace_line)

        return result

    def _append_trace(self, trace_line: dict) -> None:
        assert self.trace_log_path is not None
        self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace_line, sort_keys=True) + "\n")

    @staticmethod
    def _find_command_for_worker(worker_id, workers):
        if worker_id is None:
            return None
        for w in workers:
            if w.worker_id == worker_id:
                return getattr(w, "command", None)
        return None

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")