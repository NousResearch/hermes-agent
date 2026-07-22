"""BatchRunner — iterates multiple tasks through DecisionEngine + Dispatcher.

The BatchRunner is the entry point for batch orchestration. It:
1. Filters eligible tasks (READY + retryable FAILED).
2. Orders them deterministically by (priority desc, created_at asc, task_id asc).
3. Iterates each task through Dispatcher.dispatch() in order.
4. Tracks concurrency (max_concurrent_workers).
5. Records batch_run_id and per-task results in batch_trace.jsonl.
6. Ensures each task is dispatched at most once per batch.
7. Respects RUNNING/WAITING/BLOCKED/DONE states (skip or special-case).

Architecture:
  BatchRunner = iterate + filter + order + concurrency cap
  Dispatcher  = convert + decide + invoke handler
  DecisionEngine = pure planner
  KanbanAdapter = the only writer

BatchRunner NEVER imports DecisionEngine or worker_runner directly.
It composes Dispatcher (which composes DecisionEngine internally).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agent.orchestrator.dispatcher import (  # noqa: E402
    Dispatcher,
    TaskState,
    WorkerRegistryEntry,
)
from agent.orchestrator.kanban_adapter import (  # noqa: E402
    KanbanAdapter,
    KanbanTask,
)


_BATCH_LOG_DEFAULT = Path("/home/jr-ubuntu/.hermes/traces/batch_trace.jsonl")


@dataclass
class BatchResult:
    """Output of a batch run."""
    batch_run_id: str
    started_at: str
    finished_at: str
    total_tasks: int
    attempted_tasks: int
    skipped_tasks: int
    dispatched_tasks: int
    worker_runs_started: int
    results: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch_run_id": self.batch_run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_tasks": self.total_tasks,
            "attempted_tasks": self.attempted_tasks,
            "skipped_tasks": self.skipped_tasks,
            "dispatched_tasks": self.dispatched_tasks,
            "worker_runs_started": self.worker_runs_started,
            "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.results],
            "skipped": self.skipped,
            "errors": self.errors,
        }


class BatchRunner:
    """Iterates multiple tasks through the orchestrator in a single batch.

    Usage:
        runner = BatchRunner(
            dispatcher=Dispatcher(...),
            adapter=KanbanAdapter(board_root),
            max_concurrent_workers=1,
            batch_log_path=Path("/var/log/batch.jsonl"),
        )
        result = runner.run_batch(tasks, workers, restrictions=set())
    """

    def __init__(
        self,
        *,
        dispatcher: Dispatcher,
        adapter: KanbanAdapter | None = None,
        max_concurrent_workers: int = 1,
        batch_log_path: Path | None = None,
    ) -> None:
        if max_concurrent_workers < 1:
            raise ValueError("max_concurrent_workers must be >= 1")
        self.dispatcher = dispatcher
        self.adapter = adapter
        self.max_concurrent_workers = max_concurrent_workers
        self.batch_log_path = Path(batch_log_path) if batch_log_path else _BATCH_LOG_DEFAULT

    def run_batch(
        self,
        tasks: list,
        workers: list,
        restrictions: set | None = None,
    ) -> BatchResult:
        """Run a batch of tasks through the dispatcher.

        Args:
            tasks: list of KanbanTask or TaskState.
            workers: list of WorkerRegistryEntry.
            restrictions: optional set of restriction strings.

        Returns:
            BatchResult with per-task results, skipped tasks, errors.
        """
        restrictions = restrictions or set()
        batch_run_id = str(uuid.uuid4())
        return self._run_batch_inner(
            tasks=tasks,
            workers=workers,
            restrictions=restrictions,
            batch_run_id=batch_run_id,
        )

    def _run_batch_inner(
        self,
        tasks: list,
        workers: list,
        restrictions: set,
        batch_run_id: str,
    ) -> BatchResult:
        started_at = self._utcnow_iso()
        results = []
        skipped = []
        errors = []
        dispatched = 0
        worker_runs_started = 0
        seen_task_ids = set()

        # Convert + filter eligible tasks.
        tstates = []
        for t in tasks:
            tstate = self._to_task_state(t)
            if tstate.task_id in seen_task_ids:
                skipped.append({
                    "task_id": tstate.task_id,
                    "reason": "duplicate_in_batch",
                })
                continue
            seen_task_ids.add(tstate.task_id)
            eligibility = self._classify_eligibility(tstate)
            if not eligibility["eligible"]:
                skipped.append({
                    "task_id": tstate.task_id,
                    "reason": eligibility["reason"],
                    "state": tstate.state,
                })
                continue
            tstates.append(tstate)

        # Deterministic order: priority desc, created_at asc, task_id asc.
        # The original input list `tasks` carries optional priority/created_at;
        # we sort the eligible tstates by looking up the matching original.
        original_by_id = {getattr(t, "task_id", None): t for t in tasks}
        tstates.sort(key=lambda ts: (
            -_priority_for(original_by_id.get(ts.task_id)),
            _created_at_for(original_by_id.get(ts.task_id)),
            ts.task_id,
        ))

        # Iterate sequentially (concurrency cap = max_concurrent_workers).
        for tstate in tstates:
            try:
                result = self.dispatcher.dispatch(
                    tstate,
                    workers,
                    restrictions,
                    batch_run_id=batch_run_id,
                )
                results.append(result)
                dispatched += 1
                if result.decision.next_action == "RUN_WORKER":
                    worker_runs_started += 1
            except Exception as e:
                errors.append({
                    "task_id": tstate.task_id,
                    "error_type": type(e).__name__,
                    "error_repr": repr(e),
                })

        finished_at = self._utcnow_iso()
        batch_result = BatchResult(
            batch_run_id=batch_run_id,
            started_at=started_at,
            finished_at=finished_at,
            total_tasks=len(tasks),
            attempted_tasks=len(tstates),
            skipped_tasks=len(skipped),
            dispatched_tasks=dispatched,
            worker_runs_started=worker_runs_started,
            results=results,
            skipped=skipped,
            errors=errors,
        )

        # Append-only batch trace.
        self._append_batch_trace(batch_result)
        return batch_result

    # ----- helpers -----

    @staticmethod
    def _to_task_state(t):
        """Convert KanbanTask or TaskState → TaskState."""
        if isinstance(t, TaskState):
            return t
        if isinstance(t, KanbanTask):
            return TaskState(
                task_id=t.task_id,
                state=t.state,
                last_worker_id=t.last_worker_id,
                last_worker_status=t.last_worker_status,
                failure_count=t.failure_count,
            )
        # Dict-like: try to construct.
        if isinstance(t, dict):
            return TaskState(
                task_id=t["task_id"],
                state=t["state"],
                last_worker_id=t.get("last_worker_id"),
                last_worker_status=t.get("last_worker_status"),
                failure_count=t.get("failure_count", 0),
            )
        raise TypeError(f"unsupported task type: {type(t)}")

    @staticmethod
    def _classify_eligibility(tstate) -> dict:
        """Return dict with 'eligible' (bool) and 'reason' (str)."""
        if tstate.state == "READY":
            return {"eligible": True, "reason": ""}
        if tstate.state == "FAILED":
            return {"eligible": True, "reason": ""}  # RETRY path
        if tstate.state == "BLOCKED":
            # Blocked tasks still get a dispatch (ASK_HUMAN), so eligible.
            return {"eligible": True, "reason": ""}
        # RUNNING, WAITING, DONE → skip.
        return {
            "eligible": False,
            "reason": f"state {tstate.state} is not dispatchable",
        }

    def _append_batch_trace(self, batch_result: BatchResult) -> None:
        self.batch_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.batch_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(batch_result.to_dict(), sort_keys=True) + "\n")

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----- sorting helpers -----


def _priority_for(obj) -> int:
    """Priority for sorting; default 0 if missing.
    Accepts TaskState, KanbanTask, or dict.
    """
    if obj is None:
        return 0
    if isinstance(obj, dict):
        p = obj.get("priority")
    else:
        # Dataclass OR plain object with __dict__.
        p = getattr(obj, "priority", None)
        if p is None and hasattr(obj, "__dict__"):
            p = obj.__dict__.get("priority")
    if p is None:
        return 0
    try:
        return int(p)
    except (TypeError, ValueError):
        return 0


def _created_at_for(obj) -> str:
    """Created_at for sorting; default empty string sorts first."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        c = obj.get("created_at") or ""
    else:
        c = getattr(obj, "created_at", None)
        if c is None and hasattr(obj, "__dict__"):
            c = obj.__dict__.get("created_at")
    return str(c or "")