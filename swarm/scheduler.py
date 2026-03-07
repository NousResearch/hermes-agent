from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import asdict, field
from datetime import datetime
from typing import Any

from swarm.types import SwarmResult, SwarmTask, TaskState


class SwarmScheduler:
    """DAG-based task scheduler with dependency tracking, retries, and priority."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, SwarmTask] = {}
        self._deps: dict[str, set[str]] = defaultdict(set)  # task_id -> set of dependency ids
        self._rdeps: dict[str, set[str]] = defaultdict(set)  # dep_id -> set of dependent task ids

    def add_task(self, task: SwarmTask) -> None:
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"task {task.id} already exists")
            self._tasks[task.id] = task
            for dep_id in task.deps:
                self._deps[task.id].add(dep_id)
                self._rdeps[dep_id].add(task.id)

    def add_dependency(self, task_id: str, depends_on_id: str) -> None:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"task {task_id} not found")
            if depends_on_id not in self._tasks:
                raise KeyError(f"dependency {depends_on_id} not found")
            self._deps[task_id].add(depends_on_id)
            self._rdeps[depends_on_id].add(task_id)
            if depends_on_id not in self._tasks[task_id].deps:
                self._tasks[task_id].deps.append(depends_on_id)

    def get_ready_tasks(self) -> list[SwarmTask]:
        with self._lock:
            ready = []
            for tid, task in self._tasks.items():
                if task.state != TaskState.pending:
                    continue
                deps = self._deps.get(tid, set())
                if all(
                    self._tasks[d].state == TaskState.completed
                    for d in deps
                    if d in self._tasks
                ):
                    ready.append(task)
            ready.sort(key=lambda t: t.created_at)
            return ready

    def mark_running(self, task_id: str, worker_id: str) -> None:
        with self._lock:
            task = self._get(task_id)
            if task.state != TaskState.pending:
                raise ValueError(f"task {task_id} is {task.state.value}, expected pending")
            task.state = TaskState.running
            task.worker_id = worker_id
            task.started_at = datetime.utcnow()

    def mark_completed(self, task_id: str, result: SwarmResult) -> None:
        with self._lock:
            task = self._get(task_id)
            task.state = TaskState.completed
            task.result = result
            task.completed_at = datetime.utcnow()

    def mark_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._get(task_id)
            task.retries += 1
            if task.retries < task.max_retries:
                task.state = TaskState.pending
                task.worker_id = None
                task.started_at = None
            else:
                task.state = TaskState.failed
                task.result = error
                task.completed_at = datetime.utcnow()
                self._cancel_downstream_unlocked(task_id)

    def cancel_downstream(self, task_id: str) -> None:
        with self._lock:
            self._cancel_downstream_unlocked(task_id)

    def _cancel_downstream_unlocked(self, task_id: str) -> None:
        stack = list(self._rdeps.get(task_id, set()))
        while stack:
            tid = stack.pop()
            task = self._tasks.get(tid)
            if task and task.state in (TaskState.pending, TaskState.queued):
                task.state = TaskState.cancelled
                stack.extend(self._rdeps.get(tid, set()))

    def is_complete(self) -> bool:
        with self._lock:
            return all(
                t.state in (TaskState.completed, TaskState.failed, TaskState.cancelled)
                for t in self._tasks.values()
            )

    def get_status(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = defaultdict(int)
            for task in self._tasks.values():
                counts[task.state.value] += 1
            return dict(counts)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            tasks = []
            for t in self._tasks.values():
                d = {
                    "id": t.id,
                    "name": t.name,
                    "prompt": t.prompt,
                    "deps": t.deps,
                    "state": t.state.value,
                    "worker_id": t.worker_id,
                    "retries": t.retries,
                    "max_retries": t.max_retries,
                    "role": t.role,
                    "model": t.model,
                    "created_at": t.created_at.isoformat(),
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                }
                if isinstance(t.result, SwarmResult):
                    d["result"] = asdict(t.result)
                else:
                    d["result"] = t.result
                tasks.append(d)
            return {
                "tasks": tasks,
                "deps": {k: sorted(v) for k, v in self._deps.items()},
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwarmScheduler:
        sched = cls()
        for td in data["tasks"]:
            result_raw = td.get("result")
            result: Any = None
            if isinstance(result_raw, dict) and "task_id" in result_raw:
                result = SwarmResult(**result_raw)
            else:
                result = result_raw
            task = SwarmTask(
                name=td["name"],
                prompt=td["prompt"],
                id=td["id"],
                deps=td.get("deps", []),
                state=TaskState(td["state"]),
                result=result,
                worker_id=td.get("worker_id"),
                created_at=datetime.fromisoformat(td["created_at"]),
                started_at=datetime.fromisoformat(td["started_at"]) if td.get("started_at") else None,
                completed_at=datetime.fromisoformat(td["completed_at"]) if td.get("completed_at") else None,
                retries=td.get("retries", 0),
                max_retries=td.get("max_retries", 3),
                role=td.get("role", "worker"),
                model=td.get("model", ""),
            )
            sched._tasks[task.id] = task
        for tid, dep_ids in data.get("deps", {}).items():
            for dep_id in dep_ids:
                sched._deps[tid].add(dep_id)
                sched._rdeps[dep_id].add(tid)
        return sched

    def _get(self, task_id: str) -> SwarmTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"task {task_id} not found")
        return task
