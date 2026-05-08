"""Orchestrator — Async multi-agent task manager with DAG dependencies.

Provides persistent, cross-turn task orchestration with:
- Fire-and-forget async delegation
- DAG-based task dependency resolution
- Progress tracking across turns
- Integration with WebhookDispatcher for completion callbacks
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OrchestrationTask:
    """A single orchestrated task."""
    task_id: str
    goal: str
    context: str = ""
    toolsets: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    progress: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    subagent_id: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal[:200],
            "status": self.status.value,
            "depends_on": self.depends_on,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class Orchestrator:
    """Persistent multi-agent orchestrator that survives across turns."""

    def __init__(self, agent_factory: Callable, max_workers: int = 4,
                 webhook_dispatcher=None):
        """
        Args:
            agent_factory: Callable(toolsets, skip_memory) -> AIAgent
            max_workers: Max concurrent tasks.
            webhook_dispatcher: Optional WebhookDispatcher for completion callbacks.
        """
        self._agent_factory = agent_factory
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, OrchestrationTask] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.RLock()
        self._webhook_dispatcher = webhook_dispatcher
        self._callbacks: List[Callable] = []

    def submit(self, goal: str, context: str = "", depends_on: Optional[List[str]] = None,
               toolsets: Optional[List[str]] = None) -> str:
        """Submit an async task. Returns task_id immediately.

        If depends_on is specified, the task won't start until all dependencies complete.
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = OrchestrationTask(
            task_id=task_id,
            goal=goal,
            context=context,
            toolsets=toolsets or [],
            depends_on=depends_on or [],
        )

        with self._lock:
            self._tasks[task_id] = task

            # If no dependencies, submit immediately
            if not depends_on:
                self._submit_to_executor(task)
            else:
                log.info("Task %s waiting for dependencies: %s", task_id, depends_on)

        return task_id

    def get_status(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get status of one or all tasks."""
        with self._lock:
            if task_id:
                task = self._tasks.get(task_id)
                return [task.to_dict()] if task else []
            return [t.to_dict() for t in self._tasks.values()]

    def wait(self, task_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Block until a specific task completes."""
        future = self._futures.get(task_id)
        if not future:
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task.to_dict() if task else {}
            return {"status": "error", "message": f"Task {task_id} not found or not running"}

        try:
            result = future.result(timeout=timeout)
            return result
        except (TimeoutError, FuturesTimeoutError):
            return {"status": "timeout", "task_id": task_id}
        except Exception as e:
            return {"status": "error", "task_id": task_id, "error": str(e)}

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

            # Cancel the future if possible
            future = self._futures.get(task_id)
            if future:
                future.cancel()

            self._resolve_dependencies()
            return True

    def register_callback(self, callback: Callable):
        """Register a callback for task completion events."""
        self._callbacks.append(callback)

    def _submit_to_executor(self, task: OrchestrationTask):
        """Submit task to the thread pool executor."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        future = self._executor.submit(self._run_task, task)
        self._futures[task.task_id] = future

        future.add_done_callback(
            lambda f, tid=task.task_id: self._on_future_done(tid, f)
        )

    def _run_task(self, task: OrchestrationTask) -> Dict[str, Any]:
        """Execute a task using an AIAgent."""
        try:
            agent = self._agent_factory(
                toolsets=task.toolsets or None,
                skip_memory=True,
            )
            result = agent.run_conversation(
                user_message=task.goal + ("\n\nContext: " + task.context if task.context else ""),
            )
            return result if isinstance(result, dict) else {"response": str(result)}
        except Exception as e:
            log.error("Task %s failed: %s", task.task_id, e)
            raise

    def _on_future_done(self, task_id: str, future: Future):
        """Callback when a future completes."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return

            if task.status == TaskStatus.CANCELLED:
                return

            if future.exception():
                task.status = TaskStatus.FAILED
                task.error = str(future.exception())
            else:
                task.status = TaskStatus.COMPLETED
                task.result = future.result()

            task.completed_at = time.time()

        # Fire webhook
        if self._webhook_dispatcher:
            event = "task.completed" if task.status == TaskStatus.COMPLETED else "task.failed"
            self._webhook_dispatcher.dispatch(event, task.to_dict())

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(task_id, task.to_dict())
            except Exception as e:
                log.warning("Callback error for task %s: %s", task_id, e)

        # Resolve dependent tasks
        self._resolve_dependencies()

    def _resolve_dependencies(self):
        """Check which pending tasks have all deps satisfied, submit them."""
        with self._lock:
            for task in self._tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                if not task.depends_on:
                    continue

                # Check for unknown dependencies — fail the task immediately
                unknown_deps = [dep for dep in task.depends_on if dep not in self._tasks]
                if unknown_deps:
                    task.status = TaskStatus.FAILED
                    task.error = f"Unknown dependencies: {', '.join(unknown_deps)}"
                    task.completed_at = time.time()
                    continue

                all_done = all(
                    self._tasks[dep].status
                    in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
                    for dep in task.depends_on
                )

                if all_done:
                    # Check if any dependency failed
                    any_failed = any(
                        self._tasks.get(dep, OrchestrationTask(task_id="", goal="")).status
                        == TaskStatus.FAILED
                        for dep in task.depends_on
                    )
                    if any_failed:
                        task.status = TaskStatus.FAILED
                        task.error = "Dependency failed"
                        task.completed_at = time.time()
                    else:
                        self._submit_to_executor(task)

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)

    def summary(self) -> Dict[str, Any]:
        """Get orchestrator summary."""
        with self._lock:
            counts = {}
            for status in TaskStatus:
                counts[status.value] = sum(
                    1 for t in self._tasks.values() if t.status == status
                )
            return {
                "total_tasks": len(self._tasks),
                "status_counts": counts,
                "active_futures": len([f for f in self._futures.values() if not f.done()]),
            }
