"""Swarm orchestrator — coordinates scheduler, workers, router, artifacts, and messaging."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import Any

from swarm.artifacts import ArtifactStore
from swarm.messaging import MessageBus
from swarm.router import ModelRouter
from swarm.scheduler import SwarmScheduler
from swarm.types import (
    SwarmConfig,
    SwarmResult,
    SwarmTask,
    SwarmWorker,
    TaskState,
    WorkerState,
)
from swarm.worker import WorkerPool

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Top-level coordinator that ties all swarm subsystems together."""

    def __init__(self, config: SwarmConfig | None = None) -> None:
        self._config = config or SwarmConfig()
        self._scheduler = SwarmScheduler()
        self._pool = WorkerPool(self._config)
        self._router = ModelRouter(self._config)
        self._artifacts = ArtifactStore(Path(self._config.artifact_dir))
        self._bus = MessageBus()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._events: list[dict[str, Any]] = []

        # Register orchestrator on the bus
        self._bus._ensure_agent("orchestrator")

    # ------------------------------------------------------------------
    # Task creation helpers
    # ------------------------------------------------------------------

    def add_task(
        self,
        name: str,
        prompt: str,
        role: str = "executor",
        deps: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        task = SwarmTask(
            name=name,
            prompt=prompt,
            role=role,
            deps=deps or [],
            model=model or "",
            max_retries=self._config.max_retries,
        )
        self._scheduler.add_task(task)
        self._log_event("task_added", task_id=task.id, name=name, role=role)
        return task.id

    def add_plan(self, tasks: list[dict]) -> list[str]:
        ids: list[str] = []
        name_to_id: dict[str, str] = {}

        for spec in tasks:
            name = spec["name"]
            prompt = spec.get("prompt", "")
            role = spec.get("role", "executor")
            model = spec.get("model", None)
            raw_deps = spec.get("deps", [])

            # Resolve dep names to ids (supports both names and raw ids)
            resolved_deps = [name_to_id.get(d, d) for d in raw_deps]

            task_id = self.add_task(
                name=name,
                prompt=prompt,
                role=role,
                deps=resolved_deps,
                model=model,
            )
            name_to_id[name] = task_id
            ids.append(task_id)

        return ids

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self, max_turns: int = 1000, poll_interval: float = 0.5) -> dict:
        self._cancelled.clear()
        self._ensure_workers()

        turn = 0
        while turn < max_turns and not self._cancelled.is_set():
            if self._scheduler.is_complete():
                break

            ready = self._scheduler.get_ready_tasks()
            dispatched = 0

            for task in ready:
                if self._cancelled.is_set():
                    break

                worker = self._pool.get_available_worker()
                if worker is None:
                    break

                # Select model via router if not explicitly set
                if not task.model:
                    task.model = self._router.select_model(task.role)

                self._scheduler.mark_running(task.id, worker.id)
                self._log_event(
                    "task_dispatched",
                    task_id=task.id,
                    worker_id=worker.id,
                    model=task.model,
                )

                self._dispatch_task(task, worker)
                dispatched += 1

            # Check for completed / failed tasks
            self._collect_results()

            if dispatched == 0 and not self._has_running_tasks():
                # Nothing dispatched and nothing running — either done or stuck
                if self._scheduler.is_complete():
                    break
                # Could be waiting for workers to free up; sleep briefly
                time.sleep(poll_interval)

            turn += 1
            if dispatched == 0:
                time.sleep(poll_interval)

        # Drain remaining running tasks
        self._drain_running(timeout=self._config.timeout_seconds)
        self._collect_results()

        return self._build_summary()

    def _build_live_table(self) -> "Table":
        """Build a Rich Table showing current swarm status."""
        from rich.table import Table
        from rich.text import Text

        table = Table(
            title="🐝 Swarm",
            show_header=True,
            header_style="bold #DAA520",
            border_style="dim",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Worker", style="bold", width=12)
        table.add_column("Task", width=30)
        table.add_column("Role", width=10)
        table.add_column("Status", width=10)
        table.add_column("Time", justify="right", width=8)

        sched_status = self._scheduler.get_status()

        # Show workers with their current tasks
        for wid, worker in sorted(self._pool._workers.items()):
            task_name = ""
            role = ""
            status_text = Text(worker.state.value)
            elapsed = ""

            if worker.current_task_id:
                task = self._scheduler._tasks.get(worker.current_task_id)
                if task:
                    task_name = task.name[:28]
                    role = task.role
                    if task.started_at:
                        from datetime import datetime, timezone
                        now = datetime.now(timezone.utc)
                        started = task.started_at.replace(tzinfo=timezone.utc)
                        secs = (now - started).total_seconds()
                        elapsed = f"{secs:.0f}s"

            if worker.state.value == "busy":
                status_text = Text("running", style="yellow")
            elif worker.state.value == "idle":
                status_text = Text("idle", style="dim")
            elif worker.state.value == "dead":
                status_text = Text("dead", style="red")

            table.add_row(
                worker.name[:10],
                task_name,
                role,
                status_text,
                elapsed,
            )

        # Summary row
        completed = sched_status.get("completed", 0)
        failed = sched_status.get("failed", 0)
        pending = sched_status.get("pending", 0)
        running = sched_status.get("running", 0)
        total = sum(sched_status.values())

        table.add_section()
        summary = f"✅ {completed}  ⏳ {running}  📋 {pending}  ❌ {failed}  / {total} total"
        table.add_row("", Text(summary, style="dim"), "", "", "")

        return table

    def _get_task_elapsed(self, task: SwarmTask) -> float:
        if not task.started_at:
            return 0.0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        started = task.started_at.replace(tzinfo=timezone.utc)
        if task.completed_at:
            ended = task.completed_at.replace(tzinfo=timezone.utc)
            return (ended - started).total_seconds()
        return (now - started).total_seconds()

    def _get_task_output_preview(self, task: SwarmTask, max_len: int = 60) -> str:
        r = task.result
        if not r:
            return ""
        out = getattr(r, 'output', r) if not isinstance(r, dict) else r
        if isinstance(out, dict):
            text = out.get('output', '') or out.get('final_response', '') or ''
        elif isinstance(out, str):
            text = out
        else:
            text = str(out) if out else ''
        text = (text or '').strip().replace('\n', ' ')
        if len(text) > max_len:
            text = text[:max_len] + "..."
        return text

    def _snapshot_display(self) -> list[dict]:
        """Build display data for the prompt_toolkit widget."""
        result = []
        for task in self._scheduler._tasks.values():
            elapsed = self._get_task_elapsed(task)
            preview = self._get_task_output_preview(task, 40) if task.state in (TaskState.completed, TaskState.failed) else ""
            result.append({
                "name": task.name,
                "state": task.state.value,
                "elapsed": elapsed,
                "preview": preview,
            })
        return result

    def run_with_display(self, display_callback=None, max_turns: int = 1000, poll_interval: float = 0.5, **kw) -> dict:
        """Run the swarm, calling display_callback with task data on each tick."""
        self._cancelled.clear()
        self._ensure_workers()

        turn = 0
        while turn < max_turns and not self._cancelled.is_set():
            if self._scheduler.is_complete():
                break

            ready = self._scheduler.get_ready_tasks()
            dispatched = 0

            for task in ready:
                if self._cancelled.is_set():
                    break

                worker = self._pool.get_available_worker()
                if worker is None:
                    break

                if not task.model:
                    task.model = self._router.select_model(task.role)

                self._scheduler.mark_running(task.id, worker.id)
                self._log_event(
                    "task_dispatched",
                    task_id=task.id,
                    worker_id=worker.id,
                    model=task.model,
                )

                self._dispatch_task(task, worker)
                dispatched += 1

            self._collect_results()

            if display_callback:
                display_callback(self._snapshot_display())

            if dispatched == 0 and not self._has_running_tasks():
                if self._scheduler.is_complete():
                    break
                time.sleep(poll_interval)

            turn += 1
            if dispatched == 0:
                time.sleep(poll_interval)

        self._drain_running(timeout=self._config.timeout_seconds)
        self._collect_results()

        if display_callback:
            display_callback(self._snapshot_display())

        return self._build_summary()

    def run_async(self) -> Future:
        future: Future = Future()

        def _run() -> None:
            try:
                result = self.run()
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)

        t = threading.Thread(target=_run, daemon=True, name="orchestrator-async")
        t.start()
        return future

    def cancel(self) -> None:
        self._cancelled.set()
        self._log_event("cancelled")
        # Cancel all pending tasks in scheduler
        for task in self._scheduler.get_ready_tasks():
            self._scheduler.mark_failed(task.id, "orchestrator cancelled")
        self._pool.shutdown()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "scheduler": self._scheduler.get_status(),
            "workers": self._pool.get_status(),
            "router": self._router.get_spend_summary(),
            "artifacts": self._artifacts.export_manifest(),
            "events": len(self._events),
            "cancelled": self._cancelled.is_set(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_workers(self) -> None:
        """Scale up workers if the pool is empty."""
        status = self._pool.get_status()
        if status["total"] == 0:
            self._pool.scale_up(self._config.max_workers)
            self._log_event("workers_scaled", count=self._config.max_workers)

    def _dispatch_task(self, task: SwarmTask, worker: SwarmWorker) -> None:
        """Run a task on a worker in a background thread."""

        def _execute() -> None:
            try:
                result = self._execute_task(task, worker)
                self._on_task_complete(task, worker, result)
            except Exception as exc:
                logger.error("Task %s failed: %s", task.id, exc)
                self._on_task_failed(task, worker, str(exc))

        t = threading.Thread(
            target=_execute, daemon=True, name=f"task-{task.id}"
        )
        t.start()

    def _execute_task(self, task: SwarmTask, worker: SwarmWorker) -> SwarmResult:
        """Run a single task on a worker. Stub that uses the worker backend."""
        backend = self._pool._get_backend(worker.backend)

        with self._lock:
            worker.state = WorkerState.busy
            worker.current_task_id = task.id

        start = time.monotonic()
        try:
            raw_result = backend.run_task(worker, task)
        finally:
            elapsed = time.monotonic() - start
            with self._lock:
                worker.state = WorkerState.idle
                worker.current_task_id = None

        return SwarmResult(
            task_id=task.id,
            success=True,
            output=raw_result,
            duration_seconds=round(elapsed, 3),
        )

    def _on_task_complete(
        self, task: SwarmTask, worker: SwarmWorker, result: SwarmResult
    ) -> None:
        self._scheduler.mark_completed(task.id, result)

        # Store any artifacts referenced in the result
        if result.artifacts:
            for path_str in result.artifacts:
                p = Path(path_str)
                if p.exists():
                    ref = self._artifacts.store(task.id, p)
                    task.artifacts.append(ref)

        self._log_event(
            "task_completed",
            task_id=task.id,
            worker_id=worker.id,
            duration=result.duration_seconds,
        )

        # Broadcast completion on the message bus
        self._bus.broadcast(
            "orchestrator",
            "status",
            {"event": "task_completed", "task_id": task.id},
        )

    def _on_task_failed(self, task: SwarmTask, worker: SwarmWorker, error: str) -> None:
        self._scheduler.mark_failed(task.id, error)
        self._log_event(
            "task_failed",
            task_id=task.id,
            worker_id=worker.id,
            error=error,
            retries=task.retries,
        )

        self._bus.broadcast(
            "orchestrator",
            "error",
            {"event": "task_failed", "task_id": task.id, "error": error},
        )

    def _collect_results(self) -> None:
        """No-op for thread-based execution — results are collected via callbacks."""
        pass

    def _has_running_tasks(self) -> bool:
        status = self._scheduler.get_status()
        return status.get("running", 0) > 0

    def _drain_running(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._has_running_tasks():
                return
            time.sleep(0.1)
        logger.warning("Drain timed out with running tasks remaining")

    def _build_summary(self) -> dict:
        sched_status = self._scheduler.get_status()
        spend = self._router.get_spend_summary()
        return {
            "tasks": sched_status,
            "total_tasks": sum(sched_status.values()),
            "spend": spend,
            "artifacts": len(self._artifacts.list_all()),
            "events": len(self._events),
            "cancelled": self._cancelled.is_set(),
        }

    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        entry = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        with self._lock:
            self._events.append(entry)
        logger.info("orchestrator event: %s %s", event_type, kwargs)
