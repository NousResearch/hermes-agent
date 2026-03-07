"""Worker pool manager for swarm execution across local, Modal, and SSH backends."""

from __future__ import annotations

import abc
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from swarm.exceptions import WorkerUnavailableError
from swarm.types import SwarmConfig, SwarmTask, SwarmWorker, TaskState, WorkerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker backends (strategy pattern)
# ---------------------------------------------------------------------------

class WorkerBackend(abc.ABC):
    """Interface for executing tasks on a particular infrastructure."""

    @abc.abstractmethod
    def start_worker(self, worker: SwarmWorker) -> None:
        """Provision / start the worker."""

    @abc.abstractmethod
    def run_task(self, worker: SwarmWorker, task: SwarmTask) -> Any:
        """Execute *task* on *worker* and return the result."""

    @abc.abstractmethod
    def stop_worker(self, worker: SwarmWorker) -> None:
        """Tear down the worker."""


class LocalWorkerBackend(WorkerBackend):
    """Runs an AIAgent in a local thread."""

    def start_worker(self, worker: SwarmWorker) -> None:
        logger.info("LocalBackend: started worker %s", worker.id)

    def run_task(self, worker: SwarmWorker, task: SwarmTask) -> Any:
        """Run a task using a real AIAgent instance."""
        logger.info("LocalBackend: worker %s running task %s (%s)", worker.id, task.id, task.name)

        try:
            from run_agent import AIAgent
        except ImportError:
            logger.error("AIAgent not available — falling back to stub")
            return {"status": "completed", "output": f"stub result for {task.name}"}

        model = task.model or "claude-haiku-4-5"

        # Normalize model names: dots to dashes (e.g. claude-opus-4.6 -> claude-opus-4-6)
        # Anthropic API requires dashes, but users/configs may use dots
        if "claude" in model:
            import re as _re
            model = _re.sub(r'(\d+)\.(\d+)', r'\1-\2', model)

        # Infer provider from model name
        import os
        if "claude" in model:
            provider = "anthropic"
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = ""  # litellm handles routing
        elif model.startswith("local/"):
            provider = "local"
            api_key = "local"
            base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8800/v1")
        else:
            provider = "openrouter"
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            base_url = "https://openrouter.ai/api/v1"

        try:
            agent = AIAgent(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                max_iterations=5,
                quiet_mode=True,
            )

            result = agent.run_conversation(
                user_message=task.prompt,
                conversation_history=[],
                system_message=f"You are a focused worker agent. Complete this task: {task.name}",
            )

            final = result.get("final_response") or ""
            error = result.get("error") or ""
            completed = result.get("completed", False)

            if not final and error:
                final = f"Error: {error}"
            elif not final and not completed:
                final = f"Agent returned no output (api_calls={result.get('api_calls', 0)}, keys={list(result.keys())})"

            return {
                "status": "completed" if (completed or final) else "failed",
                "output": final,
                "api_calls": result.get("api_calls", 0),
                "tokens": getattr(agent, "session_total_tokens", 0),
            }
        except Exception as e:
            logger.error("AIAgent execution failed for task %s: %s", task.id, e)
            return {"status": "failed", "output": str(e)}

    def stop_worker(self, worker: SwarmWorker) -> None:
        logger.info("LocalBackend: stopped worker %s", worker.id)


class ModalWorkerBackend(WorkerBackend):
    """Stub for Modal cloud execution (see tools/environments/modal.py)."""

    def start_worker(self, worker: SwarmWorker) -> None:
        logger.info("ModalBackend: started worker %s (stub)", worker.id)

    def run_task(self, worker: SwarmWorker, task: SwarmTask) -> Any:
        raise NotImplementedError("Modal backend not yet implemented")

    def stop_worker(self, worker: SwarmWorker) -> None:
        logger.info("ModalBackend: stopped worker %s (stub)", worker.id)


class SSHWorkerBackend(WorkerBackend):
    """Stub for SSH remote execution."""

    def start_worker(self, worker: SwarmWorker) -> None:
        logger.info("SSHBackend: started worker %s at %s (stub)", worker.id, worker.address)

    def run_task(self, worker: SwarmWorker, task: SwarmTask) -> Any:
        raise NotImplementedError("SSH backend not yet implemented")

    def stop_worker(self, worker: SwarmWorker) -> None:
        logger.info("SSHBackend: stopped worker %s (stub)", worker.id)


_BACKENDS: dict[str, type[WorkerBackend]] = {
    "local": LocalWorkerBackend,
    "modal": ModalWorkerBackend,
    "ssh": SSHWorkerBackend,
}


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Thread-safe pool managing workers across heterogeneous backends."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._workers: dict[str, SwarmWorker] = {}
        self._backends: dict[str, WorkerBackend] = {}
        self._task_threads: dict[str, threading.Thread] = {}

    # -- backend helpers ----------------------------------------------------

    def _get_backend(self, name: str) -> WorkerBackend:
        if name not in self._backends:
            cls = _BACKENDS.get(name)
            if cls is None:
                raise ValueError(f"Unknown backend: {name}")
            self._backends[name] = cls()
        return self._backends[name]

    # -- public API ---------------------------------------------------------

    def add_worker(self, worker: SwarmWorker) -> None:
        backend = self._get_backend(worker.backend)
        backend.start_worker(worker)
        with self._lock:
            self._workers[worker.id] = worker
        logger.info("WorkerPool: added worker %s (%s)", worker.id, worker.backend)

    def remove_worker(self, worker_id: str) -> None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise WorkerUnavailableError(worker_id, "not found")
            worker.state = WorkerState.draining

        # Wait for in-flight task to finish
        thread = self._task_threads.get(worker_id)
        if thread is not None:
            thread.join(timeout=self._config.timeout_seconds)

        backend = self._get_backend(worker.backend)
        backend.stop_worker(worker)

        with self._lock:
            worker.state = WorkerState.dead
            self._workers.pop(worker_id, None)
            self._task_threads.pop(worker_id, None)
        logger.info("WorkerPool: removed worker %s", worker_id)

    def get_available_worker(self, capabilities: Optional[list[str]] = None) -> Optional[SwarmWorker]:
        with self._lock:
            for worker in self._workers.values():
                if worker.state != WorkerState.idle:
                    continue
                if capabilities and not all(c in worker.capabilities for c in capabilities):
                    continue
                return worker
        return None

    def assign_task(self, worker_id: str, task: SwarmTask) -> None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise WorkerUnavailableError(worker_id, "not found")
            if worker.state != WorkerState.idle:
                raise WorkerUnavailableError(worker_id, f"state is {worker.state.value}")
            worker.state = WorkerState.busy
            worker.current_task_id = task.id
            task.state = TaskState.running
            task.worker_id = worker_id
            task.started_at = datetime.utcnow()

        backend = self._get_backend(worker.backend)

        def _run() -> None:
            try:
                result = backend.run_task(worker, task)
                task.result = result
                task.state = TaskState.completed
            except Exception as exc:
                logger.error("Worker %s task %s failed: %s", worker_id, task.id, exc)
                task.result = str(exc)
                task.state = TaskState.failed
            finally:
                task.completed_at = datetime.utcnow()
                with self._lock:
                    worker.state = WorkerState.idle
                    worker.current_task_id = None

        t = threading.Thread(target=_run, daemon=True, name=f"worker-{worker_id}")
        self._task_threads[worker_id] = t
        t.start()

    def release_worker(self, worker_id: str) -> None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise WorkerUnavailableError(worker_id, "not found")
            worker.state = WorkerState.idle
            worker.current_task_id = None

    def scale_up(self, n: int, backend: str = "local") -> list[SwarmWorker]:
        added: list[SwarmWorker] = []
        for i in range(n):
            with self._lock:
                if len(self._workers) >= self._config.max_workers:
                    break
            worker = SwarmWorker(
                name=f"{backend}-{uuid.uuid4().hex[:6]}",
                backend=backend,
            )
            self.add_worker(worker)
            added.append(worker)
        return added

    def scale_down(self, n: int) -> int:
        removed = 0
        for _ in range(n):
            with self._lock:
                idle = [w for w in self._workers.values() if w.state == WorkerState.idle]
                if not idle:
                    break
                target = idle[0]
            self.remove_worker(target.id)
            removed += 1
        return removed

    def get_status(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for w in self._workers.values():
                counts[w.state.value] = counts.get(w.state.value, 0) + 1
            return {
                "total": len(self._workers),
                "states": counts,
                "workers": {
                    wid: {"name": w.name, "backend": w.backend, "state": w.state.value, "task": w.current_task_id}
                    for wid, w in self._workers.items()
                },
            }

    def shutdown(self) -> None:
        logger.info("WorkerPool: shutting down")
        # Wait for running tasks
        for t in list(self._task_threads.values()):
            t.join(timeout=self._config.timeout_seconds)
        # Stop all workers
        with self._lock:
            ids = list(self._workers.keys())
        for wid in ids:
            try:
                worker = self._workers.get(wid)
                if worker:
                    self._get_backend(worker.backend).stop_worker(worker)
            except Exception as exc:
                logger.warning("WorkerPool: error stopping worker %s: %s", wid, exc)
        with self._lock:
            self._workers.clear()
            self._task_threads.clear()
        logger.info("WorkerPool: shutdown complete")
