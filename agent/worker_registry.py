"""Worker Registry — sub-agent lifecycle state machine.
Inspired by Claw Code's worker boot pattern."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    SPAWNING = "spawning"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class WorkerInfo:
    worker_id: str
    state: WorkerState = WorkerState.SPAWNING
    goal: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            WorkerState.FINISHED,
            WorkerState.FAILED,
            WorkerState.TIMEOUT,
        )


class WorkerRegistry:
    """Track all sub-agent workers and their lifecycle."""

    def __init__(self, max_concurrent: int = 3):
        self._workers: Dict[str, WorkerInfo] = {}
        self._max_concurrent = max_concurrent

    def register(self, worker_id: str, goal: str = "", **metadata) -> WorkerInfo:
        """Register a new worker in SPAWNING state."""
        if len(self._active_workers) >= self._max_concurrent:
            raise RuntimeError(
                f"Max concurrent workers ({self._max_concurrent}) reached"
            )
        info = WorkerInfo(worker_id=worker_id, goal=goal, metadata=metadata)
        self._workers[worker_id] = info
        logger.info("Worker registered: %s (goal: %s)", worker_id, goal[:50])
        return info

    def update_state(self, worker_id: str, state: WorkerState, **kwargs) -> None:
        """Update worker state."""
        info = self._workers.get(worker_id)
        if not info:
            raise KeyError(f"Unknown worker: {worker_id}")
        info.state = state
        if state == WorkerState.FINISHED:
            info.finished_at = time.time()
            info.result = kwargs.get("result")
        elif state in (WorkerState.FAILED, WorkerState.TIMEOUT):
            info.finished_at = time.time()
            info.error = kwargs.get("error")
        elif state in (WorkerState.READY, WorkerState.RUNNING):
            info.finished_at = None
        if "retry_count" in kwargs:
            info.retry_count = int(kwargs["retry_count"])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            info.metadata.update(kwargs["metadata"])
        logger.info("Worker %s -> %s", worker_id, state.value)

    def get(self, worker_id: str) -> Optional[WorkerInfo]:
        return self._workers.get(worker_id)

    @property
    def _active_workers(self) -> List[WorkerInfo]:
        return [worker for worker in self._workers.values() if not worker.is_terminal]

    @property
    def active_count(self) -> int:
        return len(self._active_workers)

    def summary(self) -> str:
        """Human-readable status summary."""
        lines = [f"Workers: {len(self._workers)} total, {self.active_count} active"]
        for worker in self._workers.values():
            status = f"  {worker.worker_id}: {worker.state.value} ({worker.elapsed:.1f}s)"
            if worker.goal:
                status += f" — {worker.goal[:40]}"
            lines.append(status)
        return "\n".join(lines)

    def cleanup(self, max_age_seconds: float = 3600) -> None:
        """Remove old finished workers."""
        now = time.time()
        to_remove = [
            worker_id
            for worker_id, worker in self._workers.items()
            if worker.is_terminal
            and worker.finished_at
            and (now - worker.finished_at) > max_age_seconds
        ]
        for worker_id in to_remove:
            del self._workers[worker_id]
