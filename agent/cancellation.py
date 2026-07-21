"""Preemptive task cancellation primitives for Hermes Agent.

Provides:
- JobState: state machine (queued -> running -> cancel_requested -> cancelling -> cancelled)
- CancellationToken: per-job AbortController equivalent (thread-safe)
- JobManager: tracks all running jobs, supports cancel_job() and cancel_all()
- CancellationResult: metadata about a cancelled job

When the feature flag (HERMES_PREEMPTIVE_CANCELLATION) is off, the existing
_interrupt_requested cooperative mechanism is preserved unchanged.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class JobState(Enum):
    """State machine: queued -> running -> cancel_requested -> cancelling -> cancelled."""
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass
class CancellationResult:
    """Result of cancelling a job."""
    job_id: str
    state: JobState
    last_completed_step: Optional[str] = None
    cancelled_step: Optional[str] = None
    remaining_processes: list = field(default_factory=list)
    cancelled_at: float = field(default_factory=time.time)


class CancellationToken:
    """Per-job cancellation token (Python AbortController equivalent).

    Thread-safe: can be signalled from the Discord listener thread while
    the agent loop runs in another thread.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._state = JobState.RUNNING
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._created_at = time.time()
        self._current_step: Optional[str] = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    @property
    def current_step(self) -> Optional[str]:
        with self._lock:
            return self._current_step

    def set_current_step(self, step: str) -> None:
        with self._lock:
            self._current_step = step

    def request_cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        with self._lock:
            if self._cancelled.is_set():
                return
            self._state = JobState.CANCEL_REQUESTED
        self._cancelled.set()
        # Fire callbacks outside lock to prevent deadlock
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    def set_cancelling(self) -> None:
        """Transition to CANCELLING state during graceful shutdown."""
        with self._lock:
            self._state = JobState.CANCELLING

    def set_cancelled(self) -> None:
        """Transition to CANCELLED state after cleanup completes."""
        with self._lock:
            self._state = JobState.CANCELLED

    def register(self, callback: Callable[[], None]) -> None:
        """Register a callback to fire when cancellation is requested."""
        if self._cancelled.is_set():
            callback()
        else:
            self._callbacks.append(callback)

    def throw_if_cancelled(self) -> None:
        """Raise CancelledError if cancellation was requested."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled by user request")

    def check_cancelled(self) -> bool:
        """Check if cancelled without raising. Returns True if cancelled."""
        return self._cancelled.is_set()

    async def sleep(self, seconds: float) -> None:
        """Cancellable sleep. Raises CancelledError if cancelled during or after sleep."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled during sleep")
        await asyncio.sleep(seconds)
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled after sleep")


class JobManager:
    """Manages all running agent jobs and their cancellation tokens.

    Thread-safe. The gateway Discord listener can call cancel_job() /
    cancel_all() from a different thread than the agent execution loop.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[CancellationToken, JobState]] = {}
        self._lock = threading.RLock()

    def create_job(self, job_id: Optional[str] = None) -> str:
        """Register a new job and return its ID."""
        jid = job_id or str(uuid.uuid4())
        token = CancellationToken()
        with self._lock:
            self._jobs[jid] = (token, JobState.RUNNING)
        return jid

    def get_token(self, job_id: str) -> Optional[CancellationToken]:
        with self._lock:
            entry = self._jobs.get(job_id)
            return entry[0] if entry else None

    def get_state(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            entry = self._jobs.get(job_id)
            return entry[1] if entry else None

    def set_state(self, job_id: str, state: JobState) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry:
                token = entry[0]
                self._jobs[job_id] = (token, state)

    def cancel_job(
        self,
        job_id: str,
        *,
        last_completed_step: Optional[str] = None,
        cancelled_step: Optional[str] = None,
        remaining_processes: Optional[list] = None,
    ) -> Optional[CancellationResult]:
        """Request cancellation of a specific job. Returns result or None if not found."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if not entry:
                return None
            token, _ = entry

        token.request_cancel()

        result = CancellationResult(
            job_id=job_id,
            state=JobState.CANCELLED,
            last_completed_step=last_completed_step,
            cancelled_step=cancelled_step or token.current_step,
            remaining_processes=remaining_processes or [],
        )

        with self._lock:
            self._jobs[job_id] = (token, JobState.CANCELLED)

        return result

    def cancel_all(self) -> list[CancellationResult]:
        """Cancel all running jobs. Returns results for each cancelled job."""
        with self._lock:
            job_ids = list(self._jobs.keys())

        results = []
        for jid in job_ids:
            result = self.cancel_job(jid)
            if result:
                results.append(result)
        return results

    def unregister_job(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def list_running_jobs(self) -> list[str]:
        with self._lock:
            return [
                jid for jid, (_, state) in self._jobs.items()
                if state in (JobState.RUNNING, JobState.CANCEL_REQUESTED, JobState.CANCELLING)
            ]

    def set_current_step(self, job_id: str, step: str) -> None:
        """Track the current step for cancellation metadata."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry:
                token = entry[0]
                token.set_current_step(step)


# Global singleton
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global JobManager singleton."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def is_preemptive_cancellation_enabled() -> bool:
    """Check if preemptive cancellation feature flag is enabled.

    Resolution order:
    1. Env var HERMES_PREEMPTIVE_CANCELLATION (highest priority)
    2. config.yaml agent.preemptive_cancellation
    3. Default: off
    """
    env_val = os.getenv("HERMES_PREEMPTIVE_CANCELLATION", "")
    if env_val.lower() in ("true", "1", "yes"):
        return True
    if env_val.lower() in ("false", "0", "no"):
        return False
    # Check config.yaml
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return bool(config.get("agent", {}).get("preemptive_cancellation", False))
    except Exception:
        return False
