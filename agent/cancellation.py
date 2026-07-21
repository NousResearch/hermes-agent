"""Preemptive task cancellation primitives for Hermes Agent.

Provides:
- JobState: state machine (queued -> running -> cancel_requested -> cancelling -> cancelled)
- CancellationToken: per-job AbortController equivalent (thread-safe)
- JobManager: tracks all running jobs, supports cancel_job() and cancel_all()
- CancellationResult: metadata about a cancelled job
- ProcessRegistry: tracks PIDs per job_id for process tree termination on cancel

State transitions:
  cancel_job() -> CANCEL_REQUESTED (signals token, fires callbacks)
  callbacks -> CANCELLING (process tree kill, resource cleanup)
  cleanup verified -> CANCELLED (final state, remaining_processes recorded)

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

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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
        # Event for cancellable sleep — allows instant wake on cancel
        # without waiting for the full sleep duration.
        self._cancel_event = threading.Event()

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
        """Signal cancellation. Idempotent. Sets state to CANCEL_REQUESTED."""
        with self._lock:
            if self._cancelled.is_set():
                return
            self._state = JobState.CANCEL_REQUESTED
        self._cancelled.set()
        self._cancel_event.set()
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
        """Cancellable sleep using Event.wait — wakes instantly on cancel.

        Unlike asyncio.sleep, this does NOT wait for the full duration when
        cancelled. The cancel_event is set immediately on request_cancel().
        """
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled during sleep")
        # Use the Event to wait — either the timeout expires or cancel fires
        # We need to run the Event.wait in a thread since we're in async context
        loop = asyncio.get_event_loop()
        fired = await loop.run_in_executor(
            None,
            lambda: self._cancel_event.wait(timeout=seconds)
        )
        if fired:
            raise asyncio.CancelledError("Job cancelled during sleep")

    def sleep_sync(self, seconds: float) -> None:
        """Synchronous cancellable sleep. Wakes instantly on cancel."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled during sleep")
        self._cancel_event.wait(timeout=seconds)
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled during sleep")


class ProcessRegistry:
    """Tracks PIDs per job_id for process tree termination on cancel.

    When a terminal command spawns a process, the PID is registered here
    under the current job_id. When the job is cancelled, the CancellationToken
    callback calls kill_process_tree for all registered PIDs.
    """

    def __init__(self) -> None:
        self._pids: dict[str, list[int]] = {}
        self._lock = threading.RLock()

    def register_pid(self, job_id: str, pid: int) -> None:
        """Associate a PID with a job."""
        with self._lock:
            self._pids.setdefault(job_id, []).append(pid)

    def register_pgid(self, job_id: str, pgid: int) -> None:
        """Associate a process group ID with a job."""
        self.register_pid(job_id, pgid)

    def get_pids(self, job_id: str) -> list[int]:
        with self._lock:
            return list(self._pids.get(job_id, []))

    def clear(self, job_id: str) -> list[int]:
        """Remove and return all PIDs for a job."""
        with self._lock:
            return self._pids.pop(job_id, [])

    def get_remaining(self, job_id: str) -> list[dict]:
        """Check which registered processes are still alive for a job."""
        pids = self.get_pids(job_id)
        remaining = []
        if _HAS_PSUTIL:
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    remaining.append({"pid": pid, "status": proc.status(), "name": proc.name()})
                except Exception:
                    pass  # Already dead
        else:
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    remaining.append({"pid": pid, "status": "alive", "name": "unknown"})
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        return remaining


# Global process registry singleton
_process_registry: Optional[ProcessRegistry] = None


def get_process_registry() -> ProcessRegistry:
    global _process_registry
    if _process_registry is None:
        _process_registry = ProcessRegistry()
    return _process_registry


class JobManager:
    """Manages all running agent jobs and their cancellation tokens.

    Thread-safe. The gateway Discord listener can call cancel_job() /
    cancel_all() from a different thread than the agent execution loop.

    State transitions:
      cancel_job() -> CANCEL_REQUESTED (signals token, fires callbacks)
      callbacks -> CANCELLING (process tree kill, resource cleanup)
      cleanup verified -> CANCELLED (final state, remaining_processes recorded)
    """

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[CancellationToken, JobState]] = {}
        self._lock = threading.RLock()

    def create_job(self, job_id: Optional[str] = None) -> str:
        """Register a new job, wire cancellation callback, and return its ID."""
        jid = job_id or str(uuid.uuid4())
        token = CancellationToken()
        # Wire the default cancellation callback: process tree kill + state progression.
        # Pass `self` so the callback updates THIS manager instance, not the global singleton.
        token.register(lambda: _cancel_callback(jid, token, self))
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
    ) -> Optional[CancellationResult]:
        """Request cancellation of a specific job.

        Sets state to CANCEL_REQUESTED and fires the token. The token's
        callbacks handle process tree termination and state progression:
        CANCEL_REQUESTED -> CANCELLING -> CANCELLED.

        For jobs already in CANCEL_REQUESTED or later, returns the current
        state without re-declaring cancellation (idempotent).

        Returns CancellationResult or None if job not found.
        """
        with self._lock:
            entry = self._jobs.get(job_id)
            if not entry:
                return None
            token, current_state = entry

            # If already past CANCEL_REQUESTED, return current state
            # without re-declaring cancellation
            if current_state in (JobState.CANCEL_REQUESTED, JobState.CANCELLING, JobState.CANCELLED):
                return CancellationResult(
                    job_id=job_id,
                    state=current_state,
                    last_completed_step=last_completed_step,
                    cancelled_step=cancelled_step or token.current_step,
                    remaining_processes=get_process_registry().get_remaining(job_id),
                )

        # Request cancel on the token — this fires callbacks which
        # handle process tree kill and state progression
        token.request_cancel()

        # Return result with CANCEL_REQUESTED state (not CANCELLED yet —
        # the callback will progress to CANCELLING then CANCELLED)
        return CancellationResult(
            job_id=job_id,
            state=JobState.CANCEL_REQUESTED,
            last_completed_step=last_completed_step,
            cancelled_step=cancelled_step or token.current_step,
            remaining_processes=get_process_registry().get_remaining(job_id),
        )

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

    def list_running_jobs(self) -> list[dict]:
        """List all running jobs with their job_id, state, and current step."""
        with self._lock:
            result = []
            for jid, (token, state) in self._jobs.items():
                if state in (JobState.RUNNING, JobState.CANCEL_REQUESTED, JobState.CANCELLING):
                    result.append({
                        "job_id": jid,
                        "state": state.value,
                        "current_step": token.current_step,
                    })
            return result

    def set_current_step(self, job_id: str, step: str) -> None:
        """Track the current step for cancellation metadata."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry:
                token = entry[0]
                token.set_current_step(step)

    def complete_cancellation(self, job_id: str, remaining_processes: list) -> None:
        """Mark a job as fully cancelled after cleanup is verified.

        Called by the cancellation callback after process tree termination
        and resource cleanup are complete. Records remaining_processes and
        transitions to CANCELLED only if no processes remain.  If processes
        survived the kill attempt, the state stays at CANCELLING to signal
        that cleanup was incomplete.
        """
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry:
                token = entry[0]
                if remaining_processes:
                    # Survivors — stay in CANCELLING, do not declare CANCELLED
                    self._jobs[job_id] = (token, JobState.CANCELLING)
                else:
                    # Clean — transition to CANCELLED
                    self._jobs[job_id] = (token, JobState.CANCELLED)
                    # Clear process registry for this job
                    get_process_registry().clear(job_id)


def _cancel_callback(
    job_id: str,
    token: CancellationToken,
    mgr: "JobManager",
    grace_timeout: float = 5.0,
) -> None:
    """Default cancellation callback: kill process tree, progress to CANCELLED.

    This is registered on every CancellationToken when a job is created.
    It runs in the thread that called request_cancel() (typically the
    Discord listener thread), NOT the agent execution thread.
    """
    token.set_cancelling()

    # Kill all registered processes for this job
    registry = get_process_registry()
    pids = registry.get_pids(job_id)

    killed = []
    survived = []

    if _HAS_PSUTIL and pids:
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                children = proc.children(recursive=True)
                all_procs = children + [proc]
                for p in reversed(all_procs):
                    try:
                        p.terminate()
                        killed.append(p.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # Wait for graceful termination
                deadline = time.time() + grace_timeout
                for p in all_procs:
                    remaining = max(0.1, deadline - time.time())
                    try:
                        p.wait(timeout=remaining)
                    except psutil.TimeoutExpired:
                        try:
                            p.kill()
                            p.wait(timeout=1.0)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        except psutil.TimeoutExpired:
                            survived.append(p.pid)
                    except psutil.NoSuchProcess:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    elif pids:
        # Fallback without psutil
        for pid in pids:
            try:
                if os.name == "nt":
                    import subprocess as sp
                    sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=grace_timeout)
                    killed.append(pid)
                else:
                    try:
                        os.killpg(os.getpgid(pid), 15)  # SIGTERM
                        killed.append(pid)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    time.sleep(min(1.0, grace_timeout))
                    try:
                        os.killpg(os.getpgid(pid), 9)  # SIGKILL
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            except Exception:
                survived.append(pid)

    # Check remaining
    remaining = registry.get_remaining(job_id)

    # Only set CANCELLED on the token if no processes survived.
    # If survivors exist, leave the token in CANCELLING state so
    # complete_cancellation and the JobManager both reflect the
    # incomplete cleanup.
    if not remaining:
        token.set_cancelled()

    # Update JobManager state — use the mgr passed in (the one that created the job),
    # NOT the global singleton (which may be a different instance in tests).
    mgr.complete_cancellation(job_id, remaining)


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
