"""Non-blocking, killable live execution runner for the swarm operator.

The gateway hook must not run delegate fan-out inline: synchronous execution can
block the gateway event loop, and thread timeouts cannot stop already-running
work. This module gives the hook a narrow process boundary instead. The parent
returns immediately after spawning a child process plus a monitor thread; the
monitor terminates the child if it exceeds its timeout and persists the final
state.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.swarm_executor import execute_swarm
from agent.swarm_state import AuditEvent, RoutingPlan, SwarmJob
from agent.swarm_store import SwarmStore


@dataclass
class SwarmLiveRunHandle:
    """Summary of a live runner launch attempt."""

    started: bool
    status: str
    reason: str = ""
    job_id: str = ""
    pid: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "status": self.status,
            "reason": self.reason,
            "job_id": self.job_id,
            "pid": self.pid,
        }


def _coerce_timeout(value: float | int | str, default: float = 30.0) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    if not 0 < timeout <= 30.0:
        return default
    # NaN fails both comparisons above; keep an explicit guard for readability.
    if timeout != timeout:
        return default
    return timeout


def _child_execute_swarm(
    job: SwarmJob,
    routing_plan: RoutingPlan,
    delegate_fn: Callable[..., Any],
    *,
    store_base_dir: str,
    max_children: int,
) -> None:
    setsid = getattr(os, "setsid", None)
    if callable(setsid):
        try:
            setsid()
        except OSError:
            # Best-effort process-group isolation. The monitor still kills the
            # direct child if the platform refuses to create a new session.
            pass
    store = SwarmStore(base_dir=store_base_dir)
    try:
        store.append_event(
            AuditEvent(
                "live_delegation_child_started",
                "Swarm live delegation child process started.",
                metadata={"job_id": job.job_id},
            )
        )
        execution = execute_swarm(job, routing_plan, delegate_fn, max_children=max_children)
        job.metadata["live_delegation_status"] = "executed"
        job.metadata["live_delegation_result"] = execution.to_dict()
        job.audit.append(
            AuditEvent(
                "live_delegation_executed",
                "Swarm live delegation completed in the child process.",
                metadata={"status": execution.status, "dispatched_count": len(execution.dispatched)},
            )
        )
        store.save_job(job)
        store.append_event(
            AuditEvent(
                "live_delegation_finished",
                "Swarm live delegation child process finished.",
                metadata={"job_id": job.job_id, "status": job.status},
            )
        )
    except BaseException as exc:  # noqa: BLE001 - child must persist failure instead of vanishing.
        job.metadata["live_delegation_status"] = "failed"
        job.metadata["live_delegation_error"] = str(exc)
        job.audit.append(
            AuditEvent(
                "live_delegation_failed",
                "Swarm live delegation failed in the child process.",
                metadata={"error": str(exc)},
            )
        )
        store.save_job(job)
        store.append_event(
            AuditEvent(
                "live_delegation_failed",
                "Swarm live delegation child process failed.",
                metadata={"job_id": job.job_id, "error": str(exc)},
            )
        )


def _load_latest_job(store: SwarmStore, fallback: SwarmJob) -> SwarmJob:
    try:
        return store.load_jobs().get(fallback.job_id, fallback)
    except Exception:
        return fallback


def _isolated_process_group_id(process: Any) -> Optional[int]:
    pid = getattr(process, "pid", None)
    if not pid:
        return None
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    except OSError:
        return None
    # The child calls setsid(), which makes pgid == pid. If that has not
    # happened yet (or failed), pgid may be the gateway/test-runner group; never
    # signal a non-isolated group from the monitor.
    return pgid if pgid == pid else None


def _terminate_process_tree(process: Any) -> None:
    pgid = _isolated_process_group_id(process)
    killpg = getattr(os, "killpg", None)
    if pgid is not None and callable(killpg):
        try:
            killpg(pgid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        process.terminate()
    except Exception:
        pass


def _kill_process_tree(process: Any) -> None:
    pgid = _isolated_process_group_id(process)
    killpg = getattr(os, "killpg", None)
    if pgid is not None and callable(killpg):
        try:
            killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
        except Exception:
            pass


def _persist_terminal_if_still_running(
    *,
    store: SwarmStore,
    fallback: SwarmJob,
    status: str,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> None:
    latest = _load_latest_job(store, fallback)
    if latest.metadata.get("live_delegation_status") != "running":
        return
    latest.metadata["live_delegation_status"] = status
    latest.audit.append(AuditEvent(event_type, message, metadata=metadata))
    store.save_job(latest)
    store.append_event(AuditEvent(event_type, message, metadata={"job_id": latest.job_id, **metadata}))


def _monitor_process(process: Any, *, store_base_dir: str, job: SwarmJob, timeout_seconds: float) -> None:
    process.join(timeout_seconds)
    store = SwarmStore(base_dir=store_base_dir)
    if process.is_alive():
        _terminate_process_tree(process)
        process.join(2)
        if process.is_alive():
            _kill_process_tree(process)
            process.join(2)
        _persist_terminal_if_still_running(
            store=store,
            fallback=job,
            status="timeout",
            event_type="live_delegation_timeout",
            message="Swarm live delegation exceeded its timeout and the child process tree was terminated.",
            metadata={"timeout_seconds": timeout_seconds, "pid": getattr(process, "pid", None)},
        )
        return

    exitcode = getattr(process, "exitcode", None)
    if exitcode is not None:
        _persist_terminal_if_still_running(
            store=store,
            fallback=job,
            status="failed",
            event_type="live_delegation_crashed",
            message="Swarm live delegation child process exited before saving a terminal state.",
            metadata={"exitcode": exitcode, "pid": getattr(process, "pid", None)},
        )


def start_live_swarm(
    job: SwarmJob,
    routing_plan: RoutingPlan,
    delegate_fn: Callable[..., Any],
    *,
    store: SwarmStore,
    max_children: int = 3,
    timeout_seconds: float = 30.0,
) -> SwarmLiveRunHandle:
    """Start live swarm execution in a killable child process.

    The function mutates ``job`` with launch metadata and returns immediately.
    The caller should save the job after this returns so status is visible before
    the child finishes.
    """

    if not callable(delegate_fn):
        return SwarmLiveRunHandle(started=False, status="blocked", reason="missing_delegate_fn", job_id=job.job_id)

    context_names = []
    for name in ("forkserver", "fork"):
        try:
            multiprocessing.get_context(name)
        except (ValueError, RuntimeError):
            continue
        context_names.append(name)
    if not context_names:
        return SwarmLiveRunHandle(started=False, status="blocked", reason="process_context_unavailable", job_id=job.job_id)

    timeout = _coerce_timeout(timeout_seconds)
    job.metadata["live_delegation_status"] = "running"
    job.metadata["live_delegation_timeout_seconds"] = timeout
    job.audit.append(
        AuditEvent(
            "live_delegation_started",
            "Swarm live delegation launched in a killable child process.",
            metadata={"timeout_seconds": timeout, "max_children": max_children},
        )
    )

    store.save_job(job)
    process = None
    last_start_error: Optional[Exception] = None
    for context_name in context_names:
        ctx = multiprocessing.get_context(context_name)
        job.metadata["live_delegation_context"] = context_name
        candidate = ctx.Process(
            target=_child_execute_swarm,
            args=(job, routing_plan, delegate_fn),
            kwargs={"store_base_dir": str(store.base_dir), "max_children": max_children},
            daemon=True,
        )
        try:
            candidate.start()
        except Exception as exc:
            last_start_error = exc
            continue
        process = candidate
        job.metadata["live_delegation_context"] = context_name
        break

    if process is None:
        error = str(last_start_error) if last_start_error is not None else "no process context could start"
        job.metadata["live_delegation_status"] = "blocked"
        job.metadata["live_delegation_block_reason"] = "process_start_failed"
        job.metadata["live_delegation_error"] = error
        job.audit.append(
            AuditEvent(
                "live_delegation_blocked",
                "Swarm live delegation child process failed to start.",
                metadata={"reason": "process_start_failed", "error": error},
            )
        )
        store.save_job(job)
        return SwarmLiveRunHandle(started=False, status="blocked", reason="process_start_failed", job_id=job.job_id)

    job.metadata["live_delegation_pid"] = process.pid

    monitor = threading.Thread(
        target=_monitor_process,
        kwargs={"process": process, "store_base_dir": str(store.base_dir), "job": job, "timeout_seconds": timeout},
        name=f"swarm-live-monitor-{job.job_id}",
        daemon=True,
    )
    monitor.start()

    return SwarmLiveRunHandle(started=True, status="running", job_id=job.job_id, pid=process.pid)


__all__ = ["SwarmLiveRunHandle", "start_live_swarm"]
