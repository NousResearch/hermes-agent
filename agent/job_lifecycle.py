from __future__ import annotations

"""Shared lifecycle-state helpers for long-running Hermes work.

This is the first common state model spanning cron jobs, background
processes, and future async task runners. It gives the runtime one stable
vocabulary for "where is this work right now?" without forcing every legacy
surface to give up its existing status strings immediately.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


AttemptRecord = dict[str, Any]
RunnerMetadata = dict[str, Any]
MAX_ATTEMPT_HISTORY = 10


class JobLifecycleState(str, Enum):
    queued = "queued"
    running = "running"
    retrying = "retrying"
    failed = "failed"
    completed = "completed"
    cancelled = "cancelled"
    paused = "paused"
    unknown = "unknown"


@dataclass(frozen=True)
class LifecycleSnapshot:
    state: JobLifecycleState
    terminal: bool = False
    retryable: bool = False
    detail: str = ""


def process_lifecycle_state(*, exited: bool, exit_code: Optional[int]) -> JobLifecycleState:
    """Map background-process state to the shared lifecycle vocabulary."""
    if not exited:
        return JobLifecycleState.running
    if exit_code in (-15, 130, 143):
        return JobLifecycleState.cancelled
    if exit_code == 0 or exit_code is None:
        return JobLifecycleState.completed
    return JobLifecycleState.failed


def cron_lifecycle_state(job: Mapping[str, Any]) -> JobLifecycleState:
    """Infer the shared lifecycle state for a cron job record.

    Legacy cron jobs keep their existing ``state`` field (scheduled/paused/
    completed/error). This helper projects that storage shape into the new
    cross-runtime lifecycle vocabulary without breaking old readers.
    """
    raw_state = str(job.get("state") or "").strip().lower()
    enabled = bool(job.get("enabled", True))
    next_run_at = job.get("next_run_at")
    last_status = str(job.get("last_status") or "").strip().lower()

    if raw_state == "completed":
        return JobLifecycleState.failed if last_status == "error" else JobLifecycleState.completed
    if raw_state == "paused" or not enabled:
        return JobLifecycleState.paused
    if raw_state == "running":
        return JobLifecycleState.running
    if raw_state == "error":
        return JobLifecycleState.failed
    if next_run_at:
        if last_status == "error":
            return JobLifecycleState.retrying
        return JobLifecycleState.queued
    if last_status == "error":
        return JobLifecycleState.failed
    if last_status == "ok":
        return JobLifecycleState.completed
    return JobLifecycleState.unknown


def lifecycle_snapshot_for_process(*, exited: bool, exit_code: Optional[int]) -> LifecycleSnapshot:
    state = process_lifecycle_state(exited=exited, exit_code=exit_code)
    return LifecycleSnapshot(
        state=state,
        terminal=state in {JobLifecycleState.completed, JobLifecycleState.failed, JobLifecycleState.cancelled},
        retryable=False,
    )


def lifecycle_snapshot_for_cron(job: Mapping[str, Any]) -> LifecycleSnapshot:
    state = cron_lifecycle_state(job)
    retryable = state == JobLifecycleState.retrying
    terminal = state in {JobLifecycleState.completed, JobLifecycleState.failed, JobLifecycleState.cancelled}
    if state == JobLifecycleState.queued:
        terminal = False
    if state == JobLifecycleState.paused:
        terminal = False
    return LifecycleSnapshot(state=state, terminal=terminal, retryable=retryable)


def make_attempt_record(
    *,
    attempt: int,
    state: JobLifecycleState | str,
    started_at: Optional[str],
    finished_at: Optional[str] = None,
    error: Optional[str] = None,
    retry_count: int = 0,
) -> AttemptRecord:
    lifecycle_state = state.value if isinstance(state, JobLifecycleState) else str(state)
    return {
        "attempt": max(1, int(attempt)),
        "lifecycle_state": lifecycle_state,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "retry_count": max(0, int(retry_count)),
    }


def normalize_attempt_record(record: Any) -> Optional[AttemptRecord]:
    if not isinstance(record, Mapping):
        return None
    try:
        attempt = int(record.get("attempt", 1) or 1)
    except Exception:
        attempt = 1
    try:
        retry_count = int(record.get("retry_count", 0) or 0)
    except Exception:
        retry_count = 0
    state = str(record.get("lifecycle_state") or JobLifecycleState.unknown.value).strip() or JobLifecycleState.unknown.value
    started_at = record.get("started_at")
    finished_at = record.get("finished_at")
    error = record.get("error")
    return make_attempt_record(
        attempt=attempt,
        state=state,
        started_at=str(started_at) if started_at else None,
        finished_at=str(finished_at) if finished_at else None,
        error=str(error) if error is not None else None,
        retry_count=retry_count,
    )


def default_queue_metadata(*, queued_at: Optional[str] = None) -> dict[str, Any]:
    return {
        "retry_count": 0,
        "retry_backoff_seconds": 0,
        "next_retry_at": None,
        "current_attempt": None,
        "last_attempt": None,
        "attempt_history": [],
        "last_queued_at": queued_at,
        "runner": default_runner_metadata(kind="cron", queue_name="default"),
    }


def default_runner_metadata(
    *,
    kind: str = "cron",
    queue_name: str = "default",
    priority: int = 0,
) -> RunnerMetadata:
    return {
        "kind": str(kind or "cron"),
        "queue_name": str(queue_name or "default"),
        "priority": int(priority or 0),
        "active": False,
        "claimed_at": None,
        "lease_expires_at": None,
        "worker_id": None,
        "last_started_at": None,
        "last_finished_at": None,
    }


def normalize_runner_metadata(
    runner: Any,
    *,
    kind: str = "cron",
    queue_name: str = "default",
    priority: int = 0,
) -> RunnerMetadata:
    normalized = default_runner_metadata(kind=kind, queue_name=queue_name, priority=priority)
    if not isinstance(runner, Mapping):
        return normalized
    normalized["kind"] = str(runner.get("kind") or normalized["kind"])
    normalized["queue_name"] = str(runner.get("queue_name") or normalized["queue_name"])
    try:
        normalized["priority"] = int(runner.get("priority", normalized["priority"]) or 0)
    except Exception:
        normalized["priority"] = int(priority or 0)
    normalized["active"] = bool(runner.get("active", False))
    for key in ("claimed_at", "lease_expires_at", "worker_id", "last_started_at", "last_finished_at"):
        value = runner.get(key)
        normalized[key] = str(value) if value not in (None, "") else None
    return normalized


def append_attempt_history(history: Any, attempt: AttemptRecord, *, max_entries: int = MAX_ATTEMPT_HISTORY) -> list[AttemptRecord]:
    normalized_history: list[AttemptRecord] = []
    if isinstance(history, list):
        for item in history:
            normalized_item = normalize_attempt_record(item)
            if normalized_item is not None:
                normalized_history.append(normalized_item)
    normalized_history.append(normalize_attempt_record(attempt) or attempt)
    if max_entries > 0:
        normalized_history = normalized_history[-max_entries:]
    return normalized_history


def normalize_queue_metadata(queue: Any, *, queued_at: Optional[str] = None) -> dict[str, Any]:
    normalized = default_queue_metadata(queued_at=queued_at)
    if isinstance(queue, Mapping):
        try:
            normalized["retry_count"] = max(0, int(queue.get("retry_count", 0) or 0))
        except Exception:
            normalized["retry_count"] = 0
        try:
            normalized["retry_backoff_seconds"] = max(0, int(queue.get("retry_backoff_seconds", 0) or 0))
        except Exception:
            normalized["retry_backoff_seconds"] = 0
        normalized["next_retry_at"] = queue.get("next_retry_at")
        normalized["current_attempt"] = normalize_attempt_record(queue.get("current_attempt"))
        normalized["last_attempt"] = normalize_attempt_record(queue.get("last_attempt"))
        history = queue.get("attempt_history")
        if isinstance(history, list):
            normalized["attempt_history"] = [
                item for item in (normalize_attempt_record(entry) for entry in history) if item is not None
            ][-MAX_ATTEMPT_HISTORY:]
        last_queued_at = queue.get("last_queued_at")
        if last_queued_at:
            normalized["last_queued_at"] = str(last_queued_at)
        normalized["runner"] = normalize_runner_metadata(queue.get("runner"))
    return normalized
