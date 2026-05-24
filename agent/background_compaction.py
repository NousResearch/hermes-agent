"""Stale-safe background compaction job support.

This module intentionally does not mutate live conversation state. It runs rich
summary work off the foreground turn and stores only fresh results whose caller
supplied staleness guard still passes.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Optional
import uuid


@dataclass(frozen=True)
class CompactionSnapshot:
    """Immutable identity for a background compaction candidate."""

    session_id: str
    parent_session_id: str = ""
    generation: int = 0
    message_count: int = 0
    message_hash: str = ""
    focus_topic: str = ""


@dataclass
class BackgroundCompactionResult:
    """Result from a background compaction job."""

    job_id: str
    snapshot: CompactionSnapshot
    summary: str = ""
    applied: bool = False
    stale: bool = False
    error: str = ""


@dataclass
class BackgroundCompactionJob:
    """Handle returned immediately after scheduling background compaction."""

    job_id: str
    snapshot: CompactionSnapshot
    future: Future

    @property
    def done(self) -> bool:
        return self.future.done()


SummaryBuilder = Callable[[CompactionSnapshot], str]
CurrentGuard = Callable[[CompactionSnapshot], bool]


class BackgroundCompactionManager:
    """Small in-process manager for stale-safe rich compaction summaries."""

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hermes-bg-compact")
        self._jobs: dict[str, BackgroundCompactionJob] = {}
        self._results: dict[str, BackgroundCompactionResult] = {}
        self._latest_by_session: dict[str, BackgroundCompactionResult] = {}
        self._lock = Lock()

    def schedule(
        self,
        snapshot: CompactionSnapshot,
        build_summary: SummaryBuilder,
        *,
        is_current: CurrentGuard,
    ) -> BackgroundCompactionJob:
        """Schedule rich summary work and return immediately.

        The `is_current` guard is checked after summary generation. If the
        session/generation/hash has gone stale, the result is recorded as stale
        and is not promoted to latest-by-session.
        """
        job_id = uuid.uuid4().hex
        future = self._executor.submit(self._run_job, job_id, snapshot, build_summary, is_current)
        job = BackgroundCompactionJob(job_id=job_id, snapshot=snapshot, future=future)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def _run_job(
        self,
        job_id: str,
        snapshot: CompactionSnapshot,
        build_summary: SummaryBuilder,
        is_current: CurrentGuard,
    ) -> BackgroundCompactionResult:
        result = BackgroundCompactionResult(job_id=job_id, snapshot=snapshot)
        try:
            result.summary = build_summary(snapshot) or ""
            if not is_current(snapshot):
                result.stale = True
                result.applied = False
            else:
                result.applied = bool(result.summary)
        except Exception as exc:  # pragma: no cover - exact exception type is caller-specific
            result.error = str(exc)[:500]
            result.applied = False
        with self._lock:
            self._results[job_id] = result
            if result.applied and not result.stale:
                self._latest_by_session[snapshot.session_id] = result
        return result

    def wait(self, job_id: str, *, timeout: Optional[float] = None) -> Optional[BackgroundCompactionResult]:
        """Wait for a job result for tests/controlled callers."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        try:
            return job.future.result(timeout=timeout)
        except TimeoutError:
            return None

    def latest_for_session(self, session_id: str) -> Optional[BackgroundCompactionResult]:
        """Return latest fresh applied background result for a session."""
        with self._lock:
            return self._latest_by_session.get(session_id)

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
