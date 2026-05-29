"""
Gateway job tracking system.

Tracks work items per session with workstream classification,
matter_summary, and status. Supports modifier attachment,
restart persistence, and the /interrupt /cancel /queue /status
command surface.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

# ── Workstream taxonomy (must match operating-rules.md §4) ──────────
FIXED_WORKSTREAMS = frozenset({
    "hermes_ops",
    "spring_reit_work",
    "personal_family",
    "research",
    "drafting_writing",
    "coding_debug",
    "finance_analysis",
    "misc",
})

JOB_STATUSES = frozenset({
    "active",      # currently running or about to run
    "queued",      # waiting behind an active job
    "paused",      # interrupted mid-run, waiting to resume
    "cancelled",   # explicitly cancelled
    "completed",   # finished normally
})


@dataclass
class GatewayJob:
    """A single tracked work item within a session."""

    job_id: str
    session_key: str
    workstream: str
    matter_summary: str
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Lightweight event reference (no full MessageEvent for serialization)
    trigger_text: str = ""
    reply_to_message_id: Optional[str] = None
    # For modifier attachment: which job was this attached to
    parent_job_id: Optional[str] = None
    # For /interrupt: saved context to resume
    paused_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_key": self.session_key,
            "workstream": self.workstream,
            "matter_summary": self.matter_summary,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "trigger_text": self.trigger_text,
            "reply_to_message_id": self.reply_to_message_id,
            "parent_job_id": self.parent_job_id,
            "paused_context": self.paused_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayJob":
        return cls(
            job_id=data["job_id"],
            session_key=data["session_key"],
            workstream=data.get("workstream", "misc"),
            matter_summary=data.get("matter_summary", ""),
            status=data.get("status", "active"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            trigger_text=data.get("trigger_text", ""),
            reply_to_message_id=data.get("reply_to_message_id"),
            parent_job_id=data.get("parent_job_id"),
            paused_context=data.get("paused_context"),
        )

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_active_or_queued(self) -> bool:
        return self.status in {"active", "queued", "paused"}

    def touch(self) -> None:
        self.updated_at = time.time()


class GatewayJobs:
    """
    Per-session job tracker.

    Holds all jobs (active, queued, paused, completed, cancelled) for each
    session.  Provides modifier-attachment scoring and restart persistence.

    Persisted to ``~/.hermes/gateway/jobs.json`` on every mutation so jobs
    survive gateway restarts.
    """

    _instance: Optional["GatewayJobs"] = None

    def __init__(self):
        self._jobs: Dict[str, List[GatewayJob]] = {}           # session_key → jobs
        self._active_job: Dict[str, str] = {}                  # session_key → job_id
        self._completed_jobs: Dict[str, List[GatewayJob]] = {} # session_key → (trimmed)

    # ── Singleton ────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "GatewayJobs":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Persistence ──────────────────────────────────────────────
    @property
    def _store_path(self) -> Path:
        return get_hermes_home() / "gateway" / "jobs.json"

    def save(self) -> None:
        """Persist current state to disk (atomic write)."""
        data: Dict[str, Any] = {}
        for session_key, jobs in self._jobs.items():
            data[session_key] = [j.to_dict() for j in jobs]
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self._store_path)

    def load(self) -> None:
        """Restore state from disk. Silently no-ops if file missing."""
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for session_key, job_list in data.items():
            jobs = [GatewayJob.from_dict(j) for j in job_list]
            self._jobs[session_key] = jobs
            for job in jobs:
                if job.status == "active":
                    self._active_job[session_key] = job.job_id

    def recover_after_restart(self) -> Dict[str, List[GatewayJob]]:
        """Clean up state after a gateway restart.

        Active and paused jobs lose their agent context on restart, so they
        are demoted to ``queued``.  Already-queued jobs are promoted to
        ``active`` (first by creation time).  If no other queued jobs exist,
        the demoted job(s) stay ``queued``.

        Returns a dict of ``session_key → recovered_jobs`` for any sessions
        that had interrupted work, so the gateway can notify the user.
        """
        interrupted: Dict[str, List[GatewayJob]] = {}

        for session_key in list(self._jobs.keys()):
            recovered: List[GatewayJob] = []
            demoted_ids: set[str] = set()
            for job in self._jobs[session_key]:
                if job.status in {"active", "paused"}:
                    job.status = "queued"
                    job.touch()
                    recovered.append(job)
                    demoted_ids.add(job.job_id)
            # Clear stale active slot.
            self._active_job.pop(session_key, None)
            # Promote the first already-queued job (not one we just demoted).
            already_queued = sorted(
                [j for j in self._jobs[session_key]
                 if j.status == "queued" and j.job_id not in demoted_ids],
                key=lambda j: j.created_at,
            )
            if already_queued:
                next_job = already_queued[0]
                next_job.status = "active"
                next_job.touch()
                self._active_job[session_key] = next_job.job_id
            if recovered:
                interrupted[session_key] = recovered

        if interrupted:
            self.save()
        return interrupted

    # ── CRUD ─────────────────────────────────────────────────────
    def create(
        self,
        session_key: str,
        workstream: str,
        matter_summary: str,
        trigger_text: str = "",
        reply_to_message_id: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        status: str = "queued",
    ) -> GatewayJob:
        """Create a new job. If no active job, it becomes active."""
        job = GatewayJob(
            job_id=uuid.uuid4().hex[:12],
            session_key=session_key,
            workstream=workstream if workstream in FIXED_WORKSTREAMS else "misc",
            matter_summary=matter_summary,
            status=status,
            trigger_text=trigger_text[:500] if trigger_text else "",
            reply_to_message_id=reply_to_message_id,
            parent_job_id=parent_job_id,
        )
        self._jobs.setdefault(session_key, []).append(job)
        # If no active job, promote this one
        if session_key not in self._active_job:
            self._active_job[session_key] = job.job_id
            job.status = "active"
        self.save()
        return job

    def get_job(self, session_key: str, job_id: str) -> Optional[GatewayJob]:
        for job in self._jobs.get(session_key, []):
            if job.job_id == job_id:
                return job
        return None

    def list_for_session(
        self,
        session_key: str,
        status_filter: Optional[str] = None,
    ) -> List[GatewayJob]:
        """Return jobs for a session, optionally filtered by status."""
        jobs = self._jobs.get(session_key, [])
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        return jobs

    def active_job(self, session_key: str) -> Optional[GatewayJob]:
        """Return the currently active job, if any."""
        job_id = self._active_job.get(session_key)
        if not job_id:
            return None
        return self.get_job(session_key, job_id)

    def active_or_queued_jobs(self, session_key: str) -> List[GatewayJob]:
        """Jobs that are still pending (active, queued, paused)."""
        return [
            j for j in self._jobs.get(session_key, [])
            if j.is_active_or_queued
        ]

    def set_status(self, session_key: str, job_id: str, status: str) -> bool:
        """Transition a job to a new status."""
        if status not in JOB_STATUSES:
            return False
        job = self.get_job(session_key, job_id)
        if not job:
            return False
        job.status = status
        job.touch()
        # If completing/cancelling the active job, clear active slot
        if status in {"completed", "cancelled"} and self._active_job.get(session_key) == job_id:
            del self._active_job[session_key]
            # Promote next queued job
            next_job = self._next_queued(session_key)
            if next_job:
                self._active_job[session_key] = next_job.job_id
                next_job.status = "active"
                next_job.touch()
        # If pausing the active job, keep it active but mark paused
        elif status == "paused" and self._active_job.get(session_key) == job_id:
            pass  # active slot stays
        self.save()
        return True

    def _next_queued(self, session_key: str) -> Optional[GatewayJob]:
        """Return the first queued job by creation time."""
        queued = [
            j for j in self._jobs.get(session_key, [])
            if j.status == "queued"
        ]
        queued.sort(key=lambda j: j.created_at)
        return queued[0] if queued else None

    def cancel(self, session_key: str, job_id: str) -> bool:
        return self.set_status(session_key, job_id, "cancelled")

    def complete(self, session_key: str, job_id: str) -> bool:
        return self.set_status(session_key, job_id, "completed")

    def pause(self, session_key: str, job_id: str) -> bool:
        return self.set_status(session_key, job_id, "paused")

    def resume(self, session_key: str, job_id: str) -> bool:
        """Resume a paused job — re-activates it and clears pause context."""
        job = self.get_job(session_key, job_id)
        if not job or job.status != "paused":
            return False
        job.status = "active"
        job.paused_context = None
        job.touch()
        self._active_job[session_key] = job_id
        self.save()
        return True

    # ── Modifier attachment ───────────────────────────────────────
    def find_matching_jobs(
        self,
        session_key: str,
        workstream: str,
        matter_hint: str = "",
    ) -> List[GatewayJob]:
        """
        Return pending jobs that could be the target of a modifier message,
        scored by workstream match + matter keyword overlap.

        Only returns active/queued/paused jobs.
        """
        candidates = self.active_or_queued_jobs(session_key)
        if not candidates:
            return []

        scored: List[tuple[GatewayJob, int]] = []
        for job in candidates:
            score = 0
            # Exact workstream match = high score
            if job.workstream == workstream:
                score += 10
            elif job.workstream == "misc" or workstream == "misc":
                score += 3  # weak match
            # Keyword overlap in matter_summary
            if matter_hint:
                hint_words = set(matter_hint.lower().split())
                summary_words = set(job.matter_summary.lower().split())
                overlap = hint_words & summary_words
                score += len(overlap) * 2
            scored.append((job, score))

        # Sort by score descending, then by most recently updated
        scored.sort(key=lambda x: (-x[1], -x[0].updated_at))
        return [job for job, _ in scored]

    def best_match(
        self,
        session_key: str,
        workstream: str,
        matter_hint: str = "",
    ) -> Optional[GatewayJob]:
        """Return the single best matching job, or None."""
        matches = self.find_matching_jobs(session_key, workstream, matter_hint)
        return matches[0] if matches else None

    def attach_modifier(
        self,
        session_key: str,
        target_job_id: str,
        trigger_text: str,
        workstream: Optional[str] = None,
    ) -> Optional[GatewayJob]:
        """Record a modifier message attached to an existing job.

        Creates a child job linked via parent_job_id so the modifier is tracked
        as a distinct entry but routed to the parent.
        """
        parent = self.get_job(session_key, target_job_id)
        if not parent:
            return None
        child = GatewayJob(
            job_id=uuid.uuid4().hex[:12],
            session_key=session_key,
            workstream=workstream or parent.workstream,
            matter_summary=f"modifier: {trigger_text[:80]}",
            status="queued",
            trigger_text=trigger_text[:500],
            parent_job_id=target_job_id,
        )
        self._jobs.setdefault(session_key, []).append(child)
        parent.touch()
        self.save()
        return child

    # ── Cleanup ───────────────────────────────────────────────────
    def clear_session(self, session_key: str) -> None:
        """Remove all jobs for a session (on /new /reset)."""
        self._jobs.pop(session_key, None)
        self._completed_jobs.pop(session_key, None)
        self._active_job.pop(session_key, None)
        self.save()

    def trim_completed(self, session_key: str, max_completed: int = 20) -> None:
        """Keep only the most recent N completed jobs per session."""
        completed = [
            j for j in self._jobs.get(session_key, [])
            if j.status in {"completed", "cancelled"}
        ]
        if len(completed) <= max_completed:
            return
        completed.sort(key=lambda j: j.updated_at, reverse=True)
        keep_ids = {j.job_id for j in completed[:max_completed]}
        self._jobs[session_key] = [
            j for j in self._jobs[session_key]
            if j.status not in {"completed", "cancelled"} or j.job_id in keep_ids
        ]
        self.save()

    # ── Status summary ───────────────────────────────────────────
    def status_summary(self, session_key: str) -> str:
        """Human-readable status of all jobs for a session."""
        jobs = self._jobs.get(session_key, [])
        if not jobs:
            return "No tracked jobs for this session."

        lines = []
        active = [j for j in jobs if j.status == "active"]
        queued = [j for j in jobs if j.status == "queued"]
        paused = [j for j in jobs if j.status == "paused"]

        if active:
            for j in active:
                lines.append(f"▶ [{j.workstream}] {j.matter_summary}")
        if paused:
            for j in paused:
                lines.append(f"⏸ [{j.workstream}] {j.matter_summary}")
        if queued:
            for j in queued:
                lines.append(f"⏳ [{j.workstream}] {j.matter_summary}")

        return "\n".join(lines) if lines else "No active or queued jobs."
