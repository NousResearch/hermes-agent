"""Compact Work Ledger control surface derived from Kanban live truth.

This module is intentionally read-only.  It summarizes Kanban tasks, task_runs,
review/closeout governance fields, and authority/evidence pointers without
creating a second source of truth.  Linear/session archaeology is not consulted.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from hermes_cli import kanban_db as kb

CONTROL_SURFACE_AUTHORITY = (
    "projection_only; authorities=kanban.tasks,kanban.task_runs,"
    "kanban.review_phase,kanban.closeout_evidence,github_links_in_evidence"
)

BUCKETS = ("active", "blocked", "stale", "failed", "worker_done", "review_ready", "closed")

_SECRET_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*[^\s,;]+"),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(r"(?i)([?&](?:access_)?token=)[^&\s]+"),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{12,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)sk-[a-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"(?i)xox[baprs]-[a-z0-9-]{8,}"), "[REDACTED]"),
    (
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])", re.IGNORECASE),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (re.compile(r"<@!?\d{12,}>"), "<@[REDACTED]>"),
)


def redact_secret_text(value: Any, *, limit: int = 120) -> str:
    """Return compact, secret-safe display text for untrusted ledger fields."""

    text = "" if value is None else str(value).replace("\n", " ").strip()
    for pattern, replacement in _SECRET_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _metadata_work_state(run: Optional[kb.Run]) -> str:
    if not run or not isinstance(run.metadata, dict):
        return ""
    state = run.metadata.get("state")
    if not state and isinstance(run.metadata.get("work_state"), dict):
        state = run.metadata["work_state"].get("state")
    return _norm(state)


def _metadata_next_action(run: Optional[kb.Run]) -> Optional[str]:
    if not run or not isinstance(run.metadata, dict):
        return None
    for key in ("next_action", "proof"):
        value = run.metadata.get(key)
        if value:
            return str(value)
    work_state = run.metadata.get("work_state")
    if isinstance(work_state, dict):
        for key in ("next_action", "proof"):
            value = work_state.get(key)
            if value:
                return str(value)
    return None


def _evidence_links(task: kb.Task, latest_run: Optional[kb.Run]) -> list[str]:
    links: list[str] = []
    closeout = task.closeout_evidence if isinstance(task.closeout_evidence, dict) else {}
    github = closeout.get("github") if isinstance(closeout.get("github"), dict) else {}
    for key in ("pr_url", "url", "checks_url"):
        value = github.get(key) if isinstance(github, dict) else None
        if value:
            links.append(str(value))
    for key in ("pr_url", "evidence_url", "url"):
        value = closeout.get(key)
        if value:
            links.append(str(value))
    if latest_run:
        links.append(f"run#{latest_run.id}:{latest_run.status or '-'}:{latest_run.outcome or '-'}")
    if task.current_run_id:
        links.append(f"current_run#{task.current_run_id}")
    # Preserve order while deduping.
    seen: set[str] = set()
    out: list[str] = []
    for link in links:
        safe = redact_secret_text(link, limit=160)
        if safe and safe not in seen:
            seen.add(safe)
            out.append(safe)
    return out[:4]


def _task_authorities(task: kb.Task, latest_run: Optional[kb.Run]) -> list[str]:
    authorities = ["kanban.tasks"]
    if latest_run:
        authorities.append("kanban.task_runs")
    if task.review_phase or task.closeout_evidence:
        authorities.append("kanban.closeout_governance")
    closeout = task.closeout_evidence if isinstance(task.closeout_evidence, dict) else {}
    if closeout.get("github") or closeout.get("pr_url"):
        authorities.append("github.evidence_link")
    return authorities


@dataclass(frozen=True)
class LedgerEntry:
    task_id: str
    public_id: Optional[str]
    title: str
    tenant: Optional[str]
    owner: str
    status: str
    phase: Optional[str]
    queue_labels: tuple[str, ...]
    priority: int
    age_seconds: int
    next_action: str
    authorities: tuple[str, ...]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "public_id": self.public_id,
            "title": self.title,
            "tenant": self.tenant,
            "owner": self.owner,
            "status": self.status,
            "phase": self.phase,
            "queue_labels": list(self.queue_labels),
            "priority": self.priority,
            "age_seconds": self.age_seconds,
            "next_action": self.next_action,
            "authorities": list(self.authorities),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class LedgerSurface:
    generated_at: int
    authority: str
    tenant: Optional[str]
    entries: tuple[LedgerEntry, ...]
    queues: dict[str, tuple[LedgerEntry, ...]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {bucket: len(self.queues.get(bucket, ())) for bucket in BUCKETS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "authority": self.authority,
            "tenant": self.tenant,
            "counts": self.counts(),
            "queues": {
                bucket: [entry.to_dict() for entry in self.queues.get(bucket, ())]
                for bucket in BUCKETS
            },
        }


def _queue_labels(task: kb.Task, latest_run: Optional[kb.Run], *, now: int) -> tuple[str, ...]:
    labels: list[str] = []
    run_state = _metadata_work_state(latest_run)
    run_status = _norm(latest_run.status if latest_run else None)
    run_outcome = _norm(latest_run.outcome if latest_run else None)

    if task.status in {"ready", "running", "todo"}:
        labels.append("active")
    if task.status == "blocked":
        labels.append("blocked")
    if task.claim_expires and task.claim_expires < now and task.status == "running":
        labels.append("stale")
    if run_state in {"stale", "handoff_needed", "retry_needed"}:
        labels.append("stale")
    if task.last_failure_error or task.consecutive_failures > 0:
        labels.append("failed")
    if run_status == "failed" or run_outcome in {"failed", "gave_up", "crashed", "timed_out"} or run_state == "failed":
        labels.append("failed")
    if task.review_phase == "worker_done":
        labels.append("worker_done")
    if task.review_phase == "review_ready":
        labels.append("review_ready")
    if task.review_phase == "closed" or task.status in {"done", "archived"}:
        labels.append("closed")

    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return tuple(label for label in BUCKETS if label in deduped)


def _next_action(task: kb.Task, latest_run: Optional[kb.Run], labels: tuple[str, ...]) -> str:
    from_run = _metadata_next_action(latest_run)
    if from_run:
        return redact_secret_text(from_run, limit=120)
    if task.last_failure_error:
        return redact_secret_text(task.last_failure_error, limit=120)
    if "review_ready" in labels:
        return "review PR/checks/evidence; merge requires explicit approval"
    if "worker_done" in labels:
        return "verify PR/checks/evidence/cleanup before review_ready"
    if "failed" in labels:
        return "inspect latest run/error and decide retry or blocker"
    if "stale" in labels:
        return "recover stale worker/session or mark blocked with proof"
    if "blocked" in labels:
        return "resolve blocker recorded in Kanban/run evidence"
    if "active" in labels:
        return "continue execution from Kanban task/run state"
    if "closed" in labels:
        return "no action"
    return "inspect task"


def build_control_surface(
    conn: sqlite3.Connection,
    *,
    tenant: Optional[str] = None,
    include_archived: bool = False,
    limit_per_bucket: int = 8,
    now: Optional[int] = None,
) -> LedgerSurface:
    """Build a read-only daily Work Ledger control surface."""

    ts = int(time.time()) if now is None else int(now)
    kb.recompute_ready(conn)
    tasks = kb.list_tasks(conn, tenant=tenant, include_archived=include_archived)
    entries: list[LedgerEntry] = []
    queues: dict[str, list[LedgerEntry]] = {bucket: [] for bucket in BUCKETS}

    for task in tasks:
        latest_run = kb.latest_run(conn, task.id)
        labels = _queue_labels(task, latest_run, now=ts)
        if not labels:
            continue
        entry = LedgerEntry(
            task_id=task.id,
            public_id=task.public_id,
            title=redact_secret_text(task.title, limit=96),
            tenant=task.tenant,
            owner=redact_secret_text(task.assignee or "unassigned", limit=40),
            status=task.status,
            phase=task.review_phase,
            queue_labels=labels,
            priority=task.priority,
            age_seconds=max(0, ts - int(task.created_at or ts)),
            next_action=_next_action(task, latest_run, labels),
            authorities=tuple(_task_authorities(task, latest_run)),
            evidence=tuple(_evidence_links(task, latest_run)),
        )
        entries.append(entry)
        for label in labels:
            queues[label].append(entry)

    ordered_queues: dict[str, tuple[LedgerEntry, ...]] = {}
    for bucket in BUCKETS:
        bucket_entries = sorted(
            queues[bucket],
            key=lambda e: (-e.priority, e.age_seconds, e.task_id),
        )
        ordered_queues[bucket] = tuple(bucket_entries[: max(0, int(limit_per_bucket))])

    return LedgerSurface(
        generated_at=ts,
        authority=CONTROL_SURFACE_AUTHORITY,
        tenant=tenant,
        entries=tuple(entries),
        queues=ordered_queues,
    )


def format_control_surface(surface: LedgerSurface) -> str:
    """Format a compact human-readable Work Ledger control surface."""

    counts = surface.counts()
    lines = [
        "Work Ledger control surface",
        f"authority: {surface.authority}",
        "counts: " + " ".join(f"{bucket}={counts[bucket]}" for bucket in BUCKETS),
    ]
    if surface.tenant:
        lines.append(f"tenant: {surface.tenant}")
    for bucket in BUCKETS:
        entries = surface.queues.get(bucket, ())
        lines.append(f"\n[{bucket}] {len(entries)}")
        if not entries:
            lines.append("  - (empty)")
            continue
        for entry in entries:
            public = f"/{entry.public_id}" if entry.public_id else ""
            phase = entry.phase or "-"
            auth = "+".join(entry.authorities)
            evidence = ", ".join(entry.evidence) if entry.evidence else "-"
            lines.append(
                f"  - {entry.task_id}{public} [{entry.tenant or '-'}] "
                f"owner={entry.owner} status={entry.status} phase={phase} "
                f"auth={auth} next={entry.next_action} evidence={evidence} :: {entry.title}"
            )
    return "\n".join(lines)
