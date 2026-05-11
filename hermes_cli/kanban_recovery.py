"""Read-only Kanban blocker/crash recovery reporting.

This module turns blocked/failing board rows into a compact operator report.
It deliberately does not mutate tasks: recommendations are safe hints unless a
human chooses to run the suggested command or create a bounded retry task.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

RECOVERY_CATEGORIES = (
    "human_decision_required",
    "worker_crash_or_pid_dead",
    "auth_or_credential_issue",
    "missing_artifact_or_path",
    "stale_or_superseded",
    "unknown_needs_review",
)

_BANNER_PATTERNS = (
    re.compile(r"^╔[═=].*", re.MULTILINE),
    re.compile(r"^║.*", re.MULTILINE),
    re.compile(r"^╚[═=].*", re.MULTILINE),
    re.compile(r"^\s*(?:Available tools|Tools|Toolsets|Skills|Model|Provider):.*$", re.I | re.M),
    re.compile(r"^\s*[-•]\s*(?:browser_|terminal|read_file|write_file|kanban_|discord_|web_|search_|todo\b).*$", re.I | re.M),
    re.compile(r"^\s*\{\s*\"name\"\s*:\s*\"[a-z0-9_]+\".*$", re.I | re.M),
)

_AUTH_RE = re.compile(
    r"\b(auth|oauth|credential|token|api[_ -]?key|permission denied|forbidden|unauthorized|401|403|login required)\b",
    re.I,
)
_CRASH_RE = re.compile(
    r"\b(pid\s+\d+\s+not\s+alive|not alive|crashed|segmentation fault|sigkill|sigterm|oom|out of memory|traceback|spawn_failed|timed_out|timeout|exit code\s*-?\d+)\b",
    re.I,
)
_MISSING_RE = re.compile(
    r"\b(no such file|file not found|missing (?:artifact|file|path|workspace)|does not exist|not found|enoent|cannot find|couldn't find|not a directory)\b",
    re.I,
)
_STALE_RE = re.compile(
    r"\b(stale|superseded|obsolete|duplicate|already done|already completed|no longer needed|archived|replaced by)\b",
    re.I,
)
_HUMAN_RE = re.compile(
    r"\b(need(?:s)? (?:human|user|matthew|operator|approval|decision|input|clarification)|blocked on|waiting for|please confirm|which should|choose|approve|credential from human)\b",
    re.I,
)


@dataclass
class RecoveryItem:
    task_id: str
    title: str
    status: str
    assignee: str
    category: str
    confidence: float
    summary: str
    recommendation: str
    retry_task_spec: Optional[dict[str, Any]] = None
    evidence: list[str] = field(default_factory=list)
    run_outcomes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "assignee": self.assignee,
            "category": self.category,
            "confidence": self.confidence,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "retry_task_spec": self.retry_task_spec,
            "evidence": self.evidence,
            "run_outcomes": self.run_outcomes,
        }


def _field(obj: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(obj, "keys") and name in obj.keys():
            return obj[name]
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def sanitize_child_stdout(text: str, *, max_chars: int = 900) -> str:
    """Remove Hermes banners/tool dumps and return a bounded plain summary."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    for pattern in _BANNER_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    lines: list[str] = []
    skipped = 0
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if (
            len(line) > 240 and ("tool" in lower or "schema" in lower)
        ) or lower.startswith(("usage:", "options:")):
            skipped += 1
            continue
        lines.append(line)

    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    if skipped:
        compact = f"{compact}\n[omitted {skipped} banner/tool/help line(s)]".strip()
    if len(compact) > max_chars:
        compact = compact[: max_chars - 1].rstrip() + "…"
    return compact


def _event_payload(ev: Any) -> dict[str, Any]:
    payload = _field(ev, "payload", {})
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _collect_text(task: Any, events: Iterable[Any], runs: Iterable[Any]) -> str:
    parts = [
        str(_field(task, "title", "") or ""),
        str(_field(task, "body", "") or ""),
        str(_field(task, "result", "") or ""),
        str(_field(task, "last_failure_error", "") or ""),
    ]
    for run in runs:
        parts.extend([
            str(_field(run, "outcome", "") or ""),
            str(_field(run, "summary", "") or ""),
            str(_field(run, "error", "") or ""),
        ])
    for ev in events:
        if _field(ev, "kind") in {"blocked", "crashed", "spawn_failed", "reclaimed"}:
            parts.append(json.dumps(_event_payload(ev), ensure_ascii=False))
    return sanitize_child_stdout("\n".join(parts), max_chars=2400)


def classify_recovery(task: Any, events: list[Any], runs: list[Any]) -> tuple[str, float, list[str]]:
    text = _collect_text(task, events, runs)
    status = str(_field(task, "status", "") or "")
    failures = int(_field(task, "consecutive_failures", 0) or 0)
    outcomes = [str(_field(r, "outcome", "") or "") for r in runs]
    tail_outcomes = [o for o in outcomes[-5:] if o]
    evidence: list[str] = []

    def hit(regex: re.Pattern[str], label: str) -> bool:
        match = regex.search(text)
        if match:
            evidence.append(f"{label}: {match.group(0)[:120]}")
            return True
        return False

    if hit(_AUTH_RE, "auth"):
        return "auth_or_credential_issue", 0.88, evidence
    if hit(_MISSING_RE, "missing"):
        return "missing_artifact_or_path", 0.86, evidence
    if hit(_STALE_RE, "stale"):
        return "stale_or_superseded", 0.82, evidence
    if failures > 0 or any(o in {"crashed", "spawn_failed", "timed_out"} for o in tail_outcomes):
        if hit(_CRASH_RE, "crash") or failures > 0:
            if failures > 0:
                evidence.append(f"consecutive_failures={failures}")
            if tail_outcomes:
                evidence.append(f"recent_outcomes={','.join(tail_outcomes)}")
            return "worker_crash_or_pid_dead", 0.84, evidence
    if status == "blocked" and hit(_HUMAN_RE, "human"):
        return "human_decision_required", 0.78, evidence
    if status == "blocked":
        evidence.append("status=blocked without a known machine-failure signature")
        return "human_decision_required", 0.55, evidence
    return "unknown_needs_review", 0.35, evidence or ["no classifier signature matched"]


def _recommendation(category: str, task: Any) -> str:
    task_id = _field(task, "id")
    assignee = _field(task, "assignee") or "worker"
    if category == "worker_crash_or_pid_dead":
        return f"Inspect `hermes kanban log {task_id}` and `hermes kanban runs {task_id}`; reclaim only after fixing the profile/runtime cause, or create the bounded retry task below."
    if category == "auth_or_credential_issue":
        return f"Run `hermes -p {assignee} doctor` and refresh credentials for @{assignee}; do not retry until auth is fixed."
    if category == "missing_artifact_or_path":
        return "Verify the referenced workspace/artifact path exists or create a follow-up task to regenerate it; avoid blind retry."
    if category == "stale_or_superseded":
        return "Confirm the superseding task/artifact, then archive or complete with a concise superseded note."
    if category == "human_decision_required":
        return "Answer the blocker or unblock with a specific instruction; do not auto-retry without the requested decision."
    return "Open the task, read recent runs/comments/logs, then classify manually before reclaiming or reassigning."


def _retry_spec(category: str, task: Any, summary: str) -> Optional[dict[str, Any]]:
    if category != "worker_crash_or_pid_dead":
        return None
    task_id = _field(task, "id")
    assignee = _field(task, "assignee") or "worker"
    return {
        "title": f"Recover crashed Kanban task {task_id}",
        "assignee": assignee,
        "parents": [task_id],
        "body": textwrap.dedent(f"""
            Bounded recovery retry for {task_id}.

            Start by reading `hermes kanban show {task_id}`, `hermes kanban runs {task_id}`, and `hermes kanban log {task_id}`. Fix the crash cause first; do not repeat the same command blindly. If the task touches red-lane artifacts, report only and route to red-antonetta.

            Sanitized failure summary:
            {summary}
            """).strip(),
    }


def build_recovery_items(*, task_id: Optional[str] = None, include_ready_failures: bool = True) -> list[RecoveryItem]:
    with kb.connect() as conn:
        if task_id:
            tasks = [kb.get_task(conn, task_id)]
            if tasks[0] is None:
                raise ValueError(f"no such task: {task_id}")
        else:
            statuses = ("blocked", "running", "ready") if include_ready_failures else ("blocked", "running")
            placeholders = ",".join("?" for _ in statuses)
            tasks = conn.execute(
                f"SELECT * FROM tasks WHERE status IN ({placeholders}) AND status != 'archived'",
                statuses,
            ).fetchall()
        items: list[RecoveryItem] = []
        for task in tasks:
            if task is None:
                continue
            tid = _field(task, "id")
            runs = kb.list_runs(conn, tid)
            failures = int(_field(task, "consecutive_failures", 0) or 0)
            if _field(task, "status") != "blocked" and failures <= 0:
                continue
            events = kb.list_events(conn, tid)
            category, confidence, evidence = classify_recovery(task, events, runs)
            collected = _collect_text(task, events, runs)
            summary = collected.splitlines()[0] if collected else "No failure text captured."
            if len(summary) > 260:
                summary = summary[:259].rstrip() + "…"
            item = RecoveryItem(
                task_id=tid,
                title=_field(task, "title", "") or "(untitled)",
                status=_field(task, "status", "") or "?",
                assignee=_field(task, "assignee", "") or "(unassigned)",
                category=category,
                confidence=confidence,
                summary=summary,
                recommendation=_recommendation(category, task),
                retry_task_spec=_retry_spec(category, task, summary),
                evidence=evidence,
                run_outcomes=[str(_field(r, "outcome", "") or "") for r in runs[-5:]],
            )
            items.append(item)
    rank = {name: i for i, name in enumerate(RECOVERY_CATEGORIES)}
    items.sort(key=lambda i: (rank.get(i.category, 99), i.task_id))
    return items


def render_recovery_report(items: list[RecoveryItem], *, json_output: bool = False) -> str:
    if json_output:
        counts = {cat: 0 for cat in RECOVERY_CATEGORIES}
        for item in items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return json.dumps({"generated_at": int(time.time()), "counts": counts, "items": [i.to_dict() for i in items]}, indent=2, ensure_ascii=False)
    if not items:
        return "No blocked/failing Kanban tasks need recovery classification."
    counts: dict[str, int] = {}
    for item in items:
        counts[item.category] = counts.get(item.category, 0) + 1
    lines = ["Kanban recovery report", "", "Counts:"]
    for cat in RECOVERY_CATEGORIES:
        if counts.get(cat):
            lines.append(f"  {cat}: {counts[cat]}")
    lines.append("")
    for item in items:
        lines.extend([
            f"{item.task_id}  {item.status}  @{item.assignee}  [{item.category} {item.confidence:.2f}]",
            f"  {item.title}",
            f"  summary: {item.summary}",
            f"  recommendation: {item.recommendation}",
        ])
        if item.evidence:
            lines.append(f"  evidence: {'; '.join(item.evidence[:3])}")
        if item.retry_task_spec:
            lines.append("  bounded_retry_task: available with --json (not created automatically)")
        lines.append("")
    return "\n".join(lines).rstrip()
