"""Metadata-only autonomous-loop inventory for Hermes.

Tier 6 is read-only: it inventories standing automation surfaces (cron jobs and
persisted /goal rows) and reports loop-safety issue codes without creating,
updating, pausing, resuming, or deleting jobs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
CONTENT_POLICY = "metadata_only"
MODE = "audit_only_no_create"
SIDE_EFFECT_TOOLSETS = {
    "terminal",
    "file",
    "browser",
    "computer_use",
    "discord",
    "discord_admin",
    "homeassistant",
    "kanban",
    "spotify",
    "x_search",
    "yuanbao",
}
SILENCE_MARKERS = (
    "[silent]",
    "silent when",
    "stay silent",
    "nothing to report",
    "no new",
    "only report",
    "empty stdout",
)
APPROVAL_MARKERS = (
    "approval",
    "ask before",
    "confirm before",
    "read-only",
    "read only",
    "dry-run",
    "dry run",
    "no external side effects",
    "do not change",
)
TERMINAL_GOAL_STATUSES = {"done", "cleared"}
PAUSED_GOAL_STATUSES = {"paused"}
ACTIVE_GOAL_STATUSES = {"active"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _issue(code: str, *, severity: str = "warning", count: int = 1, job_id: Optional[str] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {"code": code, "severity": severity, "count": count}
    if job_id:
        item["job_id_sha256"] = _sha256_text(job_id)
    return item


def _read_jobs(hermes_home: Path) -> List[Dict[str, Any]]:
    path = hermes_home / "cron" / "jobs.json"
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def _schedule_kind(job: Dict[str, Any]) -> str:
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        return str(schedule.get("kind") or "unknown")
    if schedule is None:
        return "unknown"
    return "legacy"


def _is_enabled(job: Dict[str, Any]) -> bool:
    if job.get("enabled") is False:
        return False
    state = str(job.get("state") or "").strip().lower()
    return state not in {"paused", "cancelled", "canceled", "disabled"}


def _is_recurring(job: Dict[str, Any]) -> bool:
    return _schedule_kind(job) in {"interval", "cron"}


def _toolsets(job: Dict[str, Any]) -> List[str]:
    raw = job.get("enabled_toolsets")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _has_silence_condition(job: Dict[str, Any]) -> bool:
    if bool(job.get("no_agent")):
        return bool(job.get("script"))
    prompt = str(job.get("prompt") or "").lower()
    return any(marker in prompt for marker in SILENCE_MARKERS)


def _has_side_effect_policy(job: Dict[str, Any]) -> bool:
    prompt = str(job.get("prompt") or "").lower()
    return any(marker in prompt for marker in APPROVAL_MARKERS)


def _normalize_deliver_value(deliver: Any) -> str:
    if deliver is None or deliver == "":
        return "local"
    if isinstance(deliver, (list, tuple)):
        parts = [str(part).strip() for part in deliver if str(part).strip()]
        return ",".join(parts) if parts else "local"
    return str(deliver)


def _delivery_classes(job: Dict[str, Any]) -> List[str]:
    value = _normalize_deliver_value(job.get("deliver"))
    classes: List[str] = []
    for part in value.split(","):
        target = part.strip()
        if not target:
            continue
        lowered = target.lower()
        if lowered in {"origin", "local", "all"}:
            classes.append(lowered)
        elif ":" in lowered:
            classes.append("explicit")
        else:
            classes.append("home")
    return classes or ["origin"]


def _count_by(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _read_goal_rows(hermes_home: Path) -> List[Dict[str, Any]]:
    db_path = hermes_home / "state.db"
    if not db_path.exists() or not db_path.is_file():
        return []
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'goal:%'"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    parsed: List[Dict[str, Any]] = []
    for key, value in rows:
        try:
            payload = json.loads(value or "{}")
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        parsed.append({
            "key_hash": _sha256_text(str(key)),
            "status": str(payload.get("status") or "unknown"),
            "turns_used": int(payload.get("turns_used") or 0),
            "max_turns": int(payload.get("max_turns") or 0),
            "subgoal_count": len(payload.get("subgoals") or []),
        })
    return parsed


def _guidance() -> List[Dict[str, Any]]:
    return [
        {
            "id": "cron_agent_prompt",
            "requires": [
                "self-contained prompt",
                "explicit silence condition such as [SILENT] when nothing changed",
                "toolset scope narrower than default when possible",
            ],
        },
        {
            "id": "script_only_watchdog",
            "requires": [
                "no_agent=True",
                "script path present",
                "empty stdout means silent",
                "non-zero exit alerts the user",
            ],
        },
        {
            "id": "side_effect_approval",
            "requires": [
                "external or irreversible side effects require explicit approval",
                "recurring jobs should prefer read-only/dry-run checks",
                "delivery fanout should be intentional and narrow",
            ],
        },
        {
            "id": "loop_budget",
            "requires": [
                "repeat limit or clear recurring intent",
                "quiet successful no-op behavior",
                "route proof should remain content-safe",
            ],
        },
    ]


def audit_autonomous_loops(*, hermes_home: Optional[Path] = None) -> Dict[str, Any]:
    """Return a content-safe inventory of autonomous loops.

    Reads cron job storage and goal state directly. It never creates, updates,
    pauses, resumes, runs, or deletes jobs/goals.
    """
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    jobs = _read_jobs(home)
    issues: List[Dict[str, Any]] = []

    active_jobs = [job for job in jobs if _is_enabled(job)]
    recurring_jobs = [job for job in jobs if _is_recurring(job)]
    active_recurring = [job for job in active_jobs if _is_recurring(job)]
    no_agent_jobs = [job for job in jobs if bool(job.get("no_agent"))]
    agent_jobs = [job for job in jobs if not bool(job.get("no_agent"))]

    schedule_counts = _count_by(_schedule_kind(job) for job in jobs)
    delivery_counts: Dict[str, int] = {}
    for job in jobs:
        for cls in _delivery_classes(job):
            delivery_counts[cls] = delivery_counts.get(cls, 0) + 1

    missing_silence = 0
    broad_delivery = 0
    side_effect_missing = 0
    unbounded_tool_scope = 0
    no_agent_missing_script = 0
    chained_jobs = 0
    profile_jobs = 0
    workdir_jobs = 0

    for job in active_recurring:
        job_id = str(job.get("id") or "")
        if not _has_silence_condition(job):
            missing_silence += 1
            issues.append(_issue("loop_missing_silence_condition", job_id=job_id))
        if "all" in _delivery_classes(job):
            broad_delivery += 1
            issues.append(_issue("loop_broad_delivery", severity="error", job_id=job_id))
        toolsets = _toolsets(job)
        if not bool(job.get("no_agent")) and not toolsets:
            unbounded_tool_scope += 1
            issues.append(_issue("loop_tool_scope_unbounded", job_id=job_id))
        if SIDE_EFFECT_TOOLSETS.intersection(set(toolsets)) and not _has_side_effect_policy(job):
            side_effect_missing += 1
            issues.append(_issue("loop_side_effect_policy_missing", severity="error", job_id=job_id))

    for job in no_agent_jobs:
        job_id = str(job.get("id") or "")
        if not str(job.get("script") or "").strip():
            no_agent_missing_script += 1
            issues.append(_issue("loop_no_agent_missing_script", severity="error", job_id=job_id))

    for job in jobs:
        if job.get("context_from"):
            chained_jobs += 1
        if job.get("profile"):
            profile_jobs += 1
        if job.get("workdir"):
            workdir_jobs += 1

    goal_rows = _read_goal_rows(home)
    goal_status_counts = _count_by(row["status"] for row in goal_rows)
    active_goal_count = sum(1 for row in goal_rows if row["status"] in ACTIVE_GOAL_STATUSES)
    paused_goal_count = sum(1 for row in goal_rows if row["status"] in PAUSED_GOAL_STATUSES)
    terminal_goal_count = sum(1 for row in goal_rows if row["status"] in TERMINAL_GOAL_STATUSES)

    return {
        "schema_version": SCHEMA_VERSION,
        "content_policy": CONTENT_POLICY,
        "mode": MODE,
        "cron": {
            "job_count": len(jobs),
            "active_count": len(active_jobs),
            "recurring_count": len(recurring_jobs),
            "active_recurring_count": len(active_recurring),
            "agent_job_count": len(agent_jobs),
            "script_only_watchdog_count": len(no_agent_jobs),
            "schedule_counts": schedule_counts,
            "delivery_counts": delivery_counts,
            "chained_job_count": chained_jobs,
            "profile_job_count": profile_jobs,
            "workdir_job_count": workdir_jobs,
            "missing_silence_condition_count": missing_silence,
            "broad_delivery_count": broad_delivery,
            "side_effect_policy_missing_count": side_effect_missing,
            "tool_scope_unbounded_count": unbounded_tool_scope,
            "no_agent_missing_script_count": no_agent_missing_script,
        },
        "goals": {
            "total_goal_rows": len(goal_rows),
            "active_goal_count": active_goal_count,
            "paused_goal_count": paused_goal_count,
            "terminal_goal_count": terminal_goal_count,
            "status_counts": goal_status_counts,
            "turns_used_total": sum(row["turns_used"] for row in goal_rows),
            "subgoal_count_total": sum(row["subgoal_count"] for row in goal_rows),
        },
        "guidance": _guidance(),
        "issues": issues,
        "issue_count": len(issues),
    }
