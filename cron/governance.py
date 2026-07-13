"""Deterministic pre-execution policy for Hermes recurring jobs."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from croniter import croniter


@dataclass(frozen=True)
class CronGovernanceDecision:
    allow: bool
    code: str
    reason: str


def _blocked(code: str, reason: str) -> CronGovernanceDecision:
    return CronGovernanceDecision(False, code, reason)


def _schedule_period_minutes(schedule: dict[str, Any]) -> int | None:
    """Return the recurring schedule period in whole minutes when determinable."""
    kind = schedule.get("kind")
    if kind == "interval":
        minutes = schedule.get("minutes")
        return minutes if isinstance(minutes, int) else None
    if kind == "cron":
        expr = schedule.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            return None
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        iterator = croniter(expr, base)
        first = iterator.get_next(datetime)
        second = iterator.get_next(datetime)
        return max(1, int((second - first).total_seconds() // 60))
    return None


def evaluate_cron_governance_policy(job: dict[str, Any]) -> CronGovernanceDecision:
    """Return an allow/block decision without running, advancing, or delivering a job."""
    if job.get("enabled") is not True:
        return _blocked("disabled", "job is disabled")
    if job.get("state") != "scheduled":
        return _blocked("paused_state", "job state must be scheduled")

    governance = job.get("governance")
    if not isinstance(governance, dict) or not governance.get("owner_issue"):
        return _blocked("missing_owner_issue", "job requires an accountable owner issue")

    cadence = governance.get("cadence")
    if not isinstance(cadence, dict):
        return _blocked("missing_cadence_policy", "job requires a cadence policy")
    minimum = cadence.get("min_minutes")
    schedule = job.get("schedule")
    if not isinstance(minimum, int) or minimum < 1:
        return _blocked("missing_cadence_policy", "cadence minimum must be a positive integer")
    if not isinstance(schedule, dict):
        return _blocked("unknown_cadence", "job schedule is not governable")
    minutes = _schedule_period_minutes(schedule)
    if minutes is None:
        return _blocked("unknown_cadence", "schedule period could not be determined")
    if minutes < minimum:
        return _blocked("over_cadence", "schedule is more frequent than its policy permits")

    if not job.get("no_agent", False):
        model_policy = governance.get("model_policy")
        budget_policy = governance.get("budget_policy")
        if not isinstance(model_policy, dict) or not isinstance(budget_policy, dict):
            return _blocked("missing_model_budget_policy", "agent jobs require model and budget policies")
        if not model_policy.get("provider") or not model_policy.get("model"):
            return _blocked("missing_model_budget_policy", "agent model policy requires provider and model")
        if "max_usd_per_run" not in budget_policy:
            return _blocked("missing_model_budget_policy", "agent budget policy requires a per-run limit")

    if job.get("deliver", "local") != "local" and governance.get("allow_nonlocal_delivery") is not True:
        return _blocked("nonlocal_delivery_not_authorized", "nonlocal delivery requires explicit authorization")

    return CronGovernanceDecision(True, "allowed", "job passed governance policy")


def append_governance_audit(
    audit_path: Path, job: dict[str, Any], decision: CronGovernanceDecision
) -> None:
    """Append a bounded, secret-free decision record for operational visibility."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists() and audit_path.stat().st_size >= 5 * 1024 * 1024:
        audit_path.replace(audit_path.with_suffix(audit_path.suffix + ".1"))
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job_id": job.get("id"),
        "name": job.get("name"),
        "decision": "allow" if decision.allow else "block",
        "code": decision.code,
        "enabled": job.get("enabled") is True,
        "state": job.get("state"),
        "schedule": job.get("schedule"),
        "deliver": job.get("deliver", "local"),
        "owner_issue_present": bool((job.get("governance") or {}).get("owner_issue")),
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
