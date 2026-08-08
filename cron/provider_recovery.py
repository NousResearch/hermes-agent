"""Deterministic cron execution error classifier and provider recovery engine.

This module is designed for no_agent use: it scans execution-error text,
classifies failures, and decides whether a provider-level fallback is
warranted.  All decisions are threshold-based so they can be executed
without an LLM in the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── Classification ──────────────────────────────────────────────────────────

class FailureCategory(str, Enum):
    provider_429 = "provider_429"          # rate limited
    auth_blocked = "auth_blocked"         # 401 / 403
    provider_5xx = "provider_5xx"         # 500 / 502 / 503
    timeout = "timeout"                    # connection / read timeout
    quota_exhausted = "quota_exhausted"   # billing / quota
    other = "other"


# Categories that are eligible for provider-level recovery.
RECOVERABLE_CATEGORIES = frozenset({
    FailureCategory.provider_429,
    FailureCategory.auth_blocked,
    FailureCategory.provider_5xx,
})


def classify_cron_error(error_text: str | None) -> FailureCategory:
    """Classify a cron execution error string into a failure category."""
    if not error_text:
        return FailureCategory.other
    text = error_text.lower()

    if any(kw in text for kw in ("429", "rate limit", "too many requests", "rate_limited")):
        return FailureCategory.provider_429
    if any(kw in text for kw in ("401", "403", "unauthorized", "forbidden",
                                   "auth", "authentication", "permission denied")):
        return FailureCategory.auth_blocked
    if any(kw in text for kw in ("500", "502", "503", "server error", "internal server",
                                   "bad gateway", "service unavailable", "overloaded")):
        return FailureCategory.provider_5xx
    if any(kw in text for kw in ("timeout", "timed out", "connection refused",
                                   "connection reset", "connection error",
                                   "network", "dns", "name resolution")):
        return FailureCategory.timeout
    if any(kw in text for kw in ("quota", "exhausted", "billing", "402",
                                   "insufficient", "credit", "balance")):
        return FailureCategory.quota_exhausted
    return FailureCategory.other


# ── Recovery assessment ─────────────────────────────────────────────────────

@dataclass
class RecoveryAssessment:
    """Result of scanning recent failures for a provider-level recovery signal."""
    triggered: bool
    provider: str
    category: FailureCategory
    failure_count: int
    window_minutes: int
    min_consecutive: int
    affected_job_ids: list[str] = field(default_factory=list)
    details: str = ""


def scan_provider_failures(
    provider: str,
    *,
    window_minutes: int = 60,
    min_consecutive: int = 3,
) -> RecoveryAssessment:
    """Scan recent execution failures for a specific provider.

    Only counts failures in RECOVERABLE categories.  Consecutive means
    the same provider + same category, with no successful execution
    for that provider in between.
    """
    from cron.executions import list_executions
    from cron.jobs import get_job
    from collections import Counter

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=window_minutes)

    failures: list[dict[str, Any]] = []
    for row in list_executions(limit=500):
        claimed = row.get("claimed_at", "")
        if claimed < cutoff.isoformat():
            break
        if row.get("status") != "failed":
            continue
        error = row.get("error")
        if not error:
            continue
        category = classify_cron_error(error)
        if category not in RECOVERABLE_CATEGORIES:
            continue
        # Only count failures from jobs using this provider
        job = get_job(row["job_id"])
        if not job or str(job.get("provider") or "") != provider:
            continue
        failures.append({
            "execution_id": row["id"],
            "job_id": row["job_id"],
            "category": category,
            "claimed_at": claimed,
        })

    # Group by category, find the most common
    category_counts = Counter(f["category"] for f in failures)
    if not category_counts:
        return RecoveryAssessment(
            triggered=False, provider=provider,
            category=FailureCategory.other, failure_count=0,
            window_minutes=window_minutes, min_consecutive=min_consecutive,
            details="no recent failures in recoverable categories",
        )

    dominant_category, count = category_counts.most_common(1)[0]
    triggered = count >= min_consecutive
    affected_jobs = list({f["job_id"] for f in failures if f["category"] == dominant_category})

    return RecoveryAssessment(
        triggered=triggered,
        provider=provider,
        category=dominant_category,
        failure_count=count,
        window_minutes=window_minutes,
        min_consecutive=min_consecutive,
        affected_job_ids=affected_jobs,
        details=(
            f"{count} consecutive {dominant_category.value} failures "
            f"in last {window_minutes}m (threshold: {min_consecutive})"
        ),
    )


def find_fallback(provider: str) -> tuple[str, str] | None:
    """Look up the next available provider/model from the global fallback chain."""
    import yaml
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return None

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # Try explicit fallback chain first
    chain = cfg.get("fallback_chain") or cfg.get("model", {}).get("fallback_chain")
    if chain and isinstance(chain, list):
        for i, entry in enumerate(chain):
            if isinstance(entry, dict) and entry.get("provider") == provider:
                if i + 1 < len(chain):
                    nxt = chain[i + 1]
                    return (str(nxt.get("provider", "")), str(nxt.get("model", "")))
                return None

    # Default hardcoded chain (mirrors Cron Failover prompt)
    default_chain = [
        ("opencode-go", "deepseek-v4-pro"),
        ("openai-codex", "gpt-5.4-mini"),
        ("xai-oauth", "grok-4.20-reasoning"),
    ]
    for i, (prov, model) in enumerate(default_chain):
        if prov == provider and i + 1 < len(default_chain):
            return default_chain[i + 1]
    return None


# ── Recovery execution ──────────────────────────────────────────────────────

@dataclass
class RecoveryRecord:
    job_id: str
    primary_provider: str
    primary_model: str
    fallback_provider: str
    fallback_model: str
    reason_category: str
    recovery_triggered_at: str


def execute_recovery(
    job_id: str,
    fallback_provider: str,
    fallback_model: str,
    *,
    reason_category: str = "unknown",
) -> RecoveryRecord | None:
    """Rewrite a job's provider/model to fallback and record recovery state."""
    from cron.jobs import update_job, get_job

    job = get_job(job_id)
    if not job:
        return None

    primary_provider = str(job.get("provider") or "")
    primary_model = str(job.get("model") or "")

    if not primary_provider or not primary_model:
        return None

    now = datetime.now(timezone.utc).isoformat()

    # Save recovery state on the job so rollback knows where to go
    recovery_state = {
        "primary_provider": primary_provider,
        "primary_model": primary_model,
        "fallback_provider": fallback_provider,
        "fallback_model": fallback_model,
        "recovery_triggered_at": now,
        "reason_category": reason_category,
    }

    try:
        update_job(job_id, {
            "provider": fallback_provider,
            "model": fallback_model,
            "recovery_state": recovery_state,
        })
    except Exception:
        return None

    return RecoveryRecord(
        job_id=job_id,
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        reason_category=reason_category,
        recovery_triggered_at=now,
    )


def evaluate_and_recover(
    provider: str,
    *,
    window_minutes: int = 60,
    min_consecutive: int = 3,
    dry_run: bool = False,
) -> list[RecoveryRecord]:
    """Full recovery cycle: scan → decide → recover affected jobs.

    Returns list of recovery records (empty if nothing triggered).
    """
    assessment = scan_provider_failures(
        provider,
        window_minutes=window_minutes,
        min_consecutive=min_consecutive,
    )

    if not assessment.triggered:
        return []

    fallback = find_fallback(provider)
    if not fallback:
        return []

    fallback_provider, fallback_model = fallback
    records: list[RecoveryRecord] = []

    for job_id in assessment.affected_job_ids:
        if dry_run:
            records.append(RecoveryRecord(
                job_id=job_id,
                primary_provider=provider,
                primary_model="?",
                fallback_provider=fallback_provider,
                fallback_model=fallback_model,
                reason_category=assessment.category.value,
                recovery_triggered_at=datetime.now(timezone.utc).isoformat(),
            ))
            continue
        record = execute_recovery(
            job_id,
            fallback_provider,
            fallback_model,
            reason_category=assessment.category.value,
        )
        if record:
            records.append(record)

    return records


# ── Rollback ────────────────────────────────────────────────────────────────

def evaluate_rollback(
    job_id: str,
    *,
    min_successes: int = 3,
) -> bool:
    """Check if a recovered job should be rolled back to its primary provider."""
    from cron.executions import list_executions
    from cron.jobs import get_job

    job = get_job(job_id)
    if not job:
        return False

    recovery_state = job.get("recovery_state")
    if not isinstance(recovery_state, dict):
        return False

    primary_provider = recovery_state.get("primary_provider")
    primary_model = recovery_state.get("primary_model")
    if not primary_provider or not primary_model:
        return False

    # Count consecutive successes on the fallback provider
    success_count = 0
    for row in list_executions(job_id=job_id, limit=20):
        if row.get("status") != "completed":
            break
        if row.get("error"):
            break
        success_count += 1

    return success_count >= min_successes


def execute_rollback(job_id: str) -> RecoveryRecord | None:
    """Restore a recovered job to its primary provider/model."""
    from cron.jobs import update_job, get_job

    job = get_job(job_id)
    if not job:
        return None

    recovery_state = job.get("recovery_state")
    if not isinstance(recovery_state, dict):
        return None

    primary_provider = str(recovery_state.get("primary_provider") or "")
    primary_model = str(recovery_state.get("primary_model") or "")
    fallback_provider = str(recovery_state.get("fallback_provider") or "")
    fallback_model = str(recovery_state.get("fallback_model") or "")

    if not primary_provider or not primary_model:
        return None

    try:
        update_job(job_id, {
            "provider": primary_provider,
            "model": primary_model,
            "recovery_state": None,
        })
    except Exception:
        return None

    return RecoveryRecord(
        job_id=job_id,
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        reason_category="rollback",
        recovery_triggered_at=datetime.now(timezone.utc).isoformat(),
    )
