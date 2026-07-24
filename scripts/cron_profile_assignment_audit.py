#!/usr/bin/env python3
"""Audit Hermes cron jobs and recommend profile ownership.

This script is intentionally standalone. It reads cron JSON stores directly so
operators can inspect a profile's jobs without importing ``cron.jobs`` into a
long-running process whose module globals may already point at another profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TARGET_PROFILES = (
    "default",
    "cpa-tax-researcher",
    "personal",
    "rva-dev",
    "rva-firm-ops",
    "rva-leads",
    "rva-profit-pulse",
)

EXACT_JOB_TARGETS = {
    "backup-daily": "default",
    "crm-daily-scan": "rva-leads",
    "crm-lead-check": "rva-leads",
    "crm-weekly-scores": "rva-leads",
    "taxdome-client-alert": "rva-firm-ops",
    "cron-integrity-selfcheck": "default",
    "calendar-aware-nudges": "rva-firm-ops",
    "client-email-capture": "rva-firm-ops",
    "crm-lead-reconcile": "rva-leads",
    "gbrain-live-sync": "default",
    "gbrain-daily-check": "default",
    "gbrain-weekly-health": "default",
    "tax-research-archive-lint": "cpa-tax-researcher",
    "critical-repo-git-sync": "default",
    "crm-gbrain-client-export": "rva-leads",
    "content-squarespace-sync": "rva-profit-pulse",
    "content-topic-miner": "rva-profit-pulse",
    "content-distribution-audit": "rva-profit-pulse",
    "content-weekly-strategy-report": "rva-profit-pulse",
    "gmail-watch-refresh": "rva-firm-ops",
    "crm-meeting-notes-sync": "rva-leads",
    "content-idea-proposer": "rva-profit-pulse",
    "process-meeting-notes": "rva-firm-ops",
    "squarespace-hourly-intake": "rva-leads",
    "lead-no-response-watchdog": "rva-leads",
    "crm-funnel-digest": "rva-leads",
    "stall-followup-draft": "rva-leads",
    "crm-quote-suggestions": "rva-leads",
    "square-daily-sales-capture": "rva-profit-pulse",
    "infra-self-heal-loop": "default",
    "hermes-backend-update-canary": "default",
    "hermes-loop:contract-registry": "default",
    "hermes-loop:semantic-audit": "default",
    "hermes-loop:loop-truth-maintenance": "default",
    "hermes-loop:delivery-integrity": "default",
    "hermes-loop:cron-wrapper-truth": "default",
    "hermes-loop:high-risk-gates": "default",
    "hermes-loop:observability-reconciliation": "default",
    "hermes-loop:dashboard-loop-health": "default",
    "hermes-loop:gateway-plugin-readiness": "default",
    "hermes-loop:model-provider-readiness": "default",
    "hermes-loop:cost-no-progress-guard": "default",
    "hermes-loop:crm-lead-lifecycle": "rva-leads",
    "hermes-loop:meeting-to-action": "default",
    "hermes-loop:tax-research-integrity": "cpa-tax-researcher",
    "hermes-loop:content-production-quality": "rva-profit-pulse",
    "hermes-loop:square-sales-pipeline-verification": "rva-profit-pulse",
    "hermes-loop:gbrain-bridge-health": "default",
    "hermes-loop:credential-security-drift": "default",
    "hermes-loop:backup-restore-verification": "default",
    "hermes-loop:skill-promotion-upgrade": "default",
    "hermes-loop:test-suite-truth-targeted": "default",
    "hermes-loop:skill-promotions": "default",
    "hermes-loop:ingest-observability": "default",
    "hermes-loop:orphan-metric-cleanup": "default",
    "rva-profit-pulse-topic-discovery-weekly": "rva-profit-pulse",
    "square-weekly-digest": "rva-profit-pulse",
    "ringcentral-sms-opt-out-watchdog": "rva-leads",
    "notcrawl-nightly-sync": "default",
    "stale-draft-alert": "rva-firm-ops",
}


@dataclass(frozen=True)
class Assignment:
    job_id: str
    name: str
    target_profile: str
    execution_type: str
    reason: str
    review_needed: bool = False
    script: str | None = None
    workdir: str | None = None
    enabled: bool = True


def default_jobs_file() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "cron" / "jobs.json"
    except Exception:
        return Path.home() / ".hermes" / "cron" / "jobs.json"


def load_job_store(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load cron jobs from wrapped or legacy list-form JSON stores."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(job) for job in raw], {"shape": "list"}
    if isinstance(raw, dict) and isinstance(raw.get("jobs"), list):
        metadata = {k: v for k, v in raw.items() if k != "jobs"}
        metadata["shape"] = "wrapped"
        return [dict(job) for job in raw["jobs"]], metadata
    raise ValueError(f"Unsupported cron jobs store shape in {path}")


def job_enabled(job: dict[str, Any]) -> bool:
    return bool(job.get("enabled", True))


def execution_type(job: dict[str, Any]) -> str:
    if job.get("no_agent"):
        return "script-only"
    if job.get("script"):
        return "agent-with-script"
    return "agent"


def _job_text(job: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("name", "script", "workdir", "prompt"):
        value = job.get(key)
        if value:
            parts.append(str(value).lower())
    return " ".join(parts)


def classify_job(job: dict[str, Any]) -> tuple[str, str, bool]:
    """Return ``(target_profile, reason, review_needed)`` for a cron job."""
    name = str(job.get("name") or job.get("id") or "").strip()
    if name in EXACT_JOB_TARGETS:
        target = EXACT_JOB_TARGETS[name]
        return target, f"exact match: {name} -> {target}", False

    text = _job_text(job)
    if name.startswith("hermes-loop:"):
        if "crm" in text or "lead" in text:
            return "rva-leads", "Hermes loop monitors CRM/lead lifecycle", False
        if "tax-research" in text or "tax_research" in text:
            return "cpa-tax-researcher", "Hermes loop monitors tax research integrity", False
        if "content" in text or "square" in text or "sales" in text:
            return "rva-profit-pulse", "Hermes loop monitors content/profit workflows", False
        return "default", "Hermes runtime diagnostic loop stays self-referential", False

    if name.startswith("crm-") or any(
        token in text
        for token in (
            "crm",
            "lead",
            "squarespace",
            "quote",
            "ringcentral",
            "sms opt",
        )
    ):
        return "rva-leads", "CRM/lead workflow", False
    if any(token in text for token in ("taxdome", "gmail", "calendar", "meeting-notes", "client_email")):
        return "rva-firm-ops", "Firm operations/client communications workflow", False
    if any(token in text for token in ("content", "square", "profit-pulse", "sales")):
        return "rva-profit-pulse", "Content/profit workflow", False
    if "tax-research" in text or "tax_research" in text:
        return "cpa-tax-researcher", "Tax research workflow", False
    if "gbrain" in text or "backup" in text or "infra" in text or "repo" in text:
        return "default", "Shared runtime infrastructure", False
    return "default", "unknown job shape; default is safest pending review", True


def assignment_for_job(job: dict[str, Any]) -> Assignment:
    target, reason, review_needed = classify_job(job)
    return Assignment(
        job_id=str(job.get("id") or ""),
        name=str(job.get("name") or job.get("id") or ""),
        target_profile=target,
        execution_type=execution_type(job),
        reason=reason,
        review_needed=review_needed,
        script=str(job["script"]) if job.get("script") else None,
        workdir=str(job["workdir"]) if job.get("workdir") else None,
        enabled=job_enabled(job),
    )


def build_report(jobs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    paused: list[dict[str, Any]] = []
    target_counts = {profile: 0 for profile in TARGET_PROFILES}

    for job in jobs:
        assignment = assignment_for_job(job)
        row = asdict(assignment)
        if assignment.enabled:
            active.append(row)
            target_counts[assignment.target_profile] += 1
        else:
            paused.append(row)

    target_counts = {k: v for k, v in target_counts.items() if v or k in {"personal", "rva-dev"}}
    return {
        "summary": {
            "active_total": len(active),
            "paused_total": len(paused),
            "target_counts": target_counts,
            "review_needed": sum(1 for row in active if row["review_needed"]),
        },
        "active": active,
        "paused": paused,
    }


def report_as_markdown(report: dict[str, Any]) -> str:
    lines = ["# Cron Profile Assignment Audit", ""]
    summary = report["summary"]
    lines.append(f"Active jobs: {summary['active_total']}")
    lines.append(f"Paused jobs: {summary['paused_total']}")
    lines.append("")
    lines.append("## Target Counts")
    lines.append("")
    for profile, count in summary["target_counts"].items():
        lines.append(f"- `{profile}`: {count}")
    lines.append("")
    lines.append("## Active Jobs")
    lines.append("")
    lines.append("| Job | Target | Execution | Review | Reason |")
    lines.append("|---|---|---|---|---|")
    for row in report["active"]:
        review = "yes" if row["review_needed"] else "no"
        lines.append(
            f"| `{row['name']}` | `{row['target_profile']}` | {row['execution_type']} | {review} | {row['reason']} |"
        )
    if report["paused"]:
        lines.append("")
        lines.append("## Paused Follow-up")
        lines.append("")
        lines.append("| Job | Candidate Target | Reason |")
        lines.append("|---|---|---|")
        for row in report["paused"]:
            lines.append(f"| `{row['name']}` | `{row['target_profile']}` | {row['reason']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, default=default_jobs_file())
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args(argv)

    jobs, _metadata = load_job_store(args.jobs_file)
    report = build_report(jobs)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report_as_markdown(report), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
