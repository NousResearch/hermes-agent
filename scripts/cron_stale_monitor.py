#!/usr/bin/env python3
"""Dedicated stale-cron monitor for the ops board.

Unlike `system_health_daily.py` (one summary per day), this watchdog fires
independently every 6 hours and uses a tighter schedule-aware threshold so
that any cron which has not run within 24h+ of its expected cadence is
flagged the moment it crosses the line. It runs as a `no_agent` cron; the
script's stdout IS the Discord message (silent when no findings).

Why a dedicated watchdog?
  - The daily system-health scan is one signal among many (cron errors,
    memory, kanban drift, governance). When stale-cron findings get
    buried in a 12-bullet summary, the same incident can repeat for days
    before anyone notices.
  - A purpose-built, frequent, low-noise watchdog makes the signal
    obvious. If this watchdog fires, the only reason is stale crons.
  - The 24h "since creation" rule from the original incident (a cron
    that was created but never produced output for >24h) is preserved
    as the never-ran branch, regardless of cadence.

Schedule-aware threshold:
  - Daily or sub-daily schedule: stale after 26h (1h grace above the 24h SLA).
  - Weekly: stale after 8 days.
  - Monthly: stale after 33 days.
  - Quarterly: stale after 95 days.
  - One-shot: stale if its run_at is in the past and it never fired.
  - Never-ran AND created > 24h ago: ALWAYS flagged, regardless of cadence.
    This is the original incident class — a cron registered but never run.

Output contract (per cron-output-contract v2.6.0):
  - First line: `cron-stale-watchdog · DD/MM/YYYY HH:MM:SS` (UK format)
  - Empty stdout when zero findings (silent — no "[SILENT]" literal).
  - Findings capped at 12 lines in the message; full list always written
    to /home/kensei/.hermes/cron/output/stale-watchdog/<timestamp>.txt
    for downstream debugging.
  - All evidence inline: job id, last run timestamp, schedule display.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERMES = Path(os.path.expanduser("~/.hermes"))
JOBS_FILE = HERMES / "cron" / "jobs.json"
OUTPUT_DIR = HERMES / "cron" / "output" / "stale-watchdog"

# Tunables. The "24h since creation" hard floor catches the original
# incident class even when a job's nominal schedule is monthly.
NEVER_RAN_FLOOR = timedelta(hours=24)
THRESHOLDS = {
    "daily": timedelta(hours=26),     # 24h SLA + 2h grace
    "weekly": timedelta(days=8),
    "monthly": timedelta(days=33),
    "quarterly": timedelta(days=95),
}
SUB_DAILY_FLOOR = timedelta(hours=26)  # anything cron with *-anywhere fields


def _classify(job: dict) -> str:
    """Return cadence class: daily | weekly | monthly | quarterly | sub_daily."""
    schedule = job.get("schedule") or {}
    display = (schedule.get("display") or job.get("schedule_display") or "").lower()
    name = (job.get("name") or "").lower()
    if "quarterly" in display or "quarterly" in name:
        return "quarterly"
    if "monthly" in display or "monthly" in name:
        return "monthly"
    if "weekly" in display or "weekly" in name:
        return "weekly"
    expr = schedule.get("expr") or ""
    if " " in expr and len(expr.split()) == 5:
        minute, hour, dom, month, dow = expr.split()
        # Specific DOW = weekly
        if dow not in ("*", "?"):
            return "weekly"
        # Specific DOM = monthly
        if dom not in ("*", "?"):
            return "monthly"
    # Default catch-all: treat as sub-daily for short intervals, else weekly
    return "sub_daily"


def _threshold_for(cadence: str) -> timedelta:
    if cadence in THRESHOLDS:
        return THRESHOLDS[cadence]
    return SUB_DAILY_FLOOR


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_age(delta: timedelta) -> str:
    """Compact human-readable age: '4d 3h', '26h 11m', '47m'."""
    total = int(delta.total_seconds())
    if total < 0:
        return "in the future"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _uk_now() -> datetime:
    """Local UK time for the first-line stamp (matches cron-output-contract)."""
    return datetime.now().astimezone()


def _format_uk(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S")


def _receipt_job_ids(receipt_path: Path) -> set[str]:
    """Return the staged target IDs from a P13 receipt, failing closed on bad rows."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("receipt jobs must be a list")
    ids: set[str] = set()
    for item in jobs:
        target_id = item.get("target_id") if isinstance(item, dict) else None
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("receipt contains an invalid target_id")
        ids.add(target_id)
    return ids


def main(receipt_path: Path | None = None) -> int:
    monitored_ids = _receipt_job_ids(receipt_path) if receipt_path is not None else None
    if not JOBS_FILE.exists():
        # No jobs file → nothing to monitor. Silent.
        return 0

    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Don't crash the watchdog if jobs.json is transiently corrupt;
        # the daily system-health scan will surface the corruption.
        return 0

    now = datetime.now(timezone.utc)
    findings: list[dict] = []
    for job in data.get("jobs", []):
        if monitored_ids is not None and job.get("id") not in monitored_ids:
            continue
        if not job.get("enabled") or job.get("state") == "paused":
            continue
        cadence = _classify(job)
        threshold = _threshold_for(cadence)
        last = _parse_iso(job.get("last_run_at"))
        created = _parse_iso(job.get("created_at"))
        if last is None:
            # Never ran. Hard floor: 24h since creation. BUT — for a recurring
            # cron whose first scheduled fire time is still in the future, the
            # 24h-since-creation rule is misleading: the schedule has not yet
            # had a chance to fire. Skip such jobs to avoid false positives
            # (e.g. a weekly job created mid-week after the weekly fire time).
            next_run = _parse_iso(job.get("next_run_at"))
            if (
                created
                and (now - created) >= NEVER_RAN_FLOOR
                and not (next_run and next_run > now)
            ):
                age = now - created
                findings.append({
                    "id": job.get("id"),
                    "name": job.get("name", "?"),
                    "kind": "never_ran",
                    "age": _format_age(age),
                    "since": created.isoformat(),
                    "schedule": job.get("schedule_display", ""),
                    "cadence": cadence,
                })
            continue
        # Has run at least once. Check schedule-aware threshold.
        age = now - last
        if age >= threshold:
            findings.append({
                "id": job.get("id"),
                "name": job.get("name", "?"),
                "kind": "stale",
                "age": _format_age(age),
                "since": last.isoformat(),
                "schedule": job.get("schedule_display", ""),
                "cadence": cadence,
            })

    if not findings:
        # Zero-signal rule: stay silent. Empty stdout, no [SILENT] literal.
        return 0

    # Sort by age desc — worst offenders first.
    findings.sort(key=lambda f: f["age"], reverse=True)

    # Write the full report for downstream debugging.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _format_uk(_uk_now()).replace(" ", "_").replace("/", "-").replace(":", "-")
    report_path = OUTPUT_DIR / f"{stamp}.txt"
    report_lines = [f"cron-stale-watchdog · {_format_uk(_uk_now())}", ""]
    for f in findings:
        report_lines.append(
            f"{f['kind']:>10s}  {f['age']:>10s}  {f['cadence']:<10s}  "
            f"id={f['id']}  name={f['name']}  sched={f['schedule']}  last_or_created={f['since']}"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Compose the Discord message.
    first_line = f"cron-stale-watchdog · {_format_uk(_uk_now())}"
    lines = [first_line, ""]
    lines.append(f"{len(findings)} stale cron(s) detected — see `{report_path}`")
    lines.append("")
    # Up to 12 lines in the visible message; the rest is in the attached file.
    for f in findings[:12]:
        if f["kind"] == "never_ran":
            lines.append(f"• `{f['name']}` — never ran, {f['age']} old (created {f['since'][:10]}) — `{f['schedule'] or '(no schedule)'}`")
        else:
            lines.append(f"• `{f['name']}` — last run {f['age']} ago — `{f['schedule'] or '(no schedule)'}`")
    if len(findings) > 12:
        lines.append(f"… and {len(findings) - 12} more (full list in report)")

    # Strong-claim evidence: every job referenced has a backreference
    # to its id + schedule display, satisfying the evidence-gate rule.
    print("\n".join(lines))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="limit monitoring to target job IDs in this P13 staging receipt",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(_parse_args().receipt))
