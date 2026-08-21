#!/usr/bin/env python3
"""Audita jobs cron Hermes: paused, overdue, ticker — sem expor prompts.

Somente leitura. Saída JSON no stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _default_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / "AppData" / "Local" / "hermes"


def _profiles_root() -> Path:
    return Path.home() / "AppData" / "Local" / "hermes" / "profiles"


def _emit_json(payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    except (AttributeError, OSError):
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _now_aware() -> datetime:
    return datetime.now().astimezone()


def _read_config_version(home: Path) -> int | None:
    path = home / "config.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            version = raw.get("_config_version")
            return int(version) if version is not None else None
    except Exception:
        return None
    return None


def _discover_scopes(all_profiles: bool, hermes_home: Path | None) -> list[tuple[str, Path]]:
    if hermes_home is not None:
        return [("custom", hermes_home)]
    scopes: list[tuple[str, Path]] = [("default", _default_hermes_home())]
    if not all_profiles:
        return scopes
    root = _profiles_root()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "cron" / "jobs.json").is_file():
                scopes.append((child.name, child))
    return scopes


def _schedule_display(job: dict[str, Any]) -> str:
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        display = schedule.get("display") or schedule.get("expr") or schedule.get("value")
        if display:
            return str(display)
    fallback = job.get("schedule_display")
    return str(fallback) if fallback else "?"


def _audit_job(job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    from cron.jobs import effective_job_state

    state = effective_job_state(job)
    enabled = bool(job.get("enabled", True))
    paused = state == "paused" or (not enabled and state not in {"completed", "error"})
    next_dt = _parse_iso(job.get("next_run_at"))
    overdue = False
    overdue_seconds: float | None = None
    if enabled and state in {"scheduled", "active"} and next_dt is not None:
        if next_dt <= now:
            overdue = True
            overdue_seconds = (now - next_dt).total_seconds()

    deliver = job.get("deliver") or ["local"]
    if isinstance(deliver, str):
        deliver = [deliver]

    latest = job.get("latest_execution")
    latest_exec: dict[str, Any] | None = None
    if isinstance(latest, dict):
        latest_exec = {
            "id": latest.get("id"),
            "status": latest.get("status"),
            "claimed_at": latest.get("claimed_at"),
        }

    return {
        "id": job.get("id"),
        "name": job.get("name"),
        "schedule": _schedule_display(job),
        "state": state,
        "paused": paused,
        "enabled": enabled,
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_error": job.get("last_error"),
        "last_delivery_error": job.get("last_delivery_error"),
        "overdue": overdue,
        "overdue_seconds": round(overdue_seconds, 1) if overdue_seconds is not None else None,
        "never_run": enabled and job.get("last_run_at") is None and state == "scheduled",
        "provider": job.get("provider"),
        "model": job.get("model"),
        "deliver": deliver,
        "skills": job.get("skills") or ([job["skill"]] if job.get("skill") else []),
        "no_agent": bool(job.get("no_agent")),
        "latest_execution": latest_exec,
    }


def _ticker_health(home: Path) -> dict[str, Any]:
    from cron.jobs import (
        TICKER_INTERVAL_SECONDS,
        get_ticker_heartbeat_age,
        get_ticker_last_error,
        get_ticker_success_age,
        use_cron_store,
    )

    stale_after = TICKER_INTERVAL_SECONDS * 3 + 20
    with use_cron_store(home):
        hb_age = get_ticker_heartbeat_age()
        ok_age = get_ticker_success_age()
        last_error = get_ticker_last_error()

    stalled = hb_age is not None and hb_age > stale_after
    ticks_failing = (
        hb_age is not None
        and ok_age is not None
        and ok_age > stale_after
        and not stalled
    )

    return {
        "stale_after_seconds": stale_after,
        "heartbeat_age_seconds": round(hb_age, 1) if hb_age is not None else None,
        "success_age_seconds": round(ok_age, 1) if ok_age is not None else None,
        "stalled": stalled,
        "ticks_failing": ticks_failing,
        "last_tick_error": last_error,
    }


def _scheduler_provider() -> str:
    try:
        from cron.scheduler_provider import resolve_cron_scheduler

        return resolve_cron_scheduler().name or "builtin"
    except Exception:
        return "builtin"


def _gateway_running() -> bool:
    try:
        from hermes_cli.gateway import find_gateway_pids

        return bool(find_gateway_pids())
    except Exception:
        return False


def _audit_scope(scope_name: str, home: Path, *, include_disabled: bool, now: datetime) -> dict[str, Any]:
    from cron.jobs import list_jobs, use_cron_store

    jobs_path = home / "cron" / "jobs.json"
    with use_cron_store(home):
        raw_jobs = list_jobs(include_disabled=include_disabled)

    audited = [_audit_job(job, now=now) for job in raw_jobs]
    overdue_jobs = [j["id"] for j in audited if j.get("overdue")]
    paused_jobs = [j["id"] for j in audited if j.get("paused")]
    failed_jobs = [j["id"] for j in audited if j.get("last_status") not in {None, "ok"}]

    provider = _scheduler_provider()
    ticker = _ticker_health(home) if provider == "builtin" else None
    gateway = _gateway_running()

    warnings: list[str] = []
    if provider == "builtin" and not gateway:
        warnings.append("gateway_not_running")
    if ticker and ticker.get("stalled"):
        warnings.append("ticker_stalled")
    if ticker and ticker.get("ticks_failing"):
        warnings.append("ticker_ticks_failing")
    if overdue_jobs:
        warnings.append("jobs_overdue")

    return {
        "scope": scope_name,
        "hermes_home": str(home),
        "jobs_file": str(jobs_path),
        "jobs_file_exists": jobs_path.is_file(),
        "config_version": _read_config_version(home),
        "scheduler_provider": provider,
        "gateway_running": gateway,
        "ticker": ticker,
        "warnings": warnings,
        "jobs": audited,
        "summary": {
            "total": len(audited),
            "active": sum(1 for j in audited if j.get("enabled") and not j.get("paused")),
            "paused": len(paused_jobs),
            "overdue": len(overdue_jobs),
            "never_run": sum(1 for j in audited if j.get("never_run")),
            "last_failed": len(failed_jobs),
            "overdue_ids": overdue_jobs,
            "paused_ids": paused_jobs,
            "failed_ids": failed_jobs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes cron audit (read-only JSON)")
    parser.add_argument(
        "--hermes-home",
        default="",
        help="HERMES_HOME override (single scope)",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Audit default plus every profile with cron/jobs.json",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include paused/disabled jobs (default: active only)",
    )
    args = parser.parse_args()

    home_override = Path(args.hermes_home).expanduser() if args.hermes_home else None
    scopes = _discover_scopes(args.all_profiles, home_override)
    now = _now_aware()

    results = [
        _audit_scope(name, home, include_disabled=args.include_disabled, now=now)
        for name, home in scopes
    ]

    total_jobs = sum(r["summary"]["total"] for r in results)
    total_overdue = sum(r["summary"]["overdue"] for r in results)
    all_warnings = sorted({w for r in results for w in r.get("warnings", [])})

    ok = any(r.get("jobs_file_exists") for r in results) or total_jobs == 0
    out = {
        "ok": ok,
        "audited_at": now.isoformat(),
        "include_disabled": args.include_disabled,
        "scopes": results,
        "summary": {
            "scope_count": len(results),
            "job_count": total_jobs,
            "overdue_count": total_overdue,
            "warnings": all_warnings,
        },
    }
    _emit_json(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
