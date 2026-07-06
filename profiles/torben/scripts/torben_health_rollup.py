from __future__ import annotations

import json
import sys
from pathlib import Path

from torben_job_contract import health_path, load_json, torben_home


def _load_jobs(profile_home: Path) -> list[dict]:
    jobs_path = profile_home / "cron" / "jobs.json"
    payload = load_json(jobs_path, {})
    jobs = payload.get("jobs") if isinstance(payload, dict) else []
    return [job for job in jobs if isinstance(job, dict)]


def _job_name(job: dict) -> str:
    return str(job.get("name") or "").strip()


def main() -> int:
    profile_home = torben_home()
    jobs = [
        job
        for job in _load_jobs(profile_home)
        if _job_name(job) and not _job_name(job).startswith("torben-desk-v2")
    ]
    print("job\tenabled\tstatus\tconsecutive_failures\tlast_run_at\tlast_error")
    worst = 0
    for job in sorted(jobs, key=_job_name):
        name = _job_name(job)
        health = load_json(health_path(name, profile_home), {})
        status = str(health.get("status") or "missing") if isinstance(health, dict) else "missing"
        if not job.get("enabled") and status == "missing":
            status = "disabled"
        failures = health.get("consecutive_failures", "") if isinstance(health, dict) else ""
        last_run_at = health.get("last_run_at", "") if isinstance(health, dict) else ""
        last_error = ""
        if isinstance(health, dict) and isinstance(health.get("last_error"), dict):
            last_error = str(health["last_error"].get("message") or health["last_error"].get("type") or "")
        print(f"{name}\t{job.get('enabled')}\t{status}\t{failures}\t{last_run_at}\t{last_error}")
        if job.get("enabled") and status == "failed":
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
