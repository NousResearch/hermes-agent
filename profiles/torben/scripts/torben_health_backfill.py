from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torben_job_contract import health_path, isoformat, load_json, torben_home, write_health, write_json_atomic


def _load_jobs(profile_home: Path) -> list[dict[str, Any]]:
    payload = load_json(profile_home / "cron" / "jobs.json", {})
    jobs = payload.get("jobs") if isinstance(payload, dict) else []
    return [job for job in jobs if isinstance(job, dict)]


def _job_name(job: dict[str, Any]) -> str:
    return str(job.get("name") or "").strip()


def _latest_output_path(profile_home: Path, job: dict[str, Any]) -> Path | None:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return None
    output_dir = profile_home / "cron" / "output" / job_id
    if not output_dir.is_dir():
        return None
    outputs = sorted(path for path in output_dir.iterdir() if path.is_file())
    return outputs[-1] if outputs else None


def backfill_missing_health(*, profile_home: Path, apply: bool = False, include_disabled: bool = False) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for job in sorted(_load_jobs(profile_home), key=_job_name):
        name = _job_name(job)
        if not name or name.startswith("torben-desk-v2"):
            continue
        enabled = bool(job.get("enabled"))
        if not enabled and not include_disabled:
            records.append({"job": name, "status": "skipped", "reason": "disabled"})
            continue
        if health_path(name, profile_home).exists():
            records.append({"job": name, "status": "skipped", "reason": "health_exists"})
            continue
        last_status = str(job.get("last_status") or "").strip().lower()
        if last_status != "ok":
            records.append({"job": name, "status": "skipped", "reason": f"last_status_{last_status or 'missing'}"})
            continue
        last_run_at = str(job.get("last_run_at") or "").strip()
        if not last_run_at:
            records.append({"job": name, "status": "skipped", "reason": "missing_last_run_at"})
            continue
        output_path = _latest_output_path(profile_home, job)
        if output_path is None:
            records.append({"job": name, "status": "skipped", "reason": "missing_cron_output"})
            continue
        record = {
            "job": name,
            "status": "would_backfill",
            "last_run_at": last_run_at,
            "cron_output_path": str(output_path.relative_to(profile_home)),
        }
        if apply:
            health = write_health(
                name,
                status="ok",
                started_at=last_run_at,
                finished_at=last_run_at,
                exit_code=0,
                profile_home=profile_home,
            )
            health.update(
                {
                    "source": "scheduler_history_backfill",
                    "backfilled_at": isoformat(),
                    "scheduler_job_id": job.get("id"),
                    "scheduler_last_status": job.get("last_status"),
                    "cron_output_path": record["cron_output_path"],
                }
            )
            write_json_atomic(health_path(name, profile_home), health)
            record["status"] = "backfilled"
        records.append(record)
    return {
        "profile_home": str(profile_home),
        "apply": apply,
        "backfilled": sum(1 for record in records if record["status"] == "backfilled"),
        "would_backfill": sum(1 for record in records if record["status"] == "would_backfill"),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Torben job-health files from retained scheduler output.")
    parser.add_argument("--apply", action="store_true", help="Write missing health files instead of reporting candidates")
    parser.add_argument("--include-disabled", action="store_true", help="Also consider disabled jobs")
    args = parser.parse_args(argv)
    result = backfill_missing_health(profile_home=torben_home(), apply=args.apply, include_disabled=args.include_disabled)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
