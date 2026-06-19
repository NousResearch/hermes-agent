#!/usr/bin/env python3
"""Register or update the job-seeker daily scan cron job in ~/.hermes/cron/jobs.json."""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "a1ee7700job1"
JOB_NAME = "求職スキャン（BizReach/Findy/LAPRAS/クラウドワークス）"
SCRIPT_NAME = "job-seeker-telegram-digest.py"
OPS_DIR = Path(r"C:/Users/downl/Documents/ops/job-seeker")

PROMPT = r"""（no_agent モード: このプロンプトは未使用。stdout が Telegram に配信される）

スキャン本体は run_scan.py。求人ダイジェスト（タイトル・会社・URL）が Telegram に届く。
詳細レポート: ops/job-seeker/reports/YYYY-MM-DD-scan.md
"""


def _hermes_home() -> Path:
    import os

    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _load_jobs(path: Path) -> dict:
    if not path.exists():
        return {"jobs": []}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_jobs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_job(*, profile: str, telegram_origin: dict | None = None) -> dict:
    now = datetime.now(timezone.utc).astimezone().isoformat()
    origin = telegram_origin or {
        "platform": "telegram",
        "chat_id": "7201110294",
        "chat_name": "Home",
        "thread_id": None,
    }
    return {
        "id": JOB_ID,
        "name": JOB_NAME,
        "prompt": PROMPT,
        "skills": ["google-workspace", "ai-employee-org"],
        "skill": "ai-employee-org",
        "model": None,
        "provider": None,
        "base_url": None,
        "script": SCRIPT_NAME,
        "no_agent": True,
        "context_from": None,
        "schedule": {"kind": "cron", "expr": "0 9,18 * * *", "display": "0 9,18 * * *"},
        "schedule_display": "0 9,18 * * *",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": None,
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        "deliver": "origin",
        "origin": origin,
        "enabled_toolsets": ["web", "browser", "terminal", "file"],
        "workdir": str(OPS_DIR),
        "profile": profile,
    }


def _deploy_cron_script(profile: str) -> Path:
    """Copy telegram digest wrapper into the profile's HERMES_HOME/scripts."""
    import os
    import shutil

    src = Path(__file__).resolve().parent / SCRIPT_NAME
    if not src.is_file():
        raise FileNotFoundError(f"Missing cron script source: {src}")

    if profile and profile.lower() != "default":
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        # Installer runs from default shell — resolve profile home explicitly.
        profiles_root = Path.home() / ".hermes" / "profiles"
        home = profiles_root / profile
    else:
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    scripts_dir = home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    dest = scripts_dir / SCRIPT_NAME
    shutil.copy2(src, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="job-seeker", help="Hermes profile for cron")
    parser.add_argument("--telegram-chat-id", default="", help="Telegram chat_id for delivery")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    (OPS_DIR / "reports").mkdir(exist_ok=True)
    (OPS_DIR / "applications").mkdir(exist_ok=True)
    seen = OPS_DIR / "seen.json"
    if not seen.exists():
        seen.write_text("[]\n", encoding="utf-8")

    jobs_path = _hermes_home() / "cron" / "jobs.json"
    store = _load_jobs(jobs_path)
    jobs = store.setdefault("jobs", [])
    origin = None
    if args.telegram_chat_id.strip():
        origin = {
            "platform": "telegram",
            "chat_id": args.telegram_chat_id.strip(),
            "chat_name": "Home",
            "thread_id": None,
        }
    new_job = build_job(profile=args.profile, telegram_origin=origin)

    replaced = False
    for i, job in enumerate(jobs):
        if job.get("id") == JOB_ID or job.get("name") == JOB_NAME:
            jobs[i] = {**job, **new_job, "repeat": job.get("repeat", new_job["repeat"])}
            replaced = True
            break
    if not replaced:
        jobs.append(new_job)

    if args.dry_run:
        print(json.dumps(new_job, ensure_ascii=False, indent=2))
        return 0

    deployed = _deploy_cron_script(args.profile)
    _save_jobs(jobs_path, store)
    print(f"{'Updated' if replaced else 'Added'} cron job {JOB_ID} in {jobs_path}")
    print(f"Deployed script: {deployed}")
    print("Schedule: 09:00 and 18:00 daily - no_agent script -> Telegram digest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
