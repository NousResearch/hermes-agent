#!/usr/bin/env python3
"""Register secretary daily heartbeat cron in ~/.hermes/cron/jobs.json."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "a3ee7700sec1"
JOB_NAME = "秘書ハートビート（Kanban棚卸し）"

PROMPT = r"""あなたは secretary プロファイルの秘書エージェントです。ai-company ボードを点検します。

【必須手順】
1. `hermes kanban list --status ready,running,blocked,todo` で全体を把握。
2. ready が空なら、blocked のうち人間判断待ち以外を comment し、必要なら promote。
3. ready/todo が完全に空なら、kanban_create で種タスク1件（assignee=secretary, 優先度低）を追加し、子タスク案を comment のみ（勝手に大量作成しない）。
4. Telegram 要約10行以内。応募・返信・公開は block のまま触らない。

【ルール】
- 429 時は fallback_providers に従う。推測でモデルを変えない。
"""


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _load_jobs(path: Path) -> dict:
    if not path.exists():
        return {"jobs": []}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_jobs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
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
        "skills": ["ai-employee-org"],
        "skill": "ai-employee-org",
        "model": None,
        "provider": None,
        "base_url": None,
        "script": None,
        "no_agent": False,
        "context_from": None,
        "schedule": {"kind": "cron", "expr": "30 8 * * *", "display": "30 8 * * *"},
        "schedule_display": "30 8 * * * (daily 08:30)",
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
        "enabled_toolsets": ["terminal", "file", "kanban"],
        "workdir": None,
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="secretary")
    parser.add_argument("--telegram-chat-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    origin = None
    if args.telegram_chat_id.strip():
        origin = {
            "platform": "telegram",
            "chat_id": args.telegram_chat_id.strip(),
            "chat_name": "Home",
            "thread_id": None,
        }

    jobs_path = _hermes_home() / "cron" / "jobs.json"
    store = _load_jobs(jobs_path)
    jobs = store.setdefault("jobs", [])
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

    _save_jobs(jobs_path, store)
    print(f"{'Updated' if replaced else 'Added'} cron job {JOB_ID} in {jobs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
