#!/usr/bin/env python3
"""Register job-recruiter weekly posting review cron in ~/.hermes/cron/jobs.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "a4ee7700rec1"
JOB_NAME = "求人票レビュー（下書き・公開準備）"
OPS_DIR = Path(r"C:/Users/downl/Documents/ops/job-recruiter")

PROMPT = r"""あなたは job-recruiter プロファイルの求人エージェントです。公開中求人票と新規ドラフトを整備します。

【永続パス】C:/Users/downl/Documents/ops/job-recruiter（postings/, drafts/, templates/ を維持）

【必須手順】
1. `dir:C:/Users/downl/Documents/ops/job-recruiter` を workspace として listings を確認（無ければ templates/ から1件ドラフト作成）。
2. `web_search` で同職種のタイトル・年収レンジの相場を1回だけ調べ、既存ドラフトに patch。
3. 公開待ちがあれば `kanban_block`（人間承認必須）。ドラフトのみなら `reports/YYYY-MM-DD-recruiter.md` に保存して complete 可。
4. 新規募集ニーズが秘書タスクに無い場合は、kanban_create しない（重複禁止）。

【出力】
- reports/YYYY-MM-DD-recruiter.md（日本語、表禁止）
- Telegram には要約10行以内

【ルール】
- 外部投稿・公開は必ず block。ドラフト保存までが自動、公開は人間承認。
- 429 時は fallback_providers に従う。
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
        "schedule": {"kind": "cron", "expr": "0 11 * * 2,5", "display": "0 11 * * 2,5"},
        "schedule_display": "0 11 * * 2,5 (Tue/Fri 11:00)",
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
        "enabled_toolsets": ["web", "terminal", "file"],
        "workdir": str(OPS_DIR),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="job-recruiter", help="Hermes profile for cron")
    parser.add_argument("--telegram-chat-id", default="", help="Telegram chat_id for delivery")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("postings", "drafts", "templates", "reports"):
        (OPS_DIR / sub).mkdir(exist_ok=True)

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

    _save_jobs(jobs_path, store)
    print(f"{'Updated' if replaced else 'Added'} cron job {JOB_ID} in {jobs_path}")
    print("Schedule: Tue/Fri 11:00 (cron 0 11 * * 2,5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
