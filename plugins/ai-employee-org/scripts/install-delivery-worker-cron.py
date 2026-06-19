#!/usr/bin/env python3
"""Register delivery-worker daily queue check cron in ~/.hermes/cron/jobs.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "a5ee7700del1"
JOB_NAME = "受注キュー点検（進行中案件・ブロッカー）"
OPS_DIR = Path(r"C:/Users/downl/Documents/ops/delivery")

PROMPT = r"""あなたは delivery-worker プロファイルの受注達成エージェントです。進行中の受注タスクを点検します。

【永続パス】C:/Users/downl/Documents/ops/delivery（projects/, reports/ を維持）

【必須手順】
1. `hermes kanban list --status ready --assignee delivery-worker` で未着手を確認。
2. `hermes kanban list --status running --assignee delivery-worker` で長時間スタックを確認。
3. ready が空なら、board ai-company で assignee=delivery-worker の todo/blocked を確認し、着手可能なら promote または comment のみ（勝手に complete しない）。
4. 進行中案件ごとに reports/YYYY-MM-DD-delivery.md にブロッカー・次アクションを記録。
5. 受入基準が曖昧な案件は `kanban_block` で秘書/人間へエスカレーション。

【出力】
- reports/YYYY-MM-DD-delivery.md（日本語、表禁止）
- Telegram 要約10行以内

【ルール】
- 納品物パスは dir: workspace 配下に置く。scratch のみにしない。
- テスト可能なリポジトリでは complete 前にテストコマンドを実行。
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
        "schedule": {"kind": "cron", "expr": "0 10 * * 1-5", "display": "0 10 * * 1-5"},
        "schedule_display": "0 10 * * 1-5 (weekdays 10:00)",
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
        "workdir": str(OPS_DIR),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="delivery-worker", help="Hermes profile for cron")
    parser.add_argument("--telegram-chat-id", default="", help="Telegram chat_id for delivery")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("projects", "reports"):
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
    print("Schedule: weekdays 10:00 (cron 0 10 * * 1-5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
