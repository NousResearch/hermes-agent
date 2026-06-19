#!/usr/bin/env python3
"""Register or update the self-improver weekly review cron job in ~/.hermes/cron/jobs.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "a2ee7700si01"
JOB_NAME = "自己改善週次レビュー（curator + skill棚卸し）"
WORKSPACE = Path(r"C:/Users/downl/Documents/New project/hermes-agent")
OPS_INBOX = Path(r"C:/Users/downl/Documents/ops/cursor-learning-inbox")

PROMPT = r"""あなたは self-improver プロファイルの自己改善エージェントです。週次の棚卸しと skill ライブラリのメンテを行います。

【永続パス】
- ワークスペース: C:/Users/downl/Documents/New project/hermes-agent（_docs/ へレポート）
- Cursor学習インボックス: C:/Users/downl/Documents/ops/cursor-learning-inbox（*.md を取り込み）

【必須手順 — この順で実行】
1. `hermes curator status` を実行し、agent作成 skill の状態（active/stale/archived、use_count、patch_count）を把握する。
2. `hermes curator run` を実行する（config で consolidate: true — 重複 skill の統合も試みる）。
3. `hermes logs --level WARNING` で直近の警告を確認（terminal 経由、最大30行要約）。
4. Cursor インボックス `C:/Users/downl/Documents/ops/cursor-learning-inbox/*.md` を読み、未取り込みがあれば skill_manage で既存の傘 skill へ patch または references/ へ write_file。取り込み済みは `processed/` へ移動。
5. agent作成 skill（created_by: agent）のうち patch 候補を最大3件選び、skill_manage action=patch で改善（新規 create は傘が無い場合のみ）。
6. レポートを `C:/Users/downl/Documents/New project/hermes-agent/_docs/YYYY-MM-DD_self-review.md` に書く（日本語、表禁止、見出し+箇条書き）。

【レポート必須セクション】
## サマリー（3行以内）
## curator 結果（統合/アーカイブ/prune の有無）
## skill 更新（patch/create/write_file の一覧）
## Cursor インボックス取り込み
## 警告ログの要点
## 次週の推奨アクション

【ルール】
- bundled / hub-installed skill は編集禁止。
- pinned skill は patch 可、delete/archive 不可。
- 新規 skill は class-level umbrella のみ。セッション固有名は references/ へ。
- Telegram 配信は要約10行以内。秘密情報・token は出力しない。
- 何も変更がなければ「変更なし」と明記し、無理に create しない。
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
        "skills": ["ai-employee-org", "cursor-hermes-skill-sync"],
        "skill": "ai-employee-org",
        "model": None,
        "provider": None,
        "base_url": None,
        "script": None,
        "no_agent": False,
        "context_from": None,
        "schedule": {"kind": "cron", "expr": "0 18 * * 0", "display": "0 18 * * 0"},
        "schedule_display": "0 18 * * 0 (Sun 18:00)",
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
        "enabled_toolsets": ["terminal", "file", "skills", "search"],
        "workdir": str(WORKSPACE),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="self-improver", help="Hermes profile for cron")
    parser.add_argument("--telegram-chat-id", default="", help="Telegram chat_id for delivery")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "_docs").mkdir(exist_ok=True)
    OPS_INBOX.mkdir(parents=True, exist_ok=True)
    (OPS_INBOX / "processed").mkdir(exist_ok=True)

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
            merged = {**job, **new_job}
            merged["repeat"] = job.get("repeat", new_job["repeat"])
            jobs[i] = merged
            replaced = True
            break
    if not replaced:
        jobs.append(new_job)

    if args.dry_run:
        print(json.dumps(new_job, ensure_ascii=False, indent=2))
        return 0

    _save_jobs(jobs_path, store)
    print(f"{'Updated' if replaced else 'Added'} cron job {JOB_ID} in {jobs_path}")
    print("Schedule: Sunday 18:00 (cron 0 18 * * 0), profile=self-improver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
