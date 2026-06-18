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
OPS_DIR = Path(r"C:/Users/downl/Documents/ops/job-seeker")

PROMPT = r"""あなたは job-seeker プロファイルの求職エージェントです。スキル ai-employee-org の references/job-sources-jp.md に従い、次を実行してください。

【永続パス】C:/Users/downl/Documents/ops/job-seeker（seen.json, reports/, applications/ を維持）

【必須ツール】
- Gmail: google-workspace skill の google_api.py（himalaya 禁止）
- 公開情報: web_search
- ログイン必須サイト: browser_navigate（ビズリーチ、ファインディ、LAPRAS、クラウドワークス）
- LAPRAS API: optional-skills/.../scripts/lapras_jobs.py（LAPRAS_API_KEY がある場合）

【監視先】
1. ビズリーチ — AI/ML スカウト・求人
2. ファインディ — AIエンジニア / MLOps
3. LAPRAS — API またはブラウザ + Gmail
4. クラウドワークス — ソフトウェア外注案件（開発・AI）

【Gmail 例】
py -3 C:/Users/downl/.hermes/skills/productivity/google-workspace/scripts/google_api.py gmail search '(ビズリーチ OR bizreach OR findy OR lapras OR クラウドワークス) (AI OR 機械学習 OR MLOps OR スカウト) newer_than:3d' --max 20

【重複排除】
seen.json に URL/ハッシュを追記。新規のみ kanban_create（board: ai-company, assignee: job-seeker, workspace: dir:C:/Users/downl/Documents/ops/job-seeker）。

【承認】
応募・入札・返信は kanban_block。ドラフト保存のみ complete 可。

【429】
主 NVIDIA → fallback_providers の Nous Free → 最後のみ 127.0.0.1:8080 llama。推測でモデルを変えない。

【出力】
reports/YYYY-MM-DD-scan.md に要約（日本語、表禁止、見出し+箇条書き）。Telegram には先頭10行のみ要約。
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
        "script": None,
        "no_agent": False,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="job-seeker", help="Hermes profile for cron")
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
    new_job = build_job(profile=args.profile)

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
    print("Schedule: 09:00 and 18:00 daily (cron 0 9,18 * * *)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
