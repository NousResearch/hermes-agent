#!/usr/bin/env python3
"""Install/update the daily Bilibili Hermes Agent weekly-top cron job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


JOB_NAME = "Daily Bilibili Hermes Agent Weekly Top Videos to Feishu"
SKILL_NAME = "bilibili-hermes-agent-weekly-top-feishu"
RUNTIME_FETCH_SCRIPT = "bilibili_hermes_agent_weekly_top.py"
DEFAULT_SCHEDULE = "40 23 * * *"

CRON_PROMPT = """请根据 Script Output 中的 JSON，生成一条适合发送到飞书的中文 B 站 Hermes Agent 本周视频热榜。

要求：
1. 标题使用“B站 Hermes Agent 本周播放量 Top 10（YYYY-MM-DD）”，日期使用 generated_at 的日期。
2. 只输出 items 中排名最高的 10 条；如果不足 10 条，按实际数量说明。
3. 每条包含：排名、标题、UP 主、播放量、点赞、收藏、评论/弹幕、发布时间、视频链接。
4. 开头说明统计周区间（week_start 到 filter_end）、搜索关键词、抓取结果数量、匹配视频数量。
5. 如果 items 为空，说明本周暂未抓到匹配视频，并列出 errors 中的关键错误（如有）。
6. 不要编造 Script Output 里没有的视频、播放量、作者或链接。
7. 保持简洁，适合在飞书中阅读。
"""


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def skill_dir_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_hermes_env(hermes_home: Path) -> dict[str, str]:
    env_path = hermes_home / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def copy_skill_to_hermes_home(skill_dir: Path, hermes_home: Path) -> Path:
    target = hermes_home / "skills" / "research" / SKILL_NAME
    if skill_dir.resolve() == target.resolve():
        return target
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(skill_dir, target, ignore=ignore)
    return target


def install_fetch_script(skill_dir: Path, hermes_home: Path) -> Path:
    source = skill_dir / "scripts" / "fetch_bilibili_hermes_agent_weekly_top.py"
    if not source.exists():
        raise FileNotFoundError(f"missing crawler script: {source}")
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / RUNTIME_FETCH_SCRIPT
    shutil.copy2(source, target)
    target.chmod(0o700)
    return target


def find_existing_job(jobs: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for job in jobs:
        if job.get("name") == name:
            return job
    return None


def create_or_update_job(args: argparse.Namespace) -> dict[str, Any]:
    from cron.jobs import create_job, list_jobs, trigger_job, update_job

    existing = find_existing_job(list_jobs(include_disabled=True), args.name)
    updates = {
        "prompt": CRON_PROMPT,
        "schedule": args.schedule,
        "deliver": args.deliver,
        "skills": [SKILL_NAME],
        "skill": SKILL_NAME,
        "script": RUNTIME_FETCH_SCRIPT,
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
    }
    if existing:
        job = update_job(existing["id"], updates)
        action = "updated"
    else:
        job = create_job(
            prompt=CRON_PROMPT,
            schedule=args.schedule,
            name=args.name,
            deliver=args.deliver,
            skills=[SKILL_NAME],
            script=RUNTIME_FETCH_SCRIPT,
        )
        action = "created"

    if args.trigger_now and job:
        job = trigger_job(job["id"]) or job
        action += "_and_triggered"

    return {"action": action, "job": job}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/update the daily Bilibili Hermes Agent Feishu cron job.")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="Cron schedule, default: 40 23 * * *")
    parser.add_argument("--deliver", default="feishu", help="Delivery target, e.g. feishu or feishu:oc_xxx")
    parser.add_argument("--name", default=JOB_NAME)
    parser.add_argument("--trigger-now", action="store_true", help="Also run on the next scheduler tick.")
    parser.add_argument("--skip-skill-install", action="store_true", help="Do not copy the skill into HERMES_HOME/skills.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    repo_root = repo_root_from_script()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    env_values = load_hermes_env(hermes_home)
    skill_dir = skill_dir_from_script()

    installed_skill = None
    if not args.skip_skill_install:
        installed_skill = copy_skill_to_hermes_home(skill_dir, hermes_home)

    runtime_script = install_fetch_script(skill_dir, hermes_home)
    result = create_or_update_job(args)

    warnings = []
    if args.deliver.strip().lower() == "feishu" and not env_values.get("FEISHU_HOME_CHANNEL") and not os.getenv("FEISHU_HOME_CHANNEL"):
        warnings.append(
            "deliver=feishu needs FEISHU_HOME_CHANNEL in HERMES_HOME/.env, or use --deliver 'feishu:<chat_id>'."
        )

    output = {
        "success": True,
        "action": result["action"],
        "job": result["job"],
        "runtime_script": str(runtime_script),
        "installed_skill": str(installed_skill) if installed_skill else None,
        "warnings": warnings,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        job = result["job"]
        print(f"{output['action']}: {job['name']} ({job['id']})")
        print(f"schedule: {job.get('schedule_display')}")
        print(f"deliver: {job.get('deliver')}")
        print(f"next_run_at: {job.get('next_run_at')}")
        print(f"runtime_script: {runtime_script}")
        if installed_skill:
            print(f"installed_skill: {installed_skill}")
        for warning in warnings:
            print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
