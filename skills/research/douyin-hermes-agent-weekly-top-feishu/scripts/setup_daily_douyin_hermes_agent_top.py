#!/usr/bin/env python3
"""Install/update the daily Douyin Hermes Agent weekly-top cron job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


JOB_NAME = "Daily Douyin Hermes Agent Weekly Top Videos to Feishu"
SKILL_NAME = "douyin-hermes-agent-weekly-top-feishu"
RUNTIME_FETCH_SCRIPT = "douyin_hermes_agent_weekly_top.py"
DEFAULT_SCHEDULE = "10 23 * * *"

CRON_PROMPT = """请根据 Script Output 中的 JSON，生成一条适合发送到飞书的中文抖音 Hermes Agent 本周视频热榜。

要求：
1. 标题使用“抖音 Hermes Agent 本周播放量 Top 10（YYYY-MM-DD）”，日期使用 generated_at 的日期。
2. 只输出 items 中排名最高的 10 条；如果不足 10 条，按实际数量说明。
3. 每条包含：排名、标题/描述、作者、播放量、点赞、评论、分享、收藏、发布时间、视频链接。
4. 开头说明统计周区间（week_start 到 filter_end）、搜索关键词、抓取结果数量、匹配视频数量。
5. 如果 items 为空，说明本周暂未抓到匹配视频，并列出 errors 中的关键错误（如有），尤其是抖音登录 Cookie 缺失或接口要求登录。
6. 如果 notes 提示部分视频没有 play_count，说明这些视频按 0 播放量参与排序，不要自行补全播放量。
7. 不要编造 Script Output 里没有的视频、播放量、作者或链接。
8. 保持简洁，适合在飞书中阅读。
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
    source = skill_dir / "scripts" / "fetch_douyin_hermes_agent_weekly_top.py"
    if not source.exists():
        raise FileNotFoundError(f"missing crawler script: {source}")
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / RUNTIME_FETCH_SCRIPT
    shutil.copy2(source, target)
    target.chmod(0o700)
    return target


def load_fetch_module(skill_dir: Path):
    source = skill_dir / "scripts" / "fetch_douyin_hermes_agent_weekly_top.py"
    spec = importlib.util.spec_from_file_location("douyin_fetch_for_setup", source)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def douyin_cookie_source(skill_dir: Path, env_values: dict[str, str]) -> str | None:
    explicit_keys = ("DOUYIN_COOKIE", "DOUYIN_SEARCH_COOKIE", "DOUYIN_WEB_COOKIE")
    if any(env_values.get(key) or os.getenv(key) for key in explicit_keys):
        return "environment"

    try:
        fetch_module = load_fetch_module(skill_dir)
        if fetch_module and fetch_module.firefox_cookie_header():
            return "firefox"
    except Exception:  # noqa: BLE001 - setup should warn, not fail, on cookie probing
        return None
    return None


def find_existing_job(jobs: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for job in jobs:
        if job.get("name") == name:
            return job
    return None


def first_feishu_channel_from_directory(hermes_home: Path) -> str | None:
    directory_path = hermes_home / "channel_directory.json"
    if not directory_path.exists():
        return None
    try:
        payload = json.loads(directory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    feishu_channels = ((payload.get("platforms") or {}).get("feishu") or [])
    if len(feishu_channels) != 1:
        return None
    channel_id = str(feishu_channels[0].get("id") or "").strip()
    if not channel_id:
        return None
    return f"feishu:{channel_id}"


def resolve_default_deliver(deliver: str, hermes_home: Path, env_values: dict[str, str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    requested = deliver.strip() or "feishu"
    if requested.lower() != "feishu":
        return requested, warnings
    if env_values.get("FEISHU_HOME_CHANNEL") or os.getenv("FEISHU_HOME_CHANNEL"):
        return requested, warnings

    explicit_channel = first_feishu_channel_from_directory(hermes_home)
    if explicit_channel:
        warnings.append(f"FEISHU_HOME_CHANNEL is not set; using channel_directory target {explicit_channel}.")
        return explicit_channel, warnings

    warnings.append("deliver=feishu needs FEISHU_HOME_CHANNEL in HERMES_HOME/.env, or use --deliver 'feishu:<chat_id>'.")
    return requested, warnings


def create_or_update_job(args: argparse.Namespace, deliver: str) -> dict[str, Any]:
    from cron.jobs import create_job, list_jobs, trigger_job, update_job

    existing = find_existing_job(list_jobs(include_disabled=True), args.name)
    updates = {
        "prompt": CRON_PROMPT,
        "schedule": args.schedule,
        "deliver": deliver,
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
            deliver=deliver,
            skills=[SKILL_NAME],
            script=RUNTIME_FETCH_SCRIPT,
        )
        action = "created"

    if args.trigger_now and job:
        job = trigger_job(job["id"]) or job
        action += "_and_triggered"

    return {"action": action, "job": job}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/update the daily Douyin Hermes Agent Feishu cron job.")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="Cron schedule, default: 10 23 * * *")
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
    resolved_deliver, warnings = resolve_default_deliver(args.deliver, hermes_home, env_values)
    result = create_or_update_job(args, resolved_deliver)

    cookie_source = douyin_cookie_source(skill_dir, env_values)
    if cookie_source == "firefox":
        warnings.append("No explicit DOUYIN_COOKIE is set; live Douyin search will use Firefox profile cookies.")
    elif not cookie_source:
        warnings.append(
            "No Douyin login cookie found. Set DOUYIN_COOKIE/DOUYIN_SEARCH_COOKIE, "
            "or log in to Douyin with Firefox and set DOUYIN_FIREFOX_PROFILE if needed."
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
