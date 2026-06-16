#!/usr/bin/env python3
"""One-off repair: migrate cron jobs off exhausted openai-codex pins."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

HERMES_HOME = Path.home() / ".hermes"
JOBS_PATH = HERMES_HOME / "cron" / "jobs.json"
CONFIG_PATH = HERMES_HOME / "config.yaml"
STALE_WORKDIR = "C:/Users/downl/Desktop/clawdbot-main3/hermes-agent-upstream"
CURRENT_REPO = r"C:\Users\downl\Documents\New project\hermes-agent"

PINNED_PROVIDER = "openai-codex"
PINNED_MODELS = {"gpt-5.5", "gpt-5.4", "gpt-5.3-codex"}


def main() -> int:
    if not JOBS_PATH.exists():
        print(f"jobs.json not found: {JOBS_PATH}")
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = JOBS_PATH.with_suffix(f".json.bak-{stamp}")
    shutil.copy2(JOBS_PATH, backup)
    print(f"backup: {backup}")

    data = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    jobs = data.get("jobs", data if isinstance(data, list) else [])
    if not isinstance(jobs, list):
        print("unexpected jobs.json shape")
        return 1

    migrated = 0
    path_fixed = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        prov = (job.get("provider") or "").strip().lower()
        model = (job.get("model") or "").strip()
        if prov == PINNED_PROVIDER or model in PINNED_MODELS:
            job["provider"] = None
            job["model"] = None
            migrated += 1
            print(f"  cleared pin: {job.get('id')} {job.get('name', '')[:50]}")

        prompt = job.get("prompt")
        if isinstance(prompt, str) and STALE_WORKDIR in prompt:
            job["prompt"] = prompt.replace(STALE_WORKDIR, CURRENT_REPO)
            path_fixed += 1
            print(f"  fixed path: {job.get('id')}")

    if isinstance(data, dict):
        data["jobs"] = jobs
        out = data
    else:
        out = jobs

    JOBS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"migrated={migrated} path_fixed={path_fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
