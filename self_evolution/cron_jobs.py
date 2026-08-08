"""
Self Evolution Plugin — Cron Job Registration
==============================================

Registers two cron jobs:
  1. dream_time (1:00):  Run dream consolidation
  2. propose_time (19:00): Push proposals via Feishu

Uses Hermes' existing cron system (cron/jobs.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from self_evolution.paths import CRON_DIR

CRON_FILE = CRON_DIR / "jobs.json"

DREAM_JOB_ID = "self_evolution_dream"
PROPOSE_JOB_ID = "self_evolution_propose"


def register_cron_jobs():
    """Register the two self_evolution cron jobs if not already present."""
    CRON_DIR.mkdir(parents=True, exist_ok=True)

    jobs = _load_jobs()

    # Resolve model config from hermes unified config
    from self_evolution.reflection_engine import _resolve_runtime_config
    runtime = _resolve_runtime_config()
    model = runtime.get("model", "")
    provider = runtime.get("provider", "")

    # Dream consolidation at 1:00
    if not any(j.get("id") == DREAM_JOB_ID for j in jobs):
        jobs.append({
            "id": DREAM_JOB_ID,
            "name": "Self Evolution - Dream Consolidation",
            "prompt": "运行自我进化的梦境整理：分析前日session的错误和浪费时间问题，生成进化提案。",
            "schedule": "0 1 * * *",
            "model": model,
            "provider": provider,
            "deliver": "[SILENT]",
            "skill": "self_evolution:dream",
        })

    # Proposal push at 19:00
    if not any(j.get("id") == PROPOSE_JOB_ID for j in jobs):
        jobs.append({
            "id": PROPOSE_JOB_ID,
            "name": "Self Evolution - Proposal Push",
            "prompt": "推送今日自我进化提案到飞书。",
            "schedule": "0 19 * * *",
            "model": model,
            "provider": provider,
            "deliver": "[SILENT]",
            "skill": "self_evolution:propose",
        })

    _save_jobs(jobs)
    logger.info("Registered self_evolution cron jobs: dream=1:00, propose=19:00")


def run_dream_job():
    """Execute the dream consolidation job.

    Called by the cron system at 1:00.
    Uses hermes unified runtime provider for model config.
    """
    from self_evolution.reflection_engine import DreamEngine

    # DreamEngine() with no args auto-resolves via resolve_runtime_provider()
    engine = DreamEngine()
    report = engine.run(hours=24, max_runtime_seconds=6 * 3600)

    if report:
        logger.info("Dream consolidation complete: score=%.3f, proposals generated", report.avg_score)
    else:
        logger.info("Dream consolidation: no data to analyze")


def run_propose_job():
    """Execute the proposal push job.

    Called by the cron system at 19:00.
    """
    from self_evolution.feishu_notifier import FeishuNotifier

    notifier = FeishuNotifier()
    notifier.send_daily_report()


def _load_jobs() -> list:
    """Load existing cron jobs."""
    if not CRON_FILE.exists():
        return []
    try:
        return json.loads(CRON_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_jobs(jobs: list):
    """Save cron jobs."""
    CRON_FILE.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
