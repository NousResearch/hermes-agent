from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse
from apscheduler.triggers.cron import CronTrigger

from hermes_snapshot_manager.core.config import DEFAULT_EXCLUDE_PATTERNS, load_settings, update_settings
from hermes_snapshot_manager.core.paths import build_paths

router = APIRouter(tags=["settings"])


@router.get("/api/settings")
def get_settings() -> dict:
    return asdict(load_settings(build_paths()))


@router.post("/settings")
def update_settings_from_ui(
    schedule_enabled: bool = Form(default=False),
    schedule_cron: str = Form(default="0 */6 * * *"),
    retention_hourly: int = Form(default=24),
    retention_daily: int = Form(default=30),
    retention_weekly: int = Form(default=12),
    exclude_patterns: str = Form(default=""),
) -> RedirectResponse:
    app_paths = build_paths()
    current = load_settings(app_paths)
    parsed_excludes = [line.strip() for line in exclude_patterns.splitlines() if line.strip()]
    schedule_cron = schedule_cron.strip() or "0 */6 * * *"
    try:
        CronTrigger.from_crontab(schedule_cron)
    except ValueError as exc:
        return RedirectResponse(url=f"/settings?error={str(exc)}", status_code=303)
    update_settings(
        {
            "schedule_enabled": schedule_enabled,
            "schedule_cron": schedule_cron,
            "retention_hourly": retention_hourly,
            "retention_daily": retention_daily,
            "retention_weekly": retention_weekly,
            "exclude_patterns": parsed_excludes or current.exclude_patterns,
        },
        app_paths,
    )
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/excludes/defaults")
def reset_excludes_to_defaults() -> RedirectResponse:
    app_paths = build_paths()
    update_settings({"exclude_patterns": DEFAULT_EXCLUDE_PATTERNS.copy()}, app_paths)
    return RedirectResponse(url="/settings?saved=1", status_code=303)
