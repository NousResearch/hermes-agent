from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone

from hermes_snapshot_manager.core.config import load_settings
from hermes_snapshot_manager.core.paths import AppPaths, build_paths
from hermes_snapshot_manager.services.retention_service import RetentionService
from hermes_snapshot_manager.services.snapshot_service import SnapshotService


class SchedulerService:
    def __init__(self, paths: AppPaths | None = None):
        self.paths = paths or build_paths()
        self.settings = load_settings(self.paths)
        self.snapshot_service = SnapshotService(self.paths, self.settings)
        self.retention_service = RetentionService(self.snapshot_service)
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def refresh_settings(self):
        self.settings = load_settings(self.paths)
        self.snapshot_service.settings = self.settings
        return self.settings

    def describe_schedule(self) -> str:
        return describe_cron(self.refresh_settings().schedule_cron)

    def next_run_info(self) -> dict:
        settings = self.refresh_settings()
        if not settings.schedule_enabled:
            return {
                "enabled": False,
                "description": "Automatic snapshots are off",
                "next_run_utc": None,
                "next_run_relative": "Scheduling disabled",
            }
        trigger = CronTrigger.from_crontab(settings.schedule_cron, timezone="UTC")
        now = datetime.now(timezone.utc)
        next_run = trigger.get_next_fire_time(previous_fire_time=None, now=now)
        return {
            "enabled": True,
            "description": describe_cron(settings.schedule_cron),
            "next_run_utc": None if next_run is None else next_run.strftime("%Y-%m-%d %H:%M UTC"),
            "next_run_relative": _humanize_future_delta(next_run, now),
        }

    def start(self) -> None:
        settings = self.refresh_settings()
        if not settings.schedule_enabled:
            return
        trigger = CronTrigger.from_crontab(settings.schedule_cron)
        self.scheduler.add_job(self.snapshot_service.create_snapshot, trigger=trigger, kwargs={"trigger_type": "scheduled"}, id="snapshot-job", replace_existing=True)
        self.scheduler.add_job(self.retention_service.cleanup, trigger=trigger, id="retention-job", replace_existing=True)
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)


def _humanize_future_delta(target: datetime | None, now: datetime | None = None) -> str:
    if target is None:
        return "No future run"
    baseline = now or datetime.now(timezone.utc)
    seconds = max(int((target - baseline).total_seconds()), 0)
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    pieces = []
    if days:
        pieces.append(f"{days}d")
    if hours:
        pieces.append(f"{hours}h")
    if minutes:
        pieces.append(f"{minutes}m")
    return "in <1m" if not pieces else "in " + " ".join(pieces[:2])


def describe_cron(cron: str) -> str:
    parts = cron.split()
    if len(parts) != 5:
        return "Custom cron schedule"
    minute, hour, day_of_month, month, day_of_week = parts
    if minute == "0" and hour.startswith("*/") and day_of_month == month == day_of_week == "*":
        return f"Every {hour[2:]} hours"
    if day_of_month == month == day_of_week == "*" and minute.isdigit() and hour.isdigit():
        return f"Daily at {int(hour):02d}:{int(minute):02d} UTC"
    if day_of_month == month == "*" and day_of_week.isdigit() and minute.isdigit() and hour.isdigit():
        weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return f"{weekday_names[int(day_of_week)]} at {int(hour):02d}:{int(minute):02d} UTC"
    return "Custom cron schedule"
