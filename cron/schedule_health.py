"""Operator-facing reasons for recurring schedules that cannot compute a next run."""


def next_run_error(schedule: dict, croniter_error: str | None) -> str:
    if schedule.get("kind") == "cron":
        cause = croniter_error or "croniter is unavailable or the cron expression is invalid"
        return (
            f"Cannot compute the next cron run: {cause}. The scheduler retries automatically. "
            "Check the cron.jobs warning for the running interpreter; if its dependencies are "
            "broken, run `hermes pm repair` in the affected installation."
        )
    return "Cannot compute the next recurring run; check the stored schedule."
