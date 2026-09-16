"""Subprocess cron provider.

Keeps the built-in, profile-aware ticker while making every claimed execution
cross the scheduler's durable detached-worker handoff.  The child runs the
shared ``run_one_job`` path, including drift/preflight guards and queued
delivery, so this provider changes process isolation rather than semantics.
"""

from cron.scheduler_provider import InProcessCronScheduler


class SubprocessCronScheduler(InProcessCronScheduler):
    """Built-in trigger with mandatory subprocess execution."""

    @property
    def name(self) -> str:
        return "subprocess"

    @property
    def uses_detached_workers(self) -> bool:
        return True


def register(ctx) -> None:
    ctx.register_cron_scheduler(SubprocessCronScheduler())