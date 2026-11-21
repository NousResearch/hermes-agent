"""Cron job scheduling for Hermes Agent: scheduled tasks (cron expressions, intervals, one-shot),
self-scheduled reminders, isolated sessions. The gateway daemon (``hermes gateway [install]``) ticks
the scheduler every 60 seconds; a file lock prevents duplicate execution across processes.
"""

# The restart-safe external worker runs as ``-m cron.scheduler``, which executes this package
# first: boot PM dependencies before ``cron.jobs`` reaches a third-party import. A no-op
# unless ``_launch_external_cron_worker`` marked this process. See cron/worker_bootstrap.py.
from cron.worker_bootstrap import worker_bootstrap as _boot_external_worker

_boot_external_worker()

from cron.jobs import (
    create_job,
    get_job,
    list_jobs,
    remove_job,
    update_job,
    pause_job,
    resume_job,
    trigger_job,
    rearm_oneshot,
    JOBS_FILE,
)


def __getattr__(name: str):
    """Resolve ``tick`` on first access instead of importing ``cron.scheduler`` here.

    ``python -m cron.scheduler`` executes this package first, so a module-level
    ``from cron.scheduler import tick`` puts the target module in ``sys.modules`` before
    runpy executes it, emitting

        RuntimeWarning: 'cron.scheduler' found in sys.modules after import of package
        'cron', but prior to execution of 'cron.scheduler'; this may result in
        unpredictable behaviour

    on every external cron worker start. The warning is harmless, but it is the first
    line of worker stderr, so it reads like the cause in cron failure notifications.
    Deferring the import keeps ``cron.tick`` available to callers while leaving
    ``sys.modules`` clean during runpy execution.
    """
    if name == "tick":
        from cron.scheduler import tick

        return tick
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JOBS_FILE",
    "create_job",
    "get_job",
    "list_jobs",
    "pause_job",
    "rearm_oneshot",
    "remove_job",
    "resume_job",
    "tick",
    "trigger_job",
    "update_job",
]
