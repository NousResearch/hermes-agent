"""Cron job scheduling for Hermes Agent: scheduled tasks (cron expressions, intervals, one-shot),
self-scheduled reminders, isolated sessions. The gateway daemon (``hermes gateway [install]``) ticks
the scheduler every 60 seconds; a file lock prevents duplicate execution across processes.
"""

# The restart-safe external worker runs as ``-m cron.scheduler``, which executes this package
# first: boot PM dependencies before ``cron.jobs`` reaches a third-party import. A no-op
# unless ``_launch_external_cron_worker`` marked this process. See cron/worker_bootstrap.py.
from cron.worker_bootstrap import worker_bootstrap as _boot_external_worker

_boot_external_worker()

from cron.jobs import (  # noqa: E402
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
from cron.scheduler import tick  # noqa: E402

__all__ = [
    "create_job",
    "get_job",
    "list_jobs",
    "remove_job",
    "update_job",
    "pause_job",
    "resume_job",
    "trigger_job",
    "rearm_oneshot",
    "tick",
    "JOBS_FILE",
]
