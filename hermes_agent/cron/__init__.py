"""Cron job scheduling for Hermes Agent: scheduled tasks (cron expressions, intervals, one-shot),
self-scheduled reminders, isolated sessions. The gateway daemon (``hermes gateway [install]``) ticks
the scheduler every 60 seconds; a file lock prevents duplicate execution across processes.
"""

# The restart-safe external worker runs as ``-m hermes_agent.cron.scheduler``, which executes this package
# first: boot PM dependencies before ``hermes_agent.cron.jobs`` reaches a third-party import. A no-op
# unless ``_launch_external_cron_worker`` marked this process. See hermes_agent/cron/worker_bootstrap.py.
from hermes_agent.cron.worker_bootstrap import worker_bootstrap as _boot_external_worker

_boot_external_worker()

from hermes_agent.cron.jobs import (
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
from hermes_agent.cron.scheduler import tick

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
