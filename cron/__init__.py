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
from cron.scheduler import tick

# The completion tail (mark_job_run -> quota_hold / unreachable_retry / occurrences, finish_execution
# -> incidents / sqlite_util) runs minutes after a job started. Its late imports are the one seam a
# process spanning an in-place `hermes update` crosses on MIXED code: the new file on disk against
# this process's cached siblings (`cannot import name 'safe_strftime' from 'hermes_time'`), after
# the output was written and before it was delivered or recorded. Load them with the package so
# the whole bookkeeping path is pinned to the generation this process booted on.
from cron import (  # noqa: E402, F401
    incidents, lifecycle_guard, notepad, occurrences, quota_hold, scheduler_failure_copy,
    unreachable_retry,
)
from gateway import response_filters  # noqa: E402, F401
from hermes_cli import sqlite_util  # noqa: E402, F401

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
