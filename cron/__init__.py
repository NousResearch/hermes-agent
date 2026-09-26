"""Cron job scheduling for Hermes Agent: scheduled tasks (cron expressions, intervals, one-shot),
self-scheduled reminders, isolated sessions. The gateway daemon (``hermes gateway [install]``) ticks
the scheduler every 60 seconds; a file lock prevents duplicate execution across processes.
"""

import sys

# ``python -m cron.scheduler --external-worker-file ...`` imports this package
# before loading the scheduler module, so activate PM dependencies here before
# cron.jobs and its dependency-bearing import graph.
if "--external-worker-file" in sys.argv:
    import hermes_bootstrap  # noqa: F401

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
