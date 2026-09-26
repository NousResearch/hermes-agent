"""Cron job scheduling for Hermes Agent: scheduled tasks (cron expressions, intervals, one-shot),
self-scheduled reminders, isolated sessions. The gateway daemon (``hermes gateway [install]``) ticks
the scheduler every 60 seconds; a file lock prevents duplicate execution across processes.
"""

# Cron external worker: run PM's dependency boot (lease + activate the committed generation's
# site-packages) in the package prelude, *before* the first application import below
# (``cron.jobs`` -> ``utils`` -> ``hermes_yaml`` -> ``ruamel``).  The marked restart-safe
# external worker is a fresh store-Python interpreter with no third-party dependencies of its
# own; without this it dies at its first dependency import, and the PM collector can reclaim
# the generation it is importing from in the window between the gateway's exit and that import.
# No-op for the gateway and every unmarked importer (they already booted through
# hermes_bootstrap -> activate_dependencies).  See cron/worker_bootstrap.py.
from cron.worker_bootstrap import worker_bootstrap as _cron_worker_bootstrap

_cron_worker_bootstrap()

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
