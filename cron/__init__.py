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
# Lazy import to avoid the runpy RuntimeWarning when this package is loaded as
# `python -m cron.scheduler`. Symptom: `'cron.scheduler' found in sys.modules
# after import of package 'cron', but prior to execution of 'cron.scheduler'`.
# Root cause: `from cron.scheduler import tick` at package init time causes the
# scheduler module to be imported into sys.modules BEFORE runpy gets to
# "execute" it, which is the standard trigger runpy warns about (it's a CPython
# 3.10+ safe-import warning, not a hard error). The lazy __getattr__ resolves
# `cron.tick` only when something actually asks for it.
#
# Loki Q157 (Oct 16 2026) confirmed the diagnosis. Affected jobs (paper-trader
# intraday-midday/late, paper-trader-morning-research, forge-stack-weekly-rollup)
# were dying with cron status "error" + the warning in stderr; silencing the
# warning alone doesn't fix the SIGTERM-after-adopt root cause, but it stops
# the false-positive stderr noise and makes the next diagnostic cleaner.
def __getattr__(name):
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
