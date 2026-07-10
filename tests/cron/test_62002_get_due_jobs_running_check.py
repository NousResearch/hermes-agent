"""
Regression test for issue #62002 - get_due_jobs deletes a live
one-shot whose run is still alive in the same process.

The bug: get_due_jobs()'s one-shot stale-entry recovery (#38758)
deletes a job when its run_claim TTL expires, even if the run is
very much still alive in the same process. A network-stalled run
(or a laptop that slept mid-run) can outlive the 1800s TTL.

The fix: before the stale-entry removal, consult
cron.scheduler.get_running_job_ids(). If the job is in the
running set, skip the removal.

This is a static-source tripwire - it verifies that the fix is
present in the source code by checking that the get_running_job_ids
check is in place before the stale-entry removal log message.
"""

import re
from pathlib import Path


def test_static_check_consults_get_running_job_ids():
    """Static-source tripwire."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    jobs_py = (worktree / "cron" / "jobs.py").read_text()

    # Find the stale-entry removal LOG MESSAGE
    m = re.search(r"one-shot dispatch limit reached", jobs_py)
    assert m, "stale-entry removal site not found"
    # Look 1500 chars BEFORE the log message: the running-check
    # must be present in the if-block immediately above the log.
    site_before = jobs_py[max(0, m.start() - 1500):m.start()]
    assert "get_running_job_ids" in site_before, (
        "#62002 regression: get_due_jobs does not consult "
        "get_running_job_ids before removing stale due entries. "
        "A live run in the same process can have its job deleted "
        "underneath it when the run_claim TTL expires."
    )
