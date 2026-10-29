"""Cron's turn in the gateway's memory-aware admission (``gateway/admission_memory.py``).

A cron run is background work: it waits while memory is merely low and behind any person
waiting for a turn, so a scheduled job never takes the last headroom from someone typing. The
wait is bounded by ``gateway.memory_guard.max_wait_seconds``; on expiry the run is recorded as
failed with a plain reason instead of starting into a host that is about to OOM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def hold_for_memory(job: dict, job_id: str, job_name: str, cancel_event: Any) -> Optional[tuple]:
    """None to proceed; else ``run_job``'s failure tuple (timed out or cancelled while waiting).
    Runs before the job's pre-run script and before a script-only (``no_agent``) job, so no part
    of a scheduled run starts into a host that is out of memory."""
    from gateway.admission_memory import BACKGROUND, get_gate, wait_blocking

    verdict = wait_blocking(BACKGROUND, still_wanted=lambda: cancel_event is None or not cancel_event.is_set())
    if verdict == "admitted":
        return None
    waited = int(get_gate().settings.max_wait_seconds)
    error = ("cancelled while waiting for free memory" if verdict == "abandoned" else
             f"skipped: host memory stayed low for {waited}s (gateway.memory_guard); the next run tries again")
    logger.warning("Job '%s' (ID: %s) %s", job_name, job_id, error)
    from cron.scheduler import _run_doc_header

    header = _run_doc_header(job, f"{job_name} (FAILED)", job_id, str(job.get("prompt") or ""))
    return False, header + f"\n{error}\n", "", error
