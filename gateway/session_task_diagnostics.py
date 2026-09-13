"""Reporting only for completed authority tasks; never repair or replay work."""
import json
import logging


logger = logging.getLogger(__name__)


def drain_task_reporter(*, profile_id: str, session_id: str, epoch: int):
    # Capture owner identity now, not from mutable authority/route state later.
    owner = json.dumps({"profile_id": profile_id, "session_id": session_id, "epoch": epoch},
                       ensure_ascii=True)
    reported = False

    def completed(task):
        nonlocal reported
        if reported or not task.done():
            return
        reported = True
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            # Exception messages, chained tracebacks, task reprs and results can
            # contain user data. Only the class and fixed code location are safe.
            try:
                logger.error("Session drain task failed: owner=%s error_type=%s "
                             "callsite=SessionAuthority._drain", owner, type(error).__name__)
            except Exception:
                # A failing sink must not send the Future (and its exception
                # repr) to asyncio's callback-error fallback. No retry here.
                return

    return completed
