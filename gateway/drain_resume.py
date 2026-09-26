"""Shutdown-time disposition of boot auto-resume turns.

Incident 2026-09-20 (Apollo). Boot marked 4 in-flight sessions resumable at
09:55:09, but Discord could not connect until 10:02:26, so
``_schedule_resume_pending_sessions`` only ran at 10:02:45. Each scheduled
resume claimed its ``_running_agents`` slot with ``_AGENT_PENDING_SENTINEL``
and dispatched an internal event within ~0.5s — but the turn BODY
(``run_sync``, which executes off-loop on the runner's shared
``ThreadPoolExecutor(max_workers=10)``) did not begin until 10:12:36-46, ~590s
later and 30s INTO a shutdown drain that started at 10:12:05.

Two things went wrong and both are fixed here:

1. The drain waited on them. ``_drain_active_agents`` gates on
   ``len(self._running_agents)``, which COUNTS sentinels, so the drain burned
   its whole 50s cap on work that existed only because of pending resumes.
   Meanwhile the summary line reports ``active_at_start`` from
   ``_snapshot_running_agents()``, which EXCLUDES sentinels — hence the
   contradictory ``active_at_start=0, active_now=5``.

2. They were then interrupted at the cap, the ``.clean_shutdown`` marker was
   skipped, and the next boot resumed all five AGAIN as
   ``reason=shutdown_timeout kind=sibling``. Every affected channel ran the
   resume machinery twice.

The fix: at shutdown start, a boot resume that was scheduled but whose turn
never actually began is CANCELLED and the session RE-MARKED resumable with its
ORIGINAL boot reason, so the next boot resumes it exactly once. A resume whose
turn already started is drained like any other work.

Kept as a separate module (rather than inline in the runner) so the
disposition predicate is unit-testable without constructing a runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Disposition verdicts for one tracked boot-resume registration at shutdown.
DISPOSITION_CANCEL = "cancel_and_remark"
DISPOSITION_DRAIN = "drain_normally"
DISPOSITION_GONE = "already_finished"


@dataclass(frozen=True)
class BootResumeRegistration:
    """One scheduled boot auto-resume, tracked from schedule time to start.

    ``session_key`` is the routing key; ``resume_reason`` is the reason the
    boot path read off the durable mark (``restart_interrupted``,
    ``shutdown_timeout``, …) and is what must be RE-marked if the resume is
    cancelled — re-marking under a fresh/derived reason is what turned a
    ``restart_interrupted`` resume into a second-generation
    ``shutdown_timeout`` one in the incident.
    """

    session_key: str
    resume_reason: Optional[str]
    scheduled_at: float


def classify_boot_resume_at_shutdown(
    registration: BootResumeRegistration,
    *,
    slot_value: Any,
    pending_sentinel: Any,
    task_done: bool,
) -> str:
    """Decide what shutdown should do with one scheduled boot resume.

    * The session slot is GONE (or the task already finished) -> nothing to do;
      the turn ran and released, or was never claimed.
    * The slot still holds the PENDING SENTINEL -> the turn body never began.
      Cancel it and re-mark the session resumable, so the drain does not wait
      on phantom work and the next boot recovers it exactly once.
    * The slot holds a REAL agent -> the turn genuinely started. Drain it like
      any other in-flight work; the normal shutdown-mark path owns it.
    """
    if slot_value is None:
        return DISPOSITION_GONE
    if slot_value is pending_sentinel:
        # A finished task that never promoted its sentinel is a leaked slot,
        # not a live resume — releasing it is still the right move, and the
        # re-mark is harmless (the durable mark is idempotent per session).
        return DISPOSITION_CANCEL
    if task_done:
        return DISPOSITION_GONE
    return DISPOSITION_DRAIN


def drain_admission_reason(
    *,
    is_boot_resume: bool,
    is_internal: bool,
) -> str:
    """Classify WHY a turn started while shutdown was already in progress.

    This is the string carried by ``PHASE=drain_admission``. The incident was
    invisible precisely because nothing named the class of work being admitted:
    five turns appeared in ``active_now`` with no line saying where they came
    from. ``boot_resume`` is the value that would have named it.
    """
    if is_boot_resume:
        return "boot_resume"
    if is_internal:
        return "internal_event"
    return "user_message"
