"""Person-facing phrases for kanban terminal events.

Both the desktop notifier and the messaging notifier must describe the recorded
event. A timeout is not a failed start, and a stop must not promise a retry
the breaker has already spent.
"""

from __future__ import annotations

from typing import Mapping, Optional


def timeout_promises_retry(payload: Optional[Mapping]) -> bool:
    """True only when the event itself says a retry will happen.

    Missing ``will_retry`` does not promise one. The timeout event used to be
    emitted before the breaker decision, so a bare timeout must not say
    "will retry".
    """
    if not payload:
        return False
    return payload.get("will_retry") is True


def gave_up_reason(payload: Optional[Mapping]) -> str:
    """Why the board stopped, from ``trigger_outcome``, never a fixed spawn story."""
    trigger = str((payload or {}).get("trigger_outcome") or "")
    if trigger == "timed_out":
        return "the time limit was reached"
    if trigger == "spawn_failed":
        return "it failed to start"
    if trigger == "crashed":
        return "the worker stopped unexpectedly"
    if trigger:
        return f"the run ended ({trigger})"
    return "it could not continue"


def gave_up_count(payload: Optional[Mapping]) -> str:
    """Failure count without calling one failure "repeated"."""
    raw = (payload or {}).get("failures")
    if raw is None:
        return ""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    noun = "failure" if n == 1 else "failures"
    return f" after {n} {noun}"
