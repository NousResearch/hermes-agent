"""Failure classification for a cron run that never reached the model.

The success path in ``cron.scheduler.run_job`` consults this guard before
building the success tuple, so a zero-inference run lands on the existing
except handler and gets a proper failure tuple and ``last_status``
(#100180) — the inverse of #70427, where empty *successful* runs were
mis-recorded as failures.
"""

from __future__ import annotations

CRON_FAILURE_MARKER = "[CRON_FAILURE]"


def cron_failure_marker_error(text: str) -> str | None:
    """Return failure evidence when an agent response declares a cron failure.

    Only the exact, standalone first line is control text. The caller keeps the complete response
    in the saved run output while routing this evidence through normal failure bookkeeping.
    """
    lines = (text or "").splitlines()
    if not lines or lines[0].rstrip() != CRON_FAILURE_MARKER:
        return None
    evidence = "\n".join(lines[1:]).strip()
    return evidence or "Cron agent reported failure."


def zero_inference_failure_reason(result: dict) -> str:
    """Return a failure reason when a cron run never reached the model.

    A run that made ZERO inference calls did none of its advertised work:
    the transcript stops before the first tool result (or before any tool
    call at all) and yet lands on the success path. Recording that as
    ``ok`` is a false positive that defeats monitoring — an operator
    watching ``last_status`` sees green while scheduled maintenance
    silently never ran (#100180).

    Returns ``""`` when the run is fine. ``api_calls`` is authoritative
    and cheap: the conversation loop sets it to the provider round-trip
    count on every return path (agent/conversation_loop.py), so an
    explicit 0 means the model was never reached. A missing key (older
    result shapes, test doubles) is NOT treated as zero — the guard only
    fires on an explicit 0, so it can never fail a run whose call count
    simply wasn't reported.
    """
    api_calls = result.get("api_calls")
    if isinstance(api_calls, bool) or not isinstance(api_calls, int):
        return ""
    if api_calls > 0:
        return ""
    return (
        "cron run made zero inference calls (api_calls=0) — the run was "
        "interrupted before reaching the model, so none of the job's work "
        "was performed; refusing to record it as a successful run"
    )
