"""Per-step readiness confirmation for guarded desktop runs — RFC #112639.

An admitted guarded run executes its input calls back-to-back (no per-step
capture, no model round trip). After each step whose verdict allowed
continuation, this module asks the driver whether the target is still in a
sane state via ``backend.verify_readiness`` — bounded, screenshot-free, and
far cheaper than capture + inference.

Stop rules: ``unsatisfied`` or ``error`` stops the run with the evidence
attached. ``unknown`` (the driver cannot tell) degrades to the verdict-only
behavior from ``agent.guarded_desktop_runs``. When no check applies — no
sticky target, or a backend without the readiness entry point — the run also
proceeds on verdicts alone. Fail open, never fail closed, on missing signal.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from agent.guarded_desktop_runs import find_guarded_desktop_runs

logger = logging.getLogger(__name__)

# Bounded wait for the driver's postcondition check per run step: long enough
# for a window to settle after input, short enough to stay cheaper than a
# capture + model round trip.
GUARDED_RUN_READINESS_TIMEOUT_MS = 2000

# The only postcondition that holds for every input action: the target window
# still exists. Action-specific predicates are a later slice.
_READINESS_EXPECT = [{"window": {"exists": True}}]


def backend_for_session(session_id: str):
    """The computer_use backend for this session (the one dispatch uses), or None."""
    from tools.computer_use.tool import _get_backend

    try:
        return _get_backend(session_id=session_id or "")
    except Exception:
        logger.warning("guarded run: computer_use backend unavailable", exc_info=True)
        return None


def _run_target_of(backend: Any) -> Optional[Tuple[int, int]]:
    # Sticky target set by capture()/focus_app(); _last_target survives a
    # target-clearing capture so a mid-run check still has an identity.
    pid = getattr(backend, "_active_pid", None)
    window_id = getattr(backend, "_active_window_id", None)
    if pid is None or window_id is None:
        last = getattr(backend, "_last_target", None) or {}
        pid, window_id = last.get("pid"), last.get("window_id")
    if pid is None or window_id is None:
        return None
    return int(pid), int(window_id)


def confirm_run_step(backend: Any, *, timeout_ms: int = GUARDED_RUN_READINESS_TIMEOUT_MS):
    """Bounded readiness confirmation after one admitted run step.

    Returns the ``ReadinessResult``, or None when no check applies (no sticky
    target, or the backend has no readiness entry point) — the caller then
    degrades to verdict-only behavior.
    """
    verify = getattr(backend, "verify_readiness", None)
    if not callable(verify):
        return None
    target = _run_target_of(backend)
    if target is None:
        return None
    pid, window_id = target
    return verify(
        pid=pid,
        window_id=window_id,
        expect=list(_READINESS_EXPECT),
        timeout_ms=timeout_ms,
        include_screenshot=False,
    )


def run_step_needs_confirmation(tool_calls: List[Any], just_executed_index: int) -> bool:
    """True when the call is a mid-run input a readiness check could confirm.

    Pure admission check — no backend touch, so the executor can decide whether
    a backend lookup is even warranted (the lookup would otherwise spawn a
    driver daemon for sessions that never use computer_use).
    """
    return any(
        start <= just_executed_index < end - 1
        for start, end in find_guarded_desktop_runs(tool_calls)
    )


def guarded_run_readiness_stop(
    tool_calls: List[Any],
    just_executed_index: int,
    backend: Any,
    *,
    timeout_ms: int = GUARDED_RUN_READINESS_TIMEOUT_MS,
) -> Optional[Tuple[int, str]]:
    """``(run_end, reason)`` when the readiness check stops the run after this step.

    ``just_executed_index`` is the 0-based index of the call that just ran.
    Returns None when the call is not a mid-run input, when no check applies,
    or when the driver reports ``satisfied``/``unknown``.
    """
    for start, end in find_guarded_desktop_runs(tool_calls):
        if not start <= just_executed_index < end - 1:
            continue
        result = confirm_run_step(backend, timeout_ms=timeout_ms)
        if result is None:
            return None  # no check applies: verdict-only behavior
        if result.status in ("satisfied", "unknown"):
            return None
        reason = (
            f"readiness check {result.status}: {result.detail} "
            f"({result.duration_ms:.0f}ms)"
        )
        return end, reason
    return None
