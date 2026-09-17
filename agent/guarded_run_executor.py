"""Per-step readiness confirmation for guarded desktop runs — RFC #112639.

An admitted guarded run executes its input calls back-to-back (no per-step
capture, no model round trip). After each step whose verdict allowed
continuation, this module asks the driver whether the step's postcondition
holds via ``backend.verify_readiness`` — bounded, screenshot-free, and far
cheaper than capture + inference.

The postcondition is per-action (``tools.computer_use.readiness_predicates``):
a ``type`` step confirms the field holds the typed text, a click with known
target info confirms the element still exists, and anything without a
driver-expressible postcondition falls back to ``window.exists``. The caller
threads in the SOM labels from the capture snapshot the run's ``element``
indices are bound to; without them every step degrades to the fallback.
Nothing here invents target info.

Stop rules: ``unsatisfied`` or ``error`` stops the run with the evidence
attached. ``unknown`` (the driver cannot tell) degrades to the verdict-only
behavior from ``agent.guarded_desktop_runs``. When no check applies — no
sticky target, or a backend without the readiness entry point — the run also
proceeds on verdicts alone. Fail open, never fail closed, on missing signal.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.guarded_desktop_runs import _tool_args, find_guarded_desktop_runs
from tools.computer_use import readiness_predicates

logger = logging.getLogger(__name__)

# Bounded wait for the driver's postcondition check per run step: long enough
# for a window to settle after input, short enough to stay cheaper than a
# capture + model round trip.
GUARDED_RUN_READINESS_TIMEOUT_MS = 2000

# The only postcondition that holds for every input action; also the fallback
# when a step's own args cannot build a sharper predicate.
_READINESS_FALLBACK_EXPECT = [{"window": {"exists": True}}]

# SOM index (the ``element`` arg, bound to the snapshot taken before the run)
# -> (label, role) of the target element from that snapshot's capture.
# Threaded in by the caller; absent means every element-keyed expectation
# degrades to the window.exists fallback.
SomLabels = Dict[int, Tuple[Optional[str], Optional[str]]]


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


def _som_label_role(
    args: Dict[str, Any], som_labels: Optional[SomLabels]
) -> Tuple[Optional[str], Optional[str]]:
    idx = args.get("element")
    if not isinstance(idx, int) or not som_labels:
        return None, None
    found = som_labels.get(idx)
    return found if found is not None else (None, None)


def _kwargs_type(args: Dict[str, Any], som_labels: Optional[SomLabels]) -> Dict[str, Any]:
    label, role = _som_label_role(args, som_labels)
    return {"text": args.get("text") or "", "field_label": label, "field_role": role}


def _kwargs_set_value(args: Dict[str, Any], som_labels: Optional[SomLabels]) -> Dict[str, Any]:
    label, role = _som_label_role(args, som_labels)
    return {"value": args.get("value") or "", "field_label": label, "field_role": role}


def _kwargs_click(args: Dict[str, Any], som_labels: Optional[SomLabels]) -> Dict[str, Any]:
    label, role = _som_label_role(args, som_labels)
    return {"expect_label": label, "expect_role": role}


def _kwargs_key(args: Dict[str, Any], som_labels: Optional[SomLabels]) -> Dict[str, Any]:
    label, role = _som_label_role(args, som_labels)
    return {"expect_label": label, "expect_role": role}


def _kwargs_fallback(args: Dict[str, Any], som_labels: Optional[SomLabels]) -> Dict[str, Any]:
    return {}


# Table-driven: no elif ladder on the action name (repo shape rule).
_EXPECT_KWARG_BUILDERS: Dict[
    str, Callable[[Dict[str, Any], Optional[SomLabels]], Dict[str, Any]]
] = {
    "type": _kwargs_type,
    "set_value": _kwargs_set_value,
    "click": _kwargs_click,
    "double_click": _kwargs_click,
    "right_click": _kwargs_click,
    "middle_click": _kwargs_click,
    "key": _kwargs_key,
    "drag": _kwargs_fallback,
    "scroll": _kwargs_fallback,
    "focus_app": _kwargs_fallback,
    "wait": _kwargs_fallback,
    "capture": _kwargs_fallback,
}


def _expect_for_run_step(
    tool_call: Any, som_labels: Optional[SomLabels]
) -> List[Dict[str, Any]]:
    """The ``verify_state`` expect list for one executed run step.

    Sharp per-action postconditions when the call's own args supply them
    (plus SOM labels the caller threaded in); the ``window.exists`` fallback
    otherwise. Never raises: a builder rejection (e.g. empty text) degrades to
    the fallback, fail open.
    """
    args = _tool_args(tool_call) or {}
    action = args.get("action")
    builder = _EXPECT_KWARG_BUILDERS.get(action, _kwargs_fallback) if isinstance(
        action, str
    ) else _kwargs_fallback
    try:
        return readiness_predicates.predicates_for_action(
            action if isinstance(action, str) else "", **builder(args, som_labels)
        )
    except (TypeError, ValueError):
        return list(_READINESS_FALLBACK_EXPECT)


def confirm_run_step(
    backend: Any,
    *,
    timeout_ms: int = GUARDED_RUN_READINESS_TIMEOUT_MS,
    tool_call: Any = None,
    som_labels: Optional[SomLabels] = None,
):
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
        expect=_expect_for_run_step(tool_call, som_labels),
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
    som_labels: Optional[SomLabels] = None,
) -> Optional[Tuple[int, str]]:
    """``(run_end, reason)`` when the readiness check stops the run after this step.

    ``just_executed_index`` is the 0-based index of the call that just ran.
    ``som_labels`` threads in the target labels from the capture snapshot the
    run's ``element`` indices are bound to. Returns None when the call is not a
    mid-run input, when no check applies, or when the driver reports
    ``satisfied``/``unknown``.
    """
    for start, end in find_guarded_desktop_runs(tool_calls):
        if not start <= just_executed_index < end - 1:
            continue
        result = confirm_run_step(
            backend,
            timeout_ms=timeout_ms,
            tool_call=tool_calls[just_executed_index],
            som_labels=som_labels,
        )
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
