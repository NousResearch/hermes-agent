"""Guarded ordered desktop runs — RFC #112639 revised critical path.

Hermes already executes several tool calls from one model response as ordered
execution segments, and ``computer_use`` calls are sequential barriers, so a
short run of desktop inputs can share one model decision. Two gaps made that
unsafe, fixed here:

1. ``_deduplicate_tool_calls`` collapsed intentional repeats: two Tab presses
   are one ``(name, args)`` key. Repeats survive inside an admitted run.
2. Sequential execution had no GUI stop rule. An admitted run now stops at the
   first step whose effect is not confirmed, and every unstarted call gets an
   explicit skipped result via the existing ``_append_skipped_tool_results``
   machinery.

Admission is decided BEFORE dedup runs, so normalization cannot eat the
repeats. Each operation keeps its existing permission and dispatch path —
this is orchestration, not an authorization bypass.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Proposed safety setting, not a proven optimum: a bounded contiguous run of
# input operations against one known target.
GUARDED_RUN_MAX_OPS = 4

# computer_use actions that are GUI inputs (not observation, not waiting).
_GUARDED_INPUT_ACTIONS = frozenset({
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "drag",
    "scroll",
    "type",
    "key",
    "set_value",
})

# computer_use action results always carry verdict.decision; the regex covers
# results with text appended after the JSON (guardrail guidance).
_VERDICT_DECISION_RE = re.compile(r'"verdict"\s*:\s*\{[^}]*?"decision"\s*:\s*"([A-Za-z_]+)"')

# Executor-side markers for synthesized non-delivery results (private classes
# in agent.tool_executor; matched by name to avoid an import cycle).
_NON_DELIVERY_RESULT_NAMES = frozenset({"_ToolTimeoutResult", "_ToolCancelledResult"})


def _tool_name(tool_call: Any) -> Optional[str]:
    return getattr(getattr(tool_call, "function", None), "name", None)


def _tool_args(tool_call: Any) -> Optional[dict]:
    raw = getattr(getattr(tool_call, "function", None), "arguments", None)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_run_input(tool_call: Any) -> bool:
    if _tool_name(tool_call) != "computer_use":
        return False
    args = _tool_args(tool_call)
    return args is not None and args.get("action") in _GUARDED_INPUT_ACTIONS


def _run_target(tool_call: Any) -> str:
    # One known target: the `app` argument. Omitted means the frontmost
    # window — still a single identity for the run's duration.
    app = (_tool_args(tool_call) or {}).get("app")
    return "" if app is None else str(app)


def find_guarded_desktop_runs(tool_calls: List[Any]) -> List[Tuple[int, int]]:
    """Contiguous ``(start, end)`` runs of computer_use inputs sharing one target.

    Each run is 2..GUARDED_RUN_MAX_OPS calls long. Only the first call may
    carry ``element``: a SOM index is bound to the snapshot taken before the
    run, and a later UI transition can invalidate it, so a second element use
    ends the run before that call.
    """
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(tool_calls)
    while i < n:
        if not _is_run_input(tool_calls[i]):
            i += 1
            continue
        target = _run_target(tool_calls[i])
        j = i
        while (
            j < n
            and (j - i) < GUARDED_RUN_MAX_OPS
            and _is_run_input(tool_calls[j])
            and _run_target(tool_calls[j]) == target
        ):
            if j > i and (_tool_args(tool_calls[j]) or {}).get("element") is not None:
                break
            j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = j if j > i else i + 1
    return runs


def guarded_run_preserve_indices(
    tool_calls: List[Any], runs: Optional[List[Tuple[int, int]]] = None
) -> frozenset:
    """Call indices exempt from ``_deduplicate_tool_calls``.

    Intentional repeats (two Tabs) survive only inside an admitted run;
    normal dedup applies everywhere else.
    """
    if runs is None:
        runs = find_guarded_desktop_runs(tool_calls)
    return frozenset(idx for start, end in runs for idx in range(start, end))


def _verdict_decision(result: Any) -> Optional[str]:
    if not isinstance(result, str):
        return None
    try:
        verdict = json.loads(result).get("verdict")
        if isinstance(verdict, dict) and verdict.get("decision"):
            return str(verdict["decision"])
    except (ValueError, TypeError, AttributeError):
        pass
    match = _VERDICT_DECISION_RE.search(result)
    return match.group(1) if match else None


def guarded_run_continues(managed: Any) -> Tuple[bool, str]:
    """Whether the run may proceed after this result.

    Continues only on a confirmed effect (verdict decision ``done``). A blocked
    call (approval denied / policy block), a timeout or cancellation, a missing
    verdict, or any non-``done`` decision stops the run: the next step's
    precondition — a known GUI state — is not established.
    """
    if getattr(managed, "blocked", False):
        return False, "the previous action was blocked (approval denied or policy block)"
    result = getattr(managed, "result", None)
    if type(result).__name__ in _NON_DELIVERY_RESULT_NAMES:
        return False, "the previous action timed out or was cancelled"
    decision = _verdict_decision(result)
    if decision == "done":
        return True, ""
    if decision is None:
        return False, "the previous action returned no usable verdict"
    return False, f"the previous action's effect was not confirmed (verdict: {decision})"


def guarded_run_stop(
    tool_calls: List[Any], just_executed_index: int, managed: Any
) -> Optional[Tuple[int, str]]:
    """``(run_end, reason)`` when the admitted run must stop after this result.

    ``just_executed_index`` is the 0-based index of the call that just ran.
    Returns None when the call is not inside an admitted run, when it is the
    run's last call, or when the verdict allows continuation.
    """
    for start, end in find_guarded_desktop_runs(tool_calls):
        if start <= just_executed_index < end - 1:
            ok, reason = guarded_run_continues(managed)
            return None if ok else (end, reason)
    return None
