"""Failure classification and recovery strategy selection.

Never blindly retry: classify first, then pick a strategy from evidence.
The identical-failure rule forces a strategy change when the same action,
failure, and hypothesis repeat — deterministic loops are refused.
"""

from __future__ import annotations

from typing import Dict, Mapping

_TRANSIENT_SIGNALS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "503",
    "temporar",
    "unavailab",
    "network",
    "econn",
    "try again",
    "connection reset",
    "overloaded",
)


class FailureClass:
    TRANSIENT = "TRANSIENT"
    DETERMINISTIC = "DETERMINISTIC"
    UNKNOWN = "UNKNOWN"


class Strategy:
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    RETRIEVE_MORE_CONTEXT = "RETRIEVE_MORE_CONTEXT"
    INSPECT_DEPENDENCY = "INSPECT_DEPENDENCY"
    CHANGE_IMPLEMENTATION = "CHANGE_IMPLEMENTATION"
    ROLLBACK = "ROLLBACK"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


def classify_failure(message: str, *, transient: bool = False) -> str:
    """Classify a failure. Explicit transient flags win; otherwise match
    transport-level signals, else structural (deterministic). Empty text
    without a flag is UNKNOWN rather than a guess."""
    if transient:
        return FailureClass.TRANSIENT
    lowered = (message or "").lower()
    if not lowered.strip():
        return FailureClass.UNKNOWN
    if any(signal in lowered for signal in _TRANSIENT_SIGNALS):
        return FailureClass.TRANSIENT
    return FailureClass.DETERMINISTIC


# First-seen strategy per class (table, not a branch ladder).
_FIRST_SEEN: Dict[str, str] = {
    FailureClass.TRANSIENT: Strategy.RETRY,
    FailureClass.DETERMINISTIC: Strategy.REPLAN,
    FailureClass.UNKNOWN: Strategy.INSPECT_DEPENDENCY,
}

# Forced escalation once the identical triple repeats: never repeat the
# same failing strategy. Counts: 1st repeat retrieves, 2nd changes the
# implementation, further repeats escalate, then stop.
_REPEAT_LADDER = (
    Strategy.RETRIEVE_MORE_CONTEXT,
    Strategy.CHANGE_IMPLEMENTATION,
    Strategy.ESCALATE,
    Strategy.STOP,
)


def decide(
    failure_fp: str,
    hypothesis_fp: str,
    action_fp: str,
    failure_class: str,
    seen: Mapping[str, int],
) -> str:
    repeats = seen.get(failure_fp + "\x00" + hypothesis_fp + "\x00" + action_fp, 0)
    if repeats <= 1:
        return _FIRST_SEEN.get(failure_class, Strategy.ESCALATE)
    rung = min(repeats - 2, len(_REPEAT_LADDER) - 1)
    return _REPEAT_LADDER[rung]


def progress_made(
    *,
    new_evidence: bool = False,
    implementation_progress: bool = False,
    reduced_uncertainty: bool = False,
    verified_knowledge: bool = False,
    successful_verification: bool = False,
    completed_milestone: bool = False,
) -> bool:
    """Progress invariant: at least one must hold per non-terminal iteration."""
    return any((
        new_evidence,
        implementation_progress,
        reduced_uncertainty,
        verified_knowledge,
        successful_verification,
        completed_milestone,
    ))
