"""Step-transition validation.

Defines which step types may legally follow which. Illegal transitions are
rejected before they land in ``AgentMemory`` — the loop converts a rejection
into a ``TransitionRejected`` failure rather than appending the step.

The state machine is intentionally tight. Adding a new step type requires
updating ``LEGAL_TRANSITIONS`` explicitly so the graph remains auditable.
"""

from __future__ import annotations

from .steps import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep, TaskStep

# Maps "previous step type" → set of legal "next step types".
# ``None`` represents the initial state (no prior step).
LEGAL_TRANSITIONS: dict[type | None, frozenset[type]] = {
    None: frozenset({TaskStep}),
    TaskStep: frozenset({ActionStep, PlanningStep, FinalAnswerStep}),
    PlanningStep: frozenset({ActionStep, PlanningStep, FinalAnswerStep}),
    ActionStep: frozenset({ActionStep, PlanningStep, FinalAnswerStep}),
    FinalAnswerStep: frozenset(),  # terminal
}


class TransitionGuard:
    """Validates ``previous → next`` against ``LEGAL_TRANSITIONS``.

    Stateless. Construct once, call ``check()`` repeatedly. The guard never
    raises — it returns a ``(ok, reason)`` pair and lets the loop decide
    how to handle a rejection.
    """

    def check(self, previous: MemoryStep | None, next_step: MemoryStep) -> tuple[bool, str]:
        prev_key: type | None = type(previous) if previous is not None else None
        allowed = LEGAL_TRANSITIONS.get(prev_key)
        if allowed is None:
            return False, f"unknown previous step type: {prev_key}"
        if type(next_step) not in allowed:
            prev_name = prev_key.__name__ if prev_key else "<start>"
            return False, f"illegal transition {prev_name} -> {type(next_step).__name__}"
        return True, ""
