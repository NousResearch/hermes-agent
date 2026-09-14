"""Collecting an objective back from the runtime.

Submission is one direction; this is the other. An objective's state is not stored by NOVA
— it is derived, every time, from the work items the runtime currently holds. That is
deliberate: a NOVA-side copy of "which steps are done" would be a second source of truth
that drifts the moment anyone touches the board through the runtime's own CLI or dashboard,
and the board is a shared surface by design.

So the mapping is one-way and cheap: step -> idempotency key -> the runtime's task. What
NOVA contributes is the objective-level reading the runtime has no concept of — whether the
objective is blocked, what is blocking it, and which step a reviewer should look at first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from nova.runtime.base import AgentRuntime, TaskView
from nova.spec.objective import ObjectiveSpec, PlanStep
from nova.supervisor.submit import work_key

#: Objective-level states, derived from its steps. Deliberately fewer than the runtime's
#: task states: an objective is a business question, and "which of six task statuses" is
#: not the question anyone asks about a month-end close.
NOT_STARTED = "not_started"
RUNNING = "running"
BLOCKED = "blocked"
NEEDS_REVIEW = "needs_review"
DONE = "done"

OBJECTIVE_STATES = (NOT_STARTED, RUNNING, BLOCKED, NEEDS_REVIEW, DONE)


@dataclass(frozen=True)
class StepReport:
    """One step, as the runtime currently holds it."""

    step_id: str
    title: str
    assignee: str
    key: str
    submitted: bool
    task_id: str = ""
    state: str = ""
    runtime_status: str = ""
    consecutive_failures: int = 0
    last_error: str = ""
    depends_on: tuple[str, ...] = ()

    @property
    def needs_attention(self) -> bool:
        return self.state == "blocked" or self.consecutive_failures > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "assignee": self.assignee,
            "key": self.key,
            "submitted": self.submitted,
            "task_id": self.task_id,
            "state": self.state,
            "runtime_status": self.runtime_status,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "needs_attention": self.needs_attention,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class ObjectiveReport:
    """One objective's current state, derived from the runtime."""

    objective_id: str
    title: str
    owner: str
    state: str
    steps: tuple[StepReport, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def submitted_steps(self) -> tuple[StepReport, ...]:
        return tuple(step for step in self.steps if step.submitted)

    @property
    def blocking(self) -> tuple[StepReport, ...]:
        """The steps a reviewer should look at first, worst first."""
        return tuple(
            sorted(
                (step for step in self.submitted_steps if step.needs_attention),
                key=lambda step: (-step.consecutive_failures, step.step_id),
            )
        )

    @property
    def progress(self) -> tuple[int, int]:
        """``(done, total)`` across every declared step, submitted or not."""
        return sum(1 for step in self.steps if step.state == "done"), len(self.steps)

    def summary(self) -> str:
        done, total = self.progress
        line = f"{self.objective_id}: {self.state}  ({done}/{total} steps done)"
        if self.blocking:
            first = self.blocking[0]
            line += f"  — blocked on {first.step_id} ({first.assignee})"
        return line

    def to_dict(self) -> dict[str, Any]:
        done, total = self.progress
        return {
            "objective_id": self.objective_id,
            "title": self.title,
            "owner": self.owner,
            "state": self.state,
            "done": done,
            "total": total,
            "steps": [step.to_dict() for step in self.steps],
            "blocking": [step.step_id for step in self.blocking],
            "warnings": list(self.warnings),
        }


def collect(
    objective: ObjectiveSpec,
    runtime: AgentRuntime,
    *,
    tasks: Optional[Sequence[TaskView]] = None,
) -> ObjectiveReport:
    """Read ``objective``'s current state out of the runtime.

    ``tasks`` lets a caller reporting on several objectives read the board once rather than
    once per objective — the control plane does exactly that.
    """
    views = list(tasks) if tasks is not None else runtime.list_tasks(limit=1000)
    by_key = _index_by_key(views)

    steps = tuple(_step_report(objective, step, by_key) for step in objective.steps)
    warnings: list[str] = []

    missing = [step.step_id for step in steps if not step.submitted]
    if missing and len(missing) != len(steps):
        warnings.append(
            f"step(s) {', '.join(missing)} have no work item on the board — this objective "
            "was submitted before they were declared. Re-run submit to add them"
        )

    return ObjectiveReport(
        objective_id=objective.id,
        title=objective.title,
        owner=objective.owner,
        state=derive_state(steps),
        steps=steps,
        warnings=tuple(warnings),
    )


def _index_by_key(views: Sequence[TaskView]) -> dict[str, TaskView]:
    """Work items by the idempotency key NOVA gave them.

    The key is carried in ``TaskView.detail`` because the contract has no field for "the
    caller's own identifier" and inventing one would put a NOVA concept into an interface
    every adapter has to implement. An adapter that cannot surface it yields nothing here,
    and the report says the steps are unsubmitted rather than guessing by title — matching
    on a human-readable title is exactly how two objectives with a "Review" step end up
    reporting each other's progress.
    """
    indexed: dict[str, TaskView] = {}
    for view in views:
        key = str((view.detail or {}).get("idempotency_key") or "")
        if key and key not in indexed:
            indexed[key] = view
    return indexed


def _step_report(
    objective: ObjectiveSpec, step: PlanStep, by_key: dict[str, TaskView]
) -> StepReport:
    key = work_key(objective.id, step.id)
    view = by_key.get(key)
    if view is None:
        return StepReport(
            step_id=step.id,
            title=step.title,
            assignee=step.assignee,
            key=key,
            submitted=False,
            depends_on=step.depends_on,
        )
    return StepReport(
        step_id=step.id,
        title=step.title,
        assignee=step.assignee,
        key=key,
        submitted=True,
        task_id=view.task_id,
        state=view.state,
        runtime_status=view.runtime_status,
        consecutive_failures=view.consecutive_failures,
        last_error=view.last_error,
        depends_on=step.depends_on,
    )


def derive_state(steps: Sequence[StepReport]) -> str:
    """One objective state from its steps.

    Ordered by what a reader needs to know first, not by what is most common. An objective
    with one blocked step and five finished ones is blocked — reporting it as "running"
    because most of it is fine is how a stuck process goes unnoticed for a week.
    """
    submitted = [step for step in steps if step.submitted]
    if not submitted:
        return NOT_STARTED
    if any(step.needs_attention for step in submitted):
        return BLOCKED
    if any(step.state == "review" for step in submitted):
        return NEEDS_REVIEW
    if len(submitted) == len(steps) and all(step.state == "done" for step in submitted):
        return DONE
    return RUNNING
