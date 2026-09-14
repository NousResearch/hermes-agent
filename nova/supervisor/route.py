"""Whether a plan may be routed as written — the check that makes delegation a real control.

Pure. No runtime, no filesystem, no clock. Everything here is a function of the objective
and the tenant's declared agents, which is what lets the control plane simulate a routing
decision without submitting anything and lets the CLI explain a refusal before any work
exists.

The rule the whole phase rests on:

    A step may be assigned to the objective's owner, or to an agent the owner declared in
    ``delegation.may_assign_to``. Nothing else.

``may_assign_to`` has been in the spec since Phase 1, validated at bundle load, compiled
into the profile under a key the runtime ignores — and enforced nowhere. The runtime's own
decomposer picks any profile on the host and checks only that it exists. So until this
module ran, a tenant could declare that support may delegate to operations and nothing
prevented support's work landing on finance.

**This check is ``hard_preemptive``**, in the vocabulary of :mod:`nova.policy.limits`, and
it earns that classification for one specific reason: NOVA is the writer. A refusal here
means no task is created, rather than a task created and then reported on. The boundary of
that claim is equally specific and is stated wherever the claim is — NOVA governs the work
NOVA submits, and nothing else on the board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from nova.errors import NovaError
from nova.spec import AgentSpec
from nova.spec.objective import ObjectiveSpec, PlanStep

#: Why a step was refused. Carried as data rather than prose so the control plane can group
#: refusals and a test can assert on the reason rather than on a sentence.
UNKNOWN_OWNER = "unknown_owner"
UNKNOWN_ASSIGNEE = "unknown_assignee"
DISABLED_ASSIGNEE = "disabled_assignee"
NOT_DELEGABLE = "not_delegable"
OWNER_DISABLED = "owner_disabled"

#: Reasons that mean the objective cannot run at all, as opposed to one step being wrong.
OBJECTIVE_LEVEL = frozenset({UNKNOWN_OWNER, OWNER_DISABLED})


class RoutingError(NovaError):
    """A plan that may not be submitted as written."""


@dataclass(frozen=True)
class StepRouting:
    """The decision for one step."""

    step_id: str
    assignee: str
    allowed: bool
    reason: str = ""
    detail: str = ""
    #: True when the runtime's own decomposer will route this step's children, which NOVA
    #: cannot govern. Recorded so nothing downstream presents the step as fully governed.
    delegated_routing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "assignee": self.assignee,
            "allowed": self.allowed,
            "reason": self.reason,
            "detail": self.detail,
            "delegated_routing": self.delegated_routing,
        }


@dataclass(frozen=True)
class RoutingDecision:
    """Whether a whole objective may be submitted, and why not where it may not."""

    objective_id: str
    owner: str
    steps: tuple[StepRouting, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def allowed(self) -> bool:
        return all(step.allowed for step in self.steps) and bool(self.steps)

    @property
    def refusals(self) -> tuple[StepRouting, ...]:
        return tuple(step for step in self.steps if not step.allowed)

    @property
    def ungoverned_steps(self) -> tuple[str, ...]:
        return tuple(step.step_id for step in self.steps if step.delegated_routing)

    def explain(self) -> str:
        """Why this objective was refused, in the order a reader needs it."""
        if self.allowed:
            return f"{self.objective_id}: {len(self.steps)} step(s) route cleanly"
        lines = [f"{self.objective_id}: {len(self.refusals)} step(s) may not be routed"]
        for step in self.refusals:
            lines.append(f"  {step.step_id} -> {step.assignee}: {step.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "owner": self.owner,
            "allowed": self.allowed,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "ungoverned_steps": list(self.ungoverned_steps),
        }


def route_objective(
    objective: ObjectiveSpec,
    agents: Sequence[AgentSpec],
) -> RoutingDecision:
    """Decide whether every step of ``objective`` may be routed as declared.

    Never raises on a bad plan — it returns the decision. Refusing is a normal outcome that
    an operator needs the detail of, and an exception would reduce "these three steps go to
    agents you never authorised" to one line about the first of them.
    """
    by_id = {spec.id: spec for spec in agents}
    owner = by_id.get(objective.owner)

    if owner is None:
        known = ", ".join(sorted(by_id)) or "(none)"
        return RoutingDecision(
            objective_id=objective.id,
            owner=objective.owner,
            steps=tuple(
                StepRouting(
                    step_id=step.id,
                    assignee=step.assignee,
                    allowed=False,
                    reason=UNKNOWN_OWNER,
                    detail=(
                        f"objective owner {objective.owner!r} is not a declared agent; "
                        f"declared agents: {known}"
                    ),
                )
                for step in objective.steps
            ),
        )

    warnings: list[str] = []
    if not owner.enabled:
        warnings.append(
            f"owner {owner.id!r} is disabled; nothing will run this objective's own work item"
        )

    permitted = frozenset(owner.delegation.may_assign_to) | {owner.id}
    steps = tuple(
        _route_step(step, owner=owner, permitted=permitted, by_id=by_id)
        for step in objective.steps
    )

    delegated = [step.step_id for step in steps if step.delegated_routing]
    if delegated:
        warnings.append(
            f"step(s) {', '.join(delegated)} are decomposed by the runtime, so the routing "
            "of their children is chosen by the runtime's decomposer and is not governed by "
            f"{owner.id!r}'s delegation policy"
        )

    return RoutingDecision(
        objective_id=objective.id,
        owner=objective.owner,
        steps=steps,
        warnings=tuple(warnings),
    )


def _route_step(
    step: PlanStep,
    *,
    owner: AgentSpec,
    permitted: frozenset,
    by_id: Mapping[str, AgentSpec],
) -> StepRouting:
    assignee = by_id.get(step.assignee)

    if assignee is None:
        return StepRouting(
            step_id=step.id,
            assignee=step.assignee,
            allowed=False,
            reason=UNKNOWN_ASSIGNEE,
            detail=(
                f"{step.assignee!r} is not a declared agent; declared agents: "
                f"{', '.join(sorted(by_id)) or '(none)'}"
            ),
        )

    if step.assignee not in permitted:
        allowed_names = ", ".join(sorted(permitted - {owner.id})) or "(nothing)"
        return StepRouting(
            step_id=step.id,
            assignee=step.assignee,
            allowed=False,
            reason=NOT_DELEGABLE,
            detail=(
                f"{owner.id!r} may not assign work to {step.assignee!r}. Its "
                f"delegation.may_assign_to permits: {allowed_names}. Add "
                f"{step.assignee!r} there if this delegation is intended"
            ),
        )

    if not assignee.enabled:
        return StepRouting(
            step_id=step.id,
            assignee=step.assignee,
            allowed=False,
            reason=DISABLED_ASSIGNEE,
            detail=(
                f"{step.assignee!r} is disabled in its spec, so it has no profile to run "
                "this step. Enable it, or route the step elsewhere"
            ),
        )

    return StepRouting(
        step_id=step.id,
        assignee=step.assignee,
        allowed=True,
        delegated_routing=step.decompose,
    )


def plan_order(objective: ObjectiveSpec) -> tuple[PlanStep, ...]:
    """Steps in an order where every dependency precedes its dependants.

    Submission needs this because a work item's parents must already exist when it is
    created — the runtime resolves parents by id, not by promise. Ties are broken by
    declaration order rather than by id, so the board reads the way the file reads.

    Assumes an acyclic plan; :func:`nova.supervisor.objective._check_dependencies` has
    already refused anything else at load.
    """
    remaining = {step.id: step for step in objective.steps}
    placed: list[PlanStep] = []
    satisfied: set[str] = set()

    while remaining:
        ready = [
            step
            for step in objective.steps
            if step.id in remaining and all(parent in satisfied for parent in step.depends_on)
        ]
        if not ready:
            # Unreachable for a loaded objective; a plan built in code could still get here,
            # and silently emitting a partial order would be worse than saying so.
            raise RoutingError(
                f"objective {objective.id!r} has steps that can never start: "
                f"{', '.join(sorted(remaining))}"
            )
        for step in ready:
            placed.append(step)
            satisfied.add(step.id)
            del remaining[step.id]
    return tuple(placed)
