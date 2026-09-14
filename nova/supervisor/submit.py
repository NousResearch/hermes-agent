"""Turning a routed objective into work on a runtime's board.

Three things happen here in a fixed order, and the order is the design:

1. **Route.** If any step may not go where it was declared to go, nothing is submitted.
   Not "submit the valid steps and report the rest" — a plan is a unit, and half a
   month-end close running while the other half is refused is worse than none of it.
2. **Order.** Steps are topologically sorted so every dependency exists before the item
   that waits on it.
3. **Submit.** Through the runtime contract, idempotently, with every item audited.

The idempotency key is ``<objective id>:<step id>``, which makes re-submission safe by
construction. That matters more than it looks: submission touches an external system one
item at a time, so a crash halfway leaves half a plan on the board. Re-running finishes it
rather than duplicating it, and an operator does not have to reason about which half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from nova.audit import AuditLog, new_correlation_id
from nova.runtime.base import AgentRuntime, SubmitResult, WorkItem
from nova.spec import AgentSpec
from nova.spec.objective import ObjectiveSpec, PlanStep
from nova.supervisor.route import RoutingDecision, plan_order, route_objective


def work_key(objective_id: str, step_id: str) -> str:
    """The idempotency key for one step of one objective. One definition, used everywhere."""
    return f"{objective_id}:{step_id}"


@dataclass(frozen=True)
class SubmitReport:
    """What submitting one objective did, or would have done."""

    objective_id: str
    correlation_id: str
    routing: RoutingDecision
    result: Optional[SubmitResult] = None
    dry_run: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def submitted(self) -> bool:
        return self.result is not None

    @property
    def refused(self) -> bool:
        return not self.routing.allowed

    def summary(self) -> str:
        if self.refused:
            return self.routing.explain()
        if self.result is None:
            return f"{self.objective_id}: routed cleanly, nothing submitted"
        verb = "would create" if self.dry_run else "created"
        parts = [
            f"{self.objective_id}: {verb} {len(self.result.created)} work item(s)",
        ]
        if self.result.existing:
            parts.append(f"{len(self.result.existing)} already existed")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "correlation_id": self.correlation_id,
            "dry_run": self.dry_run,
            "refused": self.refused,
            "routing": self.routing.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "warnings": list(self.warnings),
        }


def submit_objective(
    objective: ObjectiveSpec,
    agents: Sequence[AgentSpec],
    runtime: AgentRuntime,
    *,
    audit: AuditLog,
    tenant_id: str = "",
    correlation_id: Optional[str] = None,
    dry_run: bool = False,
) -> SubmitReport:
    """Route ``objective`` and, if it routes cleanly, place its steps on the runtime.

    Returns a report rather than raising on a refusal: being refused is a normal outcome
    that an operator needs the detail of, and it is the outcome this whole phase exists to
    produce.
    """
    correlation_id = correlation_id or new_correlation_id()
    routing = route_objective(objective, agents)
    warnings = list(routing.warnings)

    if not objective.enabled:
        warnings.append(f"objective {objective.id!r} is disabled in its spec")

    if not runtime.capabilities.work_submission:
        warnings.append(
            f"runtime {runtime.name!r} cannot accept submitted work; this objective can be "
            "planned and routed but not run"
        )

    if routing.refusals:
        # Recorded even though nothing was written. A refused delegation is the single most
        # interesting thing this subsystem produces, and an audit log that only contains
        # successful submissions cannot answer "did anyone try".
        audit.record(
            "objective.refused",
            correlation_id=correlation_id,
            subject=objective.id,
            detail={
                "owner": objective.owner,
                "refusals": [step.to_dict() for step in routing.refusals],
            },
        )
        return SubmitReport(
            objective_id=objective.id,
            correlation_id=correlation_id,
            routing=routing,
            dry_run=dry_run,
            warnings=tuple(warnings),
        )

    # No second refusal for a runtime that cannot submit: the contract's default
    # ``submit_work`` already raises, and two definitions of one refusal drift apart. A dry
    # run still goes through, because planning and routing are useful on a runtime that
    # cannot yet run the plan — that is what the warning above is for.
    items = build_work_items(objective, agents, tenant_id=tenant_id)
    result = runtime.submit_work(
        items, audit=audit, correlation_id=correlation_id, dry_run=dry_run
    )
    warnings.extend(result.warnings)

    return SubmitReport(
        objective_id=objective.id,
        correlation_id=correlation_id,
        routing=routing,
        result=result,
        dry_run=dry_run,
        warnings=tuple(warnings),
    )


def build_work_items(
    objective: ObjectiveSpec,
    agents: Sequence[AgentSpec] = (),
    *,
    tenant_id: str = "",
) -> tuple[WorkItem, ...]:
    """The objective's steps as runtime-agnostic work items, in dependency order.

    A step that does not state a runtime cap or a retry limit inherits its **assignee's**
    declared limits. That inheritance is the difference between a control and a decoration:
    the dispatcher genuinely enforces a per-task runtime cap — SIGTERM, grace, SIGKILL at
    ``kanban_db_dispatch.py::enforce_max_runtime`` — so an agent whose spec says
    ``max_task_runtime_seconds: 900`` should get 900 seconds on every task NOVA creates for
    it, not only on the steps whose author happened to restate the number.

    Before this, the agent-level limit was compiled into the profile under the ``nova:`` key
    the runtime ignores, and dropped everywhere else. A limit that is declared, enforceable
    and never applied is exactly the shape of control this platform exists to eliminate.
    """
    by_id = {spec.id: spec for spec in agents}

    def limits_for(step: PlanStep) -> tuple[Optional[int], Optional[int]]:
        spec = by_id.get(step.assignee)
        agent_runtime = spec.limits.max_task_runtime_seconds if spec else None
        agent_retries = spec.limits.max_retries if spec else None
        # The step wins: it is the narrower statement, made by whoever wrote this plan.
        return (
            step.max_runtime_seconds if step.max_runtime_seconds is not None else agent_runtime,
            step.max_retries if step.max_retries is not None else agent_retries,
        )

    items: list[WorkItem] = []
    for step in plan_order(objective):
        runtime_cap, retry_cap = limits_for(step)
        items.append(
            WorkItem(
                key=work_key(objective.id, step.id),
                title=step.title,
                assignee=step.assignee,
                body=_body(objective, step),
                depends_on=tuple(work_key(objective.id, p) for p in step.depends_on),
                priority=step.priority,
                max_runtime_seconds=runtime_cap,
                max_retries=retry_cap,
                decompose=step.decompose,
                tenant_id=tenant_id,
            )
        )
    return tuple(items)


def _body(objective: ObjectiveSpec, step: PlanStep) -> str:
    """The step's brief, as the worker will read it with no other context.

    A worker starts fresh: it sees this text and nothing else about why the work exists.
    So the objective and its acceptance criteria are restated on every step rather than
    assumed — the alternative is an agent that completes its task correctly and misses the
    point of it, which is the expensive kind of wrong.
    """
    parts: list[str] = []
    if step.body:
        parts.append(step.body.strip())

    context = [f"Part of objective **{objective.title}** ({objective.id}), step `{step.id}`."]
    if objective.description:
        context.append(objective.description.strip())
    if objective.acceptance:
        context.append(f"The objective is complete when: {objective.acceptance.strip()}")
    if step.depends_on:
        context.append(
            "This step runs after: " + ", ".join(f"`{name}`" for name in step.depends_on) + "."
        )
    parts.append("\n\n".join(context))

    return "\n\n---\n\n".join(parts).strip() + "\n"
