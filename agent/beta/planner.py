"""Dependency-aware planning for Beta orchestration.

The planner converts a request into auditable steps tied to capabilities rather
than hard-coded agent names. It is deterministic and model-independent so it
can be unit tested and safely used by CLI and gateway surfaces.
"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.beta.risk import RiskLevel, classify_risk
from agent.beta.router import RoutingDecision
from agent.beta.specialists import SpecialistRegistry


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    objective: str
    capability: str
    specialist_id: str
    dependencies: tuple[str, ...] = ()
    execution_mode: Literal["direct", "delegate", "kanban"] = "delegate"
    risk: RiskLevel = RiskLevel.LOW
    constraints: tuple[str, ...] = ()
    expected_deliverable: str = "Evidence-backed structured findings"
    status: Literal["pending", "completed", "failed", "blocked"] = "pending"


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    request: str
    steps: tuple[PlanStep, ...]
    blocked_reason: str | None = None
    revision: int = 1

    def ordered_steps(self) -> tuple[PlanStep, ...]:
        """Return a stable topological order and reject cyclic plans."""
        by_id = {step.id: step for step in self.steps}
        incoming = {step.id: len(step.dependencies) for step in self.steps}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for step in self.steps:
            for dependency in step.dependencies:
                if dependency not in by_id:
                    raise ValueError(f"unknown plan dependency: {dependency}")
                outgoing[dependency].append(step.id)
        queue = deque(step.id for step in self.steps if incoming[step.id] == 0)
        result: list[PlanStep] = []
        while queue:
            current = queue.popleft()
            result.append(by_id[current])
            for child in outgoing[current]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        if len(result) != len(self.steps):
            raise ValueError("plan contains a dependency cycle")
        return tuple(result)

    def ready_steps(self) -> tuple[PlanStep, ...]:
        completed = {step.id for step in self.steps if step.status == "completed"}
        return tuple(
            step for step in self.ordered_steps()
            if step.status == "pending" and set(step.dependencies).issubset(completed)
        )


def build_plan(
    request: str,
    decision: RoutingDecision,
    registry: SpecialistRegistry,
    *,
    durable_threshold: int = 3,
) -> ExecutionPlan:
    """Build a capability-based plan from a routing decision."""
    if not decision.delegation_needed:
        return ExecutionPlan(
            id=f"plan-{uuid.uuid4().hex}",
            request=request,
            steps=(PlanStep(
                id="respond",
                objective=request,
                capability="conversation",
                specialist_id="beta",
                execution_mode="direct",
                risk=RiskLevel.LOW,
                expected_deliverable="Direct answer to the Chief",
            ),),
        )

    steps: list[PlanStep] = []
    for index, specialist_id in enumerate(decision.specialists):
        specialist = registry.get(specialist_id)
        if specialist is None or not specialist.enabled:
            continue
        capability = specialist.capabilities[0] if specialist.capabilities else specialist.id
        risk = classify_risk(request)
        mode: Literal["delegate", "kanban"] = (
            "kanban" if len(decision.specialists) >= durable_threshold else "delegate"
        )
        steps.append(PlanStep(
            id=f"investigate-{index + 1}",
            objective=f"Investigate the request as {specialist.name}",
            capability=capability,
            specialist_id=specialist.id,
            execution_mode=mode,
            risk=risk,
            constraints=(
                "Use read-only tools unless an exact approved operation is supplied",
                "Return facts, hypotheses, evidence, confidence, risks, and recommendations",
                "Do not contact the Chief directly",
            ),
        ))

    dependencies = tuple(step.id for step in steps)
    steps.append(PlanStep(
        id="consolidate",
        objective="Validate evidence, reconcile conflicts, and prepare one answer",
        capability="quality-assurance",
        specialist_id="qa-auditor",
        dependencies=dependencies,
        execution_mode="delegate",
        risk=RiskLevel.LOW,
        expected_deliverable="Consolidated response with facts, uncertainty, risk, and next step",
    ))
    return ExecutionPlan(id=f"plan-{uuid.uuid4().hex}", request=request, steps=tuple(steps))


def replan(plan: ExecutionPlan, failed_step_id: str, registry: SpecialistRegistry) -> ExecutionPlan:
    """Replace one failed step with the next compatible available specialist."""
    failed = next((step for step in plan.steps if step.id == failed_step_id), None)
    if failed is None:
        raise ValueError(f"unknown failed step: {failed_step_id}")
    candidates = registry.find_by_capability(failed.capability)
    replacement = next((item for item in candidates if item.id != failed.specialist_id and item.enabled), None)
    if replacement is None:
        updated = tuple(
            step.model_copy(update={"status": "blocked"}) if step.id == failed_step_id else step
            for step in plan.steps
        )
        return plan.model_copy(update={
            "steps": updated,
            "blocked_reason": f"No replacement specialist for capability {failed.capability}",
            "revision": plan.revision + 1,
        })
    updated = tuple(
        step.model_copy(update={"specialist_id": replacement.id, "status": "pending"})
        if step.id == failed_step_id else step
        for step in plan.steps
    )
    return plan.model_copy(update={"steps": updated, "revision": plan.revision + 1})
