"""Beta orchestration flow over Hermes delegation primitives."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.beta.chief_profile import ChiefProfile
from agent.beta.consolidation import ConsolidatedResponse, consolidate_results
from agent.beta.delegation import (
    SpecialistResult,
    create_delegation_tasks,
    execute_delegations,
    task_operation,
)
from agent.beta.planner import ExecutionPlan, build_plan
from agent.beta.risk import ApprovalGate, ApprovalReceipt, Operation, RiskLevel, classify_risk
from agent.beta.router import RoutingDecision, route_request
from agent.beta.specialists import SpecialistRegistry, default_specialist_registry


class ExecutedAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_fingerprint: str
    action: str
    status: str
    evidence: str


class BetaRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    plan: ExecutionPlan
    specialist_results: tuple[SpecialistResult, ...]
    response: ConsolidatedResponse
    approval_requests: tuple[Operation, ...] = ()
    approval_receipts: tuple[ApprovalReceipt, ...] = ()
    executed_actions: tuple[ExecutedAction, ...] = ()
    chief_profile_revision: int = 0


def _qa_validator(
    request: str,
    decision: RoutingDecision,
    registry: SpecialistRegistry,
    parent_agent: Any,
    delegate: Callable[..., str] | None,
    approval_gate: ApprovalGate,
) -> Callable[[tuple[SpecialistResult, ...], tuple[str, ...]], SpecialistResult]:
    def validate(results: tuple[SpecialistResult, ...], contradictions: tuple[str, ...]) -> SpecialistResult:
        review_context = json.dumps({
            "request": request,
            "specialist_results": [result.model_dump(mode="json") for result in results],
            "contradictions": contradictions,
        }, ensure_ascii=False)
        review_decision = decision.model_copy(update={
            "specialists": ("qa-auditor",),
            "initial_risk": "low",
            "delegation_needed": True,
            "parallelizable": False,
        })
        tasks = create_delegation_tasks(
            review_context,
            review_decision,
            registry,
            correlation_id=results[0].correlation_id if results else None,
        )
        if not tasks:
            return SpecialistResult(
                task_id="qa-unavailable",
                specialist_id="qa-auditor",
                correlation_id=results[0].correlation_id if results else "beta",
                status="failed",
                errors=("QA specialist unavailable",),
            )
        return execute_delegations(
            (tasks[0],), parent_agent, delegate=delegate, approval_gate=approval_gate
        )[0]

    return validate


def _recommended_operations(request: str, response: ConsolidatedResponse) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for action in response.recommendation:
        risk = classify_risk(action)
        if risk != RiskLevel.HIGH:
            continue
        operations.append(Operation(
            target=request,
            action=action,
            impact="May change a production system, security posture, data, cost, or active work",
            rollback="Executor must return a concrete rollback plan before changing state",
            risk=risk,
        ))
    return tuple(operations)


def _execute_approved(
    operations: tuple[Operation, ...],
    receipts: dict[str, ApprovalReceipt],
    gate: ApprovalGate,
    executor: Callable[[Operation], str] | None,
) -> tuple[ExecutedAction, ...]:
    if executor is None:
        return ()
    executed: list[ExecutedAction] = []
    for operation in operations:
        receipt = receipts.get(operation.fingerprint)
        if not gate.authorized(operation, receipt):
            continue
        try:
            evidence = executor(operation)
            executed.append(ExecutedAction(
                operation_fingerprint=operation.fingerprint,
                action=operation.action,
                status="completed",
                evidence=str(evidence),
            ))
        except Exception as exc:
            executed.append(ExecutedAction(
                operation_fingerprint=operation.fingerprint,
                action=operation.action,
                status="failed",
                evidence=f"executor error: {exc}",
            ))
    return tuple(executed)


def orchestrate_request(
    request: str,
    parent_agent: Any,
    *,
    delegate: Callable[..., str] | None = None,
    executor: Callable[[Operation], str] | None = None,
    registry: SpecialistRegistry | None = None,
    approval_gate: ApprovalGate | None = None,
    approval_receipts: dict[str, ApprovalReceipt] | None = None,
    chief_profile: ChiefProfile | None = None,
) -> BetaRun:
    """Plan, route, delegate, validate, consolidate, approve, and execute."""
    registry = registry or default_specialist_registry()
    gate = approval_gate or ApprovalGate()
    receipts = approval_receipts or {}
    decision = route_request(request, registry)
    plan = build_plan(request, decision, registry)

    if not decision.delegation_needed:
        response = ConsolidatedResponse(
            understanding=request,
            agents_activated=(),
            result="Direct conversational response required",
            evidence=(),
            facts=(),
            hypotheses=(),
            probable_cause=None,
            confidence=1.0,
            risk=RiskLevel.LOW,
            recommendation=("Answer directly using the active conversation model",),
            authorization_required=False,
            next_step="Respond directly to the Chief",
        )
        return BetaRun(
            decision=decision,
            plan=plan,
            specialist_results=(),
            response=response,
            chief_profile_revision=chief_profile.revision if chief_profile else 0,
        )

    tasks = create_delegation_tasks(request, decision, registry)
    task_receipts = {
        task.task_id: receipts[operation.fingerprint]
        for task in tasks
        for operation in (task_operation(task),)
        if operation.fingerprint in receipts
    }
    results = execute_delegations(
        tasks,
        parent_agent,
        delegate=delegate,
        approval_gate=gate,
        approval_receipts=task_receipts,
    )
    response = consolidate_results(
        request,
        decision,
        results,
        qa_validator=_qa_validator(request, decision, registry, parent_agent, delegate, gate),
    )

    operations = _recommended_operations(request, response)
    issued: list[ApprovalReceipt] = []
    available = dict(receipts)
    for operation in operations:
        existing = available.get(operation.fingerprint)
        if existing is not None and gate.authorized(operation, existing):
            continue
        receipt = gate.request(operation)
        if receipt is not None:
            issued.append(receipt)
            available[operation.fingerprint] = receipt

    executed = _execute_approved(operations, available, gate, executor)
    return BetaRun(
        decision=decision,
        plan=plan,
        specialist_results=results,
        response=response,
        approval_requests=operations,
        approval_receipts=tuple(issued),
        executed_actions=executed,
        chief_profile_revision=chief_profile.revision if chief_profile else 0,
    )
