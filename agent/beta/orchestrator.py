"""Thin Beta orchestration flow over Hermes delegation primitives."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.beta.consolidation import ConsolidatedResponse, consolidate_results
from agent.beta.delegation import (
    SpecialistResult,
    create_delegation_tasks,
    execute_delegations,
)
from agent.beta.risk import ApprovalGate, Operation, RiskLevel, classify_risk
from agent.beta.router import RoutingDecision, route_request
from agent.beta.specialists import SpecialistRegistry, default_specialist_registry


class BetaRun(BaseModel):
    """Auditable result of one Beta orchestration turn."""

    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    specialist_results: tuple[SpecialistResult, ...]
    response: ConsolidatedResponse
    approval_requests: tuple[Operation, ...] = ()
    approved_operations: tuple[str, ...] = ()
    executed_actions: tuple[str, ...] = ()


def _qa_validator(
    request: str,
    decision: RoutingDecision,
    registry: SpecialistRegistry,
    parent_agent: Any,
    delegate: Callable[..., str] | None,
    approval_gate: ApprovalGate,
) -> Callable[[tuple[SpecialistResult, ...], tuple[str, ...]], SpecialistResult]:
    def validate(
        results: tuple[SpecialistResult, ...],
        contradictions: tuple[str, ...],
    ) -> SpecialistResult:
        review_context = json.dumps(
            {
                "request": request,
                "specialist_results": [result.model_dump(mode="json") for result in results],
                "contradictions": contradictions,
            },
            ensure_ascii=False,
        )
        review_decision = decision.model_copy(
            update={
                "specialists": ("qa-auditor",),
                "initial_risk": "low",
                "delegation_needed": True,
                "parallelizable": False,
            }
        )
        task = create_delegation_tasks(
            review_context,
            review_decision,
            registry,
            correlation_id=results[0].correlation_id if results else None,
        )[0]
        return execute_delegations(
            (task,),
            parent_agent,
            delegate=delegate,
            approval_gate=approval_gate,
        )[0]

    return validate


def orchestrate_request(
    request: str,
    parent_agent: Any,
    *,
    delegate: Callable[..., str] | None = None,
    registry: SpecialistRegistry | None = None,
    approval_gate: ApprovalGate | None = None,
) -> BetaRun:
    """Route, delegate, validate, consolidate, and request exact approvals."""
    registry = registry or default_specialist_registry()
    gate = approval_gate or ApprovalGate()
    decision = route_request(request, registry)
    tasks = create_delegation_tasks(request, decision, registry)
    results = execute_delegations(
        tasks,
        parent_agent,
        delegate=delegate,
        approval_gate=gate,
    )
    response = consolidate_results(
        request,
        decision,
        results,
        qa_validator=_qa_validator(
            request,
            decision,
            registry,
            parent_agent,
            delegate,
            gate,
        ),
    )

    operations = tuple(
        Operation(
            target=request,
            action=action,
            impact="May change a production system or interrupt active work",
            rollback="Revert the exact operation or restore the affected service state",
            risk=RiskLevel.HIGH,
        )
        for action in response.recommendation
        if classify_risk(action) == RiskLevel.HIGH
    )
    approved = tuple(
        operation.fingerprint
        for operation in operations
        if gate.request(operation) is not None
    )
    return BetaRun(
        decision=decision,
        specialist_results=results,
        response=response,
        approval_requests=operations,
        approved_operations=approved,
    )
