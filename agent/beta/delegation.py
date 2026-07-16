"""Structured delegation contracts over Hermes' existing delegate_task."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.beta.router import RoutingDecision
from agent.beta.risk import ApprovalGate, ApprovalReceipt, Operation, RiskLevel
from agent.beta.specialists import Specialist, SpecialistRegistry, default_specialist_registry
from toolsets import resolve_multiple_toolsets


class DelegationTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    specialist_id: str
    objective: str
    minimal_context: str
    constraints: tuple[str, ...]
    risk: Literal["low", "medium", "high"]
    allowed_tools: tuple[str, ...]
    expected_deliverable: str
    timeout_seconds: int = Field(default=300, gt=0)
    correlation_id: str

    def delegate_entry(self) -> dict[str, str]:
        response_schema = SpecialistResult.model_json_schema()
        context = {
            "task": self.model_dump(),
            "response_schema": response_schema,
            "instructions": (
                "Work only on this isolated task. Do not contact the Chief, write strategic "
                "memory, or use tools outside allowed_tools. Return one JSON object only."
            ),
        }
        return {
            "goal": self.objective,
            "context": json.dumps(context, ensure_ascii=False),
            "role": "leaf",
        }


class SpecialistResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    specialist_id: str
    correlation_id: str
    status: Literal["completed", "failed", "timeout", "contract_error"]
    summary: str = ""
    evidence: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    confidence: float = Field(default=0, ge=0, le=1)
    recommended_actions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    authorization_required: bool = False
    errors: tuple[str, ...] = ()


def _allowed_tools(specialist: Specialist) -> tuple[str, ...]:
    blocked = set(specialist.blocked_tools)
    return tuple(
        tool
        for tool in resolve_multiple_toolsets(list(specialist.allowed_toolsets))
        if tool not in blocked
    )


def create_delegation_tasks(
    request: str,
    decision: RoutingDecision,
    registry: SpecialistRegistry | None = None,
    *,
    correlation_id: str | None = None,
    timeout_seconds: int = 300,
) -> tuple[DelegationTask, ...]:
    """Create isolated specialist tasks from a routing decision."""
    registry = registry or default_specialist_registry()
    correlation_id = correlation_id or f"beta-{uuid.uuid4().hex}"
    tasks = []
    for specialist_id in decision.specialists:
        specialist = registry.get(specialist_id)
        if specialist is None or not specialist.enabled:
            continue
        tasks.append(
            DelegationTask(
                task_id=f"{correlation_id}:{specialist.id}",
                specialist_id=specialist.id,
                objective=f"Investigate as {specialist.name}: {request}",
                minimal_context=request,
                constraints=(
                    "Use read-only investigation for diagnosis",
                    "Do not perform impactful changes",
                    "Do not contact the Chief directly",
                    "Do not write Beta strategic memory",
                ),
                risk=decision.initial_risk,
                allowed_tools=_allowed_tools(specialist),
                expected_deliverable="Evidence-backed structured findings",
                timeout_seconds=timeout_seconds,
                correlation_id=correlation_id,
            )
        )
    return tuple(tasks)


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("response is not a JSON object")
    text = raw.strip()
    if text.startswith("```"):
        text = re_sub_fence(text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def re_sub_fence(text: str) -> str:
    """Strip one Markdown JSON fence without accepting surrounding prose."""
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def _contract_error(task: DelegationTask, error: str, status: str = "contract_error") -> SpecialistResult:
    return SpecialistResult(
        task_id=task.task_id,
        specialist_id=task.specialist_id,
        correlation_id=task.correlation_id,
        status=status,
        errors=(error,),
    )


def task_operation(task: DelegationTask) -> Operation:
    """Build the exact approval scope for one delegated operation."""
    return Operation(
        target=task.minimal_context,
        action=task.objective,
        impact="May change production or another high-impact system",
        rollback="Use the specialist-provided rollback plan",
        risk=RiskLevel(task.risk),
    )


def execute_delegations(
    tasks: Iterable[DelegationTask],
    parent_agent: Any,
    *,
    delegate: Callable[..., str] | None = None,
    approval_gate: ApprovalGate | None = None,
    approval_receipts: dict[str, ApprovalReceipt] | None = None,
) -> tuple[SpecialistResult, ...]:
    """Run one delegate_task batch and validate each specialist response."""
    task_list = tuple(tasks)
    if not task_list:
        return ()
    gate = approval_gate or ApprovalGate()
    receipts = approval_receipts or {}
    blocked = {
        index: task
        for index, task in enumerate(task_list)
        if not gate.authorized(task_operation(task), receipts.get(task.task_id))
    }
    runnable = tuple(task for index, task in enumerate(task_list) if index not in blocked)
    if not runnable:
        return tuple(
            SpecialistResult(
                task_id=task.task_id,
                specialist_id=task.specialist_id,
                correlation_id=task.correlation_id,
                status="failed",
                authorization_required=True,
                errors=("Explicit approval required before high-risk execution",),
            )
            for task in task_list
        )
    if delegate is None:
        from tools.delegate_tool import delegate_task

        delegate = delegate_task

    try:
        payload = _json_object(
            delegate(
                tasks=[task.delegate_entry() for task in runnable],
                background=False,
                parent_agent=parent_agent,
            )
        )
    except (ValueError, json.JSONDecodeError) as exc:
        delegated = tuple(_contract_error(task, f"delegation response invalid: {exc}") for task in runnable)
        return _merge_blocked(task_list, delegated)

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        delegated = tuple(_contract_error(task, "delegation response missing results list") for task in runnable)
        return _merge_blocked(task_list, delegated)

    by_index = {
        entry.get("task_index"): entry
        for entry in raw_results
        if isinstance(entry, dict) and isinstance(entry.get("task_index"), int)
    }
    results = []
    for index, task in enumerate(runnable):
        entry = by_index.get(index)
        if entry is None:
            results.append(_contract_error(task, "missing specialist result"))
            continue
        outer_status = entry.get("status")
        if outer_status in {"timeout", "interrupted"}:
            results.append(_contract_error(task, entry.get("error") or "specialist timed out", "timeout"))
            continue
        if outer_status != "completed":
            results.append(_contract_error(task, entry.get("error") or "specialist failed", "failed"))
            continue
        try:
            result = SpecialistResult.model_validate(_json_object(entry.get("summary")))
            if (result.task_id, result.specialist_id, result.correlation_id) != (
                task.task_id,
                task.specialist_id,
                task.correlation_id,
            ):
                raise ValueError("response identifiers do not match delegated task")
            results.append(result)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            results.append(_contract_error(task, f"invalid specialist response: {exc}"))
    return _merge_blocked(task_list, tuple(results))


def _merge_blocked(
    all_tasks: tuple[DelegationTask, ...],
    delegated: tuple[SpecialistResult, ...],
) -> tuple[SpecialistResult, ...]:
    by_task = {result.task_id: result for result in delegated}
    return tuple(
        by_task.get(task.task_id)
        or SpecialistResult(
            task_id=task.task_id,
            specialist_id=task.specialist_id,
            correlation_id=task.correlation_id,
            status="failed",
            authorization_required=True,
            errors=("Explicit approval required before high-risk execution",),
        )
        for task in all_tasks
    )
