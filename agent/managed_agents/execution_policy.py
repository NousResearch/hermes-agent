"""Execution policy decisions for managed-agent runs.

P3-D/P3-E turns ledger outcomes into an orchestration plan.  The policy engine
does not spawn processes; it is the deterministic kernel layer that decides
what the next executor should do after a run succeeds, times out, fails, or
needs revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .failure_reroute import FailureRerouteDecision, decide_failure_reroute
from .model_tier_router import ModelTierDecision, resolve_model_tier
from .registry import AgentRegistry


SUCCESS_CLASSIFICATIONS = {"ok", "no_findings", "approved"}
REVISION_CLASSIFICATIONS = {"revision_needed", "needs_revision"}
UNKNOWN_STOP_CLASSIFICATIONS = {"manual_review", "blocked", "rejected"}


from enum import Enum


class TaskType(str, Enum):
    implementation = "implementation"
    bugfix = "bugfix"
    refactor = "refactor"
    test = "test"
    review = "review"
    architecture = "architecture"
    documentation = "documentation"
    smoke = "smoke"
    migration = "migration"
    investigation = "investigation"

    @staticmethod
    def from_task_type(value: str | None) -> "TaskType":
        default = TaskType.investigation
        if not value:
            return default
        # Backward-compatible aliases for common legacy abbreviations
        _aliases = {
            "tests": TaskType.test,
            "code_review": TaskType.review,
            "architecture_review": TaskType.architecture,
        }
        if value in _aliases:
            return _aliases[value]
        try:
            return TaskType(value)
        except ValueError:
            return default


@dataclass(frozen=True, slots=True)
class ExecutionPolicyDecision:
    task_id: str
    status: str
    action: str
    reason: str
    task_type: str
    risk_level: str
    current_agent_id: str | None
    current_model_ref: str | None
    next_agent_id: str | None
    next_model_ref: str | None
    max_attempts: int
    attempt_count: int
    should_execute: bool
    requires_human_approval: bool
    latest_run_id: str | None = None
    latest_classification: str | None = None
    reroute: FailureRerouteDecision | None = None
    model_route: ModelTierDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "task_type": self.task_type,
            "risk_level": self.risk_level,
            "current_agent_id": self.current_agent_id,
            "current_model_ref": self.current_model_ref,
            "next_agent_id": self.next_agent_id,
            "next_model_ref": self.next_model_ref,
            "max_attempts": self.max_attempts,
            "attempt_count": self.attempt_count,
            "should_execute": self.should_execute,
            "requires_human_approval": self.requires_human_approval,
            "latest_run_id": self.latest_run_id,
            "latest_classification": self.latest_classification,
            "model_route": self.model_route.to_dict() if self.model_route else None,
            "reroute": self.reroute.to_dict() if self.reroute else None,
        }


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _latest_run(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    runs = task.get("runs")
    if isinstance(runs, list):
        candidates = [
            row for row in runs
            if isinstance(row, Mapping) and row.get("event") != "lifecycle_event"
        ]
        if candidates:
            return max(
                candidates,
                key=lambda row: str(row.get("started_at") or row.get("finished_at") or ""),
            )
        for row in runs:
            if isinstance(row, Mapping) and row.get("event") != "lifecycle_event":
                return row
    return None


def _classification_for(task: Mapping[str, Any], run: Mapping[str, Any] | None) -> str:
    raw = (
        (run.get("classification") if run else None)
        or task.get("latest_classification")
        or task.get("status")
        or "unknown"
    )
    return str(raw).strip().lower() or "unknown"


def _agent_for(task: Mapping[str, Any], run: Mapping[str, Any] | None) -> str | None:
    return _as_str((run.get("agent_id") if run else None) or task.get("latest_agent_id"))


def _model_for(run: Mapping[str, Any] | None) -> str | None:
    if not run:
        return None
    for key in ("model_ref", "model", "failed_model_ref"):
        value = _as_str(run.get(key))
        if value:
            return value
    return None


def _attempt_count(task: Mapping[str, Any]) -> int:
    value = task.get("run_count")
    if isinstance(value, int):
        return max(0, value)
    runs = task.get("runs")
    if not isinstance(runs, list):
        return 0
    return len([
        row for row in runs
        if isinstance(row, Mapping) and row.get("event") != "lifecycle_event"
    ])


def _max_attempts(risk_level: str) -> int:
    if risk_level in {"R3", "R4"}:
        return 1
    if risk_level == "R2":
        return 2
    return 3


def _resolve_route_or_none(
    registry: AgentRegistry,
    models_cfg: Mapping[str, Any],
    *,
    agent_id: str | None,
    task_type: str,
    risk_level: str,
    effectiveness: Mapping[str, Mapping[str, Any]] | None = None,
) -> ModelTierDecision | None:
    try:
        return resolve_model_tier(
            registry,
            models_cfg,
            agent_id=agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
    except Exception:
        if not agent_id:
            return None
    try:
        return resolve_model_tier(
            registry,
            models_cfg,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
    except Exception:
        return None


def _manual(
    *,
    task_id: str,
    status: str,
    reason: str,
    task_type: str,
    risk_level: str,
    current_agent_id: str | None,
    current_model_ref: str | None,
    latest_run_id: str | None,
    latest_classification: str | None,
    attempt_count: int,
    max_attempts: int,
) -> ExecutionPolicyDecision:
    return ExecutionPolicyDecision(
        task_id=task_id,
        status=status,
        action="manual_review",
        reason=reason,
        task_type=task_type,
        risk_level=risk_level,
        current_agent_id=current_agent_id,
        current_model_ref=current_model_ref,
        next_agent_id=None,
        next_model_ref=None,
        max_attempts=max_attempts,
        attempt_count=attempt_count,
        should_execute=False,
        requires_human_approval=True,
        latest_run_id=latest_run_id,
        latest_classification=latest_classification,
    )


def decide_execution_policy(
    registry: AgentRegistry,
    models_cfg: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    task_type: str = "tests",
    risk_level: str = "R1",
    effectiveness: Mapping[str, Mapping[str, Any]] | None = None,
) -> ExecutionPolicyDecision:
    task_id = str(task.get("task_id") or "unknown")
    status = str(task.get("status") or "unknown")
    run = _latest_run(task)
    classification = _classification_for(task, run)
    current_agent_id = _agent_for(task, run)
    current_model_ref = _model_for(run)
    latest_run_id = _as_str((run.get("run_id") if run else None) or task.get("latest_run_id"))
    attempts = _attempt_count(task)
    max_attempts = _max_attempts(risk_level)

    if status == "running":
        route = _resolve_route_or_none(
            registry,
            models_cfg,
            agent_id=current_agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        return ExecutionPolicyDecision(
            task_id=task_id,
            status=status,
            action="continue",
            reason="run_still_active",
            task_type=task_type,
            risk_level=route.risk_level if route else risk_level,
            current_agent_id=current_agent_id,
            current_model_ref=current_model_ref,
            next_agent_id=route.agent_id if route else current_agent_id,
            next_model_ref=route.model_ref if route else current_model_ref,
            max_attempts=max_attempts,
            attempt_count=attempts,
            should_execute=False,
            requires_human_approval=False,
            latest_run_id=latest_run_id,
            latest_classification=classification,
            model_route=route,
        )

    if classification in SUCCESS_CLASSIFICATIONS or status == "ok":
        route = _resolve_route_or_none(
            registry,
            models_cfg,
            agent_id=current_agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        return ExecutionPolicyDecision(
            task_id=task_id,
            status=status,
            action="complete",
            reason="latest_run_succeeded",
            task_type=task_type,
            risk_level=route.risk_level if route else risk_level,
            current_agent_id=current_agent_id,
            current_model_ref=current_model_ref,
            next_agent_id=route.agent_id if route else current_agent_id,
            next_model_ref=route.model_ref if route else current_model_ref,
            max_attempts=max_attempts,
            attempt_count=attempts,
            should_execute=False,
            requires_human_approval=False,
            latest_run_id=latest_run_id,
            latest_classification=classification,
            model_route=route,
        )

    if classification in UNKNOWN_STOP_CLASSIFICATIONS:
        return _manual(
            task_id=task_id,
            status=status,
            reason=f"{classification}_requires_manual_review",
            task_type=task_type,
            risk_level=risk_level,
            current_agent_id=current_agent_id,
            current_model_ref=current_model_ref,
            latest_run_id=latest_run_id,
            latest_classification=classification,
            attempt_count=attempts,
            max_attempts=max_attempts,
        )

    if attempts >= max_attempts:
        return _manual(
            task_id=task_id,
            status=status,
            reason="max_attempts_exhausted",
            task_type=task_type,
            risk_level=risk_level,
            current_agent_id=current_agent_id,
            current_model_ref=current_model_ref,
            latest_run_id=latest_run_id,
            latest_classification=classification,
            attempt_count=attempts,
            max_attempts=max_attempts,
        )

    failure = "revision_needed" if classification in REVISION_CLASSIFICATIONS else classification
    reroute = decide_failure_reroute(
        registry,
        models_cfg,
        task_type=task_type,
        risk_level=risk_level,
        failure=failure,
        failed_agent_id=current_agent_id,
        failed_model_ref=current_model_ref,
        effectiveness=effectiveness,
    )
    return ExecutionPolicyDecision(
        task_id=task_id,
        status=status,
        action=reroute.action,
        reason=reroute.reason,
        task_type=task_type,
        risk_level=reroute.risk_level,
        current_agent_id=current_agent_id,
        current_model_ref=current_model_ref,
        next_agent_id=reroute.next_agent_id,
        next_model_ref=reroute.next_model_ref,
        max_attempts=max_attempts,
        attempt_count=attempts,
        should_execute=reroute.action in {"retry_same_agent", "switch_agent", "switch_model"},
        requires_human_approval=reroute.requires_human_approval,
        latest_run_id=latest_run_id,
        latest_classification=classification,
        reroute=reroute,
    )
