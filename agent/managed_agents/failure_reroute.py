"""Failure-based reroute decisions for managed agents.

P3-C consumes failure outcomes from Run Ledger / gate records and proposes the
next orchestration action.  It does not execute retries itself; it returns a
small, explainable plan that dispatchers and dashboards can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .capability_matrix import build_capability_matrix, preview_route
from .model_tier_router import ModelTierDecision, resolve_model_tier
from .registry import AgentRegistry


MODEL_RETRY_FAILURES = {"rate_limited", "quota_exceeded", "server_error", "empty_final_content"}
AGENT_SWITCH_FAILURES = {"timeout", "ineffective", "process_error", "failed"}
MANUAL_FAILURES = {"permission_error", "auth_error", "rejected"}


@dataclass(frozen=True, slots=True)
class FailureRerouteDecision:
    action: str
    reason: str
    task_type: str
    risk_level: str
    failed_agent_id: str | None
    failed_model_ref: str | None
    next_agent_id: str | None
    next_model_ref: str | None
    fallback_chain: tuple[str, ...]
    fallback_on: tuple[str, ...]
    timeout_seconds: int
    max_context_chars: int
    requires_human_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "task_type": self.task_type,
            "risk_level": self.risk_level,
            "failed_agent_id": self.failed_agent_id,
            "failed_model_ref": self.failed_model_ref,
            "next_agent_id": self.next_agent_id,
            "next_model_ref": self.next_model_ref,
            "fallback_chain": list(self.fallback_chain),
            "fallback_on": list(self.fallback_on),
            "timeout_seconds": self.timeout_seconds,
            "max_context_chars": self.max_context_chars,
            "requires_human_approval": self.requires_human_approval,
        }


def _normal_failure(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "revision_needed":
        return raw
    if raw in {"ok", "complete", "completed", "approved"}:
        return "ok"
    if raw in {"timeout", "timed_out"}:
        return "timeout"
    if raw in {"ineffective", "no_effective_output"}:
        return "ineffective"
    if raw in {"failed", "process_error", "error"}:
        return "process_error"
    return raw or "unknown"


def _decision_from_route(
    *,
    action: str,
    reason: str,
    route: ModelTierDecision,
    failed_agent_id: str | None,
    failed_model_ref: str | None,
    requires_human_approval: bool = False,
) -> FailureRerouteDecision:
    return FailureRerouteDecision(
        action=action,
        reason=reason,
        task_type=route.task_type,
        risk_level=route.risk_level,
        failed_agent_id=failed_agent_id,
        failed_model_ref=failed_model_ref,
        next_agent_id=route.agent_id or None,
        next_model_ref=route.model_ref,
        fallback_chain=route.fallback_chain,
        fallback_on=route.fallback_on,
        timeout_seconds=route.timeout_seconds,
        max_context_chars=route.max_context_chars,
        requires_human_approval=requires_human_approval,
    )


def _manual_decision(
    *,
    reason: str,
    task_type: str,
    risk_level: str,
    failed_agent_id: str | None,
    failed_model_ref: str | None,
) -> FailureRerouteDecision:
    return FailureRerouteDecision(
        action="manual_review",
        reason=reason,
        task_type=task_type,
        risk_level=risk_level,
        failed_agent_id=failed_agent_id,
        failed_model_ref=failed_model_ref,
        next_agent_id=None,
        next_model_ref=None,
        fallback_chain=(),
        fallback_on=(),
        timeout_seconds=0,
        max_context_chars=0,
        requires_human_approval=True,
    )


def decide_failure_reroute(
    registry: AgentRegistry,
    models_cfg: Mapping[str, Any],
    *,
    task_type: str,
    risk_level: str,
    failure: str,
    failed_agent_id: str | None = None,
    failed_model_ref: str | None = None,
    effectiveness: Mapping[str, Mapping[str, Any]] | None = None,
) -> FailureRerouteDecision:
    normalized = _normal_failure(failure)
    if normalized == "ok":
        route = resolve_model_tier(
            registry,
            models_cfg,
            agent_id=failed_agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        return _decision_from_route(
            action="complete",
            reason="previous_run_succeeded",
            route=route,
            failed_agent_id=failed_agent_id,
            failed_model_ref=failed_model_ref,
        )

    if normalized in MANUAL_FAILURES:
        return _manual_decision(
            reason=f"{normalized}_requires_human_intervention",
            task_type=task_type,
            risk_level=risk_level,
            failed_agent_id=failed_agent_id,
            failed_model_ref=failed_model_ref,
        )

    if normalized == "revision_needed":
        route = resolve_model_tier(
            registry,
            models_cfg,
            agent_id=failed_agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        return _decision_from_route(
            action="retry_same_agent",
            reason="revision_needed_is_recoverable",
            route=route,
            failed_agent_id=failed_agent_id,
            failed_model_ref=failed_model_ref,
        )

    if normalized in MODEL_RETRY_FAILURES and failed_agent_id:
        route = resolve_model_tier(
            registry,
            models_cfg,
            agent_id=failed_agent_id,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        chain = tuple(ref for ref in route.fallback_chain if ref != failed_model_ref)
        if chain:
            return FailureRerouteDecision(
                action="switch_model",
                reason=f"{normalized}_uses_next_model_in_chain",
                task_type=route.task_type,
                risk_level=route.risk_level,
                failed_agent_id=failed_agent_id,
                failed_model_ref=failed_model_ref,
                next_agent_id=route.agent_id,
                next_model_ref=chain[0],
                fallback_chain=chain,
                fallback_on=route.fallback_on,
                timeout_seconds=route.timeout_seconds,
                max_context_chars=route.max_context_chars,
            )

    if normalized in AGENT_SWITCH_FAILURES:
        route_preview = preview_route(
            registry,
            task_type=task_type,
            risk_level=risk_level,
            effectiveness=effectiveness,
        )
        candidates = [
            agent_id for agent_id in route_preview.get("candidate_agents", [])
            if agent_id != failed_agent_id
        ]
        if candidates:
            route = resolve_model_tier(
                registry,
                models_cfg,
                agent_id=candidates[0],
                task_type=task_type,
                risk_level=risk_level,
                effectiveness=effectiveness,
            )
            return _decision_from_route(
                action="switch_agent",
                reason=f"{normalized}_uses_next_capable_agent",
                route=route,
                failed_agent_id=failed_agent_id,
                failed_model_ref=failed_model_ref,
            )
        if failed_agent_id:
            route = resolve_model_tier(
                registry,
                models_cfg,
                agent_id=failed_agent_id,
                task_type=task_type,
                risk_level=risk_level,
                effectiveness=effectiveness,
            )
            return _decision_from_route(
                action="retry_same_agent",
                reason=f"{normalized}_no_alternate_agent",
                route=route,
                failed_agent_id=failed_agent_id,
                failed_model_ref=failed_model_ref,
            )

    return _manual_decision(
        reason=f"unclassified_failure:{normalized}",
        task_type=task_type,
        risk_level=risk_level,
        failed_agent_id=failed_agent_id,
        failed_model_ref=failed_model_ref,
    )
