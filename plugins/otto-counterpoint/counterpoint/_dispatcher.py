# -*- coding: utf-8 -*-
"""Single-entry admission policy for bounded user calls.

This module is deliberately model-free. It normalizes the call envelope, applies
hard scope/approval gates, and chooses one of ``direct``, ``counterpoint``,
``human_review`` or ``blocked`` before a runtime is allowed to execute a turn.
The original user message is never stored in this module's audit payload.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from ._policy import RoutePolicyError, decide_counterpoint

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RISKS = frozenset({"low", "medium", "high", "critical"})
_COMPLEXITIES = frozenset({"simple", "moderate", "complex", "frontier"})


class DispatchConflict(RoutePolicyError):
    """An idempotency key was reused for a different request."""


class DispatchDecision(str, Enum):
    """Admission result; no value authorizes an external effect by itself."""

    DIRECT = "direct"
    COUNTERPOINT = "counterpoint"
    HUMAN_REVIEW = "human_review"
    BLOCKED = "blocked"



def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoutePolicyError(f"{field} must be a non-empty string")
    return value.strip()



def _text_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, frozenset)):
        raise RoutePolicyError(f"{field} must be a list or tuple")
    result = []
    for item in value:
        result.append(_required_text(item, field))
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class RequestEnvelope:
    """Metadata required before a user call can reach an execution path.

    ``objective_ref`` and ``scope_ref`` point to supervisor-owned material. They
    are references, not the user prompt, so serializing this envelope cannot
    accidentally put the raw request in the ledger.
    """

    run_id: str
    task_id: str
    idempotency_key: str
    source: str
    source_message_id: str | None
    request_sha256: str
    objective_ref: str
    scope_ref: str
    risk: str
    complexity: str
    requested_effects: tuple[str, ...] = ()
    forbidden_effects: tuple[str, ...] = ()
    external_effect: bool = False
    sensitive_scope: bool = False
    evidence_conflict: bool = False
    uncertainty: bool = False
    sampling_selected: bool = False
    supervisor_approved: bool = False
    counterpoint_route_ready: bool = False
    workflow_version: str = "counterpoint.dispatch.v1"

    def __post_init__(self) -> None:
        for field in (
            "run_id", "task_id", "idempotency_key", "source", "request_sha256",
            "objective_ref", "scope_ref", "workflow_version",
        ):
            _required_text(getattr(self, field), field)
        if self.source_message_id is not None:
            _required_text(self.source_message_id, "source_message_id")
        if not _SHA256.fullmatch(self.request_sha256):
            raise RoutePolicyError("request_sha256 must be a lowercase SHA-256 hex digest")

        risk = _required_text(self.risk, "risk").lower()
        complexity = _required_text(self.complexity, "complexity").lower()
        if risk not in _RISKS:
            raise RoutePolicyError(f"unknown risk: {self.risk!r}")
        if complexity not in _COMPLEXITIES:
            raise RoutePolicyError(f"unknown complexity: {self.complexity!r}")
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "complexity", complexity)

        object.__setattr__(self, "requested_effects", _text_tuple(self.requested_effects, "requested_effects"))
        object.__setattr__(self, "forbidden_effects", _text_tuple(self.forbidden_effects, "forbidden_effects"))
        for field in (
            "external_effect", "sensitive_scope", "evidence_conflict", "uncertainty",
            "sampling_selected", "supervisor_approved", "counterpoint_route_ready",
        ):
            if not isinstance(getattr(self, field), bool):
                raise RoutePolicyError(f"{field} must be a boolean")

    def to_audit_dict(self) -> dict[str, Any]:
        """Return bounded metadata only; never include prompt or model transcript."""
        return {
            "workflow_version": self.workflow_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "idempotency_key": self.idempotency_key,
            "source": self.source,
            "source_message_id": self.source_message_id,
            "request_sha256": self.request_sha256,
            "objective_ref": self.objective_ref,
            "scope_ref": self.scope_ref,
            "risk": self.risk,
            "complexity": self.complexity,
            "requested_effects": list(self.requested_effects),
            "forbidden_effects": list(self.forbidden_effects),
            "external_effect": self.external_effect,
            "sensitive_scope": self.sensitive_scope,
            "evidence_conflict": self.evidence_conflict,
            "uncertainty": self.uncertainty,
            "sampling_selected": self.sampling_selected,
            "supervisor_approved": self.supervisor_approved,
            "counterpoint_route_ready": self.counterpoint_route_ready,
        }


@dataclass(frozen=True)
class DispatchOutcome:
    """One admission decision correlated to one request envelope."""

    run_id: str
    task_id: str
    idempotency_key: str
    request_sha256: str
    decision: DispatchDecision
    reason: str
    next_actor: str
    replayed: bool = False

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "idempotency_key": self.idempotency_key,
            "request_sha256": self.request_sha256,
            "decision": self.decision.value,
            "reason": self.reason,
            "next_actor": self.next_actor,
            "replayed": self.replayed,
        }


class SingleEntryDispatcher:
    """In-memory admission boundary for one process or isolated rehearsal.

    Durable queue persistence is intentionally a later layer. This class still
    enforces the most important local invariant: a request key cannot create a
    second decision, and the same key cannot be reused for different bytes.
    """

    def __init__(self) -> None:
        self._admissions: dict[str, DispatchOutcome] = {}

    def admit(self, request: RequestEnvelope) -> DispatchOutcome:
        if not isinstance(request, RequestEnvelope):
            raise RoutePolicyError("request must be a RequestEnvelope")

        prior = self._admissions.get(request.idempotency_key)
        if prior is not None:
            if prior.request_sha256 != request.request_sha256:
                raise DispatchConflict(
                    "idempotency key is already bound to a different request hash"
                )
            return replace(prior, replayed=True)

        outcome = self._decide(request)
        self._admissions[request.idempotency_key] = outcome
        return outcome

    @staticmethod
    def _decide(request: RequestEnvelope) -> DispatchOutcome:
        requested = set(request.requested_effects)
        forbidden = set(request.forbidden_effects)
        common: dict[str, Any] = {
            "run_id": request.run_id,
            "task_id": request.task_id,
            "idempotency_key": request.idempotency_key,
            "request_sha256": request.request_sha256,
        }

        if requested.intersection(forbidden):
            return DispatchOutcome(
                **common,
                decision=DispatchDecision.BLOCKED,
                reason="forbidden_effect",
                next_actor="none",
            )

        if request.external_effect and not request.supervisor_approved:
            return DispatchOutcome(
                **common,
                decision=DispatchDecision.HUMAN_REVIEW,
                reason="approval_required",
                next_actor="human",
            )

        if request.sensitive_scope:
            trigger_reason = "sensitive_scope"
            counterpoint_required = True
        else:
            trigger = decide_counterpoint(
                risk=request.risk,
                complexity=request.complexity,
                external_effect=request.external_effect,
                evidence_conflict=request.evidence_conflict,
                uncertainty=request.uncertainty,
                sampling_selected=request.sampling_selected,
            )
            trigger_reason = trigger.reason
            counterpoint_required = trigger.required

        if counterpoint_required:
            if not request.counterpoint_route_ready:
                return DispatchOutcome(
                    **common,
                    decision=DispatchDecision.HUMAN_REVIEW,
                    reason="no_independent_route",
                    next_actor="human",
                )
            return DispatchOutcome(
                **common,
                decision=DispatchDecision.COUNTERPOINT,
                reason=trigger_reason,
                next_actor="counterpoint_workflow",
            )

        return DispatchOutcome(
            **common,
            decision=DispatchDecision.DIRECT,
            reason=trigger_reason,
            next_actor="normal_executor",
        )


__all__ = [
    "DispatchConflict",
    "DispatchDecision",
    "DispatchOutcome",
    "RequestEnvelope",
    "SingleEntryDispatcher",
]
