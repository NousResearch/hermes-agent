from __future__ import annotations
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, asdict, fields, replace
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Callable
import pytest
from agent import dynamic_orchestration as orchestration
from scripts import bench_dynamic_orchestration as benchmark
from agent.dynamic_orchestration import (
    AcceptanceThresholdV1,
    AttemptState,
    AttemptStateEvent,
    AuditedModelJustification,
    CandidateEvaluation,
    CandidateScoreV1,
    CompensationEscalationV1,
    CredentialState,
    CredentialStateEvent,
    DecisionRelation,
    DomainValidationError,
    ErrorKind,
    EligibilityDisposition,
    EscalationAction,
    IndependentReviewAttestationV1,
    InitialSelectionTriggerV1,
    QualityCompensationPlanV1,
    ReservationState,
    ReservationStateEvent,
    ReviewState,
    ReviewStateEvent,
    RouteState,
    RouteStateEvent,
    RouteDecisionV1,
    RouteEligibilityFactsV1,
    RouteV1,
    RuntimeErrorClassificationV1,
    TaskEnvelope,
    TaskState,
    TaskStateEvent,
    validate_attempt_transition,
    validate_credential_transition,
    validate_reservation_transition,
    validate_review_transition,
    validate_route_transition,
    validate_task_transition,
    evaluate_route_eligibility,
    replan_after_capacity_exhaustion,
    score_eligible_candidates,
    CapacityObservationV1,
    CapacityValueV1,
    PoolCapacityV1,
    CapacityViewV1,
    build_capacity_view,
    CapacityDimension,
    CapacityFreshness,
    CapacityConfidence,
    CapacityCompleteness,
    CapacityConflictDisposition,
    ExecutionEnvelopeV1,
    ExecutionIdentityAttestationV1,
    ExecutionVerificationAttestationV1,
    ExecutionOutcomeV1,
    ExecutionOutcome,
    ExecutionDispatchState,
)
from agent.dynamic_orchestration.validation import (
    _MAX_GRAPH_DEPTH as _MAX_GRAPH_DEPTH,
    _MAX_GRAPH_NODES as _MAX_GRAPH_NODES,
    _MAX_MAPPING_FIELDS as _MAX_MAPPING_FIELDS,
    _MAX_TEXT_COLLECTION_ITEMS as _MAX_TEXT_COLLECTION_ITEMS,
    _mapping_snapshot as _mapping_snapshot,
    _reject_sensitive as _reject_sensitive,
)
__all__ = [
    'json',
    'os',
    'subprocess',
    'sys',
    'Iterator',
    'Mapping',
    'FrozenInstanceError',
    'asdict',
    'fields',
    'replace',
    'datetime',
    'timezone',
    'permutations',
    'Path',
    'Any',
    'Callable',
    'pytest',
    'orchestration',
    'benchmark',
    'AcceptanceThresholdV1',
    'AttemptState',
    'AttemptStateEvent',
    'AuditedModelJustification',
    'CandidateEvaluation',
    'CandidateScoreV1',
    'CompensationEscalationV1',
    'CredentialState',
    'CredentialStateEvent',
    'DecisionRelation',
    'DomainValidationError',
    'ErrorKind',
    'EligibilityDisposition',
    'EscalationAction',
    'IndependentReviewAttestationV1',
    'InitialSelectionTriggerV1',
    'QualityCompensationPlanV1',
    'ReservationState',
    'ReservationStateEvent',
    'ReviewState',
    'ReviewStateEvent',
    'RouteState',
    'RouteStateEvent',
    'RouteDecisionV1',
    'RouteEligibilityFactsV1',
    'RouteV1',
    'RuntimeErrorClassificationV1',
    'TaskEnvelope',
    'TaskState',
    'TaskStateEvent',
    'validate_attempt_transition',
    'validate_credential_transition',
    'validate_reservation_transition',
    'validate_review_transition',
    'validate_route_transition',
    'validate_task_transition',
    'evaluate_route_eligibility',
    'replan_after_capacity_exhaustion',
    'score_eligible_candidates',
    'CapacityObservationV1',
    'CapacityValueV1',
    'PoolCapacityV1',
    'CapacityViewV1',
    'build_capacity_view',
    'CapacityDimension',
    'CapacityFreshness',
    'CapacityConfidence',
    'CapacityCompleteness',
    'CapacityConflictDisposition',
    'ExecutionEnvelopeV1',
    'ExecutionIdentityAttestationV1',
    'ExecutionVerificationAttestationV1',
    'ExecutionOutcomeV1',
    'ExecutionOutcome',
    'ExecutionDispatchState',
    'FIXTURES',
    'UNSELECTED_ROUTE_ID',
    'route',
    'scored_candidate',
    'canonical_dataclass_json',
    'INDEPENDENT_REVIEW_ROUTE',
    'INDEPENDENT_REVIEW_ROUTE_ID',
    'classification',
    'decision_metadata',
    'replan',
    'task_payload',
    'IterationBombList',
    'HugePrefixMapping',
    'HostileMapping',
    'HOSTILE_MAPPING_HOOKS',
    'HOSTILE_MAPPING_MARKER',
    'nested_lists',
    'nested_dicts',
    '_MAX_GRAPH_DEPTH',
    '_MAX_GRAPH_NODES',
    '_MAX_MAPPING_FIELDS',
    '_MAX_TEXT_COLLECTION_ITEMS',
    '_mapping_snapshot',
    '_reject_sensitive',
    'valid_plan',
    'audit',
    'ELIGIBILITY_GATES',
    'eligibility_facts',
    'capacity_observation',
    'capacity_view',
    'execution_authority',
    'execution_payload',
    'reservation_payload',
    'capacity_reservation',
    'breaker_payload',
    'ExplosiveString',
    'HashBomb',
    'DeepcopyBomb',
    '_invoke_untrusted_label_boundary',
    'ExplosiveNumber',
    '_invoke_public_numeric_boundary',
    '_invoke_public_status_boundary',
    '_task_with_context_bounds',
    '_invoke_public_integer_boundary',
]
FIXTURES = Path(__file__).parents[2] / "specs" / "001-dynamic-orchestration" / "fixtures" / "route-v1"
UNSELECTED_ROUTE_ID = "route-v1:" + "f" * 64
def route(**overrides: object) -> RouteV1:
    values: dict[str, object] = {
        "provider": "OpenAI",
        "product": "ChatGPT",
        "surface": "API",
        "account_id": "café",
        "billing_pool_id": "team-a",
        "quota_pool_id": "team-a",
        "model": "GPT-5",
        "endpoint": "HTTPS://API.EXAMPLE.COM:443/v1/",
        "region": "US",
    }
    values.update(overrides)
    return RouteV1.from_mapping(values)
def scored_candidate(candidate_route: RouteV1, score: float = 1.0) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_route,
        True,
        score=score,
        score_factors=("quality",),
    )
def canonical_dataclass_json(value: object) -> str:
    return json.dumps(
        asdict(value),  # type: ignore[arg-type]
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
INDEPENDENT_REVIEW_ROUTE = route(
    provider="Independent",
    product="Review",
    account_id="reviewer-account",
    billing_pool_id="review-billing-pool",
    quota_pool_id="review-quota-pool",
    model="review-model",
    endpoint="https://review.example.com/v1",
)
INDEPENDENT_REVIEW_ROUTE_ID = INDEPENDENT_REVIEW_ROUTE.route_id
def classification(
    attempted: RouteV1,
    quota_pool_id: str | None = None,
    billing_pool_id: str | None = None,
) -> RuntimeErrorClassificationV1:
    return RuntimeErrorClassificationV1(
        kind=ErrorKind.CAPACITY_EXHAUSTED,
        source="typed-runtime-error",
        attempted_route_id=attempted.route_id,
        quota_pool_id=quota_pool_id or attempted.quota_pool_id,
        billing_pool_id=billing_pool_id,
        classified_at="2026-07-26T17:00:00Z",
    )
def decision_metadata() -> dict[str, object]:
    return {
        "created_at": "2026-07-26T17:00:01Z",
        "policy_version": "policy/v1",
        "router_version": "router/pure-v1",
        "capacity_view_id": "capacity-view:test",
        "effort": "E2",
        "verification": "V2",
    }
def replan(**kwargs: Any):
    verification = str(kwargs.pop("task_verification_minimum", "V0"))
    kwargs.setdefault(
        "trusted_reviewer_routes",
        {INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE},
    )
    kwargs.setdefault(
        "trusted_execution_routes",
        {"review-execution": INDEPENDENT_REVIEW_ROUTE},
    )
    kwargs.setdefault(
        "trusted_execution_evidence",
        {"review-execution": ("evidence:test-run",)},
    )
    kwargs.setdefault("trusted_evidence_refs", ("evidence:test-run",))
    kwargs.setdefault("trusted_threshold_results", {"evidence:test-run": True})
    kwargs.setdefault("parent_decision_id", f"parent-{kwargs.get('decision_id', 'decision')}")
    metadata = decision_metadata()
    metadata["verification"] = verification
    task_id = str(kwargs.get("task_id", "task-1"))
    task_policy_version = str(metadata["policy_version"])
    task_effort = str(metadata["effort"])
    kwargs.setdefault(
        "trusted_task",
        TaskEnvelope.from_mapping(
            task_payload(
                task_id=task_id,
                policy_version=task_policy_version,
                effort=task_effort,
                verification={
                    "minimum": verification,
                    "independent_required": int(verification[1:]) >= 2,
                    "human_gate_required": verification == "V4",
                },
            )
        ),
    )
    return replan_after_capacity_exhaustion(**metadata, **kwargs)
def task_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "task-envelope/v1",
        "task_id": "task-1",
        "objective": "contract",
        "deliverables": ["module"],
        "capabilities_required": ["filesystem.write"],
        "tools_allowed": ["patch"],
        "permissions_required": ["repository.write"],
        "context": {
            "classification": "internal",
            "max_tokens": 900,
            "token_count": 12,
            "allowed_sources": ["repository"],
        },
        "privacy": {
            "classification": "internal",
            "outbound_allowed": False,
            "retention": "ephemeral",
        },
        "risk": {
            "level": "medium",
            "reversibility": "reversible",
            "impact": "repository-only",
        },
        "effort": "E2",
        "budget": {
            "currency": "USD",
            "paid_allowed": False,
            "soft_cap": 0,
            "hard_cap": 0,
        },
        "verification": {
            "minimum": "V2",
            "independent_required": True,
            "human_gate_required": False,
        },
        "policy_version": "policy/v1",
    }
    values.update(overrides)
    return values
class IterationBombList(list[object]):
    """Fails if a boundary iterates before checking concrete cardinality."""

    def __iter__(self):
        raise AssertionError("ITERATED_BEFORE_CARDINALITY_REJECTION")
class HugePrefixMapping(Mapping[str, object]):
    """Advertise a huge size while permitting only the bounded key prefix."""

    def __init__(self) -> None:
        self.keys_yielded = 0
        self.values_read = 0

    def __len__(self) -> int:
        return 1_000_000_000

    def __iter__(self) -> Iterator[str]:
        for index in range(129):
            self.keys_yielded += 1
            yield f"field_{index}"
        raise AssertionError("mapping key scan exceeded its bounded prefix")

    def __getitem__(self, key: str) -> object:
        self.values_read += 1
        raise AssertionError(f"oversized mapping value must not be read: {key}")
HOSTILE_MAPPING_HOOKS = ("len", "iter", "getitem", "items", "get")
HOSTILE_MAPPING_MARKER = "HOSTILE_MAPPING_"
class HostileMapping(Mapping):
    """A custom ``Mapping`` whose one targeted container hook raises.

    Public ``from_mapping`` boundaries must snapshot decoded mappings and turn a
    hostile adapter's ``RuntimeError``/``TypeError``/``ValueError`` into a stable
    ``DomainValidationError`` rather than leaking it. Hooks other than ``mode``
    delegate to the ``collections.abc.Mapping`` mixin so iteration/item access
    genuinely reach ``__iter__``/``__getitem__`` exactly as a real adapter would.
    """

    def __init__(
        self,
        mode: str,
        *,
        exc: type[BaseException] = RuntimeError,
        base: Mapping[str, object] | None = None,
    ) -> None:
        self.mode = mode
        self.exc = exc
        self._base = dict(base or {})

    def __len__(self) -> int:
        if self.mode == "len":
            raise self.exc(f"{HOSTILE_MAPPING_MARKER}LEN")
        return len(self._base)

    def __iter__(self) -> Iterator[str]:
        if self.mode == "iter":
            raise self.exc(f"{HOSTILE_MAPPING_MARKER}ITER")
        return iter(self._base)

    def __getitem__(self, key: str) -> object:
        if self.mode == "getitem":
            raise self.exc(f"{HOSTILE_MAPPING_MARKER}GETITEM")
        return self._base[key]

    def items(self):  # type: ignore[override]
        if self.mode == "items":
            raise self.exc(f"{HOSTILE_MAPPING_MARKER}ITEMS")
        return super().items()

    def get(self, key, default=None):  # type: ignore[override]
        if self.mode == "get":
            raise self.exc(f"{HOSTILE_MAPPING_MARKER}GET")
        return super().get(key, default)
def nested_lists(depth: int, leaf: object = "leaf") -> object:
    """Return ``depth`` single-element lists wrapped around ``leaf``.

    ``nested_lists(0)`` is the bare leaf; each level adds one list, so the leaf
    is walked at graph depth ``depth`` by the sensitive-value scanner.
    """

    value: object = leaf
    for _ in range(depth):
        value = [value]
    return value
def nested_dicts(depth: int, leaf: object = "leaf") -> object:
    """Return ``depth`` single-key dicts wrapped around ``leaf``."""

    value: object = leaf
    for _ in range(depth):
        value = {"nested": value}
    return value
def valid_plan(
    decision_id: str,
    prior_route: RouteV1 | str,
    selected_route: RouteV1 | str,
    **overrides: object,
) -> QualityCompensationPlanV1:
    reviewed_execution_id = str(overrides.pop("reviewed_execution_id", "attempt-1"))
    prior_route_id = prior_route.route_id if type(prior_route) is RouteV1 else prior_route
    selected_route_id = selected_route.route_id if type(selected_route) is RouteV1 else selected_route
    values: dict[str, object] = {
        "plan_id": "plan-1",
        "decision_id": decision_id,
        "prior_route_id": prior_route_id,
        "selected_route_id": selected_route_id,
        "prior_quota_pool_id": (
            prior_route.quota_pool_id if type(prior_route) is RouteV1 else "unresolved-prior-quota"
        ),
        "prior_billing_pool_id": (
            prior_route.billing_pool_id if type(prior_route) is RouteV1 else "unresolved-prior-billing"
        ),
        "selected_quota_pool_id": (
            selected_route.quota_pool_id
            if type(selected_route) is RouteV1
            else "unresolved-selected-quota"
        ),
        "selected_billing_pool_id": (
            selected_route.billing_pool_id
            if type(selected_route) is RouteV1
            else "unresolved-selected-billing"
        ),
        "trigger_kind": "capacity_exhausted",
        "quality_delta_codes": ("different_model_family",),
        "required_verification": "V2",
        "independence_required": True,
        "required_reviewers": ("review-route-independent",),
        "review_attestations": (
            IndependentReviewAttestationV1(
                reviewer="review-route-independent",
                route_id=INDEPENDENT_REVIEW_ROUTE_ID,
                quota_pool_id="review-quota-pool",
                billing_pool_id="review-billing-pool",
                reviewed_execution_id=reviewed_execution_id,
                execution_id="review-execution",
                evidence_ref="evidence:test-run",
            ),
        ),
        "acceptance_thresholds": (
            AcceptanceThresholdV1(
                metric="focused_tests",
                operator=">=",
                value=1,
                unit="suite",
                evidence_required=True,
                evidence_ref="evidence:test-run",
                met=True,
            ),
        ),
        "escalation": CompensationEscalationV1(
            on_unmet=EscalationAction.BLOCK_DISPATCH,
            owner="policy-owner",
        ),
        "evidence_refs": ("evidence:test-run",),
        "created_at": "2026-07-26T17:00:00Z",
        "policy_version": "policy/v1",
    }
    values.update(overrides)
    return QualityCompensationPlanV1(**values)
def audit(identity_field: str) -> dict[str, str]:
    return {
        identity_field: "entity-1",
        "actor": "policy-engine",
        "timestamp": "2026-07-26T17:00:00Z",
        "reason": "contract transition",
        "correlation_id": "corr-1",
    }
ELIGIBILITY_GATES = (
    "identity_policy",
    "privacy_permission",
    "capability_tool",
    "context",
    "freshness_confidence",
    "budget",
    "breaker_cooldown",
    "concurrency_reservation",
)
def eligibility_facts(candidate_route: RouteV1, **overrides: object) -> RouteEligibilityFactsV1:
    values: dict[str, object] = {
        gate: EligibilityDisposition.PASS for gate in ELIGIBILITY_GATES
    }
    values.update(overrides)
    return RouteEligibilityFactsV1(route=candidate_route, **values)  # type: ignore[arg-type]
def capacity_observation(
    *,
    observation_id: str = "obs-1",
    quota_pool_id: str = "pool-a",
    source: str = "provider-a",
    freshness: str = "fresh",
    confidence: float | str = 0.9,
    value: float | None = 100,
    unit: str | None = "requests",
    captured_at: str = "2026-07-26T17:00:00Z",
    valid_until: str = "2026-07-26T18:00:00Z",
) -> CapacityObservationV1:
    return CapacityObservationV1(
        observation_id=observation_id,
        quota_pool_id=quota_pool_id,
        metric="remaining",
        source=source,
        captured_at=captured_at,
        valid_until=valid_until,
        freshness=freshness,
        confidence=confidence,
        is_estimated=False,
        value=value,
        unit=unit,
    )
def capacity_view(*observations: CapacityObservationV1, **kwargs: object) -> CapacityViewV1:
    values: dict[str, object] = {
        "view_id": "view-1",
        "built_at": "2026-07-26T17:30:00Z",
        "observations": observations,
        "conflict_disposition": CapacityConflictDisposition.EXCLUDE,
        "canonical_units": {"remaining": "requests"},
    }
    values.update(kwargs)
    return build_capacity_view(**values)  # type: ignore[arg-type]
def execution_authority(
    *,
    verification: str = "V0",
    reservation_id: str | None = None,
) -> tuple[TaskEnvelope, RouteDecisionV1, dict[str, RouteV1]]:
    selected = route(model="execution-authority", quota_pool_id="execution-authority")
    task = TaskEnvelope.from_mapping(task_payload(
        task_id="task-execution", effort="E2",
        verification={
            "minimum": verification,
            "independent_required": int(verification[1:]) >= 2,
            "human_gate_required": verification == "V4",
        },
    ))
    decision = RouteDecisionV1(
        decision_id="decision-execution", task_id=task.task_id, attempt_id="attempt-execution",
        created_at="2026-07-26T17:00:00Z", policy_version=task.policy_version,
        router_version="router/v1", capacity_view_id="view-execution", effort=task.effort,
        verification=verification, fallback=False, relation=DecisionRelation.INITIAL,
        candidates=(scored_candidate(selected),), selected_route_id=selected.route_id,
        trigger=InitialSelectionTriggerV1("initial-selection-trigger/v1", "initial_selection", "policy", "2026-07-26T17:00:00Z"),
        reason_codes=("initial_selection",),
        trusted_task=task, trusted_routes={selected.route_id: selected},
    )
    return task, decision, {selected.route_id: selected}
def execution_payload(task: TaskEnvelope, decision: RouteDecisionV1) -> dict[str, object]:
    return {
        "schema_version": "execution-envelope/v1", "execution_id": "exec-1", "task_id": task.task_id,
        "attempt_id": decision.attempt_id, "decision_id": decision.decision_id, "route_id": decision.selected_route_id,
        "reservation_id": None, "effort": task.effort, "verification": decision.verification,
        "workdir": None, "allowed_context_refs": (), "allowed_tools": (), "permissions": (), "timeout": 30,
        "budget": asdict(task.budget), "trace_context": None, "telemetry_consent_id": None,
        "requested_identity": decision.selected_route_id, "resolved_identity": decision.selected_route_id,
        "effective_identity": decision.selected_route_id, "dispatch_state": "PENDING", "result_evidence_refs": (),
        "normalized_error": None, "reconciliation_state": "pending",
    }
def reservation_payload(
    bound_route: RouteV1,
    *,
    reservation_id: str = "reservation-1",
    owner_attempt_id: str = "attempt-execution",
    status: str = "HELD",
    estimated_amount: object = 10,
    held_amount: object = 5,
    expires_at: object = "2026-07-26T18:00:00Z",
    version: object = 0,
) -> dict[str, object]:
    return {
        "reservation_id": reservation_id,
        "route_id": bound_route.route_id,
        "quota_pool_id": bound_route.quota_pool_id,
        "billing_pool_id": bound_route.billing_pool_id,
        "owner_attempt_id": owner_attempt_id,
        "unit": "requests",
        "estimated_amount": estimated_amount,
        "held_amount": held_amount,
        "expires_at": expires_at,
        "status": status,
        "version": version,
    }
def capacity_reservation(
    bound_route: RouteV1,
    **overrides: object,
):
    payload = reservation_payload(bound_route, **overrides)
    if payload["status"] in {"PENDING", "RELEASED", "EXPIRED", "RECONCILED"}:
        payload.setdefault("held_amount", 0)
        if "held_amount" not in overrides:
            payload["held_amount"] = 0
    return orchestration.CapacityReservationV1.from_mapping(
        payload,
        trusted_route=bound_route,
    )
def breaker_payload(
    bound_route: RouteV1,
    *,
    status: str = "CLOSED",
    cooldown_until: object = None,
    probe_budget: object = 0,
    reason_code: object = None,
    version: object = 0,
) -> dict[str, object]:
    return {
        "breaker_id": "breaker-1",
        "route_id": bound_route.route_id,
        "quota_pool_id": bound_route.quota_pool_id,
        "status": status,
        "cooldown_until": cooldown_until,
        "probe_budget": probe_budget,
        "reason_code": reason_code,
        "version": version,
    }
class ExplosiveString:
    def __init__(self) -> None:
        self.called = False

    def __str__(self) -> str:
        self.called = True
        raise RuntimeError("public boundary invoked __str__")
class HashBomb:
    def __init__(self) -> None:
        self.called = False

    def __hash__(self) -> int:
        self.called = True
        raise RuntimeError("ATTACKER_HASH_MESSAGE")
class DeepcopyBomb:
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise RuntimeError("ATTACKER_DEEPCOPY_MESSAGE")
def _invoke_untrusted_label_boundary(case: str, value: object) -> None:
    if case == "task_verification":
        orchestration.TaskVerificationV1(
            minimum=value,  # type: ignore[arg-type]
            independent_required=True,
            human_gate_required=False,
        )
    elif case == "task_effort":
        TaskEnvelope.from_mapping(task_payload(effort=value))
    elif case == "quality_verification":
        failed = route(model="label-quality-failed", quota_pool_id="label-quality-failed")
        selected = route(
            model="label-quality-selected",
            quota_pool_id="label-quality-selected",
        )
        replace(
            valid_plan("decision-label-quality", failed, selected),
            required_verification=value,  # type: ignore[arg-type]
        )
    elif case == "decision_effort":
        task, decision, routes = execution_authority()
        replace(
            decision,
            effort=value,  # type: ignore[arg-type]
            trusted_task=task,
            trusted_routes=routes,
        )
    elif case == "execution_verification":
        task, decision, routes = execution_authority()
        ExecutionEnvelopeV1.from_mapping(
            {**execution_payload(task, decision), "verification": value},
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    else:  # pragma: no cover - table completeness guard
        raise AssertionError(case)
class ExplosiveNumber:
    def __init__(self) -> None:
        self.called = False

    def __float__(self) -> float:
        self.called = True
        raise RuntimeError("public boundary invoked __float__")
def _invoke_public_numeric_boundary(case: str, value: object) -> None:
    if case == "capacity_value":
        CapacityValueV1(known=True, value=value, unit="requests")  # type: ignore[arg-type]
    elif case in {"observation_value", "observation_confidence", "observation_price"}:
        field_name = case.removeprefix("observation_")
        CapacityObservationV1.from_mapping(
            {**asdict(capacity_observation()), field_name: value}
        )
    elif case == "task_budget":
        orchestration.TaskBudgetV1(
            currency="USD",
            paid_allowed=True,
            soft_cap=value,  # type: ignore[arg-type]
        )
    elif case == "candidate_score":
        CandidateEvaluation(
            route(model="numeric-candidate"),
            True,
            score=value,  # type: ignore[arg-type]
            score_factors=("quality",),
        )
    elif case == "score":
        CandidateScoreV1(value, ("quality",))  # type: ignore[arg-type]
    elif case == "threshold":
        AcceptanceThresholdV1("quality", ">=", value)  # type: ignore[arg-type]
    elif case in {"reservation_estimated", "reservation_held"}:
        bound_route = route(model=f"numeric-{case}", quota_pool_id=f"numeric-{case}")
        field_name = case.removeprefix("reservation_") + "_amount"
        orchestration.CapacityReservationV1.from_mapping(
            {**reservation_payload(bound_route), field_name: value},
            trusted_route=bound_route,
        )
    elif case == "execution_timeout":
        task, decision, routes = execution_authority()
        ExecutionEnvelopeV1.from_mapping(
            {**execution_payload(task, decision), "timeout": value},
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    else:  # pragma: no cover - table completeness guard
        raise AssertionError(case)
def _invoke_public_status_boundary(case: str, value: object) -> None:
    if case == "candidate":
        candidate_route = route(model="status-candidate")
        candidate = CandidateEvaluation(candidate_route, True)
        CandidateEvaluation.from_mapping(
            {**asdict(candidate), "deterministic_status": value},
            route_context=candidate_route,
        )
    elif case == "relation":
        task, decision, routes = execution_authority()
        RouteDecisionV1.from_mapping(
            {**asdict(decision), "relation": value},
            trusted_task=task,
            trusted_routes=routes,
        )
    elif case == "escalation":
        CompensationEscalationV1.from_mapping(
            {"on_unmet": value, "owner": "policy-owner", "deadline": None}
        )
    elif case == "reservation":
        bound_route = route(model="status-reservation", quota_pool_id="status-reservation")
        orchestration.CapacityReservationV1.from_mapping(
            {**reservation_payload(bound_route), "status": value},
            trusted_route=bound_route,
        )
    elif case == "breaker":
        bound_route = route(model="status-breaker", quota_pool_id="status-breaker")
        orchestration.CircuitBreakerV1.from_mapping(
            {**breaker_payload(bound_route), "status": value},
            trusted_route=bound_route,
        )
    elif case == "outcome":
        ExecutionOutcomeV1.from_mapping({"outcome": value})
    elif case == "dispatch":
        task, decision, routes = execution_authority()
        ExecutionEnvelopeV1.from_mapping(
            {**execution_payload(task, decision), "dispatch_state": value},
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    else:  # pragma: no cover - table completeness guard
        raise AssertionError(case)
def _task_with_context_bounds(*, max_tokens: object, token_count: object) -> TaskEnvelope:
    return TaskEnvelope.from_mapping(
        task_payload(
            context={
                "classification": "internal",
                "max_tokens": max_tokens,
                "token_count": token_count,
                "allowed_sources": ["repository"],
            }
        )
    )
def _invoke_public_integer_boundary(case: str, value: object) -> None:
    bound_route = route(model=f"integer-{case}", quota_pool_id=f"integer-{case}")
    if case == "context_max_tokens":
        _task_with_context_bounds(max_tokens=value, token_count=0)
    elif case == "context_token_count":
        _task_with_context_bounds(max_tokens=None, token_count=value)
    elif case == "max_concurrency":
        CapacityObservationV1.from_mapping(
            {**asdict(capacity_observation()), "max_concurrency": value}
        )
    elif case == "reservation_version":
        orchestration.CapacityReservationV1.from_mapping(
            {**reservation_payload(bound_route), "version": value},
            trusted_route=bound_route,
        )
    elif case == "breaker_probe_budget":
        orchestration.CircuitBreakerV1.from_mapping(
            breaker_payload(
                bound_route,
                status="HALF_OPEN",
                probe_budget=value,
            ),
            trusted_route=bound_route,
        )
    elif case == "breaker_version":
        orchestration.CircuitBreakerV1.from_mapping(
            breaker_payload(bound_route, version=value),
            trusted_route=bound_route,
        )
    else:  # pragma: no cover - table completeness guard
        raise AssertionError(case)
