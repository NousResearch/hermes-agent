"""Decision authority validation and fail-closed policy application."""

from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

from typing import Mapping, Sequence

from .eligibility import (
    CandidateEvaluation,
)

from .quality import QualityCompensationPlanV1

from .route import (
    RouteV1,
    _revalidated_route,
    _revalidated_route_registry,
)

from .task import (
    TaskEnvelope,
)

from .validation import (
    DomainValidationError,
    _MAX_DECISION_CANDIDATES,
    _MAX_ROUTE_REGISTRY_ITEMS,
    _immutable_string_collection,
    _reject_sensitive,
    _safe_asdict,
    _task_text,
    _validated_verification_rank,
)


class DecisionRelation(str, Enum):
    INITIAL = "INITIAL"
    FALLBACK = "FALLBACK"
    REPLAN = "REPLAN"
    WAITING = "WAITING"


def _coerce_quality_plan(
    value: QualityCompensationPlanV1 | Mapping[str, object] | None,
) -> QualityCompensationPlanV1 | None:
    if value is None:
        return None
    if type(value) is QualityCompensationPlanV1:
        try:
            return QualityCompensationPlanV1.from_mapping(
                _safe_asdict(value, "quality compensation plan")
            )
        except (DomainValidationError, TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        try:
            return QualityCompensationPlanV1.from_mapping(value)
        except DomainValidationError:
            return None
    return None

def _require_scored_eligible_candidates(
    candidates: Sequence[CandidateEvaluation],
    *,
    code: str,
) -> None:
    if any(
        candidate.eligible
        and (candidate.score is None or not candidate.score_factors)
        for candidate in candidates
    ):
        raise DomainValidationError(
            code,
            "every eligible candidate requires a finite score and at least one score factor",
        )

def _validated_decision_candidates(
    raw_candidates: object,
    *,
    route_registry: Mapping[str, RouteV1],
    effort: str,
    selected_route_id: str | None,
    prior_route_id: str | None,
    relation: DecisionRelation,
) -> tuple[tuple[CandidateEvaluation, ...], tuple[CandidateEvaluation, ...]]:
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(raw_candidates, Sequence):
        raise DomainValidationError("decision.candidates_invalid", "candidates must be a collection")
    if len(raw_candidates) > _MAX_DECISION_CANDIDATES:
        raise DomainValidationError(
            "decision.candidates_invalid",
            f"candidates must contain at most {_MAX_DECISION_CANDIDATES} entries",
        )
    candidates = tuple(raw_candidates)
    if any(type(item) is not CandidateEvaluation for item in candidates):
        raise DomainValidationError("decision.candidates_invalid", "candidate DTO required")

    validated_candidates: list[CandidateEvaluation] = []
    for candidate in candidates:
        validated_route_context = route_registry.get(candidate.route_id)
        if validated_route_context is None:
            raise DomainValidationError(
                "decision.trusted_route_context_required",
                "every candidate requires trusted RouteV1 registry binding",
            )
        validated_candidates.append(
            CandidateEvaluation(
                validated_route_context,
                candidate.deterministic_status,
                candidate.rejection_codes,
                candidate.score,
                candidate.score_factors,
            )
        )
    candidates = tuple(
        sorted(validated_candidates, key=lambda candidate: candidate.route_id)
    )

    if effort == "E0" and (
        candidates
        or selected_route_id is not None
        or prior_route_id is not None
        or relation is not DecisionRelation.INITIAL
    ):
        raise DomainValidationError(
            "decision.effort_route_prohibited",
            "E0 decisions cannot evaluate or select model routes",
        )
    if effort != "E0":
        _require_scored_eligible_candidates(
            candidates,
            code="decision.candidate_score_required",
        )
    candidate_ids = tuple(candidate.route_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DomainValidationError("decision.candidates_invalid", "candidate route IDs must be unique")

    selected_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.eligible and candidate.route_id == selected_route_id
    )
    if selected_route_id is not None and len(selected_candidates) != 1:
        raise DomainValidationError(
            "decision.selected_candidate_ineligible",
            "selected route must be exactly one eligible candidate",
        )
    eligible_ranked = sorted(
        (candidate for candidate in candidates if candidate.eligible),
        key=lambda candidate: (
            -candidate.score,  # type: ignore[operator]
            candidate.route_id,
        ),
    )
    if (
        selected_route_id is not None
        and eligible_ranked
        and selected_route_id != eligible_ranked[0].route_id
    ):
        raise DomainValidationError(
            "decision.selection_rank_invalid",
            "selected route must be the highest-ranked eligible candidate",
        )
    if (
        relation is DecisionRelation.INITIAL
        and effort != "E0"
        and selected_route_id is None
    ):
        raise DomainValidationError(
            "decision.selected_candidate_required",
            "initial decision requires an eligible selected candidate",
        )
    return candidates, selected_candidates

@dataclass(frozen=True)
class _DecisionAuthorities:
    task: TaskEnvelope
    route_registry: Mapping[str, RouteV1]
    reviewer_routes: Mapping[str, RouteV1]
    approval_refs: frozenset[str]
    execution_routes: Mapping[str, RouteV1]
    execution_evidence: Mapping[str, frozenset[str]]
    evidence_refs: frozenset[str]
    threshold_results: Mapping[str, bool]

def _validated_unsealed_decision_authorities(
    *,
    trusted_reviewer_routes: Mapping[str, RouteV1] | None,
    trusted_human_approval_refs: Mapping[str, str] | None,
    trusted_execution_routes: Mapping[str, RouteV1] | None,
    trusted_execution_evidence: Mapping[str, Sequence[str]] | None,
    trusted_evidence_refs: Sequence[str] | None,
    trusted_threshold_results: Mapping[str, bool] | None,
) -> tuple[
    Mapping[str, RouteV1],
    frozenset[str],
    Mapping[str, RouteV1],
    Mapping[str, frozenset[str]],
    frozenset[str],
    Mapping[str, bool],
]:
    """Validate caller-supplied structural authority data without elevating it."""

    reviewer_routes = _revalidated_route_registry(
        trusted_reviewer_routes,
        code="decision.trusted_reviewer_context_invalid",
        location="trusted reviewer registry",
    )

    if trusted_human_approval_refs is not None and not isinstance(
        trusted_human_approval_refs,
        Mapping,
    ):
        raise DomainValidationError(
            "decision.trusted_human_approval_invalid",
            "trusted human approvals must map approval reference to task ID",
        )
    if (
        trusted_human_approval_refs is not None
        and len(trusted_human_approval_refs) > _MAX_ROUTE_REGISTRY_ITEMS
    ):
        raise DomainValidationError(
            "decision.trusted_human_approval_invalid",
            f"trusted human approvals must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} entries",
        )
    for approval_ref, approved_task_id in (trusted_human_approval_refs or {}).items():
        normalized_approval_ref = _task_text(approval_ref, "human_approval_ref")
        normalized_approved_task_id = _task_text(
            approved_task_id,
            "approved_task_id",
        )
        _reject_sensitive(
            {normalized_approval_ref: normalized_approved_task_id},
            "trusted human approvals",
        )
    # This pure/unwired phase has no external approval authority. Public
    # mappings are validated as untrusted input but can never elevate policy.
    approval_refs: frozenset[str] = frozenset()

    execution_routes: dict[str, RouteV1] = {}
    if trusted_execution_routes is not None:
        if not isinstance(trusted_execution_routes, Mapping):
            raise DomainValidationError(
                "decision.trusted_execution_context_invalid",
                "trusted execution registry must be a mapping",
            )
        if len(trusted_execution_routes) > _MAX_ROUTE_REGISTRY_ITEMS:
            raise DomainValidationError(
                "decision.trusted_execution_context_invalid",
                f"trusted execution registry must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} entries",
            )
        for execution_id, execution_route in trusted_execution_routes.items():
            normalized_execution_id = _task_text(execution_id, "execution_id")
            _reject_sensitive(
                {normalized_execution_id: None},
                "trusted execution registry",
            )
            execution_routes[normalized_execution_id] = _revalidated_route(
                execution_route,
                "trusted execution route",
            )

    execution_evidence: dict[str, frozenset[str]] = {}
    if trusted_execution_evidence is not None:
        if not isinstance(trusted_execution_evidence, Mapping):
            raise DomainValidationError(
                "decision.trusted_execution_context_invalid",
                "trusted execution evidence must be a mapping",
            )
        if len(trusted_execution_evidence) > _MAX_ROUTE_REGISTRY_ITEMS:
            raise DomainValidationError(
                "decision.trusted_execution_context_invalid",
                f"trusted execution evidence must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} entries",
            )
        for execution_id, execution_refs in trusted_execution_evidence.items():
            normalized_execution_id = _task_text(execution_id, "execution_id")
            normalized_execution_refs = _immutable_string_collection(
                execution_refs,
                "execution_evidence_refs",
                require_nonempty=True,
            )
            _reject_sensitive(
                {normalized_execution_id: normalized_execution_refs},
                "trusted execution evidence",
            )
            execution_evidence[normalized_execution_id] = frozenset(
                normalized_execution_refs
            )

    normalized_evidence_refs = _immutable_string_collection(
        trusted_evidence_refs or (),
        "trusted_evidence_refs",
    )
    _reject_sensitive(normalized_evidence_refs, "trusted evidence references")
    evidence_refs = frozenset(normalized_evidence_refs)
    threshold_results: dict[str, bool] = {}
    if trusted_threshold_results is not None:
        if not isinstance(trusted_threshold_results, Mapping):
            raise DomainValidationError(
                "decision.trusted_threshold_context_invalid",
                "trusted threshold results must be a mapping",
            )
        if len(trusted_threshold_results) > _MAX_ROUTE_REGISTRY_ITEMS:
            raise DomainValidationError(
                "decision.trusted_threshold_context_invalid",
                f"trusted threshold results must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} entries",
            )
        for evidence_ref, result in trusted_threshold_results.items():
            normalized_ref = _task_text(evidence_ref, "threshold_evidence_ref")
            if type(result) is not bool:
                raise DomainValidationError(
                    "decision.trusted_threshold_context_invalid",
                    "trusted threshold results must be boolean",
                )
            _reject_sensitive(
                {normalized_ref: result},
                "trusted threshold results",
            )
            threshold_results[normalized_ref] = result
    return (
        reviewer_routes,
        approval_refs,
        execution_routes,
        execution_evidence,
        evidence_refs,
        threshold_results,
    )

def _validated_decision_authorities(
    decision: "RouteDecisionV1",
    *,
    trusted_task: TaskEnvelope | None,
    trusted_routes: Mapping[str, RouteV1] | None,
    trusted_reviewer_routes: Mapping[str, RouteV1] | None,
    trusted_human_approval_refs: Mapping[str, str] | None,
    trusted_execution_routes: Mapping[str, RouteV1] | None,
    trusted_execution_evidence: Mapping[str, Sequence[str]] | None,
    trusted_evidence_refs: Sequence[str] | None,
    trusted_threshold_results: Mapping[str, bool] | None,
) -> _DecisionAuthorities:
    if type(trusted_task) is not TaskEnvelope:
        raise DomainValidationError(
            "decision.trusted_task_context_required",
            "route decisions require trusted TaskEnvelope context",
        )
    validated_task = TaskEnvelope.from_mapping(
        _safe_asdict(trusted_task, "trusted task")
    )
    _, decision_verification_rank = _validated_verification_rank(
        decision.verification,
        code="decision.verification_invalid",
        message="verification must be V0 through V4",
    )
    _, task_verification_rank = _validated_verification_rank(
        validated_task.verification.minimum,
        code="task.verification_invalid",
        message="verification.minimum must be V0 through V4",
    )
    if (
        decision.task_id != validated_task.task_id
        or decision.policy_version != validated_task.policy_version
        or decision.effort != validated_task.effort
        or decision_verification_rank < task_verification_rank
    ):
        raise DomainValidationError(
            "decision.trusted_task_mismatch",
            "decision identity, policy, effort, and verification must match trusted task authority",
        )
    if not isinstance(trusted_routes, Mapping):
        raise DomainValidationError(
            "decision.trusted_route_context_required",
            "route decisions require a trusted RouteV1 registry",
        )
    route_registry = _revalidated_route_registry(
        trusted_routes,
        code="decision.trusted_route_context_invalid",
        location="trusted route registry",
    )
    (
        reviewer_routes,
        approval_refs,
        execution_routes,
        execution_evidence,
        evidence_refs,
        threshold_results,
    ) = _validated_unsealed_decision_authorities(
        trusted_reviewer_routes=trusted_reviewer_routes,
        trusted_human_approval_refs=trusted_human_approval_refs,
        trusted_execution_routes=trusted_execution_routes,
        trusted_execution_evidence=trusted_execution_evidence,
        trusted_evidence_refs=trusted_evidence_refs,
        trusted_threshold_results=trusted_threshold_results,
    )
    return _DecisionAuthorities(
        task=validated_task,
        route_registry=route_registry,
        reviewer_routes=reviewer_routes,
        approval_refs=approval_refs,
        execution_routes=execution_routes,
        execution_evidence=execution_evidence,
        evidence_refs=evidence_refs,
        threshold_results=threshold_results,
    )

def _apply_decision_policy(
    decision: "RouteDecisionV1",
    *,
    authorities: _DecisionAuthorities,
    validated_prior_route: RouteV1 | None,
    selected_candidates: Sequence[CandidateEvaluation],
) -> None:
    if decision.relation in {DecisionRelation.FALLBACK, DecisionRelation.REPLAN}:
        if (
            not decision.selected_route_id
            or not decision.prior_route_id
            or not decision.parent_decision_id
        ):
            raise DomainValidationError(
                "decision.route_identity_required",
                "fallback/replan requires parent, prior, and selected route identity",
            )
        plan = _coerce_quality_plan(decision.quality_compensation_plan)
        object.__setattr__(decision, "quality_compensation_plan", plan)
        # A syntactically valid plan remains structural unsealed data in this
        # phase. No caller review/evidence/threshold registry can satisfy it;
        # a future opaque sealed review artifact is required.
        object.__setattr__(
            decision,
            "policy_status",
            "ACTIVATION_BLOCKED_QUALITY_COMPENSATION",
        )
        object.__setattr__(
            decision,
            "activation_block_reason",
            "quality_compensation_insufficient",
        )
        if "quality_compensation_insufficient" not in decision.reason_codes:
            object.__setattr__(
                decision,
                "reason_codes",
                decision.reason_codes + ("quality_compensation_insufficient",),
            )
        if decision.reservation_id is not None:
            raise DomainValidationError(
                "decision.blocked_reservation",
                "activation-blocked fallback must not claim a reservation",
            )
        return

    if decision.relation is DecisionRelation.WAITING:
        if decision.selected_route_id is not None:
            raise DomainValidationError(
                "decision.waiting_selected_route",
                "waiting must not select a route",
            )
        if not decision.recheck_evidence:
            raise DomainValidationError(
                "replan.recheck_evidence_required",
                "waiting requires recheck evidence",
            )
        if any(candidate.eligible for candidate in decision.candidates):
            raise DomainValidationError(
                "decision.waiting_eligible_candidate",
                "waiting requires every retained candidate to be ineligible",
            )
        if decision.reservation_id is not None:
            raise DomainValidationError(
                "decision.waiting_reservation",
                "waiting must not hold a reservation",
            )
        object.__setattr__(decision, "policy_status", "WAITING_FOR_CAPACITY")
        return

    if decision.effort == "E0":
        object.__setattr__(decision, "policy_status", "NO_ROUTE_REQUIRED")
    elif authorities.task.verification.independent_required:
        object.__setattr__(
            decision,
            "policy_status",
            "ACTIVATION_BLOCKED_INDEPENDENT_REVIEW",
        )
        object.__setattr__(
            decision,
            "activation_block_reason",
            "independent_review_required",
        )
    elif (
        authorities.task.verification.human_gate_required
        and not authorities.approval_refs
    ):
        object.__setattr__(decision, "policy_status", "ACTIVATION_BLOCKED_HUMAN_GATE")
        object.__setattr__(decision, "activation_block_reason", "human_gate_required")
        if "human_gate_required" not in decision.reason_codes:
            object.__setattr__(
                decision,
                "reason_codes",
                decision.reason_codes + ("human_gate_required",),
            )
    else:
        object.__setattr__(
            decision,
            "policy_status",
            "ACTIVATION_BLOCKED_EXTERNAL_AUTHORITY_UNAVAILABLE",
        )
        object.__setattr__(
            decision,
            "activation_block_reason",
            "external_authority_store_unimplemented",
        )
    if not decision.dispatchable and decision.reservation_id is not None:
        raise DomainValidationError(
            "decision.blocked_reservation",
            "non-dispatchable initial decision must not claim a reservation",
        )
