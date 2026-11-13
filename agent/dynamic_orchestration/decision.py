"""Route decisions and optimized capacity-exhaustion replanning."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass


from typing import Mapping, Sequence

from .eligibility import (
    CandidateEvaluation,
    InitialSelectionTriggerV1,
    RuntimeErrorClassificationV1,
)

from .quality import (
    QualityCompensationPlanV1,
    _raise_candidate,
)

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
    _ACTIVATION_BLOCK_REASON_LABELS,
    _EFFORT_LABELS,
    _MAX_DECISION_CANDIDATES,
    _POLICY_STATUS_LABELS,
    _PROHIBITED_FIELD_NAMES,
    _SENSITIVE_VALUE_PATTERN,
    _immutable_string_collection,
    _mapping_snapshot,
    _parse_exact_enum,
    _reject_sensitive,
    _safe_asdict,
    _stable_identifier_collection,
    _task_text,
    _validated_exact_label,
    _validated_mapping_keys,
    _validated_route_id,
    _validated_verification_rank,
)
from .policy import (
    DecisionRelation,
    _apply_decision_policy,
    _coerce_quality_plan,
    _require_scored_eligible_candidates,
    _validated_decision_authorities,
    _validated_decision_candidates,
    _validated_unsealed_decision_authorities,
)


@dataclass(frozen=True)
class RouteDecisionV1:
    decision_id: str
    task_id: str
    attempt_id: str
    created_at: str
    policy_version: str
    router_version: str
    capacity_view_id: str
    effort: str
    verification: str
    fallback: bool
    relation: DecisionRelation
    candidates: tuple[CandidateEvaluation, ...]
    selected_route_id: str | None
    trigger: InitialSelectionTriggerV1 | RuntimeErrorClassificationV1
    reason_codes: tuple[str, ...]
    prior_route_id: str | None = None
    parent_decision_id: str | None = None
    reservation_id: str | None = None
    quality_compensation_plan: QualityCompensationPlanV1 | None = None
    recheck_evidence: tuple[str, ...] = ()
    schema_version: str = "route-decision/v1"
    trusted_task: InitVar[TaskEnvelope | None] = None
    trusted_routes: InitVar[Mapping[str, RouteV1] | None] = None
    trusted_reviewer_routes: InitVar[Mapping[str, RouteV1] | None] = None
    trusted_prior_route: InitVar[RouteV1 | None] = None
    trusted_human_approval_refs: InitVar[Mapping[str, str] | None] = None
    trusted_execution_routes: InitVar[Mapping[str, RouteV1] | None] = None
    trusted_execution_evidence: InitVar[Mapping[str, Sequence[str]] | None] = None
    trusted_evidence_refs: InitVar[Sequence[str] | None] = None
    trusted_threshold_results: InitVar[Mapping[str, bool] | None] = None
    policy_status: str = field(default="", init=False)
    activation_block_reason: str | None = field(default=None, init=False)

    def __post_init__(
        self,
        trusted_task: TaskEnvelope | None,
        trusted_routes: Mapping[str, RouteV1] | None,
        trusted_reviewer_routes: Mapping[str, RouteV1] | None,
        trusted_prior_route: RouteV1 | None,
        trusted_human_approval_refs: Mapping[str, str] | None,
        trusted_execution_routes: Mapping[str, RouteV1] | None,
        trusted_execution_evidence: Mapping[str, Sequence[str]] | None,
        trusted_evidence_refs: Sequence[str] | None,
        trusted_threshold_results: Mapping[str, bool] | None,
    ) -> None:
        if self.schema_version != "route-decision/v1":
            raise DomainValidationError("decision.schema_invalid", "schema_version must be route-decision/v1")
        for name in (
            "decision_id",
            "task_id",
            "attempt_id",
            "created_at",
            "policy_version",
            "router_version",
            "capacity_view_id",
        ):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        if self.selected_route_id is not None:
            object.__setattr__(
                self,
                "selected_route_id",
                _validated_route_id(self.selected_route_id, "selected_route_id"),
            )
        if self.prior_route_id is not None:
            object.__setattr__(
                self,
                "prior_route_id",
                _validated_route_id(self.prior_route_id, "prior_route_id"),
            )
        for name in ("parent_decision_id", "reservation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _task_text(value, name))
        object.__setattr__(
            self,
            "effort",
            _validated_exact_label(
                self.effort,
                _EFFORT_LABELS,
                code="decision.effort_invalid",
                message="effort must be E0 through E4",
            ),
        )
        verification, _ = _validated_verification_rank(
            self.verification,
            code="decision.verification_invalid",
            message="verification must be V0 through V4",
        )
        object.__setattr__(self, "verification", verification)
        authorities = _validated_decision_authorities(
            self,
            trusted_task=trusted_task,
            trusted_routes=trusted_routes,
            trusted_reviewer_routes=trusted_reviewer_routes,
            trusted_human_approval_refs=trusted_human_approval_refs,
            trusted_execution_routes=trusted_execution_routes,
            trusted_execution_evidence=trusted_execution_evidence,
            trusted_evidence_refs=trusted_evidence_refs,
            trusted_threshold_results=trusted_threshold_results,
        )
        route_registry = authorities.route_registry
        if not isinstance(self.fallback, bool) or self.fallback is not (
            self.relation is DecisionRelation.FALLBACK
        ):
            raise DomainValidationError("decision.fallback_invalid", "fallback must exactly match FALLBACK relation")
        if type(self.relation) is not DecisionRelation:
            raise DomainValidationError("decision.relation_invalid", "invalid decision relation")
        expected_trigger_type = (
            InitialSelectionTriggerV1
            if self.relation is DecisionRelation.INITIAL
            else RuntimeErrorClassificationV1
        )
        if type(self.trigger) is not expected_trigger_type:
            raise DomainValidationError(
                "decision.trigger_invalid",
                "trigger type must match the decision relation",
            )
        validated_trigger = (
            InitialSelectionTriggerV1.from_mapping(
                _safe_asdict(self.trigger, "decision trigger")
            )
            if type(self.trigger) is InitialSelectionTriggerV1
            else RuntimeErrorClassificationV1.from_mapping(
                _safe_asdict(self.trigger, "decision trigger")
            )
        )
        object.__setattr__(self, "trigger", validated_trigger)
        validated_prior_route: RouteV1 | None = None
        if self.relation in {
            DecisionRelation.FALLBACK,
            DecisionRelation.REPLAN,
            DecisionRelation.WAITING,
        }:
            if type(trusted_prior_route) is not RouteV1:
                raise DomainValidationError(
                    "decision.trusted_prior_route_required",
                    "dynamic decisions require trusted prior RouteV1 context",
                )
            validated_prior_route = _revalidated_route(
                trusted_prior_route,
                "trusted prior route",
            )
        if (
            self.relation
            in {DecisionRelation.FALLBACK, DecisionRelation.REPLAN, DecisionRelation.WAITING}
            and type(self.trigger) is RuntimeErrorClassificationV1
            and (
                self.trigger.attempted_route_id != self.prior_route_id
                or validated_prior_route is None
                or self.prior_route_id != validated_prior_route.route_id
                or self.trigger.quota_pool_id != validated_prior_route.quota_pool_id
                or self.trigger.billing_pool_id != validated_prior_route.billing_pool_id
            )
        ):
            raise DomainValidationError(
                "decision.trigger_route_mismatch",
                "fallback/replan/waiting trigger must identify the prior attempted route",
            )
        candidates, selected_candidates = _validated_decision_candidates(
            self.candidates,
            route_registry=route_registry,
            effort=self.effort,
            selected_route_id=self.selected_route_id,
            prior_route_id=self.prior_route_id,
            relation=self.relation,
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "reason_codes",
            _stable_identifier_collection(
                self.reason_codes,
                "reason_codes",
                code="decision.reason_codes_invalid",
            ),
        )
        object.__setattr__(
            self,
            "recheck_evidence",
            _immutable_string_collection(self.recheck_evidence, "recheck_evidence"),
        )

        _reject_sensitive(
            {
                "reason_codes": self.reason_codes,
                "recheck_evidence": self.recheck_evidence,
            },
            "decision",
        )

        _apply_decision_policy(
            self,
            authorities=authorities,
            validated_prior_route=validated_prior_route,
            selected_candidates=selected_candidates,
        )
        _reject_sensitive(self, "decision")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        trusted_routes: Mapping[str, RouteV1] | None = None,
        trusted_task: TaskEnvelope | None = None,
        trusted_reviewer_routes: Mapping[str, RouteV1] | None = None,
        trusted_human_approval_refs: Mapping[str, str] | None = None,
        trusted_execution_routes: Mapping[str, RouteV1] | None = None,
        trusted_execution_evidence: Mapping[str, Sequence[str]] | None = None,
        trusted_evidence_refs: Sequence[str] | None = None,
        trusted_threshold_results: Mapping[str, bool] | None = None,
    ) -> "RouteDecisionV1":
        if not isinstance(payload, Mapping):
            raise DomainValidationError(
                "decision.schema_invalid", "route decision must be a mapping"
            )
        payload = _mapping_snapshot(
            payload,
            code="decision.schema_invalid",
            location="route decision",
        )
        _reject_sensitive(payload, "decision")
        allowed = {field_.name for field_ in fields(cls)}
        payload_keys = _validated_mapping_keys(
            payload,
            code="decision.unexpected_field",
            location="route decision",
        )
        unknown = payload_keys - allowed
        if unknown:
            raise DomainValidationError(
                "decision.unexpected_field",
                f"unexpected decision fields: {sorted(unknown)}",
            )
        required_derived = {"policy_status", "activation_block_reason"}
        if not required_derived.issubset(payload):
            raise DomainValidationError(
                "decision.persisted_status_required",
                "persisted decision requires policy_status and activation_block_reason",
            )
        values = dict(payload)
        persisted_policy_status = _validated_exact_label(
            values.pop("policy_status"),
            _POLICY_STATUS_LABELS,
            code="decision.policy_status_invalid",
            message="policy_status must be a supported derived decision status",
        )
        persisted_block_reason = values.pop("activation_block_reason")
        if persisted_block_reason is not None:
            persisted_block_reason = _validated_exact_label(
                persisted_block_reason,
                _ACTIVATION_BLOCK_REASON_LABELS,
                code="decision.activation_block_reason_invalid",
                message="activation_block_reason must be a supported derived reason",
            )
        try:
            relation = values.get("relation")
            values["relation"] = _parse_exact_enum(
                relation,
                DecisionRelation,
                code="decision.relation_invalid",
                message="relation must be INITIAL, FALLBACK, REPLAN, or WAITING",
            )
            resolved_relation = values["relation"]
            prior_route: RouteV1 | None = None
            dynamic_relations = {
                DecisionRelation.FALLBACK,
                DecisionRelation.REPLAN,
                DecisionRelation.WAITING,
            }
            if type(trusted_task) is not TaskEnvelope:
                raise DomainValidationError(
                    "decision.trusted_task_context_required",
                    "persisted route decision requires trusted TaskEnvelope context",
                )
            validated_task = TaskEnvelope.from_mapping(
                _safe_asdict(trusted_task, "trusted task")
            )
            persisted_effort = _validated_exact_label(
                values.get("effort"),
                _EFFORT_LABELS,
                code="decision.effort_invalid",
                message="effort must be E0 through E4",
            )
            values["effort"] = persisted_effort
            persisted_verification, persisted_verification_rank = (
                _validated_verification_rank(
                    values.get("verification"),
                    code="decision.verification_invalid",
                    message="verification must be V0 through V4",
                )
            )
            values["verification"] = persisted_verification
            _, task_verification_rank = _validated_verification_rank(
                validated_task.verification.minimum,
                code="task.verification_invalid",
                message="verification.minimum must be V0 through V4",
            )
            if (
                values.get("task_id") != validated_task.task_id
                or values.get("policy_version") != validated_task.policy_version
                or persisted_effort != validated_task.effort
                or persisted_verification_rank < task_verification_rank
            ):
                raise DomainValidationError(
                    "decision.trusted_task_context_invalid",
                    "persisted decision must satisfy trusted task identity and policy",
                )
            if not isinstance(trusted_routes, Mapping):
                raise DomainValidationError(
                    "decision.trusted_route_context_required",
                    "persisted route decision requires trusted route registry rehydration",
                )
            trusted_routes = _revalidated_route_registry(
                trusted_routes,
                code="decision.trusted_route_context_invalid",
                location="trusted route registry",
            )
            trigger = values.get("trigger")
            if isinstance(trigger, Mapping):
                trigger = _mapping_snapshot(
                    trigger,
                    code="decision.trigger_invalid",
                    location="decision trigger",
                )
                if trigger.get("schema_version") == "initial-selection-trigger/v1":
                    values["trigger"] = InitialSelectionTriggerV1.from_mapping(trigger)
                else:
                    values["trigger"] = RuntimeErrorClassificationV1.from_mapping(trigger)
            plan = values.get("quality_compensation_plan")
            if isinstance(plan, Mapping):
                values["quality_compensation_plan"] = QualityCompensationPlanV1.from_mapping(
                    plan
                )
            raw_candidates = values.get("candidates", ())
            if isinstance(raw_candidates, (str, bytes, bytearray)) or not isinstance(
                raw_candidates,
                Sequence,
            ):
                raise TypeError("candidates")
            if len(raw_candidates) > _MAX_DECISION_CANDIDATES:
                raise DomainValidationError(
                    "decision.candidates_invalid",
                    f"candidates must contain at most {_MAX_DECISION_CANDIDATES} entries",
                )
            candidates: list[CandidateEvaluation] = []
            for item in raw_candidates:
                if type(item) is CandidateEvaluation:
                    if type(resolved_relation) is DecisionRelation:
                        route_context = trusted_routes.get(item.route_id) if trusted_routes else None
                        if route_context is None:
                            raise DomainValidationError(
                                "decision.trusted_route_context_required",
                                "every persisted fallback candidate requires trusted RouteV1 context",
                            )
                        candidate = CandidateEvaluation(
                            route_context,
                            item.deterministic_status,
                            item.rejection_codes,
                            item.score,
                            item.score_factors,
                        )
                    else:
                        candidate = item
                elif isinstance(item, Mapping):
                    item = _mapping_snapshot(
                        item,
                        code="candidate.invalid",
                        location="candidate",
                    )
                    persisted_route_id = _validated_route_id(item.get("route_id"), "route_id")
                    route_context = (
                        trusted_routes.get(persisted_route_id)
                        if trusted_routes is not None
                        else None
                    )
                    if (
                        type(resolved_relation) is DecisionRelation
                        and route_context is None
                    ):
                        raise DomainValidationError(
                            "decision.trusted_route_context_required",
                            "every persisted fallback candidate requires trusted RouteV1 context",
                        )
                    candidate = CandidateEvaluation.from_mapping(
                        item,
                        route_context=route_context,
                    )
                else:
                    candidate = _raise_candidate()
                candidates.append(candidate)
            values["candidates"] = tuple(candidates)
            if resolved_relation in dynamic_relations:
                prior_route_id = _validated_route_id(values.get("prior_route_id"), "prior_route_id")
                prior_route = trusted_routes.get(prior_route_id) if trusted_routes else None
                selected_route = None
                if resolved_relation in {DecisionRelation.FALLBACK, DecisionRelation.REPLAN}:
                    selected_route_id = _validated_route_id(
                        values.get("selected_route_id"),
                        "selected_route_id",
                    )
                    selected_route = trusted_routes.get(selected_route_id) if trusted_routes else None
                if prior_route is None or (
                    resolved_relation in {DecisionRelation.FALLBACK, DecisionRelation.REPLAN}
                    and selected_route is None
                ):
                    raise DomainValidationError(
                        "decision.trusted_route_context_required",
                        "prior and selected routes require trusted RouteV1 context",
                    )
                parsed_trigger = values.get("trigger")
                if (
                    type(parsed_trigger) is not RuntimeErrorClassificationV1
                    or parsed_trigger.attempted_route_id != prior_route.route_id
                    or parsed_trigger.quota_pool_id != prior_route.quota_pool_id
                    or parsed_trigger.billing_pool_id != prior_route.billing_pool_id
                ):
                    raise DomainValidationError(
                        "decision.trigger_route_mismatch",
                        "persisted trigger identity and pools must match the trusted prior route",
                    )
                parsed_plan = values.get("quality_compensation_plan")
                if (
                    type(parsed_plan) is QualityCompensationPlanV1
                    and type(selected_route) is RouteV1
                    and not parsed_plan.matches_route_context(prior_route, selected_route)
                ):
                    raise DomainValidationError(
                        "quality.route_binding_invalid",
                        "persisted compensation pool identities do not match trusted routes",
                    )
            for collection_name in ("reason_codes", "recheck_evidence"):
                if collection_name in values:
                    raw_collection = values[collection_name]
                    values[collection_name] = _immutable_string_collection(
                        raw_collection,
                        collection_name,
                    )
            values["trusted_task"] = validated_task
            values["trusted_routes"] = trusted_routes
            values["trusted_reviewer_routes"] = trusted_reviewer_routes
            values["trusted_human_approval_refs"] = trusted_human_approval_refs
            values["trusted_execution_routes"] = trusted_execution_routes
            values["trusted_execution_evidence"] = trusted_execution_evidence
            values["trusted_evidence_refs"] = trusted_evidence_refs
            values["trusted_threshold_results"] = trusted_threshold_results
            values["trusted_prior_route"] = (
                prior_route if resolved_relation in dynamic_relations else None
            )
            decision = cls(**values)  # type: ignore[arg-type]
            if (
                persisted_policy_status != decision.policy_status
                or persisted_block_reason != decision.activation_block_reason
            ):
                raise DomainValidationError(
                    "decision.persisted_status_mismatch",
                    "persisted derived decision status does not match contract validation",
                )
            return decision
        except (TypeError, ValueError) as exc:
            if isinstance(exc, DomainValidationError):
                raise
            raise DomainValidationError(
                "decision.schema_invalid", "route decision payload is malformed"
            ) from exc

    @property
    def dispatchable(self) -> bool:
        # Pure-phase decisions are structural policy records only. A future
        # wired phase must consult an opaque, sealed external authorization
        # artifact rather than deriving dispatch authority from this DTO.
        return False

def replan_after_capacity_exhaustion(
    *,
    trusted_task: TaskEnvelope,
    task_id: str,
    attempt_id: str,
    failed_route: RouteV1,
    classification: RuntimeErrorClassificationV1,
    candidates: Sequence[CandidateEvaluation],
    decision_id: str,
    parent_decision_id: str,
    created_at: str,
    policy_version: str,
    router_version: str,
    capacity_view_id: str,
    effort: str,
    verification: str,
    quality_compensation_plan: QualityCompensationPlanV1 | Mapping[str, object] | None = None,
    trusted_reviewer_routes: Mapping[str, RouteV1] | None = None,
    trusted_human_approval_refs: Mapping[str, str] | None = None,
    trusted_execution_routes: Mapping[str, RouteV1] | None = None,
    trusted_execution_evidence: Mapping[str, Sequence[str]] | None = None,
    trusted_evidence_refs: Sequence[str] | None = None,
    trusted_threshold_results: Mapping[str, bool] | None = None,
    recheck_evidence: Sequence[str] = (),
) -> RouteDecisionV1:
    if type(trusted_task) is not TaskEnvelope:
        raise DomainValidationError(
            "replan.trusted_task_required",
            "capacity replan requires trusted TaskEnvelope authority",
        )
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise DomainValidationError("decision.candidates_invalid", "candidates must be a sequence")
    if len(candidates) > _MAX_DECISION_CANDIDATES:
        raise DomainValidationError(
            "decision.candidates_invalid",
            f"candidates must contain at most {_MAX_DECISION_CANDIDATES} entries",
        )
    validated_recheck_evidence = _immutable_string_collection(
        recheck_evidence,
        "recheck_evidence",
    )
    trusted_task = TaskEnvelope.from_mapping(
        _safe_asdict(trusted_task, "trusted task")
    )
    if trusted_task.effort == "E0":
        raise DomainValidationError(
            "replan.effort_route_prohibited",
            "E0 tasks cannot enter route-scoped capacity replanning",
        )
    effort = _validated_exact_label(
        effort,
        _EFFORT_LABELS,
        code="replan.trusted_task_mismatch",
        message="replan effort must be E0 through E4",
    )
    verification, verification_rank = _validated_verification_rank(
        verification,
        code="replan.trusted_task_mismatch",
        message="replan verification must be V0 through V4",
    )
    _, task_verification_rank = _validated_verification_rank(
        trusted_task.verification.minimum,
        code="task.verification_invalid",
        message="verification.minimum must be V0 through V4",
    )
    if (
        task_id != trusted_task.task_id
        or policy_version != trusted_task.policy_version
        or effort != trusted_task.effort
        or verification_rank < task_verification_rank
    ):
        raise DomainValidationError(
            "replan.trusted_task_mismatch",
            "replan task, policy, effort, and verification must satisfy trusted task",
        )
    failed_route = _revalidated_route(failed_route, "failed route")
    if type(classification) is not RuntimeErrorClassificationV1:
        raise DomainValidationError(
            "classification.capacity_scope_required",
            "validated runtime classification is required",
        )
    classification = RuntimeErrorClassificationV1.from_mapping(
        _safe_asdict(classification, "runtime error classification")
    )
    if (
        classification.attempted_route_id != failed_route.route_id
        or classification.quota_pool_id.casefold() != failed_route.quota_pool_id
        or (
            classification.billing_pool_id is not None
            and classification.billing_pool_id != failed_route.billing_pool_id
        )
    ):
        raise DomainValidationError(
            "classification.capacity_scope_required",
            "classification must match attempted route and quota/billing pools",
        )
    if classification.billing_pool_id is None:
        classification = RuntimeErrorClassificationV1(
            kind=classification.kind,
            source=classification.source,
            attempted_route_id=classification.attempted_route_id,
            quota_pool_id=classification.quota_pool_id,
            billing_pool_id=failed_route.billing_pool_id,
            classified_at=classification.classified_at,
            evidence_code=classification.evidence_code,
        )
    evaluated_candidates: list[CandidateEvaluation] = []
    for candidate in candidates:
        if type(candidate) is not CandidateEvaluation:
            raise DomainValidationError("decision.candidates_invalid", "candidate DTO required")
        candidate_route = _revalidated_route(candidate.route, "candidate route")
        validated_candidate = CandidateEvaluation(
            candidate_route,
            candidate.deterministic_status,
            candidate.rejection_codes,
            candidate.score,
            candidate.score_factors,
        )
        if candidate_route.quota_pool_id == failed_route.quota_pool_id:
            rejection_codes = validated_candidate.rejection_codes
            if "same_exhausted_quota_pool" not in rejection_codes:
                rejection_codes += ("same_exhausted_quota_pool",)
            evaluated_candidates.append(
                CandidateEvaluation(
                    candidate_route,
                    False,
                    rejection_codes,
                )
            )
        else:
            evaluated_candidates.append(validated_candidate)
    _require_scored_eligible_candidates(
        evaluated_candidates,
        code="replan.candidate_score_required",
    )
    canonical_candidates = tuple(
        sorted(evaluated_candidates, key=lambda candidate: candidate.route_id)
    )
    candidate_ids = tuple(candidate.route_id for candidate in canonical_candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DomainValidationError(
            "decision.candidates_invalid",
            "candidate route IDs must be unique",
        )
    eligible = sorted(
        (candidate for candidate in canonical_candidates if candidate.eligible),
        key=lambda candidate: (
            -candidate.score,  # type: ignore[operator]
            candidate.route_id,
        ),
    )

    # The public RouteDecisionV1 constructor must continue to distrust and
    # revalidate typed DTOs. This top-level operation has already revalidated
    # its task, failed route, classification, and every candidate route/value,
    # so re-entering that public boundary would repeat O(n) graph scans and
    # canonicalization. Validate the remaining caller-supplied structural
    # registries once, then keep the optimized construction lexical to this
    # call path so no caller can supply allegedly prevalidated state.
    normalized_decision_fields = {
        name: _task_text(value, name)
        for name, value in (
            ("decision_id", decision_id),
            ("task_id", task_id),
            ("attempt_id", attempt_id),
            ("created_at", created_at),
            ("policy_version", policy_version),
            ("router_version", router_version),
            ("capacity_view_id", capacity_view_id),
        )
    }
    normalized_parent_decision_id = _task_text(
        parent_decision_id,
        "parent_decision_id",
    )
    _validated_unsealed_decision_authorities(
        trusted_reviewer_routes=trusted_reviewer_routes,
        trusted_human_approval_refs=trusted_human_approval_refs,
        trusted_execution_routes=trusted_execution_routes,
        trusted_execution_evidence=trusted_execution_evidence,
        trusted_evidence_refs=trusted_evidence_refs,
        trusted_threshold_results=trusted_threshold_results,
    )

    def reject_sensitive_revalidated_decision_values(value: object) -> None:
        """Scan only the bounded local graph revalidated by this invocation."""

        if isinstance(value, str):
            if _SENSITIVE_VALUE_PATTERN.search(value):
                raise DomainValidationError(
                    "decision.sensitive_field_prohibited",
                    "sensitive value prohibited in decision",
                )
            return
        value_type = type(value)
        if value_type is CandidateEvaluation:
            # Route identity is a computed SHA-256 label, status is an exact
            # enum-like constant, and score is finite numeric. These two
            # bounded collections are the candidate's remaining caller text.
            reject_sensitive_revalidated_decision_values(value.rejection_codes)
            reject_sensitive_revalidated_decision_values(value.score_factors)
            return
        if value_type is tuple:
            for item in value:
                reject_sensitive_revalidated_decision_values(item)
            return
        if value_type is dict:
            for key, item in value.items():  # type: ignore[union-attr]
                if (
                    type(key) is str
                    and key.casefold() in _PROHIBITED_FIELD_NAMES
                ):
                    raise DomainValidationError(
                        "decision.sensitive_field_prohibited",
                        "sensitive field prohibited in decision",
                    )
                reject_sensitive_revalidated_decision_values(key)
                reject_sensitive_revalidated_decision_values(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for dataclass_field in fields(value):
                reject_sensitive_revalidated_decision_values(
                    getattr(value, dataclass_field.name)
                )

    def build_revalidated_decision(
        *,
        relation: DecisionRelation,
        selected_route_id: str | None,
        reason_codes: tuple[str, ...],
        plan: QualityCompensationPlanV1 | None,
        decision_recheck_evidence: tuple[str, ...] = (),
    ) -> RouteDecisionV1:
        """Construct only from values revalidated above in this invocation."""

        validated_reason_codes = _stable_identifier_collection(
            reason_codes,
            "reason_codes",
            code="decision.reason_codes_invalid",
        )
        if relation is DecisionRelation.FALLBACK:
            validated_reason_codes += ("quality_compensation_insufficient",)
            policy_status = "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
            activation_block_reason = "quality_compensation_insufficient"
        else:
            if selected_route_id is not None:
                raise DomainValidationError(
                    "decision.waiting_selected_route",
                    "waiting must not select a route",
                )
            if not decision_recheck_evidence:
                raise DomainValidationError(
                    "replan.recheck_evidence_required",
                    "waiting requires recheck evidence",
                )
            if any(candidate.eligible for candidate in canonical_candidates):
                raise DomainValidationError(
                    "decision.waiting_eligible_candidate",
                    "waiting requires every retained candidate to be ineligible",
                )
            _reject_sensitive(
                {"recheck_evidence": decision_recheck_evidence},
                "decision",
            )
            policy_status = "WAITING_FOR_CAPACITY"
            activation_block_reason = None

        decision = object.__new__(RouteDecisionV1)
        values: dict[str, object] = {
            **normalized_decision_fields,
            "effort": effort,
            "verification": verification,
            "fallback": relation is DecisionRelation.FALLBACK,
            "relation": relation,
            "candidates": canonical_candidates,
            "selected_route_id": selected_route_id,
            "trigger": classification,
            "reason_codes": validated_reason_codes,
            "prior_route_id": failed_route.route_id,
            "parent_decision_id": normalized_parent_decision_id,
            "reservation_id": None,
            "quality_compensation_plan": plan,
            "recheck_evidence": decision_recheck_evidence,
            "schema_version": "route-decision/v1",
            "policy_status": policy_status,
            "activation_block_reason": activation_block_reason,
        }
        reject_sensitive_revalidated_decision_values(values)
        for name, value in values.items():
            object.__setattr__(decision, name, value)
        return decision

    if eligible:
        selected = eligible[0]
        plan = _coerce_quality_plan(quality_compensation_plan)
        if plan is not None and not plan.matches_route_context(failed_route, selected.route):
            plan = None
        return build_revalidated_decision(
            relation=DecisionRelation.FALLBACK,
            selected_route_id=selected.route_id,
            reason_codes=("route_capacity_exhausted",),
            plan=plan,
        )
    return build_revalidated_decision(
        relation=DecisionRelation.WAITING,
        selected_route_id=None,
        reason_codes=("no_eligible_routes",),
        plan=None,
        decision_recheck_evidence=validated_recheck_evidence,
    )
