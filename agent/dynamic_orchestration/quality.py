"""Quality thresholds, escalation, review attestations, and compensation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping, Sequence

from .eligibility import (
    CandidateEvaluation,
)

from .route import (
    RouteV1,
)

from .validation import (
    DomainValidationError,
    QUALITY_COMPENSATION_SCHEMA_VERSION,
    _MAX_TEXT_COLLECTION_ITEMS,
    _ascii_trimmed_nfc,
    _canonical_pool_identity,
    _finite_number,
    _immutable_string_collection,
    _mapping_snapshot,
    _parse_exact_enum,
    _reject_sensitive,
    _safe_asdict,
    _task_text,
    _validated_exact_label,
    _validated_mapping_keys,
    _validated_route_id,
    _validated_verification_rank,
)
class EscalationAction(str, Enum):
    BLOCK_DISPATCH = "BLOCK_DISPATCH"
    HUMAN_GATE = "HUMAN_GATE"

@dataclass(frozen=True)
class AcceptanceThresholdV1:
    metric: str
    operator: str
    value: str | int | float
    unit: str | None = None
    evidence_required: bool = True
    evidence_ref: str | None = None
    met: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _task_text(self.metric, "metric"))
        object.__setattr__(
            self,
            "operator",
            _validated_exact_label(
                self.operator,
                frozenset({">", ">=", "<", "<=", "==", "!="}),
                code="quality.threshold_invalid",
                message="threshold operator is not supported",
            ),
        )
        if type(self.value) not in {str, int, float}:
            raise DomainValidationError(
                "quality.threshold_invalid",
                "threshold value must be a finite scalar",
            )
        if type(self.value) is str:
            object.__setattr__(
                self,
                "value",
                _ascii_trimmed_nfc(
                    self.value,
                    field_name="threshold value",
                    code="quality.threshold_invalid",
                ),
            )
        else:
            object.__setattr__(
                self,
                "value",
                _finite_number(
                    self.value,
                    code="quality.threshold_invalid",
                    field_name="threshold value",
                ),
            )
        if self.unit is not None:
            object.__setattr__(self, "unit", _task_text(self.unit, "unit"))
        if not isinstance(self.evidence_required, bool):
            raise DomainValidationError(
                "quality.threshold_invalid",
                "evidence_required must be boolean",
            )
        if self.evidence_ref is not None:
            object.__setattr__(
                self,
                "evidence_ref",
                _task_text(self.evidence_ref, "evidence_ref"),
            )
        if self.met is not None and not isinstance(self.met, bool):
            raise DomainValidationError("quality.threshold_invalid", "met must be boolean or unknown")
        _reject_sensitive(self, "quality.threshold")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AcceptanceThresholdV1":
        payload = _mapping_snapshot(
            payload,
            code="quality.threshold_invalid",
            location="acceptance threshold",
        )
        expected = {field_.name for field_ in fields(cls)}
        if _validated_mapping_keys(
            payload,
            code="quality.threshold_invalid",
            location="acceptance threshold",
        ) - expected:
            raise DomainValidationError("quality.threshold_invalid", "unexpected threshold fields")
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("quality.threshold_invalid", "malformed threshold") from exc

@dataclass(frozen=True)
class CompensationEscalationV1:
    on_unmet: EscalationAction
    owner: str
    deadline: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "on_unmet",
            _parse_exact_enum(
                self.on_unmet,
                EscalationAction,
                code="quality.escalation_invalid",
                message="escalation action must block dispatch or require a human gate",
            ),
        )
        object.__setattr__(self, "owner", _task_text(self.owner, "owner"))
        if self.deadline is not None:
            object.__setattr__(self, "deadline", _task_text(self.deadline, "deadline"))
        _reject_sensitive(self, "quality.escalation")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CompensationEscalationV1":
        payload = _mapping_snapshot(
            payload,
            code="quality.escalation_invalid",
            location="compensation escalation",
        )
        expected = {field_.name for field_ in fields(cls)}
        if _validated_mapping_keys(
            payload,
            code="quality.escalation_invalid",
            location="compensation escalation",
        ) - expected:
            raise DomainValidationError("quality.escalation_invalid", "unexpected escalation fields")
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("quality.escalation_invalid", "malformed escalation") from exc

@dataclass(frozen=True)
class IndependentReviewAttestationV1:
    reviewer: str
    route_id: str
    quota_pool_id: str
    billing_pool_id: str
    reviewed_execution_id: str
    execution_id: str
    evidence_ref: str

    def __post_init__(self) -> None:
        for name in (
            "reviewer",
            "route_id",
            "reviewed_execution_id",
            "execution_id",
            "evidence_ref",
        ):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        object.__setattr__(self, "route_id", _validated_route_id(self.route_id, "route_id"))
        for name in ("quota_pool_id", "billing_pool_id"):
            object.__setattr__(
                self,
                name,
                _canonical_pool_identity(
                    getattr(self, name),
                    code="quality.independence_invalid",
                    field_name=name,
                ),
            )
        _reject_sensitive(self, "quality.review")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "IndependentReviewAttestationV1":
        payload = _mapping_snapshot(
            payload,
            code="quality.independence_invalid",
            location="review attestation",
        )
        expected = {field_.name for field_ in fields(cls)}
        if _validated_mapping_keys(
            payload,
            code="quality.independence_invalid",
            location="review attestation",
        ) != expected:
            raise DomainValidationError(
                "quality.independence_invalid",
                "review attestation must contain the exact v1 identity fields",
            )
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "quality.independence_invalid",
                "malformed independent review attestation",
            ) from exc

@dataclass(frozen=True)
class QualityCompensationPlanV1:
    plan_id: str
    decision_id: str
    prior_route_id: str
    selected_route_id: str
    prior_quota_pool_id: str
    prior_billing_pool_id: str
    selected_quota_pool_id: str
    selected_billing_pool_id: str
    trigger_kind: str
    quality_delta_codes: tuple[str, ...]
    required_verification: str
    independence_required: bool
    required_reviewers: tuple[str, ...]
    review_attestations: tuple[IndependentReviewAttestationV1, ...]
    acceptance_thresholds: tuple[AcceptanceThresholdV1, ...]
    escalation: CompensationEscalationV1
    evidence_refs: tuple[str, ...]
    created_at: str
    policy_version: str
    schema_version: str = QUALITY_COMPENSATION_SCHEMA_VERSION
    human_approval_ref: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != QUALITY_COMPENSATION_SCHEMA_VERSION:
            raise DomainValidationError(
                "quality.schema_invalid",
                "schema_version must be quality-compensation/v1",
            )
        for name in (
            "plan_id",
            "decision_id",
            "prior_route_id",
            "selected_route_id",
            "trigger_kind",
            "created_at",
            "policy_version",
        ):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "prior_route_id",
            _validated_route_id(self.prior_route_id, "prior_route_id"),
        )
        object.__setattr__(
            self,
            "selected_route_id",
            _validated_route_id(self.selected_route_id, "selected_route_id"),
        )
        for name in (
            "prior_quota_pool_id",
            "prior_billing_pool_id",
            "selected_quota_pool_id",
            "selected_billing_pool_id",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_pool_identity(
                    getattr(self, name),
                    code="quality.route_binding_invalid",
                    field_name=name,
                ),
            )
        for name, nonempty in (
            ("quality_delta_codes", True),
            ("required_reviewers", False),
            ("evidence_refs", True),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_string_collection(
                    getattr(self, name),
                    name,
                    require_nonempty=nonempty,
                ),
            )
        required_verification, _ = _validated_verification_rank(
            self.required_verification,
            code="quality.verification_invalid",
            message="required_verification must be V0 through V4",
        )
        object.__setattr__(
            self,
            "required_verification",
            required_verification,
        )
        if not isinstance(self.independence_required, bool):
            raise DomainValidationError(
                "quality.independence_invalid",
                "independence_required must be boolean",
            )
        if isinstance(self.acceptance_thresholds, (str, bytes)) or not isinstance(
            self.acceptance_thresholds,
            Sequence,
        ):
            raise DomainValidationError(
                "quality.threshold_invalid",
                "acceptance_thresholds must be a collection",
            )
        if len(self.acceptance_thresholds) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "quality.threshold_invalid",
                f"acceptance_thresholds must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        thresholds = tuple(self.acceptance_thresholds)
        if not thresholds or any(type(item) is not AcceptanceThresholdV1 for item in thresholds):
            raise DomainValidationError(
                "quality.threshold_invalid",
                "at least one validated acceptance threshold is required",
            )
        object.__setattr__(
            self,
            "acceptance_thresholds",
            tuple(
                AcceptanceThresholdV1.from_mapping(
                    _safe_asdict(item, "quality acceptance threshold")
                )
                for item in thresholds
            ),
        )
        if isinstance(self.review_attestations, (str, bytes)) or not isinstance(
            self.review_attestations,
            Sequence,
        ):
            raise DomainValidationError(
                "quality.independence_invalid",
                "review_attestations must be a collection",
            )
        if len(self.review_attestations) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "quality.independence_invalid",
                f"review_attestations must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        attestations = tuple(self.review_attestations)
        if any(type(item) is not IndependentReviewAttestationV1 for item in attestations):
            raise DomainValidationError(
                "quality.independence_invalid",
                "review_attestations must contain immutable validated attestations",
            )
        object.__setattr__(
            self,
            "review_attestations",
            tuple(
                IndependentReviewAttestationV1.from_mapping(
                    _safe_asdict(item, "quality review attestation")
                )
                for item in attestations
            ),
        )
        if type(self.escalation) is not CompensationEscalationV1:
            raise DomainValidationError(
                "quality.escalation_invalid",
                "validated escalation is required",
            )
        object.__setattr__(
            self,
            "escalation",
            CompensationEscalationV1.from_mapping(
                _safe_asdict(self.escalation, "quality escalation")
            ),
        )
        if self.human_approval_ref is not None:
            object.__setattr__(
                self,
                "human_approval_ref",
                _task_text(self.human_approval_ref, "human_approval_ref"),
            )
        _reject_sensitive(self, "quality_compensation")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "QualityCompensationPlanV1":
        payload = _mapping_snapshot(
            payload,
            code="quality.schema_invalid",
            location="quality plan",
        )
        payload_keys = _validated_mapping_keys(
            payload,
            code="quality.schema_invalid",
            location="quality plan",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = payload_keys - expected
        if unknown:
            raise DomainValidationError(
                "quality.schema_invalid",
                f"unexpected quality plan fields: {sorted(unknown)}",
            )
        values = dict(payload)
        thresholds = values.get("acceptance_thresholds", ())
        if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence):
            raise DomainValidationError("quality.threshold_invalid", "malformed thresholds")
        if len(thresholds) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "quality.threshold_invalid",
                f"thresholds must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        values["acceptance_thresholds"] = tuple(
            AcceptanceThresholdV1.from_mapping(
                _safe_asdict(item, "quality acceptance threshold")
            )
            if type(item) is AcceptanceThresholdV1
            else AcceptanceThresholdV1.from_mapping(item)
            if isinstance(item, Mapping)
            else _raise_quality_threshold()
            for item in thresholds
        )
        escalation = values.get("escalation")
        if type(escalation) is CompensationEscalationV1:
            values["escalation"] = CompensationEscalationV1.from_mapping(
                _safe_asdict(escalation, "quality escalation")
            )
        elif isinstance(escalation, Mapping):
            values["escalation"] = CompensationEscalationV1.from_mapping(escalation)
        attestations = values.get("review_attestations", ())
        if isinstance(attestations, (str, bytes)) or not isinstance(attestations, Sequence):
            raise DomainValidationError("quality.independence_invalid", "malformed review attestations")
        if len(attestations) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "quality.independence_invalid",
                f"review attestations must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        values["review_attestations"] = tuple(
            IndependentReviewAttestationV1.from_mapping(
                _safe_asdict(item, "quality review attestation")
            )
            if type(item) is IndependentReviewAttestationV1
            else IndependentReviewAttestationV1.from_mapping(item)
            if isinstance(item, Mapping)
            else _raise_review_attestation()
            for item in attestations
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("quality.schema_invalid", "malformed quality plan") from exc

    def valid_for(
        self,
        *,
        decision_id: str,
        prior_route_id: str,
        selected_route_id: str,
        attempt_id: str,
        task_verification_minimum: str,
        task_independence_required: bool,
        task_human_gate_required: bool,
        decision_policy_version: str,
        decision_verification: str,
        trusted_reviewer_routes: Mapping[str, RouteV1] | None,
        trusted_human_approval_refs: frozenset[str],
        trusted_execution_routes: Mapping[str, RouteV1],
        trusted_execution_evidence: Mapping[str, frozenset[str]],
        trusted_evidence_refs: frozenset[str],
        trusted_threshold_results: Mapping[str, bool],
    ) -> bool:
        # These arguments are structural, caller-supplied DTO data. This
        # pure/unwired phase has no opaque sealed review artifact and therefore
        # no authority that can satisfy a compensation plan.
        del (
            decision_id,
            prior_route_id,
            selected_route_id,
            attempt_id,
            task_independence_required,
            task_human_gate_required,
            decision_policy_version,
            trusted_reviewer_routes,
            trusted_human_approval_refs,
            trusted_execution_routes,
            trusted_execution_evidence,
            trusted_evidence_refs,
            trusted_threshold_results,
        )
        try:
            QualityCompensationPlanV1.from_mapping(
                _safe_asdict(self, "quality compensation plan")
            )
            _validated_verification_rank(
                task_verification_minimum,
                code="quality.verification_invalid",
                message="task verification must be V0 through V4",
            )
            _validated_verification_rank(
                decision_verification,
                code="quality.verification_invalid",
                message="decision verification must be V0 through V4",
            )
        except DomainValidationError:
            return False
        return False

    def matches_route_context(self, prior_route: RouteV1, selected_route: RouteV1) -> bool:
        return (
            self.prior_route_id == prior_route.route_id
            and self.selected_route_id == selected_route.route_id
            and self.prior_quota_pool_id == prior_route.quota_pool_id
            and self.prior_billing_pool_id == prior_route.billing_pool_id
            and self.selected_quota_pool_id == selected_route.quota_pool_id
            and self.selected_billing_pool_id == selected_route.billing_pool_id
        )

def _raise_candidate() -> CandidateEvaluation:
    raise DomainValidationError("candidate.invalid", "malformed candidate")

def _raise_quality_threshold() -> AcceptanceThresholdV1:
    raise DomainValidationError("quality.threshold_invalid", "malformed threshold")

def _raise_review_attestation() -> IndependentReviewAttestationV1:
    raise DomainValidationError("quality.independence_invalid", "malformed review attestation")
