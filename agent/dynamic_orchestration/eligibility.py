"""Eligibility facts, triggers, runtime classification, and scoring."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .route import (
    RouteV1,
    _revalidated_route,
)

from .validation import (
    DomainValidationError,
    _MAX_DECISION_CANDIDATES,
    _MAX_ROUTE_REGISTRY_ITEMS,
    _ascii_trimmed_nfc,
    _canonical_pool_identity,
    _finite_number,
    _mapping_snapshot,
    _reject_sensitive,
    _stable_identifier_collection,
    _task_text,
    _validated_mapping_keys,
    _validated_route_id,
)
class ErrorKind(str, Enum):
    CAPACITY_EXHAUSTED = "capacity_exhausted"

class EligibilityDisposition(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    UNKNOWN = "UNKNOWN"

_ELIGIBILITY_GATES = (
    "identity_policy",
    "privacy_permission",
    "capability_tool",
    "context",
    "freshness_confidence",
    "budget",
    "breaker_cooldown",
    "concurrency_reservation",
)

@dataclass(frozen=True)
class RouteEligibilityFactsV1:
    route: RouteV1
    identity_policy: EligibilityDisposition
    privacy_permission: EligibilityDisposition
    capability_tool: EligibilityDisposition
    context: EligibilityDisposition
    freshness_confidence: EligibilityDisposition
    budget: EligibilityDisposition
    breaker_cooldown: EligibilityDisposition
    concurrency_reservation: EligibilityDisposition

    def __post_init__(self) -> None:
        if type(self.route) is not RouteV1:
            raise DomainValidationError(
                "eligibility.invalid",
                "eligibility facts require a validated RouteV1",
            )
        object.__setattr__(self, "route", _revalidated_route(self.route, "eligibility facts"))
        for gate in _ELIGIBILITY_GATES:
            if type(getattr(self, gate)) is not EligibilityDisposition:
                raise DomainValidationError(
                    "eligibility.invalid",
                    f"{gate} requires a typed EligibilityDisposition",
                )

@dataclass(frozen=True)
class InitialSelectionTriggerV1:
    schema_version: str
    kind: str
    source: str
    evaluated_at: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "initial-selection-trigger/v1"
            or type(self.kind) is not str
            or self.kind != "initial_selection"
        ):
            raise DomainValidationError(
                "decision.trigger_invalid",
                "initial trigger must be initial-selection-trigger/v1",
            )
        for name in ("source", "evaluated_at"):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        _reject_sensitive(self, "initial trigger")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "InitialSelectionTriggerV1":
        payload = _mapping_snapshot(
            payload,
            code="decision.trigger_invalid",
            location="initial trigger",
        )
        expected = {field_.name for field_ in fields(cls)}
        if _validated_mapping_keys(
            payload,
            code="decision.trigger_invalid",
            location="initial trigger",
        ) != expected:
            raise DomainValidationError(
                "decision.trigger_invalid",
                "initial trigger must contain the exact v1 fields",
            )
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("decision.trigger_invalid", "malformed initial trigger") from exc

@dataclass(frozen=True)
class RuntimeErrorClassificationV1:
    kind: ErrorKind
    source: str
    attempted_route_id: str
    quota_pool_id: str
    classified_at: str
    billing_pool_id: str | None = None
    evidence_code: str | None = None

    def __post_init__(self) -> None:
        if self.kind is not ErrorKind.CAPACITY_EXHAUSTED:
            raise DomainValidationError(
                "classification.capacity_scope_required",
                "capacity exhaustion classification kind is required",
            )
        object.__setattr__(
            self,
            "attempted_route_id",
            _ascii_trimmed_nfc(
                self.attempted_route_id,
                field_name="attempted_route_id",
                code="classification.capacity_scope_required",
            ),
        )
        object.__setattr__(
            self,
            "quota_pool_id",
            _canonical_pool_identity(
                self.quota_pool_id,
                field_name="quota_pool_id",
                code="classification.capacity_scope_required",
            ),
        )
        object.__setattr__(
            self,
            "attempted_route_id",
            _validated_route_id(self.attempted_route_id, "attempted_route_id"),
        )
        if self.billing_pool_id is not None:
            normalized = _canonical_pool_identity(
                self.billing_pool_id,
                field_name="billing_pool_id",
                code="classification.scalar_invalid",
            )
            object.__setattr__(self, "billing_pool_id", normalized)
        if self.evidence_code is not None:
            object.__setattr__(
                self,
                "evidence_code",
                _task_text(self.evidence_code, "evidence_code"),
            )
        object.__setattr__(self, "source", _task_text(self.source, "classification source"))
        object.__setattr__(
            self,
            "classified_at",
            _task_text(self.classified_at, "classification timestamp"),
        )
        _reject_sensitive(self, "classification")

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object]
    ) -> "RuntimeErrorClassificationV1":
        if not isinstance(payload, Mapping):
            raise DomainValidationError(
                "classification.capacity_scope_required",
                "capacity classification must be a mapping with route and pool scope",
            )
        payload = _mapping_snapshot(
            payload,
            code="classification.capacity_scope_required",
            location="classification",
        )
        if "provider" in payload or "model" in payload:
            raise DomainValidationError(
                "classification.capacity_scope_required",
                "provider/model labels cannot replace attempted_route_id and quota_pool_id",
            )
        required_text = {
            name: _ascii_trimmed_nfc(
                payload.get(name),
                field_name=name,
                code="classification.capacity_scope_required",
            )
            for name in ("source", "classified_at", "attempted_route_id", "quota_pool_id")
        }
        _reject_sensitive(payload, "classification")
        allowed = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="classification.unexpected_field",
            location="classification",
        ) - allowed
        if unknown:
            raise DomainValidationError(
                "classification.unexpected_field",
                f"unexpected classification fields: {sorted(unknown)}",
            )
        try:
            raw_kind = payload.get("kind", "")
            if type(raw_kind) is ErrorKind:
                kind = raw_kind
            elif type(raw_kind) is str:
                kind = ErrorKind(raw_kind)
            else:
                raise ValueError
        except ValueError as exc:
            raise DomainValidationError(
                "classification.capacity_scope_required",
                "typed capacity exhaustion classification is required",
            ) from exc
        optional_text: dict[str, str | None] = {}
        for name in ("billing_pool_id", "evidence_code"):
            raw_value = payload.get(name)
            optional_text[name] = (
                None
                if raw_value is None
                else _ascii_trimmed_nfc(
                    raw_value,
                    field_name=name,
                    code="classification.scalar_invalid",
                )
            )
        return cls(
            kind=kind,
            source=required_text["source"],
            attempted_route_id=required_text["attempted_route_id"],
            quota_pool_id=required_text["quota_pool_id"],
            classified_at=required_text["classified_at"],
            billing_pool_id=optional_text["billing_pool_id"],
            evidence_code=optional_text["evidence_code"],
        )

@dataclass(frozen=True, init=False)
class CandidateEvaluation:
    route_id: str
    deterministic_status: str
    rejection_codes: tuple[str, ...]
    score: float | None
    score_factors: tuple[str, ...]

    def __init__(
        self,
        route_or_id: RouteV1 | str,
        deterministic_status: bool | str,
        rejection_codes: Sequence[str] = (),
        score: float | None = None,
        score_factors: Sequence[str] = (),
    ) -> None:
        route_context = route_or_id if type(route_or_id) is RouteV1 else None
        route_id = (
            route_context.route_id
            if route_context is not None
            else _validated_route_id(route_or_id, "route_id")
        )
        status = (
            "ELIGIBLE" if deterministic_status is True else "REJECTED"
            if deterministic_status is False
            else _task_text(deterministic_status, "deterministic_status").upper()
        )
        if status not in {"ELIGIBLE", "REJECTED"}:
            raise DomainValidationError(
                "candidate.invalid",
                "deterministic_status must be ELIGIBLE or REJECTED",
            )
        normalized_rejections = _stable_identifier_collection(
            rejection_codes,
            "rejection_codes",
            code="candidate.invalid",
        )
        normalized_factors = _stable_identifier_collection(
            score_factors,
            "score_factors",
            code="candidate.invalid",
        )
        if status == "REJECTED" and not normalized_rejections:
            raise DomainValidationError(
                "candidate.invalid",
                "rejected candidate requires at least one rejection code",
            )
        if status == "ELIGIBLE" and normalized_rejections:
            raise DomainValidationError(
                "candidate.invalid",
                "eligible candidate must not contain rejection codes",
            )
        if status == "REJECTED" and score is not None:
            raise DomainValidationError("candidate.invalid", "rejected candidate must not be scored")
        normalized_score = (
            None
            if score is None
            else _finite_number(
                score,
                code="candidate.invalid",
                field_name="score",
            )
        )
        object.__setattr__(self, "route_id", route_id)
        object.__setattr__(self, "deterministic_status", status)
        object.__setattr__(self, "rejection_codes", normalized_rejections)
        object.__setattr__(self, "score", normalized_score)
        object.__setattr__(self, "score_factors", normalized_factors)
        object.__setattr__(self, "_route_context", route_context)

    @property
    def eligible(self) -> bool:
        return self.deterministic_status == "ELIGIBLE"

    @property
    def route(self) -> RouteV1:
        route_context = getattr(self, "_route_context", None)
        if type(route_context) is not RouteV1:
            raise DomainValidationError(
                "candidate.route_context_required",
                "route context is required for policy recomputation",
            )
        return route_context

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        route_context: RouteV1 | None = None,
    ) -> "CandidateEvaluation":
        payload = _mapping_snapshot(
            payload,
            code="candidate.invalid",
            location="candidate",
        )
        expected = {
            "route_id",
            "deterministic_status",
            "rejection_codes",
            "score",
            "score_factors",
        }
        payload_keys = _validated_mapping_keys(
            payload,
            code="candidate.invalid",
            location="candidate",
        )
        unknown = payload_keys - expected
        required = {"route_id", "deterministic_status", "rejection_codes", "score_factors"}
        if unknown or not required.issubset(payload_keys):
            raise DomainValidationError(
                "candidate.invalid",
                "candidate mapping must match CandidateEvaluation v1",
            )
        persisted_route_id = _validated_route_id(payload["route_id"], "route_id")
        if route_context is not None and route_context.route_id != persisted_route_id:
            raise DomainValidationError(
                "candidate.route_binding_invalid",
                "trusted route context does not match persisted candidate route_id",
            )
        raw_status = payload["deterministic_status"]
        if type(raw_status) is not str:
            raise DomainValidationError(
                "candidate.invalid",
                "persisted deterministic_status must be ELIGIBLE or REJECTED",
            )
        return cls(
            route_context or persisted_route_id,
            raw_status,
            payload["rejection_codes"],  # type: ignore[arg-type]
            payload.get("score"),  # type: ignore[arg-type]
            payload["score_factors"],  # type: ignore[arg-type]
        )

@dataclass(frozen=True)
class CandidateScoreV1:
    score: float
    score_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        score = _finite_number(
            self.score,
            code="score.invalid",
            field_name="score",
        )
        factors = _stable_identifier_collection(
            self.score_factors,
            "score_factors",
            code="score.invalid",
            require_nonempty=True,
        )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "score_factors", factors)
        _reject_sensitive({"score_factors": factors}, "score")

def evaluate_route_eligibility(
    facts: Iterable[RouteEligibilityFactsV1],
) -> tuple[CandidateEvaluation, ...]:
    try:
        iterator = iter(facts)
    except TypeError as exc:
        raise DomainValidationError(
            "eligibility.invalid",
            "eligibility facts must be iterable",
        ) from exc
    facts_buffer: list[RouteEligibilityFactsV1] = []
    for item in iterator:
        if len(facts_buffer) >= _MAX_DECISION_CANDIDATES:
            raise DomainValidationError(
                "eligibility.invalid",
                f"eligibility facts must contain at most {_MAX_DECISION_CANDIDATES} entries",
            )
        facts_buffer.append(item)
    facts_items = tuple(facts_buffer)

    validated: list[RouteEligibilityFactsV1] = []
    for item in facts_items:
        if type(item) is not RouteEligibilityFactsV1:
            raise DomainValidationError(
                "eligibility.invalid",
                "every eligibility item must be RouteEligibilityFactsV1",
            )
        validated.append(
            RouteEligibilityFactsV1(
                route=item.route,
                **{gate: getattr(item, gate) for gate in _ELIGIBILITY_GATES},
            )
        )

    route_ids = tuple(item.route.route_id for item in validated)
    if len(set(route_ids)) != len(route_ids):
        raise DomainValidationError(
            "eligibility.duplicate_route",
            "eligibility facts must contain unique route IDs",
        )

    evaluations: list[CandidateEvaluation] = []
    for item in sorted(validated, key=lambda candidate: candidate.route.route_id):
        rejection_codes = tuple(
            f"{gate}_{'rejected' if disposition is EligibilityDisposition.REJECT else 'unknown'}"
            for gate in _ELIGIBILITY_GATES
            if (disposition := getattr(item, gate)) is not EligibilityDisposition.PASS
        )
        evaluations.append(
            CandidateEvaluation(
                item.route,
                not rejection_codes,
                rejection_codes,
            )
        )
    return tuple(evaluations)

def score_eligible_candidates(
    candidates: Sequence[CandidateEvaluation],
    scores_by_route_id: Mapping[str, CandidateScoreV1],
) -> tuple[CandidateEvaluation, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise DomainValidationError("score.invalid", "candidates must be a sequence")
    if len(candidates) > _MAX_DECISION_CANDIDATES:
        raise DomainValidationError(
            "score.invalid",
            f"candidates must contain at most {_MAX_DECISION_CANDIDATES} entries",
        )
    if not isinstance(scores_by_route_id, Mapping):
        raise DomainValidationError("score.invalid", "scores must be keyed by route ID")
    if len(scores_by_route_id) > _MAX_ROUTE_REGISTRY_ITEMS:
        raise DomainValidationError(
            "score.invalid",
            f"scores must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} entries",
        )

    validated_candidates: list[CandidateEvaluation] = []
    for candidate in candidates:
        if type(candidate) is not CandidateEvaluation:
            raise DomainValidationError(
                "score.invalid",
                "every candidate must be CandidateEvaluation",
            )
        try:
            candidate_route = _revalidated_route(candidate.route, "candidate scoring")
            validated_candidate = CandidateEvaluation(
                candidate_route,
                candidate.deterministic_status,
                candidate.rejection_codes,
                candidate.score,
                candidate.score_factors,
            )
        except DomainValidationError as exc:
            raise DomainValidationError(
                "score.invalid",
                "candidate must retain a trusted RouteV1 binding",
            ) from exc
        if validated_candidate.score is not None or validated_candidate.score_factors:
            raise DomainValidationError(
                "score.invalid",
                "candidate must be unscored before score assignment",
            )
        validated_candidates.append(validated_candidate)

    candidate_ids = tuple(candidate.route_id for candidate in validated_candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DomainValidationError(
            "score.duplicate_candidate",
            "candidates must contain unique route IDs",
        )

    validated_scores: dict[str, CandidateScoreV1] = {}
    for route_id, candidate_score in scores_by_route_id.items():
        if type(route_id) is not str or type(candidate_score) is not CandidateScoreV1:
            raise DomainValidationError(
                "score.invalid",
                "scores require string route IDs and validated CandidateScoreV1 values",
            )
        try:
            _validated_route_id(route_id, "score route_id")
            validated_scores[route_id] = CandidateScoreV1(
                candidate_score.score,
                candidate_score.score_factors,
            )
        except DomainValidationError as exc:
            raise DomainValidationError("score.invalid", "malformed candidate score") from exc

    eligible_ids = {
        candidate.route_id for candidate in validated_candidates if candidate.eligible
    }
    if set(validated_scores) != eligible_ids:
        raise DomainValidationError(
            "score.invalid",
            "scores must contain exactly one entry for every eligible route and no others",
        )

    scored: list[CandidateEvaluation] = []
    for candidate in sorted(validated_candidates, key=lambda item: item.route_id):
        if not candidate.eligible:
            scored.append(candidate)
            continue
        candidate_score = validated_scores[candidate.route_id]
        scored.append(
            CandidateEvaluation(
                candidate.route,
                True,
                score=candidate_score.score,
                score_factors=candidate_score.score_factors,
            )
        )
    return tuple(scored)
