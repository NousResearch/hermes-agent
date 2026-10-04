"""Immutable task-envelope contracts and nested task DTOs."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields
from datetime import datetime
from typing import Any, Mapping, Sequence

from .validation import (
    DomainValidationError,
    _EFFORT_LABELS,
    _KNOWN_CAPABILITY_IDENTIFIERS,
    _KNOWN_PERMISSION_IDENTIFIERS,
    _KNOWN_TOOL_IDENTIFIERS,
    _MAX_CONTEXT_TOKENS,
    _MAX_TEXT_COLLECTION_ITEMS,
    _ascii_trimmed_nfc,
    _bounded_non_negative_integer,
    _finite_number,
    _immutable_string_collection,
    _mapping_snapshot,
    _normalized_policy_identifiers,
    _reject_sensitive,
    _safe_asdict,
    _task_text,
    _validated_exact_label,
    _validated_future_timestamp,
    _validated_mapping_keys,
    _validated_verification_rank,
)
@dataclass(frozen=True)
class AuditedModelJustification:
    policy_version: str
    reason: str
    evidence_refs: tuple[str, ...]
    author: str
    expires_at: str
    identity_claims: tuple[tuple[str, str], ...] = ()
    validation_time: InitVar[datetime | None] = None

    def __post_init__(self, validation_time: datetime | None) -> None:
        for name in ("policy_version", "reason", "author"):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "expires_at",
            _validated_future_timestamp(
                self.expires_at,
                "expires_at",
                reference_time=validation_time,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _immutable_string_collection(
                self.evidence_refs,
                "evidence_refs",
                require_nonempty=True,
            ),
        )
        if isinstance(self.identity_claims, (str, bytes)) or not isinstance(
            self.identity_claims,
            Sequence,
        ):
            raise DomainValidationError(
                "task.justification_invalid",
                "identity_claims must be a collection",
            )
        if len(self.identity_claims) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "task.justification_invalid",
                f"identity_claims must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        claims: list[tuple[str, str]] = []
        for claim in self.identity_claims:
            if (
                not isinstance(claim, Sequence)
                or isinstance(claim, (str, bytes, bytearray))
                or len(claim) != 2
                or not all(isinstance(part, str) for part in claim)
            ):
                raise DomainValidationError(
                    "task.justification_invalid",
                    "identity_claims must contain audited field/value pairs",
                )
            claims.append(
                (
                    _task_text(claim[0], "identity claim field"),
                    _task_text(claim[1], "identity claim value"),
                )
            )
        object.__setattr__(self, "identity_claims", tuple(sorted(claims)))
        _reject_sensitive(self, "task.justification")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        reference_time: datetime | None = None,
    ) -> "AuditedModelJustification":
        payload = _mapping_snapshot(
            payload,
            code="task.unexpected_field",
            location="audited justification",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="task.unexpected_field",
            location="audited justification",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "task.unexpected_field",
                f"unexpected audited justification fields: {sorted(unknown)}",
            )
        return cls(
            policy_version=payload.get("policy_version"),  # type: ignore[arg-type]
            reason=payload.get("reason"),  # type: ignore[arg-type]
            evidence_refs=payload.get("evidence_refs", ()),  # type: ignore[arg-type]
            author=payload.get("author"),  # type: ignore[arg-type]
            expires_at=payload.get("expires_at"),  # type: ignore[arg-type]
            identity_claims=payload.get("identity_claims", ()),  # type: ignore[arg-type]
            validation_time=reference_time,
        )

@dataclass(frozen=True)
class TaskContextV1:
    classification: str
    max_tokens: int | None
    allowed_sources: tuple[str, ...]
    token_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "classification", _task_text(self.classification, "context.classification"))
        object.__setattr__(
            self,
            "allowed_sources",
            _immutable_string_collection(self.allowed_sources, "context.allowed_sources"),
        )
        for name in ("max_tokens", "token_count"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _bounded_non_negative_integer(
                        value,
                        code="task.context_bounds_invalid",
                        field_name=f"context.{name}",
                        maximum=_MAX_CONTEXT_TOKENS,
                    ),
                )
        if (
            self.max_tokens is not None
            and self.max_tokens == 0
        ):
            raise DomainValidationError(
                "task.context_bounds_invalid",
                "context.max_tokens must be positive when supplied",
            )
        if (
            self.max_tokens is not None
            and self.token_count is not None
            and self.token_count > self.max_tokens
        ):
            raise DomainValidationError(
                "task.context_bounds_invalid",
                "context.token_count must not exceed context.max_tokens",
            )
        _reject_sensitive(self, "task.context")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaskContextV1":
        return _dto_from_mapping(cls, payload, "task.context")

@dataclass(frozen=True)
class TaskPrivacyV1:
    classification: str
    outbound_allowed: bool
    retention: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "classification", _task_text(self.classification, "privacy.classification"))
        object.__setattr__(self, "retention", _task_text(self.retention, "privacy.retention"))
        if not isinstance(self.outbound_allowed, bool):
            raise DomainValidationError("task.privacy_invalid", "privacy.outbound_allowed must be boolean")
        _reject_sensitive(self, "task.privacy")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaskPrivacyV1":
        return _dto_from_mapping(cls, payload, "task.privacy")

@dataclass(frozen=True)
class TaskRiskV1:
    level: str
    reversibility: str
    impact: str

    def __post_init__(self) -> None:
        for name in ("level", "reversibility", "impact"):
            object.__setattr__(self, name, _task_text(getattr(self, name), f"risk.{name}"))
        _reject_sensitive(self, "task.risk")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaskRiskV1":
        return _dto_from_mapping(cls, payload, "task.risk")

@dataclass(frozen=True)
class TaskBudgetV1:
    currency: str
    paid_allowed: bool
    soft_cap: float | None = None
    hard_cap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _task_text(self.currency, "budget.currency"))
        if not isinstance(self.paid_allowed, bool):
            raise DomainValidationError("task.budget_invalid", "budget.paid_allowed must be boolean")
        for name in ("soft_cap", "hard_cap"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_number(
                        value,
                        code="task.budget_invalid",
                        field_name=f"budget.{name}",
                        non_negative=True,
                    ),
                )
        if self.soft_cap is not None and self.hard_cap is not None and self.soft_cap > self.hard_cap:
            raise DomainValidationError("task.budget_invalid", "budget.soft_cap must not exceed hard_cap")
        _reject_sensitive(self, "task.budget")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaskBudgetV1":
        return _dto_from_mapping(cls, payload, "task.budget")

@dataclass(frozen=True)
class TaskVerificationV1:
    minimum: str
    independent_required: bool
    human_gate_required: bool

    def __post_init__(self) -> None:
        minimum, minimum_rank = _validated_verification_rank(
            self.minimum,
            code="task.verification_invalid",
            message="verification.minimum must be V0 through V4",
        )
        object.__setattr__(self, "minimum", minimum)
        if not isinstance(self.independent_required, bool) or not isinstance(self.human_gate_required, bool):
            raise DomainValidationError("task.verification_invalid", "verification flags must be boolean")
        if minimum_rank >= 2 and not self.independent_required:
            raise DomainValidationError(
                "task.verification_invariant",
                "V2 through V4 require independent verification",
            )
        if self.minimum == "V4" and not self.human_gate_required:
            raise DomainValidationError(
                "task.verification_invariant",
                "V4 requires a human or external authority gate",
            )
        _reject_sensitive(self, "task.verification")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaskVerificationV1":
        return _dto_from_mapping(cls, payload, "task.verification")

def _dto_from_mapping(dto: type[Any], payload: Mapping[str, object], location: str) -> Any:
    payload = _mapping_snapshot(payload, code="task.schema_invalid", location=location)
    expected = {field_.name for field_ in fields(dto)}
    unknown = _validated_mapping_keys(
        payload,
        code="task.unexpected_field",
        location=location,
    ) - expected
    if unknown:
        raise DomainValidationError("task.unexpected_field", f"unexpected {location} fields: {sorted(unknown)}")
    try:
        return dto(**payload)
    except TypeError as exc:
        raise DomainValidationError("task.schema_invalid", f"malformed {location}") from exc

@dataclass(frozen=True)
class TaskEnvelope:
    schema_version: str
    task_id: str
    objective: str
    deliverables: tuple[str, ...]
    capabilities_required: tuple[str, ...]
    tools_allowed: tuple[str, ...]
    permissions_required: tuple[str, ...]
    context: TaskContextV1
    privacy: TaskPrivacyV1
    risk: TaskRiskV1
    effort: str
    budget: TaskBudgetV1
    verification: TaskVerificationV1
    policy_version: str
    root_task_id: str | None = None
    audited_model_justification: AuditedModelJustification | None = None
    validation_time: InitVar[datetime | None] = None

    def __post_init__(self, validation_time: datetime | None) -> None:
        if self.schema_version != "task-envelope/v1":
            raise DomainValidationError("task.schema_invalid", "schema_version must be task-envelope/v1")
        for name in ("task_id", "objective", "policy_version"):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        if self.root_task_id is not None:
            object.__setattr__(self, "root_task_id", _task_text(self.root_task_id, "root_task_id"))
        object.__setattr__(
            self,
            "deliverables",
            _immutable_string_collection(self.deliverables, "deliverables", require_nonempty=True),
        )
        for name, known in (
            ("capabilities_required", _KNOWN_CAPABILITY_IDENTIFIERS),
            ("tools_allowed", _KNOWN_TOOL_IDENTIFIERS),
            ("permissions_required", _KNOWN_PERMISSION_IDENTIFIERS),
        ):
            object.__setattr__(
                self,
                name,
                _normalized_policy_identifiers(getattr(self, name), name, known),
            )
        expected_dtos = (
            ("context", TaskContextV1),
            ("privacy", TaskPrivacyV1),
            ("risk", TaskRiskV1),
            ("budget", TaskBudgetV1),
            ("verification", TaskVerificationV1),
        )
        for name, dto_type in expected_dtos:
            value = getattr(self, name)
            if type(value) is not dto_type:
                raise DomainValidationError("task.schema_invalid", f"{name} must be an immutable validated DTO")
            object.__setattr__(
                self,
                name,
                dto_type.from_mapping(_safe_asdict(value, f"task.{name}")),
            )
        object.__setattr__(
            self,
            "effort",
            _validated_exact_label(
                self.effort,
                _EFFORT_LABELS,
                code="task.effort_invalid",
                message="effort must be E0 through E4",
            ),
        )
        justification = self.audited_model_justification
        if justification is not None:
            if type(justification) is not AuditedModelJustification:
                raise DomainValidationError(
                    "task.justification_invalid", "invalid audited_model_justification"
                )
            validated_justification = AuditedModelJustification.from_mapping(
                _safe_asdict(justification, "task.justification"),
                reference_time=validation_time,
            )
            if validated_justification.policy_version != self.policy_version:
                raise DomainValidationError(
                    "task.justification_policy_mismatch",
                    "audited model justification must match task policy_version",
                )
            object.__setattr__(
                self,
                "audited_model_justification",
                validated_justification,
            )
        _reject_sensitive(self, "task")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        reference_time: datetime | None = None,
    ) -> "TaskEnvelope":
        payload = _mapping_snapshot(
            payload,
            code="task.schema_invalid",
            location="task envelope",
        )
        _validated_mapping_keys(
            payload,
            code="task.unexpected_field",
            location="task",
        )
        _reject_sensitive(payload, "task")
        identity_fields = {
            key
            for key in payload
            if isinstance(key, str)
            and key.casefold() in {"model", "provider", "route_id", "selected_route_id"}
        }
        justification_payload = payload.get("audited_model_justification")
        if identity_fields and not isinstance(justification_payload, Mapping):
            raise DomainValidationError(
                "task.unaudited_model_identity",
                "model/provider/route identity requires audited_model_justification",
            )
        safe_payload = {key: value for key, value in payload.items() if key not in identity_fields}
        allowed = {field_.name for field_ in fields(cls)}
        unknown = set(safe_payload) - allowed
        if unknown:
            raise DomainValidationError("task.unexpected_field", f"unexpected task fields: {sorted(unknown)}")
        values = dict(safe_payload)
        converters = {
            "context": TaskContextV1,
            "privacy": TaskPrivacyV1,
            "risk": TaskRiskV1,
            "budget": TaskBudgetV1,
            "verification": TaskVerificationV1,
        }
        for name, dto_type in converters.items():
            value = values.get(name)
            if isinstance(value, Mapping):
                values[name] = dto_type.from_mapping(value)
        justification = values.get("audited_model_justification")
        if isinstance(justification, Mapping):
            audited_payload = dict(
                _mapping_snapshot(
                    justification,
                    code="task.justification_invalid",
                    location="audited model justification",
                )
            )
            if identity_fields:
                requested_identity = {
                    name: _ascii_trimmed_nfc(
                        payload[name],
                        field_name=name,
                        code="task.scalar_invalid",
                    )
                    for name in identity_fields
                }
                supplied_claims_raw = audited_payload.get("identity_claims", ())
                supplied_claims: dict[str, str] = {}
                if isinstance(supplied_claims_raw, (str, bytes, bytearray)) or not isinstance(
                    supplied_claims_raw,
                    Sequence,
                ):
                    raise DomainValidationError(
                        "task.justification_invalid",
                        "identity_claims must contain field/value pairs",
                    )
                if len(supplied_claims_raw) > _MAX_TEXT_COLLECTION_ITEMS:
                    raise DomainValidationError(
                        "task.justification_invalid",
                        f"identity_claims must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
                    )
                if supplied_claims_raw:
                    for claim in supplied_claims_raw:
                        if (
                            isinstance(claim, (str, bytes, bytearray))
                            or not isinstance(claim, Sequence)
                            or len(claim) != 2
                        ):
                            raise DomainValidationError(
                                "task.justification_invalid",
                                "identity_claims must contain field/value pairs",
                            )
                        claim_name = _ascii_trimmed_nfc(
                            claim[0],
                            field_name="identity claim field",
                            code="task.justification_invalid",
                        )
                        claim_value = _ascii_trimmed_nfc(
                            claim[1],
                            field_name="identity claim value",
                            code="task.justification_invalid",
                        )
                        supplied_claims[claim_name] = claim_value
                if any(
                    name in supplied_claims
                    and supplied_claims[name] != requested_identity[name]
                    for name in identity_fields
                ):
                    raise DomainValidationError(
                        "task.unaudited_model_identity",
                        "audited identity claims must match the requested identity",
                    )
                supplied_claims.update(requested_identity)
                audited_payload["identity_claims"] = tuple(sorted(supplied_claims.items()))
            values["audited_model_justification"] = AuditedModelJustification.from_mapping(
                audited_payload,
                reference_time=reference_time,
            )
        try:
            return cls(**values, validation_time=reference_time)
        except TypeError as exc:
            raise DomainValidationError("task.schema_invalid", "malformed task envelope") from exc
