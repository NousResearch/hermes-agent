"""Execution identity, verification, outcome, and envelope contracts."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Mapping

from .decision import (
    RouteDecisionV1,
)

from .route import (
    RouteV1,
    _revalidated_route_registry,
)

from .task import (
    TaskBudgetV1,
    TaskEnvelope,
)

from .validation import (
    DomainValidationError,
    _EFFORT_LABELS,
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
EXECUTION_ENVELOPE_SCHEMA_VERSION = "execution-envelope/v1"

class ExecutionDispatchState(str, Enum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    RESULT_RECORDED = "RESULT_RECORDED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RECONCILED = "RECONCILED"

class ExecutionOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class ExecutionIdentityAttestationV1:
    """Requested/resolved/effective identity refs, kept separate and never collapsed."""

    requested_identity: str
    resolved_identity: str | None = None
    effective_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_identity", _task_text(self.requested_identity, "requested_identity"))
        for name in ("resolved_identity", "effective_identity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _task_text(value, name))
        _reject_sensitive(self, "execution identity")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionIdentityAttestationV1":
        payload = _mapping_snapshot(
            payload,
            code="execution.schema_invalid",
            location="execution identity",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="execution.schema_invalid",
            location="execution identity",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "execution.schema_invalid",
                f"unexpected execution identity fields: {sorted(unknown)}",
            )
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "execution.schema_invalid",
                "malformed execution identity",
            ) from exc

@dataclass(frozen=True)
class ExecutionVerificationAttestationV1:
    """Required vs effective verification level, with explicit un-attested gaps."""

    required_verification: str
    effective_verification: str | None = None
    attestation_refs: tuple[str, ...] = ()
    unmet_reason: str | None = None

    def __post_init__(self) -> None:
        required_verification, _ = _validated_verification_rank(
            self.required_verification,
            code="execution.verification_invalid",
            message="required_verification must be V0 through V4",
        )
        object.__setattr__(self, "required_verification", required_verification)
        if self.effective_verification is not None:
            effective_verification, _ = _validated_verification_rank(
                self.effective_verification,
                code="execution.verification_invalid",
                message="effective_verification must be V0 through V4",
            )
            object.__setattr__(
                self,
                "effective_verification",
                effective_verification,
            )
        object.__setattr__(
            self,
            "attestation_refs",
            _immutable_string_collection(self.attestation_refs, "attestation_refs"),
        )
        if self.unmet_reason is not None:
            object.__setattr__(self, "unmet_reason", _task_text(self.unmet_reason, "unmet_reason"))
        _reject_sensitive(self, "execution verification")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionVerificationAttestationV1":
        payload = _mapping_snapshot(
            payload,
            code="execution.schema_invalid",
            location="execution verification",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="execution.schema_invalid",
            location="execution verification",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "execution.schema_invalid",
                f"unexpected execution verification fields: {sorted(unknown)}",
            )
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "execution.schema_invalid",
                "malformed execution verification",
            ) from exc

@dataclass(frozen=True)
class ExecutionOutcomeV1:
    """Bounded outcome/error/timestamp/provenance; no secret material."""

    outcome: ExecutionOutcome
    error_code: str | None = None
    error_message: str | None = None
    recorded_at: str | None = None
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _parse_exact_enum(
                self.outcome,
                ExecutionOutcome,
                code="execution.outcome_invalid",
                message="outcome must be SUCCESS, ERROR, CANCELLED, or UNKNOWN",
            ),
        )
        for name in ("error_code", "error_message", "recorded_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _task_text(value, name))
        object.__setattr__(
            self,
            "provenance_refs",
            _immutable_string_collection(self.provenance_refs, "provenance_refs"),
        )
        _reject_sensitive(self, "execution outcome")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionOutcomeV1":
        payload = _mapping_snapshot(
            payload,
            code="execution.schema_invalid",
            location="execution outcome",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="execution.schema_invalid",
            location="execution outcome",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "execution.schema_invalid",
                f"unexpected execution outcome fields: {sorted(unknown)}",
            )
        raw_outcome = payload.get("outcome")
        values = dict(payload)
        values["outcome"] = _parse_exact_enum(
            raw_outcome,
            ExecutionOutcome,
            code="execution.outcome_invalid",
            message="outcome must be SUCCESS, ERROR, CANCELLED, or UNKNOWN",
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "execution.schema_invalid",
                "malformed execution outcome",
            ) from exc

@dataclass(frozen=True)
class ExecutionEnvelopeV1:
    """Canonical execution-envelope/v1 wire DTO, bound to trusted task/decision authority."""

    schema_version: str
    execution_id: str
    task_id: str
    attempt_id: str
    decision_id: str
    route_id: str
    effort: str
    verification: str
    allowed_context_refs: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    permissions: tuple[str, ...]
    timeout: float
    budget: TaskBudgetV1
    requested_identity: str
    dispatch_state: ExecutionDispatchState
    result_evidence_refs: tuple[str, ...]
    reconciliation_state: str
    reservation_id: str | None = None
    workdir: str | None = None
    trace_context: str | None = None
    telemetry_consent_id: str | None = None
    resolved_identity: str | None = None
    effective_identity: str | None = None
    normalized_error: str | None = None
    trusted_task: InitVar[TaskEnvelope | None] = None
    trusted_decision: InitVar[RouteDecisionV1 | None] = None
    trusted_routes: InitVar[Mapping[str, RouteV1] | None] = None
    activation_status: str = field(default="", init=False)
    activation_block_reason: str | None = field(default=None, init=False)

    def __post_init__(
        self,
        trusted_task: TaskEnvelope | None,
        trusted_decision: RouteDecisionV1 | None,
        trusted_routes: Mapping[str, RouteV1] | None,
    ) -> None:
        if self.schema_version != EXECUTION_ENVELOPE_SCHEMA_VERSION:
            raise DomainValidationError("execution.schema_invalid", "schema_version must be execution-envelope/v1")
        for name in ("execution_id", "task_id", "attempt_id", "decision_id", "reconciliation_state"):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        object.__setattr__(self, "route_id", _validated_route_id(self.route_id, "route_id"))
        object.__setattr__(
            self,
            "effort",
            _validated_exact_label(
                self.effort,
                _EFFORT_LABELS,
                code="execution.effort_invalid",
                message="effort must be E0 through E4",
            ),
        )
        verification, verification_rank = _validated_verification_rank(
            self.verification,
            code="execution.verification_invalid",
            message="verification must be V0 through V4",
        )
        object.__setattr__(self, "verification", verification)
        if type(trusted_task) is not TaskEnvelope:
            raise DomainValidationError("execution.trusted_task_required", "execution requires trusted TaskEnvelope authority")
        if type(trusted_decision) is not RouteDecisionV1:
            raise DomainValidationError("execution.trusted_decision_required", "execution requires trusted RouteDecision authority")
        routes = _revalidated_route_registry(trusted_routes, code="execution.trusted_route_invalid", location="trusted execution routes")
        task = TaskEnvelope.from_mapping(
            _safe_asdict(trusted_task, "trusted task")
        )
        try:
            decision = RouteDecisionV1.from_mapping(
                _safe_asdict(trusted_decision, "trusted route decision"),
                trusted_task=task,
                trusted_routes=routes,
            )
        except DomainValidationError as exc:
            if exc.code == "contract.payload_too_complex":
                raise
            raise DomainValidationError("execution.trusted_decision_invalid", "trusted route decision failed authoritative rehydration") from exc
        _, task_verification_rank = _validated_verification_rank(
            task.verification.minimum,
            code="task.verification_invalid",
            message="verification.minimum must be V0 through V4",
        )
        if (self.task_id != task.task_id or self.attempt_id != decision.attempt_id or self.decision_id != decision.decision_id or self.route_id != decision.selected_route_id or self.effort != task.effort or self.effort != decision.effort or self.verification != decision.verification or verification_rank < task_verification_rank):
            raise DomainValidationError("execution.authorization_mismatch", "execution must bind trusted task, attempt, decision, route, effort, and verification")
        for name in ("requested_identity", "resolved_identity", "effective_identity"):
            value = getattr(self, name)
            if value is not None:
                value = _validated_route_id(value, name)
                if value != self.route_id:
                    raise DomainValidationError("execution.identity_mismatch", "identities must bind the authorized route")
                object.__setattr__(self, name, value)
        if self.reservation_id is not None:
            object.__setattr__(self, "reservation_id", _task_text(self.reservation_id, "reservation_id"))
        state = _parse_exact_enum(
            self.dispatch_state,
            ExecutionDispatchState,
            code="execution.dispatch_state_invalid",
            message="dispatch_state must be a supported execution state",
        )
        object.__setattr__(self, "dispatch_state", state)
        if state is not ExecutionDispatchState.PENDING:
            raise DomainValidationError(
                "execution.external_authority_unavailable",
                "only PENDING execution is representable before an external authority store exists",
            )
        if self.reservation_id is not None:
            raise DomainValidationError("execution.reservation_state_invalid", "PENDING must not hold a reservation")
        object.__setattr__(
            self,
            "timeout",
            _finite_number(
                self.timeout,
                code="execution.timeout_invalid",
                field_name="timeout",
                non_negative=True,
            ),
        )
        if type(self.budget) is not TaskBudgetV1:
            raise DomainValidationError("execution.budget_invalid", "budget must be a validated TaskBudgetV1")
        budget = TaskBudgetV1.from_mapping(
            _safe_asdict(self.budget, "execution budget")
        )
        if budget != task.budget:
            raise DomainValidationError("execution.budget_invalid", "execution budget must match trusted task budget")
        object.__setattr__(self, "budget", budget)
        for name in ("allowed_context_refs", "allowed_tools", "permissions", "result_evidence_refs"):
            object.__setattr__(self, name, _immutable_string_collection(getattr(self, name), name))
        if not set(self.allowed_context_refs).issubset(task.context.allowed_sources) or not set(self.allowed_tools).issubset(task.tools_allowed) or not set(self.permissions).issubset(task.permissions_required):
            raise DomainValidationError("execution.authorization_mismatch", "execution controls exceed trusted task authorization")
        for name in ("workdir", "trace_context", "telemetry_consent_id", "normalized_error"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _task_text(getattr(self, name), name))
        if self.result_evidence_refs or self.normalized_error is not None:
            raise DomainValidationError("execution.state_invariant", "PENDING has no outcome or evidence")
        object.__setattr__(
            self,
            "activation_status",
            "ACTIVATION_BLOCKED_EXTERNAL_AUTHORITY_UNAVAILABLE",
        )
        object.__setattr__(
            self,
            "activation_block_reason",
            "external_authority_store_unimplemented",
        )
        _reject_sensitive(self, "execution envelope")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        trusted_task: TaskEnvelope | None = None,
        trusted_decision: RouteDecisionV1 | None = None,
        trusted_routes: Mapping[str, RouteV1] | None = None,
    ) -> "ExecutionEnvelopeV1":
        payload = _mapping_snapshot(payload, code="execution.schema_invalid", location="execution envelope")
        expected = {field_.name for field_ in fields(cls) if field_.init}
        keys = _validated_mapping_keys(payload, code="execution.schema_invalid", location="execution envelope")
        if keys - expected:
            raise DomainValidationError("execution.schema_invalid", "execution envelope must match canonical v1 fields")
        values = dict(payload)
        if isinstance(values.get("budget"), Mapping):
            values["budget"] = TaskBudgetV1.from_mapping(values["budget"])
        values["dispatch_state"] = _parse_exact_enum(
            values.get("dispatch_state"),
            ExecutionDispatchState,
            code="execution.dispatch_state_invalid",
            message="dispatch_state must be a supported execution state",
        )
        values.update(
            trusted_task=trusted_task,
            trusted_decision=trusted_decision,
            trusted_routes=trusted_routes,
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("execution.schema_invalid", "malformed execution envelope") from exc

    @property
    def outcome(self) -> ExecutionOutcome | None:
        return {ExecutionDispatchState.COMPLETED: ExecutionOutcome.SUCCESS, ExecutionDispatchState.FAILED: ExecutionOutcome.ERROR, ExecutionDispatchState.CANCELLED: ExecutionOutcome.CANCELLED}.get(self.dispatch_state)

    def canonical_object(self) -> dict[str, object]:
        """Return exactly the v1 wire fields; derived activation state is never serialized."""
        payload = {
            field_.name: getattr(self, field_.name)
            for field_ in fields(self)
            if field_.init
        }
        payload["dispatch_state"] = self.dispatch_state.value
        payload["budget"] = _safe_asdict(self.budget, "execution budget")
        return payload
