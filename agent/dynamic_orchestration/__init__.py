"""Pure contracts for proposed dynamic multimodel orchestration.

This package intentionally has no runtime wiring, provider calls, persistence,
or credential-pool mutation. The package root preserves the original public
import surface while cohesive modules own each contract definition.
"""

from __future__ import annotations

from .capacity import (
    CAPACITY_OBSERVATION_SCHEMA_VERSION,
    CAPACITY_POOL_SCHEMA_VERSION,
    CAPACITY_VIEW_SCHEMA_VERSION,
    CapacityCompleteness,
    CapacityConfidence,
    CapacityConflictDisposition,
    CapacityDimension,
    CapacityFreshness,
    CapacityMetric,
    CapacityObservationV1,
    CapacityReservationV1,
    CapacityValueV1,
    CapacityViewV1,
    CircuitBreakerStatus,
    CircuitBreakerV1,
    PoolCapacityV1,
    build_capacity_view,
)

from .decision import (
    DecisionRelation,
    RouteDecisionV1,
    replan_after_capacity_exhaustion,
)

from .eligibility import (
    CandidateEvaluation,
    CandidateScoreV1,
    EligibilityDisposition,
    ErrorKind,
    InitialSelectionTriggerV1,
    RouteEligibilityFactsV1,
    RuntimeErrorClassificationV1,
    evaluate_route_eligibility,
    score_eligible_candidates,
)

from .execution import (
    EXECUTION_ENVELOPE_SCHEMA_VERSION,
    ExecutionDispatchState,
    ExecutionEnvelopeV1,
    ExecutionIdentityAttestationV1,
    ExecutionOutcome,
    ExecutionOutcomeV1,
    ExecutionVerificationAttestationV1,
)

from .quality import (
    AcceptanceThresholdV1,
    CompensationEscalationV1,
    EscalationAction,
    IndependentReviewAttestationV1,
    QualityCompensationPlanV1,
)

from .route import (
    RouteV1,
)

from .state import (
    AttemptState,
    AttemptStateEvent,
    CredentialState,
    CredentialStateEvent,
    E,
    ReservationState,
    ReservationStateEvent,
    ReviewState,
    ReviewStateEvent,
    RouteState,
    RouteStateEvent,
    S,
    TaskState,
    TaskStateEvent,
    validate_attempt_transition,
    validate_credential_transition,
    validate_reservation_transition,
    validate_review_transition,
    validate_route_transition,
    validate_task_transition,
)

from .task import (
    AuditedModelJustification,
    TaskBudgetV1,
    TaskContextV1,
    TaskEnvelope,
    TaskPrivacyV1,
    TaskRiskV1,
    TaskVerificationV1,
)

from .validation import (
    DomainValidationError,
    QUALITY_COMPENSATION_SCHEMA_VERSION,
    ROUTE_CANONICALIZATION_VERSION,
    _MAX_BREAKER_PROBE_BUDGET as _MAX_BREAKER_PROBE_BUDGET,
    _MAX_CAPACITY_CONCURRENCY as _MAX_CAPACITY_CONCURRENCY,
    _MAX_CONTEXT_TOKENS as _MAX_CONTEXT_TOKENS,
    _MAX_CONTRACT_VERSION as _MAX_CONTRACT_VERSION,
    _safe_asdict as _safe_asdict,
)

__all__ = (
    "AcceptanceThresholdV1",
    "AttemptState",
    "AttemptStateEvent",
    "AuditedModelJustification",
    "CAPACITY_OBSERVATION_SCHEMA_VERSION",
    "CAPACITY_POOL_SCHEMA_VERSION",
    "CAPACITY_VIEW_SCHEMA_VERSION",
    "CandidateEvaluation",
    "CandidateScoreV1",
    "CapacityCompleteness",
    "CapacityConfidence",
    "CapacityConflictDisposition",
    "CapacityDimension",
    "CapacityFreshness",
    "CapacityMetric",
    "CapacityObservationV1",
    "CapacityReservationV1",
    "CapacityValueV1",
    "CapacityViewV1",
    "CircuitBreakerStatus",
    "CircuitBreakerV1",
    "CompensationEscalationV1",
    "CredentialState",
    "CredentialStateEvent",
    "DecisionRelation",
    "DomainValidationError",
    "E",
    "EXECUTION_ENVELOPE_SCHEMA_VERSION",
    "EligibilityDisposition",
    "ErrorKind",
    "EscalationAction",
    "ExecutionDispatchState",
    "ExecutionEnvelopeV1",
    "ExecutionIdentityAttestationV1",
    "ExecutionOutcome",
    "ExecutionOutcomeV1",
    "ExecutionVerificationAttestationV1",
    "IndependentReviewAttestationV1",
    "InitialSelectionTriggerV1",
    "PoolCapacityV1",
    "QUALITY_COMPENSATION_SCHEMA_VERSION",
    "QualityCompensationPlanV1",
    "ROUTE_CANONICALIZATION_VERSION",
    "ReservationState",
    "ReservationStateEvent",
    "ReviewState",
    "ReviewStateEvent",
    "RouteDecisionV1",
    "RouteEligibilityFactsV1",
    "RouteState",
    "RouteStateEvent",
    "RouteV1",
    "RuntimeErrorClassificationV1",
    "S",
    "TaskBudgetV1",
    "TaskContextV1",
    "TaskEnvelope",
    "TaskPrivacyV1",
    "TaskRiskV1",
    "TaskState",
    "TaskStateEvent",
    "TaskVerificationV1",
    "build_capacity_view",
    "evaluate_route_eligibility",
    "replan_after_capacity_exhaustion",
    "score_eligible_candidates",
    "validate_attempt_transition",
    "validate_credential_transition",
    "validate_reservation_transition",
    "validate_review_transition",
    "validate_route_transition",
    "validate_task_transition",
)
