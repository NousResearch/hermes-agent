"""Task and attempt status enums with legal transition rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

RunStatus = str
TaskStatus = str
AttemptStatus = str

RUN_CREATED: Final = "created"
RUN_RUNNING: Final = "running"
RUN_COMPLETED: Final = "completed"
RUN_FAILED: Final = "failed"
RUN_CANCELLED: Final = "cancelled"

TASK_CREATED: Final = "created"
TASK_RUNNING: Final = "running"
TASK_BLOCKED: Final = "blocked"
TASK_COMPLETED: Final = "completed"
TASK_FAILED: Final = "failed"
TASK_CANCELLED: Final = "cancelled"

ATTEMPT_CREATED: Final = "created"
ATTEMPT_RUNNING: Final = "running"
ATTEMPT_RESULT_SUBMITTED: Final = "result_submitted"
ATTEMPT_VERIFICATION_PASSED: Final = "verification_passed"
ATTEMPT_VERIFICATION_FAILED: Final = "verification_failed"
ATTEMPT_HEAL_REQUIRED: Final = "heal_required"
ATTEMPT_COMPLETED: Final = "completed"
ATTEMPT_FAILED: Final = "failed"
ATTEMPT_CANCELLED: Final = "cancelled"

RUN_STATUSES: frozenset[str] = frozenset(
    {
        RUN_CREATED,
        RUN_RUNNING,
        RUN_COMPLETED,
        RUN_FAILED,
        RUN_CANCELLED,
    }
)

TASK_STATUSES: frozenset[str] = frozenset(
    {
        TASK_CREATED,
        TASK_RUNNING,
        TASK_BLOCKED,
        TASK_COMPLETED,
        TASK_FAILED,
        TASK_CANCELLED,
    }
)

ATTEMPT_STATUSES: frozenset[str] = frozenset(
    {
        ATTEMPT_CREATED,
        ATTEMPT_RUNNING,
        ATTEMPT_RESULT_SUBMITTED,
        ATTEMPT_VERIFICATION_PASSED,
        ATTEMPT_VERIFICATION_FAILED,
        ATTEMPT_HEAL_REQUIRED,
        ATTEMPT_COMPLETED,
        ATTEMPT_FAILED,
        ATTEMPT_CANCELLED,
    }
)

TASK_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED}
)

RUN_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED}
)

ATTEMPT_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {ATTEMPT_COMPLETED, ATTEMPT_FAILED, ATTEMPT_CANCELLED}
)

TASK_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    TASK_CREATED: frozenset({TASK_RUNNING, TASK_CANCELLED}),
    TASK_RUNNING: frozenset(
        {TASK_BLOCKED, TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED}
    ),
    TASK_BLOCKED: frozenset({TASK_RUNNING, TASK_CANCELLED}),
}

RUN_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    RUN_CREATED: frozenset({RUN_RUNNING, RUN_COMPLETED, RUN_CANCELLED}),
    RUN_RUNNING: frozenset({RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED}),
}

ATTEMPT_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    ATTEMPT_CREATED: frozenset({ATTEMPT_RUNNING, ATTEMPT_CANCELLED}),
    ATTEMPT_RUNNING: frozenset(
        {ATTEMPT_RESULT_SUBMITTED, ATTEMPT_FAILED, ATTEMPT_CANCELLED}
    ),
    ATTEMPT_RESULT_SUBMITTED: frozenset(
        {ATTEMPT_VERIFICATION_PASSED, ATTEMPT_VERIFICATION_FAILED, ATTEMPT_HEAL_REQUIRED}
    ),
    ATTEMPT_VERIFICATION_PASSED: frozenset({ATTEMPT_COMPLETED}),
    ATTEMPT_VERIFICATION_FAILED: frozenset(
        {ATTEMPT_HEAL_REQUIRED, ATTEMPT_FAILED}
    ),
    ATTEMPT_HEAL_REQUIRED: frozenset({ATTEMPT_FAILED}),
}


class HTRStateError(Exception):
    """Base error for HTR state and event operations."""


class InvalidTransition(HTRStateError):
    """Raised when a lifecycle status transition is not allowed."""


class EventConflict(HTRStateError):
    """Raised when the same event_id is reused with different semantics."""


class AttemptAlreadyRegistered(HTRStateError):
    """Raised when an attempt_id is registered more than once."""


class EventValidationError(HTRStateError):
    """Raised when an event fails schema validation."""


ERROR_CODE_RUN_FINALIZED: Final = "RUN_FINALIZED"
ERROR_CODE_RUN_SEAL_BLOCKED: Final = "RUN_SEAL_BLOCKED"


class RunFinalizedError(HTRStateError):
    """Raised when a valid final closure seals the run against mutation."""

    def __init__(
        self,
        message: str = "Original run is finalized and cannot be modified.",
        *,
        run_id: str | None = None,
        error_code: str = ERROR_CODE_RUN_FINALIZED,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.error_code = error_code


class RunSealBlockedError(HTRStateError):
    """Raised when closure state is untrusted or indeterminate for mutation."""

    def __init__(
        self,
        message: str = "Run closure state is untrusted; mutation blocked.",
        *,
        run_id: str | None = None,
        error_code: str = ERROR_CODE_RUN_SEAL_BLOCKED,
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.error_code = error_code
        self.reason_codes = reason_codes


ERROR_CODE_APPROVAL_VALIDATION: Final = "APPROVAL_VALIDATION_FAILED"
ERROR_CODE_APPROVAL_CONFLICT: Final = "APPROVAL_CONFLICT"
ERROR_CODE_APPROVAL_FINALIZED: Final = "APPROVAL_FINALIZED_RUN_BLOCKED"
ERROR_CODE_APPROVAL_STATE: Final = "APPROVAL_ILLEGAL_STATE"


class ApprovalControlError(HTRStateError):
    """Base error for Task 24 approval-control operations."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_APPROVAL_VALIDATION,
        approval_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.approval_id = approval_id


class ApprovalValidationError(ApprovalControlError):
    """Raised when approval inputs or derived validation fail."""


class ApprovalConflictError(ApprovalControlError):
    """Raised when an immutable record replay conflicts with existing evidence."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_APPROVAL_CONFLICT, **kwargs)


class ApprovalFinalizedRunError(ApprovalControlError):
    """Raised when a lifecycle approval targets a finalized original run."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_APPROVAL_FINALIZED, **kwargs)


class ApprovalStateError(ApprovalControlError):
    """Raised when an approval transition is not legal for current evidence."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_APPROVAL_STATE, **kwargs)


ERROR_CODE_INVOKE_STALE: Final = "INVOKE_STALE_REJECTION"
ERROR_CODE_INVOKE_AMBIGUOUS: Final = "INVOKE_AMBIGUOUS_OUTCOME"
ERROR_CODE_INVOKE_OUTCOME_PERSISTENCE: Final = "INVOKE_OUTCOME_PERSISTENCE_FAILED"
ERROR_CODE_INVOKE_CLEANUP_DURABILITY: Final = "INVOKE_CLEANUP_DURABILITY_FAILED"


class InvokeRunCompletionError(HTRStateError):
    """Base error for Task 25 human-gated run-completion invoke."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        approval_id: str | None = None,
        claim_id: str | None = None,
        reason_code: str | None = None,
        mutation_may_have_committed: bool = False,
        safe_to_retry: bool = False,
        outcome_evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.approval_id = approval_id
        self.claim_id = claim_id
        self.reason_code = reason_code
        self.mutation_may_have_committed = mutation_may_have_committed
        self.safe_to_retry = safe_to_retry
        self.outcome_evidence = outcome_evidence


class InvokeStaleApprovalError(InvokeRunCompletionError):
    """Raised when pre-claim validation rejects the approval as stale."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_INVOKE_STALE,
            safe_to_retry=False,
            **kwargs,
        )


class InvokeAmbiguousOutcomeError(InvokeRunCompletionError):
    """Raised after a durable claim when invoke outcome is ambiguous."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_INVOKE_AMBIGUOUS,
            safe_to_retry=False,
            **kwargs,
        )


class InvokeOutcomePersistenceError(InvokeRunCompletionError):
    """Raised when outcome.json cannot be persisted after claim."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_INVOKE_OUTCOME_PERSISTENCE,
            safe_to_retry=False,
            **kwargs,
        )


class InvokeCleanupDurabilityError(InvokeRunCompletionError):
    """Raised when marker cleanup fails after a consumed outcome."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_INVOKE_CLEANUP_DURABILITY,
            safe_to_retry=False,
            **kwargs,
        )


@dataclass(frozen=True)
class InvokeRunCompletionResult:
    """Immutable success result for Task 25 run-completion invoke."""

    approval_id: str
    claim_id: str
    run_id: str
    event_id: str
    completion_record_fingerprint: str
    event_semantic_fingerprint: str
    pre_observation_digest: str
    post_observation_digest: str
    outcome_digest: str


ERROR_CODE_RECONCILIATION_INSPECTION: Final = "RECONCILIATION_INSPECTION_FAILED"
ERROR_CODE_RECONCILIATION_UNSUPPORTED: Final = "RECONCILIATION_UNSUPPORTED_APPROVAL"
ERROR_CODE_RECONCILIATION_EVIDENCE: Final = "RECONCILIATION_EVIDENCE_INTEGRITY"


class ReconciliationInspectionError(HTRStateError):
    """Base error for Task 26A read-only reconciliation inspection."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_RECONCILIATION_INSPECTION,
        approval_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.approval_id = approval_id


class ReconciliationUnsupportedApprovalError(ReconciliationInspectionError):
    """Raised when approval is outside Task 26A pilot scope."""

    def __init__(self, message: str, *, approval_id: str | None = None) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_RECONCILIATION_UNSUPPORTED,
            approval_id=approval_id,
        )


class ReconciliationEvidenceIntegrityError(ReconciliationInspectionError):
    """Raised when inspection cannot resolve trustworthy identity/evidence."""

    def __init__(self, message: str, *, approval_id: str | None = None) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_RECONCILIATION_EVIDENCE,
            approval_id=approval_id,
        )


@dataclass(frozen=True)
class RunCompletionReconciliationInspection:
    """Immutable read-only reconciliation inspection for Task 25 pilot."""

    inspection_schema_version: str
    inspection_projection_version: str
    approval_id: str
    approval_digest: str
    claim_id: str | None
    claim_digest: str | None
    outcome_class: str | None
    outcome_digest: str | None
    run_id: str
    bound_api: str
    event_id: str
    htr_runs_root_path_digest: str
    approval_control_state: str
    marker_state: str
    lifecycle_evidence_state: str
    integrity_state: str
    overall_classification: str
    reason_codes: tuple[str, ...]
    observed_completion_record_fingerprint: str | None
    observed_event_semantic_fingerprint: str | None
    observed_manifest_status: str | None
    current_observation_semantic_digest: str | None
    source_observation_digest: str
    inspection_semantic_digest: str
    safe_to_retry: bool
    marker_disposition_allowed: bool
    reconciliation_case_required: bool
    recovery_protocol_required: bool
    observed_at: str | None


ERROR_CODE_RECONCILIATION_VALIDATION: Final = "RECONCILIATION_VALIDATION_FAILED"
ERROR_CODE_RECONCILIATION_CONFLICT: Final = "RECONCILIATION_CONFLICT"
ERROR_CODE_RECONCILIATION_STATE: Final = "RECONCILIATION_ILLEGAL_STATE"
ERROR_CODE_RECONCILIATION_DURABILITY: Final = "RECONCILIATION_DURABILITY_FAILED"


class ReconciliationStateError(HTRStateError):
    """Base error for Task 26B reconciliation case operations."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_RECONCILIATION_STATE,
        case_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.case_id = case_id


class ReconciliationValidationError(ReconciliationStateError):
    """Raised when reconciliation inputs or derived validation fail."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_RECONCILIATION_VALIDATION, **kwargs)


class ReconciliationConflictError(ReconciliationStateError):
    """Raised when an immutable reconciliation record conflicts with existing evidence."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_RECONCILIATION_CONFLICT, **kwargs)


DurabilityStage = Literal[
    "record_write",
    "record_fsync",
    "case_dir_fsync",
    "control_dir_fsync",
    "parent_dir_fsync",
]

ExactReplayStatus = Literal["yes", "no", "indeterminate"]

ReconciliationRecordName = Literal["open.json", "observation.json", "decision.json"]


class ReconciliationDurabilityError(ReconciliationStateError):
    """Raised when reconciliation control-record durability cannot be confirmed."""

    def __init__(
        self,
        message: str,
        *,
        record_may_have_committed: bool,
        exact_replay_status: ExactReplayStatus,
        durability_stage: DurabilityStage,
        case_id: str,
        record_name: ReconciliationRecordName,
    ) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_RECONCILIATION_DURABILITY,
            case_id=case_id,
        )
        self.record_may_have_committed = record_may_have_committed
        self.exact_replay_status = exact_replay_status
        self.durability_stage = durability_stage
        self.record_name = record_name


@dataclass(frozen=True)
class ReconciliationWriteMetadata:
    exact_replay: bool
    exact_replay_status: ExactReplayStatus
    record_may_have_committed: bool
    durability_indeterminate: bool


@dataclass(frozen=True)
class ReconciliationCaseOpenRecord:
    case_id: str
    case_open_digest: str
    approval_id: str
    approval_issue_digest: str
    run_id: str
    bound_api: str
    event_id: str
    htr_runs_root_path_digest: str
    htr_project_dir_path_digest: str
    opened_by: str
    scope_reason: str
    opened_at: str


@dataclass(frozen=True)
class ReconciliationObservationRecord:
    case_id: str
    observation_digest: str
    case_open_digest: str
    observed_by: str
    observed_at: str
    inspection_semantic_digest: str


@dataclass(frozen=True)
class ReconciliationDecisionRecord:
    case_id: str
    decision_digest: str
    case_open_digest: str
    observation_digest: str
    decided_by: str
    decided_at: str
    requested_decision_class: str
    decision_class: str
    derived_rationale_codes: tuple[str, ...]


@dataclass(frozen=True)
class ReconciliationCaseBundle:
    case_id: str
    open_record: ReconciliationCaseOpenRecord
    observation_record: ReconciliationObservationRecord | None
    decision_record: ReconciliationDecisionRecord | None


ERROR_CODE_MARKER_DISPOSITION_VALIDATION: Final = "MARKER_DISPOSITION_VALIDATION_FAILED"
ERROR_CODE_MARKER_DISPOSITION_CONFLICT: Final = "MARKER_DISPOSITION_CONFLICT"
ERROR_CODE_MARKER_DISPOSITION_STATE: Final = "MARKER_DISPOSITION_ILLEGAL_STATE"
ERROR_CODE_MARKER_DISPOSITION_DURABILITY: Final = "MARKER_DISPOSITION_DURABILITY_FAILED"


MarkerDispositionDurabilityStage = Literal[
    "record_write",
    "record_fsync",
    "disposition_dir_fsync",
    "control_dir_fsync",
    "parent_dir_fsync",
    "lock_directory_fsync",
]

MarkerDispositionRecordName = Literal[
    "request.json",
    "issue.json",
    "revoke.json",
    "claim.json",
    "attempt.json",
    "outcome.json",
]


class MarkerDispositionStateError(HTRStateError):
    """Base error for Task 26C marker disposition operations."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_MARKER_DISPOSITION_STATE,
        disposition_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.disposition_id = disposition_id


class MarkerDispositionValidationError(MarkerDispositionStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_MARKER_DISPOSITION_VALIDATION, **kwargs)


class MarkerDispositionConflictError(MarkerDispositionStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_MARKER_DISPOSITION_CONFLICT, **kwargs)


class MarkerDispositionDurabilityError(MarkerDispositionStateError):
    def __init__(
        self,
        message: str,
        *,
        disposition_id: str,
        record_name: MarkerDispositionRecordName,
        durability_stage: MarkerDispositionDurabilityStage,
        record_may_have_committed: bool,
        exact_replay_status: ExactReplayStatus,
        marker_may_have_been_removed: bool = False,
        marker_directory_durability_indeterminate: bool = False,
    ) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_MARKER_DISPOSITION_DURABILITY,
            disposition_id=disposition_id,
        )
        self.record_name = record_name
        self.durability_stage = durability_stage
        self.record_may_have_committed = record_may_have_committed
        self.exact_replay_status = exact_replay_status
        self.marker_may_have_been_removed = marker_may_have_been_removed
        self.marker_directory_durability_indeterminate = marker_directory_durability_indeterminate


@dataclass(frozen=True)
class MarkerDispositionWriteMetadata:
    exact_replay: bool
    exact_replay_status: ExactReplayStatus
    record_may_have_committed: bool
    durability_indeterminate: bool


@dataclass(frozen=True)
class MarkerDispositionRequestRecord:
    disposition_id: str
    request_digest: str
    reconciliation_case_id: str
    run_id: str
    requested_by: str
    requested_at: str


@dataclass(frozen=True)
class MarkerDispositionIssueRecord:
    disposition_id: str
    disposition_approval_id: str
    issue_digest: str
    issued_by: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class MarkerDispositionClaimRecord:
    disposition_id: str
    claim_id: str
    claim_digest: str
    claimant: str
    claimed_at: str


@dataclass(frozen=True)
class MarkerDispositionAttemptRecord:
    disposition_id: str
    attempt_id: str
    attempt_digest: str
    executor: str
    attempted_at: str


@dataclass(frozen=True)
class MarkerDispositionOutcomeRecord:
    disposition_id: str
    outcome_class: str
    outcome_digest: str
    recorded_at: str


@dataclass(frozen=True)
class MarkerDispositionBundle:
    disposition_id: str
    request_record: MarkerDispositionRequestRecord | None
    issue_record: MarkerDispositionIssueRecord | None
    revoke_record: dict[str, Any] | None
    claim_record: MarkerDispositionClaimRecord | None
    attempt_record: MarkerDispositionAttemptRecord | None
    outcome_record: MarkerDispositionOutcomeRecord | None


@dataclass(frozen=True)
class MarkerDispositionExecutionResult:
    disposition_id: str
    outcome_class: str
    outcome_digest: str
    exact_replay: bool
    marker_removed_by_this_execution: bool


@dataclass(frozen=True)
class MarkerDispositionReconcileResult:
    disposition_id: str
    classification: str
    outcome_record: MarkerDispositionOutcomeRecord | None
    marker_present: bool | None
    notes: tuple[str, ...] = ()


ERROR_CODE_RECOVERY_RUN_VALIDATION: Final = "RECOVERY_RUN_VALIDATION_FAILED"
ERROR_CODE_RECOVERY_RUN_CONFLICT: Final = "RECOVERY_RUN_CONFLICT"
ERROR_CODE_RECOVERY_RUN_STATE: Final = "RECOVERY_RUN_ILLEGAL_STATE"
ERROR_CODE_RECOVERY_RUN_DURABILITY: Final = "RECOVERY_RUN_DURABILITY_FAILED"


RecoveryRunDurabilityStage = Literal[
    "record_write",
    "record_fsync",
    "recovery_case_dir_fsync",
    "control_dir_fsync",
    "parent_dir_fsync",
    "successor_root_reservation",
    "successor_root_fsync",
    "runs_root_fsync",
    "recovery_origin_write",
    "bootstrap_write",
]

RecoveryRunRecordName = Literal[
    "request.json",
    "issue.json",
    "revoke.json",
    "claim.json",
    "attempt.json",
    "outcome.json",
    "recovery_origin.json",
]


class RecoveryRunStateError(HTRStateError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_RECOVERY_RUN_STATE,
        recovery_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.recovery_request_id = recovery_request_id


class RecoveryRunValidationError(RecoveryRunStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_RECOVERY_RUN_VALIDATION, **kwargs)


class RecoveryRunConflictError(RecoveryRunStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_RECOVERY_RUN_CONFLICT, **kwargs)


class RecoveryRunDurabilityError(RecoveryRunStateError):
    def __init__(
        self,
        message: str,
        *,
        recovery_request_id: str,
        successor_run_id: str | None = None,
        record_name: RecoveryRunRecordName | str,
        durability_stage: RecoveryRunDurabilityStage | str,
        record_may_have_committed: bool,
        successor_may_have_been_created: bool,
        exact_replay_status: ExactReplayStatus,
    ) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_RECOVERY_RUN_DURABILITY,
            recovery_request_id=recovery_request_id,
        )
        self.successor_run_id = successor_run_id
        self.record_name = record_name
        self.durability_stage = durability_stage
        self.record_may_have_committed = record_may_have_committed
        self.successor_may_have_been_created = successor_may_have_been_created
        self.exact_replay_status = exact_replay_status


@dataclass(frozen=True)
class RecoveryRunWriteMetadata:
    exact_replay: bool
    exact_replay_status: ExactReplayStatus
    record_may_have_committed: bool
    durability_indeterminate: bool


@dataclass(frozen=True)
class RecoveryRunRequestRecord:
    recovery_request_id: str
    request_digest: str
    reconciliation_case_id: str
    recovery_of_run_id: str
    successor_run_id: str
    recovery_scope: str
    requested_by: str
    requested_at: str


@dataclass(frozen=True)
class RecoveryRunIssueRecord:
    recovery_request_id: str
    recovery_approval_id: str
    issue_digest: str
    issued_by: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class RecoveryRunClaimRecord:
    recovery_request_id: str
    claim_id: str
    claim_digest: str
    claimant: str
    claimed_at: str


@dataclass(frozen=True)
class RecoveryRunAttemptRecord:
    recovery_request_id: str
    attempt_id: str
    attempt_digest: str
    executor: str
    attempted_at: str


@dataclass(frozen=True)
class RecoveryRunOutcomeRecord:
    recovery_request_id: str
    outcome_class: str
    outcome_digest: str
    recorded_at: str


@dataclass(frozen=True)
class RecoveryRunBundle:
    recovery_request_id: str
    request_record: RecoveryRunRequestRecord | None
    issue_record: RecoveryRunIssueRecord | None
    revoke_record: dict[str, Any] | None
    claim_record: RecoveryRunClaimRecord | None
    attempt_record: RecoveryRunAttemptRecord | None
    outcome_record: RecoveryRunOutcomeRecord | None


@dataclass(frozen=True)
class RecoveryRunExecutionResult:
    recovery_request_id: str
    successor_run_id: str
    outcome_class: str
    outcome_digest: str
    exact_replay: bool


@dataclass(frozen=True)
class RecoveryRunReconcileResult:
    recovery_request_id: str
    classification: str
    outcome_record: RecoveryRunOutcomeRecord | None
    successor_present: bool | None
    notes: tuple[str, ...] = ()


ERROR_CODE_BOUNDED_ACTION_VALIDATION: Final = "BOUNDED_ACTION_VALIDATION_FAILED"
ERROR_CODE_BOUNDED_ACTION_CONFLICT: Final = "BOUNDED_ACTION_CONFLICT"
ERROR_CODE_BOUNDED_ACTION_STATE: Final = "BOUNDED_ACTION_ILLEGAL_STATE"
ERROR_CODE_BOUNDED_ACTION_DURABILITY: Final = "BOUNDED_ACTION_DURABILITY_FAILED"
ERROR_CODE_BOUNDED_ACTION_PRECONDITION: Final = "BOUNDED_ACTION_PRECONDITION_FAILED"


class BoundedActionStateError(HTRStateError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = ERROR_CODE_BOUNDED_ACTION_STATE,
        proposal_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.proposal_id = proposal_id


class BoundedActionValidationError(BoundedActionStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_BOUNDED_ACTION_VALIDATION, **kwargs)


class BoundedActionConflictError(BoundedActionStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_BOUNDED_ACTION_CONFLICT, **kwargs)


class BoundedActionPreconditionError(BoundedActionStateError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code=ERROR_CODE_BOUNDED_ACTION_PRECONDITION, **kwargs)


class BoundedActionDurabilityError(BoundedActionStateError):
    def __init__(
        self,
        message: str,
        *,
        proposal_id: str,
        record_name: str,
        durability_stage: str,
        record_may_have_committed: bool,
    ) -> None:
        super().__init__(
            message,
            error_code=ERROR_CODE_BOUNDED_ACTION_DURABILITY,
            proposal_id=proposal_id,
        )
        self.record_name = record_name
        self.durability_stage = durability_stage
        self.record_may_have_committed = record_may_have_committed


def is_valid_task_transition(from_status: str, to_status: str) -> bool:
    """Return True when *to_status* is legal from *from_status*."""
    if from_status not in TASK_STATUSES or to_status not in TASK_STATUSES:
        return False
    allowed = TASK_LEGAL_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


def is_valid_attempt_transition(from_status: str, to_status: str) -> bool:
    """Return True when *to_status* is legal from *from_status*."""
    if from_status not in ATTEMPT_STATUSES or to_status not in ATTEMPT_STATUSES:
        return False
    allowed = ATTEMPT_LEGAL_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


def assert_valid_task_transition(from_status: str, to_status: str) -> None:
    """Raise :class:`InvalidTransition` when the transition is not legal."""
    if not is_valid_task_transition(from_status, to_status):
        raise InvalidTransition(
            f"illegal task transition: {from_status!r} -> {to_status!r}"
        )


def assert_valid_attempt_transition(from_status: str, to_status: str) -> None:
    """Raise :class:`InvalidTransition` when the transition is not legal."""
    if not is_valid_attempt_transition(from_status, to_status):
        raise InvalidTransition(
            f"illegal attempt transition: {from_status!r} -> {to_status!r}"
        )


def is_valid_run_transition(from_status: str, to_status: str) -> bool:
    """Return True when *to_status* is legal from *from_status*."""
    if from_status not in RUN_STATUSES or to_status not in RUN_STATUSES:
        return False
    allowed = RUN_LEGAL_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


def assert_valid_run_transition(from_status: str, to_status: str) -> None:
    """Raise :class:`InvalidTransition` when the transition is not legal."""
    if not is_valid_run_transition(from_status, to_status):
        raise InvalidTransition(
            f"illegal run transition: {from_status!r} -> {to_status!r}"
        )


def is_terminal_task_status(status: str) -> bool:
    """Return True when *status* is a terminal task status."""
    return status in TASK_TERMINAL_STATUSES


def is_terminal_attempt_status(status: str) -> bool:
    """Return True when *status* is a terminal attempt status."""
    return status in ATTEMPT_TERMINAL_STATUSES


def is_terminal_run_status(status: str) -> bool:
    """Return True when *status* is a terminal run status."""
    return status in RUN_TERMINAL_STATUSES
