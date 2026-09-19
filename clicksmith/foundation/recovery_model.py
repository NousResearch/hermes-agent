"""FND-06 Recovery Model — Phase 1 recovery semantics boundary.

FND-06 decides whether a supplied execution disruption has a bounded
recovery action. It does not execute recovery, retry, authorize, persist,
or mutate execution identity/state/evidence.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping

from .execution_identity import ExecutionIdentity, ExecutionIdentityError
from .execution_state import ExecutionState
from .evidence_model import EvidenceRecord, EvidenceModelError

SCHEMA_VERSION = "1.0"
RECOVERY_ACTIONS = frozenset({"RECOVERY_REQUIRED", "NO_RECOVERY", "RETRY_REQUIRED"})
RECOVERABILITY = frozenset({"RECOVERABLE", "NON_RECOVERABLE"})

ERROR_CODES = frozenset({
    "INVALID_IDENTITY",
    "INVALID_RECOVERY_INPUT",
    "INVALID_RECOVERY_CONTEXT",
    "UNSUPPORTED_RECOVERY_ACTION",
    "RECOVERY_EVALUATION_ERROR",
    "AMBIGUOUS_RECOVERY",
    "SCHEMA_ERROR",
    "RECOVERY_EXECUTION_ERROR",
    "RECOVERY_BOUNDARY_VIOLATION",
})


class RecoveryModelError(ValueError):
    """Base FND-06 boundary error."""
    code = "INVALID_RECOVERY_INPUT"


class InvalidIdentityError(RecoveryModelError):
    code = "INVALID_IDENTITY"


class InvalidRecoveryInputError(RecoveryModelError):
    code = "INVALID_RECOVERY_INPUT"


class InvalidRecoveryContextError(RecoveryModelError):
    code = "INVALID_RECOVERY_CONTEXT"


class UnsupportedRecoveryActionError(RecoveryModelError):
    code = "UNSUPPORTED_RECOVERY_ACTION"


class RecoveryEvaluationError(RecoveryModelError):
    code = "RECOVERY_EVALUATION_ERROR"


class AmbiguousRecoveryError(RecoveryModelError):
    code = "AMBIGUOUS_RECOVERY"


class SchemaError(RecoveryModelError):
    code = "SCHEMA_ERROR"


class RecoveryExecutionError(RecoveryModelError):
    code = "RECOVERY_EXECUTION_ERROR"


class RecoveryBoundaryViolationError(RecoveryModelError):
    code = "RECOVERY_BOUNDARY_VIOLATION"


@dataclasses.dataclass(frozen=True)
class FailureOrDisruption:
    """Explicit condition classification supplied to the recovery boundary.

    FND-06 deliberately does not encode provider-specific retry rules. The
    condition is classified by its bounded semantic properties, supplied by
    the caller or an explicitly authorized classifier.
    """

    condition_code: str
    recoverability: str
    retry_required: bool
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        _nonempty("condition_code", self.condition_code)
        if self.recoverability not in RECOVERABILITY:
            raise AmbiguousRecoveryError(
                "recoverability must be exactly RECOVERABLE or NON_RECOVERABLE."
            )
        if type(self.retry_required) is not bool:
            raise InvalidRecoveryInputError("retry_required must be boolean.")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")
        if self.recoverability == "NON_RECOVERABLE" and self.retry_required:
            raise AmbiguousRecoveryError(
                "a non-recoverable condition cannot require retry."
            )


@dataclasses.dataclass(frozen=True)
class RecoveryRequest:
    """Explicit inputs for one recovery evaluation."""

    task: Any
    execution_identity: ExecutionIdentity
    execution_state: ExecutionState
    failure_or_disruption: FailureOrDisruption
    applicable_evidence: tuple[EvidenceRecord, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.execution_identity, ExecutionIdentity):
            raise InvalidIdentityError("execution_identity must be an ExecutionIdentity.")
        try:
            self.execution_identity.validate()
        except ExecutionIdentityError as exc:
            raise InvalidIdentityError(str(exc)) from exc

        _validate_task(self.task, self.execution_identity)
        if not isinstance(self.execution_state, ExecutionState):
            raise InvalidRecoveryContextError(
                "execution_state must be an ExecutionState."
            )
        if self.execution_state.identity != self.execution_identity:
            raise InvalidRecoveryContextError(
                "execution_state.identity must exactly match execution_identity."
            )
        if not isinstance(self.failure_or_disruption, FailureOrDisruption):
            raise InvalidRecoveryInputError(
                "failure_or_disruption must be a FailureOrDisruption."
            )
        self.failure_or_disruption.validate()
        if not isinstance(self.applicable_evidence, tuple):
            raise InvalidRecoveryInputError(
                "applicable_evidence must be an immutable tuple."
            )
        for evidence in self.applicable_evidence:
            if not isinstance(evidence, EvidenceRecord):
                raise InvalidRecoveryInputError(
                    "applicable_evidence contains a non-EvidenceRecord item."
                )
            try:
                evidence.validate()
            except EvidenceModelError as exc:
                raise InvalidRecoveryInputError(
                    "applicable_evidence contains invalid evidence."
                ) from exc
            if evidence.identity_applicability == "EXECUTION":
                if evidence.identity != self.execution_identity:
                    raise InvalidRecoveryContextError(
                        "execution evidence must match the recovery execution identity."
                    )


@dataclasses.dataclass(frozen=True)
class RecoveryDecision:
    """Immutable semantic result; no recovery action is executed."""

    execution_identity: ExecutionIdentity
    action: str
    condition_code: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if not isinstance(self.execution_identity, ExecutionIdentity):
            raise InvalidIdentityError("decision execution_identity must be an ExecutionIdentity.")
        try:
            self.execution_identity.validate()
        except ExecutionIdentityError as exc:
            raise InvalidIdentityError(str(exc)) from exc
        if self.action not in RECOVERY_ACTIONS:
            raise UnsupportedRecoveryActionError(
                "action must be one of the three canonical Phase 1 recovery actions."
            )
        _nonempty("condition_code", self.condition_code)
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "action": self.action,
            "condition_code": self.condition_code,
            "execution_identity": self.execution_identity.to_dict(),
            "schema_version": self.schema_version,
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


def evaluate(request: RecoveryRequest) -> RecoveryDecision:
    """Evaluate explicit recovery semantics without executing the result."""
    if not isinstance(request, RecoveryRequest):
        raise SchemaError("request must be a RecoveryRequest.")
    try:
        request.validate()
        failure = request.failure_or_disruption
        if failure.recoverability == "NON_RECOVERABLE":
            action = "NO_RECOVERY"
        elif failure.retry_required:
            action = "RETRY_REQUIRED"
        else:
            action = "RECOVERY_REQUIRED"
        decision = RecoveryDecision(
            execution_identity=request.execution_identity,
            action=action,
            condition_code=failure.condition_code,
        )
        decision.validate()
        return decision
    except RecoveryModelError:
        raise
    except Exception as exc:
        raise RecoveryEvaluationError("recovery evaluation failed.") from exc


def _validate_task(task: Any, identity: ExecutionIdentity) -> None:
    if task is None:
        raise InvalidRecoveryInputError("task is required.")
    task_id = getattr(task, "task_id", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise InvalidRecoveryInputError("task must expose a non-empty task_id.")
    if task_id != identity.task_id:
        raise InvalidRecoveryContextError("task.task_id must match execution_identity.task_id.")
    validate = getattr(task, "validate", None)
    if not callable(validate):
        raise InvalidRecoveryInputError("task must expose validate().")
    try:
        validate()
    except Exception as exc:
        raise InvalidRecoveryInputError("task contract validation failed.") from exc


def _nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string.")
