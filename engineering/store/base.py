"""Storage-independent contract for durable engineering facts."""

from __future__ import annotations

from typing import Protocol, Sequence

from engineering.domain import (
    Evidence,
    ReviewResult,
    VerificationResult,
    WorkflowRun,
)


class EngineeringStoreError(Exception):
    """Base error for EngineeringStore operations."""


class WorkflowAlreadyExists(EngineeringStoreError):
    """Raised when creating a workflow whose identity already exists."""


class WorkflowNotFound(EngineeringStoreError):
    """Raised when a requested workflow does not exist."""


class EvidenceNotFound(EngineeringStoreError):
    """Raised when requested evidence does not exist."""


class VerificationNotFound(EngineeringStoreError):
    """Raised when a requested verification result does not exist."""


class ReviewNotFound(EngineeringStoreError):
    """Raised when a requested review result does not exist."""


class EngineeringStoreConflict(EngineeringStoreError):
    """Base error for immutable-result persistence conflicts."""


class VerificationAlreadyExists(EngineeringStoreConflict):
    """Raised when a verification already exists for an attempt."""


class ReviewAlreadyExists(EngineeringStoreConflict):
    """Raised when a review already exists for an attempt."""


class EvidenceAlreadyExists(EngineeringStoreConflict):
    """Raised when an evidence identity already exists in the store."""


class EngineeringStoreCorruption(EngineeringStoreError):
    """Raised when persisted Engineering data cannot be reconstructed."""


class InvalidWorkflowIdentifier(EngineeringStoreError, ValueError):
    """Raised when a workflow identity is unsafe for path composition."""


class EngineeringStore(Protocol):
    """Capabilities required from an Engineering persistence provider."""

    def create_workflow(self, workflow: WorkflowRun) -> None: ...

    def get_workflow(self, workflow_run_id: str) -> WorkflowRun: ...

    def save_workflow(self, workflow: WorkflowRun) -> None: ...

    def append_evidence(self, evidence: Evidence) -> None:
        """Append evidence, rejecting an ID already present in this store."""
        ...

    def list_evidence(
        self,
        workflow_run_id: str,
        attempt: int | None = None,
    ) -> Sequence[Evidence]: ...

    def get_evidence(self, evidence_id: str) -> Evidence: ...

    def save_verification(self, result: VerificationResult) -> None: ...

    def get_verification(
        self,
        workflow_run_id: str,
        attempt: int,
    ) -> VerificationResult: ...

    def save_review(self, result: ReviewResult) -> None: ...

    def get_review(
        self,
        workflow_run_id: str,
        attempt: int,
    ) -> ReviewResult: ...
