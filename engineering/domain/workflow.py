"""Hermes-independent engineering workflow domain model."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4


class WorkflowState(str, Enum):
    """Engineering workflow states; these are not Hermes turn states."""

    CREATED = "CREATED"
    UNDERSTANDING = "UNDERSTANDING"
    EXPLORING = "EXPLORING"
    PLANNING = "PLANNING"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


TERMINAL_STATES: frozenset[WorkflowState] = frozenset(
    {
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
        WorkflowState.COMPLETED,
    }
)


ALLOWED_STATE_TRANSITIONS: Mapping[
    WorkflowState, frozenset[WorkflowState]
] = MappingProxyType(
    {
        WorkflowState.CREATED: frozenset(
            {
                WorkflowState.UNDERSTANDING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.UNDERSTANDING: frozenset(
            {
                WorkflowState.EXPLORING,
                WorkflowState.PLANNING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.EXPLORING: frozenset(
            {
                WorkflowState.UNDERSTANDING,
                WorkflowState.PLANNING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.PLANNING: frozenset(
            {
                WorkflowState.EXPLORING,
                WorkflowState.IMPLEMENTING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.IMPLEMENTING: frozenset(
            {
                WorkflowState.VERIFYING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.VERIFYING: frozenset(
            {
                WorkflowState.IMPLEMENTING,
                WorkflowState.REVIEWING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
            }
        ),
        WorkflowState.REVIEWING: frozenset(
            {
                WorkflowState.IMPLEMENTING,
                WorkflowState.VERIFYING,
                WorkflowState.BLOCKED,
                WorkflowState.FAILED,
                WorkflowState.COMPLETED,
            }
        ),
        WorkflowState.BLOCKED: frozenset(),
        WorkflowState.FAILED: frozenset(),
        WorkflowState.COMPLETED: frozenset(),
    }
)


class WorkflowDomainError(ValueError):
    """Base error for deterministic workflow-domain failures."""


class InvalidWorkflowTransition(WorkflowDomainError):
    """Raised when a requested workflow state transition is not allowed."""

    def __init__(
        self, current_state: WorkflowState, target_state: WorkflowState
    ) -> None:
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(
            "Invalid engineering workflow transition: "
            f"{current_state.value} -> {target_state.value}"
        )


class AttemptLimitExceeded(WorkflowDomainError):
    """Raised after an exhausted workflow is deterministically failed."""

    def __init__(self, workflow_run_id: str, max_attempts: int) -> None:
        self.workflow_run_id = workflow_run_id
        self.max_attempts = max_attempts
        super().__init__(
            "Engineering workflow attempt limit exceeded: "
            f"workflow_run_id={workflow_run_id}, max_attempts={max_attempts}"
        )


class WorkflowRun:
    """A single engineering workflow run.

    ``engineering_completed`` is derived exclusively from the engineering
    workflow state. A completed Hermes turn is execution evidence for a future
    orchestrator; it does not mutate this model and cannot complete a workflow.

    ``attempt`` starts at one. ``begin_next_attempt`` advances it while budget
    remains. Requesting another attempt after the configured limit moves the
    workflow to ``FAILED`` and raises ``AttemptLimitExceeded``; exhaustion can
    never produce ``COMPLETED``.
    """

    __slots__ = (
        "_attempt",
        "_created_at",
        "_max_attempts",
        "_state",
        "_updated_at",
        "_workflow_run_id",
    )

    def __init__(self, *, max_attempts: int = 3) -> None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than zero")

        now = _utc_now()
        self._workflow_run_id = str(uuid4())
        self._state = WorkflowState.CREATED
        self._created_at = now
        self._updated_at = now
        self._attempt = 1
        self._max_attempts = max_attempts

    @classmethod
    def restore(
        cls,
        *,
        workflow_run_id: str,
        state: WorkflowState,
        created_at: datetime,
        updated_at: datetime,
        attempt: int,
        max_attempts: int,
    ) -> "WorkflowRun":
        """Reconstruct a validated workflow snapshot from durable facts."""

        if not isinstance(workflow_run_id, str) or not workflow_run_id.strip():
            raise ValueError("workflow_run_id must be a non-empty string")
        if not isinstance(state, WorkflowState):
            raise TypeError("state must be a WorkflowState")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than zero")
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be an integer greater than zero")
        if attempt > max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")
        for name, timestamp in (
            ("created_at", created_at),
            ("updated_at", updated_at),
        ):
            if not isinstance(timestamp, datetime):
                raise TypeError(f"{name} must be a datetime")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if updated_at < created_at:
            raise ValueError("updated_at cannot be earlier than created_at")

        run = cls(max_attempts=max_attempts)
        run._workflow_run_id = workflow_run_id
        run._state = state
        run._created_at = created_at
        run._updated_at = updated_at
        run._attempt = attempt
        return run

    @property
    def workflow_run_id(self) -> str:
        return self._workflow_run_id

    @property
    def state(self) -> WorkflowState:
        return self._state

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def engineering_completed(self) -> bool:
        """Whether the engineering workflow—not a Hermes turn—completed."""

        return self._state is WorkflowState.COMPLETED

    def can_transition_to(self, target_state: WorkflowState) -> bool:
        """Return whether ``target_state`` is allowed from the current state."""

        if not isinstance(target_state, WorkflowState):
            return False
        return target_state in ALLOWED_STATE_TRANSITIONS[self._state]

    def transition_to(
        self, target_state: WorkflowState, *, at: datetime | None = None
    ) -> None:
        """Apply an allowed state transition or fail without partial mutation."""

        if not isinstance(target_state, WorkflowState):
            raise TypeError("target_state must be a WorkflowState")
        if not self.can_transition_to(target_state):
            raise InvalidWorkflowTransition(self._state, target_state)

        timestamp = self._validated_timestamp(at)
        self._state = target_state
        self._updated_at = timestamp

    def begin_next_attempt(self, *, at: datetime | None = None) -> int:
        """Begin the next attempt, or fail the workflow when none remain."""

        if self.is_terminal:
            raise WorkflowDomainError(
                "Cannot begin an attempt for terminal engineering workflow: "
                f"{self._state.value}"
            )

        timestamp = self._validated_timestamp(at)
        if self._attempt >= self._max_attempts:
            self._state = WorkflowState.FAILED
            self._updated_at = timestamp
            raise AttemptLimitExceeded(
                self._workflow_run_id, self._max_attempts
            )

        self._attempt += 1
        self._updated_at = timestamp
        return self._attempt

    def _validated_timestamp(self, at: datetime | None) -> datetime:
        timestamp = _utc_now() if at is None else at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("workflow timestamps must be timezone-aware")
        if timestamp < self._updated_at:
            raise ValueError("workflow timestamps cannot move backwards")
        return timestamp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
