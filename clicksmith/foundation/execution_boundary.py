"""FND-03 Phase 1 Clicksmith-side execution boundary.

This module wraps an existing execution callable behind the canonical
Clicksmith execution-state contract.

The boundary owns Clicksmith state transitions only. It does not perform
authorization, persistence, recovery, verification, artifact handling,
scheduling, orchestration, or Hermes lifecycle management.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .execution_state import (
    AUTHORIZED,
    CANCELLED,
    FAILED,
    RUNNING,
    SUCCEEDED,
    ExecutionState,
)


class ExecutionBoundaryError(ValueError):
    """Base error for Clicksmith execution-boundary operations."""


class ExecutionNotAuthorizedError(ExecutionBoundaryError):
    """Execution was attempted before authorization was established."""


class InvalidExecutorError(ExecutionBoundaryError):
    """The supplied executor is not callable."""


@dataclass(frozen=True)
class ExecutionBoundaryResult:
    """Result returned by the Clicksmith execution boundary."""

    state: ExecutionState
    result: Any = None


class ExecutionBoundary:
    """Wrap an existing execution callable behind FND-03 state semantics."""

    def __init__(
        self,
        execution_state: ExecutionState,
        executor: Callable[[], Any],
    ) -> None:
        if not isinstance(execution_state, ExecutionState):
            raise ExecutionBoundaryError(
                "execution_state must be an ExecutionState."
            )

        if not callable(executor):
            raise InvalidExecutorError("executor must be callable.")

        self.execution_state = execution_state
        self.executor = executor

    def execute(self) -> ExecutionBoundaryResult:
        """Execute once from AUTHORIZED and map the result to FND-03 state.

        Authorization must already have occurred outside this boundary.
        """
        if self.execution_state.state != AUTHORIZED:
            raise ExecutionNotAuthorizedError(
                "ExecutionBoundary requires an AUTHORIZED execution state."
            )

        self.execution_state.transition(RUNNING)

        try:
            result = self.executor()
        except Exception:
            self.execution_state.transition(FAILED)
            raise

        if _is_explicit_cancellation(result):
            self.execution_state.transition(CANCELLED)
        elif _is_success(result):
            self.execution_state.transition(SUCCEEDED)
        else:
            self.execution_state.transition(FAILED)

        return ExecutionBoundaryResult(
            state=self.execution_state,
            result=result,
        )


def _is_explicit_cancellation(result: Any) -> bool:
    """Return True only for an explicit cancellation result."""
    if isinstance(result, Mapping):
        return result.get("cancelled") is True

    return False


def _is_success(result: Any) -> bool:
    """Return True only for an explicit successful completion result."""
    if not isinstance(result, Mapping):
        return False

    if result.get("cancelled") is True:
        return False

    if result.get("error") is not None:
        return False

    return result.get("completed") is True


__all__ = [
    "ExecutionBoundary",
    "ExecutionBoundaryError",
    "ExecutionBoundaryResult",
    "ExecutionNotAuthorizedError",
    "InvalidExecutorError",
]
