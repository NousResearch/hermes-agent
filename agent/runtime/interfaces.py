"""Narrow protocols the loop calls into.

The loop is deliberately ignorant of provider quirks, real wall-clock time,
and id generation. Everything that varies between production and replay is
injected as a protocol implementation at construction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .steps import FailureKind, StepFailure, ToolCall, ToolOutput


# ----- model + tool handler --------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Provider-agnostic shape of a single model generation."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length" | "content_filter" | "error"
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class ModelProtocol(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelOutput: ...


@runtime_checkable
class ToolHandlerProtocol(Protocol):
    """Batch tool execution surface.

    The handler receives only calls that have been **allowed** by the
    governance gate. Denied calls never reach the handler; they appear as
    error-flagged ``ToolOutput`` entries in the step record.
    """

    def handle(self, calls: list[ToolCall]) -> list[ToolOutput]: ...


# ----- governance ------------------------------------------------------------


GovernanceVerdict = Literal["allow", "deny", "require_approval"]


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """Outcome of submitting a single tool call to the governance gate."""

    call_id: str
    tool_name: str
    verdict: GovernanceVerdict
    reason: str = ""
    policy: str = ""  # policy identifier (e.g. "deny-all-default", "approval-required")


@dataclass(frozen=True, slots=True)
class GovernanceContext:
    """What the governance policy is allowed to see when deciding.

    Read-only by contract — the gate receives a snapshot, never a live
    reference to ``RunState``.
    """

    step_number: int
    task: str
    prior_tool_names: tuple[str, ...]
    state_snapshot: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GovernanceProtocol(Protocol):
    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision: ...


# ----- clock + id source -----------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


@runtime_checkable
class IdSource(Protocol):
    def call_id(self) -> str: ...


class SystemClock:
    """Default production clock — wall time."""

    def now(self) -> float:
        return time.time()


class FrozenClock:
    """Replay clock — starts at 0, advances only when ``advance()`` is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class UuidIdSource:
    """Default production id source — random uuid4 prefix."""

    def call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:12]}"


class SequentialIdSource:
    """Replay id source — deterministic counter starting at 1."""

    def __init__(self) -> None:
        self._n = 0

    def call_id(self) -> str:
        self._n += 1
        return f"call_{self._n:08d}"


# ----- misc ------------------------------------------------------------------


FINAL_ANSWER_TOOL = "final_answer"


def failure_from_exception(
    kind: FailureKind,
    exc: BaseException,
    *,
    extra: dict[str, Any] | None = None,
) -> StepFailure:
    """Helper for building a ``StepFailure`` from a caught exception.

    Used at narrow boundaries where the loop explicitly opts to convert an
    exception into a typed failure — never to silence one. ``kind`` must
    be a ``FailureKind`` literal so the taxonomy stays enforced.
    """
    details: dict[str, Any] = {"exception_type": type(exc).__name__}
    if extra:
        details.update(extra)
    return StepFailure(kind=kind, message=f"{type(exc).__name__}: {exc}", details=details)
