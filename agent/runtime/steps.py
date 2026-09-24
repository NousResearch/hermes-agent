"""Typed step records for the agent loop.

Every iteration of the loop appends exactly one ``MemoryStep`` subclass to the
agent's ``AgentMemory``. Steps are **frozen dataclasses** — once recorded
they do not change. This is what makes replay deterministic and makes step
history safe to share across threads.

Every step that can fail carries a typed ``StepFailure`` rather than a free
``error: str`` field, so consumers can reason structurally about what went
wrong (model error vs governance denial vs limit breach vs check rejection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ----- failure taxonomy ------------------------------------------------------

FailureKind = Literal[
    "model_error",          # provider call raised
    "tool_error",           # tool handler raised
    "governance_denied",    # governance gate refused a tool call
    "transition_rejected",  # transition guard refused an append
    "check_failed",         # final_answer_check returned False
    "limit_exceeded",       # max_steps / token / wall-time cap hit
    "executor_refused",     # sandboxed executor refused to run code
    "interrupted",          # cooperative interrupt
]


@dataclass(frozen=True, slots=True)
class StepFailure:
    """Structured record of why a step did not produce useful output.

    ``kind`` is the discriminator. ``message`` is a one-line human summary.
    ``details`` carries any extra structured context (provider name, policy
    id, exception type, …) — kept as a plain dict because the dataclass is
    frozen anyway.
    """

    kind: FailureKind
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ----- tool-call records -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation requested by the model.

    ``id`` is the provider's call id when present (Anthropic/OpenAI return
    one). The model shim is expected to either pass the provider id through
    or substitute one from the loop's injected ``IdSource`` — the loop
    asserts a non-empty id before submitting calls to the governance gate.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def new(name: str, arguments: dict[str, Any], call_id: str | None = None) -> "ToolCall":
        """Test/construction convenience.

        Production code path always goes through the loop's ``IdSource`` —
        this helper is for fixtures and direct callers that need a quick id.
        """
        import uuid

        return ToolCall(
            id=call_id or f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
        )


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """The result of executing a ``ToolCall``.

    ``synthesized`` is ``True`` when the loop fabricated this output instead
    of asking the handler to run it — e.g. when governance denied the call.
    Replay uses this flag to skip the synthetic entries when rebuilding the
    handler's id map (so a recorded run replays without spurious
    ``ReplayMissingToolOutput`` errors).
    """

    id: str  # mirrors ToolCall.id
    name: str
    output: Any
    is_error: bool = False
    is_final_answer: bool = False
    synthesized: bool = False


# ----- step base + variants --------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryStep:
    """Base type for everything that lands in ``AgentMemory``.

    ``started_at`` is injected by the loop's ``Clock`` — never read directly
    from ``time.time()`` inside step construction so that replay traces are
    byte-identical.
    """

    step_number: int
    started_at: float = 0.0


@dataclass(frozen=True, slots=True)
class TaskStep(MemoryStep):
    """The initial user task. Always step 0."""

    task: str = ""
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionStep(MemoryStep):
    """A think → act → observe cycle.

    ``model_input_messages`` is the prompt the model saw for *this* step.
    ``tool_calls`` and ``tool_outputs`` line up by call id (not by index —
    governance may deny one of N calls).

    ``governance_decisions`` records every decision made by the gate for
    this step, even denials, so the audit trail is complete.

    ``failure`` is set iff the step did not produce useful output. The loop
    consults ``failure`` to decide whether to continue (fail-open policy)
    or terminate (fail-closed default).
    """

    model_input_messages: tuple[dict[str, Any], ...] = ()
    model_output: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_outputs: tuple[ToolOutput, ...] = ()
    governance_decisions: tuple[Any, ...] = ()  # tuple[GovernanceDecision, ...] — avoids import cycle
    failure: StepFailure | None = None
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PlanningStep(MemoryStep):
    """A periodic explicit plan generation step."""

    model_input_messages: tuple[dict[str, Any], ...] = ()
    plan: str = ""
    facts: str = ""
    duration_s: float = 0.0
    failure: StepFailure | None = None


@dataclass(frozen=True, slots=True)
class FinalAnswerStep(MemoryStep):
    """Terminal step — the agent's answer to the original task.

    ``triggered_by`` says *how* the run ended:
      * ``"final_answer_tool"``   — explicit final_answer tool call
      * ``"empty_tool_calls"``    — model produced text with no tool calls
      * ``"max_steps"``           — exhausted the step budget
      * ``"interrupted"``         — cooperative interrupt
      * a FailureKind value       — fail-closed termination

    When ``failure`` is set, ``output`` is whatever the last useful step
    produced (often empty / partial). Callers should always inspect
    ``failure`` before trusting ``output``.
    """

    output: Any = None
    triggered_by: str = "final_answer_tool"
    failure: StepFailure | None = None
