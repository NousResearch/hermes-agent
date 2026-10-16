"""Deterministic replay of a recorded run.

A run records every model output, governance decision, and tool output in
its ``AgentMemory``. Replay constructs a ``MultiStepLoop`` whose injected
``ModelProtocol`` and ``ToolHandlerProtocol`` are *scripted from the recorded
trace*. With a ``FrozenClock`` + ``SequentialIdSource`` (or by pinning ids
through the recorded trace), the replayed run produces a byte-identical
JSONL.

The contract:

  * ``RecordedModel.generate()`` returns the next recorded ``ModelOutput``
    in step order. If the model is asked for more outputs than were
    recorded, it raises ``ReplayExhausted``.
  * ``RecordedToolHandler.handle()`` returns the tool outputs that were
    recorded for the matching tool call ids; missing ids raise
    ``ReplayMissingToolOutput``.
  * Default governance during replay is ``ScriptedGovernance`` built from
    the recorded decisions — the strictest mode, so the audit trail is
    reproduced byte-for-byte. ``AllowAllGovernance`` is used only when
    the trace contains no governance decisions (e.g. a no-tool run).
    Pass an override to ``build_replay_loop`` if you want to see what
    a different policy would have done against the same model trace.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .governance import AllowAllGovernance
from .interfaces import (
    FrozenClock,
    GovernanceContext,
    GovernanceDecision,
    GovernanceProtocol,
    ModelOutput,
    SequentialIdSource,
)
from .loop import MultiStepLoop
from .memory import AgentMemory
from .result import RunResult
from .steps import ActionStep, FinalAnswerStep, PlanningStep, TaskStep, ToolCall, ToolOutput


class ReplayExhausted(RuntimeError):
    """Replay was asked for more model outputs than the trace contains."""


class ReplayMissingToolOutput(RuntimeError):
    """Replay was asked for a tool output the trace doesn't contain."""


# ----- scripted protocol impls -----------------------------------------------


class RecordedModel:
    """Returns recorded ``ModelOutput`` in step order."""

    def __init__(self, outputs: list[ModelOutput]) -> None:
        self._outputs = list(outputs)
        self._cursor = 0

    def generate(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> ModelOutput:
        if self._cursor >= len(self._outputs):
            raise ReplayExhausted(
                f"replay requested generation #{self._cursor + 1} but only {len(self._outputs)} recorded"
            )
        out = self._outputs[self._cursor]
        self._cursor += 1
        return out

    @property
    def remaining(self) -> int:
        return max(0, len(self._outputs) - self._cursor)


class RecordedToolHandler:
    """Returns tool outputs keyed by call id."""

    def __init__(self, outputs_by_call_id: dict[str, ToolOutput]) -> None:
        self._by_id = dict(outputs_by_call_id)

    def handle(self, calls: list[ToolCall]) -> list[ToolOutput]:
        results: list[ToolOutput] = []
        for call in calls:
            recorded = self._by_id.get(call.id)
            if recorded is None:
                raise ReplayMissingToolOutput(f"no recorded output for tool call id={call.id!r} name={call.name!r}")
            results.append(recorded)
        return results


class ScriptedGovernance:
    """Returns recorded decisions in order of call.

    The strictest replay mode — ensures the governance trace is preserved
    even if the live policy would have decided differently.
    """

    policy_id = "scripted-replay"

    def __init__(self, decisions: list[GovernanceDecision]) -> None:
        self._decisions = list(decisions)
        self._cursor = 0

    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision:
        if self._cursor >= len(self._decisions):
            raise ReplayExhausted("no more recorded governance decisions")
        decision = self._decisions[self._cursor]
        self._cursor += 1
        # Use the recorded decision but mirror the live call id (in case
        # the live id source assigned a different one).
        return replace(decision, call_id=call.id, tool_name=call.name)


# ----- replay runner ---------------------------------------------------------


def build_replay_loop(memory: AgentMemory, *, governance: GovernanceProtocol | None = None) -> MultiStepLoop:
    """Construct a ``MultiStepLoop`` wired to replay the given memory.

    The caller still drives execution with ``loop.run(task)`` — the loop
    will consume model outputs from the recorded trace and the recorded
    tool outputs from each ``ActionStep``.

    By default a ``ScriptedGovernance`` is built from the recorded
    decisions so the governance trail replays exactly. Pass an override
    if you want to test what would happen under a different policy.
    """
    model_outputs: list[ModelOutput] = []
    tool_outputs_by_id: dict[str, ToolOutput] = {}
    recorded_decisions: list[GovernanceDecision] = []

    for step in memory:
        if isinstance(step, ActionStep):
            # The recorded ``model_output`` and ``tool_calls`` together
            # reconstruct what the model returned for this step.
            model_outputs.append(
                ModelOutput(
                    content=step.model_output,
                    tool_calls=step.tool_calls,
                    input_tokens=step.input_tokens,
                    output_tokens=step.output_tokens,
                    finish_reason="tool_calls" if step.tool_calls else "stop",
                )
            )
            for output in step.tool_outputs:
                # Synthesized outputs (governance denials, etc.) were
                # fabricated by the loop, not produced by the handler.
                # They must NOT enter the handler's id map — the replay
                # loop will re-synthesize them via the same governance
                # path it took the first time.
                if output.synthesized:
                    continue
                tool_outputs_by_id[output.id] = output
            for decision in step.governance_decisions:
                recorded_decisions.append(decision)
        elif isinstance(step, PlanningStep):
            model_outputs.append(
                ModelOutput(content=step.plan, tool_calls=(), input_tokens=0, output_tokens=0)
            )

    chosen_governance: GovernanceProtocol = governance or (
        ScriptedGovernance(recorded_decisions) if recorded_decisions else AllowAllGovernance()
    )

    return MultiStepLoop(
        model=RecordedModel(model_outputs),
        tool_handler=RecordedToolHandler(tool_outputs_by_id),
        governance=chosen_governance,
        clock=FrozenClock(),
        id_source=SequentialIdSource(),
        continue_on_error=False,
    )


def run_from_trace(jsonl_blob: str, *, governance: GovernanceProtocol | None = None) -> RunResult:
    """Convenience: rebuild memory from JSONL and replay it.

    The task is taken from the recorded ``TaskStep``. Returns the
    ``RunResult`` produced by the replayed loop.
    """
    memory = AgentMemory.from_jsonl(jsonl_blob)
    task_step = memory.task_step()
    if task_step is None:
        raise ValueError("recorded trace has no TaskStep — cannot replay")
    loop = build_replay_loop(memory, governance=governance)
    return loop.run(task_step.task, images=task_step.images)
