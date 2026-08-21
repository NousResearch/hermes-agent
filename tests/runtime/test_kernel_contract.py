"""End-to-end goal contract.

This test pins the kernel's stated goal:

    Prompt → Typed Step Machine → Validated Transition →
    Governed Tool Action → Replayable Runtime Trace

It runs a deterministic scripted scenario and asserts every property
required by the goal:

  1. Every step is a typed frozen dataclass.
  2. Every transition was validated (no rogue step appended).
  3. Every tool call passed through governance — decisions are on record.
  4. State mutations carry typed reasons.
  5. The trace replays byte-identically.
  6. The default loop is fail-closed.
"""

from __future__ import annotations

import dataclasses
import unittest

from agent.runtime import (
    ActionStep,
    AgentMemory,
    AllowListGovernance,
    DenyAllGovernance,
    FINAL_ANSWER_TOOL,
    FinalAnswerStep,
    FrozenClock,
    MemoryStep,
    ModelOutput,
    MultiStepLoop,
    PlanningStep,
    SequentialIdSource,
    StepFailure,
    TaskStep,
    ToolCall,
    ToolOutput,
    run_from_trace,
)


class ScriptedModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, messages, tools=None, **kwargs):
        return self._outputs.pop(0)


class EchoHandler:
    def handle(self, calls):
        return [
            ToolOutput(
                id=c.id,
                name=c.name,
                output=c.arguments,
                is_final_answer=(c.name == FINAL_ANSWER_TOOL),
            )
            for c in calls
        ]


def _build_scenario() -> MultiStepLoop:
    """A run with: tool call, denied call, final answer."""
    safe_call = ToolCall.new(name="lookup", arguments={"q": "weather"}, call_id="lookup_1")
    risky_call = ToolCall.new(name="delete_everything", arguments={}, call_id="risky_1")
    final = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"answer": "sunny"}, call_id="final_1")
    model = ScriptedModel(
        [
            ModelOutput(content="checking", tool_calls=(safe_call, risky_call), input_tokens=8, output_tokens=4),
            ModelOutput(content="answering", tool_calls=(final,), input_tokens=10, output_tokens=2),
        ]
    )
    return MultiStepLoop(
        model=model,
        tool_handler=EchoHandler(),
        governance=AllowListGovernance(allowed={"lookup", FINAL_ANSWER_TOOL}),
        clock=FrozenClock(),
        id_source=SequentialIdSource(),
        max_steps=5,
    )


class KernelContractTests(unittest.TestCase):
    def test_every_step_is_typed_frozen_dataclass(self) -> None:
        loop = _build_scenario()
        result = loop.run("what's the weather?")
        self.assertTrue(result.completed)
        for step in loop.memory:
            self.assertIsInstance(step, MemoryStep)
            self.assertTrue(dataclasses.is_dataclass(step))
            self.assertTrue(step.__dataclass_params__.frozen)  # type: ignore[attr-defined]

    def test_only_legal_transitions_landed_in_memory(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        step_types = [type(s).__name__ for s in loop.memory]
        # Legal walk: TaskStep -> ActionStep -> ActionStep -> FinalAnswerStep
        self.assertEqual(step_types[0], "TaskStep")
        self.assertEqual(step_types[-1], "FinalAnswerStep")
        for prev, nxt in zip(step_types, step_types[1:]):
            self.assertNotEqual(prev, "FinalAnswerStep", "terminal step had a successor")

    def test_every_tool_call_has_a_governance_decision(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        for step in loop.memory:
            if isinstance(step, ActionStep):
                self.assertEqual(
                    len(step.governance_decisions),
                    len(step.tool_calls),
                    "every tool call must have a governance decision on record",
                )
                # Decision ids correlate with call ids.
                decision_ids = {d.call_id for d in step.governance_decisions}
                call_ids = {c.id for c in step.tool_calls}
                self.assertEqual(decision_ids, call_ids)

    def test_denied_tool_call_did_not_execute(self) -> None:
        loop = _build_scenario()
        result = loop.run("what's the weather?")
        # Walk the audit trail and confirm the risky call was denied.
        action0 = result.steps[1]  # task is steps[0], first action is steps[1]
        self.assertEqual(action0["type"], "ActionStep")
        verdicts = {d["tool_name"]: d["verdict"] for d in action0["governance_decisions"]}
        self.assertEqual(verdicts["lookup"], "allow")
        self.assertEqual(verdicts["delete_everything"], "deny")
        # The denied call appears as an error tool output, never an executed one.
        denied = [o for o in action0["tool_outputs"] if o["name"] == "delete_everything"]
        self.assertEqual(len(denied), 1)
        self.assertTrue(denied[0]["is_error"])
        self.assertEqual(denied[0]["output"]["denied"], True)

    def test_state_mutations_all_have_reasons(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        for mutation in loop.state.mutations:
            self.assertTrue(mutation.reason, f"mutation has empty reason: {mutation}")

    def test_state_is_frozen_after_run(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        self.assertTrue(loop.state.frozen)

    def test_trace_replays_byte_identically(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        original = loop.memory.to_jsonl()

        # Drive the replay through ``build_replay_loop`` so we test the
        # actual scripted model + scripted governance + recorded handler
        # path — not just JSONL round-trip serde.
        from agent.runtime import build_replay_loop

        memory = AgentMemory.from_jsonl(original)
        task_step = memory.task_step()
        self.assertIsNotNone(task_step)
        replay_loop = build_replay_loop(memory)
        result = replay_loop.run(task_step.task, images=task_step.images)  # type: ignore[union-attr]
        self.assertTrue(result.completed)

        # The replayed loop's memory must serialize to the same bytes as
        # the original. This is the actual byte-identical-replay guarantee.
        self.assertEqual(replay_loop.memory.to_jsonl(), original)

    def test_loop_refuses_reuse(self) -> None:
        loop = _build_scenario()
        loop.run("what's the weather?")
        with self.assertRaises(RuntimeError):
            loop.run("ask twice")

    def test_default_loop_is_fail_closed_on_tool_use(self) -> None:
        # No governance provided → DenyAllGovernance.
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="want to act", tool_calls=(call,))])
        loop = MultiStepLoop(model=model, tool_handler=EchoHandler(), max_steps=2)
        result = loop.run("do it")
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "governance_denied")

    def test_failure_carries_typed_kind(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="want to act", tool_calls=(call,))])
        loop = MultiStepLoop(model=model, tool_handler=EchoHandler(), max_steps=2)
        loop.run("do it")
        final = next(s for s in loop.memory if isinstance(s, FinalAnswerStep))
        self.assertIsInstance(final.failure, StepFailure)
        self.assertEqual(final.failure.kind, "governance_denied")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
