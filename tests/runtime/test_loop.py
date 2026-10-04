"""End-to-end loop semantics with stub model + tool handler.

Every test that needs to actually execute tools opts into
``AllowAllGovernance`` — the loop's default is deny-all.
"""

from __future__ import annotations

import unittest
from typing import Any

from agent.runtime import (
    ActionStep,
    AgentMemory,
    AllowAllGovernance,
    CallbackRegistry,
    DenyAllGovernance,
    FINAL_ANSWER_TOOL,
    MemoryStep,
    ModelOutput,
    MultiStepLoop,
    RunState,
    ToolCall,
    ToolOutput,
)


# ---- test doubles -----------------------------------------------------------


class ScriptedModel:
    """Returns a pre-recorded sequence of ModelOutputs."""

    def __init__(self, outputs: list[ModelOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages, tools=None, **kwargs) -> ModelOutput:
        self.calls.append(list(messages))
        if not self._outputs:
            raise AssertionError("ScriptedModel exhausted — loop made an extra call")
        return self._outputs.pop(0)


class EchoToolHandler:
    """Returns each tool call's arguments back as the output."""

    def __init__(self) -> None:
        self.received: list[ToolCall] = []

    def handle(self, calls):
        self.received.extend(calls)
        return [
            ToolOutput(
                id=call.id,
                name=call.name,
                output=call.arguments,
                is_error=False,
                is_final_answer=(call.name == FINAL_ANSWER_TOOL),
            )
            for call in calls
        ]


# ---- tests ------------------------------------------------------------------


class LoopTests(unittest.TestCase):
    def test_plain_text_response_is_final_answer(self) -> None:
        model = ScriptedModel([ModelOutput(content="42", input_tokens=3, output_tokens=1)])
        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
        )
        result = loop.run("what is the answer?")

        self.assertTrue(result.completed)
        self.assertEqual(result.output, "42")
        self.assertEqual(result.termination_reason, "empty_tool_calls")
        self.assertEqual(result.token_usage.input_tokens, 3)
        self.assertEqual(result.token_usage.output_tokens, 1)
        types = [s["type"] for s in result.steps]
        self.assertEqual(types, ["TaskStep", "ActionStep", "FinalAnswerStep"])

    def test_tool_call_then_final_answer(self) -> None:
        call = ToolCall.new(name="lookup", arguments={"q": "x"}, call_id="c1")
        final_call = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"answer": "done"}, call_id="c2")
        model = ScriptedModel(
            [
                ModelOutput(content="thinking", tool_calls=(call,), input_tokens=10, output_tokens=4),
                ModelOutput(content="", tool_calls=(final_call,), input_tokens=12, output_tokens=2),
            ]
        )
        handler = EchoToolHandler()
        loop = MultiStepLoop(
            model=model,
            tool_handler=handler,
            governance=AllowAllGovernance(),
            max_steps=5,
        )
        result = loop.run("do the thing")

        self.assertTrue(result.completed)
        self.assertEqual(result.output, {"answer": "done"})
        self.assertEqual(len(handler.received), 2)
        self.assertEqual(handler.received[0].name, "lookup")
        self.assertEqual(result.token_usage.input_tokens, 22)
        self.assertEqual(result.token_usage.output_tokens, 6)

    def test_max_steps_exhausted_terminates_failclosed(self) -> None:
        call = ToolCall.new(name="loop_forever", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="step", tool_calls=(call,)) for _ in range(3)])
        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=3,
        )
        result = loop.run("never finish")

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "max_steps")
        # 1 task + 3 actions + 1 synthetic final = 5
        self.assertEqual(len(result.steps), 5)
        self.assertEqual(result.steps[-1]["type"], "FinalAnswerStep")
        self.assertEqual(result.steps[-1]["triggered_by"], "max_steps")
        # The synthetic final step carries a typed failure.
        self.assertEqual(result.steps[-1]["failure"]["kind"], "limit_exceeded")

    def test_interrupt_terminates_between_steps(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="hi", tool_calls=(call,)) for _ in range(5)])
        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
        )

        def trip(step, state):
            loop.interrupt()

        registry = CallbackRegistry()
        registry.register(ActionStep, trip)
        loop.callbacks = registry

        result = loop.run("be interrupted")
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "interrupted")

    def test_callbacks_run_for_each_step_type(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        final_call = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"answer": 1}, call_id="f")
        model = ScriptedModel(
            [
                ModelOutput(content="t", tool_calls=(call,)),
                ModelOutput(content="t", tool_calls=(final_call,)),
            ]
        )

        seen: dict[str, int] = {}

        def counter(step, state):
            key = type(step).__name__
            seen[key] = seen.get(key, 0) + 1

        registry = CallbackRegistry()
        registry.register(MemoryStep, counter)

        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
            callbacks=registry,
        )
        loop.run("x")
        self.assertEqual(seen["TaskStep"], 1)
        self.assertEqual(seen["ActionStep"], 2)
        self.assertEqual(seen["FinalAnswerStep"], 1)

    def test_final_answer_check_rejects_then_loop_continues(self) -> None:
        final_call_bad = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"v": "no"}, call_id="bad")
        final_call_good = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"v": "yes"}, call_id="good")
        model = ScriptedModel(
            [
                ModelOutput(content="trying", tool_calls=(final_call_bad,)),
                ModelOutput(content="trying again", tool_calls=(final_call_good,)),
            ]
        )

        def reject_no(output, memory: AgentMemory):
            if isinstance(output, dict) and output.get("v") == "no":
                return False, "v cannot be 'no'"
            return True, ""

        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
            final_answer_checks=[reject_no],
        )
        result = loop.run("get the right answer")
        self.assertTrue(result.completed)
        self.assertEqual(result.output, {"v": "yes"})
        rejections = result.state.get("rejected_final_answers", [])
        self.assertEqual(len(rejections), 1)

    def test_planning_interval_inserts_planning_steps(self) -> None:
        call = ToolCall.new(name="x", arguments={}, call_id="c")
        final_call = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments="done", call_id="f")
        model = ScriptedModel(
            [
                ModelOutput(content="step1", tool_calls=(call,)),
                ModelOutput(content="step2", tool_calls=(call,)),
                ModelOutput(content="plan-text"),  # planning at step 3
                ModelOutput(content="step3", tool_calls=(final_call,)),
            ]
        )
        loop = MultiStepLoop(
            model=model,
            tool_handler=EchoToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=10,
            planning_interval=2,
        )
        result = loop.run("plan and execute")
        self.assertTrue(result.completed)
        types = [s["type"] for s in result.steps]
        self.assertIn("PlanningStep", types)


if __name__ == "__main__":
    unittest.main()
