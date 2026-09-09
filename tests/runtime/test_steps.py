"""Step dataclass behavior."""

from __future__ import annotations

import dataclasses
import unittest

from agent.runtime import ActionStep, FinalAnswerStep, PlanningStep, TaskStep, ToolCall, ToolOutput


class StepTests(unittest.TestCase):
    def test_task_step_is_frozen(self) -> None:
        step = TaskStep(step_number=0, task="hello")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            step.task = "mutated"  # type: ignore[misc]

    def test_action_step_defaults(self) -> None:
        step = ActionStep(step_number=1)
        self.assertEqual(step.model_output, "")
        self.assertEqual(step.tool_calls, ())
        self.assertEqual(step.tool_outputs, ())
        self.assertEqual(step.governance_decisions, ())
        self.assertIsNone(step.failure)
        self.assertEqual(step.input_tokens, 0)

    def test_planning_step_carries_plan(self) -> None:
        step = PlanningStep(step_number=2, plan="step 1; step 2")
        self.assertEqual(step.plan, "step 1; step 2")

    def test_final_answer_step_default_trigger(self) -> None:
        step = FinalAnswerStep(step_number=5, output={"answer": 42})
        self.assertEqual(step.output, {"answer": 42})
        self.assertEqual(step.triggered_by, "final_answer_tool")

    def test_tool_call_factory_generates_id(self) -> None:
        a = ToolCall.new(name="x", arguments={})
        b = ToolCall.new(name="x", arguments={})
        self.assertNotEqual(a.id, b.id)
        self.assertTrue(a.id.startswith("call_"))

    def test_tool_call_factory_uses_supplied_id(self) -> None:
        call = ToolCall.new(name="x", arguments={}, call_id="provided")
        self.assertEqual(call.id, "provided")

    def test_tool_output_id_correlates(self) -> None:
        call = ToolCall.new(name="x", arguments={}, call_id="abc")
        output = ToolOutput(id=call.id, name="x", output="result")
        self.assertEqual(call.id, output.id)


if __name__ == "__main__":
    unittest.main()
