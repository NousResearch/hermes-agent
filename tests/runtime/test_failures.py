"""Fail-closed failure handling."""

from __future__ import annotations

import unittest

from agent.runtime import (
    AllowAllGovernance,
    ModelOutput,
    MultiStepLoop,
    ToolCall,
    ToolOutput,
)


class ExplodingModel:
    def generate(self, messages, tools=None, **kwargs):
        raise RuntimeError("provider boom")


class ExplodingToolHandler:
    def handle(self, calls):
        raise RuntimeError("tool boom")


class ScriptedModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, messages, tools=None, **kwargs):
        return self._outputs.pop(0)


class FailClosedTests(unittest.TestCase):
    def test_model_error_terminates_failclosed_by_default(self) -> None:
        loop = MultiStepLoop(
            model=ExplodingModel(),
            tool_handler=ExplodingToolHandler(),  # unreachable
            governance=AllowAllGovernance(),
            max_steps=5,
        )
        result = loop.run("t")
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "model_error")
        # Action step recorded the failure.
        action = next(s for s in result.steps if s["type"] == "ActionStep")
        self.assertEqual(action["failure"]["kind"], "model_error")
        self.assertIn("RuntimeError", action["failure"]["details"]["exception_type"])

    def test_tool_error_terminates_failclosed_by_default(self) -> None:
        call = ToolCall.new(name="explode", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="thinking", tool_calls=(call,))])
        loop = MultiStepLoop(
            model=model,
            tool_handler=ExplodingToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
        )
        result = loop.run("t")
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "tool_error")

    def test_continue_on_error_keeps_going_after_model_error(self) -> None:
        # Two consecutive model calls — first raises, second succeeds with
        # a plain-text final answer.
        class FlakeyModel:
            def __init__(self):
                self._calls = 0

            def generate(self, messages, tools=None, **kwargs):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("transient")
                return ModelOutput(content="recovered", input_tokens=1, output_tokens=1)

        loop = MultiStepLoop(
            model=FlakeyModel(),
            tool_handler=ExplodingToolHandler(),
            governance=AllowAllGovernance(),
            max_steps=5,
            continue_on_error=True,
        )
        result = loop.run("t")
        self.assertTrue(result.completed)
        self.assertEqual(result.output, "recovered")

    def test_governance_default_denies_all_tools_terminates(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="trying", tool_calls=(call,))])
        # No governance arg → fail-closed default.
        loop = MultiStepLoop(model=model, tool_handler=ExplodingToolHandler(), max_steps=2)
        result = loop.run("t")
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "governance_denied")


if __name__ == "__main__":
    unittest.main()
