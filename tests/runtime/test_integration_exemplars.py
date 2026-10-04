"""Smoke tests for the integration exemplars.

These tests use **fake hermes internals** to prove the shim shape is
sound — the kernel can be wired up against a provider adapter and a
function-call dispatcher without modification.

They do NOT test real hermes code (which isn't installed here). When
this scaffold is dropped into hermes-agent, replace the fakes with real
adapter imports.
"""

from __future__ import annotations

import unittest

from agent.runtime import (
    AllowAllGovernance,
    FINAL_ANSWER_TOOL,
    MultiStepLoop,
)
from integration.aiagent_delegation import AIAgentKernelAdapter
from integration.hermes_model_shim import HermesModelShim
from integration.hermes_tool_handler_shim import HermesToolHandler


# ---- fake hermes internals ---------------------------------------------------


class FakeAnthropicAdapter:
    """Stands in for hermes's real adapter.

    Returns a dict that the shim's extractors understand. The real
    adapter returns Anthropic SDK objects; the shim has TODOs marking
    where to plug in real extraction.
    """

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)

    def call(self, messages, tools=None, **kwargs):
        return self._responses.pop(0)


def fake_handle_function_call(name: str, arguments: dict, **kwargs):
    """Stands in for ``model_tools.handle_function_call``."""
    if name == "compute":
        return {"result": arguments["x"] * 2}
    if name == FINAL_ANSWER_TOOL:
        return arguments
    raise ValueError(f"unknown tool: {name}")


# ---- shim shape tests --------------------------------------------------------


class HermesModelShimTests(unittest.TestCase):
    def test_shim_translates_response_into_model_output(self) -> None:
        adapter = FakeAnthropicAdapter([
            {
                "content": "thinking",
                "tool_calls": [{"id": "c1", "name": "compute", "arguments": {"x": 21}}],
                "input_tokens": 10,
                "output_tokens": 3,
                "finish_reason": "tool_calls",
            }
        ])
        shim = HermesModelShim(adapter)
        output = shim.generate(messages=[{"role": "user", "content": "double 21"}])
        self.assertEqual(output.content, "thinking")
        self.assertEqual(len(output.tool_calls), 1)
        self.assertEqual(output.tool_calls[0].name, "compute")
        self.assertEqual(output.input_tokens, 10)
        self.assertEqual(output.output_tokens, 3)
        self.assertEqual(output.finish_reason, "tool_calls")

    def test_shim_preserves_raw_payload_for_telemetry(self) -> None:
        adapter = FakeAnthropicAdapter([{"content": "x", "tool_calls": [], "input_tokens": 1, "output_tokens": 1}])
        shim = HermesModelShim(adapter)
        output = shim.generate(messages=[])
        self.assertIsNotNone(output.raw)
        self.assertEqual(output.raw["content"], "x")


class HermesToolHandlerTests(unittest.TestCase):
    def test_handler_dispatches_each_call(self) -> None:
        from agent.runtime import ToolCall
        handler = HermesToolHandler(dispatch=fake_handle_function_call)
        outputs = handler.handle([
            ToolCall.new(name="compute", arguments={"x": 5}, call_id="c1"),
            ToolCall.new(name=FINAL_ANSWER_TOOL, arguments={"answer": 10}, call_id="c2"),
        ])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].output, {"result": 10})
        self.assertTrue(outputs[1].is_final_answer)

    def test_handler_converts_exception_to_typed_observation(self) -> None:
        from agent.runtime import ToolCall
        handler = HermesToolHandler(dispatch=fake_handle_function_call)
        outputs = handler.handle([ToolCall.new(name="unknown", arguments={}, call_id="c")])
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].is_error)
        self.assertEqual(outputs[0].output["error_type"], "ValueError")


# ---- end-to-end delegation --------------------------------------------------


class AIAgentDelegationTests(unittest.TestCase):
    def test_kernel_delegation_returns_legacy_dict(self) -> None:
        adapter = FakeAnthropicAdapter([
            {
                "content": "doubling",
                "tool_calls": [{"id": "c1", "name": "compute", "arguments": {"x": 21}}],
                "input_tokens": 10,
                "output_tokens": 3,
                "finish_reason": "tool_calls",
            },
            {
                "content": "",
                "tool_calls": [{"id": "c2", "name": FINAL_ANSWER_TOOL, "arguments": {"answer": 42}}],
                "input_tokens": 12,
                "output_tokens": 2,
                "finish_reason": "tool_calls",
            },
        ])
        agent = AIAgentKernelAdapter(
            provider_adapter=adapter,
            config={"agent": {"governance": "allow-all", "max_steps": 5}},
            tools=[],
            tool_dispatch=fake_handle_function_call,
        )
        result = agent.run("double 21")
        # Legacy dict shape — what existing AIAgent.run() callers consume.
        self.assertTrue(result["completed"])
        self.assertEqual(result["output"], {"answer": 42})
        self.assertEqual(result["input_tokens"], 22)
        self.assertEqual(result["output_tokens"], 5)
        # The kernel is still accessible for callers who want the typed result.
        self.assertIsInstance(agent.loop, MultiStepLoop)

    def test_kernel_delegation_fail_closed_by_default(self) -> None:
        adapter = FakeAnthropicAdapter([
            {
                "content": "want to act",
                "tool_calls": [{"id": "c", "name": "anything", "arguments": {}}],
                "input_tokens": 1,
                "output_tokens": 1,
                "finish_reason": "tool_calls",
            },
        ])
        agent = AIAgentKernelAdapter(
            provider_adapter=adapter,
            config={"agent": {"max_steps": 2}},  # default governance = deny-all
            tools=[],
        )
        result = agent.run("do it")
        self.assertFalse(result["completed"])
        self.assertEqual(result["termination_reason"], "governance_denied")

    def test_telemetry_sink_receives_every_step(self) -> None:
        adapter = FakeAnthropicAdapter([
            {"content": "answer", "tool_calls": [], "input_tokens": 1, "output_tokens": 1, "finish_reason": "stop"},
        ])
        sink_log: list[dict] = []
        agent = AIAgentKernelAdapter(
            provider_adapter=adapter,
            config={"agent": {"governance": "allow-all", "max_steps": 3}},
            tools=[],
            telemetry_sink=sink_log.append,
        )
        agent.run("x")
        types_seen = [entry["type"] for entry in sink_log]
        self.assertIn("TaskStep", types_seen)
        self.assertIn("ActionStep", types_seen)
        self.assertIn("FinalAnswerStep", types_seen)


if __name__ == "__main__":
    unittest.main()
