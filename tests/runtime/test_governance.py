"""Governance gate semantics.

Tests that:
  * The default policy is fail-closed (deny-all).
  * Denied tool calls never reach the handler.
  * Allowlist policy admits only enumerated tools.
  * All decisions land on the ActionStep audit trail.
  * Run terminates fail-closed when every call is denied (governance_denied).
"""

from __future__ import annotations

import unittest

from agent.runtime import (
    AllowAllGovernance,
    AllowListGovernance,
    DenyAllGovernance,
    FINAL_ANSWER_TOOL,
    GovernanceContext,
    ModelOutput,
    MultiStepLoop,
    ToolCall,
    ToolOutput,
)


class ScriptedModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, messages, tools=None, **kwargs):
        return self._outputs.pop(0)


class RecordingHandler:
    def __init__(self):
        self.received: list[ToolCall] = []

    def handle(self, calls):
        self.received.extend(calls)
        return [ToolOutput(id=c.id, name=c.name, output={"ran": True}) for c in calls]


# ---- policy unit tests -------------------------------------------------------


class GovernancePolicyTests(unittest.TestCase):
    def _ctx(self) -> GovernanceContext:
        return GovernanceContext(step_number=1, task="t", prior_tool_names=(), state_snapshot={})

    def test_deny_all_denies_everything(self) -> None:
        policy = DenyAllGovernance()
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        decision = policy.decide(call, self._ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertEqual(decision.policy, "deny-all-default")

    def test_allow_all_allows_everything(self) -> None:
        policy = AllowAllGovernance()
        decision = policy.decide(ToolCall.new(name="x", arguments={}, call_id="c"), self._ctx())
        self.assertEqual(decision.verdict, "allow")

    def test_allowlist_admits_only_enumerated(self) -> None:
        policy = AllowListGovernance(allowed={"safe", FINAL_ANSWER_TOOL})
        allowed = policy.decide(ToolCall.new(name="safe", arguments={}, call_id="a"), self._ctx())
        denied = policy.decide(ToolCall.new(name="dangerous", arguments={}, call_id="b"), self._ctx())
        self.assertEqual(allowed.verdict, "allow")
        self.assertEqual(denied.verdict, "deny")
        self.assertIn("not on allowlist", denied.reason)


# ---- loop integration --------------------------------------------------------


class GovernanceIntegrationTests(unittest.TestCase):
    def test_default_loop_is_deny_all(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="trying", tool_calls=(call,))])
        handler = RecordingHandler()
        # No governance arg → DenyAllGovernance.
        loop = MultiStepLoop(model=model, tool_handler=handler, max_steps=2)
        result = loop.run("t")
        # Handler never saw the call.
        self.assertEqual(handler.received, [])
        # Run terminated fail-closed.
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "governance_denied")

    def test_denied_call_appears_in_audit_trail(self) -> None:
        call = ToolCall.new(name="anything", arguments={}, call_id="c")
        model = ScriptedModel([ModelOutput(content="trying", tool_calls=(call,))])
        loop = MultiStepLoop(
            model=model,
            tool_handler=RecordingHandler(),
            governance=DenyAllGovernance(),
            max_steps=2,
        )
        result = loop.run("t")
        action = next(s for s in result.steps if s["type"] == "ActionStep")
        self.assertEqual(len(action["governance_decisions"]), 1)
        self.assertEqual(action["governance_decisions"][0]["verdict"], "deny")
        # Denied calls appear as error tool outputs.
        self.assertEqual(len(action["tool_outputs"]), 1)
        self.assertTrue(action["tool_outputs"][0]["is_error"])
        self.assertEqual(action["tool_outputs"][0]["output"]["denied"], True)

    def test_allowlist_lets_safe_through_blocks_unsafe(self) -> None:
        safe = ToolCall.new(name="safe", arguments={}, call_id="a")
        unsafe = ToolCall.new(name="unsafe", arguments={}, call_id="b")
        final = ToolCall.new(name=FINAL_ANSWER_TOOL, arguments="done", call_id="f")
        model = ScriptedModel(
            [
                ModelOutput(content="mixed batch", tool_calls=(safe, unsafe)),
                ModelOutput(content="", tool_calls=(final,)),
            ]
        )
        handler = RecordingHandler()
        loop = MultiStepLoop(
            model=model,
            tool_handler=handler,
            governance=AllowListGovernance(allowed={"safe", FINAL_ANSWER_TOOL}),
            max_steps=5,
        )
        result = loop.run("t")
        self.assertTrue(result.completed)
        # Handler only saw the allowed call (plus the final-answer call later).
        handler_names = [c.name for c in handler.received]
        self.assertIn("safe", handler_names)
        self.assertNotIn("unsafe", handler_names)


if __name__ == "__main__":
    unittest.main()
