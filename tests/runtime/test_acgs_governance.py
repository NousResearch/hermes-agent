"""ACGS constitutional governance backend."""

from __future__ import annotations

import unittest

from agent.runtime import (
    ACGSDecisionReceipt,
    ACGSGovernance,
    ACGSRule,
    FINAL_ANSWER_TOOL,
    FrozenClock,
    GovernanceContext,
    LocalACGSClient,
    ModelOutput,
    MultiStepLoop,
    SequentialIdSource,
    ToolCall,
    ToolOutput,
    build_acgs_governance_from_config,
)


class _Model:
    def __init__(self, outputs):
        self._o = list(outputs)

    def generate(self, messages, tools=None, **kwargs):
        return self._o.pop(0)


class _Handler:
    def handle(self, calls):
        return [ToolOutput(id=c.id, name=c.name, output={"ran": c.name}) for c in calls]


def _ctx() -> GovernanceContext:
    return GovernanceContext(step_number=1, task="t", prior_tool_names=(), state_snapshot={})


class LocalACGSClientTests(unittest.TestCase):
    def test_first_deny_wins(self) -> None:
        client = LocalACGSClient([
            ACGSRule(rule_id="R-allow", description="ok", effect="allow", applies_to_tools=("lookup",)),
            ACGSRule(rule_id="R-deny", description="no", effect="deny", applies_to_tools=("rm",)),
        ])
        receipt = client.evaluate("rm", {"path": "/"}, _ctx())
        self.assertEqual(receipt.verdict, "deny")
        self.assertEqual(receipt.matched_rules, ("R-deny",))

    def test_no_matching_rule_is_fail_closed(self) -> None:
        client = LocalACGSClient([
            ACGSRule(rule_id="R-x", description="x", effect="allow", applies_to_tools=("only_this",)),
        ])
        receipt = client.evaluate("something_else", {}, _ctx())
        self.assertEqual(receipt.verdict, "deny")
        self.assertEqual(receipt.matched_rules, ())

    def test_empty_rule_set_is_fail_closed(self) -> None:
        client = LocalACGSClient([])
        receipt = client.evaluate("anything", {}, _ctx())
        self.assertEqual(receipt.verdict, "deny")

    def test_receipt_hash_is_deterministic_across_argument_order(self) -> None:
        client = LocalACGSClient([
            ACGSRule(rule_id="R-allow", description="ok", effect="allow"),
        ])
        a = client.evaluate("t", {"x": 1, "y": 2}, _ctx())
        b = client.evaluate("t", {"y": 2, "x": 1}, _ctx())
        self.assertEqual(a.arguments_hash, b.arguments_hash)
        self.assertEqual(a.receipt_hash, b.receipt_hash)

    def test_rule_set_hash_changes_with_rules(self) -> None:
        a = LocalACGSClient([ACGSRule(rule_id="A", description="", effect="allow")])
        b = LocalACGSClient([
            ACGSRule(rule_id="A", description="", effect="allow"),
            ACGSRule(rule_id="B", description="", effect="deny"),
        ])
        self.assertNotEqual(a.rule_set_hash, b.rule_set_hash)


class ACGSGovernanceTests(unittest.TestCase):
    def test_translates_receipt_to_decision(self) -> None:
        client = LocalACGSClient([
            ACGSRule(rule_id="R-allow", description="ok", effect="allow", applies_to_tools=("lookup",)),
        ], rule_set_id="hermes-test")
        gov = ACGSGovernance(client)
        decision = gov.decide(
            ToolCall(id="c1", name="lookup", arguments={"q": "x"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "allow")
        self.assertEqual(decision.call_id, "c1")
        self.assertTrue(decision.policy.startswith("acgs:hermes-test@"))

    def test_client_exception_fails_closed(self) -> None:
        class Boom:
            rule_set_id = "x"
            rule_set_hash = "y"

            def evaluate(self, *_a, **_k):
                raise RuntimeError("network down")

        gov = ACGSGovernance(Boom())
        decision = gov.decide(ToolCall(id="c", name="lookup", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs_client_error", decision.reason)
        self.assertEqual(decision.policy, "acgs:error")

    def test_unmapped_verdict_fails_closed(self) -> None:
        class Weird:
            rule_set_id = "rs"
            rule_set_hash = "rh"

            def evaluate(self, tool_name, arguments, context):
                return ACGSDecisionReceipt(
                    receipt_hash="x",
                    rule_set_id=self.rule_set_id,
                    rule_set_hash=self.rule_set_hash,
                    tool_name=tool_name,
                    arguments_hash="a",
                    verdict="maybe",
                    reason="who knows",
                )

        gov = ACGSGovernance(Weird())
        decision = gov.decide(ToolCall(id="c", name="x", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs_unmapped_verdict", decision.reason)

    def test_end_to_end_with_loop(self) -> None:
        gov = ACGSGovernance(LocalACGSClient([
            ACGSRule(rule_id="R-allow-lookup", description="reads ok", effect="allow", applies_to_tools=("lookup",)),
            ACGSRule(rule_id="R-allow-final", description="final ok", effect="allow", applies_to_tools=(FINAL_ANSWER_TOOL,)),
            ACGSRule(rule_id="R-deny-rm", description="no destruction", effect="deny", applies_to_tools=("rm",)),
        ], rule_set_id="hermes-test"))

        ok = ToolCall(id="c1", name="lookup", arguments={"q": "w"})
        bad = ToolCall(id="c2", name="rm", arguments={"p": "/"})
        final = ToolCall(id="c3", name=FINAL_ANSWER_TOOL, arguments={"answer": "done"})
        model = _Model([
            ModelOutput(content="probe", tool_calls=(ok, bad)),
            ModelOutput(content="answer", tool_calls=(final,)),
        ])
        loop = MultiStepLoop(
            model=model,
            tool_handler=_Handler(),
            governance=gov,
            clock=FrozenClock(),
            id_source=SequentialIdSource(),
            max_steps=5,
            final_answer_tool_name=FINAL_ANSWER_TOOL,
        )
        result = loop.run("do a thing")
        self.assertTrue(result.completed)

        # First action: lookup allowed, rm denied via an ACGS rule.
        action0 = result.steps[1]
        verdicts = {d["tool_name"]: d["verdict"] for d in action0["governance_decisions"]}
        self.assertEqual(verdicts["lookup"], "allow")
        self.assertEqual(verdicts["rm"], "deny")
        # Denied output present, synthesized, not executed.
        denied = [o for o in action0["tool_outputs"] if o["name"] == "rm"]
        self.assertEqual(len(denied), 1)
        self.assertTrue(denied[0]["is_error"])
        self.assertTrue(denied[0]["synthesized"])

    def test_config_builder_wires_rules(self) -> None:
        gov = build_acgs_governance_from_config({
            "agent": {
                "governance": "acgs",
                "acgs": {
                    "rule_set_id": "hermes-dev",
                    "rules": [
                        {"rule_id": "R-deny", "effect": "deny", "applies_to_tools": ["rm"]},
                        {"rule_id": "R-allow", "effect": "allow", "applies_to_tools": ["lookup"]},
                    ],
                },
            }
        })
        decision = gov.decide(ToolCall(id="c", name="rm", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertTrue(decision.policy.startswith("acgs:hermes-dev@"))


if __name__ == "__main__":
    unittest.main()
