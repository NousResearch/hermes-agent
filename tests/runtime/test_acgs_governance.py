"""ACGS-Lite constitutional governance backend."""

from __future__ import annotations

import unittest

from agent.runtime import (
    ACGSDecisionReceipt,
    ACGSGovernance,
    ACGSRule,
    Constitution,
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


def _constitution(*rules: ACGSRule) -> Constitution:
    return Constitution.from_yaml_dict({"rules": [
        {
            "id": r.id,
            "pattern": r.pattern,
            "description": r.description,
            "effect": r.effect,
            "severity": r.severity,
            "applies_to_tools": list(r.applies_to_tools),
        }
        for r in rules
    ]})


class ConstitutionTests(unittest.TestCase):
    def test_constitutional_hash_is_computed_when_absent(self) -> None:
        c = Constitution.from_yaml_dict({
            "rules": [{"id": "x", "pattern": "y", "effect": "HARD_DENY", "severity": "HIGH"}],
        })
        self.assertTrue(c.constitutional_hash)
        self.assertEqual(len(c.constitutional_hash), 64)  # sha256 hex

    def test_constitutional_hash_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            Constitution.from_yaml_dict({
                "constitutional_hash": "deadbeef",  # wrong on purpose
                "rules": [{"id": "x", "pattern": "y", "effect": "HARD_DENY", "severity": "HIGH"}],
            })

    def test_constitutional_hash_matches_when_correct(self) -> None:
        rules = [{"id": "x", "pattern": "y", "effect": "HARD_DENY", "severity": "HIGH"}]
        computed = Constitution.from_yaml_dict({"rules": rules}).constitutional_hash
        c = Constitution.from_yaml_dict({"constitutional_hash": computed, "rules": rules})
        self.assertEqual(c.constitutional_hash, computed)


class LocalACGSClientTests(unittest.TestCase):
    def test_hard_deny_wins_over_softer_match(self) -> None:
        client = LocalACGSClient(_constitution(
            ACGSRule(id="R-allow", pattern="lookup", effect="ALLOW", severity="INFO"),
            ACGSRule(id="R-hard", pattern="rm -rf|drop table", effect="HARD_DENY", severity="CRITICAL"),
        ))
        receipt = client.evaluate("shell", {"cmd": "rm -rf /"}, _ctx())
        self.assertEqual(receipt.verdict, "HARD_DENY")
        self.assertIn("R-hard", receipt.matched_rules)

    def test_no_match_is_fail_closed_hard_deny(self) -> None:
        client = LocalACGSClient(_constitution(
            ACGSRule(id="R-allow", pattern="^lookup$", effect="ALLOW", severity="INFO",
                     applies_to_tools=("lookup",)),
        ))
        receipt = client.evaluate("delete_everything", {}, _ctx())
        self.assertEqual(receipt.verdict, "HARD_DENY")
        self.assertEqual(receipt.matched_rules, ())

    def test_structured_review_required_path(self) -> None:
        client = LocalACGSClient(_constitution(
            ACGSRule(id="R-pay", pattern="transfer|wire", effect="STRUCTURED_REVIEW_REQUIRED",
                     severity="HIGH"),
        ))
        receipt = client.evaluate("bank_transfer", {"action": "wire 100 USD"}, _ctx())
        self.assertEqual(receipt.verdict, "STRUCTURED_REVIEW_REQUIRED")

    def test_allow_with_controls_surfaces_controls(self) -> None:
        client = LocalACGSClient(_constitution(
            ACGSRule(id="R-read", pattern="read_file", effect="ALLOW_WITH_CONTROLS",
                     severity="LOW"),
        ))
        receipt = client.evaluate("read_file", {"path": "/etc/hosts"}, _ctx())
        self.assertEqual(receipt.verdict, "ALLOW_WITH_CONTROLS")
        self.assertEqual(receipt.controls, ("R-read",))

    def test_receipt_hash_deterministic_across_argument_order(self) -> None:
        client = LocalACGSClient(_constitution(
            ACGSRule(id="R-allow", pattern="t", effect="ALLOW", severity="INFO"),
        ))
        a = client.evaluate("t", {"x": 1, "y": 2}, _ctx())
        b = client.evaluate("t", {"y": 2, "x": 1}, _ctx())
        self.assertEqual(a.arguments_hash, b.arguments_hash)
        self.assertEqual(a.receipt_hash, b.receipt_hash)

    def test_constitutional_hash_changes_with_rules(self) -> None:
        a = LocalACGSClient(_constitution(
            ACGSRule(id="A", pattern="x", effect="ALLOW", severity="INFO"),
        ))
        b = LocalACGSClient(_constitution(
            ACGSRule(id="A", pattern="x", effect="ALLOW", severity="INFO"),
            ACGSRule(id="B", pattern="y", effect="HARD_DENY", severity="HIGH"),
        ))
        self.assertNotEqual(a.constitutional_hash, b.constitutional_hash)

    def test_invalid_regex_surfaces_at_init(self) -> None:
        import re
        with self.assertRaises(re.error):
            LocalACGSClient(_constitution(
                ACGSRule(id="bad", pattern="(unclosed", effect="ALLOW", severity="INFO"),
            ))


class ACGSGovernanceTests(unittest.TestCase):
    def test_allow_maps_to_kernel_allow(self) -> None:
        gov = ACGSGovernance(LocalACGSClient(_constitution(
            ACGSRule(id="R-ok", pattern="lookup", effect="ALLOW", severity="INFO"),
        )))
        decision = gov.decide(
            ToolCall(id="c1", name="lookup", arguments={"q": "x"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "allow")
        self.assertIn("acgs=ALLOW", decision.reason)
        self.assertIn("receipt=", decision.reason)
        self.assertTrue(decision.policy.startswith("acgs-lite:"))

    def test_structured_review_maps_to_require_approval(self) -> None:
        gov = ACGSGovernance(LocalACGSClient(_constitution(
            ACGSRule(id="R-pay", pattern="wire|transfer", effect="STRUCTURED_REVIEW_REQUIRED",
                     severity="HIGH"),
        )))
        decision = gov.decide(
            ToolCall(id="c", name="bank", arguments={"action": "wire"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "require_approval")
        self.assertIn("acgs=STRUCTURED_REVIEW_REQUIRED", decision.reason)

    def test_hard_deny_maps_to_deny(self) -> None:
        gov = ACGSGovernance(LocalACGSClient(_constitution(
            ACGSRule(id="R-rm", pattern="rm -rf", effect="HARD_DENY", severity="CRITICAL"),
        )))
        decision = gov.decide(
            ToolCall(id="c", name="shell", arguments={"cmd": "rm -rf /"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs=HARD_DENY", decision.reason)

    def test_transform_required_maps_to_deny(self) -> None:
        gov = ACGSGovernance(LocalACGSClient(_constitution(
            ACGSRule(id="R-trans", pattern="bulk", effect="TRANSFORM_REQUIRED", severity="MEDIUM"),
        )))
        decision = gov.decide(
            ToolCall(id="c", name="x", arguments={"mode": "bulk"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs=TRANSFORM_REQUIRED", decision.reason)

    def test_client_exception_fails_closed(self) -> None:
        class Boom:
            constitutional_hash = "rs"

            def evaluate(self, *_a, **_k):
                raise RuntimeError("network down")

        gov = ACGSGovernance(Boom())
        decision = gov.decide(ToolCall(id="c", name="lookup", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs_client_error", decision.reason)
        self.assertEqual(decision.policy, "acgs-lite:error")

    def test_unmapped_verdict_fails_closed(self) -> None:
        class Weird:
            constitutional_hash = "rh"

            def evaluate(self, tool_name, arguments, context):
                return ACGSDecisionReceipt(
                    receipt_hash="x",
                    constitutional_hash=self.constitutional_hash,
                    tool_name=tool_name,
                    arguments_hash="a",
                    verdict="maybe",  # type: ignore[arg-type]
                    reason="who knows",
                )

        gov = ACGSGovernance(Weird())
        decision = gov.decide(ToolCall(id="c", name="x", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs_unmapped_verdict", decision.reason)

    def test_missing_receipt_hash_fails_closed(self) -> None:
        class NoHash:
            constitutional_hash = "rh"

            def evaluate(self, tool_name, arguments, context):
                return ACGSDecisionReceipt(
                    receipt_hash="",  # missing
                    constitutional_hash=self.constitutional_hash,
                    tool_name=tool_name,
                    arguments_hash="a",
                    verdict="ALLOW",
                    reason="ok",
                )

        gov = ACGSGovernance(NoHash())
        decision = gov.decide(ToolCall(id="c", name="x", arguments={}), _ctx())
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs_missing_receipt_hash", decision.reason)

    def test_end_to_end_with_loop(self) -> None:
        gov = ACGSGovernance(LocalACGSClient(_constitution(
            ACGSRule(id="R-ok-lookup", pattern="^lookup", effect="ALLOW",
                     severity="INFO", applies_to_tools=("lookup",)),
            ACGSRule(id="R-ok-final", pattern=".*", effect="ALLOW",
                     severity="INFO", applies_to_tools=(FINAL_ANSWER_TOOL,)),
            ACGSRule(id="R-deny-rm", pattern="rm -rf|drop table", effect="HARD_DENY",
                     severity="CRITICAL"),
        )))

        ok = ToolCall(id="c1", name="lookup", arguments={"q": "weather"})
        bad = ToolCall(id="c2", name="shell", arguments={"cmd": "rm -rf /"})
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

        action0 = result.steps[1]
        verdicts = {d["tool_name"]: d["verdict"] for d in action0["governance_decisions"]}
        self.assertEqual(verdicts["lookup"], "allow")
        self.assertEqual(verdicts["shell"], "deny")
        denied = [o for o in action0["tool_outputs"] if o["name"] == "shell"]
        self.assertEqual(len(denied), 1)
        self.assertTrue(denied[0]["is_error"])
        self.assertTrue(denied[0]["synthesized"])

    def test_config_builder_wires_yaml_dict(self) -> None:
        gov = build_acgs_governance_from_config({
            "agent": {
                "governance": "acgs",
                "acgs": {
                    "rules": [
                        {"id": "no-rm", "pattern": "rm -rf", "effect": "HARD_DENY",
                         "severity": "CRITICAL"},
                        {"id": "ok-read", "pattern": "read_file", "effect": "ALLOW",
                         "severity": "LOW"},
                    ],
                },
            }
        })
        decision = gov.decide(
            ToolCall(id="c", name="shell", arguments={"cmd": "rm -rf /tmp"}),
            _ctx(),
        )
        self.assertEqual(decision.verdict, "deny")
        self.assertIn("acgs=HARD_DENY", decision.reason)


if __name__ == "__main__":
    unittest.main()
