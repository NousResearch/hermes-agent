"""Tests for L0-L3 rule priority governance."""

import sys, os, unittest
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
from rule_priority import (  # noqa: E402
    Rule, resolve_conflicts, load_rules, inject_system_prompt,
    check_tool_block, RulePriorityPlugin, P_L0, P_L1, P_L2, P_L3,
)


class TestResolveConflicts(unittest.TestCase):
    """L0 > L3 > L1 > L2. Same level → last-write-wins."""

    def test_same_level_last_wins(self):
        r = resolve_conflicts([
            Rule("a", P_L1, "Be concise."),
            Rule("b", P_L1, "Be concise."),
        ])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].id, "b")  # last-write-wins

    def test_l0_wins_over_l3(self):
        r = resolve_conflicts([
            Rule("l3r", P_L3, "Never execute shell."),
            Rule("l0r", P_L0, "Never execute shell."),
        ])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].id, "l0r")

    def test_l3_wins_over_l1(self):
        r = resolve_conflicts([
            Rule("l1r", P_L1, "Use Python."),
            Rule("l3r", P_L3, "Use shell scripts."),
        ])
        # Different content → both survive; L3 comes first
        self.assertEqual(r[0].id, "l3r")

    def test_l1_wins_over_l2_conflict(self):
        r = resolve_conflicts([
            Rule("l2r", P_L2, "Use tabs."),
            Rule("l1r", P_L1, "Use tabs."),
        ])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].id, "l1r")

    def test_full_ordering(self):
        r = resolve_conflicts([
            Rule("l2", P_L2, "Be creative."),
            Rule("l1", P_L1, "Be factual."),
            Rule("l3", P_L3, "Be concise."),
            Rule("l0", P_L0, "Be helpful."),
        ])
        self.assertEqual([rr.id for rr in r], ["l0", "l3", "l1", "l2"])

    def test_empty(self):
        self.assertEqual(resolve_conflicts([]), [])


class TestCheckToolBlock(unittest.TestCase):
    """L3 tool blocking."""

    def test_block_specific_call(self):
        r = [Rule("b", P_L3, "No rm -rf /",
                  tool_block={"tool": "bash", "args": {"command": "rm -rf /"}})]
        self.assertFalse(check_tool_block(r, "bash", {"command": "rm -rf /"}))
        self.assertTrue(check_tool_block(r, "bash", {"command": "ls -la"}))

    def test_block_by_tool_name(self):
        r = [Rule("b", P_L3, "No shutdown", tool_block={"tool": "shutdown"})]
        self.assertFalse(check_tool_block(r, "shutdown", {"reason": "test"}))
        self.assertTrue(check_tool_block(r, "bash", {"command": "echo hi"}))

    def test_no_block_rules(self):
        r = [Rule("r", P_L1, "Be nice.")]
        self.assertTrue(check_tool_block(r, "bash", {"command": "rm -rf /"}))

    def test_args_must_match(self):
        r = [Rule("b", P_L3, "No write /etc",
                  tool_block={"tool": "write_file", "args": {"path": "/etc/passwd"}})]
        self.assertTrue(check_tool_block(r, "write_file", {"path": "/tmp/x"}))
        self.assertFalse(check_tool_block(r, "write_file", {"path": "/etc/passwd"}))


class TestInjectSystemPrompt(unittest.TestCase):
    """Hard constraints before soft rules."""

    def test_hard_before_soft(self):
        p = inject_system_prompt([
            Rule("l1", P_L1, "Use Python."),
            Rule("l3", P_L3, "No network."),
            Rule("l0", P_L0, "Safe always."),
        ], "Base.")
        self.assertLess(p.index("Hard Constraints"), p.index("Soft Rules"))

    def test_labels_present(self):
        p = inject_system_prompt([
            Rule("r0", P_L0, "C."), Rule("r1", P_L1, "P."),
            Rule("r2", P_L2, "U."), Rule("r3", P_L3, "G."),
        ], "B.")
        for L in ("[L0]", "[L1]", "[L2]", "[L3]"):
            self.assertIn(L, p)

    def test_no_rules(self):
        self.assertEqual(inject_system_prompt([], "Hello"), "Hello")

    def test_no_base(self):
        self.assertIn("Soft Rules", inject_system_prompt([Rule("r", P_L1, "Go.")]))


class TestLoadRules(unittest.TestCase):
    def test_from_config(self):
        r = load_rules(config={"rules": [
            {"id": "a", "priority": 1, "content": "A"},
            {"id": "b", "priority": 3, "content": "B"},
        ]})
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0].id, "a")

    def test_from_skill_meta(self):
        r = load_rules(skill_metadata=[{
            "name": "safety", "rule_priority": "L3",
            "rules": [{"id": "s1", "content": "Safe first"}],
        }])
        self.assertEqual(r[0].priority, P_L3)

    def test_no_sources(self):
        self.assertEqual(load_rules(), [])


class TestPlugin(unittest.TestCase):
    def test_disabled(self):
        p = RulePriorityPlugin({"enabled": False})
        self.assertEqual(p.pre_llm_call("Base"), "Base")
        self.assertTrue(p.pre_tool_call("bash", {"command": "rm -rf /"}))

    def test_enabled_no_rules(self):
        self.assertEqual(RulePriorityPlugin({"enabled": True}).pre_llm_call("B"), "B")

    def test_enabled_with_rules(self):
        p = RulePriorityPlugin({"enabled": True, "rules": [
            {"id": "r1", "priority": 0, "content": "Core rule"},
        ]})
        self.assertIn("Core rule", p.pre_llm_call("Base"))

    def test_tool_blocking(self):
        p = RulePriorityPlugin({"enabled": True, "rules": [
            {"id": "b", "priority": 3, "content": "No rm -rf",
             "tool_block": {"tool": "bash", "args": {"command": "rm -rf /"}}},
        ]})
        self.assertFalse(p.pre_tool_call("bash", {"command": "rm -rf /"}))
        self.assertTrue(p.pre_tool_call("bash", {"command": "ls"}))

    def test_reload(self):
        cfg = {"enabled": True, "rules": [{"id": "r", "priority": 0, "content": "A"}]}
        p = RulePriorityPlugin(cfg)
        self.assertIn("A", p.pre_llm_call(""))
        cfg["rules"][0]["content"] = "B"
        p.reload()
        self.assertIn("B", p.pre_llm_call(""))

    def test_rules_copy(self):
        r = RulePriorityPlugin({"enabled": True}).rules
        r.append(Rule("x", P_L1, "X"))
        self.assertEqual(len(RulePriorityPlugin({"enabled": True}).rules), 0)


if __name__ == "__main__":
    unittest.main()
