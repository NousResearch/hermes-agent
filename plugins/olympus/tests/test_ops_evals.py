import json
import unittest

from _load_plugin_api import plugin_api


class OpsEvalTest(unittest.TestCase):
    def test_operational_evals_rank_reliability_routing_skill_and_efficiency(self):
        payload = plugin_api.build_ops_evals(
            sessions=[
                {"state": "stale", "tool_call_count": 45, "message_count": 48, "avg_input_tokens": 62000},
                {"state": "completed", "tool_call_count": 24, "message_count": 12, "avg_input_tokens": 1800},
            ],
            kanban={
                "open": 8,
                "totals": {"blocked": 2, "ready": 2},
                "attention": [{"severity": "warning", "label": "Blocked work"}],
                "assignee_load": [{"assignee": "Default", "open": 8, "running": 3, "blocked": 2}],
                "recent_tasks": [{"status": "ready", "assignee": "unassigned"}],
                "recent_runs": [{"status": "failed", "outcome": "crashed"}],
            },
            skill_coverage={"summary": {"zero_skill_profiles": 1, "forced_skill_tasks": 2}},
            skill_hygiene={"summary": {"forced_skill_metadata_gaps": 1, "hub_audit_warn": 1, "hub_audit_fail": 1}},
            config_policy={"summary": {"max_turns": 120, "hard_loop_stop": False, "browser_private_flags": 1}},
        )

        by_id = {item["id"]: item for item in payload["items"]}
        self.assertEqual(set(by_id), {"reliability", "routing", "skill_use", "efficiency"})
        self.assertEqual(payload["summary"]["checks"], 4)
        self.assertLess(payload["summary"]["score"], 100)
        self.assertIn(payload["summary"]["state"], {"warning", "critical"})
        self.assertEqual(by_id["reliability"]["recommended_view"], "/kanban")
        self.assertEqual(by_id["routing"]["recommended_view"], "/kanban")
        self.assertEqual(by_id["skill_use"]["recommended_view"], "/skills")
        self.assertEqual(by_id["efficiency"]["recommended_view"], "/sessions")

        serialized = json.dumps(payload).lower()
        self.assertNotIn("answer quality", serialized)
        self.assertNotIn("hallucination", serialized)
        self.assertNotIn("benchmark", serialized)

    def test_operational_evals_pass_for_clean_operational_inputs(self):
        payload = plugin_api.build_ops_evals(
            sessions=[{"state": "completed", "tool_call_count": 5, "message_count": 8, "avg_input_tokens": 1200}],
            kanban={"open": 0, "totals": {"blocked": 0, "ready": 0}, "recent_runs": [], "recent_tasks": []},
            skill_coverage={"summary": {"zero_skill_profiles": 0, "forced_skill_tasks": 0}},
            skill_hygiene={"summary": {"forced_skill_metadata_gaps": 0, "hub_audit_warn": 0, "hub_audit_fail": 0}},
            config_policy={"summary": {"max_turns": 40, "hard_loop_stop": True, "browser_private_flags": 0}},
        )

        self.assertEqual(payload["summary"]["state"], "ok")
        self.assertEqual(payload["summary"]["score"], 100)
        self.assertEqual(payload["summary"]["warnings"], 0)
        self.assertTrue(all(item["state"] == "ok" for item in payload["items"]))


if __name__ == "__main__":
    unittest.main()
