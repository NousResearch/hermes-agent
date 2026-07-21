import unittest
from pathlib import Path

from tests.evals.no_tool_eval_buckets import (
    CandidateTrace,
    DIMENSIONS,
    Fixture,
    load_fixtures,
    score_trace,
    summarize_fixture_set,
    validate_fixture,
)

ROOT = Path(__file__).parent
FIXTURE_PATH = ROOT / "fixtures" / "no_tool_eval_fixtures.json"


class NoToolEvalBucketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {fixture.id: fixture for fixture in load_fixtures(FIXTURE_PATH)}

    def test_seed_fixture_set_is_valid_and_balanced(self):
        summary = summarize_fixture_set(list(self.fixtures.values()))
        self.assertEqual(summary["invalid"], {})
        self.assertGreaterEqual(summary["categories"].get("no_tool", 0), 1)
        self.assertGreaterEqual(summary["categories"].get("clarification", 0), 1)
        self.assertGreaterEqual(summary["categories"].get("loaded_context", 0), 1)
        self.assertGreaterEqual(summary["categories"].get("required_tool", 0), 1)
        self.assertEqual(summary["dimension_count"], 6)

    def test_no_tool_fixture_fails_if_agent_calls_tool(self):
        fixture = self.fixtures["telegram-ambient-bot-no-reply"]
        result = score_trace(fixture, CandidateTrace(final_action="no_reply", tools_called=("send_message",)))
        self.assertFalse(result["ok"])
        self.assertEqual(result["dimensions"]["tool_selection"], "fail")
        self.assertIn("called tools", result["failures"][0])

    def test_loaded_context_fixture_requires_answer_content_without_search(self):
        fixture = self.fixtures["answer-from-loaded-context-no-search"]
        good = score_trace(
            fixture,
            CandidateTrace(
                final_action="answer_from_loaded_context",
                final_text="Inspect the latest injected output first, then the route packet.",
            ),
        )
        self.assertTrue(good["ok"])

        bad = score_trace(
            fixture,
            CandidateTrace(final_action="answer_from_loaded_context", final_text="Check the logs."),
        )
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["dimensions"]["result_utilization"], "fail")

    def test_required_tool_fixture_fails_without_tool_and_passes_with_allowed_tool(self):
        fixture = self.fixtures["arithmetic-requires-tool"]
        missing = score_trace(fixture, CandidateTrace(final_action="call_tool"))
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["dimensions"]["tool_selection"], "fail")

        present = score_trace(fixture, CandidateTrace(final_action="call_tool", tools_called=("terminal",), final_text="422963"))
        self.assertTrue(present["ok"])

    def test_fixture_validation_rejects_conflicting_tool_contract(self):
        fixture = self.fixtures["arithmetic-requires-tool"]
        broken = fixture.__class__(
            id="broken",
            category="bad",
            prompt="bad",
            expected_action="call_tool",
            must_not_call_tools=True,
            must_call_tool=True,
            expected_dimensions=fixture.expected_dimensions,
        )
        errors = validate_fixture(broken)
        self.assertTrue(any("cannot require both" in error for error in errors))

    def test_score_trace_marks_invalid_fixture_not_ok_even_when_trace_matches(self):
        invalid = Fixture(
            id="invalid-fixture",
            category="bad",
            prompt="bad",
            expected_action="call_tool",
            must_not_call_tools=True,
            expected_dimensions={name: "pass" for name in DIMENSIONS},
        )
        result = score_trace(invalid, CandidateTrace(final_action="call_tool"))
        self.assertFalse(result["ok"])
        self.assertTrue(any("must use a no-tool expected_action" in error for error in result["failures"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
