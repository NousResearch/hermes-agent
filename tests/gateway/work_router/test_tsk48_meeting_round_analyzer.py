import unittest

from gateway.work_router.meeting_analyzer import (
    MeetingRoundAnalyzer,
    RoundAnalysisParseError,
)


class MeetingRoundAnalyzerTest(unittest.TestCase):

    def setUp(self):
        self.analyzer = MeetingRoundAnalyzer(
            participants=("hans", "mason", "watson"),
        )

    def test_parser_accepts_structured_demian_analysis(self):
        response = """
        {
          "key_conflicts": [
            "Hans prefers A while Mason prefers B."
          ],
          "uncertainties": [],
          "evidence_gaps": [
            "Lifecycle evidence is not verified."
          ],
          "challenged_assumptions": [
            "The available budget is assumed to be fixed."
          ],
          "promising_alternatives": [
            "Phase the investment across two budget periods."
          ],
          "participants_needed_next": ["hans", "mason", "watson"],
          "next_round_focus": "Resolve the investment conflict and verify lifecycle evidence.",
          "convergence_recommended": false
        }
        """

        analysis = self.analyzer.parse(response)

        self.assertEqual(len(analysis.key_conflicts), 1)
        self.assertEqual(len(analysis.evidence_gaps), 1)
        self.assertEqual(
            analysis.participants_needed_next,
            ("hans", "mason", "watson"),
        )
        self.assertFalse(analysis.convergence_recommended)

    def test_parser_rejects_unknown_participant(self):
        response = """
        {
          "key_conflicts": ["A conflict remains."],
          "uncertainties": [],
          "evidence_gaps": [],
          "challenged_assumptions": [],
          "promising_alternatives": [],
          "participants_needed_next": ["unknown-agent"],
          "next_round_focus": "Resolve the conflict.",
          "convergence_recommended": false
        }
        """

        with self.assertRaises(RoundAnalysisParseError):
            self.analyzer.parse(response)

    def test_parser_rejects_missing_required_field(self):
        response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": []
        }
        """

        with self.assertRaises(RoundAnalysisParseError):
            self.analyzer.parse(response)

    def test_parser_rejects_false_convergence_with_evidence_gap(self):
        response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": ["EOS date remains unverified."],
          "challenged_assumptions": [],
          "promising_alternatives": [],
          "participants_needed_next": ["watson"],
          "next_round_focus": "Verify EOS evidence.",
          "convergence_recommended": true
        }
        """

        analysis = self.analyzer.parse(response)

        self.assertFalse(analysis.convergence_recommended)

    def test_prompt_does_not_predetermine_the_answer(self):
        prompt = self.analyzer.build_prompt(
            objective="Choose the strongest proposal.",
            round_id=1,
            contributions={
                "hans": "Option A is cheaper.",
                "mason": "Option B has better strategic value.",
                "watson": "Evidence is incomplete.",
            },
        )

        self.assertIn("Option A is cheaper.", prompt)
        self.assertIn("Option B has better strategic value.", prompt)
        self.assertIn("Evidence is incomplete.", prompt)

        self.assertIn("Do not predetermine the conclusion", prompt)
        self.assertIn("minority", prompt.lower())
        self.assertIn("reframe", prompt.lower())


if __name__ == "__main__":
    unittest.main()