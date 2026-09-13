import unittest

from gateway.work_router.meeting_plan import MeetingPlan
from gateway.work_router.meeting_controller import (
    MeetingRoundController,
    MeetingRoundOutcome,
)


class DynamicMeetingControllerTest(unittest.TestCase):

    def setUp(self):
        self.plan = MeetingPlan.minimal(
            objective="Choose the strongest supported proposal.",
            participants=("hans", "mason", "watson"),
        )
        self.controller = MeetingRoundController(self.plan)

    def test_round_one_creates_independent_assignments(self):
        assignments = self.controller.start_round(
            round_id=1,
            prior_contributions={},
        )

        self.assertEqual(
            tuple(a.profile for a in assignments),
            ("hans", "mason", "watson"),
        )
        self.assertTrue(all(a.independent for a in assignments))

    def test_completed_round_uses_demian_analysis_to_continue(self):
        analyzer_response = """
        {
          "key_conflicts": ["Hans prefers A while Mason prefers B."],
          "uncertainties": [],
          "evidence_gaps": ["Lifecycle evidence remains unverified."],
          "challenged_assumptions": [],
          "promising_alternatives": [],
          "participants_needed_next": ["hans", "mason", "watson"],
          "next_round_focus": "Resolve the disagreement and verify evidence.",
          "convergence_recommended": false
        }
        """

        outcome = self.controller.complete_round(
            round_id=1,
            contributions={
                "hans": "I prefer A.",
                "mason": "I prefer B.",
                "watson": "Evidence remains incomplete.",
            },
            analyzer_response=analyzer_response,
        )

        self.assertIsInstance(outcome, MeetingRoundOutcome)
        self.assertTrue(outcome.continue_meeting)
        self.assertFalse(outcome.converged)
        self.assertEqual(outcome.next_round_id, 2)
        self.assertEqual(
            outcome.next_participants,
            ("hans", "mason", "watson"),
        )

    def test_next_round_can_use_subset_of_staff(self):
        analyzer_response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": ["Lifecycle evidence remains unverified."],
          "challenged_assumptions": [],
          "promising_alternatives": [],
          "participants_needed_next": ["watson"],
          "next_round_focus": "Verify lifecycle evidence.",
          "convergence_recommended": false
        }
        """

        outcome = self.controller.complete_round(
            round_id=1,
            contributions={
                "hans": "Commercial position is settled.",
                "mason": "Strategy position is settled.",
                "watson": "Evidence remains incomplete.",
            },
            analyzer_response=analyzer_response,
        )

        self.assertTrue(outcome.continue_meeting)
        self.assertEqual(outcome.next_participants, ("watson",))

    def test_clean_analysis_converges_early(self):
        analyzer_response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": [],
          "challenged_assumptions": [],
          "promising_alternatives": ["A phased rollout is viable."],
          "participants_needed_next": [],
          "next_round_focus": "",
          "convergence_recommended": true
        }
        """

        outcome = self.controller.complete_round(
            round_id=2,
            contributions={
                "hans": "The phased option is commercially viable.",
                "mason": "The phased option resolves the strategic concern.",
                "watson": "The evidence now supports the assumptions.",
            },
            analyzer_response=analyzer_response,
        )

        self.assertFalse(outcome.continue_meeting)
        self.assertTrue(outcome.converged)
        self.assertIsNone(outcome.next_round_id)

    def test_analyzer_cannot_force_false_convergence(self):
        analyzer_response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": ["A material fact remains unverified."],
          "challenged_assumptions": [],
          "promising_alternatives": [],
          "participants_needed_next": ["watson"],
          "next_round_focus": "Verify the remaining fact.",
          "convergence_recommended": true
        }
        """

        outcome = self.controller.complete_round(
            round_id=1,
            contributions={
                "hans": "No objection.",
                "mason": "No objection.",
                "watson": "One fact remains unverified.",
            },
            analyzer_response=analyzer_response,
        )

        self.assertTrue(outcome.continue_meeting)
        self.assertFalse(outcome.converged)


if __name__ == "__main__":
    unittest.main()