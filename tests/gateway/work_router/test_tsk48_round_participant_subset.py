import unittest

from gateway.work_router.meeting_controller import MeetingRoundController
from gateway.work_router.meeting_plan import (
    ConvergencePolicy,
    MeetingPlan,
    ParticipantBrief,
    RoundPolicy,
)


class RoundParticipantSubsetTest(unittest.TestCase):

    def setUp(self):
        self.plan = MeetingPlan(
            objective="Evaluate the strongest supported proposal.",
            participants=("hans", "mason"),
            participant_briefs=(
                ParticipantBrief("hans", "commercial", "Assess commercial risk."),
                ParticipantBrief("mason", "strategy", "Assess strategic alternatives."),
            ),
            round_policy=RoundPolicy(),
            convergence_policy=ConvergencePolicy(),
            initial_participants=("hans", "mason"),
            participant_pool=("hans", "mason", "watson"),
        )
        self.controller = MeetingRoundController(self.plan)

    def test_round_one_defaults_to_initial_participants(self):
        assignments = self.controller.start_round(
            round_id=1,
            prior_contributions={},
        )

        self.assertEqual(
            tuple(assignment.profile for assignment in assignments),
            ("hans", "mason"),
        )

    def test_later_round_can_target_pool_subset(self):
        assignments = self.controller.start_round(
            round_id=2,
            prior_contributions={"hans": "Commercial risk is settled."},
            round_participants=("watson",),
        )

        self.assertEqual(
            tuple(assignment.profile for assignment in assignments),
            ("watson",),
        )

    def test_subset_completion_waits_only_for_selected_participants(self):
        response = """
        {
          "key_conflicts": [],
          "uncertainties": [],
          "evidence_gaps": [],
          "challenged_assumptions": [],
          "promising_alternatives": ["Evidence is sufficient."],
          "participants_needed_next": [],
          "next_round_focus": "",
          "convergence_recommended": true
        }
        """

        outcome = self.controller.complete_round(
            round_id=2,
            round_participants=("watson",),
            contributions={"watson": "The evidence is now verified."},
            analyzer_response=response,
        )

        self.assertTrue(outcome.converged)
        self.assertFalse(outcome.continue_meeting)

    def test_round_subset_rejects_profile_outside_pool(self):
        with self.assertRaises(ValueError):
            self.controller.start_round(
                round_id=2,
                prior_contributions={},
                round_participants=("unknown",),
            )

    def test_round_subset_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            self.controller.start_round(
                round_id=2,
                prior_contributions={},
                round_participants=("watson", "watson"),
            )

    def test_round_subset_rejects_demian(self):
        with self.assertRaises(ValueError):
            self.controller.start_round(
                round_id=2,
                prior_contributions={},
                round_participants=("Demian",),
            )


if __name__ == "__main__":
    unittest.main()