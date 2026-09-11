import unittest

from gateway.work_router.meeting_plan import MeetingPlan
from gateway.work_router.meeting_orchestrator import (
    MeetingRoundOrchestrator,
    RoundContribution,
)


class DynamicMeetingRoundTest(unittest.TestCase):

    def setUp(self):
        self.plan = MeetingPlan.minimal(
            objective="Choose the strongest supported proposal.",
            participants=("hans", "mason", "watson"),
        )
        self.orchestrator = MeetingRoundOrchestrator(self.plan)

    def test_round_one_assignments_are_independent(self):
        assignments = self.orchestrator.build_round(
            round_id=1,
            prior_contributions={
                "mason": "Option B is best.",
                "watson": "The evidence is incomplete.",
            },
        )

        self.assertEqual(
            tuple(item.profile for item in assignments),
            ("hans", "mason", "watson"),
        )

        hans = next(item for item in assignments if item.profile == "hans")

        self.assertNotIn("Option B is best.", hans.context)
        self.assertNotIn("The evidence is incomplete.", hans.context)
        self.assertTrue(hans.independent)

    def test_round_two_can_include_peer_arguments(self):
        assignments = self.orchestrator.build_round(
            round_id=2,
            prior_contributions={
                "mason": "Option B is best.",
                "watson": "The evidence is incomplete.",
            },
        )

        hans = next(item for item in assignments if item.profile == "hans")

        self.assertIn("Option B is best.", hans.context)
        self.assertIn("The evidence is incomplete.", hans.context)
        self.assertFalse(hans.independent)

    def test_round_does_not_advance_until_all_participants_submit(self):
        contributions = (
            RoundContribution(
                profile="hans",
                statement="Option A has lower commercial risk.",
            ),
            RoundContribution(
                profile="mason",
                statement="Option B has stronger strategic value.",
            ),
        )

        decision = self.orchestrator.evaluate_round(
            round_id=1,
            contributions=contributions,
            key_conflicts_remaining=True,
            material_uncertainties_remaining=False,
        )

        self.assertFalse(decision.round_complete)
        self.assertFalse(decision.advance)
        self.assertEqual(decision.missing_participants, ("watson",))

    def test_complete_round_can_advance_when_conflict_remains(self):
        contributions = (
            RoundContribution(profile="hans", statement="Prefer A."),
            RoundContribution(profile="mason", statement="Prefer B."),
            RoundContribution(
                profile="watson",
                statement="Evidence does not resolve A versus B.",
            ),
        )

        decision = self.orchestrator.evaluate_round(
            round_id=1,
            contributions=contributions,
            key_conflicts_remaining=True,
            material_uncertainties_remaining=False,
        )

        self.assertTrue(decision.round_complete)
        self.assertTrue(decision.advance)
        self.assertEqual(decision.next_round_id, 2)

    def test_complete_round_can_converge_early(self):
        contributions = (
            RoundContribution(profile="hans", statement="Prefer A."),
            RoundContribution(profile="mason", statement="A is acceptable."),
            RoundContribution(
                profile="watson",
                statement="Evidence supports A.",
            ),
        )

        decision = self.orchestrator.evaluate_round(
            round_id=1,
            contributions=contributions,
            key_conflicts_remaining=False,
            material_uncertainties_remaining=False,
        )

        self.assertTrue(decision.round_complete)
        self.assertFalse(decision.advance)
        self.assertTrue(decision.converged)
        self.assertIsNone(decision.next_round_id)


if __name__ == "__main__":
    unittest.main()