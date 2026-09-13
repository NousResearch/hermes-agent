import unittest

from gateway.work_router.meeting_plan import MeetingPlan


class MeetingParticipantPoolTest(unittest.TestCase):

    def setUp(self):
        self.plan = MeetingPlan.minimal(
            objective="Evaluate proposal",
            participants=("hans", "mason"),
        )

    def test_plan_exposes_participant_pool(self):
        self.assertTrue(hasattr(self.plan, "participant_pool"))

    def test_plan_exposes_initial_participants(self):
        self.assertTrue(hasattr(self.plan, "initial_participants"))

    def test_initial_participants_preserve_original_selection(self):
        self.assertEqual(
            self.plan.initial_participants,
            ("hans", "mason"),
        )

    def test_initial_participants_are_inside_pool(self):
        self.assertTrue(
            set(self.plan.initial_participants)
            <= set(self.plan.participant_pool)
        )

    def test_legacy_participants_remain_compatible(self):
        self.assertEqual(
            self.plan.participants,
            self.plan.initial_participants,
        )


class MeetingParticipantPoolValidationTest(unittest.TestCase):

    def test_initial_participants_must_be_subset_of_pool(self):
        with self.assertRaises(ValueError):
            MeetingPlan(
                objective="Evaluate proposal",
                participants=("hans",),
                participant_briefs=(),
                round_policy=__import__(
                    "gateway.work_router.meeting_plan",
                    fromlist=["RoundPolicy"],
                ).RoundPolicy(),
                convergence_policy=__import__(
                    "gateway.work_router.meeting_plan",
                    fromlist=["ConvergencePolicy"],
                ).ConvergencePolicy(),
                initial_participants=("hans",),
                participant_pool=("mason",),
            )

    def test_participant_pool_rejects_demian(self):
        with self.assertRaises(ValueError):
            MeetingPlan.minimal(
                objective="Evaluate proposal",
                participants=("hans", "demian"),
            )

    def test_participant_pool_rejects_duplicates(self):
        plan = MeetingPlan.minimal(
            objective="Evaluate proposal",
            participants=("hans", "mason"),
        )

        with self.assertRaises(ValueError):
            MeetingPlan(
                objective=plan.objective,
                participants=plan.participants,
                participant_briefs=plan.participant_briefs,
                round_policy=plan.round_policy,
                convergence_policy=plan.convergence_policy,
                initial_participants=plan.initial_participants,
                participant_pool=("hans", "hans", "mason"),
            )


if __name__ == "__main__":
    unittest.main()