import unittest

from gateway.work_router.meeting_plan import (
    ConvergencePolicy,
    MeetingPlan,
    ParticipantBrief,
    RoundPolicy,
)


class DynamicMeetingPlanTest(unittest.TestCase):
    def test_dynamic_meeting_plan_preserves_staff_freedom(self):
        plan = MeetingPlan(
            objective="Compare alternatives and reach the strongest supported conclusion.",
            participants=("hans", "mason", "watson"),
            participant_briefs=(
                ParticipantBrief(
                    profile="hans",
                    role="commercial perspective",
                    question="Assess cost, feasibility, and commercial risks.",
                ),
                ParticipantBrief(
                    profile="mason",
                    role="strategy perspective",
                    question="Challenge assumptions and identify stronger alternatives.",
                ),
                ParticipantBrief(
                    profile="watson",
                    role="evidence perspective",
                    question="Verify material claims and identify unsupported assumptions.",
                ),
            ),
            round_policy=RoundPolicy(
                initial_independence=True,
                allow_cross_domain_challenge=True,
                allow_position_revision=True,
                allow_problem_reframing=True,
                safety_round_cap=5,
            ),
            convergence_policy=ConvergencePolicy(
                require_resolved_key_conflicts=True,
                require_uncertainties_reported=True,
                allow_early_finish=True,
            ),
        )

        self.assertEqual(plan.participants, ("hans", "mason", "watson"))
        self.assertTrue(plan.round_policy.initial_independence)
        self.assertTrue(plan.round_policy.allow_position_revision)
        self.assertTrue(plan.round_policy.allow_problem_reframing)
        self.assertTrue(plan.convergence_policy.allow_early_finish)

    def test_round_one_context_is_independent(self):
        plan = MeetingPlan.minimal(
            objective="Evaluate the proposal.",
            participants=("hans", "mason", "watson"),
        )

        context = plan.context_for_round(
            profile="hans",
            round_id=1,
            prior_contributions={
                "mason": "Adopt option B.",
                "watson": "Evidence is incomplete.",
            },
        )

        self.assertNotIn("Adopt option B.", context)
        self.assertNotIn("Evidence is incomplete.", context)

    def test_later_round_can_expose_peer_contributions(self):
        plan = MeetingPlan.minimal(
            objective="Evaluate the proposal.",
            participants=("hans", "mason", "watson"),
        )

        context = plan.context_for_round(
            profile="hans",
            round_id=2,
            prior_contributions={
                "mason": "Adopt option B.",
                "watson": "Evidence is incomplete.",
            },
        )

        self.assertIn("Adopt option B.", context)
        self.assertIn("Evidence is incomplete.", context)

    def test_safety_round_cap_is_not_required_round_count(self):
        plan = MeetingPlan.minimal(
            objective="Evaluate the proposal.",
            participants=("hans", "mason", "watson"),
        )

        self.assertFalse(
            plan.should_continue(
                round_id=1,
                key_conflicts_remaining=False,
                material_uncertainties_remaining=False,
            )
        )
        self.assertTrue(
            plan.should_continue(
                round_id=1,
                key_conflicts_remaining=True,
                material_uncertainties_remaining=False,
            )
        )
        self.assertFalse(
            plan.should_continue(
                round_id=5,
                key_conflicts_remaining=True,
                material_uncertainties_remaining=True,
            )
        )

    def test_plan_rejects_duplicate_participants(self):
        with self.assertRaises(ValueError):
            MeetingPlan.minimal(
                objective="Evaluate the proposal.",
                participants=("hans", "hans", "mason"),
            )
