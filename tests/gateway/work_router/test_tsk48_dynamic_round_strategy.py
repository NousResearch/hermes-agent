import unittest

from gateway.work_router.meeting_strategy import (
    DynamicRoundStrategy,
    RoundAnalysis,
)


class DynamicRoundStrategyTest(unittest.TestCase):

    def setUp(self):
        self.strategy = DynamicRoundStrategy(
            participants=("hans", "mason", "watson"),
        )

    def test_conflict_selects_only_relevant_next_participants(self):
        analysis = RoundAnalysis(
            key_conflicts=(
                "Hans and Mason disagree on whether the higher initial cost is justified.",
            ),
            uncertainties=(),
            evidence_gaps=(
                "Lifecycle policy requires verification.",
            ),
            challenged_assumptions=(),
            promising_alternatives=(),
            participants_needed_next=("hans", "mason", "watson"),
            next_round_focus=(
                "Resolve investment justification and verify lifecycle evidence."
            ),
            convergence_recommended=False,
        )

        decision = self.strategy.decide_next_round(
            round_id=1,
            analysis=analysis,
        )

        self.assertTrue(decision.continue_meeting)
        self.assertEqual(
            decision.participants,
            ("hans", "mason", "watson"),
        )
        self.assertIn("investment", decision.focus.lower())

    def test_strategy_can_converge_without_using_all_rounds(self):
        analysis = RoundAnalysis(
            key_conflicts=(),
            uncertainties=(),
            evidence_gaps=(),
            challenged_assumptions=(),
            promising_alternatives=(),
            participants_needed_next=(),
            next_round_focus="",
            convergence_recommended=True,
        )

        decision = self.strategy.decide_next_round(
            round_id=2,
            analysis=analysis,
        )

        self.assertFalse(decision.continue_meeting)
        self.assertTrue(decision.converged)
        self.assertEqual(decision.participants, ())

    def test_unresolved_evidence_prevents_false_convergence(self):
        analysis = RoundAnalysis(
            key_conflicts=(),
            uncertainties=(),
            evidence_gaps=("EOS date is not verified.",),
            challenged_assumptions=(),
            promising_alternatives=(),
            participants_needed_next=("watson",),
            next_round_focus="Verify the EOS evidence.",
            convergence_recommended=True,
        )

        decision = self.strategy.decide_next_round(
            round_id=1,
            analysis=analysis,
        )

        self.assertTrue(decision.continue_meeting)
        self.assertFalse(decision.converged)
        self.assertEqual(decision.participants, ("watson",))

    def test_strategy_rejects_unknown_participant(self):
        analysis = RoundAnalysis(
            key_conflicts=("A conflict remains.",),
            uncertainties=(),
            evidence_gaps=(),
            challenged_assumptions=(),
            promising_alternatives=(),
            participants_needed_next=("unknown-agent",),
            next_round_focus="Resolve conflict.",
            convergence_recommended=False,
        )

        with self.assertRaises(ValueError):
            self.strategy.decide_next_round(
                round_id=1,
                analysis=analysis,
            )

    def test_safety_cap_stops_even_when_conflict_remains(self):
        strategy = DynamicRoundStrategy(
            participants=("hans", "mason", "watson"),
            safety_round_cap=5,
        )

        analysis = RoundAnalysis(
            key_conflicts=("Conflict remains.",),
            uncertainties=("Decision remains uncertain.",),
            evidence_gaps=(),
            challenged_assumptions=(),
            promising_alternatives=(),
            participants_needed_next=("hans", "mason"),
            next_round_focus="Continue challenging the unresolved positions.",
            convergence_recommended=False,
        )

        decision = strategy.decide_next_round(
            round_id=5,
            analysis=analysis,
        )

        self.assertFalse(decision.continue_meeting)
        self.assertFalse(decision.converged)
        self.assertTrue(decision.safety_stop)


if __name__ == "__main__":
    unittest.main()