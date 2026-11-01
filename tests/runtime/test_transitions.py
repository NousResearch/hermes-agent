"""Transition guard: legal step orderings."""

from __future__ import annotations

import unittest

from agent.runtime import (
    ActionStep,
    FinalAnswerStep,
    PlanningStep,
    TaskStep,
    TransitionGuard,
)


class TransitionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = TransitionGuard()

    def test_initial_state_only_allows_task_step(self) -> None:
        ok, _ = self.guard.check(None, TaskStep(step_number=0, task="t"))
        self.assertTrue(ok)
        ok, reason = self.guard.check(None, ActionStep(step_number=1))
        self.assertFalse(ok)
        self.assertIn("illegal transition", reason)

    def test_task_can_go_to_action_or_planning(self) -> None:
        task = TaskStep(step_number=0, task="t")
        self.assertTrue(self.guard.check(task, ActionStep(step_number=1))[0])
        self.assertTrue(self.guard.check(task, PlanningStep(step_number=1))[0])

    def test_action_can_go_to_action_planning_or_final(self) -> None:
        action = ActionStep(step_number=1)
        self.assertTrue(self.guard.check(action, ActionStep(step_number=2))[0])
        self.assertTrue(self.guard.check(action, PlanningStep(step_number=2))[0])
        self.assertTrue(self.guard.check(action, FinalAnswerStep(step_number=2))[0])

    def test_action_cannot_go_back_to_task(self) -> None:
        action = ActionStep(step_number=1)
        ok, reason = self.guard.check(action, TaskStep(step_number=2, task="t"))
        self.assertFalse(ok)
        self.assertIn("ActionStep -> TaskStep", reason)

    def test_final_answer_is_terminal(self) -> None:
        final = FinalAnswerStep(step_number=5, output="x")
        for next_step in (
            TaskStep(step_number=6, task="t"),
            ActionStep(step_number=6),
            PlanningStep(step_number=6),
            FinalAnswerStep(step_number=6, output="y"),
        ):
            ok, _ = self.guard.check(final, next_step)
            self.assertFalse(ok, f"FinalAnswerStep should be terminal, but allowed {type(next_step).__name__}")


if __name__ == "__main__":
    unittest.main()
