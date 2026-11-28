"""CallbackRegistry dispatch semantics with RunState."""

from __future__ import annotations

import unittest

from agent.runtime import (
    ActionStep,
    CallbackRegistry,
    FinalAnswerStep,
    FrozenClock,
    MemoryStep,
    PlanningStep,
    RunState,
    TaskStep,
)


def make_state() -> RunState:
    return RunState(FrozenClock())


class CallbackTests(unittest.TestCase):
    def test_registration_rejects_non_memorystep_type(self) -> None:
        registry = CallbackRegistry()
        with self.assertRaises(TypeError):
            registry.register(str, lambda s, state: None)  # type: ignore[arg-type]

    def test_dispatch_by_concrete_type(self) -> None:
        registry = CallbackRegistry()
        fired: list[str] = []
        registry.register(ActionStep, lambda s, state: fired.append("action"))
        registry.register(FinalAnswerStep, lambda s, state: fired.append("final"))

        registry.dispatch(ActionStep(step_number=1), make_state())
        registry.dispatch(FinalAnswerStep(step_number=2, output=None), make_state())
        registry.dispatch(TaskStep(step_number=0, task="t"), make_state())

        self.assertEqual(fired, ["action", "final"])

    def test_dispatch_by_base_class_catches_all(self) -> None:
        registry = CallbackRegistry()
        seen: list[str] = []
        registry.register(MemoryStep, lambda s, state: seen.append(type(s).__name__))

        registry.dispatch(TaskStep(step_number=0, task="t"), make_state())
        registry.dispatch(ActionStep(step_number=1), make_state())
        registry.dispatch(PlanningStep(step_number=2), make_state())
        registry.dispatch(FinalAnswerStep(step_number=3, output=None), make_state())

        self.assertEqual(seen, ["TaskStep", "ActionStep", "PlanningStep", "FinalAnswerStep"])

    def test_multiple_callbacks_fire_in_registration_order(self) -> None:
        registry = CallbackRegistry()
        order: list[str] = []
        registry.register(ActionStep, lambda s, state: order.append("first"))
        registry.register(ActionStep, lambda s, state: order.append("second"))
        registry.dispatch(ActionStep(step_number=1), make_state())
        self.assertEqual(order, ["first", "second"])

    def test_callbacks_mutate_state_via_typed_api(self) -> None:
        registry = CallbackRegistry()

        def increment(step: MemoryStep, state: RunState) -> None:
            state.increment("budget", reason="action step consumed budget")

        registry.register(ActionStep, increment)
        shared = make_state()
        registry.dispatch(ActionStep(step_number=1), shared)
        registry.dispatch(ActionStep(step_number=2), shared)
        self.assertEqual(shared.get("budget"), 2)
        # Every mutation recorded.
        self.assertEqual(len(shared.mutations), 2)
        self.assertTrue(all(m.reason for m in shared.mutations))


if __name__ == "__main__":
    unittest.main()
