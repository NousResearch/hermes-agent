"""RunState typed mutation log."""

from __future__ import annotations

import unittest

from agent.runtime import FrozenClock, RunState, StateFrozenError


def make_state() -> RunState:
    return RunState(FrozenClock())


class RunStateTests(unittest.TestCase):
    def test_set_records_mutation_with_reason(self) -> None:
        state = make_state()
        state.set("model", "claude", reason="caller chose model")
        self.assertEqual(state.get("model"), "claude")
        self.assertEqual(len(state.mutations), 1)
        m = state.mutations[0]
        self.assertEqual(m.kind, "set")
        self.assertEqual(m.key, "model")
        self.assertEqual(m.value, "claude")
        self.assertEqual(m.reason, "caller chose model")

    def test_mutation_without_reason_is_rejected(self) -> None:
        state = make_state()
        with self.assertRaises(ValueError):
            state.set("x", 1, reason="")

    def test_append_to_non_list_raises(self) -> None:
        state = make_state()
        state.set("not_a_list", 1, reason="set scalar")
        with self.assertRaises(TypeError):
            state.append("not_a_list", "x", reason="should fail")

    def test_increment_on_non_numeric_raises(self) -> None:
        state = make_state()
        state.set("not_a_number", "x", reason="set string")
        with self.assertRaises(TypeError):
            state.increment("not_a_number", reason="should fail")

    def test_increment_starts_from_zero(self) -> None:
        state = make_state()
        new_value = state.increment("budget", reason="first tick")
        self.assertEqual(new_value, 1)
        new_value = state.increment("budget", amount=5, reason="big jump")
        self.assertEqual(new_value, 6)
        self.assertEqual(state.get("budget"), 6)

    def test_frozen_state_rejects_further_mutation(self) -> None:
        state = make_state()
        state.set("x", 1, reason="initial")
        state.freeze()
        self.assertTrue(state.frozen)
        with self.assertRaises(StateFrozenError):
            state.set("x", 2, reason="post-freeze")
        with self.assertRaises(StateFrozenError):
            state.increment("x", reason="post-freeze")
        # But reads still work.
        self.assertEqual(state.get("x"), 1)

    def test_snapshot_is_a_copy(self) -> None:
        state = make_state()
        state.set("k", [1, 2], reason="seed")
        snap = state.snapshot()
        snap["k"].append(3)  # mutation of snapshot copy
        # Underlying list is shared by reference (shallow copy) — this is
        # documented behavior. Asserting it so the contract is explicit.
        self.assertEqual(state.get("k"), [1, 2, 3])

    def test_delete_records_mutation(self) -> None:
        state = make_state()
        state.set("k", 1, reason="seed")
        state.delete("k", reason="cleanup")
        self.assertIsNone(state.get("k"))
        self.assertEqual(state.mutations[-1].kind, "delete")


if __name__ == "__main__":
    unittest.main()
