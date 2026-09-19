import json
import unittest

from clicksmith.foundation.execution_identity import ExecutionIdentity
from clicksmith.foundation.execution_state import (
    AUTHORIZED,
    CANCELLED,
    CANONICAL_STATES,
    FAILED,
    PENDING,
    PAUSED,
    RUNNING,
    SUCCEEDED,
    TERMINAL_STATES,
    ExecutionState,
    InvalidIdentityError,
    InvalidStateError,
    InvalidTransitionError,
    StateSchemaError,
    TerminalStateMutationError,
)


class TestExecutionState(unittest.TestCase):
    def setUp(self):
        self.identity = ExecutionIdentity(
            task_id="task-001",
            run_id="run-001",
            attempt_id="attempt-001",
            session_id="session-001",
            revision=1,
        )

    def test_canonical_states(self):
        self.assertEqual(
            CANONICAL_STATES,
            {
                PENDING,
                AUTHORIZED,
                RUNNING,
                PAUSED,
                SUCCEEDED,
                FAILED,
                CANCELLED,
            },
        )

    def test_terminal_states(self):
        self.assertEqual(
            TERMINAL_STATES,
            {SUCCEEDED, FAILED, CANCELLED},
        )

    def test_valid_transition_matrix(self):
        valid_paths = [
            (PENDING, AUTHORIZED),
            (PENDING, CANCELLED),
            (AUTHORIZED, RUNNING),
            (AUTHORIZED, CANCELLED),
            (RUNNING, PAUSED),
            (RUNNING, SUCCEEDED),
            (RUNNING, FAILED),
            (RUNNING, CANCELLED),
            (PAUSED, RUNNING),
            (PAUSED, CANCELLED),
        ]

        for source, target in valid_paths:
            state = ExecutionState(self.identity, source)
            state.transition(target)
            self.assertEqual(state.state, target)

    def test_invalid_transitions_fail_closed(self):
        invalid_paths = [
            (PENDING, RUNNING),
            (PENDING, SUCCEEDED),
            (PENDING, FAILED),
            (AUTHORIZED, PAUSED),
            (AUTHORIZED, SUCCEEDED),
            (RUNNING, PENDING),
            (RUNNING, AUTHORIZED),
            (PAUSED, PENDING),
            (PAUSED, AUTHORIZED),
            (PAUSED, SUCCEEDED),
        ]

        for source, target in invalid_paths:
            state = ExecutionState(self.identity, source)

            with self.assertRaises(InvalidTransitionError):
                state.transition(target)

            self.assertEqual(state.state, source)

    def test_terminal_states_are_immutable(self):
        for terminal in TERMINAL_STATES:
            state = ExecutionState(self.identity, terminal)

            for target in CANONICAL_STATES:
                with self.assertRaises(TerminalStateMutationError):
                    state.transition(target)

                self.assertEqual(state.state, terminal)

    def test_unknown_state_rejected(self):
        with self.assertRaises(InvalidStateError):
            ExecutionState(self.identity, "UNKNOWN")

    def test_non_string_state_rejected(self):
        with self.assertRaises(InvalidStateError):
            ExecutionState(self.identity, 123)

    def test_non_execution_identity_rejected(self):
        with self.assertRaises(InvalidIdentityError):
            ExecutionState("not-an-identity")

    def test_identity_preserved_across_transition(self):
        state = ExecutionState(self.identity)

        before = state.identity.to_dict()

        state.transition(AUTHORIZED)
        state.transition(RUNNING)
        state.transition(PAUSED)
        state.transition(RUNNING)
        state.transition(FAILED)

        self.assertEqual(state.identity.to_dict(), before)

    def test_to_dict_is_canonical(self):
        state = ExecutionState(self.identity, RUNNING)

        self.assertEqual(
            state.to_dict(),
            {
                "identity": self.identity.to_dict(),
                "schema_version": "1.0",
                "state": RUNNING,
            },
        )

    def test_serialization_is_deterministic(self):
        state = ExecutionState(self.identity, RUNNING)

        first = state.serialize()
        second = state.serialize()

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def test_round_trip_serialization(self):
        original = ExecutionState(self.identity, PAUSED)

        restored = ExecutionState.deserialize(original.serialize())

        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertEqual(restored.serialize(), original.serialize())

    def test_missing_required_field_rejected(self):
        value = ExecutionState(self.identity, RUNNING).to_dict()

        for field in ("identity", "schema_version", "state"):
            malformed = dict(value)
            del malformed[field]

            with self.assertRaises(StateSchemaError):
                ExecutionState.from_dict(malformed)

    def test_unknown_field_rejected(self):
        value = ExecutionState(self.identity, RUNNING).to_dict()
        value["unexpected"] = "reject-me"

        with self.assertRaises(StateSchemaError):
            ExecutionState.from_dict(value)

    def test_unsupported_schema_rejected(self):
        value = ExecutionState(self.identity, RUNNING).to_dict()
        value["schema_version"] = "99.0"

        with self.assertRaises(StateSchemaError):
            ExecutionState.from_dict(value)

    def test_invalid_identity_rejected(self):
        value = ExecutionState(self.identity, RUNNING).to_dict()
        value["identity"] = "invalid"

        with self.assertRaises(InvalidIdentityError):
            ExecutionState.from_dict(value)

    def test_invalid_serialized_json_rejected(self):
        with self.assertRaises(StateSchemaError):
            ExecutionState.deserialize("{not-valid-json")

    def test_non_string_serialized_payload_rejected(self):
        with self.assertRaises(StateSchemaError):
            ExecutionState.deserialize(123)

    def test_error_and_non_lifecycle_values_rejected(self):
        for value in (
            "UNKNOWN",
            "ERROR",
            "VERIFIED",
            "UNVERIFIED",
            "BLOCKED",
            "RECOVERING",
            "RETRYING",
        ):
            with self.assertRaises(InvalidStateError):
                ExecutionState(self.identity, value)

    def test_failed_transition_does_not_mutate_identity(self):
        state = ExecutionState(self.identity, RUNNING)

        before_identity = state.identity.to_dict()
        before_state = state.state

        with self.assertRaises(InvalidTransitionError):
            state.transition(PENDING)

        self.assertEqual(state.state, before_state)
        self.assertEqual(state.identity.to_dict(), before_identity)


if __name__ == "__main__":
    unittest.main()
