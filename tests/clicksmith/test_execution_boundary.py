import unittest

from clicksmith.foundation.execution_boundary import (
    ExecutionBoundary,
    ExecutionNotAuthorizedError,
)
from clicksmith.foundation.execution_identity import ExecutionIdentity
from clicksmith.foundation.execution_state import (
    AUTHORIZED,
    CANCELLED,
    FAILED,
    PENDING,
    RUNNING,
    SUCCEEDED,
    ExecutionState,
)


class TestExecutionBoundary(unittest.TestCase):
    def setUp(self):
        self.identity = ExecutionIdentity(
            task_id="task-001",
            run_id="run-001",
            attempt_id="attempt-001",
            session_id="session-001",
            revision=1,
        )

    def make_state(self, state=AUTHORIZED):
        return ExecutionState(self.identity, state)

    def test_requires_authorized_state(self):
        calls = []

        def executor():
            calls.append(True)
            return {"completed": True}

        boundary = ExecutionBoundary(
            self.make_state(PENDING),
            executor,
        )

        with self.assertRaises(ExecutionNotAuthorizedError):
            boundary.execute()

        self.assertEqual(calls, [])

    def test_successful_execution(self):
        calls = []

        def executor():
            calls.append(True)
            return {
                "completed": True,
                "partial": False,
                "error": None,
            }

        state = self.make_state()
        result = ExecutionBoundary(state, executor).execute()

        self.assertEqual(calls, [True])
        self.assertEqual(result.state.state, SUCCEEDED)

    def test_failed_result(self):
        def executor():
            return {
                "completed": False,
                "partial": True,
                "error": "execution failed",
            }

        state = self.make_state()
        result = ExecutionBoundary(state, executor).execute()

        self.assertEqual(result.state.state, FAILED)

    def test_incomplete_mapping_is_failed(self):
        state = self.make_state()

        result = ExecutionBoundary(
            state,
            lambda: {},
        ).execute()

        self.assertEqual(result.state.state, FAILED)

    def test_completed_must_be_explicitly_true(self):
        for payload in (
            {"completed": None},
            {"error": None},
        ):
            state = self.make_state()

            result = ExecutionBoundary(
                state,
                lambda payload=payload: payload,
            ).execute()

            self.assertEqual(result.state.state, FAILED)

    def test_non_mapping_result_is_failed(self):
        state = self.make_state()

        result = ExecutionBoundary(
            state,
            lambda: "completed",
        ).execute()

        self.assertEqual(result.state.state, FAILED)

    def test_explicit_cancellation(self):
        def executor():
            return {
                "completed": False,
                "cancelled": True,
            }

        state = self.make_state()
        result = ExecutionBoundary(state, executor).execute()

        self.assertEqual(result.state.state, CANCELLED)

    def test_executor_exception_transitions_to_failed(self):
        def executor():
            raise RuntimeError("boom")

        state = self.make_state()

        with self.assertRaises(RuntimeError):
            ExecutionBoundary(state, executor).execute()

        self.assertEqual(state.state, FAILED)

    def test_executor_called_once(self):
        calls = []

        def executor():
            calls.append(len(calls))
            return {"completed": True}

        state = self.make_state()
        ExecutionBoundary(state, executor).execute()

        self.assertEqual(calls, [0])

    def test_running_is_not_accepted_as_start_state(self):
        state = self.make_state(RUNNING)

        with self.assertRaises(ExecutionNotAuthorizedError):
            ExecutionBoundary(
                state,
                lambda: {"completed": True},
            ).execute()


if __name__ == "__main__":
    unittest.main()
