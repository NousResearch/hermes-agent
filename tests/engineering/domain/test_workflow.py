from __future__ import annotations

import ast
import inspect
from datetime import timedelta
from uuid import UUID

import pytest

import engineering.domain.workflow as workflow_module
from engineering.domain.workflow import (
    ALLOWED_STATE_TRANSITIONS,
    TERMINAL_STATES,
    AttemptLimitExceeded,
    InvalidWorkflowTransition,
    WorkflowRun,
    WorkflowState,
)


EXPECTED_TRANSITIONS = {
    WorkflowState.CREATED: {
        WorkflowState.UNDERSTANDING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.UNDERSTANDING: {
        WorkflowState.EXPLORING,
        WorkflowState.PLANNING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.EXPLORING: {
        WorkflowState.UNDERSTANDING,
        WorkflowState.PLANNING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.PLANNING: {
        WorkflowState.EXPLORING,
        WorkflowState.IMPLEMENTING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.IMPLEMENTING: {
        WorkflowState.VERIFYING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.VERIFYING: {
        WorkflowState.IMPLEMENTING,
        WorkflowState.REVIEWING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
    },
    WorkflowState.REVIEWING: {
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
        WorkflowState.COMPLETED,
    },
    WorkflowState.BLOCKED: set(),
    WorkflowState.FAILED: set(),
    WorkflowState.COMPLETED: set(),
}


def test_declares_the_explicit_workflow_contract() -> None:
    assert [state.value for state in WorkflowState] == [
        "CREATED",
        "UNDERSTANDING",
        "EXPLORING",
        "PLANNING",
        "IMPLEMENTING",
        "VERIFYING",
        "REVIEWING",
        "BLOCKED",
        "FAILED",
        "COMPLETED",
    ]
    assert TERMINAL_STATES == {
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
        WorkflowState.COMPLETED,
    }
    assert {
        state: set(targets)
        for state, targets in ALLOWED_STATE_TRANSITIONS.items()
    } == EXPECTED_TRANSITIONS


def test_standard_lifecycle_reaches_engineering_completion() -> None:
    run = WorkflowRun()

    for state in (
        WorkflowState.UNDERSTANDING,
        WorkflowState.EXPLORING,
        WorkflowState.PLANNING,
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
    ):
        run.transition_to(state)
        assert run.engineering_completed is False

    run.transition_to(WorkflowState.COMPLETED)

    assert run.state is WorkflowState.COMPLETED
    assert run.is_terminal is True
    assert run.engineering_completed is True


def test_invalid_transition_fails_without_mutating_the_run() -> None:
    run = WorkflowRun()
    original_updated_at = run.updated_at

    with pytest.raises(
        InvalidWorkflowTransition,
        match="Invalid engineering workflow transition: CREATED -> COMPLETED",
    ):
        run.transition_to(WorkflowState.COMPLETED)

    assert run.state is WorkflowState.CREATED
    assert run.updated_at == original_updated_at
    assert run.engineering_completed is False


@pytest.mark.parametrize(
    ("path", "rejected_target"),
    [
        ((WorkflowState.BLOCKED,), WorkflowState.FAILED),
        ((WorkflowState.FAILED,), WorkflowState.BLOCKED),
        (
            (
                WorkflowState.UNDERSTANDING,
                WorkflowState.PLANNING,
                WorkflowState.IMPLEMENTING,
                WorkflowState.VERIFYING,
                WorkflowState.REVIEWING,
                WorkflowState.COMPLETED,
            ),
            WorkflowState.FAILED,
        ),
    ],
)
def test_terminal_states_reject_all_outgoing_transitions(
    path: tuple[WorkflowState, ...], rejected_target: WorkflowState
) -> None:
    run = WorkflowRun()
    for state in path:
        run.transition_to(state)

    assert run.state in TERMINAL_STATES
    assert ALLOWED_STATE_TRANSITIONS[run.state] == frozenset()
    with pytest.raises(InvalidWorkflowTransition):
        run.transition_to(rejected_target)


def test_workflow_ids_and_timestamps_are_generated_by_code() -> None:
    first = WorkflowRun()
    second = WorkflowRun()

    assert UUID(first.workflow_run_id).version == 4
    assert UUID(second.workflow_run_id).version == 4
    assert first.workflow_run_id != second.workflow_run_id
    assert first.created_at == first.updated_at
    assert first.created_at.tzinfo is not None
    assert first.created_at.utcoffset() is not None


def test_transition_updates_timestamp_without_allowing_time_to_move_backwards() -> None:
    run = WorkflowRun()
    next_timestamp = run.updated_at + timedelta(seconds=1)

    run.transition_to(WorkflowState.UNDERSTANDING, at=next_timestamp)

    assert run.updated_at == next_timestamp
    with pytest.raises(ValueError, match="timestamps cannot move backwards"):
        run.transition_to(
            WorkflowState.EXPLORING,
            at=next_timestamp - timedelta(microseconds=1),
        )
    assert run.state is WorkflowState.UNDERSTANDING
    assert run.updated_at == next_timestamp


def test_attempt_exhaustion_fails_and_never_completes_the_workflow() -> None:
    run = WorkflowRun(max_attempts=2)
    run.transition_to(WorkflowState.UNDERSTANDING)

    assert run.begin_next_attempt() == 2
    with pytest.raises(AttemptLimitExceeded, match="max_attempts=2"):
        run.begin_next_attempt()

    assert run.attempt == 2
    assert run.state is WorkflowState.FAILED
    assert run.is_terminal is True
    assert run.engineering_completed is False


@pytest.mark.parametrize("max_attempts", [0, -1, True, 1.5, "3"])
def test_max_attempts_must_be_a_positive_integer(max_attempts: object) -> None:
    with pytest.raises(
        ValueError, match="max_attempts must be an integer greater than zero"
    ):
        WorkflowRun(max_attempts=max_attempts)  # type: ignore[arg-type]


def test_domain_model_has_no_hermes_runtime_imports() -> None:
    tree = ast.parse(inspect.getsource(workflow_module))
    forbidden_roots = {"run_agent", "agent", "hermes_cli", "tools"}
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots.isdisjoint(forbidden_roots)
