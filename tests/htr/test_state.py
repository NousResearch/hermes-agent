import pytest

from htr.state import (
    ATTEMPT_CANCELLED,
    ATTEMPT_COMPLETED,
    ATTEMPT_CREATED,
    ATTEMPT_FAILED,
    ATTEMPT_HEAL_REQUIRED,
    ATTEMPT_RESULT_SUBMITTED,
    ATTEMPT_RUNNING,
    ATTEMPT_VERIFICATION_FAILED,
    ATTEMPT_VERIFICATION_PASSED,
    InvalidTransition,
    TASK_BLOCKED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_FAILED,
    TASK_RUNNING,
    assert_valid_attempt_transition,
    assert_valid_task_transition,
    is_terminal_attempt_status,
    is_terminal_task_status,
    is_valid_attempt_transition,
    is_valid_task_transition,
)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (TASK_CREATED, TASK_RUNNING),
        (TASK_CREATED, TASK_CANCELLED),
        (TASK_RUNNING, TASK_BLOCKED),
        (TASK_RUNNING, TASK_COMPLETED),
        (TASK_RUNNING, TASK_FAILED),
        (TASK_RUNNING, TASK_CANCELLED),
        (TASK_BLOCKED, TASK_RUNNING),
        (TASK_BLOCKED, TASK_CANCELLED),
    ],
)
def test_legal_task_transitions_pass(from_status, to_status):
    assert is_valid_task_transition(from_status, to_status)
    assert_valid_task_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (TASK_CREATED, TASK_COMPLETED),
        (TASK_RUNNING, TASK_CREATED),
        (TASK_COMPLETED, TASK_RUNNING),
        (TASK_FAILED, TASK_RUNNING),
        (TASK_CANCELLED, TASK_RUNNING),
    ],
)
def test_illegal_task_transitions_rejected(from_status, to_status):
    assert not is_valid_task_transition(from_status, to_status)
    with pytest.raises(InvalidTransition):
        assert_valid_task_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TASK_COMPLETED, True),
        (TASK_FAILED, True),
        (TASK_CANCELLED, True),
        (TASK_RUNNING, False),
        (TASK_BLOCKED, False),
        (TASK_CREATED, False),
    ],
)
def test_terminal_task_helper(status, expected):
    assert is_terminal_task_status(status) is expected


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (ATTEMPT_CREATED, ATTEMPT_RUNNING),
        (ATTEMPT_CREATED, ATTEMPT_CANCELLED),
        (ATTEMPT_RUNNING, ATTEMPT_RESULT_SUBMITTED),
        (ATTEMPT_RUNNING, ATTEMPT_FAILED),
        (ATTEMPT_RUNNING, ATTEMPT_CANCELLED),
        (ATTEMPT_RESULT_SUBMITTED, ATTEMPT_VERIFICATION_PASSED),
        (ATTEMPT_RESULT_SUBMITTED, ATTEMPT_VERIFICATION_FAILED),
        (ATTEMPT_VERIFICATION_PASSED, ATTEMPT_COMPLETED),
        (ATTEMPT_VERIFICATION_FAILED, ATTEMPT_HEAL_REQUIRED),
        (ATTEMPT_VERIFICATION_FAILED, ATTEMPT_FAILED),
        (ATTEMPT_HEAL_REQUIRED, ATTEMPT_FAILED),
    ],
)
def test_legal_attempt_transitions_pass(from_status, to_status):
    assert is_valid_attempt_transition(from_status, to_status)
    assert_valid_attempt_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (ATTEMPT_CREATED, ATTEMPT_COMPLETED),
        (ATTEMPT_RUNNING, ATTEMPT_COMPLETED),
        (ATTEMPT_RESULT_SUBMITTED, ATTEMPT_COMPLETED),
        (ATTEMPT_HEAL_REQUIRED, ATTEMPT_COMPLETED),
        (ATTEMPT_COMPLETED, ATTEMPT_RUNNING),
        (ATTEMPT_FAILED, ATTEMPT_RUNNING),
    ],
)
def test_illegal_attempt_transitions_rejected(from_status, to_status):
    assert not is_valid_attempt_transition(from_status, to_status)
    with pytest.raises(InvalidTransition):
        assert_valid_attempt_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ATTEMPT_COMPLETED, True),
        (ATTEMPT_FAILED, True),
        (ATTEMPT_CANCELLED, True),
        (ATTEMPT_HEAL_REQUIRED, False),
        (ATTEMPT_RUNNING, False),
        (ATTEMPT_RESULT_SUBMITTED, False),
    ],
)
def test_terminal_attempt_helper(status, expected):
    assert is_terminal_attempt_status(status) is expected
