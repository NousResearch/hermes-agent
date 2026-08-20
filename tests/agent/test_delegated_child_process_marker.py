"""Process-lineage regression coverage for delegated child isolation."""

import os

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_subprocess_env,
    is_delegated_child_process_context,
)


def test_real_child_process_marker_remains_fail_closed_across_repeated_checks(
    monkeypatch,
):
    """A real child must not become privileged after its first guard check."""
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "parent-task")

    assert is_delegated_child_process_context() is True
    assert is_delegated_child_process_context() is True
    assert DELEGATED_CHILD_ENV_MARKER in os.environ

    child_env = delegated_child_subprocess_env()
    assert child_env is not None
    assert child_env[DELEGATED_CHILD_ENV_MARKER] == "1"
    assert "HERMES_KANBAN_TASK" not in child_env
