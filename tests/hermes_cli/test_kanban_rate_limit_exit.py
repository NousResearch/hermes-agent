import pytest

from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
from hermes_cli.kanban_worker import worker_exit_code


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_quota_failure_uses_rate_limit_exit_sentinel(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")

    assert worker_exit_code({"partial": True, "failure_reason": reason}) == (
        KANBAN_RATE_LIMIT_EXIT_CODE
    )


def test_generic_failure_keeps_failure_exit_code(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")

    assert worker_exit_code({"failed": True, "failure_reason": "tool_error"}) == 1


def test_rate_limit_sentinel_is_scoped_to_kanban_workers(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert worker_exit_code({"partial": True, "failure_reason": "rate_limit"}) == 0