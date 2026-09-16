"""A delegate_task child inherits HERMES_KANBAN_TASK via os.environ; on its own budget
exhaustion it must not record a terminal outcome against the dispatcher's task (#112817)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.delegation_context import delegated_child_context
from agent.turn_finalizer import _resolve_budget_fallback


def _agent():
    return SimpleNamespace(
        max_iterations=10,
        iteration_budget=SimpleNamespace(remaining=0),
        quiet_mode=True,
        _emit_status=lambda *a, **k: None,
        _safe_print=lambda *a, **k: None,
        _handle_max_iterations=lambda messages, api_call_count: "summary",
    )


@pytest.fixture(autouse=True)
def _kanban_task_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "dispatcher-task-1")


def _call(agent):
    with patch("agent.turn_finalizer._record_kanban_budget_exhausted") as recorder:
        _resolve_budget_fallback(
            agent,
            final_response=None,
            api_call_count=10,
            interrupted=False,
            failed=False,
            messages=[],
            _turn_exit_reason="unknown",
            _pending_verification_response=None,
            _pending_verification_response_previewed=False,
            logger=SimpleNamespace(warning=lambda *a, **k: None),
        )
        return recorder


def test_delegated_child_does_not_record_parent_task_outcome():
    agent = _agent()
    with delegated_child_context():
        recorder = _call(agent)
    recorder.assert_not_called()


def test_dispatcher_owned_worker_still_records_outcome():
    agent = _agent()
    recorder = _call(agent)
    recorder.assert_called_once()
    task_id, api_call_count, max_iterations, _logger = recorder.call_args[0]
    assert (task_id, api_call_count, max_iterations) == ("dispatcher-task-1", 10, 10)
