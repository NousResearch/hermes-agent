"""The budget-exhaustion path must persist the summary it paid for (#78).

``_handle_max_iterations`` spends a dedicated toolless API call asking the
dying worker to summarise itself. Before this fix the result was assigned to
``final_response`` and then dropped: none of the three functions between it
and ``task_runs.summary`` had a ``summary`` parameter, so all 13 recorded
budget deaths stored NULL.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import _record_kanban_budget_exhausted, finalize_turn
from tests.agent.test_turn_finalizer_iteration_limit_exit import _LimitAgent


@pytest.fixture
def recorded(monkeypatch):
    """Capture the kwargs ``_record_task_failure`` is called with."""
    record = MagicMock(name="record_task_failure")
    monkeypatch.setattr(
        "hermes_cli.kanban_db_connect.connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    return record


def test_the_summary_reaches_the_failure_record(recorded, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    agent = _LimitAgent()

    finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )

    # _LimitAgent._handle_max_iterations returns "summary from extra call".
    assert recorded.call_args.kwargs["summary"] == "summary from extra call"


def test_structured_content_is_flattened(recorded, monkeypatch):
    """``final_response`` is not always a plain string."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    agent = _LimitAgent()
    agent._handle_max_iterations = lambda messages, n: [
        {"type": "text", "text": "wrote the migration"},
        {"type": "text", "text": "tests still red"},
    ]

    finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )

    assert recorded.call_args.kwargs["summary"] == "wrote the migration\ntests still red"


def test_an_empty_summary_warns_instead_of_storing_null_silently(recorded, caplog):
    logger = logging.getLogger("test-budget-summary")

    with caplog.at_level(logging.WARNING, logger="test-budget-summary"):
        _record_kanban_budget_exhausted("task-123", 60, 60, logger, summary="   ")

    assert recorded.call_args.kwargs["summary"] is None
    assert "no summary" in caplog.text.lower()


def test_a_present_summary_does_not_warn(recorded, caplog):
    logger = logging.getLogger("test-budget-summary")

    with caplog.at_level(logging.WARNING, logger="test-budget-summary"):
        _record_kanban_budget_exhausted("task-123", 60, 60, logger, summary="real work")

    assert recorded.call_args.kwargs["summary"] == "real work"
    assert "no summary" not in caplog.text


@pytest.mark.parametrize(
    "placeholder",
    [
        "I reached the iteration limit and couldn't generate a summary.",
        "I reached the maximum iterations (60) but couldn't summarize. Error: boom",
    ],
)
def test_placeholder_summary_is_treated_as_absent(recorded, caplog, placeholder):
    """``_handle_max_iterations`` never returns empty — it substitutes one of
    these two placeholders on every failure mode (chat_completion_helpers.py
    :3165, :3230, :3232, :3236). Persisting either would hand the next
    worker a prior-attempt block that says nothing (or leaks raw provider
    error text), so both must be normalized to "no summary" just like a
    blank string.
    """
    logger = logging.getLogger("test-budget-summary")

    with caplog.at_level(logging.WARNING, logger="test-budget-summary"):
        _record_kanban_budget_exhausted(
            "task-123", 60, 60, logger, summary=placeholder,
        )

    assert recorded.call_args.kwargs["summary"] is None
    assert "no summary" in caplog.text.lower()


def test_budget_exhausted_without_summary_call_records_none(recorded, monkeypatch):
    """The bounded fallback (``elif budget_exhausted:`` in finalize_turn) is
    reached only when no summary call was ever made (interrupted / failed /
    an anomalous exit_reason). It must not mislabel the last assistant turn
    as a summary.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    agent = _LimitAgent()

    finalize_turn(
        agent,
        final_response={"role": "assistant", "content": "unrelated last turn"},
        api_call_count=60,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted",
    )

    assert recorded.call_args.kwargs["summary"] is None
