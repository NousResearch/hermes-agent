"""Regression test for partial-findings salvage on budget exhaustion.

When a Kanban worker exhausts its iteration budget, ``finalize_turn``
already asks the model for a toolless summary via
``_handle_max_iterations``.  Previously that summary was discarded:
``_record_task_failure`` was called without it, so the run row closed
with ``summary=None`` and the retry worker started from scratch.

This test verifies that:
  1. The model's last text (``final_response``) is forwarded as
     ``partial_summary`` to ``_record_task_failure`` and lands in the
     run row's ``summary`` field.
  2. A ``[partial_unverified]`` comment is written to the task so the
     retry worker sees it in ``build_worker_context``'s comment thread.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.turn_finalizer import finalize_turn
from hermes_cli import kanban_db as kb


class _StubBudget:
    used = 40
    max_total = 40
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, max_iterations=40, partial_summary="PARTIAL FINDINGS"):
        self.max_iterations = max_iterations
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._partial_summary = partial_summary
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        # Simulate the model producing a partial summary.
        return self._partial_summary

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run_budget_exhausted(agent):
    """Run finalize_turn in the budget-exhausted path."""
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=None,          # triggers the budget-exhausted branch
        api_call_count=agent.max_iterations,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="budget_exhausted",
    )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_budget_exhaustion_salvages_partial_summary_to_run(kanban_home, monkeypatch):
    """The model's last text must land in the run row's summary field."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="find the bug", assignee="enrico")
        kb.claim_task(conn, t)

    monkeypatch.setenv("HERMES_KANBAN_TASK", t)
    monkeypatch.setenv("HERMES_PROFILE", "enrico")

    agent = _StubAgent(partial_summary="Root cause: None.split() in handle_x()")
    result = _run_budget_exhausted(agent)

    # The model's summary is returned as final_response.
    assert "Root cause" in result["final_response"]

    with kb.connect() as conn:
        runs = kb.list_runs(conn, t)
        assert len(runs) == 1
        run = runs[0]
        assert run.outcome == "timed_out"
        assert run.summary is not None
        assert "Root cause" in run.summary


def test_budget_exhaustion_writes_partial_unverified_comment(kanban_home, monkeypatch):
    """A [partial_unverified] comment must be written so the retry worker
    sees it in build_worker_context's comment thread."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="find the bug", assignee="enrico")
        kb.claim_task(conn, t)

    monkeypatch.setenv("HERMES_KANBAN_TASK", t)
    monkeypatch.setenv("HERMES_PROFILE", "enrico")

    agent = _StubAgent(partial_summary="Found it: line 42 is the culprit.")
    _run_budget_exhausted(agent)

    with kb.connect() as conn:
        comments = kb.list_comments(conn, t)
        assert len(comments) == 1
        assert "partial_unverified" in comments[0].body
        assert "Found it" in comments[0].body
        assert comments[0].author == "enrico"

        # build_worker_context surfaces the comment.
        ctx = kb.build_worker_context(conn, t)
        assert "Found it" in ctx
        assert "timed_out" in ctx


def test_budget_exhaustion_no_summary_no_comment(kanban_home, monkeypatch):
    """When the model produces no summary text, no comment is written
    and the run summary stays None (don't fabricate)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="find the bug", assignee="enrico")
        kb.claim_task(conn, t)

    monkeypatch.setenv("HERMES_KANBAN_TASK", t)
    monkeypatch.setenv("HERMES_PROFILE", "enrico")

    agent = _StubAgent(partial_summary="")
    _run_budget_exhausted(agent)

    with kb.connect() as conn:
        runs = kb.list_runs(conn, t)
        assert len(runs) == 1
        assert runs[0].summary is None
        comments = kb.list_comments(conn, t)
        assert len(comments) == 0


def test_budget_exhaustion_retry_worker_sees_partial_findings(kanban_home, monkeypatch):
    """End-to-end: the retry worker's build_worker_context must contain
    the partial findings from the timed-out run AND the comment."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="find the bug", assignee="enrico")
        kb.claim_task(conn, t)

    monkeypatch.setenv("HERMES_KANBAN_TASK", t)
    monkeypatch.setenv("HERMES_PROFILE", "enrico")

    agent = _StubAgent(
        partial_summary=(
            "Root cause identified: the retry handler at api.py:142 "
            "doesn't guard against None returns from _fetch(). "
            "Fix: add `if result is None: return []` before .split()."
        )
    )
    _run_budget_exhausted(agent)

    # Simulate the retry: a fresh worker reads its context.
    with kb.connect() as conn:
        ctx = kb.build_worker_context(conn, t)
        # The partial summary appears in the prior-attempts section.
        assert "Root cause identified" in ctx
        # The comment appears in the comment thread.
        assert "partial_unverified" in ctx
        # The outcome is visible so the worker knows this was a timeout.
        assert "timed_out" in ctx