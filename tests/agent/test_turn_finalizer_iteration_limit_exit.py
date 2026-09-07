"""Regression tests for iteration-limit exit normalization (#61631)."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
    )
















@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once_with(
        conn,
        "task-123",
        error=(
            "Iteration budget exhausted (60/60) — task could not complete "
            "within the allowed iterations"
        ),
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        event_payload_extra={"budget_used": 60, "budget_max": 60},
    )


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


def test_bounded_fallback_records_kanban_failure_when_interrupted(monkeypatch):
    """When budget is exhausted and the turn was interrupted,
    ``finalize_turn`` must still record a terminal kanban failure via
    the bounded fallback path (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-456")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    # Budget exhausted (60/60), interrupted, no fallback-eligible exit_reason
    result = finalize_turn(
        agent,
        final_response=None,
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
        _turn_exit_reason="interrupted_by_user",
    )

    # The bounded fallback must fire even though interrupted=True
    # makes budget_fallback_eligible=False.
    record.assert_called_once()
    args, kwargs = record.call_args
    assert args[1] == "task-456"
    assert kwargs["outcome"] == "timed_out"
    assert kwargs["release_claim"] is True
    assert kwargs["end_run"] is True
    assert kwargs["event_payload_extra"]["budget_used"] == 60
    assert kwargs["event_payload_extra"]["budget_max"] == 60


def test_bounded_fallback_records_kanban_failure_when_failed(monkeypatch):
    """When budget is exhausted and the turn failed,
    the bounded fallback must still record a terminal kanban failure (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-789")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=True,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="provider_failure",
    )

    record.assert_called_once()
    args, kwargs = record.call_args
    assert args[1] == "task-789"
    assert kwargs["outcome"] == "timed_out"


def test_bounded_fallback_does_not_fire_without_kanban_task(monkeypatch):
    """When budget is exhausted and interrupted but no kanban task is
    active, the bounded fallback must NOT fire (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
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
        _turn_exit_reason="interrupted_by_user",
    )

    record.assert_not_called()


def test_bounded_fallback_does_not_fire_when_budget_not_exhausted(monkeypatch):
    """When budget is NOT exhausted but turn is interrupted and a kanban
    task is active, the bounded fallback must NOT fire (#87096).
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-999")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._record_task_failure", record)
    agent = _LimitAgent(budget_remaining=60)

    # api_call_count=10, max_iterations=60 — budget NOT exhausted
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=10,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )

    record.assert_not_called()




# ---------------------------------------------------------------------------
# #104782: budget exhaustion with real worktree work is not a stuck task
# ---------------------------------------------------------------------------


def _git(*args, cwd):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _worktree_for(tmp_path, task_id, *, evidence):
    """Build a repo + linked worktree workspace; returns the worktree path.

    evidence: "commits" (a commit beyond the base branch), "dirty" (untracked
    changes), "clean" (untouched worktree), "not_a_repo" (plain directory).
    """
    repo = tmp_path / f"repo-{task_id}"
    repo.mkdir()
    if evidence == "not_a_repo":
        return repo
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    (repo / "base.txt").write_text("base\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "base", cwd=repo)
    wt = tmp_path / f"wt-{task_id}"
    _git("worktree", "add", "-b", f"wt/{task_id}", str(wt), cwd=repo)
    if evidence == "commits":
        (wt / "work.txt").write_text("real work\n")
        _git("add", ".", cwd=wt)
        _git("commit", "-m", "real work", cwd=wt)
    elif evidence == "dirty":
        (wt / "scratch.txt").write_text("untracked but real\n")
    return wt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _running_worktree_task(workspace_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    task_id = kb.create_task(
        conn, title="wt worker", assignee="builder",
        workspace_kind="worktree", workspace_path=str(workspace_path),
    )
    claimed = kb.claim_task(conn, task_id, claimer="builder:test")
    assert claimed is not None, "task must claim ready -> running"
    return conn, task_id


@pytest.mark.parametrize("evidence", ["commits", "dirty"])
def test_budget_exhaustion_with_worktree_work_routes_to_review(
    monkeypatch, kanban_home, tmp_path, evidence
):
    """#104782: a budget-exhausted worktree worker with real work behind it
    hands off to review WITHOUT a circuit-breaker strike — the failure counter
    stays at zero and the run ends ``review_requested``."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    wt = _worktree_for(tmp_path, "task-wt-1", evidence=evidence)
    conn, task_id = _running_worktree_task(wt)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    agent = _LimitAgent()

    _finalize(agent, final_response=None, exit_reason="unknown")

    task = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert task["status"] == "review"
    assert task["consecutive_failures"] == 0
    run = conn.execute(
        "SELECT outcome, status, ended_at FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    assert run["outcome"] == "review_requested"
    assert run["ended_at"] is not None


@pytest.mark.parametrize("evidence", ["clean", "not_a_repo"])
def test_budget_exhaustion_without_worktree_evidence_still_strikes(
    monkeypatch, kanban_home, tmp_path, evidence
):
    """#104782 invariant arm: no git evidence of work → the strike path is
    unchanged (failure counter advances, run ends ``timed_out``)."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    wt = _worktree_for(tmp_path, "task-wt-2", evidence=evidence)
    conn, task_id = _running_worktree_task(wt)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    agent = _LimitAgent()

    _finalize(agent, final_response=None, exit_reason="unknown")

    task = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert task["consecutive_failures"] == 1
    run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    assert run["outcome"] == "timed_out"
    assert run["ended_at"] is not None
