from __future__ import annotations

import sqlite3
import argparse
import sys

import pytest
from types import SimpleNamespace

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        yield conn


def test_policy_validation_and_required_round_trip(board):
    with pytest.raises(ValueError, match="goal_judge_policy"):
        kb.create_task(board, title="bad", goal_judge_policy="unknown")
    with pytest.raises(ValueError, match="goal_mode"):
        kb.create_task(board, title="bad", goal_judge_policy="required")

    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required"
    )
    assert kb.get_task(board, tid).goal_judge_policy == "required"


def test_idempotency_rejects_policy_mismatch_but_preserves_compatible_duplicate(board):
    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required",
        idempotency_key="same-request",
    )
    assert kb.create_task(
        board, title="duplicate", goal_mode=True, goal_judge_policy="required",
        idempotency_key="same-request",
    ) == tid
    with pytest.raises(ValueError, match="goal_judge_policy"):
        kb.create_task(
            board, title="unsafe duplicate", goal_mode=True,
            goal_judge_policy="best_effort", idempotency_key="same-request",
        )


def test_required_completion_needs_central_proof(board):
    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required"
    )
    with pytest.raises(kb.GoalJudgeRequiredError):
        kb.complete_task(board, tid, summary="claimed done")
    assert kb.get_task(board, tid).status != "done"
    assert kb.complete_task(board, tid, summary="proved", goal_gate_passed=True)
    assert kb.get_task(board, tid).status == "done"
    assert kb.complete_task(board, tid, summary="duplicate") is False


def test_blocked_required_completion_still_needs_central_proof(board):
    tid = kb.create_task(
        board, title="guarded blocked", goal_mode=True,
        goal_judge_policy="required",
    )
    assert kb.block_task(board, tid, reason="waiting", kind="needs_input")
    with pytest.raises(kb.GoalJudgeRequiredError):
        kb.complete_task(board, tid, summary="manual bypass")
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "blocked"
    assert kb.complete_task(
        board, tid, summary="proved", goal_gate_passed=True,
    )
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "done"


def test_required_gate_rechecks_status_inside_write_transaction(board, monkeypatch):
    tid = kb.create_task(
        board, title="guarded race", goal_mode=True,
        goal_judge_policy="required", triage=True,
    )
    real_merge = kb._merge_completion_prose_artifacts

    def promote_before_write(conn, task_id, metadata, **kwargs):
        merged = real_merge(conn, task_id, metadata, **kwargs)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()
        return merged

    monkeypatch.setattr(kb, "_merge_completion_prose_artifacts", promote_before_write)

    with pytest.raises(kb.GoalJudgeRequiredError):
        kb.complete_task(board, tid, summary="ungated")
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "ready"


def test_best_effort_completion_is_unchanged(board):
    tid = kb.create_task(board, title="legacy", goal_mode=True)
    assert kb.get_task(board, tid).goal_judge_policy == "best_effort"
    assert kb.complete_task(board, tid, summary="done")


def test_from_row_without_policy_defaults_best_effort():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE old (id TEXT, title TEXT, body TEXT, assignee TEXT, status TEXT, "
        "priority INTEGER, created_by TEXT, created_at INTEGER, started_at INTEGER, "
        "completed_at INTEGER, workspace_kind TEXT, workspace_path TEXT, claim_lock TEXT, "
        "claim_expires INTEGER)"
    )
    conn.execute(
        "INSERT INTO old VALUES ('t1','old',NULL,NULL,'ready',0,NULL,1,NULL,NULL,"
        "'scratch',NULL,NULL,NULL)"
    )
    task = kb.Task.from_row(conn.execute("SELECT * FROM old").fetchone())
    assert task.goal_judge_policy == "best_effort"


def test_direct_status_helper_rejects_done(board):
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    tid = kb.create_task(board, title="x")
    with pytest.raises(ValueError, match="done"):
        _set_status_direct(board, tid, "done")


def test_cli_rejects_required_policy_without_goal(capsys):
    from hermes_cli.kanban import _cmd_create

    args = argparse.Namespace(
        workspace="scratch", branch=None, max_runtime=None, max_retries=None,
        goal_judge_policy="required", goal_mode=False,
    )
    assert _cmd_create(args) == 2
    assert "requires --goal" in capsys.readouterr().err


def _complete_args(task_id):
    return argparse.Namespace(
        task_ids=[task_id], summary="acceptance evidence", result=None,
        metadata=None,
    )


@pytest.mark.parametrize(
    ("verdict", "expected_rc", "expected_status"),
    [
        ("continue", 1, "ready"),
        ("done", 0, "done"),
    ],
)
def test_cli_required_judge_reject_and_pass(
    board, monkeypatch, verdict, expected_rc, expected_status,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required",
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda _purpose: (object(), "judge-model"),
    )
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda **_: (verdict, "reviewed", False, None, False),
    )
    assert _cmd_complete(_complete_args(tid)) == expected_rc
    assert kb.get_task(board, tid).status == expected_status


def test_cli_required_fail_closed_block_is_run_scoped(board, monkeypatch):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required",
    )
    kb.claim_task(board, tid)
    run_id = kb.get_task(board, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda _purpose: (None, None),
    )
    assert _cmd_complete(_complete_args(tid)) == 1
    task = kb.get_task(board, tid)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_cli_required_live_worker_without_matching_env_refuses_mutation(
    board, monkeypatch,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="guarded", goal_mode=True, goal_judge_policy="required",
    )
    kb.claim_task(board, tid)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda _purpose: (None, None),
    )
    assert _cmd_complete(_complete_args(tid)) == 1
    assert kb.get_task(board, tid).status == "running"


def test_cli_required_ready_card_can_fail_closed_without_run_env(
    board, monkeypatch,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="guarded ready", goal_mode=True,
        goal_judge_policy="required",
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda _purpose: (None, None),
    )
    assert _cmd_complete(_complete_args(tid)) == 1
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_cli_required_block_handles_task_deleted_before_status_read(
    board, monkeypatch,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="guarded disappearing", goal_mode=True,
        goal_judge_policy="required",
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda _purpose: (None, None),
    )
    real_block = kb.block_task

    def block_then_delete(conn, task_id, **kwargs):
        ok = real_block(conn, task_id, **kwargs)
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return ok

    monkeypatch.setattr("hermes_cli.kanban.kb.block_task", block_then_delete)

    assert _cmd_complete(_complete_args(tid)) == 1
    assert kb.get_task(board, tid) is None


def test_cli_plain_card_does_not_import_auxiliary_judge_dependencies(
    board, monkeypatch,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(board, title="plain card")
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
    monkeypatch.setitem(sys.modules, "hermes_cli.goals", None)
    assert _cmd_complete(_complete_args(tid)) == 0
    assert kb.get_task(board, tid).status == "done"


def test_cli_best_effort_goal_card_fails_open_when_judge_imports_are_missing(
    board, monkeypatch,
):
    from hermes_cli.kanban import _cmd_complete

    tid = kb.create_task(
        board, title="legacy goal card", goal_mode=True,
        goal_judge_policy="best_effort",
    )
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
    monkeypatch.setitem(sys.modules, "hermes_cli.goals", None)

    assert _cmd_complete(_complete_args(tid)) == 0
    assert kb.get_task(board, tid).status == "done"


@pytest.mark.parametrize(
    ("available", "judge_result", "raises", "action"),
    [
        (False, None, None, "block"),
        (True, None, RuntimeError("boom"), "block"),
        (True, ("done", "bad wire", False, None, True), None, "block"),
        (True, ("done", "bad json", True, None, False), None, "block"),
        (True, ("skipped", "no verdict", False, None, False), None, "block"),
        (True, ("mystery", "unknown", False, None, False), None, "block"),
        (True, ("continue", "more work", False, None, False), None, "reject"),
        (True, ("wait", "later", False, {"reason": "external input"}, False), None, "reject"),
        (True, ("done", "accepted", False, None, False), None, "pass"),
    ],
)
def test_required_decision_mapping(available, judge_result, raises, action):
    from hermes_cli.kanban_goal_judge import evaluate_goal_completion

    task = SimpleNamespace(
        goal_mode=True, goal_judge_policy="required", title="goal", body="criteria"
    )

    def judge(**_kwargs):
        if raises:
            raise raises
        return judge_result

    decision = evaluate_goal_completion(
        task, "evidence", judge_available=lambda: available, judge=judge
    )
    assert decision.action == action


def test_wait_preserves_directive_reason_and_all_reasons_are_redacted(monkeypatch):
    from hermes_cli import kanban_goal_judge as gate

    monkeypatch.setattr(
        gate, "redact_sensitive_text", lambda text, force=False: f"safe:{text}"
    )
    task = SimpleNamespace(
        goal_mode=True, goal_judge_policy="required", title="goal", body="",
    )
    decision = gate.evaluate_goal_completion(
        task, "x", judge_available=lambda: True,
        judge=lambda **_: (
            "wait", "fallback secret", False,
            {"reason": "preferred secret"}, False,
        ),
    )
    assert decision.reason == "safe:preferred secret"


def test_best_effort_judge_exception_and_transport_done_fail_open():
    from hermes_cli.kanban_goal_judge import evaluate_goal_completion

    task = SimpleNamespace(
        goal_mode=True, goal_judge_policy="best_effort", title="goal", body=""
    )
    assert evaluate_goal_completion(
        task, "x", judge_available=lambda: True,
        judge=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    ).action == "pass"
    assert evaluate_goal_completion(
        task, "x", judge_available=lambda: True,
        judge=lambda **_: ("done", "legacy verdict wins", True, None, True),
    ).action == "pass"
