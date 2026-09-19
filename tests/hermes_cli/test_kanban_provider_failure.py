"""Provider admission evidence and independent retry budgets survive worker exit."""

import json
import os

import pytest

import cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "admission")
    monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
    monkeypatch.setattr(dispatch, "_worker_alive", lambda *args: False)
    # Recovery boundaries must work even when all native timestamps tie.
    monkeypatch.setattr(kb.time, "time", lambda: 1700000000)
    kbc.init_db()
    with kbc.connect() as conn:
        yield conn


def launch(conn, task, monkeypatch):
    lock = kb._host_prefix() + "failure-test"
    assert kb.claim_task(conn, task, claimer=lock)
    dispatch._set_worker_pid(conn, task, os.getpid())
    run = kb.get_task(conn, task).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
    return run


@pytest.mark.parametrize(
    "reason,status,transient",
    [
        ("billing", 402, False),
        ("billing", 403, False),
        ("auth", 401, False),
        ("auth_permanent", 403, False),
        ("rate_limit", 429, True),
        ("rate_limit", 402, True),
        ("overloaded", 503, True),
    ],
)
def test_provider_failure_survives_clean_exit_without_spending_model_retries(
    board, monkeypatch, reason, status, transient
):
    task = kb.create_task(
        board, title="provider rejected", assignee="worker", max_retries=2
    )
    child = kb.create_task(board, title="dependent", assignee="worker")
    kb.link_tasks(board, task, child)
    run = launch(board, task, monkeypatch)
    result = {
        "failed": True,
        "completed": False,
        "failure_reason": reason,
        "failure_status_code": status,
        "error": "SECRET raw body",
        "messages": [],
    }
    if reason == "billing":
        from agent.error_classifier import classify_api_error
        from agent.conversation_loop import _billing_failure_result

        error = RuntimeError(
            "Budget limit exceeded (monthly limit). Contact your org admin."
        )
        error.status_code = status
        classified = classify_api_error(error)
        result = _billing_failure_result(
            classified=classified,
            summary="SECRET raw body",
            messages=[],
            api_call_count=1,
            provider="custom",
            base_url="",
            model="unchanged",
            guidance="",
        )
        assert result["failure_reason"] == reason
    # Neither a delegated child nor stale run/claim may report against this owner.
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        cli._single_query_exit_code(result)
    with monkeypatch.context() as stale:
        stale.setenv("HERMES_KANBAN_RUN_ID", str(run + 1))
        cli._single_query_exit_code(result)
    with monkeypatch.context() as stale:
        stale.setenv("HERMES_KANBAN_CLAIM_LOCK", "other-owner")
        cli._single_query_exit_code(result)
    assert not any(e.kind == "provider_failure" for e in kb.list_events(board, task))
    # Report during Popen's launch window before the dispatcher records its PID.
    board.execute("UPDATE tasks SET worker_pid=NULL WHERE id=?", (task,))
    board.commit()
    cli._single_query_exit_code(result)
    dispatch._set_worker_pid(board, task, os.getpid())
    cli._single_query_exit_code(result)  # An uncertain repeated report is idempotent.
    dispatch._record_worker_exit(
        os.getpid(), 0
    )  # Even lost/incorrect exit status cannot erase evidence.
    dispatch.detect_crashed_workers(board)
    expected = "ready" if transient else "blocked"
    assert kb.get_task(board, task).status == expected
    assert kb.get_task(board, task).consecutive_failures == 0
    if not transient:
        assert task in dispatch.detect_crashed_workers._last_auto_blocked
    events = kb.list_events(board, task)
    assert len([e for e in events if e.kind == "provider_failure"]) == 1
    preserved = dict(
        board.execute("SELECT * FROM task_runs WHERE id=?", (run,)).fetchone()
    )
    evidence = json.loads(preserved["metadata"])["provider_failure"]
    assert evidence["reason"] == reason and evidence["http_status"] == status
    assert "SECRET" not in json.dumps(preserved)
    assert not any(e.kind == "protocol_violation" for e in events)
    kb.recompute_ready(board)
    assert kb.get_task(board, child).status != "ready"
    if transient:
        launch(board, task, monkeypatch)
        cli._single_query_exit_code(result)
        dispatch._record_worker_exit(os.getpid(), 0)
        dispatch.detect_crashed_workers(board)
    blocked = [e for e in kb.list_events(board, task) if e.kind == "blocked"]
    assert blocked[-1].payload["retry_status"] == "blocked"
    terminal = board.execute(
        "SELECT metadata FROM task_runs WHERE id=?", (blocked[-1].run_id,)
    ).fetchone()
    assert json.loads(terminal["metadata"])["retry_status"] == "blocked"
    for _ in range(3):
        kb.recompute_ready(board)
        assert kb.get_task(board, task).status == "blocked"
    assert (
        dict(board.execute("SELECT * FROM task_runs WHERE id=?", (run,)).fetchone())
        == preserved
    )
    assert kb.get_task(board, task).consecutive_failures == 0
    if transient:
        terminal_history = [
            dict(row) for row in board.execute("SELECT * FROM task_runs ORDER BY id")
        ]
        assert kb.unblock_task(board, task)
        # A late comment on a historical run is not a new execution attempt.
        with kb.write_txn(board):
            kb._append_event(
                board,
                task,
                "comment",
                {"note": "Prior-run audit"},
                run_id=terminal_history[-1]["id"],
            )
        for expected in ("ready", "blocked"):
            launch(board, task, monkeypatch)
            cli._single_query_exit_code(result)
            dispatch._record_worker_exit(os.getpid(), 0)
            dispatch.detect_crashed_workers(board)
            assert kb.get_task(board, task).status == expected
        assert [
            dict(row) for row in board.execute("SELECT * FROM task_runs ORDER BY id")
        ][: len(terminal_history)] == terminal_history
        assert [e for e in kb.list_events(board, task) if e.kind == "blocked"][
            -1
        ].payload["retry_status"] == "blocked"


@pytest.mark.parametrize("limit", [None, 1, 3])
def test_protocol_breaker_cannot_be_repromoted_by_unified_counter(
    board, monkeypatch, limit
):
    task = kb.create_task(
        board, title="missing terminal call", assignee="worker", max_retries=limit
    )
    bound = dispatch._PROTOCOL_VIOLATION_FAILURE_LIMIT if limit is None else limit
    for attempt in range(bound):
        launch(board, task, monkeypatch)
        dispatch._record_worker_exit(os.getpid(), 0)
        dispatch.detect_crashed_workers(board)
        if attempt < bound - 1:
            assert kb.get_task(board, task).status == "ready"
            assert kb.get_task(board, task).consecutive_failures == 0
    assert kb.get_task(board, task).status == "blocked"
    before = [
        dict(r)
        for r in board.execute("SELECT * FROM task_runs WHERE task_id=?", (task,))
    ]
    for _ in range(3):
        kb.recompute_ready(board)
        assert kb.get_task(board, task).status == "blocked"
    assert [
        dict(r)
        for r in board.execute("SELECT * FROM task_runs WHERE task_id=?", (task,))
    ] == before

    assert kb.unblock_task(board, task)
    for attempt in range(bound):
        launch(board, task, monkeypatch)
        dispatch._record_worker_exit(os.getpid(), 0)
        dispatch.detect_crashed_workers(board)
        expected = "blocked" if attempt == bound - 1 else "ready"
        assert kb.get_task(board, task).status == expected
        kb.recompute_ready(board)
        assert kb.get_task(board, task).status == expected
    assert [
        dict(r)
        for r in board.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task,)
        )
    ][: len(before)] == before
