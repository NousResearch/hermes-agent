"""Structured completion refusals (complete_task_with_reason / CLI complete).

``complete_task`` used to collapse every refusal — unfinished parents, missing
task, terminal state, stale run — into one boolean, so the CLI printed
"unknown id or terminal state" for a task that merely waited on its parents.
These tests pin the refusal codes, the blocking-parents detail and the CLI
message, and that the happy path is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban import run_slash


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    conn = kbc.connect()
    try:
        yield conn
    finally:
        conn.close()


def _child_with_open_parent(conn, parent_status="todo"):
    parent_id = kb.create_task(conn, title="parent", assignee="planner")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (parent_status, parent_id))
    child_id = kb.create_task(conn, title="child", assignee="builder", parents=[parent_id])
    return parent_id, child_id


def test_unfinished_parent_refusal_lists_parent_id_and_status(conn):
    parent_id, child_id = _child_with_open_parent(conn, parent_status="blocked")
    ok, refusal = kb.complete_task_with_reason(conn, child_id, result="source work")
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.PARENTS_NOT_SATISFIED
    assert refusal.blocking_parents == ((parent_id, "blocked"),)
    assert kb.get_task(conn, child_id).status == "todo"
    message = refusal.message(child_id)
    assert parent_id in message and "blocked" in message
    assert "complete the parents first" in message


def test_multiple_unfinished_parents_reported_in_deterministic_order(conn):
    p2 = kb.create_task(conn, title="second parent", assignee="planner")
    p1 = kb.create_task(conn, title="first parent", assignee="planner")
    child_id = kb.create_task(conn, title="child", assignee="builder", parents=[p1, p2])
    ok, refusal = kb.complete_task_with_reason(conn, child_id, result="work")
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.PARENTS_NOT_SATISFIED
    # Deterministic means parent-id order regardless of link insertion order;
    # unclaimed running tasks settle in ready.
    expected = tuple((pid, "ready") for pid in sorted((p1, p2)))
    assert refusal.blocking_parents == expected
    message = refusal.message(child_id)
    assert message.index(min(p1, p2)) < message.index(max(p1, p2))


def test_parent_reopened_between_precheck_and_txn_still_refused(conn, monkeypatch):
    parent_id, child_id = _child_with_open_parent(conn, parent_status="done")
    import hermes_cli.kanban_pr_acceptance_store as store
    real_prepare = store.prepare_acceptance

    def reopen_parent_during_gap(conn, task_id, expected_run_id, metadata):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (parent_id,))
        return real_prepare(conn, task_id, expected_run_id, metadata)

    monkeypatch.setattr(store, "prepare_acceptance", reopen_parent_during_gap)
    ok, refusal = kb.complete_task_with_reason(conn, child_id, result="work")
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.PARENTS_NOT_SATISFIED
    assert refusal.blocking_parents == ((parent_id, "todo"),)
    assert kb.get_task(conn, child_id).status != "done"


def test_unknown_task_is_distinct_from_parent_refusal(conn):
    ok, refusal = kb.complete_task_with_reason(conn, "t_0000dead", result="work")
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.UNKNOWN_TASK
    message = refusal.message("t_0000dead")
    assert "no such task" in message
    assert "parent" not in message


def test_terminal_state_refusal_is_distinct(conn):
    task_id = kb.create_task(conn, title="once", assignee="builder")
    assert kb.complete_task(conn, task_id, result="done once") is True
    ok, refusal = kb.complete_task_with_reason(conn, task_id, result="again")
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.TERMINAL_STATE
    assert "already done or archived" in refusal.message(task_id)


def test_stale_run_refusal_is_distinct(conn):
    task_id = kb.create_task(conn, title="fenced", assignee="builder")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET current_run_id = 4242 WHERE id = ?", (task_id,))
    ok, refusal = kb.complete_task_with_reason(
        conn, task_id, result="work", expected_run_id=7
    )
    assert ok is False
    assert refusal is not None
    assert refusal.code == kb.RUN_MISMATCH
    assert "stale run" in refusal.message(task_id)
    assert kb.get_task(conn, task_id).status != "done"


def test_valid_completion_unchanged(conn):
    _parent_id, child_id = _child_with_open_parent(conn, parent_status="done")
    ok, refusal = kb.complete_task_with_reason(
        conn, child_id, result="delivered", summary="handoff")
    assert ok is True
    assert refusal is None
    assert kb.get_task(conn, child_id).status == "done"
    # The boolean API still works for existing callers.
    fresh = kb.create_task(conn, title="plain", assignee="builder")
    assert kb.complete_task(conn, fresh, result="ok") is True


def test_cli_complete_reports_blocking_parents(kanban_home):
    conn = kbc.connect()
    try:
        parent_id = kb.create_task(conn, title="parent", assignee="planner")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (parent_id,))
        child_id = kb.create_task(conn, title="child", assignee="builder", parents=[parent_id])
    finally:
        conn.close()
    out = run_slash(f"complete {child_id} --result 'source work delivered'")
    assert "unsatisfied parent dependencies" in out
    assert parent_id in out and "(blocked)" in out
    assert "complete the parents first (done or archived)" in out
    assert "unknown id or terminal state" not in out
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, child_id).status != "done"
    finally:
        conn.close()


def test_cli_complete_unknown_id_stays_actionable(kanban_home):
    out = run_slash("complete t_0000dead --result 'work'")
    assert "no such task" in out
