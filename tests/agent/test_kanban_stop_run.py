"""The stop guard may continue only the run that still owns the work."""

import json
import os

import pytest

from agent.kanban_stop import build_kanban_stop_nudge
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


@pytest.fixture
def board(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("HERMES_KANBAN_"):
            monkeypatch.delenv(key)
    path = tmp_path / "board #1.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    with connect_closing(path) as conn:
        tid = kb.create_task(conn, title="Synthetic handoff", assignee="builder")
        task = kb.claim_task(conn, tid)
        assert task is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
        yield conn, tid, task.current_run_id, monkeypatch


@pytest.mark.parametrize("transition", ["review", "changes", "complete", "block"])
@pytest.mark.parametrize("history", [None, [], "failed", "attempted"])
def test_handoff_never_rearms_or_mutates_successor(board, transition, history):
    conn, tid, run_id, env = board
    if transition == "changes":
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_id)
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        run_id = review.current_run_id
        env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    # Attempts / failures are not handoffs: the genuine owner still needs a nudge.
    messages = history
    if history in ("failed", "attempted"):
        messages = [{"role": "assistant", "tool_calls": [{
            "id": "call-1", "function": {"name": "kanban_complete", "arguments": "{}"},
        }]}]
        if history == "failed":
            messages.append({"role": "tool", "name": "kanban_complete",
                             "tool_call_id": "call-1", "content": '{"error":"rejected"}'})
    assert build_kanban_stop_nudge(messages=messages) is not None
    if transition == "review":
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer",
                                 expected_run_id=run_id)
        successor = kb.claim_review_task(conn, tid)
    elif transition == "changes":
        assert kb.request_changes(conn, tid, reason="revise", expected_run_id=run_id)[0]
        successor = kb.claim_task(conn, tid)
    elif transition == "complete":
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
        successor = None
    else:
        assert kb.block_task(conn, tid, reason="need input", expected_run_id=run_id)
        assert kb.unblock_task(conn, tid)
        successor = kb.claim_task(conn, tid)
    snapshot = list(conn.iterdump())
    assert build_kanban_stop_nudge(messages=messages) is None
    assert list(conn.iterdump()) == snapshot
    if successor:
        assert not kb.complete_task(conn, tid, summary="stale", expected_run_id=run_id)
        assert not kb.block_task(conn, tid, reason="stale", expected_run_id=run_id)
        current = kb.get_task(conn, tid)
        assert current is not None
        assert current.current_run_id == successor.current_run_id
        assert current.status == "running"
        env.setenv("HERMES_KANBAN_RUN_ID", str(successor.current_run_id))
        # A prior run's successful history must not let its successor escape.
        old_success = [{"role": "tool", "name": "kanban_complete", "content": json.dumps({
            "ok": True, "task_id": tid, "run_id": run_id,
        })}]
        assert build_kanban_stop_nudge(messages=old_success) is not None


@pytest.mark.parametrize("case", [
    "wrong-run", "wrong-task", "missing-run", "missing-task", "missing-db", "bad-db", "bad-run",
    "board-isolation", "db-pin", "no-run", "delegate", "cron", "budget", "disabled",
])
def test_identity_and_unavailable_state_do_not_authorize_stale_work(board, tmp_path, case):
    conn, tid, run_id, env = board
    expected_nudge = False
    if case == "wrong-run":
        env.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 100))
    elif case == "wrong-task":
        other = kb.create_task(conn, title="Other", assignee="builder")
        kb.claim_task(conn, other)
        env.setenv("HERMES_KANBAN_TASK", other)
    elif case == "missing-run":
        conn.execute("DELETE FROM task_runs WHERE id=?", (run_id,))
    elif case == "missing-task":
        env.setenv("HERMES_KANBAN_TASK", "t_missing")
    elif case == "missing-db":
        env.setenv("HERMES_KANBAN_DB", str(tmp_path / "absent.db"))
    elif case == "bad-db":
        bad = tmp_path / "broken.db"
        bad.write_text("not sqlite", encoding="utf-8")
        env.setenv("HERMES_KANBAN_DB", str(bad))
    elif case == "bad-run":
        env.setenv("HERMES_KANBAN_RUN_ID", "invalid")
    elif case in ("board-isolation", "db-pin"):
        # Identical task/run ids on distinct boards must never share a cached verdict.
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
        assert build_kanban_stop_nudge(messages=[]) is None
        env.delenv("HERMES_KANBAN_DB")
        other_path = kb.kanban_db_path(board="other")
        with connect_closing(other_path) as other_conn:
            conn.backup(other_conn)
            other_conn.execute("UPDATE tasks SET status='running', current_run_id=? WHERE id=?",
                               (run_id, tid))
            other_conn.execute("UPDATE task_runs SET outcome=NULL, ended_at=NULL WHERE id=?", (run_id,))
        env.setenv("HERMES_KANBAN_BOARD", "other")
        if case == "db-pin":
            env.setenv("HERMES_KANBAN_DB", str(conn.execute("PRAGMA database_list").fetchone()[2]))
        else:
            expected_nudge = True
    elif case == "no-run":
        env.delenv("HERMES_KANBAN_RUN_ID")
        expected_nudge = True
    elif case in ("delegate", "cron"):
        from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
        context = delegated_child_context if case == "delegate" else non_dispatcher_owned_context
        with context():
            assert build_kanban_stop_nudge(messages=[]) is None
        assert build_kanban_stop_nudge(messages=[]) is not None
        return
    elif case == "budget":
        assert build_kanban_stop_nudge(messages=[], attempts=2) is None
        return
    else:
        env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert (build_kanban_stop_nudge(messages=[]) is not None) is expected_nudge
    assert not (tmp_path / "absent.db").exists()
