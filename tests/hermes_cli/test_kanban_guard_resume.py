"""An open PR is not a duplicate when a worker explicitly waits on parents."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()


def test_worker_dependency_wait_promoted_continues_pr_once(board):
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="implement", assignee="worker")
        claim = kb.claim_task(conn, child)
        assert claim is not None
        kb.add_comment(conn, child, "worker", "https://github.com/o/r/pull/9")
        parent = kb.create_task(conn, title="prerequisite", assignee="worker")
        kb.link_tasks(conn, parent, child, expected_child_run_id=claim.current_run_id)
        assert kb.block_task(conn, child, kind="dependency", reason="resume then complete")
        assert kb.complete_task(conn, parent, summary="prerequisite complete")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        assert kbd.check_respawn_guard(conn, child) is None
        assert kb.claim_task(conn, child)
        with kb.write_txn(conn):
            kb._append_event(conn, child, "spawned", {"pid": 42})
            conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, current_run_id=NULL WHERE id=?", (child,))
        assert kbd.check_respawn_guard(conn, child) == "active_pr"


def test_operator_requeue_ready_card(board):
    from hermes_cli import kanban as kc
    from hermes_cli.kanban_parser import build_parser
    import argparse

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="retry", assignee="worker")
        kb.add_comment(conn, task_id, "worker", "https://github.com/o/r/pull/9")
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        assert kb.requeue_task(conn, task_id, actor="operator", reason=" ") == (False, "a reason is required")
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))
    assert kc.kanban_command(parser.parse_args(["kanban", "requeue", task_id, "continue", "PR"])) == 0
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, task_id) is None
        kb.add_comment(conn, task_id, "worker", "new https://github.com/o/r/pull/10")
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        assert kb.requeue_task(conn, task_id, actor="operator", reason="again") == (True, None)
        assert kbd.check_respawn_guard(conn, task_id) is None
        kb.add_comment(conn, task_id, "worker", "status update without a PR")
        assert kbd.check_respawn_guard(conn, task_id) is None
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "spawned", {"pid": 42})
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        assert kb.block_task(conn, task_id, kind="needs_input", reason="wait")
        assert kb.requeue_task(conn, task_id, actor="operator", reason="again")[0] is False


def test_ordinary_comment_after_wait_does_not_cancel_pr_resume(board):
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="implement", assignee="worker")
        claim = kb.claim_task(conn, child)
        assert claim is not None
        kb.add_comment(conn, child, "worker", "https://github.com/o/r/pull/9")
        parent = kb.create_task(conn, title="prerequisite", assignee="worker")
        kb.link_tasks(conn, parent, child, expected_child_run_id=claim.current_run_id)
        assert kb.block_task(conn, child, kind="dependency", reason="resume")
        kb.add_comment(conn, child, "worker", "parent still pending")
        assert kb.complete_task(conn, parent, summary="prerequisite complete")
        kb.recompute_ready(conn)
        assert kbd.check_respawn_guard(conn, child) is None


def test_inline_audit_comment_does_not_shift_ready_requeue(board, monkeypatch):
    import time
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="inline audit", assignee="worker")
        with kb.write_txn(conn):
            kb._insert_comment(conn, task_id, "worker", "x" * len("https://github.com/o/r/pull/9"), now)
        kb.add_comment(conn, task_id, "worker", "https://github.com/o/r/pull/9")
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        assert kb.requeue_task(conn, task_id, actor="operator", reason="continue PR") == (True, None)
        assert kbd.check_respawn_guard(conn, task_id) is None
        kb.add_comment(conn, task_id, "worker", "https://github.com/o/r/pull/10")
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_legacy_equal_length_inline_comment_requeues(board, monkeypatch):
    import time

    now = int(time.time())
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="legacy inline audit", assignee="worker")
        pr = "https://github.com/o/r/pull/9"
        with kb.write_txn(conn):
            kb._insert_comment(conn, task_id, "worker", "x" * len(pr), now)
        kb.add_comment(conn, task_id, "worker", pr)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_remove(payload, '$.comment_id') "
                         "WHERE task_id=? AND kind='commented'", (task_id,))
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        assert kb.requeue_task(conn, task_id, actor="operator", reason="continue PR") == (True, None)
        assert kbd.check_respawn_guard(conn, task_id) is None
        kb.add_comment(conn, task_id, "worker", "https://github.com/o/r/pull/10")
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_dependency_intent_cannot_attach_to_second_promotion(board):
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="implement", assignee="worker")
        claim = kb.claim_task(conn, child)
        assert claim is not None
        kb.add_comment(conn, child, "worker", "https://github.com/o/r/pull/9")
        parent = kb.create_task(conn, title="prerequisite", assignee="worker")
        kb.link_tasks(conn, parent, child, expected_child_run_id=claim.current_run_id)
        assert kb.block_task(conn, child, kind="dependency", reason="resume")
        assert kb.complete_task(conn, parent, summary="done")
        kb.recompute_ready(conn)
        assert kbd.check_respawn_guard(conn, child) is None
        with kb.write_txn(conn):
            kb._append_event(conn, child, "spawned", {"pid": 42})
            kb._append_event(conn, child, "promoted", {"status": "ready"})
        assert kbd.check_respawn_guard(conn, child) == "active_pr"


@pytest.mark.parametrize("inline_before", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_historical_pr_wait_resumes_with_trimmed_or_inline_comment(board, monkeypatch, inline_before, padded):
    import time

    now = int(time.time())
    clock = {"now": now}
    monkeypatch.setattr(kbd.time, "time", lambda: clock["now"])
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="implement", assignee="worker")
        claim = kb.claim_task(conn, child)
        assert claim is not None
        pr = "https://github.com/o/r/pull/9"
        if inline_before:
            with kb.write_txn(conn):
                kb._insert_comment(conn, child, "worker", "x" * len(pr), now)
        kb.add_comment(conn, child, "worker", f" {pr} " if padded else pr)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_remove(payload, '$.comment_id') "
                         "WHERE task_id=? AND kind='commented'", (child,))
        parent = kb.create_task(conn, title="prerequisite", assignee="worker")
        kb.link_tasks(conn, parent, child, expected_child_run_id=claim.current_run_id)
        clock["now"] += 2
        assert kb.block_task(conn, child, kind="dependency", reason="resume")
        assert kb.complete_task(conn, parent, summary="prerequisite complete")
        kb.recompute_ready(conn)
        assert kbd.check_respawn_guard(conn, child) is None


def test_legacy_intent_equal_second_resumes_without_comment_event_mapping(board, monkeypatch):
    import time

    now = int(time.time())
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="legacy tie", assignee="worker")
        kb.add_comment(conn, task_id, "worker", " https://github.com/o/r/pull/9 ")
        assert kb.requeue_task(conn, task_id, actor="operator", reason="continue") == (True, None)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload=json_remove(payload, '$.after_comment_id') "
                "WHERE task_id=? AND kind='requeued'", (task_id,),
            )
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_pr_guard_does_not_correlate_comment_rows_with_events():
    import ast
    import inspect

    source = inspect.getsource(kbd.check_respawn_guard)
    sql_literals = [node.value.lower() for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert "after_comment_id" in source
    assert not any("kind = 'commented'" in value for value in sql_literals)
    assert not any("join task_comments" in value or "join task_events" in value
                   for value in sql_literals)


def test_new_pr_comment_after_wait_does_not_resume(board):
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="implement", assignee="worker")
        claim = kb.claim_task(conn, child)
        assert claim is not None
        kb.add_comment(conn, child, "worker", "https://github.com/o/r/pull/9")
        parent = kb.create_task(conn, title="prerequisite", assignee="worker")
        kb.link_tasks(conn, parent, child, expected_child_run_id=claim.current_run_id)
        assert kb.block_task(conn, child, kind="dependency", reason="resume")
        kb.add_comment(conn, child, "worker", "https://github.com/o/r/pull/10")
        assert kb.complete_task(conn, parent, summary="prerequisite complete")
        kb.recompute_ready(conn)
        assert kbd.check_respawn_guard(conn, child) == "active_pr"
