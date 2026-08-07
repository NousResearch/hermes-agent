"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _stuck_todo(conn, *, parents_done=True, n_parents=1):
    """Build the #28822 scenario: child in 'todo' whose parents may
    have closed as 'done' without the auto-promote logic firing.
    """
    parent_ids = [
        kb.create_task(conn, title=f"parent{i}", assignee="setup")
        for i in range(n_parents)
    ]
    child_id = kb.create_task(
        conn, title="child", parents=parent_ids, assignee="setup"
    )
    assert kb.get_task(conn, child_id).status == "todo"
    if parents_done:
        for pid in parent_ids:
            conn.execute(
                "UPDATE tasks SET status='done' WHERE id=?", (pid,)
            )
    return child_id, parent_ids


def test_promote_stuck_todo_succeeds(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    ok, err = kb.promote_task(conn, child, actor="tester")
    assert ok and err is None
    assert kb.get_task(conn, child).status == "ready"
    kinds = [event.kind for event in kb.list_events(conn, child)]
    assert kinds[-1] == "promoted_manual"
    assert "unblocked" not in kinds


def test_promote_dry_run_does_not_mutate_status_or_events(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    before = [event.kind for event in kb.list_events(conn, child)]

    ok, err = kb.promote_task(conn, child, actor="tester", dry_run=True)

    assert ok and err is None
    assert kb.get_task(conn, child).status == "todo"
    assert [event.kind for event in kb.list_events(conn, child)] == before


def test_promote_refuses_open_parent_unless_forced(conn):
    child, _ = _stuck_todo(conn, parents_done=False)
    before = [event.kind for event in kb.list_events(conn, child)]

    ok, err = kb.promote_task(conn, child, actor="tester")

    assert ok is False
    assert "unsatisfied parent dependencies" in (err or "")
    assert kb.get_task(conn, child).status == "todo"
    assert [event.kind for event in kb.list_events(conn, child)] == before

    ok, err = kb.promote_task(conn, child, actor="tester", force=True)
    assert ok and err is None
    assert kb.get_task(conn, child).status == "ready"


def test_promote_blocked_task_explicitly_releases_sticky_block(conn):
    tid = kb.create_task(conn, title="blocked then promoted", assignee="worker")
    assert kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.block_task(
        conn,
        tid,
        reason="review-required: operator release needed",
        expected_run_id=run_id,
    )

    ok, err = kb.promote_task(conn, tid, actor="operator")

    assert ok and err is None
    assert kb.get_task(conn, tid).status == "ready"
    kinds = [event.kind for event in kb.list_events(conn, tid)]
    assert kinds[-3:] == ["blocked", "unblocked", "promoted_manual"]

    # A later recoverable circuit-breaker block must not inherit the earlier
    # operator block after manual promotion explicitly released it.
    conn.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=1 "
        "WHERE id=?",
        (tid,),
    )
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'gave_up', NULL, ?)",
        (tid, int(time.time())),
    )
    conn.commit()
    assert kb.recompute_ready(conn) == 1
    assert kb.get_task(conn, tid).status == "ready"


def test_promote_reads_status_inside_write_transaction(conn, monkeypatch):
    tid = kb.create_task(conn, title="racing promotion", assignee="worker")
    conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
    conn.commit()

    real_write_txn = kb.write_txn
    injected = False

    @contextmanager
    def racing_write_txn(connection):
        nonlocal injected
        if not injected:
            injected = True
            connection.execute(
                "UPDATE tasks SET status='blocked' WHERE id=?", (tid,),
            )
            connection.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'blocked', NULL, ?)",
                (tid, int(time.time())),
            )
            connection.commit()
        with real_write_txn(connection):
            yield

    monkeypatch.setattr(kb, "write_txn", racing_write_txn)

    ok, err = kb.promote_task(conn, tid, actor="operator")

    assert ok and err is None
    kinds = [event.kind for event in kb.list_events(conn, tid)]
    assert kinds[-3:] == ["blocked", "unblocked", "promoted_manual"]








# ---------------------------------------------------------------------------
# CLI `_cmd_promote` — bulk via `--ids` (the issue's anti-respawn use case:
# promote all children of a closed parent in one command).
# ---------------------------------------------------------------------------


def _promote_ns(task_id, *, ids=None, reason=None, force=False,
                dry_run=False, as_json=False):
    return argparse.Namespace(
        task_id=task_id,
        reason=list(reason or []),
        ids=list(ids or []) or None,
        force=force,
        dry_run=dry_run,
        json=as_json,
    )


def test_cli_promote_bulk_ids_promotes_all(kanban_home, capsys):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        children = [
            kb.create_task(conn, title=f"c{i}", parents=[parent])
            for i in range(3)
        ]
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(children[0], ids=children[1:]))
    assert rc == 0
    out = capsys.readouterr().out
    for c in children:
        assert c in out
    with kb.connect() as conn:
        for c in children:
            assert kb.get_task(conn, c).status == "ready"


