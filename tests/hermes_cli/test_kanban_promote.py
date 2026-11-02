"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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
    with kbc.connect() as c:
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


def test_promote_refuses_undone_parent_and_names_the_real_remedy(conn):
    # #106195: promotion must never report a 'ready' that the first claim reverts.
    child, (parent,) = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(conn, child, actor="tester", reason="recovery")
    assert not ok
    assert err and parent in err
    assert kb.get_task(conn, child).status == "todo"
    assert kb.claim_task(conn, child) is None  # still gated; nothing pretended



def test_promote_accepts_triage_as_the_manual_accept_as_is_exit(conn):
    # Triage's only other exits (specify/decompose) route the card through the
    # auxiliary LLM; an operator who reviewed the card needs to accept it as-is.
    tid = kb.create_task(conn, title="spec is fine as written", triage=True, assignee="setup")
    assert kb.get_task(conn, tid).status == "triage"
    ok, err = kb.promote_task(conn, tid, actor="tester", reason="accepted as-is")
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "ready"
    kinds = [row["kind"] for row in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert kinds[-1] == "promoted_manual"
    # The LLM specifier can no longer move the accepted card back to todo.
    assert kb.specify_triage_task(conn, tid, title="rewritten") is False
    assert kb.get_task(conn, tid).status == "ready"


def test_promote_triage_dry_run_validates_without_mutating(conn):
    tid = kb.create_task(conn, title="t", triage=True, assignee="setup")
    ok, err = kb.promote_task(conn, tid, actor="tester", dry_run=True)
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "triage"


def test_promote_triage_refused_with_unfinished_parent(conn):
    parent = kb.create_task(conn, title="parent", assignee="setup")
    tid = kb.create_task(conn, title="child", triage=True, parents=[parent], assignee="setup")
    ok, err = kb.promote_task(conn, tid, actor="tester")
    assert not ok and parent in err
    assert kb.get_task(conn, tid).status == "triage"



def test_promoted_block_loop_card_returns_to_triage_on_the_same_block(conn):
    # A card the unblock-loop breaker parked in triage keeps its recurrence count
    # through promotion, so accepting it cannot restart an unbounded loop.
    tid = kb.create_task(conn, title="loops", assignee="worker")
    for _ in range(2):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        kb.block_task(conn, tid, reason="x", kind="capability")
        if kb.get_task(conn, tid).status == "blocked":
            kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"

    ok, err = kb.promote_task(conn, tid, actor="tester", reason="one more try")
    assert ok and err is None
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    kb.block_task(conn, tid, reason="x", kind="capability")
    assert kb.get_task(conn, tid).status == "triage"


@pytest.mark.parametrize("status", ["scheduled", "ready", "running", "review", "done", "archived"])
def test_promote_still_refuses_other_statuses(conn, status):
    tid = kb.create_task(conn, title="t", assignee="setup")
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    ok, err = kb.promote_task(conn, tid, actor="tester")
    assert not ok and "'triage', 'todo' or 'blocked'" in err
    assert kb.get_task(conn, tid).status == status




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
    with kbc.connect() as conn:
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
    with kbc.connect() as conn:
        for c in children:
            assert kb.get_task(conn, c).status == "ready"


