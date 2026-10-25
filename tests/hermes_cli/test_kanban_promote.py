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




@pytest.mark.parametrize("result", ["GATE_FAIL", "PORTABLE_GATE_FAIL"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_promote_refuses_failed_gate_and_names_result(conn, result, dry_run):
    child, (parent,) = _stuck_todo(conn)
    assert kb.edit_task(conn, parent, result=result)
    blockers = kb.unsatisfied_parents(conn, child)
    assert blockers and blockers[0][0] == parent and result in blockers[0][1]
    ok, err = kb.promote_task(conn, child, actor="tester", dry_run=dry_run)
    assert not ok and parent in err and result in err
    assert kb.get_task(conn, child).status == "todo"
    assert not any(e.kind == "promoted_manual" for e in kb.list_events(conn, child))


def test_promote_rechecks_gate_inside_write_transaction(conn, monkeypatch):
    from contextlib import contextmanager
    child, (parent,) = _stuck_todo(conn)
    original = kb.write_txn
    @contextmanager
    def change_gate_before_begin(c, **kwargs):
        # Another committed writer wins immediately before promotion's BEGIN.
        c.execute("UPDATE tasks SET result = 'PORTABLE_GATE_FAIL' WHERE id = ?", (parent,))
        with original(c, **kwargs):
            yield c
    monkeypatch.setattr(kb, "write_txn", change_gate_before_begin)
    ok, err = kb.promote_task(conn, child, actor="tester")
    assert not ok and parent in err and "PORTABLE_GATE_FAIL" in err
    assert kb.get_task(conn, child).status == "todo"


@pytest.mark.parametrize("result", ["GATE_FAIL", "PORTABLE_GATE_FAIL"])
def test_link_failed_gate_retracts_ready_child(conn, result):
    parent = kb.create_task(conn, title="gate")
    assert kb.complete_task(conn, parent, result=result)
    child = kb.create_task(conn, title="ready child")
    assert kb.link_tasks(conn, parent, child)
    assert kb.get_task(conn, child).status == "todo"
    assert not kb.claim_task(conn, child)
