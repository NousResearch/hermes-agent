"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
import json
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


def test_force_promote_runs_child_of_blocked_parent_through_lifecycle(conn):
    """An explicit override lets a support card unblock its blocked parent."""
    parent = kb.create_task(conn, title="blocked parent", assignee="setup")
    assert kb.block_task(conn, parent, reason="needs support")
    child = kb.create_task(conn, title="support child", parents=[parent], assignee="setup")

    ok, err = kb.promote_task(conn, child, actor="tester", force=True)
    assert ok and err is None
    assert kb.claim_task(conn, child, claimer="tester") is not None

    assert kb.block_task(conn, child, reason="pause support")
    assert kb.unblock_task(conn, child)
    assert kb.get_task(conn, child).status == "ready"

    assert kb.request_review(conn, child, summary="support is ready")
    assert kb.reopen_review_task(conn, child)
    assert kb.get_task(conn, child).status == "ready"
    assert kb.complete_task(conn, child)


def test_force_promote_remains_gated_by_ordinary_undone_parent(conn):
    # A force override is only for a parent that is itself blocked.
    child, (parent,) = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(conn, child, actor="tester", reason="recovery", force=True)
    assert not ok
    assert parent in err and f"unlink <parent_id> {child}" in err
    assert kb.get_task(conn, child).status == "todo"
    assert kb.claim_task(conn, child) is None  # still gated; nothing pretended


def test_cli_promote_accepts_force_flag(kanban_home):
    from hermes_cli import kanban_parser
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kanban_parser.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", "promote", "t_x", "--force"])
    assert args.force is True


def test_recompute_ready_honors_forced_blocked_parent_override(conn):
    parent = kb.create_task(conn, title="blocked parent")
    assert kb.block_task(conn, parent, reason="needs support")
    child = kb.create_task(conn, title="support child", parents=[parent])
    assert kb.promote_task(conn, child, actor="tester", force=True)[0]
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))

    assert kb.recompute_ready(conn) == 1
    assert kb.get_task(conn, child).status == "ready"


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

