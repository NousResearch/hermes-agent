"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
import logging
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


# ---------------------------------------------------------------------------
# `recompute_ready` and a parent reference that exists only in prose (#126699)
# ---------------------------------------------------------------------------


def test_recompute_ready_names_prose_only_parent_ids_without_gating(conn, caplog):
    """Prose is NOT a dependency edge, so promotion is unchanged -- but the
    ids a promoted card names in prose WITHOUT a ``task_links`` row are the
    only trace of a dependency the operator never declared, and they were
    silent: a card re-promoted every tick while the blocker it named was open.

    The relation: across one recompute, the reported set is exactly
    ``prose ids - own id - linked parent ids``, while every card that nothing
    gates still lands in ``ready`` as before.
    """
    open_parent = kb.create_task(conn, title="root blocker", assignee="setup")
    done_parent = kb.create_task(conn, title="closed gate", assignee="setup")
    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (open_parent,))
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done_parent,))

    # Names BOTH ids in prose, but only `done_parent` is a real edge.
    linked = kb.create_task(
        conn, title="linked", assignee="setup", parents=[done_parent],
        body=f"Context: {open_parent} is still open, {done_parent} is done.",
    )
    # Names `open_parent` in prose and declares no edge at all. The second id
    # resolves to nothing, so it is a phantom citation, not a missing edge.
    prose_only = kb.create_task(
        conn, title="prose child", assignee="setup",
        body=f"Depends on {open_parent} before starting, not on t_deadbeefcafe.",
    )
    # `create_task` returns `ready` for a card whose only parent is already
    # done, so both must be parked in `todo` to reach the promotion scan.
    conn.execute("UPDATE tasks SET status='todo' WHERE id IN (?, ?)", (linked, prose_only))
    assert kb.get_task(conn, linked).status == "todo"

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        kb.recompute_ready(conn)

    # Promotion behaviour is untouched: nothing gates either card.
    assert kb.get_task(conn, linked).status == "ready"
    assert kb.get_task(conn, prose_only).status == "ready"

    warned = " ".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    # The unlinked real id is named; the linked one is not reported, and the
    # id that resolves to no card is a phantom citation, not a missing edge
    # (that is `_scan_prose_for_phantom_ids`' report, on the completion path).
    assert open_parent in warned
    assert done_parent not in warned
    assert "t_deadbeefcafe" not in warned


