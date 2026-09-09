"""C1 / O1 (2026-09-06): the cap counts the assignee's own ledger only, and the
overwatch extension is once per card, never past the hard ceiling."""
from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_kanban_cost_cap import (  # noqa: F401  (fixtures)
    kanban_home, _claim_running, _make_state_db, _noop_signal, _workspace,
)
from hermes_cli import kanban_db_connect as kbc


def _add_session(db_path, cwd, cost, task_id, sid):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO sessions (id, cwd, source, title, estimated_cost_usd) VALUES (?, ?, 'cli', ?, ?)",
        (sid, cwd, f"Work kanban task {task_id} #9", cost),
    )
    con.commit()
    con.close()


def test_reviewer_ledger_does_not_count_against_the_builder(kanban_home):
    """A $0.60 build plus a $0.50 review in ANOTHER profile's ledger must not
    breach a $1.00 cap: reviewer spend is measured on the review card."""
    conn = kbc.connect()
    W = _workspace(kanban_home, "asg")
    tid = kb.create_task(conn, title="x", assignee="bob", max_cost=1.00, workspace_path=W)
    _claim_running(conn, tid)
    bob_db = _make_state_db(kanban_home / "state.db", [(W, 0.60)], task_id=tid)
    # Another profile's ledger carrying the same card id (a reviewer session).
    (kanban_home / "profiles" / "rodge").mkdir(parents=True)
    rodge_db = _make_state_db(kanban_home / "profiles" / "rodge" / "state.db", [(W, 0.50)], task_id=tid)

    assert kb.enforce_max_cost(conn, signal_fn=_noop_signal, state_db_path=bob_db) == []
    assert kb.get_task(conn, tid).status == "running"
    # The reviewer's ledger does carry the spend (so the lifetime report can see it).
    assert kb._session_cost_in_db(str(rodge_db), task_id=tid) == pytest.approx(0.50)
    # Negative control: the assignee's OWN spend over the cap still blocks.
    _add_session(bob_db, W, 0.45, tid, "s-extra")
    assert kb.enforce_max_cost(conn, signal_fn=_noop_signal, state_db_path=bob_db) == [tid]
    assert kb.get_task(conn, tid).block_kind == "cost_cap"
    conn.close()


def test_set_cap_once_and_never_past_hard_ceiling(kanban_home):
    conn = kbc.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_cost=1.00)
    # +0.50 is allowed once.
    assert kb.set_task_max_cost(conn, tid, 1.50, by="default", reason="large but legitimate") == 1.50
    assert kb.get_task(conn, tid).max_cost == 1.50
    bodies = [c.body for c in kb.list_comments(conn, tid)]
    assert any(b.startswith("cost-extension:") for b in bodies)
    # A second extension is refused: a second breach is Richie's.
    with pytest.raises(ValueError, match="already been extended"):
        kb.set_task_max_cost(conn, tid, 1.50, by="default")
    conn.close()


def test_set_cap_refuses_above_hard_ceiling_and_non_increase(kanban_home):
    conn = kbc.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_cost=1.00)
    with pytest.raises(ValueError, match="hard ceiling"):
        kb.set_task_max_cost(conn, tid, 2.00, by="default")
    with pytest.raises(ValueError, match="not above"):
        kb.set_task_max_cost(conn, tid, 0.90, by="default")
    assert kb.get_task(conn, tid).max_cost == 1.00
    conn.close()
