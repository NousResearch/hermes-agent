"""Wake-Guard: the kanban stop nudge must not lie about a dead card (t_bdd69e28).

Port of PR #91's tests/agent/test_kanban_stop_expiry.py to the upstream line.
A worker session whose ``HERMES_KANBAN_TASK`` scoping has expired (card terminal
on the board, or the pinned run already ended) must not be reminded to hand off
a card that is no longer its responsibility. Unknown freshness fails open toward
the legacy nudge; an explicit ``task_id`` argument is never second-guessed.
"""

import pytest

from agent.kanban_stop import build_kanban_stop_nudge
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _fresh_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "stop-expiry.db"))
    kb.init_db()
    conn = kbc.connect()
    return conn


def _env_task(monkeypatch, tid, run_id=None):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    if run_id is not None:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _latest_run_id(conn, tid):
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    return row["id"] if row is not None else None


def test_stop_nudge_no_lie_on_terminal_card(tmp_path, monkeypatch):
    """A completed card must not produce a 'card still running' nudge."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished before the session ended")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    nudge = build_kanban_stop_nudge(messages=None, attempts=0, task_id=None)

    assert nudge is None


def test_stop_nudge_no_lie_when_scoping_run_ended(tmp_path, monkeypatch):
    """A pinned run that already ended expires the scoping even though the card
    row itself was left non-terminal by the reclaim path."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="ended run", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = _latest_run_id(conn, tid)
        assert run_id is not None
        kb.complete_task(conn, tid, summary="run closed", expected_run_id=run_id)
    finally:
        conn.close()
    _env_task(monkeypatch, tid, run_id)

    nudge = build_kanban_stop_nudge(messages=None, attempts=0, task_id=None)

    assert nudge is None


def test_stop_nudge_keeps_firing_for_live_worker(tmp_path, monkeypatch):
    """A live (non-terminal, un-ended) scoping keeps the legacy nudge behavior."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="live card", assignee="worker")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    nudge = build_kanban_stop_nudge(messages=None, attempts=0, task_id=None)

    assert isinstance(nudge, str)
    assert "kanban_complete" in nudge


def test_stop_nudge_fail_closed_when_freshness_unknown(tmp_path, monkeypatch):
    """A missing task row is UNKNOWN freshness — the nudge is never silenced on
    a guess (the legacy reminder stands)."""
    _fresh_board(tmp_path, monkeypatch)
    _env_task(monkeypatch, "t_does_not_exist")

    nudge = build_kanban_stop_nudge(messages=None, attempts=0, task_id=None)

    assert isinstance(nudge, str)


def test_stop_nudge_explicit_task_id_keeps_nudge(tmp_path, monkeypatch):
    """An explicit ``task_id`` argument keeps the caller's semantics: the env
    scoping expiry never second-guesses it."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        dead_tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, dead_tid, summary="done")
        live_tid = kb.create_task(conn, title="explicitly named", assignee="worker")
    finally:
        conn.close()
    _env_task(monkeypatch, dead_tid)

    nudge = build_kanban_stop_nudge(messages=None, attempts=0, task_id=live_tid)

    assert isinstance(nudge, str)
    assert live_tid in nudge
