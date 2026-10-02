"""Wake-Guard vocabulary: shared terminal-status proof helpers on the kanban
DB facade (t_bdd69e28, port of PR #91's tests/hermes_cli/test_kanban_terminal_statuses.py).

``TASK_TERMINAL_STATUSES`` is a deliberate superset: this line persists
"done"/"archived" (the pair every inline gating site uses), while
"completed"/"cancelled" are legacy aliases from boards written by older lines.
"""

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_terminal_status_set_includes_legacy_completed():
    assert "completed" in kb.TASK_TERMINAL_STATUSES
    assert "cancelled" in kb.TASK_TERMINAL_STATUSES
    assert {"done", "archived"} <= set(kb.TASK_TERMINAL_STATUSES)


def test_terminal_set_covers_upstream_gating_statuses():
    """The shared vocabulary must at least cover the ("done", "archived") pair
    that every inline parent-gating / ready-promotion site tests against."""
    assert {"done", "archived"} <= set(kb.TASK_TERMINAL_STATUSES)


def test_get_scoping_freshness_reads_task_and_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "freshness.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="scoping proof", assignee="worker")
        # Unknown task row -> unknown freshness.
        assert kb.get_scoping_freshness(conn, "t_missing", None) is None

        kb.claim_task(conn, tid)
        run_row = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        assert run_row is not None
        run_id = run_row["id"]

        info = kb.get_scoping_freshness(conn, tid, run_id)
        assert info == {"status": "running", "run_ended_at": None}

        # run_id=None never treats a run as ended.
        assert kb.get_scoping_freshness(conn, tid, None)["run_ended_at"] is None
        # A run id belonging to another task is unknown, never "ended".
        other = kb.create_task(conn, title="other", assignee="worker")
        foreign = kb.get_scoping_freshness(conn, other, run_id)
        assert foreign["status"] == "ready"
        assert foreign["run_ended_at"] is None

        kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
        ended = kb.get_scoping_freshness(conn, tid, run_id)
        assert ended["status"] == "done"
        assert ended["run_ended_at"] is not None
    finally:
        conn.close()


def test_get_scoping_freshness_terminal_status_positive_proof(tmp_path, monkeypatch):
    """A terminal task row yields positive terminal proof through the helper."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "freshness-positive.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="done card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
        info = kb.get_scoping_freshness(conn, tid, None)
        assert info is not None
        assert kb.task_is_terminal(info["status"]) is True
    finally:
        conn.close()
    assert kb.task_is_terminal("done") is True
    assert kb.task_is_terminal("archived") is True
    assert kb.task_is_terminal("completed") is True
    assert kb.task_is_terminal("running") is False
    assert kb.task_is_terminal("ready") is False
    assert kb.task_is_terminal(None) is False
    assert kb.task_is_terminal("") is False
