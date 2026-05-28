"""Per-run worker session_id capture for dispatcher resume-on-respawn.

These cover the helpers added for issue #33873 — the goal is for the
dispatcher to launch the next worker for a task with ``--resume <id>``
so the new worker continues the prior conversation instead of starting
fresh. The dispatcher-side wiring lives elsewhere; this file pins the
storage primitives (schema migration + the two query helpers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors the rig used
    by test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_task_with_run(conn, *, title: str = "t") -> tuple[str, int]:
    """Create a running task and synthesize a matching run row.

    Mirrors the post-claim state the dispatcher leaves a task in: status
    ``running`` and a ``task_runs`` row pointed at by ``current_run_id``.
    Going through ``claim_task`` would require navigating ``todo → ready``
    promotion first; this short-circuit keeps the per-helper tests
    focused on the helpers themselves.

    No outer ``write_txn`` here: ``create_task`` opens its own, and
    helpers like ``set_run_session_id`` open theirs. Nesting would raise
    ``cannot start a transaction within a transaction``.
    """
    task_id = kb.create_task(
        conn, title=title, assignee="default", initial_status="running"
    )
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at) "
        "VALUES (?, 'default', 'running', strftime('%s','now'))",
        (task_id,),
    )
    run_id = int(cur.lastrowid)
    conn.execute(
        "UPDATE tasks SET current_run_id = ? WHERE id = ?",
        (run_id, task_id),
    )
    conn.commit()
    return task_id, run_id


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def test_migration_adds_session_id_to_task_runs(kanban_home):
    with kb.connect() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")}
    assert "session_id" in cols


def test_migration_is_idempotent_on_existing_db(kanban_home):
    # Re-running init_db on a DB that already has session_id should not raise
    # or duplicate the column.
    kb.init_db()
    with kb.connect() as conn:
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")]
    assert cols.count("session_id") == 1


# ---------------------------------------------------------------------------
# set_run_session_id
# ---------------------------------------------------------------------------

def test_set_run_session_id_persists(kanban_home):
    with kb.connect() as conn:
        _, run_id = _make_task_with_run(conn)
        assert kb.set_run_session_id(conn, run_id, "20260528_120000_abcdef")
        row = conn.execute(
            "SELECT session_id FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert row["session_id"] == "20260528_120000_abcdef"


def test_set_run_session_id_rejects_empty(kanban_home):
    with kb.connect() as conn:
        _, run_id = _make_task_with_run(conn)
        assert not kb.set_run_session_id(conn, run_id, "")
        assert not kb.set_run_session_id(conn, run_id, "   ")


def test_set_run_session_id_returns_false_for_unknown_run(kanban_home):
    with kb.connect() as conn:
        # No claim → no run row → unknown run id.
        assert not kb.set_run_session_id(conn, 99999, "20260528_120000_abcdef")


def test_set_run_session_id_overwrites(kanban_home):
    """A worker that re-registers (e.g. after an in-process resume) wins
    over the prior value. Per-run latest-wins semantics matter so the
    dispatcher always reads the freshest session id for that run."""
    with kb.connect() as conn:
        _, run_id = _make_task_with_run(conn)
        assert kb.set_run_session_id(conn, run_id, "first")
        assert kb.set_run_session_id(conn, run_id, "second")
        row = conn.execute(
            "SELECT session_id FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert row["session_id"] == "second"


# ---------------------------------------------------------------------------
# latest_session_id_for_task
# ---------------------------------------------------------------------------

def test_latest_session_id_none_when_no_runs(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t", assignee="default")
        assert kb.latest_session_id_for_task(conn, task_id) is None


def test_latest_session_id_none_when_no_session_recorded(kanban_home):
    """Run exists but worker never called kanban_register_session."""
    with kb.connect() as conn:
        task_id, _ = _make_task_with_run(conn)
        assert kb.latest_session_id_for_task(conn, task_id) is None


def test_latest_session_id_returns_recorded(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _make_task_with_run(conn)
        kb.set_run_session_id(conn, run_id, "sess-A")
        assert kb.latest_session_id_for_task(conn, task_id) == "sess-A"


def test_latest_session_id_picks_most_recent_run(kanban_home):
    """A task that has been reclaimed accumulates multiple runs. The
    dispatcher should resume the most recent session, not the oldest."""
    with kb.connect() as conn:
        task_id, run1 = _make_task_with_run(conn)
        kb.set_run_session_id(conn, run1, "sess-old")
        # Synthesize the reclaim → fresh claim cycle: ended row for run1,
        # new running row for run2.
        conn.execute(
            "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
            "ended_at=strftime('%s','now') WHERE id=?",
            (run1,),
        )
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, 'default', 'running', strftime('%s','now') + 1)",
            (task_id,),
        )
        run2 = int(cur.lastrowid)
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run2, task_id),
        )
        conn.commit()
        kb.set_run_session_id(conn, run2, "sess-new")
        assert kb.latest_session_id_for_task(conn, task_id) == "sess-new"


def test_latest_session_id_skips_null_sessions(kanban_home):
    """If the latest run has no session id (worker crashed before
    registering) but an earlier run does, fall back to the earlier one
    rather than returning None and losing resumability."""
    with kb.connect() as conn:
        task_id, run1 = _make_task_with_run(conn)
        kb.set_run_session_id(conn, run1, "sess-A")
        # Synthesize a later run that never registered a session.
        conn.execute(
            "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
            "ended_at=strftime('%s','now') WHERE id=?",
            (run1,),
        )
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, 'default', 'running', strftime('%s','now') + 1)",
            (task_id,),
        )
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (int(cur.lastrowid), task_id),
        )
        conn.commit()
        assert kb.latest_session_id_for_task(conn, task_id) == "sess-A"


def test_latest_session_id_isolated_per_task(kanban_home):
    with kb.connect() as conn:
        t1, run1 = _make_task_with_run(conn, title="one")
        t2, run2 = _make_task_with_run(conn, title="two")
        kb.set_run_session_id(conn, run1, "sess-one")
        kb.set_run_session_id(conn, run2, "sess-two")
        assert kb.latest_session_id_for_task(conn, t1) == "sess-one"
        assert kb.latest_session_id_for_task(conn, t2) == "sess-two"
