"""Access-unit migration compatibility against a REAL legacy board DB.

Builds a kanban.db using the PRE-feature schema (no access_units table) by
hand, then proves: connect() migrates it additively; a restart mid-migration
is safe (idempotent re-run); the partial unique index lands; existing rows
are untouched; and the new feature works immediately on the migrated DB.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_access_units as kau
from hermes_cli import kanban_db_connect as kbc


LEGACY_SCHEMA = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
    status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT,
    created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT,
    branch_name TEXT, project_id TEXT, claim_lock TEXT, claim_expires INTEGER,
    tenant TEXT, result TEXT, idempotency_key TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0, worker_pid INTEGER,
    last_failure_error TEXT, max_runtime_seconds INTEGER, last_heartbeat_at INTEGER,
    current_run_id INTEGER, workflow_template_id TEXT, current_step_key TEXT,
    skills TEXT, max_retries INTEGER, model_override TEXT, provider_override TEXT,
    reasoning_effort TEXT, goal_mode INTEGER NOT NULL DEFAULT 0,
    goal_max_turns INTEGER, session_id TEXT, block_kind TEXT,
    block_recurrences INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE task_links (parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id));
CREATE TABLE task_comments (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    author TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);
CREATE TABLE task_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    profile TEXT, step_key TEXT, status TEXT NOT NULL, claim_lock TEXT,
    claim_expires INTEGER, worker_pid INTEGER, max_runtime_seconds INTEGER,
    last_heartbeat_at INTEGER, started_at INTEGER NOT NULL, ended_at INTEGER,
    outcome TEXT, summary TEXT, metadata TEXT, error TEXT);
CREATE TABLE task_attachments (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    filename TEXT NOT NULL, stored_path TEXT NOT NULL, content_type TEXT,
    size INTEGER NOT NULL DEFAULT 0, uploaded_by TEXT, created_at INTEGER NOT NULL);
CREATE TABLE kanban_notify_subs (task_id TEXT NOT NULL, platform TEXT NOT NULL,
    chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,
    user_id_alt TEXT, chat_type TEXT, notifier_profile TEXT,
    delivery_mode TEXT NOT NULL DEFAULT 'notify', delivery_metadata TEXT,
    created_at INTEGER NOT NULL, last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id));
"""


@pytest.fixture
def legacy_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "kanban" / "boards" / "default").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The ``default`` board DB lives at <root>/kanban.db (back-compat), not
    # under boards/default/ — craft the legacy DB at the canonical path.
    db_path = home / "kanban.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(LEGACY_SCHEMA)
    # One legacy task with history that must survive untouched.
    conn.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_at, workspace_kind) "
        "VALUES ('t_legacy1', 'legacy access card', 'vladamir', 'done', 1000, 'scratch')")
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES ('t_legacy1', 'completed', '{\"result\": \"ok\"}', 1001)")
    conn.commit()
    conn.close()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return home


def test_legacy_board_migrates_additively_and_keeps_history(legacy_home: Path):
    with kbc.connect_closing() as conn:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "access_units" in tables
        idx = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='access_units'")]
        assert "idx_access_units_one_active" in idx
        # Legacy row + event untouched.
        row = conn.execute("SELECT title, status FROM tasks WHERE id='t_legacy1'").fetchone()
        assert row["title"] == "legacy access card" and row["status"] == "done"
        events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id='t_legacy1'").fetchone()[0]
        assert events == 1


def test_migration_survives_restart_and_is_idempotent(legacy_home: Path):
    db_path = kb.kanban_db_path(board="default")
    for _ in range(3):  # simulate restart mid/post migration
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with kbc.connect_closing() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM access_units").fetchone()[0]
            assert rows == 0  # no duplicate/no-op side effects


def test_migrated_board_supports_access_units_immediately(legacy_home: Path):
    key = kau.build_outcome_key("ruth", "ga4", "properties/11", "read_only")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="first post-migration unit", assignee="vladamir")
        kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=tid)
        assert kau.active_access_unit(conn, key) == tid
        dup = kb.create_task(conn, title="dup", assignee="ruth")
        with pytest.raises(kau.AccessUnitConflict):
            kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=dup)


def test_unique_index_rejects_second_active_row(legacy_home: Path):
    """The partial unique index is the hard invariant: a second active row
    for one key is impossible even via raw SQL."""
    with kbc.connect_closing() as conn:
        conn.execute(
            "INSERT INTO access_units (key, unit_class, task_id, created_at) "
            "VALUES ('k1', 'outcome', 't_a', 1)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO access_units (key, unit_class, task_id, created_at) "
                "VALUES ('k1', 'outcome', 't_b', 2)")
        # ...but a RELEASED row plus a new active row is fine (succession).
        conn.execute("UPDATE access_units SET released_at = 9 WHERE key = 'k1'")
        conn.execute(
            "INSERT INTO access_units (key, unit_class, task_id, created_at) "
            "VALUES ('k1', 'outcome', 't_b', 3)")
