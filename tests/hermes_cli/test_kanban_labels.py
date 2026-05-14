"""Tests for the ``tasks.labels`` triage column (Phase 1 of the
Kanban triage layer).

Covers:
  * fresh DBs have the ``labels`` column with the JSON-array default
  * legacy DBs missing the column are auto-migrated on ``connect()``
  * ``set_task_labels`` / ``get_task_labels`` roundtrip
  * ``set_task_labels`` rejects malformed input
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB.

    Mirrors ``tests/hermes_cli/test_kanban_db.py::kanban_home`` so the
    label tests inherit the same path-resolution behaviour.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Wipe the per-process init cache so each test starts from a
    # truly cold DB; otherwise a prior test in the same process can
    # leave the migration "already run" flag set for tmp_path.
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Schema: fresh DB
# ---------------------------------------------------------------------------

def test_fresh_db_has_labels_column(kanban_home):
    """A freshly-initialised DB must include the ``labels`` column.

    Guards against silent regressions in ``SCHEMA_SQL``.
    """
    with kb.connect() as conn:
        cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "labels" in cols, "labels column missing from fresh DB"
    info = cols["labels"]
    assert info["type"].upper() == "TEXT"
    assert int(info["notnull"]) == 1, "labels must be NOT NULL"
    # SQLite quotes string defaults so the literal '[]' shows up as
    # "'[]'" in dflt_value. Accept either form for forward compat.
    assert info["dflt_value"] in ("'[]'", "[]")


def test_fresh_task_has_empty_labels(kanban_home):
    """A task created on a fresh DB starts with an empty label list."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs triage")
        assert kb.get_task_labels(conn, tid) == []


# ---------------------------------------------------------------------------
# Migration: legacy DB
# ---------------------------------------------------------------------------

def test_legacy_db_gets_labels_column_on_connect(tmp_path, monkeypatch):
    """Manually build a DB without ``labels``, then connect.

    The connect() path must run ``_migrate_add_labels_column`` so the
    column appears without the operator having to do anything.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()

    db_path = home / "kanban.db"
    # Hand-craft a "legacy" tasks table that has every pre-labels
    # column (i.e. the schema as it existed just before this change).
    # Anything sparser breaks the index pass in
    # ``conn.executescript(SCHEMA_SQL)`` because indexes reference
    # columns ``_migrate_add_optional_columns`` has not yet added.
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.executescript(
        """
        CREATE TABLE tasks (
            id                   TEXT PRIMARY KEY,
            title                TEXT NOT NULL,
            body                 TEXT,
            assignee             TEXT,
            status               TEXT NOT NULL,
            priority             INTEGER DEFAULT 0,
            created_by           TEXT,
            created_at           INTEGER NOT NULL,
            started_at           INTEGER,
            completed_at         INTEGER,
            workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
            workspace_path       TEXT,
            claim_lock           TEXT,
            claim_expires        INTEGER,
            tenant               TEXT,
            result               TEXT,
            idempotency_key      TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid           INTEGER,
            last_failure_error   TEXT,
            max_runtime_seconds  INTEGER,
            last_heartbeat_at    INTEGER,
            current_run_id       INTEGER,
            workflow_template_id TEXT,
            current_step_key     TEXT,
            skills               TEXT,
            max_retries          INTEGER
        );
        INSERT INTO tasks (id, title, status, created_at)
        VALUES ('t_legacy', 'old task', 'ready', 1);
        """
    )
    raw.close()

    # Sanity: column really is absent before we touch it.
    raw = sqlite3.connect(str(db_path))
    before = {r[1] for r in raw.execute("PRAGMA table_info(tasks)")}
    raw.close()
    assert "labels" not in before

    # connect() auto-runs migrations.
    with kb.connect(db_path) as conn:
        after = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "labels" in after, "labels column was not added by migration"
        # Pre-existing row should now read back with the default.
        row = conn.execute(
            "SELECT labels FROM tasks WHERE id = ?", ("t_legacy",)
        ).fetchone()
        assert row["labels"] == "[]"
        assert kb.get_task_labels(conn, "t_legacy") == []


def test_migration_is_idempotent(kanban_home):
    """Running the migration twice must not error or duplicate columns."""
    with kb.connect() as conn:
        # First run was implicit via init_db; calling it directly should
        # short-circuit on the PRAGMA table_info check.
        added_again = kb._migrate_add_labels_column(conn)
        assert added_again is False
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(tasks)")]
        # Column only appears once.
        assert cols.count("labels") == 1


# ---------------------------------------------------------------------------
# Set / get roundtrip
# ---------------------------------------------------------------------------

def test_set_and_get_labels_roundtrip(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="triage me")
        assert kb.set_task_labels(conn, tid, ["bug", "infra"]) is True
        assert kb.get_task_labels(conn, tid) == ["bug", "infra"]
        # Persists across statements.
        stored = conn.execute(
            "SELECT labels FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["labels"]
        assert json.loads(stored) == ["bug", "infra"]


def test_set_labels_deduplicates_and_strips(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="dup tags")
        kb.set_task_labels(conn, tid, ["bug", "  bug  ", "", "infra", "bug"])
        # Whitespace-only entries dropped, dupes collapsed, order preserved.
        assert kb.get_task_labels(conn, tid) == ["bug", "infra"]


def test_set_labels_overwrites(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="replace me")
        kb.set_task_labels(conn, tid, ["old"])
        kb.set_task_labels(conn, tid, ["new", "tags"])
        assert kb.get_task_labels(conn, tid) == ["new", "tags"]


def test_set_labels_empty_list_clears(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="clear me")
        kb.set_task_labels(conn, tid, ["something"])
        kb.set_task_labels(conn, tid, [])
        assert kb.get_task_labels(conn, tid) == []


def test_set_labels_unknown_task_returns_false(kanban_home):
    with kb.connect() as conn:
        assert kb.set_task_labels(conn, "t_does_not_exist", ["x"]) is False


def test_get_labels_unknown_task_returns_empty(kanban_home):
    with kb.connect() as conn:
        assert kb.get_task_labels(conn, "t_ghost") == []


# ---------------------------------------------------------------------------
# Malformed input rejection
# ---------------------------------------------------------------------------

def test_set_labels_rejects_non_list(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bad input")
        with pytest.raises(TypeError, match="list of strings"):
            kb.set_task_labels(conn, tid, "bug")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="list of strings"):
            kb.set_task_labels(conn, tid, {"bug": True})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="list of strings"):
            kb.set_task_labels(conn, tid, ("bug",))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="list of strings"):
            kb.set_task_labels(conn, tid, None)  # type: ignore[arg-type]


def test_set_labels_rejects_non_string_elements(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bad items")
        with pytest.raises(TypeError, match="only strings"):
            kb.set_task_labels(conn, tid, ["ok", 7])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="only strings"):
            kb.set_task_labels(conn, tid, ["ok", None])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="only strings"):
            kb.set_task_labels(conn, tid, [["nested"]])  # type: ignore[list-item]
        # Row stays untouched after a rejected write.
        assert kb.get_task_labels(conn, tid) == []
