"""Tests that DB upgrades preserve existing data across schema migrations."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_cli import workflows_db as wfdb


# ── old-schema fixture (pre-drafts, pre-archived) ────────────────────────────

_OLD_DEFINITIONS_SQL = """
CREATE TABLE workflow_definitions (
    workflow_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    spec_json   TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    created_by  TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (workflow_id, version)
);
"""

_OLD_EXECUTIONS_SQL = """
CREATE TABLE workflow_executions (
    execution_id  TEXT PRIMARY KEY,
    workflow_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    context_json  TEXT NOT NULL,
    trigger_type  TEXT NOT NULL,
    trigger_id    TEXT,
    claim_lock    TEXT,
    claim_expires INTEGER,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
"""

_OLD_NODE_RUNS_SQL = """
CREATE TABLE workflow_node_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  TEXT NOT NULL,
    node_id       TEXT NOT NULL,
    status        TEXT NOT NULL,
    input_json    TEXT,
    output_json   TEXT,
    error         TEXT,
    started_at    INTEGER,
    completed_at  INTEGER
);
"""

_OLD_EVENTS_SQL = """
CREATE TABLE workflow_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  TEXT NOT NULL,
    node_run_id   INTEGER,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);
"""

_OLD_SCHEDULES_SQL = """
CREATE TABLE workflow_schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    version     INTEGER,
    trigger_id  TEXT,
    schedule    TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    next_run_at INTEGER,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
"""

_OLD_FEEDS_SQL = """
CREATE TABLE workflow_input_feeds (
    feed_id     TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    trigger_id  TEXT,
    status      TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
"""

_OLD_ITEMS_SQL = """
CREATE TABLE workflow_input_items (
    item_id       TEXT PRIMARY KEY,
    feed_id       TEXT NOT NULL,
    workflow_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    trigger_id    TEXT,
    status        TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    criteria_json TEXT NOT NULL,
    dedupe_value  TEXT,
    execution_id  TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
"""


def _create_old_schema_db(db_path: str) -> tuple[int, int, str]:
    """Create a DB with the pre-draft/pre-archive schema and seed data.

    Returns (definition_count, execution_count, feed_id) for assertion.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(
        _OLD_DEFINITIONS_SQL
        + _OLD_EXECUTIONS_SQL
        + _OLD_NODE_RUNS_SQL
        + _OLD_EVENTS_SQL
        + _OLD_SCHEDULES_SQL
        + _OLD_FEEDS_SQL
        + _OLD_ITEMS_SQL
    )

    now = int(time.time())
    spec = {
        "id": "old_workflow",
        "name": "Old Workflow",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
    }
    spec_json = json.dumps(spec, sort_keys=True)
    checksum = "abc123"

    conn.execute(
        "INSERT INTO workflow_definitions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("old_workflow", 1, "Old Workflow", 1, spec_json, checksum, "test", now),
    )
    conn.execute(
        "INSERT INTO workflow_definitions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("old_workflow", 2, "Old Workflow", 1, spec_json, "def456", "test", now + 1),
    )

    conn.execute(
        "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("wfexec_old1", "old_workflow", 1, "succeeded", '{"x": 1}', "{}", "manual", None, None, None, now, now),
    )
    conn.execute(
        "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("wfexec_old2", "old_workflow", 2, "queued", '{"y": 2}', "{}", "manual", None, None, None, now, now),
    )

    feed_id = "feed_old_1"
    conn.execute(
        "INSERT INTO workflow_input_feeds VALUES (?, ?, ?, ?, ?, ?, ?)",
        (feed_id, "old_workflow", 1, "manual", "open", now, now),
    )

    conn.commit()
    def_count = conn.execute("SELECT count(*) FROM workflow_definitions").fetchone()[0]
    exec_count = conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0]
    conn.close()
    return def_count, exec_count, feed_id


@pytest.fixture
def old_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    db_path = str(home / "workflows.db")
    def_count, exec_count, feed_id = _create_old_schema_db(db_path)
    # Clear init cache so init_db actually runs
    wfdb._INITIALIZED_DB_PATHS.clear()
    return db_path, def_count, exec_count, feed_id


def test_init_db_preserves_old_definitions(old_db):
    db_path, def_count, _, _ = old_db

    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    assert conn.execute("SELECT count(*) FROM workflow_definitions").fetchone()[0] == def_count
    row = conn.execute(
        "SELECT * FROM workflow_definitions WHERE workflow_id = 'old_workflow' AND version = 1"
    ).fetchone()
    assert row["name"] == "Old Workflow"
    assert json.loads(row["spec_json"])["id"] == "old_workflow"
    conn.close()


def test_init_db_preserves_old_executions(old_db):
    db_path, _, exec_count, _ = old_db

    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    assert conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0] == exec_count
    row = conn.execute(
        "SELECT * FROM workflow_executions WHERE execution_id = 'wfexec_old1'"
    ).fetchone()
    assert row["status"] == "succeeded"
    conn.close()


def test_init_db_adds_archived_column_with_default_zero(old_db):
    db_path, _, _, _ = old_db

    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(workflow_definitions)")}
    assert "archived" in columns
    row = conn.execute(
        "SELECT archived FROM workflow_definitions WHERE workflow_id = 'old_workflow' AND version = 1"
    ).fetchone()
    assert row["archived"] == 0
    conn.close()


def test_init_db_creates_workflow_drafts_table(old_db):
    db_path, _, _, _ = old_db

    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "workflow_drafts" in tables
    conn.close()


def test_init_db_preserves_continuous_feed(old_db):
    db_path, _, _, feed_id = old_db

    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM workflow_input_feeds WHERE feed_id = ?", (feed_id,)
    ).fetchone()
    assert row is not None
    assert row["status"] == "open"
    conn.close()


def test_init_db_is_idempotent(old_db):
    db_path, def_count, exec_count, _ = old_db

    wfdb.init_db(db_path)
    wfdb._INITIALIZED_DB_PATHS.clear()
    wfdb.init_db(db_path)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT count(*) FROM workflow_definitions").fetchone()[0] == def_count
    assert conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0] == exec_count
    conn.close()
