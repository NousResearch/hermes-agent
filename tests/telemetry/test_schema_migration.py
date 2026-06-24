"""Schema tests: tel_* tables exist after init; SCHEMA_VERSION bumped."""

from __future__ import annotations

import sqlite3

import hermes_state


TEL_TABLES = {
    "tel_runs", "tel_spans", "tel_model_calls", "tel_tool_calls",
    "tel_gateway_events", "tel_cron_events", "tel_skill_events",
    "tel_memory_events", "tel_feedback_events", "tel_error_events",
}


def test_schema_version_is_17_or_higher():
    assert hermes_state.SCHEMA_VERSION >= 17


def test_tel_tables_present_in_schema_sql():
    for tbl in TEL_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {tbl}" in hermes_state.SCHEMA_SQL


def test_tel_tables_created_on_executescript(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    rows = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert TEL_TABLES.issubset(rows), f"missing: {TEL_TABLES - rows}"


def test_executescript_is_idempotent(tmp_path):
    # IF NOT EXISTS means re-running on an existing DB is a no-op, not an error.
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    conn.executescript(hermes_state.SCHEMA_SQL)  # second run must not raise
    conn.close()
