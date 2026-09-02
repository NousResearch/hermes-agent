"""Regression coverage for WAL restoration during state.db repair."""

import sqlite3

import pytest

import hermes_state


def test_wal_restoration_reuses_exclusive_repair_connection(tmp_path, monkeypatch):
    """WAL must be restored before the repair guard releases the live DB."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("CREATE TABLE marker (value TEXT)")
    conn.execute("PRAGMA journal_mode=DELETE")

    def fail_if_reopened(_path):
        pytest.fail("WAL restoration reopened state.db outside the repair guard")

    monkeypatch.setattr(hermes_state, "_connect_repair_durable", fail_if_reopened)

    hermes_state._restore_journal_mode_after_repair(
        db_path,
        "delete",
        conn=conn,
    )

    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()
