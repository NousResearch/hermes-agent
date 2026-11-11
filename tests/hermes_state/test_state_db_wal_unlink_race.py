"""Regression coverage for WAL restoration during state.db repair (#101064).

Journal-mode restoration used to open a NEW connection after the exclusive
repair guard had released the live database. In WAL mode a writer could still
hold the unlinked old WAL inode while that second connection created a fresh
``state.db-wal`` path — two generations of one store. The restore must run
through the guard connection, before the guard releases.
"""

import sqlite3

import pytest

import hermes_state
import hermes_state_repair
from hermes_state_repair import repair_state_db_schema


def _make_db(path):
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("CREATE TABLE sessions (name TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('seed')")
    conn.close()


def test_wal_restoration_reuses_exclusive_repair_connection(tmp_path, monkeypatch):
    """Unit contract: given the guard connection, no reopen happens."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("CREATE TABLE marker (value TEXT)")

    def fail_if_reopened(_path):
        pytest.fail("WAL restoration reopened state.db outside the repair guard")

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", fail_if_reopened)

    hermes_state_repair._restore_journal_mode_after_repair(db_path, None, conn=conn)
    # The mode itself is whatever apply_wal_with_fallback resolves on this
    # runtime (WAL, or DELETE on WAL-reset-vulnerable SQLite builds); the
    # contract under test is the connection reuse, asserted above.
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() in ("wal", "delete")
    conn.close()


def test_repair_never_reopens_after_the_guard_releases(tmp_path, monkeypatch):
    """End to end through repair_state_db_schema: every connection the repair
    opens is opened while the exclusive guard is still held, and none after."""
    db = tmp_path / "state.db"
    _make_db(db)
    monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", lambda path: "forced-unhealthy")
    # The scratch-space pre-flight wants ~10GB headroom; irrelevant here.
    monkeypatch.setattr(hermes_state_repair, "_repair_scratch_space_error", lambda path: None)

    def fake_strategies(scratch_path, report):
        report["repaired"] = True
        report["strategy"] = "test_strategy"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", fake_strategies)

    events: list[str] = []
    real_guard = hermes_state_repair._exclusive_repair_db_guard
    real_connect = hermes_state_repair._connect_repair_durable

    from contextlib import contextmanager

    @contextmanager
    def tracing_guard(path):
        events.append("guard-enter")
        with real_guard(path) as pair:
            yield pair
        events.append("guard-exit")

    def tracing_connect(path, *a, **kw):
        events.append("connect")
        return real_connect(path, *a, **kw)

    monkeypatch.setattr(hermes_state_repair, "_exclusive_repair_db_guard", tracing_guard)
    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", tracing_connect)

    report = repair_state_db_schema(db, backup=False)
    assert report["repaired"] is True
    assert "guard-exit" in events
    after_release = events[events.index("guard-exit") + 1 :]
    assert "connect" not in after_release, events
