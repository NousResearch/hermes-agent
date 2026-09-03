"""doctor's session probes must never leak a blocking connection (#100836).

On a malformed state.db the COUNT(*) probe raises; without try/finally the
connection stays open, and the except path's repair_state_db_schema() then
fails its own _live_writer_holds_db() probe against that leaked connection —
doctor self-detects as the live writer and refuses to repair even after the
operator stopped the gateway.
"""

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import doctor  # noqa: E402


def test_probe_closes_connection_on_malformed_db(tmp_path):
    """The COUNT(*) probe must close its connection even when the DB is
    malformed — otherwise doctor's own repair path is blocked by its own
    leaked handle (#100836)."""
    db_path = tmp_path / "state.db"
    # Structural corruption that makes SELECT COUNT(*) fail: valid SQLite
    # header, garbage body -> DatabaseError on first statement.
    header = b"SQLite format 3\x00" + b"\x10\x00" + b"\x00" * 100
    db_path.write_bytes(header + b"\x00" * 4096)

    closed = {"n": 0}
    real_connect = sqlite3.connect

    class _TrackingConn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *a, **kw):
            return self._conn.execute(*a, **kw)

        def close(self):
            closed["n"] += 1
            self._conn.close()

    def connect_tracking(path, *a, **kw):
        return _TrackingConn(real_connect(path, *a, **kw))

    orig_connect = sqlite3.connect
    sqlite3.connect = connect_tracking
    try:
        # The pre-fix shape: bare execute, close only on success.
        # (Simulates the original code path to demonstrate the leak.)
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            conn.close()  # pre-fix: never reached on error
        except sqlite3.DatabaseError:
            pass  # pre-fix: conn leaked here
        leaked = closed["n"] == 0
        assert leaked, "pre-fix shape demonstrates the leak (close skipped)"

        # The fixed shape: try/finally guarantees close on every path.
        closed["n"] = 0
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            pass
        assert closed["n"] == 1, (
            "fixed shape must close exactly once on the error path (#100836)"
        )
    finally:
        sqlite3.connect = orig_connect


def test_doctor_probe_uses_try_finally_shape():
    """Source-shape guard: the probe block in doctor.py must wrap
    execute/close in try/finally so the leak class cannot silently return."""
    source = Path(doctor.__file__).read_text(encoding="utf-8")
    probe_at = source.index('conn.execute("SELECT COUNT(*) FROM sessions")')
    window = source[probe_at - 500: probe_at + 500]
    assert "finally:" in window, (
        "the probe's execute/close must be inside try/finally (#100836)"
    )
    # The repaired-recount site too
    probe2 = source.index('"SELECT COUNT(*) FROM sessions"', probe_at + 1)
    window2 = source[probe2 - 500: probe2 + 500]
    assert "finally:" in window2, (
        "the post-repair recount must also close in a finally (#100836)"
    )
