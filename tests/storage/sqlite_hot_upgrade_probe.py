"""Subprocess probe for retained pre-upgrade SQLite connections.

Only stdlib imports in 'live' and 'stale' modes so the POSIX lock regression
can also run with WSL's bare Python, independent of Hermes dependencies.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import types


root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sys.path.insert(0, str(root))
import hermes_cli  # noqa: E402


if mode == "stale":
    # A newly loaded post-cutover CLI facade has no historic live registry.
    facade = types.ModuleType("hermes_cli.sqlite_safe_read")
    facade.SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"
    sys.modules[facade.__name__] = facade
    hermes_cli.sqlite_safe_read = facade
    import storage.sqlite_safe_read as current  # noqa: E402

    assert current._live_lock is not getattr(facade, "_live_lock", None)
    assert current._live_connections == {}
    print("stale-facade isolation: OK")
    raise SystemExit(0)

# A pre-update hermes_cli.sqlite_safe_read was loaded *before* the disk
# swap. The old TrackedConnection's Python methods continue to resolve their
# globals in that cached module after a new storage owner is imported.
name = "hermes_cli.sqlite_safe_read"
archive = root / "tests" / "fixtures" / "legacy_sqlite_safe_read.py"
spec = importlib.util.spec_from_file_location(name, archive)
assert spec is not None and spec.loader is not None
legacy = importlib.util.module_from_spec(spec)
sys.modules[name] = legacy
hermes_cli.sqlite_safe_read = legacy
spec.loader.exec_module(legacy)
assert "storage.sqlite_safe_read" not in sys.modules

_INTRUDER = """
import sqlite3, sys
with sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0) as conn:
    try:
        conn.execute('PRAGMA busy_timeout=0')
        conn.execute('BEGIN IMMEDIATE')
        conn.execute("INSERT INTO t VALUES ('intruder')")
        conn.execute('COMMIT')
        print('ACQUIRED')
    except sqlite3.OperationalError:
        print('BLOCKED')
"""


def intruder(db: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _INTRUDER, str(db)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


with tempfile.TemporaryDirectory() as temp:
    db = Path(temp) / "state.db"
    setup = sqlite3.connect(str(db))
    setup.execute("PRAGMA journal_mode=DELETE")
    setup.execute("CREATE TABLE t(v TEXT)")
    setup.commit()
    setup.close()

    old = legacy.connect_tracked(db, isolation_level=None)
    old.execute("BEGIN IMMEDIATE")
    old.execute("INSERT INTO t VALUES ('old-writer')")
    try:
        assert intruder(db) == "BLOCKED", "old writer never held a real SQLite lock"

        # Source swapped: the new owner must reuse these exact Python objects.
        import storage.sqlite_safe_read as current  # noqa: E402

        assert current._live_lock is legacy._live_lock
        assert current._live_connections is legacy._live_connections
        assert legacy._live_connections[str(db.resolve())] == 1

        # No new consumer can close a raw fd and cancel the old transaction.
        assert current.has_live_connection(db)
        assert current.has_live_connection(str(db) + "-wal")
        assert current.has_live_connection(str(db) + "-shm")
        assert current.read_header_bytes_preopen(db) is None
        for target in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            try:
                with current.offline_file_access(target):
                    raise AssertionError("offline file probe bypassed old registry")
            except current.LiveConnectionError:
                pass
        assert intruder(db) == "BLOCKED", "new owner's raw probe cancelled an old lock"

        if mode == "consumer":
            from agent.context_references import preprocess_context_references

            result = preprocess_context_references(
                "@file:state.db", cwd=db.parent, context_length=16384
            )
            assert "live SQLite database file" in result.message, result.message
            assert intruder(db) == "BLOCKED", "context preview cancelled the old lock"

        # New and old connection classes must share a count *and* lock. Close
        # the legacy connection first; the new one still forbids raw access.
        fresh = current.connect_tracked(db, isolation_level=None)
        try:
            assert legacy._live_connections[str(db.resolve())] == 2
            old.close()
            assert legacy._live_connections[str(db.resolve())] == 1
            assert current.read_header_bytes_preopen(db) is None

            # Reciprocal ordering: an old reader retained after the swap must
            # observe a NEW tracked connection and refuse its own raw probe.
            fresh.execute("BEGIN IMMEDIATE")
            fresh.execute("INSERT INTO t VALUES ('new-writer')")
            assert legacy.has_live_connection(db)
            assert legacy.has_live_connection(str(db) + "-wal")
            assert legacy.read_header_bytes_preopen(db) is None
            assert intruder(db) == "BLOCKED", "old reader cancelled new writer's lock"
            fresh.execute("ROLLBACK")
        finally:
            fresh.close()
        assert not current.has_live_connection(db)
        assert current.read_header_bytes_preopen(db, length=16) == b"SQLite format 3\x00"
        assert intruder(db) == "ACQUIRED", "a closed transaction still holds locks"
    finally:
        if old._hermes_tracked_path is not None:
            old.close()

print("live legacy-to-canonical registry upgrade: OK")
