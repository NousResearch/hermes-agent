"""The raw-copy quarantine/backup paths hold the connection-lifecycle lock.

``offline_file_access`` (hermes_cli.sqlite_safe_read) exists because checking
``has_live_connection()`` and *then* doing raw file I/O is a check/use race: a
connection opened in the window between the two has its POSIX advisory locks
cancelled by the raw ``close()`` -- the exact corruption route the registry
guards against. The snapshot path in ``session_recovery`` was converted to the
context manager when it landed; these tests pin the other two raw-copy sites
to the same contract:

* ``hermes_state._backup_db_file`` (malformed-DB backup before schema surgery)
* ``hermes_cli.kanban_db._backup_corrupt_db`` (corrupt-board quarantine)

Each site gets the same pair: a live connection means refusal, and a
connection attempted MID-COPY blocks on the lifecycle lock until the copy is
done rather than slipping into the gap.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading

import pytest

import hermes_state
from hermes_cli import kanban_db
from hermes_cli import sqlite_safe_read
from hermes_cli.sqlite_safe_read import connect_tracked


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty live-connection registry.

    The registry is process-global; a connection leaked by one test would make
    ``offline_file_access`` in the next raise ``LiveConnectionError`` and
    silently skip the copy under test.
    """
    with sqlite_safe_read._live_lock:
        sqlite_safe_read._live_connections.clear()
    yield
    with sqlite_safe_read._live_lock:
        sqlite_safe_read._live_connections.clear()


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    (tmp_path / "state.db-wal").write_bytes(b"wal")
    return path


def _connect_attempt_during(monkeypatch, db_path):
    """Patch ``shutil.copy2`` to probe lock ordering, not scheduling timing.

    The fixture parks the copy thread inside the patched ``copy2`` using a
    park/release event pair, starts the connector thread, waits until the
    connector has actually called into ``connect_tracked``, then checks whether
    the connection opened while the copy still holds the lifecycle lock.  Only
    after that does it release the copy so both threads can finish cleanly.

    This is scheduling-independent: the assertion is about lock ordering
    (``connect_tracked`` must block while the copy holds the lock), not about
    which thread the scheduler happens to resume first within a fixed timeout.

    Returns ``(copier, connector, events)`` where ``events`` is a namespace
    with ``inside_copy``, ``release_copy``, ``connect_attempted``,
    ``connection_opened``, and ``holder`` (list collecting the opened conn).
    """
    real_copy2 = shutil.copy2
    inside_copy = threading.Event()
    release_copy = threading.Event()
    connect_attempted = threading.Event()
    connection_opened = threading.Event()
    holder: list[sqlite3.Connection] = []
    errors: list[str] = []
    fired = threading.Event()

    def _slow_copy2(src, dst, *a, **kw):
        result = real_copy2(src, dst, *a, **kw)
        if not fired.is_set():
            fired.set()
            inside_copy.set()
            release_copy.wait(timeout=30)
        return result

    def _racing_connect():
        # Signal immediately *before* the blocking call so a timed
        # "still blocked" assertion cannot pass merely because this thread
        # had not been scheduled yet.
        connect_attempted.set()
        try:
            conn = connect_tracked(str(db_path), check_same_thread=False)
            connection_opened.set()
            holder.append(conn)
        except Exception as exc:  # pragma: no cover
            errors.append(f"connect failed: {exc}")

    monkeypatch.setattr(shutil, "copy2", _slow_copy2)

    import types
    ns = types.SimpleNamespace(
        inside_copy=inside_copy,
        release_copy=release_copy,
        connect_attempted=connect_attempted,
        connection_opened=connection_opened,
        holder=holder,
        errors=errors,
    )
    return ns


# ---------------------------------------------------------------------------
# hermes_state._backup_db_file
# ---------------------------------------------------------------------------

def test_backup_db_file_refuses_with_live_connection(db_file):
    conn = connect_tracked(str(db_file))
    try:
        assert hermes_state._backup_db_file(db_file) is None
        assert not list(db_file.parent.glob("*.malformed-backup-*"))
    finally:
        conn.close()


def test_backup_db_file_copy_is_atomic_with_the_registry(db_file, monkeypatch):
    """A connect attempted mid-copy must block on the lock, not slip in the gap."""
    ev = _connect_attempt_during(monkeypatch, db_file)

    copier = threading.Thread(
        target=lambda: hermes_state._backup_db_file(db_file), daemon=True
    )
    connector = threading.Thread(target=lambda: None, daemon=True)  # replaced below

    connect_attempted = ev.connect_attempted
    connection_opened = ev.connection_opened

    def _do_connect():
        connect_attempted.set()
        try:
            conn = connect_tracked(str(db_file), check_same_thread=False)
            connection_opened.set()
            ev.holder.append(conn)
        except Exception as exc:  # pragma: no cover
            ev.errors.append(f"connect failed: {exc}")

    connector = threading.Thread(target=_do_connect, daemon=True)

    try:
        copier.start()
        assert ev.inside_copy.wait(30), "copy never reached the patched operation"

        connector.start()
        assert ev.connect_attempted.wait(30), "connector thread never started"
        # The connector is at the lock. While the copy holds it the connection
        # must not open.
        assert not ev.connection_opened.wait(1.0), (
            "connect_tracked() completed while the raw copy was in flight -- "
            "its POSIX locks would be cancelled by the copy's close()"
        )

        ev.release_copy.set()
        # Once the copy releases the lock the connection must open promptly.
        assert ev.connection_opened.wait(30), (
            "connect_tracked() never completed after the copy released the lock"
        )
    finally:
        ev.release_copy.set()
        copier.join(30)
        connector.join(30)

    assert not ev.errors, ev.errors[0]
    assert ev.holder, "the queued connect must succeed once the copy is done"
    ev.holder[0].close()


def test_backup_db_file_still_copies_without_the_registry(db_file, monkeypatch):
    """Constrained embeds without hermes_cli keep the best-effort copy."""
    import builtins

    real_import = builtins.__import__

    def _no_safe_read(name, *args, **kwargs):
        if name == "hermes_cli.sqlite_safe_read":
            raise ImportError("embed path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_safe_read)
    result = hermes_state._backup_db_file(db_file)
    assert result is not None and result.exists()
    assert result.with_name(result.name + "-wal").exists()


# ---------------------------------------------------------------------------
# hermes_cli.kanban_db._backup_corrupt_db
# ---------------------------------------------------------------------------

def test_backup_corrupt_db_refuses_with_live_connection(db_file):
    conn = connect_tracked(str(db_file))
    try:
        assert kanban_db._backup_corrupt_db(db_file) is None
        assert not list(db_file.parent.glob("*.corrupt.*.bak"))
    finally:
        conn.close()


def test_backup_corrupt_db_copy_is_atomic_with_the_registry(db_file, monkeypatch):
    """A connect attempted mid-quarantine must block, not land in the gap."""
    ev = _connect_attempt_during(monkeypatch, db_file)

    connect_attempted = ev.connect_attempted
    connection_opened = ev.connection_opened

    def _do_connect():
        connect_attempted.set()
        try:
            conn = connect_tracked(str(db_file), check_same_thread=False)
            connection_opened.set()
            ev.holder.append(conn)
        except Exception as exc:  # pragma: no cover
            ev.errors.append(f"connect failed: {exc}")

    copier = threading.Thread(
        target=lambda: kanban_db._backup_corrupt_db(db_file), daemon=True
    )
    connector = threading.Thread(target=_do_connect, daemon=True)

    try:
        copier.start()
        assert ev.inside_copy.wait(30), "copy never reached the patched operation"

        connector.start()
        assert ev.connect_attempted.wait(30), "connector thread never started"
        assert not ev.connection_opened.wait(1.0), (
            "connect_tracked() completed while the quarantine fingerprint/copy "
            "was in flight -- its POSIX locks would be cancelled by our close()"
        )

        ev.release_copy.set()
        assert ev.connection_opened.wait(30), (
            "connect_tracked() never completed after the quarantine released the lock"
        )
    finally:
        ev.release_copy.set()
        copier.join(30)
        connector.join(30)

    assert not ev.errors, ev.errors[0]
    assert ev.holder, "the queued connect must succeed once the quarantine is done"
    ev.holder[0].close()


def test_backup_corrupt_db_still_copies_sidecars(db_file):
    result = kanban_db._backup_corrupt_db(db_file)
    assert result is not None and result.exists()
    assert result.with_name(result.name + "-wal").exists()
