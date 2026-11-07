"""Shared helpers for the authority and worker backend tests: seed a profile store with the legacy store, run the
legacy function and the backend verb on copies of one seed, compare outcomes and rows, and count the transactions a
verb opens on its holder. The legacy store is the pinned checkout's (``tests/persistence_sqlite/_legacy_store.py``);
without one the helpers that run it skip the test."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from persistence import PersistenceError
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite._legacy_store import legacy_scope

#: The tables the authority and worker transactions write.
AUTHORITY_TABLES = ("sessions", "session_admissions", "worker_executions", "worker_receipts", "state_meta",
                    "runtime_epoch", "messages")


#: The legacy store's typed refusals and the contract class each one is raised as.
_CONTRACT_CLASSES = {
    "SessionTurnLeaseLostError": "TurnLeaseLost",
    "SessionCompressionInProgressError": "CompressionInProgress",
    "CompressionSessionBusyError": "CompressionBusy",
    "CompressionSessionClosedError": "CompressionClosed",
    "SessionActiveWriteGuardError": "SessionWriteGuarded",
}


def copy_db(src: Path, dst: Path) -> None:
    """A consistent copy of ``src`` at ``dst`` through the SQLite backup API."""
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def dump(path: Path, tables: Iterable[str] = AUTHORITY_TABLES) -> dict[str, Any]:
    """Every row of each named table that exists, in rowid order."""
    with sqlite3.connect(path) as conn:
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in tables if table in present}


def outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", value)`` or ``("error", exception)``."""
    try:
        return "ok", call()
    except Exception as exc:  # compared by reason, class and message
        return "error", exc


def assert_same(legacy: tuple[str, Any], backend: tuple[str, Any], project: Callable[[Any], Any] = lambda v: v) -> None:
    """The backend returned what the legacy function returned (after ``project``), or refused the same way: the same
    reason for an authority refusal, the contract class of a legacy typed refusal with the same text, a contract
    error chaining the same SQLite error, else the same class and text."""
    assert backend[0] == legacy[0], (legacy, backend)
    if legacy[0] == "ok":
        assert project(backend[1]) == legacy[1]
        return
    legacy_exc, exc = legacy[1], backend[1]
    if type(legacy_exc) is AuthorityRefusal or type(legacy_exc).__name__ == "RuntimeStoreError":
        assert isinstance(exc, AuthorityRefusal), (legacy_exc, exc)
        assert (exc.reason, exc.field, exc.detail) == (legacy_exc.reason, legacy_exc.field, legacy_exc.detail)
    elif type(legacy_exc).__name__ in _CONTRACT_CLASSES:
        assert type(exc).__name__ == _CONTRACT_CLASSES[type(legacy_exc).__name__], (legacy_exc, exc)
        assert str(exc) == str(legacy_exc)
    elif isinstance(legacy_exc, sqlite3.Error):
        assert isinstance(exc, PersistenceError) and type(exc.__cause__) is type(legacy_exc), (legacy_exc, exc)
    else:
        assert isinstance(exc, type(legacy_exc)), (legacy_exc, exc)
        assert str(exc) == str(legacy_exc)


def seed_legacy(path: Path, seed: Callable[[Any], None]) -> None:
    """Build the store at ``path`` with the legacy ``SessionDB`` and run ``seed(db)`` on it."""
    with legacy_scope():
        from hermes_state import SessionDB

        db = SessionDB(db_path=path)
        try:
            seed(db)
        finally:
            db.close()


def run_both(path: Path, legacy: Callable[[Any], Any], backend: Callable[[], Any],
             tables: Iterable[str] = AUTHORITY_TABLES) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """Keep a copy of the seeded store at ``path``, run ``legacy(db)`` on it with the legacy ``SessionDB``, restore
    the seed through the backup API (the live file keeps its WAL generation), run ``backend()``, and assert both runs
    left the same rows. Returns both outcomes."""
    tables = tuple(tables)
    seed = path.with_name("seed.db")
    with legacy_scope():
        from hermes_state import SessionDB

        copy_db(path, seed)
        db = SessionDB(db_path=path)
        try:
            before = outcome(lambda: legacy(db))
        finally:
            db.close()
    legacy_rows = dump(path, tables)
    copy_db(seed, path)
    after = outcome(backend)
    backend_rows = dump(path, tables)
    assert backend_rows == legacy_rows, row_diff(legacy_rows, backend_rows)
    return before, after


def row_diff(legacy: dict[str, Any], backend: dict[str, Any]) -> str:
    """The rows each table holds after only one of the two runs."""
    lines = []
    for table in sorted(set(legacy) | set(backend)):
        old, new = legacy.get(table, []), backend.get(table, [])
        if old != new:
            lines.append(f"{table}: legacy only {[r for r in old if r not in new]!r}; "
                         f"backend only {[r for r in new if r not in old]!r}")
    return "\n".join(lines)


@contextmanager
def counted_transactions(holder: Any) -> Iterator[dict[str, int]]:
    """Count the write transactions (``_execute_write``) and read contexts (``_read_ctx``) opened on ``holder``
    while the block runs; a write's lock replays count once."""
    counts = {"write": 0, "read": 0}
    write, read = holder._execute_write, holder._read_ctx

    def counted_write(*args: Any, **kwargs: Any) -> Any:
        counts["write"] += 1
        return write(*args, **kwargs)

    def counted_read(*args: Any, **kwargs: Any) -> Any:
        counts["read"] += 1
        return read(*args, **kwargs)

    holder._execute_write, holder._read_ctx = counted_write, counted_read
    try:
        yield counts
    finally:
        del holder._execute_write, holder._read_ctx
