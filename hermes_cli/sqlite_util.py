"""Shared SQLite primitives for the small per-profile / board stores.

``open_db`` is the one connect + PRAGMA stack; ``transaction`` is the one commit-and-ALWAYS-close
shape. Every hand-rolled ``_connect``/``_transaction`` pair used to re-carry the #69567 fix (a
``with conn:`` only commits — it never closes, so each call leaked a connection and its WAL/SHM fds
until GC, exhausting ``RLIMIT_NOFILE`` on long-running gateways) and at least one copy missed it.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import Callable, Iterator


def open_db(
    path: Path | str,
    *,
    db_label: str,
    busy_timeout_ms: int = 5000,
    wal: bool = True,
    foreign_keys: bool = False,
    synchronous_full: bool = False,
    row_factory=sqlite3.Row,
    check_same_thread: bool = True,
    wal_lock_retries: int = 1,
    initialize: Callable[[sqlite3.Connection], None] | None = None,
) -> sqlite3.Connection:
    """Open ``path`` (parent created), apply the PRAGMA set, run ``initialize``; closed if anything raises.

    ``busy_timeout_ms`` is the single busy knob: it is passed as ``connect(timeout=)`` AND set as the
    explicit PRAGMA so it is observable. ``wal=True`` goes through ``apply_wal_with_fallback`` — the
    only journal-mode setter that carries the WAL-reset-bug gate, the network-FS silent-refusal
    fallback and the never-live-downgrade invariant; a raw ``PRAGMA journal_mode=WAL`` bypasses all
    three. Only the transient ``database is locked`` from that pragma is retried (``wal_lock_retries``):
    a first opener initializing a shared DB can make it ignore the busy timeout, notably on Windows.
    """
    from sqlite_support import apply_wal_with_fallback

    path = Path(path)
    # A profile whose persistence provider keeps its file stores too is served from there (no file, no PRAGMAs).
    from persistence_assembly import open_compat_store

    compat = open_compat_store(path, row_factory=row_factory)
    if compat is not None:
        try:
            if initialize is not None:
                initialize(compat)
        except BaseException:
            compat.close()
            raise
        return compat
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resolved at call time: fd-leak tests patch ``sqlite3.connect`` through the caller's module.
    conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000, check_same_thread=check_same_thread)
    try:
        conn.row_factory = row_factory
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        if wal:
            for attempt in range(wal_lock_retries):
                try:
                    apply_wal_with_fallback(conn, db_label=db_label)
                    break
                except sqlite3.OperationalError as exc:
                    if str(exc).lower() != "database is locked" or attempt + 1 == wal_lock_retries:
                        raise
                    time.sleep(0.01 * (2**attempt))
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        if synchronous_full:
            conn.execute("PRAGMA synchronous=FULL")
        if initialize is not None:
            initialize(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def connect_store(path: Path | str, *, db_label: str, wal: bool = True, row_factory=None, **connect_kwargs) -> sqlite3.Connection:
    """A store that opens its own connection (not through :func:`open_db`): the provider's compatible connection when
    the profile keeps its file stores there, else ``sqlite3.connect(path, **connect_kwargs)`` with the shared WAL
    fallback. A ``:memory:`` path is always SQLite."""
    from sqlite_support import apply_wal_with_fallback

    if str(path) != ":memory:":
        from persistence_assembly import open_compat_store

        compat = open_compat_store(Path(path), row_factory=row_factory)
        if compat is not None:
            return compat
    conn = sqlite3.connect(path, **connect_kwargs)
    try:
        conn.row_factory = row_factory
        if wal and str(path) != ":memory:":
            apply_wal_with_fallback(conn, db_label=db_label)
    except BaseException:
        conn.close()
        raise
    return conn


def connect_read_only(path: Path | str, *, row_factory=None, **connect_kwargs) -> sqlite3.Connection:
    """A read-only connection on the store at ``path``: the provider's (refusing writes) when the profile keeps
    its file stores there, else ``mode=ro`` on the file."""
    from persistence_assembly import open_compat_store

    compat = open_compat_store(Path(path), row_factory=row_factory)
    if compat is not None:
        compat.read_only = True
        return compat
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, **connect_kwargs)
    conn.row_factory = row_factory
    return conn


def copy_store_contents(src: sqlite3.Connection, dst: sqlite3.Connection) -> dict[str, int]:
    """Copy every table and index of ``src`` into the empty ``dst`` (either side a SQLite file or a provider's
    store) from ``src``'s own ``sqlite_master`` text, rows in rowid order; one transaction on ``dst``, which the
    caller commits. Returns the rows copied per table."""
    if dst.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchone()[0]:
        raise sqlite3.OperationalError("refusing to copy into a store that already holds tables")
    objects = src.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
                          "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END").fetchall()
    if not dst.in_transaction:
        dst.execute("BEGIN")
    counts: dict[str, int] = {}
    for kind, _name, ddl in objects:
        if kind in ("table", "index"):
            dst.execute(ddl)
    for kind, name, _ddl in objects:
        if kind != "table":
            continue
        columns = [row[1] for row in src.execute(f'PRAGMA table_info("{name}")')]
        names = ", ".join(f'"{c}"' for c in columns)
        try:
            rows = src.execute(f'SELECT {names} FROM "{name}" ORDER BY rowid').fetchall()
        except sqlite3.OperationalError:  # a WITHOUT ROWID table
            rows = src.execute(f'SELECT {names} FROM "{name}"').fetchall()
        if rows:
            dst.executemany(f'INSERT INTO "{name}" ({names}) VALUES ({", ".join("?" for _ in columns)})',
                            [tuple(r) for r in rows])
        counts[name] = len(rows)
    return counts


def is_sqlite_file(conn: sqlite3.Connection) -> bool:
    """Whether ``conn`` is a real SQLite connection (whose file and sidecars exist on disk), not a provider's."""
    return isinstance(conn, sqlite3.Connection)


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on error, and ALWAYS close ``conn`` (see the module docstring)."""
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        with conn:
            yield conn
    finally:
        conn.close()


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``, idempotent across races: True when this call added
    it, False on the ``duplicate column name`` a concurrent migrator caused.

    ``column`` is the human-readable name for the call site; ``ddl`` carries the actual definition. See
    #21708.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """An IMMEDIATE write transaction on a long-lived connection (stays open). The explicit ROLLBACK is
    guarded so a SQLite auto-rollback (no transaction left under EIO / contention / corruption) cannot
    shadow the original error."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        conn.execute("COMMIT")
