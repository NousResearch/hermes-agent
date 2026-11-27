"""The two files that hold the hosted-room tables of one store handle, and the transactions every hosted-room verb runs.

``placement='profile'`` is the bound profile's ``state.db``: its writes go through the registered holder's ``_execute_write``, which always begins ``BEGIN IMMEDIATE``, with the hosted-room patience, and its reads through the holder's read path with the hosted-room lock wait. ``placement='shared'`` is the root ``shared-state.db`` of the install the bound store's folder belongs to; each write opens the file's connection (WAL, foreign keys, 10 s busy wait) under one lock per file per process and begins as main's transaction for the same verb begins: ``BEGIN IMMEDIATE`` when ``immediate``, else Python's implicit begin before the first data-changing statement; each read opens a connection that changes neither journal mode nor schema on a ready file. On a read-only store handle every write refuses with ``PermanentStoreError`` before any file is opened, and a shared read opens the file ``mode=ro``: a missing file is ``StoreUnavailable``, one whose steps are not current is refused, and neither is created or migrated.

A ``HostedFiles`` carries the schema steps its verbs run when they open the file: each step is a ``(ready, initialize)`` pair, and before the transaction every step whose ``ready`` is false runs ``initialize`` in its own ``BEGIN IMMEDIATE``, as main's hosted-room connections do. A read on a file whose steps are current writes nothing.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Callable, TypeVar

from persistence import PermanentStoreError, StoreContention
from persistence.gateway.hosted_rooms import HostedPlacement
from session_rules.hosted_rooms import _JOURNAL_MODE_LOCK_RETRIES
from sqlite_support import open_db, read_only_db_uri

from .._backend import SQLiteStore

__all__ = ["HostedFiles", "SchemaStep", "file_identity", "shared_state_path"]

T = TypeVar("T")

# Every hosted-room connection waits up to 10 s for the file's locks.
HOSTED_PATIENCE_S = 10.0
SHARED_STATE_NAME = "shared-state.db"
ATTACHMENT_DIR_NAME = "hosted-room-attachments"

# (ready(conn, path) -> bool, initialize(conn, path) -> None); initialize runs inside BEGIN IMMEDIATE.
SchemaStep = tuple[Callable[[sqlite3.Connection, Path], bool], Callable[[sqlite3.Connection, Path], None]]

_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def shared_state_path(profile_dir: Path) -> Path:
    """The install's root ``shared-state.db`` for ``profile_dir``: beside ``profiles/`` for a named profile."""
    home = Path(profile_dir)
    return (home.parent.parent if home.parent.name == "profiles" else home) / SHARED_STATE_NAME


def file_identity(path: Path) -> tuple[str, int, int] | None:
    """Identity of the database file, so a recreated file is set up again; None while it does not exist."""
    try:
        info = path.stat()
    except OSError:
        return None
    return str(path.resolve()), info.st_dev, info.st_ino


def _file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _file_locks_guard:
        return _file_locks.setdefault(key, threading.Lock())


def _run_steps(conn: sqlite3.Connection, path: Path, steps: tuple[SchemaStep, ...]) -> None:
    for ready, initialize in steps:
        if not ready(conn, path):
            try:
                conn.execute("BEGIN IMMEDIATE")
                initialize(conn, path)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


def _connect_shared(path: Path, steps: tuple[SchemaStep, ...]) -> sqlite3.Connection:
    """The write connection: WAL, foreign keys, 10 s busy wait, then each step that is not current in its own ``BEGIN IMMEDIATE``."""
    return open_db(path, db_label="shared-state.db (hosted_rooms)", busy_timeout_ms=int(HOSTED_PATIENCE_S * 1000),
                   foreign_keys=True, wal_lock_retries=_JOURNAL_MODE_LOCK_RETRIES,
                   initialize=lambda conn: _run_steps(conn, path, steps))


def _open_reader(path: Path) -> sqlite3.Connection:
    return open_db(path, db_label="shared-state.db", busy_timeout_ms=int(HOSTED_PATIENCE_S * 1000), wal=False,
                   foreign_keys=True)


def _open_shared_reader(path: Path, steps: tuple[SchemaStep, ...]) -> sqlite3.Connection:
    """The read connection: no journal or schema writes on a ready file; a missing or unready file gets its steps through the write connection first."""
    if not path.is_file():
        _connect_shared(path, steps).close()
    conn = _open_reader(path)
    if not all(ready(conn, path) for ready, _ in steps):
        conn.close()
        _connect_shared(path, steps).close()
        conn = _open_reader(path)
    return conn


def _open_read_only_reader(path: Path, steps: tuple[SchemaStep, ...]) -> sqlite3.Connection:
    """The read connection of a read-only handle: ``mode=ro``, so a missing file stays missing; a file whose steps are not current is refused, never set up."""
    conn = sqlite3.connect(read_only_db_uri(path), uri=True, timeout=HOSTED_PATIENCE_S)
    try:
        conn.row_factory = sqlite3.Row
        if not all(ready(conn, path) for ready, _ in steps):
            raise PermanentStoreError(f"{path} is not set up for this version; a read-only store handle never migrates it")
    except BaseException:
        conn.close()
        raise
    return conn


class HostedFiles:
    """The hosted-room files of one store handle and the write and read transactions on them, after ``steps``."""

    def __init__(self, store: SQLiteStore, steps: tuple[SchemaStep, ...] = ()) -> None:
        self._store = store
        self._steps = steps

    def path(self, placement: HostedPlacement) -> Path:
        """The database file ``placement`` names; raises ``RuntimeError`` once the store's backend has closed."""
        self._store.require_open()
        if placement == "profile":
            return Path(self._store.store_path)
        if placement == "shared":
            return shared_state_path(Path(self._store.store_path).parent)
        raise ValueError(f"unknown hosted-room placement {placement!r}")

    def attachment_root(self, placement: HostedPlacement) -> Path:
        """The attachment directory beside the file."""
        return self.path(placement).parent / ATTACHMENT_DIR_NAME

    def refuse_read_only(self) -> None:
        """Raise ``PermanentStoreError`` on a read-only handle: hosted-room writes and first-touch setup need a writable store."""
        if self._store.read_only:
            raise PermanentStoreError(f"hosted-room write refused: the store handle for {self._store.store_path} is read-only")

    def _prepare_profile(self) -> None:
        """Run each step that is not current on the profile's ``state.db``, each in its own holder write."""
        holder = self._store.holder()
        for ready, initialize in self._steps:
            with holder._read_ctx_with_timeout(HOSTED_PATIENCE_S) as conn:
                current = ready(conn, holder.db_path)
            if not current:
                holder._execute_write(lambda conn: initialize(conn, holder.db_path), patience_s=HOSTED_PATIENCE_S)

    def _hosted_write(self, placement: HostedPlacement, fn: Callable[[sqlite3.Connection], T], *, immediate: bool) -> T:
        """Run ``fn(conn)`` in one transaction on the placement's file after the steps, beginning ``BEGIN IMMEDIATE`` when ``immediate`` (always on ``state.db``); commits on success, rolls back on any exception. On ``shared-state.db`` it first waits up to the hosted-room patience for this process's other write to the file, then raises ``StoreContention``. A read-only handle refuses before opening anything."""
        self.refuse_read_only()
        if placement == "profile":
            self._prepare_profile()
            return self._store.holder()._execute_write(fn, patience_s=HOSTED_PATIENCE_S)
        path = self.path(placement)
        lock = _file_lock(path)
        if not lock.acquire(timeout=HOSTED_PATIENCE_S):
            raise StoreContention(f"another write in this process held {path} for {HOSTED_PATIENCE_S:g} s")
        try:
            with closing(_connect_shared(path, self._steps)) as conn:
                if immediate:
                    conn.execute("BEGIN IMMEDIATE")
                with conn:
                    return fn(conn)
        finally:
            lock.release()

    def _hosted_read(self, placement: HostedPlacement, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(conn)`` on a read connection of the placement's file after the steps, waiting up to the hosted-room lock wait. A read-only handle opens the shared file ``mode=ro`` and needs it set up already."""
        if placement == "profile":
            self._prepare_profile()
            with self._store.holder()._read_ctx_with_timeout(HOSTED_PATIENCE_S) as conn:
                return fn(conn)
        open_reader = _open_read_only_reader if self._store.read_only else _open_shared_reader
        with closing(open_reader(self.path(placement), self._steps)) as conn:
            return fn(conn)
