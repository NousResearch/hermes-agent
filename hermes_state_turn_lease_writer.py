"""Short, bounded turn-lease writes independent of transcript/FTS serialization."""

import sqlite3
import time

from hermes_state_errors import (
    DeletedWalGenerationError, StateDbCorruptError, StateDbReplacedError,
    _DELETED_WAL_GENERATION_MSG, _STATE_DB_REPLACED_MSG, is_sqlite_lock_error,
)


def _generation_lost(db, sidecars):
    from hermes_state_dbfile import _stat_sqlite_sidecar_identity

    if sidecars is None:
        return db._wal_generation_was_lost()
    current = _stat_sqlite_sidecar_identity(db.db_path)
    return any(current.get(suffix) != identity for suffix, identity in sidecars.items())


def _check_generation(db, sidecars=None):
    # Do not touch the transcript connection or mutate its quarantine state from this worker.
    from hermes_state_health import STORAGE_CORRUPT, storage_state

    if db._db_corrupt or storage_state(db.db_path) == STORAGE_CORRUPT:
        raise db._corrupt_error()
    if db._db_replaced or db._db_file_was_replaced():
        raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
    if db._db_wal_generation_lost or _generation_lost(db, sidecars):
        raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)


def _write(db, conn, fn, deadline, patience, sidecars):
    while True:
        _check_generation(db, sidecars)
        remaining = max(0.0, deadline - time.monotonic())
        conn.execute(f"PRAGMA busy_timeout={min(1000, int(remaining * 1000))}")
        try:
            conn.execute("BEGIN IMMEDIATE")
            _check_generation(db, sidecars)
            result = fn(conn)
            conn.commit()
            return result
        except BaseException as exc:
            conn.rollback()
            if db._is_structural_corruption_error(exc):
                from hermes_state_health import mark_storage_corrupt
                mark_storage_corrupt(db.db_path, exc)
                raise StateDbCorruptError(str(exc)) from exc
            if not is_sqlite_lock_error(exc) or not db._sleep_before_write_retry(deadline, patience):
                raise


def _close(db, conn, guard, sidecars):
    from hermes_state import _lockguard
    from hermes_state_health import STORAGE_CORRUPT, storage_state

    _lockguard.release(guard)
    if conn is not None:
        # Older Python cannot suppress a close-time checkpoint. Keep unsafe handles alive
        # through interpreter shutdown using the capability bound at SessionDB open.
        unsafe = (db._db_corrupt or storage_state(db.db_path) == STORAGE_CORRUPT
                  or _generation_lost(db, sidecars) or db._db_file_was_replaced())
        if db._retire_connection is not None and unsafe:
            db._retire_connection(conn)
        else:
            conn.close()


def _open(db, deadline):
    from hermes_state import _connect_tracked_db
    from hermes_state import _lockguard

    if not db._read_conns_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise sqlite3.OperationalError("database is locked (turn lease upkeep busy)")
    conn, guard, sidecars = None, {}, None
    try:
        if db._read_conns_closed:
            raise sqlite3.ProgrammingError("SessionDB is closed; cannot maintain a turn lease")
        _check_generation(db)
        sidecars = dict(db._db_sidecar_identity)
        # mode=rw refuses to create a replacement state.db if its path disappears.
        conn = _connect_tracked_db(
            db.db_path.resolve().as_uri() + "?mode=rw", tracking_path=db.db_path,
            uri=True, timeout=0, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        if hasattr(conn, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE"):
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
        conn.execute("PRAGMA wal_autocheckpoint=0")
        guard = _lockguard.hold(db.db_path) if db._wal_active else {}
        _check_generation(db, sidecars)
        return conn, guard, sidecars
    except BaseException:
        _close(db, conn, guard, sidecars)
        raise
    finally:
        db._read_conns_lock.release()


def execute_lease_write(db, fn, patience_s=None):
    """Only upkeep uses this path; acquisition and transcript mutations retain their ordering.

    The reader lifecycle mutex protects only connection open, never a SQLite busy wait. The
    connection owns its generation snapshot/guard even if the main handle subsequently closes.
    No schema initialization, FTS work, journal-mode change or periodic checkpoint is performed.
    """
    if db.read_only:
        raise sqlite3.OperationalError("attempt to write a readonly database")
    patience = max(0.0, float(db._WRITE_PATIENCE_S if patience_s is None else patience_s))
    deadline = time.monotonic() + patience
    conn, guard, sidecars = _open(db, deadline)
    try:
        return _write(db, conn, fn, deadline, patience, sidecars)
    finally:
        _close(db, conn, guard, sidecars)
