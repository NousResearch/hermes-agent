"""SessionDB write phases: local mutex patience and post-commit maintenance."""
from contextlib import contextmanager
import logging
import sqlite3
import sys
import threading
import time

logger = logging.getLogger("hermes_state")

# One holder line per database per window: under a real stall every queued writer times out at
# once, and N identical lines would bury the one that names the holder.
_OWNER_LOG_INTERVAL_S = 30.0
_owner_log_lock = threading.Lock()
_owner_logged_at: dict[str, float] = {}


class WriterMutexTimeout(sqlite3.OperationalError):
    """The local writer is busy, not a damaged database or a provider failure.

    Carries SQLITE_BUSY so every caller that retries or defers on lock contention
    (``is_sqlite_lock_error``, ``classify_persistence_error`` -> ``"locked"``) treats it the same
    way; the type and the ``in-process`` wording tell it apart from a foreign SQLite writer, which
    surfaces as the ``another Hermes process held the state.db write lock`` error instead."""

    def __init__(self, message: str):
        super().__init__(message)
        self.sqlite_errorcode = sqlite3.SQLITE_BUSY
        self.sqlite_errorname = "SQLITE_BUSY"


_WRAPPER_NAMES = frozenset({"_write_sql", "_write_rowcount"})


def write_site(fn):
    """Where this write comes from, as code objects (no frame: locals are never retained): the
    caller of ``_execute_write`` (or of the generic ``_write_sql``/``_write_rowcount`` wrappers)
    and the write callback. Formatted only when a waiter has to name the holder."""
    caller = None
    try:
        frame = sys._getframe(2)  # 0: here, 1: _execute_write, 2: its caller
        if frame.f_code.co_name in _WRAPPER_NAMES and frame.f_back is not None:
            frame = frame.f_back
        caller = frame.f_code
    except Exception:
        pass
    finally:
        frame = None
    return caller, getattr(fn, "__code__", None)


def _describe_code(code) -> str:
    module = (getattr(code, "co_filename", "") or "?").rsplit("/", 1)[-1].removesuffix(".py")
    return f"{module}:{getattr(code, 'co_qualname', None) or getattr(code, 'co_name', '?')}"


def describe_site(site) -> str:
    """``module:qualname`` of the recorded caller (and callback): code locations, never arguments."""
    if not isinstance(site, tuple):
        return "?"
    caller, callback = site
    parts = [_describe_code(code) for code in (caller, callback) if code is not None]
    return " -> ".join(parts) or "?"


@contextmanager
def writer_mutex_until(lock, deadline, *, db=None, site=None):
    """Do not let an observation or lease renewal queue forever behind another writer.

    With ``db``, the holder (thread name, write callback, acquire time) is recorded on
    ``db._writer_owner`` while held, so a refused waiter can name it. Only code locations are
    recorded: no SQL, parameters or frame locals."""
    if not lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        if db is not None:
            _log_writer_owner(db, site)
        raise WriterMutexTimeout(
            "database is locked (in-process state.db writer mutex exceeded write patience)"
        )
    if db is not None:
        db._writer_owner = (threading.current_thread().name, site, time.monotonic())
    try:
        yield
    finally:
        if db is not None:
            db._writer_owner = None
        lock.release()


def _log_writer_owner(db, waiter_site) -> None:
    """One rate-limited WARNING naming who holds the writer mutex a waiter just gave up on."""
    key = str(getattr(db, "db_path", "?"))
    now = time.monotonic()
    with _owner_log_lock:
        if now - _owner_logged_at.get(key, -_OWNER_LOG_INTERVAL_S) < _OWNER_LOG_INTERVAL_S:
            return
        _owner_logged_at[key] = now
    owner = getattr(db, "_writer_owner", None)
    if owner is None:
        holder = "an untracked holder (a read served on the writer connection, a checkpoint or maintenance)"
    else:
        name, site, since = owner
        holder = f"thread {name!r} in {describe_site(site)} (held for {now - since:.1f}s)"
    logger.warning(
        "%s: in-process writer mutex unavailable within write patience for %s on thread %r; held by %s. "
        "Not a foreign SQLite writer; the refused write is retried or deferred by its caller. "
        "Logged at most once per %.0fs per database.",
        getattr(getattr(db, "db_path", None), "name", key), describe_site(waiter_site),
        threading.current_thread().name, holder, _OWNER_LOG_INTERVAL_S,
    )


def maintain_after_commit(db):
    """Count committed writes and run due maintenance after releasing the writer mutex."""
    db._write_count += 1
    if db._write_count % db._CHECKPOINT_EVERY_N_WRITES == 0:
        db._try_wal_checkpoint()
    if db._write_count % db._FTS_MERGE_EVERY_N_WRITES == 0:
        db._try_incremental_merge_fts()
