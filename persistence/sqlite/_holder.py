"""One SQLite store holder per ``state.db`` file: the writer connection and its lock, ``BEGIN IMMEDIATE``
write transactions with jittered replay, the bounded ``mode=ro`` read pool, WAL setup, file and WAL-generation
guards, page-0 quarantine, structural-corruption quarantine, the asynchronous accounting queue and close.

Repositories run their SQL through ``_execute_write`` and ``_read_ctx``; ``persistence.sqlite._registry`` hands
out one holder per resolved path per process.
"""

import asyncio
import atexit
import contextlib
import hashlib
import logging
import os
import queue
import random
import sqlite3
import stat
import sys
import threading
import time
import uuid
import weakref
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar, cast

from hermes_constants import mkdir_under_hermes_home
from hermes_state_common import escape_like as _escape_like, stat_db_file_identity as _stat_db_file_identity
from hermes_state_holders import read_only_db_uri
from hermes_state_health import (
    STORAGE_CORRUPT, mark_storage_corrupt, note_storage_error, storage_corrupt_reason, storage_state,
)
from hermes_state_errors import (
    _DELETED_WAL_GENERATION_MSG, _DISK_IO_ERROR_MARKER, _STATE_DB_CORRUPT_MSG, _STATE_DB_GENERATION_KEY, _STATE_DB_REPLACED_MSG, DeletedWalGenerationError, SessionCompressionInProgressError, StateDbCorruptError, StateDbReplacedError, _is_no_more_rows, classify_persistence_error, is_malformed_db_error, is_malformed_schema_error, is_sqlite_lock_error,
)
from hermes_state_guard import (
    _STATE_DB_GUARD_BYPASS_ENV, _in_test_context, _is_production_state_db, _real_platform_state_root, _register_test_instance,
)
from hermes_state_readpool import _READ_POOL_MAX, _proc_fd_targets, _read_budget_for
import hermes_state_holders as _state_holders
import hermes_state_lockguard as _lockguard
from hermes_state_lockowners import log_write_lock_holders
from hermes_state_dbfile import (
    _connect_tracked_db, _fd_is_truly_unlinked, _prepare_connection_retirement, _read_sqlite_application_id, _stat_sqlite_sidecar_identity, _watched_sqlite_sidecar_paths, has_invalid_sqlite_header_preopen, quarantine_cross_process_lock, quarantine_invalid_state_db, RetiredGenerationCaptureError, capture_retired_wal_generation, refuse_deleted_wal_generation,
)
from hermes_state_wal import _on_disk_journal_mode, apply_wal_with_fallback
from hermes_state_repair import _claim_repair_attempt, preflight_db_writability, repair_state_db_schema
from persistence.core.errors import CompressionInProgress
from hermes_state_wal import apply_database_pragmas
from storage_user_copy import set_last_init_error

from ._schema import (
    _enter_fts_fail_open, _fts_table_probe, _init_schema, _is_fts_write_corruption_error,
    _try_incremental_merge_fts,
)
from ._schema_ddl import load_fts5_cjk_extension

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

T = TypeVar("T")

# Back off from read-only opens after one fails: not per query, but short enough that
# transient fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0
# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436): a WAL writer's checkpoint/
# reset/frame flush surfaces "disk I/O error" to a concurrent mode=ro reader for a millisecond-
# wide window — the ro connection cannot perform WAL recovery because recovery writes the -shm
# index, which mode=ro refuses. The writer closes the window on its own, so a few short retries
# make the open succeed instead of 500-ing the whole /api/sessions poll (or any other ro opener).
# Deliberately NOT for writable opens: a writer owns the transition, so an IOERR there is a real
# storage/fd problem. A persistent IOERR still exhausts the budget and propagates.
_READ_ONLY_IOERR_RETRY_ATTEMPTS, _READ_ONLY_IOERR_RETRY_BACKOFF_S = 3, 0.05
# SQLite busy handler budget for reads. Under DELETE (rollback-journal) mode a reader needs a SHARED
# lock, which every commit from another process blocks across its journal+db fsyncs; the writer
# connection's 1 s timeout exists for writes (they retry at application level) and starved readers
# into "database is locked" (dashboard 503s, `hermes sessions list` crashes) under a busy gateway.
_READ_BUSY_TIMEOUT_S = 5.0


# Live-DB guard knobs live HERE (not in ``_state.guard``): the hermetic conftest monkeypatches
# ``_STATE_DB_GUARD_BYPASS`` (``@pytest.mark.live_system_guard_bypass`` escape hatch)
# and ``_EXTRA_DENY_ROOTS`` (the pre-sandbox root, so custom-HERMES_HOME deployments are covered).
_STATE_DB_GUARD_BYPASS = False
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: tuple[Path, ...] = ()


def _ensure_test_isolation(db_path: Path) -> None:
    """Raise before any connection/mkdir/pragma/byte probe when a pytest-context process
    (env OR ancestry) resolves a production DB.

    Env alone is not enough: a child spawned with a rebuilt environment loses ``PYTEST_*`` and
    ``HERMES_HOME`` together, which is precisely the state in which it writes to production (#82770).
    """
    if _STATE_DB_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV) or not _in_test_context():
        return
    try:
        resolved = Path(db_path).expanduser().resolve()
    except Exception:  # health: allow BLE001 -- the guard checks only a path that resolves
        return
    roots = [r for r in (_real_platform_state_root(),) if r is not None]
    for extra in _STATE_DB_GUARD_EXTRA_DENY_ROOTS:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except Exception:  # health: allow BLE001 -- a deny root that does not resolve is skipped
            continue
    for root in roots:
        if _is_production_state_db(resolved, root):
            raise RuntimeError(
                "live-system guard: test attempted to open production "
                f"state.db at {resolved} (under real Hermes root {root}). "
                "Tests must run against a temporary HERMES_HOME — pass an "
                "explicit tmp db_path or let the hermetic conftest redirect "
                "HERMES_HOME. If this test genuinely needs the live database, mark it with "
                "@pytest.mark.live_system_guard_bypass — or, for a spawned "
                f"child process, export {_STATE_DB_GUARD_BYPASS_ENV}=1 in "
                "its environment."
            )


def _secure_state_db_files(db_path: Path, *, create_main: bool = False) -> None:
    """Create/tighten a writable state database and its sidecars to 0600.

    SQLite otherwise creates ``state.db``, ``-wal``, and ``-shm`` according to
    the process umask (commonly 0644 under 0022). Read-only holders
    never call this helper and remain observational.

    Existing files are tightened with ``chmod(2)`` on the path: opening the
    file and closing that descriptor would drop every POSIX ``fcntl`` lock the
    process holds on its inode — including the locks of an already-open SQLite
    connection to the same database. A lock-losing close in one process lets a
    sibling's connection take the shared-memory DMS exclusively at its own
    close, checkpoint, and unlink the sidecars while long-lived holders
    (gateway, desktop ``hermes serve``) keep using the deleted inodes.
    """
    if os.name == "nt":
        return

    main_path = db_path
    if create_main:
        # O_EXCL: only a brand-new inode gets a descriptor. Opening an existing
        # file here and closing it would drop this process's POSIX locks on it.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(main_path, flags, 0o600)
        except FileExistsError:
            pass
        except IsADirectoryError:
            # Not a database file at all; sqlite3.connect() raises the
            # canonical error for this, and a directory leaks no row data.
            return
        else:
            os.close(fd)

    for path in (
        main_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        # fchmod on an fd of a pre-existing file cannot be used here: close(fd)
        # would release this process's POSIX locks on that inode, stripping the
        # locks of any live SQLite connection to the same database. chmod(2)
        # never opens the file, so it leaves the lock state untouched.
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode):
            # Refuse a planted symlink exactly like O_NOFOLLOW would.
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        os.chmod(path, 0o600)


def _close_time_checkpoint_configurable() -> bool:
    """Whether this runtime can switch off SQLite's close-time checkpoint (Python 3.12+ ``setconfig``)."""
    return (getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None) is not None
            and hasattr(sqlite3.Connection, "setconfig"))


# Process-wide holder registry: long-lived in-process callers share ONE writer connection per resolved
# path via persistence.sqlite._registry.acquire(); one-shots open a StoreHolder and close it.
def _foreign_state_db_holders(db_path: Path) -> list[tuple[int, str]]:
    """Compatibility delegate to the state-holder authority."""
    return _state_holders.foreign_state_db_holders(db_path)


# The accounting queue applies each (session_id, delta) through the usage family's transaction,
# registered once with ``set_token_delta_applier``; the queue itself owns ordering, coalescing,
# the daemon writer, flush, the stopped-writer fallback and the exit drain.
_token_delta_applier: Optional[Callable[["StoreHolder", str, dict[str, Any]], None]] = None


def set_token_delta_applier(applier: Optional[Callable[["StoreHolder", str, dict[str, Any]], None]]) -> None:
    """Register the function that writes one queued usage delta (``update_token_counts`` semantics)."""
    global _token_delta_applier
    _token_delta_applier = applier


def _apply_token_delta(holder: "StoreHolder", session_id: str, kwargs: dict[str, Any]) -> None:
    if _token_delta_applier is None:
        raise RuntimeError("no usage delta applier is registered for the accounting queue")
    _token_delta_applier(holder, session_id, kwargs)


class StoreHolder:
    """SQLite-backed store for one ``state.db``: many reader threads, one writer (WAL)."""

    # ── Write-contention tuning ──
    # SQLite's deterministic busy handler convoys under many hermes processes: keep its
    # timeout short (1s) and retry with random jitter. Patience is TIME-based (a sibling
    # legitimately holds the lock for seconds: checkpoint at close, VACUUM, recovery, FTS
    # optimize); attempt-counted budgets destroyed turns on a healthy store. Transcript
    # writes (failure aborts the turn) get the long budget; observation-only activity
    # writes sit on the response-critical path and get a sub-second one.
    _WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
    # A live compression lock gets a short wait (compression publishes in seconds), but the lease
    # is a correctness boundary: a writer still locked out afterwards is refused.
    # Observation-only activity heartbeat/label writes (#76354 review S1): these run on (or adjacent to) the
    # response-critical path and must never wait out the full routine patience under contention. Sub-second
    # budget; a skipped write is retried naturally at the next heartbeat window.
    # A live compression lock gets its own, much shorter budget than the write lock. Compression publishes
    # in a couple of seconds, so a brief wait saves the overwhelming majority of concurrent turns (#75083).
    # It deliberately stays short: the lease is a correctness boundary, not just a busy signal (see
    # test_compression_lease_blocks_non_owner_but_allows_owner_flush), so a writer that is still locked out
    # after this budget must still be refused rather than allowed to land a stale turn in a session whose
    # compression is genuinely long-running or wedged.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150  # fast jitter for the first _SLOW_AFTER_S
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
    # PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Bounded FTS ``'merge'`` (ms of lock each) instead of ``'optimize'`` (9-18s per index on a 10GB
    # DB, longer than a writer's patience); up to _COMMANDS_PER_PASS per index, stopping on no-progress.
    _FTS_MERGE_EVERY_N_WRITES, _FTS_MERGE_MAX_PAGES_PER_INDEX, _FTS_MERGE_COMMANDS_PER_PASS = 1000, 500, 4
    # Accounting workers retire when idle so a bound-method target can't keep an abandoned holder alive.
    _TOKEN_WRITER_IDLE_SECONDS = 30.0
    # Set by the backend on the holder that replaces one the shutdown sweep closed under a live handle.
    _reopen_notice = False
    # Set by the registry's ``close_all`` (gateway shutdown) on every holder it closes.
    _closed_by_close_all = False

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    def _close_connection_quietly(conn: Optional[sqlite3.Connection]) -> None:
        """Close a partially initialized connection without masking its error."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close a SessionDB connection", exc_info=True)

    def _close_conn_logged(self, conn, label: str) -> None:
        """Close *conn*; a failing close leaks a tracked fd: logged at WARNING, never swallowed."""
        try:
            conn.close()
        except Exception as exc:  # health: allow BLE001 -- a failed close is logged at WARNING, never raised
            logger.warning("%s close failed for %s: %s", label, self.db_path, exc)

    def __init__(self, db_path: Path, read_only: bool = False, *, creation_site: Optional[str] = None,
                 read_patience_s: Optional[float] = None):
        self.db_path = db_path
        _ensure_test_isolation(self.db_path)  # before any connection/pragma/mkdir
        self.read_only = read_only
        # The lock wait a read-only holder opens its one connection with; the reads it serves inherit it.
        self._read_busy_timeout_s = _READ_BUSY_TIMEOUT_S if read_patience_s is None else read_patience_s
        # Keep only the opening call site, never a frame (which pins caller locals).
        self._creation_site = creation_site or "unknown"
        caller = None
        try:
            if creation_site is None:
                caller = sys._getframe(1)
                self._creation_site = (
                    f"{caller.f_globals.get('__name__', '?')}.{caller.f_code.co_name}:{caller.f_lineno}"
                )
        except Exception:  # health: allow BLE001 S110 -- the creation-site label must not stop the open
            pass  # Diagnostic metadata must not prevent opening the database.
        finally:
            del caller
        self._lock = threading.Lock()
        # Read-path split (WAL only): reads borrow from a BOUNDED read-only pool so they
        # never queue behind writer flushes on self._lock (see _read_ctx); unbounded
        # per-thread connections pinned fds for the process lifetime and hit EMFILE.
        self._read_pool: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=_READ_POOL_MAX)
        # Permits bound PEAK descriptors (the pool bounds only the idle set), shared per
        # DATABASE PATH; acquired non-blocking so a permitless reader degrades to the writer lock.
        # One permit per live read connection, held from before the open in _get_read_conn() until after the
        # close in _close_read_conn(). See _READ_POOL_MAX. Acquired non-blocking on purpose: a reader that
        # cannot get a permit must degrade to the writer lock, not queue here — blocking would convert fd
        # exhaustion into a stall, which is the same outage with a different stack trace. Permits are shared
        # per DATABASE PATH, not per instance: the descriptors they ration belong to the file, and one
        # process holds several store objects on the same state.db (#98573). See _PathReadBudget.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_permits = self._read_budget.permits
        self._read_conns_lock = threading.Lock()
        # Set when close() begins; an in-flight reader then closes its own connection
        # instead of re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # Read-open failure backoff is a TIMESTAMP, not a sticky bool: the likeliest trigger
        # is transient EMFILE, and a permanent flag would demote every reader forever.
        self._read_open_failed_at = 0.0
        self._wal_active, self._write_count = False, 0
        # File identity of the opened state.db, compared on every write so an out-of-band
        # replace cannot limp through in-place surgery (inode: mv/new-file; application_id: cp).
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_sidecar_identity: dict[str, tuple] = {}
        self._db_replaced = self._db_wal_generation_lost = False
        # Durable capture of a lost WAL generation (see _capture_retired_generation): once per handle.
        self._retired_generation_capture: Optional[Path] = None
        self._retired_capture_lock = threading.Lock()
        self._retire_connection: Optional[Callable[[Any], None]] = None
        self._connection_pinned = False  # one unmatched C reference taken at most once per handle
        self._wal_lock_guard: dict = {}  # lockguard.hold() record, see _open_writer
        self._db_corrupt, self._db_corrupt_reason = False, ""  # sticky quarantine (StateDbCorruptError)
        self._fts_usermerge_floor_applied = False  # one-shot usermerge-floor write guard
        self._fts_enabled = self._fts_stale = self._trigram_available = False
        # _fts_cjk_loaded: tokenizer on the writer connection; _fts_cjk_available: messages_fts_cjk
        # is queryable AND not marked stale.
        self._fts_cjk_loaded = self._fts_cjk_available = self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting; distinct from self._lock so enqueue/flush never contends with writes.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        # Opened via persistence.sqlite._registry.acquire(): close() releases a refcount instead.
        # Set True when this instance is opened via persistence.sqlite._registry.acquire(). Makes close() a no-op so the
        # registry (not individual callers) controls the connection lifecycle (#90837).
        self._shared_registry_owned = False
        initialization_complete = False
        try:
            if read_only:
                self._open_read_only()
            else:
                # Where SQLite's close-time checkpoint cannot be switched off, a lost-generation handle
                # is retired unclosed (see close()). Resolve that capability before opening a writer:
                # late cleanup must not import ctypes or look up it in a cleared module dictionary.
                if not _close_time_checkpoint_configurable():
                    self._retire_connection = _prepare_connection_retirement()
                self._open_writer()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Surface WHY via /resume and friends; callers keep their ``_session_db = None`` path.
            set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not initialization_complete:
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
            else:
                # Only a successfully opened handle owns a writer connection. Failed
                # construction must not leave a diagnostic member behind.
                self._read_budget.register(self)
                # Test-isolation runs only (gated inside the helper): register
                # for the suite-level leak sweep in tests/conftest.py.
                _register_test_instance(self)

    def _open_writer(self) -> None:
        """Writable open: preflight, zero-byte quarantine, connect + schema (one in-place repair of a
        malformed sqlite_master), generation stamp."""
        # Never materialize a deleted/archived named profile's home: a multiplexer or Desktop backend
        # still holding the profile's route would otherwise re-scaffold it on the next turn (#94590).
        mkdir_under_hermes_home(self.db_path.parent)
        try:
            # Serialize zero-byte check, quarantine, connect and schema commit so concurrent
            # openers don't race the absent-path -> schema-commit window.
            if not self.db_path.exists() or has_invalid_sqlite_header_preopen(self.db_path):
                # The directory must be writable before the lock file can be created in it; the
                # directory never vanishes mid-quarantine, so checking it unlocked is race-free.
                preflight_db_writability(self.db_path, db_label="state.db", directory_only=True)
                with quarantine_cross_process_lock(self.db_path) as lock_acquired:
                    if not lock_acquired:
                        logger.warning(
                            "startup quarantine lock for %s not acquired within 5s; proceeding",
                            self.db_path,
                        )
                    # Preflight under the lock: unlocked, it can list state.db just before a
                    # concurrent opener renames it aside, and os.access on the vanished path
                    # then reports a writable DB as "read-only for this user".
                    preflight_db_writability(self.db_path, db_label="state.db")
                    self._handle_quarantine_if_invalid(already_locked=lock_acquired)
                    self._connect_and_init_with_lock_patience()
            else:
                # Read-only file/sidecar preflight BEFORE the first connection: an actionable message
                # instead of an opaque "attempt to write a readonly database" from inside _init_schema.
                preflight_db_writability(self.db_path, db_label="state.db")
                self._handle_quarantine_if_invalid(already_locked=False)
                self._connect_and_init_with_lock_patience()
        except sqlite3.DatabaseError as exc:
            # A malformed schema fails on the very first statement (before _init_schema), so the
            # FTS-rebuild layer never sees it: repair sqlite_master in place (backup first), reopen once.
            if not is_malformed_schema_error(exc) or not _claim_repair_attempt(self.db_path):
                raise
            logger.error(
                "state.db schema is malformed (%s) — attempting automatic "
                "repair (a backup copy is made first).", exc,
            )
            self._close_connection_quietly(self._conn)
            if not repair_state_db_schema(self.db_path).get("repaired"):
                raise
            self._connect_and_init_with_lock_patience()
        # FTS optimization is OPT-IN (`hermes db optimize`); no background worker races session lifecycle.
        self._ensure_db_file_generation()
        if self._wal_active:
            # OFD copies of the two POSIX locks that keep a sibling's close from unlinking this WAL
            # generation: any in-process open()/close() of state.db or -shm cancels SQLite's own
            # (howtocorrupt §2.2); these survive it. Lifted in close().
            self._wal_lock_guard = _lockguard.hold(self.db_path)

    def _open_read_only(self) -> None:
        """Read-only attach for cross-profile aggregation: no schema init, NO write
        lock (sidebar polling never contends with that profile's backend); the DB
        must exist. FTS flags are probed with SELECTs only, and the connection is
        closed on ANY probe failure (malformed schema raises DatabaseError) so a
        leaked tracked connection cannot block the forensic backup the writable heal takes next."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
            try:
                self._conn = conn = self._connect_read_only(timeout=self._read_busy_timeout_s)
                try:
                    apply_database_pragmas(conn, db_label="state.db")
                    cursor = conn.cursor()
                    self._fts_enabled = _fts_table_probe(self, cursor, "messages_fts") is True
                    if self._fts_enabled:
                        self._trigram_available = (
                            _fts_table_probe(self, cursor, "messages_fts_trigram") is True
                        )
                except BaseException:
                    self._conn = None
                    self._close_connection_quietly(conn)
                    raise
                return
            except sqlite3.OperationalError as ioerr:
                # In-flight WAL checkpoint/reset/frame-flush on the writer side can surface
                # SQLITE_IOERR to a mode=ro reader (it can't do the -shm recovery the read
                # needs). Closes in milliseconds: retry a bounded number of times before
                # classifying the store as failed (#100436; see _READ_ONLY_IOERR_RETRY_ATTEMPTS).
                # A lock is NOT retried here: the connection already waited its read wait,
                # and a retry would multiply that wait on blocking callers (TUI, `hermes status`).
                transient = _DISK_IO_ERROR_MARKER in str(ioerr).lower()
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or not transient:
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)

    def _connect_read_only(self, timeout: float) -> sqlite3.Connection:
        """``mode=ro`` tracked connection with Row factory. check_same_thread=False: pooled connections
        are borrowed by whichever thread reads next; exclusive ownership is enforced by pool checkout."""
        conn = _connect_tracked_db(
            read_only_db_uri(self.db_path), tracking_path=self.db_path, uri=True,
            check_same_thread=False, timeout=timeout, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _handle_quarantine_if_invalid(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open; if quarantine failed,
        raise the clear message instead of opening the zeroed file."""
        if not (self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path)):
            return
        try:
            zsize = self.db_path.stat().st_size
        except OSError:
            zsize = -1
        qpath = quarantine_invalid_state_db(self.db_path, already_locked=already_locked)
        where = f"moved aside to {qpath}" if qpath else "left in place (it could not be moved aside)"
        msg = (
            f"state.db was empty or damaged ({zsize} bytes) and has been {where}; Hermes started with a "
            "fresh, empty session database. To bring old sessions back, run "
            f"`hermes sessions recover --source {qpath or self.db_path} --inspect-only`, or restore a "
            "snapshot with `/snapshot list` then `/snapshot restore <id>` (terminal `hermes` chat only)."
        )
        logger.error(msg)
        set_last_init_error(msg)
        if qpath is None and self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path):
            raise sqlite3.DatabaseError(msg)

    def _open_writer_conn(self) -> sqlite3.Connection:
        """Connect + WAL/pragma/tokenizer setup for a writer connection (no schema init). Short timeout:
        jittered application-level retry handles contention, not SQLite's busy handler;
        isolation_level=None: explicit BEGIN IMMEDIATE."""
        conn = _connect_tracked_db(
            str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            mode = apply_wal_with_fallback(conn, db_label="state.db")
            # "wal" is also the *assumed* mode when the on-disk probe was blocked by a concurrent opener
            # (#86515): the lock-free mode=ro read pool needs a confirmed WAL header, so confirm it here.
            # Unknown -> reads queue on the writer lock (slow but correct) instead of racing SQLITE_BUSY
            # on a file that may really be in rollback-journal mode.
            self._wal_active = mode == "wal" and _on_disk_journal_mode(conn) == "wal"
            # Existing WAL/SHM files may predate the main-file hardening;
            # normalize any sidecars that became visible during WAL setup.
            _secure_state_db_files(self.db_path)
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn, self.db_path.parent)
        except BaseException:
            self._close_connection_quietly(conn)
            raise
        return conn

    def _connect_and_init(self) -> None:
        # Refuse before sqlite3.connect (under the startup lock) so we cannot mint
        # a replacement WAL while a live writer still holds a deleted sidecar inode.
        refuse_deleted_wal_generation(self.db_path)
        # Create/tighten the main database before sqlite3.connect() so a
        # permissive process umask can never expose a fresh profile store.
        _secure_state_db_files(self.db_path, create_main=True)
        self._conn = self._open_writer_conn()
        _init_schema(self)

    def _connect_and_init_with_lock_patience(self) -> None:
        """Open + init, waiting out a sibling's write lock with jittered patience:
        _init_schema's DDL runs on a 1s-timeout connection, so a sibling's VACUUM
        or checkpoint used to fail the ENTIRE open and callers disabled
        persistence for the whole run. Non-lock errors propagate immediately."""
        # Lock contention during open: _init_schema's DDL/reconcile statements run on a 1s-timeout
        # connection with no retry, so a sibling process holding the write lock (VACUUM, TRUNCATE checkpoint
        # at close, a long FTS pass from an older still-running install) used to fail the ENTIRE open —
        # callers then disable persistence for the whole run ("database
        # is locked", #74478). The store is healthy; wait it out with the same jittered patience the write
        # path uses.
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        while True:
            try:
                self._connect_and_init()
                return
            except sqlite3.OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
                self._close_connection_quietly(self._conn)
                now = time.monotonic()
                if now >= deadline:
                    log_write_lock_holders(self.db_path, self._WRITE_PATIENCE_S)
                    raise
                jitter = random.uniform(self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S)
                time.sleep(min(jitter, max(deadline - now, 0.001)))

    # ── Read-path split ──

    def _get_read_conn(self) -> Optional[sqlite3.Connection]:
        """Open a fresh read-only connection, or None when unavailable (callers
        return it to self._read_pool). WAL only: WAL readers never block on the
        writer, so reads skip self._lock; under DELETE journal mode (NFS fallback)
        readers hit SQLITE_BUSY storms, so the legacy locked path stays. Autocommit
        reads see everything committed so far (read-your-writes for flush-then-search)."""
        if not self._wal_active or self.read_only:
            return None
        with self._read_conns_lock:
            failed_at = self._read_open_failed_at
            backing_off = failed_at and time.monotonic() - failed_at < _READ_OPEN_RETRY_SECONDS
            if self._read_conns_closed or backing_off:
                return None
        # Permit BEFORE the open: openers race for permits, not descriptors.
        if not self._read_budget.acquire(self):
            logger.debug(
                "read pool at capacity (%d) for %s; serving this read from the "
                "locked writer connection", _READ_POOL_MAX, self.db_path,
            )
            return None
        conn = None  # bound before the try so the handlers can close a half-open one
        try:
            conn = self._connect_read_only(timeout=_READ_BUSY_TIMEOUT_S)
            apply_database_pragmas(conn, db_label="state.db")
            if self._fts_cjk_loaded:  # registers in the connection, not the file: ro is fine
                load_fts5_cjk_extension(conn, self.db_path.parent)
        except BaseException as exc:
            # A half-open connection (open ok, extension load failed) is a live tracked descriptor,
            # the leak shape this pool exists to fix; a stranded permit would shrink the read
            # path by one slot forever. (Not _close_read_conn: callers release their own permit.)
            if conn is not None:
                self._close_conn_logged(conn, "partially-opened read conn")
            self._read_budget.release()
            if not isinstance(exc, sqlite3.Error):
                raise
            with self._read_conns_lock:
                self._read_open_failed_at = time.monotonic()
            logger.debug("read-only connection open failed for %s", self.db_path, exc_info=True)
            return None
        return conn

    def _evict_one_idle_read_conn(self) -> bool:
        """Close one idle pooled connection (a peer on the same file wants its permit); never a live one."""
        try:
            conn = self._read_pool.get_nowait()
        except queue.Empty:
            return False
        self._close_read_conn(conn)
        return True

    def _close_read_conn(self, conn) -> None:
        """Close a pooled read connection and release its permit even when the close fails (withholding
        it would narrow the read path forever). Over-releasing the BoundedSemaphore raises ValueError."""
        try:
            self._close_conn_logged(conn, "read-conn")
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection, opening on a miss; None when the read path is unavailable.
        A pool hit costs no permit (the connection already holds one)."""
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self, *, reopen: bool = True) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements: a pooled read-only
        connection with NO lock under WAL; otherwise (non-WAL, open failure,
        ceiling reached) the writer connection under self._lock — deliberate
        degradation: slower beats EMFILE, which the supervisor cannot see.
        ``reopen=False``: a read after ``close()`` raises instead of reopening the
        writer (an optional reader must not hold a home that is being deleted)."""
        self._log_reopen_notice("read")
        conn = self._checkout_read_conn()
        if conn is not None:
            try:
                yield conn
            finally:
                returned = False
                with self._read_conns_lock:
                    if not self._read_conns_closed:
                        try:
                            self._read_pool.put_nowait(conn)
                            returned = True
                        except queue.Full:
                            pass
                if not returned:
                    # close() drained the pool (or queue.Full: unreachable while
                    # permits == maxsize, load-bearing if they drift): surplus.
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:  # close() raced a still-unwinding reader
                if not reopen:
                    raise sqlite3.ProgrammingError(f"SessionDB for {self.db_path} is closed")
                self._reopen_after_close_locked(context="read")
            conn = cast(sqlite3.Connection, self._conn)
            if self._wal_active or self.read_only:  # WAL: no SHARED-lock wait; ro: opened with the read budget
                yield conn
                return
            # DELETE-mode writer connection: wait out a sibling's commit like a pooled reader would.
            previous_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.execute(f"PRAGMA busy_timeout={int(_READ_BUSY_TIMEOUT_S * 1000)}")
            try:
                yield conn
            finally:
                with suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout={int(previous_ms)}")

    @contextmanager
    def _read_ctx_with_timeout(self, busy_timeout_s: float) -> Iterator[sqlite3.Connection]:
        """``_read_ctx`` with the caller's own lock wait: the connection waits at most ``busy_timeout_s`` on a
        busy store inside the block and gets its previous budget back after it, for reads that wait less than
        ``_READ_BUSY_TIMEOUT_S``."""
        with self._read_ctx() as conn:
            previous_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_s * 1000)}")
            try:
                yield conn
            finally:
                with suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout={int(previous_ms)}")

    def _log_reopen_notice(self, context: str) -> None:
        """Log once, for the first read or write through a holder that replaces a swept one, the warning a reopen after
        ``close()`` logs: the handle's store was closed under it and is open again."""
        if not self._reopen_notice:
            return
        with self._read_conns_lock:
            pending, self._reopen_notice = self._reopen_notice, False
        if pending:
            logger.warning(
                "state.db connection for %s was closed while a %s was still in "
                "flight — reopening (teardown/worker race, #94736)", self.db_path, context,
            )

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer after ``close()`` raced a live caller (a teardown owner
        set ``_conn = None`` while a worker still had a transcript flush to land).
        Loud (WARNING) and bounded (only after an explicit close()). Caller holds
        ``self._lock``. No _init_schema: no DDL races with siblings during teardown."""
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again: a replaced file would be written through stale WAL/shm
        # assumptions; a quarantined handle must never hand a fresh connection to a damaged file.
        if self._db_corrupt and not (self._db_replaced or self._db_file_was_replaced()):
            raise self._corrupt_error(
                f"state.db connection for {self.db_path} is quarantined after "
                f"structural corruption; refusing to reopen for a {context} "
                "after close(). "
            )
        self._halt_if_db_generation_changed()
        logger.warning(
            "state.db connection for %s was closed while a %s was still in "
            "flight — reopening (teardown/worker race, #94736)", self.db_path, context,
        )
        try:
            self._conn = self._open_writer_conn()
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen failed: {exc}"
            ) from exc
        if self._wal_active:  # a reopened writer is a live generation holder like the first open
            self._wal_lock_guard = _lockguard.hold(self.db_path)

    def _execute_write(  # health: allow CC -- lock retry, rollback and repair share one deadline
        self, fn: Callable[[sqlite3.Connection], T], patience_s: Optional[float] = None,
    ) -> T:
        """Run *fn(conn)* inside BEGIN IMMEDIATE with jittered lock retry; commit
        is handled here (callers must not commit). Returns *fn*'s result.
        BEGIN IMMEDIATE takes the WAL write lock up front so contention surfaces
        immediately; on locked/busy the Python lock is released, a jitter slept,
        and the WHOLE callback retried — *fn* must stay idempotent under retry."""
        self._log_reopen_notice("write")
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        compression_deadline: Optional[float] = None  # set on the first compression-busy collision
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself (callback not run: nothing
        # replayed). Once fn has started, an IOERR leaves settlement unknown and must propagate.
        # The callback has not run at that point, so there is no durable effect to replay and the retry is
        # exactly-once safe (#99502's contract). Once the callback starts, an IOERR leaves the write's
        # settlement unknown and must propagate — this helper owns non-idempotent transcript/counter
        # mutations, not just idempotent UPSERTs.
        ioerr_begin_retried = False
        while True:
            self._raise_if_db_corrupt(storage=True)
            # NOTE: the replaced/generation live probe runs INSIDE the lock below,
            # not here. close() mutates _conn and _db_sidecar_identity under that
            # same lock, ending the WAL generation (SQLite unlinks the -wal/-shm
            # sidecars on a clean close). A lock-free probe that races close() can
            # observe the mid-teardown state — sidecars already unlinked while
            # _db_sidecar_identity is not yet cleared — and misclassify this
            # process's OWN clean close as an externally deleted generation,
            # raising a sticky DeletedWalGenerationError that permanently refuses
            # later writes (#105567). Inside the lock the probe only ever sees the
            # stable post-close state (identity cleared → adopt / reopen path).
            fn_started = False
            try:
                with self._lock:
                    self._raise_if_db_replaced()
                    if self._conn is None:  # close() raced this writer
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn_started = True
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:  # health: allow BLE001 S110 -- a failed rollback must not hide the error re-raised
                            pass
                        raise
                # Success — periodic best-effort checkpoint + FTS merge.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                if self._write_count % self._FTS_MERGE_EVERY_N_WRITES == 0:
                    _try_incremental_merge_fts(self)
                return result
            except (SessionCompressionInProgressError, CompressionInProgress):
                # Transient (see _COMPRESSION_BUSY_WAIT_S): a steer landing mid-compression must not abort.
                # A live foreign compression lock is transient: the compressor publishes in a couple of
                # seconds. Without any wait, a steer that lands mid-compression aborts the user's turn as
                # session_persistence_failed and sends the operator hunting disk space that was never the
                # problem (#75083). The budget is _COMPRESSION_BUSY_WAIT_S, not the write-lock patience: the
                # lease is a correctness boundary, so a writer still locked out after a short wait must be
                # refused rather than left to land a stale turn once a long-running or wedged compression
                # finally lets go.
                if compression_deadline is None:
                    compression_deadline = min(time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline)
                if self._sleep_before_write_retry(
                    compression_deadline, self._COMPRESSION_BUSY_WAIT_S
                ):
                    continue
                raise
            except sqlite3.Error as exc:
                # 'no more rows' is a transient engine error on contended WAL appends (some builds
                # raise it as InterfaceError, a sibling of DatabaseError): retry like locked/busy.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                err_msg = str(exc).lower()
                if isinstance(exc, sqlite3.OperationalError):
                    if is_sqlite_lock_error(exc):
                        if self._sleep_before_write_retry(deadline, patience_s):
                            continue
                        # Say what actually happened, not disk/permission damage. The holder goes to
                        # the log, not the message: classify_persistence_error() buckets by phrase and
                        # a holder's argv (a worktree named fix-corrupt-db) would flip the bucket.
                        log_write_lock_holders(self.db_path, patience_s)
                        raise sqlite3.OperationalError(
                            f"database is locked (another Hermes process held the "
                            f"state.db write lock for over {patience_s:.0f}s — "
                            "likely a long maintenance operation such as VACUUM, "
                            "a large WAL checkpoint, or an older pre-update "
                            "process; the database itself is healthy)"
                        ) from exc
                    if (
                        _DISK_IO_ERROR_MARKER in err_msg and not fn_started and not ioerr_begin_retried
                        and self._sleep_before_write_retry(deadline, patience_s)
                    ):
                        # Retry on the SAME connection: close()+reopen would cancel this process's
                        # POSIX locks for every sibling (howtocorrupt §2.2).
                        ioerr_begin_retried = True
                        continue
                    raise  # non-lock error, callback already ran, or patience exhausted
                if isinstance(exc, sqlite3.DatabaseError):
                    # An out-of-band replace surfaces as this same corruption class; in-file repair
                    # on a NEW generation amplifies the damage.
                    if (
                        "not a database" in err_msg or is_malformed_db_error(exc)
                        or _is_fts_write_corruption_error(exc)
                    ):
                        with self._lock:
                            self._raise_if_db_replaced()
                    # Corrupt FTS shadow tables fail every write via the sync triggers while canonical
                    # rows are intact: detach the derived indexes atomically and retry (never rebuild here).
                    if _enter_fts_fail_open(self, exc, deadline=deadline, patience_s=patience_s):
                        continue
                    # What survives both checks is structural damage: quarantine.
                    if self._is_structural_corruption_error(exc):
                        self._halt_db_corrupt(exc)
                raise

    def _write_sql(
        self, sql: str, params: Any = (), *, many: bool = False, patience_s: Optional[float] = None,
    ) -> None:
        """Run one INSERT/UPDATE/DELETE through ``_execute_write``."""
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)
        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(self, sql: str, params: Any = (), *, patience_s: Optional[float] = None) -> int:
        """Run one UPDATE/DELETE through ``_execute_write``; return rows changed
        (``SELECT changes()`` when the driver reports None / negative)."""
        def _do(conn):
            rowcount = conn.execute(sql, params).rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        return self._execute_write(_do, patience_s=patience_s)

    def _read_one(self, sql: str, params: Any = ()) -> Optional[sqlite3.Row]:
        """``fetchone()`` of one read-only statement via ``_read_ctx``."""
        return self._read_retrying_ioerr(lambda conn: conn.execute(sql, params).fetchone())

    def _read_all(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        """``fetchall()`` of one read-only statement via ``_read_ctx``."""
        return self._read_retrying_ioerr(lambda conn: conn.execute(sql, params).fetchall())

    def _read_retrying_ioerr(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run an idempotent SELECT through ``_read_ctx``, retrying a transient SQLITE_IOERR.

        A warm ``mode=ro`` pooled reader can hit the same millisecond-wide WAL transition window as a
        read-only OPEN (#100436) when its statement executes or steps: a sibling process's checkpoint /
        WAL reset / frame flush surfaces ``disk I/O error`` because a read-only connection cannot rewrite
        the -shm index (#100871, WSL2 ext4-on-vhdx, multi-process). The window closes on its own, so
        the statement is replayed on the SAME connection within the read-only IOERR budget -- never
        closed and reopened (close() cancels this process's POSIX locks for every sibling connection),
        never quarantined (busy is not broken). A persistent IOERR exhausts the budget and propagates."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
            try:
                with self._read_ctx() as conn:
                    return fn(conn)
            except sqlite3.OperationalError as exc:
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or _DISK_IO_ERROR_MARKER not in str(exc).lower():
                    note_storage_error(self.db_path, exc)
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)
            except sqlite3.DatabaseError as exc:
                # A reader is often the only observer (the sidebar poll on a store nobody is
                # writing to): publish structural damage so the list is not read as empty.
                note_storage_error(self.db_path, exc)
                raise

    def _ensure_db_file_generation(self) -> None:
        """Mint a once-per-file generation stamp (state_meta + application_id). First opener wins (INSERT
        OR IGNORE); application_id is written only while 0 so racers converge. PASSIVE checkpoint only.

        See #45383.
        """
        if self.read_only or self._conn is None:
            return
        token = uuid.uuid4().hex
        try:
            with self._lock:
                # Read first: the stamp is minted once per file, and a no-op INSERT OR IGNORE
                # still takes the write lock — under a sibling's transaction it blocked for the
                # busy timeout and the except below then dropped the token entirely. First
                # opener still wins via INSERT OR IGNORE; racers converge on the re-read.
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if not (row and row[0]):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                        (_STATE_DB_GENERATION_KEY, token),
                    )
                    row = self._conn.execute(
                        "SELECT value FROM state_meta WHERE key = ?",
                        (_STATE_DB_GENERATION_KEY,),
                    ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                current = int(pragma_row[0] or 0) if pragma_row else 0
                if current == 0:
                    current = (int(token[:8], 16) & 0x7FFFFFFF) or 1
                    self._conn.execute(f"PRAGMA application_id={current}")
                self._db_file_application_id = current
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
        except sqlite3.Error as exc:
            logger.debug("state.db generation stamp skipped: %s", exc)

    def _record_db_file_identity(self) -> None:
        """Snapshot inode plus the on-disk generation header when present."""
        self._db_file_identity = _stat_db_file_identity(self.db_path)
        self._db_sidecar_identity = _stat_sqlite_sidecar_identity(self.db_path)
        disk_id = _read_sqlite_application_id(self.db_path)
        if disk_id:
            self._db_file_application_id = disk_id
        elif self._conn is not None and not self._db_file_application_id:
            try:
                pragma_row = self._read_one("PRAGMA application_id")
            except sqlite3.Error:
                pragma_row = None
            if pragma_row and pragma_row[0]:
                self._db_file_application_id = int(pragma_row[0])

    def _db_file_was_replaced(self) -> bool:
        """True when the path no longer names the file this instance opened."""
        recorded = self._db_file_identity
        if recorded is not None and _stat_db_file_identity(self.db_path) != recorded:
            return True
        recorded_app = int(self._db_file_application_id or 0)
        if not recorded_app:
            return False
        # Header 0 = WAL not yet checkpointed, not a replace; a real replacement is nonzero.
        disk_app = _read_sqlite_application_id(self.db_path)
        return bool(disk_app and disk_app != recorded_app)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this handle opened is gone. Recorded
        generation: pure stat (no /proc walk on healthy writes). Empty identity
        (WAL appeared after open, or cleared by a clean close()): probe
        /proc/self/fd for deleted sidecars and adopt the current ones once clean."""
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            return any(
                _stat_db_file_identity(Path(base + suffix)) != ident for suffix, ident in recorded.items()
            )
        if not self._wal_active:  # no sidecar generation to lose; keep /proc off the hot path
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            try:
                for target, fd_path in _proc_fd_targets(os.getpid()):
                    canonical = _state_holders.canonical_sqlite_path(target)
                    if (" (deleted)" in target and canonical in watched
                            and _fd_is_truly_unlinked(fd_path, watched[canonical])):
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable): adopt the current sidecar generation.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_if_db_generation_changed(self) -> None:
        """Stop writes (logging once) when the file was replaced or its WAL/SHM generation
        is gone: never run in-file repair on a new generation, never keep committing on a
        split WAL. Both flags are sticky."""
        # A reopen resolves the PATH again — if the file at that path is no longer the one this instance
        # originally opened (out-of-band restore/cp/mv), reconnecting would write into the new generation
        # through stale WAL/shm assumptions (#89332). Refuse instead.
        if self._db_replaced or self._db_file_was_replaced():
            self._db_replaced = True
            self._disable_close_time_checkpoint()
            logger.error(_STATE_DB_REPLACED_MSG)
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._db_wal_generation_lost = True
            self._disable_close_time_checkpoint()
            try:
                self._capture_retired_generation("halt")
            except RetiredGenerationCaptureError as exc:
                logger.error(
                    "Could not capture the retired WAL generation of %s at halt: %s. close() retries "
                    "the capture and refuses to settle without it.", self.db_path, exc,
                )
            logger.error(_DELETED_WAL_GENERATION_MSG)
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _capture_retired_generation(self, trigger: str) -> Path:
        """Durably capture the lost WAL generation this handle still holds open, once per handle.

        The quarantine keeps the retired frames from being checkpointed under wrong page numbers,
        but they live only in an unlinked inode that dies with this process's last descriptor, and
        the canonical DeletedWalGenerationError remediation is to stop the writers. Capturing at the
        first halt (or at close(), whichever sees the loss first) makes "preserve" outlive the
        process. Raises RetiredGenerationCaptureError; nothing is mutated on failure."""
        with self._retired_capture_lock:
            if self._retired_generation_capture is not None:
                return self._retired_generation_capture
            artifact = capture_retired_wal_generation(
                self.db_path, sidecar_identity=dict(self._db_sidecar_identity or {}), trigger=trigger,
            )
            self._retired_generation_capture = artifact
        logger.warning(
            "Captured the retired WAL generation of %s at %s to %s; read its manifest.json before deciding "
            "whether those frames belong on top of the file now at the path.", self.db_path, trigger, artifact,
        )
        return artifact

    def _raise_if_db_replaced(self) -> None:
        """Sticky-flag fast path (no log spam on every write), then the live probe."""
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        self._halt_if_db_generation_changed()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance: canonical B-tree/schema/freelist damage,
        never repairable from the live write path."""
        return (
            isinstance(exc, sqlite3.DatabaseError)
            and not isinstance(exc, StateDbCorruptError)
            and not _is_fts_write_corruption_error(exc)
            and classify_persistence_error(exc) == "corrupt"
        )

    def _corrupt_error(self, prefix: str = "") -> StateDbCorruptError:
        """Build the quarantine error for this handle (message assembled once)."""
        return StateDbCorruptError(f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})")

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quarantine this handle and raise; never run in-file repair here."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        # Publish the profile-level state first: readiness, /api/status and the session list
        # endpoints read it, and every other handle in this process refuses writes on it.
        mark_storage_corrupt(self.db_path, exc)
        self._disable_close_time_checkpoint()
        logger.error(
            "state.db %s reported structural corruption outside the FTS "
            "indexes (%s); quarantining this handle: no further writes, no "
            "automatic reopen, no explicit WAL checkpoint at close. Stop the "
            "gateway and run `hermes sessions recover --source %s --inspect-only`.", self.db_path, exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            if getattr(exc, attr, None) is not None:
                setattr(err, attr, getattr(exc, attr))
        raise err from exc

    def _disable_close_time_checkpoint(self) -> bool:
        """Best-effort SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python 3.12+): sqlite3's
        close() otherwise runs the internal last-connection checkpoint that wrote
        the incident's pages under wrong page numbers (see StateDbCorruptError and
        the generation-loss halts).
        <3.12 has no setconfig, so a lost-generation handle is retired unclosed
        instead (see close()): closing its last descriptor could both run that
        checkpoint and discard committed data present only in an unlinked WAL."""
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if flag is None or setconfig is None:
            # Same predicate as _close_time_checkpoint_configurable() plus the per-instance
            # getattr: __init__ binds no retirement capability when either half is missing,
            # and close() must agree with that decision or the lost handle would neither
            # setconfig nor pin.
            return False
        try:
            setconfig(flag, True)
        except Exception:
            # No retention capability is bound on this runtime, so close() will let SQLite run the
            # checkpoint over the newer generation: say so where an operator can see it.
            logger.error(
                "Could not disable SQLite's close-time checkpoint on the quarantined handle for %s; "
                "closing it may checkpoint retired frames over the newer generation.",
                self.db_path, exc_info=True,
            )
            return False
        return True

    def _pin_connection(self, conn) -> None:
        """Retain the exact quarantined connection past GC and interpreter teardown (once per handle).

        Takes the connection as a parameter: callers are lock-held close paths, and the
        writer-conn thread-safety audit flags self._conn in functions outside `with self._lock`."""
        if not self._connection_pinned:
            self._retire_connection(conn)
            self._connection_pinned = True

    def _settle_lost_generation_locked(self) -> bool:
        """Capture the retired generation; return whether the handle must be retired unclosed.

        Where SQLite's close-time checkpoint cannot be switched off (no setconfig, Python < 3.12),
        sqlite3_close would write the retired frames over the newer generation, so the exact
        connection is retired unclosed instead. A failed capture leaves the handle open for a retry
        -- but the pin is taken FIRST on such a runtime: every production caller reaches close()
        through persistence.sqlite._registry.release_or_close, which swallows the error, so an interpreter
        exit before the retry must not be able to checkpoint the stale frames either."""
        self._db_wal_generation_lost = True
        retire_without_close = not self._disable_close_time_checkpoint() and self._retire_connection is not None
        try:
            artifact = self._capture_retired_generation("close")
        except RetiredGenerationCaptureError as exc:
            if retire_without_close:
                self._pin_connection(self._conn)
            logger.error(
                "Could not capture the retired WAL generation of %s at close: %s. The handle stays open "
                "and close() retries the capture; those frames are NOT yet preserved.", self.db_path, exc,
            )
            raise
        logger.warning(
            "Skipping the close-time WAL checkpoint for %s: this handle's WAL/SHM generation "
            "was deleted or replaced; the retired generation is captured at %s. Stop the other "
            "writers before reopening and inspect the capture before deciding its disposition.",
            self.db_path, artifact,
        )
        if retire_without_close:
            logger.warning(
                "Retaining the quarantined connection for %s unclosed: this runtime cannot "
                "switch off SQLite's close-time checkpoint.", self.db_path,
            )
        return retire_without_close

    def _raise_if_db_corrupt(self, *, storage: bool = False) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()
        if storage and storage_state(self.db_path) == STORAGE_CORRUPT:
            # Another handle in this process already saw structural damage on this file.
            # Quarantine this one before it touches SQLite; the error type is the same
            # StateDbCorruptError, so every transcript-diversion owner handles it unchanged.
            self._halt_db_corrupt(sqlite3.DatabaseError(
                "database disk image is malformed (reported earlier in this process: "
                f"{storage_corrupt_reason(self.db_path)})"))

    def _sleep_before_write_retry(self, deadline: float, patience_s: float) -> bool:
        """Sleep one jitter interval if the budget allows; True = retry, False = deadline passed. Small
        jitter for the first _WRITE_RETRY_SLOW_AFTER_S, then slow; never overshoots the deadline."""
        now = time.monotonic()
        if now >= deadline:
            return False
        slow = now - (deadline - patience_s) >= self._WRITE_RETRY_SLOW_AFTER_S
        jitter = random.uniform(*(
            (self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S) if slow
            else (self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
        ))
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    def _foreign_state_db_holders(self) -> list[tuple[int, str]]:
        """Foreign processes holding this DB or its WAL sidecars (see ``_state.holders``)."""
        return _foreign_state_db_holders(self.db_path)

    def _quarantine_reason(self) -> Optional[str]:
        """Why this handle must not checkpoint or run in-file repair, or None. A corrupted image has
        torn B-trees; a replaced file or a deleted/replaced WAL generation would checkpoint under
        wrong page numbers into the main DB -- the shutdown-time cause of #105670. Precedence note:
        close() evaluates generation loss BEFORE calling this (and skips it entirely when lost —
        a lost generation settles through the capture path, not the quarantine advisory), while
        the halt path checks replaced first."""
        if self._db_corrupt:
            return f"structural corruption ({self._db_corrupt_reason})"
        if self._db_replaced:
            return "a replaced state.db file"
        if self._db_wal_generation_lost:
            return "a deleted WAL generation (split-brain)"
        return None

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks writers;
        TRUNCATE corrupted B-trees on 65K+ page databases under exclusive-lock I/O pressure.

        Previous TRUNCATE strategy caused B-tree corruption on large databases (65K+ pages) due to the
        exclusive-lock I/O pressure from checkpointing thousands of frames at once (issue #45383).
        """
        if self._quarantine_reason() is not None:
            return
        try:
            with self._lock:
                if self._conn is None:
                    return  # closed underneath the timer: nothing to checkpoint, nothing to re-guard
                if self._wal_lock_guard:
                    _lockguard.hold(self.db_path, self._wal_lock_guard)  # a -shm minted after open
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result and result[1] > 0:
                    logger.debug("WAL checkpoint: %d/%d pages checkpointed", result[2], result[1])
        except Exception as exc:  # health: allow BLE001 -- a failed checkpoint is logged; the write has committed
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def __enter__(self) -> "StoreHolder":
        """``with StoreHolder(path) as holder:`` closes on exit; owners must release deterministically.

        Ownership of a holder should be released explicitly. Historically an instance with a started
        token writer pinned ITSELF (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked descriptors (#88033). The writer now
        retires after an idle window and the atexit hook holds only a weak reference, so abandoned handles
        are eventually collectible — but "eventually, after the idle window and a GC cycle" is not a release
        policy. Call sites owning a handle are still expected to close it deterministically (see the
        ownership comments in ``run_agent.py`` and ``tui_gateway/methods_session.py``).
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # never suppress the caller's exception

    def close(self):
        """Drain queued token deltas, then a PASSIVE checkpoint on writable handles
        (NOT TRUNCATE: a full WAL reset races the gateway's live writer, tearing
        B-tree pages). A registry-shared instance RELEASES one refcount instead.

        Drains queued token deltas first (the background writer needs the connection). Read-only connections
        never request a checkpoint. See #45383.
        When this instance is shared (opened via ``persistence.sqlite._registry.acquire``), ``close()`` RELEASES one
        refcount instead of tearing down the connection: the registry owns the lifecycle and only closes on
        the final release (#90837). This prevents one caller's close from tearing down the writer connection
        that other callers in the same process are still using — while still letting legacy ``close()`` call
        sites return their reference instead of leaking it.
        """
        if self._shared_registry_owned:
            from persistence.sqlite._registry import release
            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Closed flag first: an in-flight reader then closes its own connection.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while self._evict_one_idle_read_conn():
            pass
        with self._lock:
            if self._conn:
                generation_lost = not self.read_only and (
                    self._db_wal_generation_lost
                    or (bool(self._db_sidecar_identity) and self._wal_generation_was_lost())
                )
                # Loss is settled here, not at exit: the unlinked WAL inode dies with this process's
                # last descriptor (the capture raises and the handle stays open when it fails).
                retire_without_close = generation_lost and self._settle_lost_generation_locked()
                quarantine_reason = None if generation_lost else self._quarantine_reason()
                if quarantine_reason is not None:
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed %s. Take a snapshot of state.db, -wal and -shm "
                        "before restarting, then run `hermes sessions recover --source %s --inspect-only`.",
                        self.db_path, quarantine_reason, self.db_path,
                    )
                elif not self.read_only and not generation_lost:  # PASSIVE, not TRUNCATE (see docstring)
                    try:
                        # Every cron run_agent opens+closes a transient store, so a TRUNCATE here fires
                        # a full WAL reset many times/hour, racing the gateway's long-lived writer on large
                        # WAL databases and tearing hot B-tree pages -- the #45383 corruption this class's
                        # own periodic checkpoint was already made PASSIVE to avoid. TRUNCATE belongs only
                        # on a sole-opener/quiescent connection.
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:  # health: allow BLE001 -- a failed checkpoint at close is logged and the close goes on
                        logger.debug("WAL checkpoint (PASSIVE) at close failed: %s", exc)
                _lockguard.release(self._wal_lock_guard)  # before the close: see release()
                if retire_without_close:
                    self._pin_connection(self._conn)
                    self._conn = None
                else:
                    conn, self._conn = self._conn, None
                    self._close_connection_quietly(conn)
                    # Only a clean close ends the generation; retain the recorded
                    # identity when retiring an unsafe handle.
                    self._db_sidecar_identity = {}
        self._read_budget.unregister(self)  # idempotent: a never-registered (failed-init) handle is a no-op

    def __del__(self) -> None:
        """Safety net: close() if the caller forgot. Attribute access stays
        guarded: module teardown order is undefined."""
        if self.__dict__.get("_conn") is not None:
            try:
                self.close()
            except Exception:  # health: allow BLE001 S110 -- the finalizer never raises, even at interpreter exit
                pass

    # ── Async token accounting ──
    # queue_token_counts() is a deque append; a single-writer thread applies deltas in
    # order, coalescing consecutive deltas whose route fields are EQUAL (so the merged
    # UPDATE equals applying them sequentially). Exact readers call flush_token_counts().
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version", "billing_provider", "billing_base_url",
        "billing_mode", "source", "task",
    )

    # ── Meta key/value (scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key] on self._lock (not _read_ctx): fts_rebuild_step reads progress before its
        write transaction and a WAL reader would not see it."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert state_meta[key]; with ``cursor`` the write is inline (the caller already holds a
        transaction — nesting BEGIN IMMEDIATE would deadlock)."""
        sql = (
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        if cursor is not None:
            cursor.execute(sql, (key, value))
        else:
            self._write_sql(sql, (key, value))

    def list_meta_prefix(self, prefix: str) -> list[tuple[str, str]]:
        """``[(key, value), ...]`` for state_meta keys starting with the literal
        ``prefix`` (LIKE wildcards escaped) — e.g. ``loop:<session_id>`` rows."""
        if not prefix:
            return []
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (_escape_like(prefix) + "%",),
        )
        return [(row[0], row[1]) for row in rows]

    def queue_token_counts(self, session_id: str, **kwargs) -> None:
        """Enqueue a token/cost delta for the background writer (same kwargs as
        :meth:`update_token_counts`). After close() stopped the writer, falls back to the
        synchronous path and may raise."""
        if _token_delta_applier is None:
            from . import SQLiteBackend  # loads the repositories; the usage family registers the applier
        with self._token_queue_cond:
            thread = self._token_writer_thread
            writer_alive = thread is not None and thread.is_alive()
            writer_stopped = self._token_writer_stop and not writer_alive
            if not writer_stopped:
                self._token_queue.append((session_id, kwargs))
                if not writer_alive:
                    # Daemon so exit never hangs on accounting; the atexit hook drains
                    # leftovers. ``not is_alive()`` respawns a writer that died unexpectedly.
                    thread = threading.Thread(  # health: allow HX012 -- a daemon writer; the atexit hook drains its queue
                        target=self._token_writer_loop, name="session-db-token-writer", daemon=True)
                    self._token_writer_thread = thread
                    thread.start()
                    if self._token_atexit_hook is None:
                        self_ref = weakref.ref(self)

                        def _drain_at_exit() -> None:
                            db = self_ref()
                            if db is not None:
                                db._drain_token_queue_at_exit()

                        self._token_atexit_hook = _drain_at_exit
                        atexit.register(_drain_at_exit)
                self._token_queue_cond.notify_all()
        if writer_stopped:
            # close() ran: enqueueing would drop the delta silently, so apply inline.
            _apply_token_delta(self, session_id, kwargs)

    def _apply_claimed_batch(self, batch) -> None:
        """Apply a batch whose ``busy`` flag the caller already claimed, then release."""
        try:
            self._apply_token_batch(batch)
        finally:
            with self._token_queue_cond:
                self._token_writer_busy = False
                self._token_queue_cond.notify_all()

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Block until every queued token delta has been applied. False on timeout (callers
        then read totals stale by the queued deltas). Never raises."""
        # Lock-free fast path: reads queue-then-busy (see ordering notes below).
        if not self._token_queue and not self._token_writer_busy:
            return True
        batch = None
        with self._token_queue_cond:
            deadline = time.monotonic() + timeout
            while self._token_queue or self._token_writer_busy:
                # A live writer is authoritative even when stop-flagged: draining here would
                # race its in-flight batch and reorder deltas (breaking last-non-None-wins /
                # first-accounted-route / COALESCE-backfill fields). Only a dead writer lets
                # the caller take leftovers; a claimed busy means "wait".
                thread = self._token_writer_thread
                if (thread is None or not thread.is_alive()) and not self._token_writer_busy:
                    self._token_writer_busy = True
                    batch = list(self._token_queue)
                    self._token_queue.clear()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._token_queue_cond.wait(remaining)
        if batch:
            self._apply_claimed_batch(batch)
        return True

    def flush_for_read(self) -> bool:
        """Read-side token barrier. On a thread running an asyncio loop it never waits: the writer can sit
        behind another connection's write lock, and a gateway handler reading a session there would freeze
        every socket for up to the 5 s flush budget. That read sees totals stale by the queued deltas,
        which the next read picks up; write barriers keep calling ``flush_token_counts`` and still wait."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.flush_token_counts()
        return self.flush_token_counts(timeout=0.0)

    def _token_writer_loop(self) -> None:
        while True:
            with self._token_queue_cond:
                idle_deadline = time.monotonic() + self._TOKEN_WRITER_IDLE_SECONDS
                while not self._token_queue and not self._token_writer_stop:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        # Retire under the lock queue_token_counts() spawns under, so no
                        # delta strands behind an exiting worker.
                        self._token_writer_thread = None
                        return
                    self._token_queue_cond.wait(remaining)
                if not self._token_queue:
                    self._token_writer_thread = None
                    return  # stop requested and fully drained
                # busy BEFORE clearing the queue: flush's lock-free fast path must never see
                # "empty and idle" while a popped batch is unapplied.
                self._token_writer_busy = True
                batch = list(self._token_queue)
                self._token_queue.clear()
            self._apply_claimed_batch(batch)

    def _apply_token_batch(self, batch: list[tuple[str, dict[str, Any]]]) -> None:
        """Apply queued deltas in order, coalescing where safe. Never raises."""
        try:
            coalesced = self._coalesce_token_deltas(batch)
        except Exception as exc:  # health: allow BLE001 -- coalescing only optimizes; the raw batch applies
            # Coalescing must never kill the writer; the merge is only an optimization.
            logger.warning("async token accounting: coalesce failed, applying raw batch: %s", exc)
            coalesced = batch
        for session_id, kwargs in coalesced:
            try:
                _apply_token_delta(self, session_id, kwargs)
            except Exception as exc:  # health: allow BLE001 -- one failed delta is logged; the batch goes on
                # Accounting loss is logged, never raised into a turn.
                logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def _coalesce_token_deltas(self, batch: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
        """Merge adjacent incremental deltas with an identical route, so ordering across
        sessions and /model switches is preserved exactly. absolute=True never merges."""
        groups: list[tuple[Optional[tuple], str, dict[str, Any]]] = []
        for session_id, kwargs in batch:
            key = None
            if not kwargs.get("absolute"):
                key = (session_id, *(kwargs.get(f) for f in self._TOKEN_DELTA_ROUTE_FIELDS))
            if groups and key is not None and groups[-1][0] == key:
                merged = groups[-1][2]
                for f in self._TOKEN_DELTA_SUM_FIELDS:
                    merged[f] = merged.get(f, 0) + kwargs.get(f, 0)
                for f in self._TOKEN_DELTA_COST_FIELDS:
                    value = kwargs.get(f)
                    if value is not None:
                        # All-None runs stay None so COALESCE keeps the stored value.
                        merged[f] = (merged.get(f) or 0.0) + value
            else:
                groups.append((key, session_id, dict(kwargs)))
        return [(sid, kw) for _, sid, kw in groups]

    def _stop_token_writer(self, join_timeout: float = 10.0) -> None:
        """Stop the writer thread and drain remaining deltas. Never raises."""
        with self._token_queue_cond:
            self._token_writer_stop = True
            self._token_queue_cond.notify_all()
            thread = self._token_writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Writer stuck mid-apply: leave deltas unapplied rather than race it.
                logger.warning(
                    "async token accounting: writer did not stop within %.0fs; "
                    "%d queued delta(s) not persisted", join_timeout, len(self._token_queue))
                return
        # Writer gone: apply leftovers synchronously under the same busy protocol. Wait out
        # a flush caller-drain that already claimed busy — close() nulls the connection
        # right after this returns and must not yank it mid-batch.
        with self._token_queue_cond:
            deadline = time.monotonic() + join_timeout
            while self._token_writer_busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "async token accounting: concurrent drain did not "
                        "finish within %.0fs; %d queued delta(s) not persisted",
                        join_timeout, len(self._token_queue))
                    return
                self._token_queue_cond.wait(remaining)
            # busy BEFORE clearing the queue (same ordering as the writer loop).
            batch = list(self._token_queue)
            if batch:
                self._token_writer_busy = True
                self._token_queue.clear()
        if batch:
            self._apply_claimed_batch(batch)

    def _drain_token_queue_at_exit(self) -> None:
        with contextlib.suppress(Exception):  # never fatal at interpreter shutdown
            self._stop_token_writer()
