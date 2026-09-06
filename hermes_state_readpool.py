"""Read-connection budgeting for SessionDB's WAL read path: per-file and
process-wide permit ceilings plus a descriptor-headroom gate, so N handles on
one state.db cannot walk the process into EMFILE. Idle pooled connections keep
their permit and are reclaimable across peers on the same path."""

import errno
import logging
import os
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover
    from hermes_state import SessionDB

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# Ceiling on read-only connections ALIVE at once against one database FILE (idle
# pooled + checked out, over every SessionDB on that file). One constant for both
# the pool maxsize and the permit count: a LifoQueue only caps how many are
# *returned*, and EMFILE is a peak-instant condition, so a connection holds a
# permit for its whole lifetime; once permits are gone reads degrade to the
# locked writer connection — slower, but not a wedge the supervisor can't see.
# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436). A WAL database being actively written
# (checkpoint, WAL reset/truncate, frame flush) can surface "disk I/O error" to a concurrent ``mode=ro``
# reader in a millisecond-wide transition window: the read-only connection cannot perform the WAL recovery a
# read through a stale or mid-update -shm file needs, because recovery requires writing the -shm index,
# which mode=ro refuses. The window closes on its own (the writer finishes the transition), so a bounded
# number of short retries makes the open succeed instead of 500-ing the whole /api/sessions poll (or any
# other read-only opener). Deliberately NOT attempted on writable opens: a writer owns the transition, so an
# IOERR there means a real storage/fd problem.
_READ_POOL_MAX = 8

# Ceiling ALIVE in this PROCESS across every state.db (a multiplexed gateway
# opens one per profile); three profiles' worth, then readers degrade likewise.
# _READ_POOL_MAX bounds one file. A multiplexed gateway serves N profiles from one process and each profile
# has its OWN state.db, so a per-file ceiling still lets the descriptor cost grow with the profile count —
# the same shape as the per-instance bug, one level out (#98573). Past it, readers on the (N+1)th file
# degrade to their writer connection instead of opening descriptors, which is the same trade _READ_POOL_MAX
# makes and for the same reason: a slow read path is recoverable, a process-wide EMFILE is not.
_READ_POOL_PROCESS_MAX = 24

# Warn past this many SessionDB handles on one file in one process (diagnostic:
# writer connections cannot be rationed the way read connections can).
_HANDLES_PER_PATH_WARN = 4

# Descriptors kept in reserve for everything that is NOT this module (httpx
# sockets, terminal pipes, log files): the EMFILE SQLite pushes over surfaces elsewhere.
# The ceilings above bound Hermes's SQLite descriptors, which is only ever part of the fd table. The #98573
# report is exactly that case: ~20 state.db descriptors were not the whole 256, they were the share that
# pushed httpx and terminal pipes over, and the EMFILE surfaced in tools/terminal_tool.py rather than here.
# So the read pool also yields when the PROCESS is close to its limit, whatever is consuming it.
_FD_HEADROOM_RESERVE = 64

# The fd count is a directory listing; cache it briefly so a read burst isn't a
# syscall per query (staleness lets through at most the ceiling's worth of opens).
_FD_USAGE_CACHE_SECONDS = 0.25

_process_read_permits = threading.BoundedSemaphore(_READ_POOL_PROCESS_MAX)

# Read opens refused for low descriptor headroom — the only visible signal the
# guard fires. Guarded by _read_budgets_lock.
_read_open_denied_fd_headroom = 0

_fd_usage_lock = threading.Lock()
_fd_usage_cache: "tuple[float, Optional[int]]" = (0.0, None)


def _proc_fd_targets(pid: int) -> Iterator[str]:
    """readlink() of every entry in /proc/<pid>/fd (unreadable links skipped).
    Raises OSError when the fd directory itself cannot be listed."""
    fd_dir = f"/proc/{pid}/fd"
    for fd in os.listdir(fd_dir):
        try:
            yield os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue


def _open_fd_count() -> Optional[int]:
    """Open descriptors in THIS process; None when unmeasurable (Windows: no fd
    dir, correctly inert); -1 when the probe itself hit EMFILE/ENFILE (no headroom)."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError as exc:
            if exc.errno in (errno.EMFILE, errno.ENFILE):
                return -1
    return None


def _fd_soft_limit() -> Optional[int]:
    """The process's soft RLIMIT_NOFILE, or None when there is no usable one."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    if soft in (resource.RLIM_INFINITY, -1):
        return None
    return int(soft)


def _fd_headroom_ok() -> bool:
    """Can the process spare a descriptor for a new read connection? Fails OPEN
    when unmeasurable (refusing every read would be a self-inflicted convoy);
    fails CLOSED only on evidence (measured shortfall or a starved probe)."""
    soft = _fd_soft_limit()
    if soft is None:
        return True
    global _fd_usage_cache
    now = time.monotonic()
    with _fd_usage_lock:
        stamp, cached = _fd_usage_cache
        fresh = cached is not None and (now - stamp) < _FD_USAGE_CACHE_SECONDS
    if not fresh:
        cached = _open_fd_count()
        with _fd_usage_lock:
            _fd_usage_cache = (now, cached)
    if cached is None:
        return True
    return cached >= 0 and (soft - cached) > _FD_HEADROOM_RESERVE


def _reclaim_idle_read_conn_anywhere() -> bool:
    """Close one idle read connection on ANY path: the process ceiling is shared
    across files, so a quiet profile must not hold descriptors a busy one needs."""
    with _read_budgets_lock:
        budgets = list(_read_budgets.values())
    return any(budget.reclaim_idle() for budget in budgets)


class _PathReadBudget:
    """Read-connection permits for ONE database file, shared process-wide
    (per-instance semaphores let N SessionDBs peak at N x (1 + MAX)). An idle
    pooled connection keeps its permit, so a permit miss first reclaims an IDLE
    connection from a peer on the same path.

    ``_READ_POOL_MAX`` used to be enforced by a ``BoundedSemaphore`` owned by each SessionDB, which bounded
    the wrong noun: the descriptors are spent on a *file*, so N SessionDB objects on one state.db each got
    their own allowance and peak scaled as ``N x (1 + _READ_POOL_MAX)``. A long-lived gateway holds at least
    two (``SessionStore`` and ``GatewayRunner`` open independent handles per profile path) and the count
    grows with the profile count, which is how a healthy process walked into EMFILE — #98573.
    """

    def __init__(self) -> None:
        self.permits = threading.BoundedSemaphore(_READ_POOL_MAX)
        self._lock = threading.Lock()
        # Weak: a SessionDB dropped without close() must not pin peers' budget.
        self._members: "weakref.WeakSet[SessionDB]" = weakref.WeakSet()
        self._duplicate_handles_warned = False

    def register(self, db: "SessionDB") -> None:
        with self._lock:
            self._members.add(db)
            handles = len(self._members)
            warn = (handles > _HANDLES_PER_PATH_WARN and not self._duplicate_handles_warned)
            if warn:
                self._duplicate_handles_warned = True
        if warn:
            # Writer connections cannot be capped; the only bound is not opening
            # redundant handles, so make the duplicate visible before it's an incident.
            logger.warning(
                # The only real bound on writers is not opening redundant handles in the first place (which
                # is what GatewayRunner borrowing SessionStore's handle does, #98573), so the next duplicate
                # should be visible before it becomes an incident rather than inferred from an lsof after
                # one.
                "%d live SessionDB handles on %s in this process; each holds "
                "its own writer connection (read connections are capped at %d "
                "for the file). A long-lived process should share one handle per path.",
                handles, db.db_path, _READ_POOL_MAX,
            )

    def acquire(self, requester: "SessionDB") -> bool:
        """Take a permit for a new read connection, or refuse (caller degrades to the
        locked writer connection). Gates, broadest first: fd headroom, process
        ceiling, this file's ceiling."""
        if not _fd_headroom_ok():
            global _read_open_denied_fd_headroom
            with _read_budgets_lock:
                _read_open_denied_fd_headroom += 1
            return False
        if not self._acquire_process_permit():
            return False
        if self._acquire_path_permit(requester):
            return True
        _process_read_permits.release()
        return False

    def release(self) -> None:
        """Return one connection's permits. Pairs with a successful acquire()."""
        self.permits.release()
        _process_read_permits.release()

    def _acquire_process_permit(self) -> bool:
        # Another thread may take a freed permit first: legitimate loss, no looping.
        return _process_read_permits.acquire(blocking=False) or (
            _reclaim_idle_read_conn_anywhere() and _process_read_permits.acquire(blocking=False)
        )

    def _acquire_path_permit(self, requester: "SessionDB") -> bool:
        return self.permits.acquire(blocking=False) or (
            self.reclaim_idle(exclude=requester) and self.permits.acquire(blocking=False)
        )

    def reclaim_idle(self, exclude: "Optional[SessionDB]" = None) -> bool:
        """Close one idle pooled connection held by a member; True if one went.
        Its release() returns both permits, so both ceilings reclaim through here."""
        with self._lock:
            members = [db for db in self._members if db is not exclude]
        return any(member._evict_one_idle_read_conn() for member in members)


# canonical db path -> permits for that file. Weak values: the budget lives only
# while some SessionDB on the path holds it, so tmp_path churn can't grow this.
_read_budgets: "weakref.WeakValueDictionary[str, _PathReadBudget]" = (weakref.WeakValueDictionary())
_read_budgets_lock = threading.Lock()


def _read_budget_key(db_path) -> str:
    """Canonicalise a db path so two spellings share one budget."""
    try:
        return str(Path(db_path).resolve())
    except OSError:
        return str(db_path)


def _read_budget_for(db_path) -> _PathReadBudget:
    key = _read_budget_key(db_path)
    with _read_budgets_lock:
        budget = _read_budgets.get(key)
        if budget is None:
            budget = _PathReadBudget()
            _read_budgets[key] = budget
        return budget
