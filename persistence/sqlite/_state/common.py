"""Constants and helpers the SQLite backend and the legacy store share: title provenance, end reasons, the
session preview shape, LIKE escaping, id chunking, file identity and the cross-process FTS rebuild lock."""

import contextlib
import errno
import json
import logging
import os
import sys
import time
from typing import Any

from process_identity import persistent_record_pidns_checkable, pid_namespace_id
from session_rules.turns import SKILL_EXCERPT_JOINT, describe_skill_invocation


# Persisted title provenance: automatic display labels are not user-selected identities.
TITLE_SOURCE_DERIVED = "derived"
TITLE_SOURCE_LLM = "llm"
TITLE_SOURCE_USER = "user"


_PREVIEW_MAX_CHARS = 60


def escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with ``ESCAPE '\\'``.
    ``_`` is common in branch names/titles/paths and a substring match must not silently widen."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _shape_preview(raw: Any) -> str:
    """Turn a ``_preview_raw`` column into the short preview callers show."""
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    described = describe_skill_invocation(text)
    text = described if described is not None else text.split(SKILL_EXCERPT_JOINT)[0]
    return text[:_PREVIEW_MAX_CHARS] + "..." if len(text) > _PREVIEW_MAX_CHARS else text


# 'session_switch' creates no child row today, but pre-marker DBs hold legacy reset children whose parent
# ended that way.  Must stay identical to the recovery fence in find_latest_gateway_session_for_peer.
_RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")
# Deliberate conversation boundaries: the reset set plus CLI /new, which ends the predecessor as
# 'new_session' (hermes_cli/cli_session_mixin.py) without a reset child row.  A compression rotation must
# never heal one of these (#106459); tools/session_search_tool.py derives its fresh-reset set from it.
_BOUNDARY_END_REASONS = frozenset(_RESET_END_REASONS) | {"new_session"}

# Accidental end reasons recovery treats as resumable (website/docs/developer-guide/gateway-session-lifecycle.md); single source of truth for
# recovery SQL and the session store's RECOVERABLE_END_REASONS.  superseded_by_resume = sentinel-parked runtime replaced
# by a fresh session.resume; startup_orphan_reap = dead-gateway sweep, same class as ws_orphan_reap but kept
# distinct for forensics.
_RECOVERABLE_END_REASONS = ("agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap")

# End reasons written by AUTOMATIC cleanup (shutdown, orphan reapers, idle/LRU eviction), not a deliberate
# conversation boundary: "some runtime went away", so a writer that can prove liveness (e.g. a compression
# rotation holding the lease) may clear it.  Recoverable set plus the TUI gateway's automatic reasons.
# Superset of the recoverable set: those are already resumable accidents; the extra TUI reasons are the same
# accident class but were historically only known to tui_gateway's _AUTOMATIC_SESSION_END_REASONS. See
# #88197.
_AUTOMATIC_END_REASONS = frozenset(_RECOVERABLE_END_REASONS) | {
    "tui_shutdown", "ws_disconnect", "idle_timeout", "lru_evict"}


def is_automatic_end_reason(reason) -> bool:
    """True when *reason* is an automatic-cleanup end stamp; compression-liveness sites must call this.

    Single owner of the "accidental vs deliberate end" predicate — every compression-liveness site must call
    this instead of re-implementing the reason taxonomy (#88197, never-patch-predicates).
    """
    return isinstance(reason, str) and reason in _AUTOMATIC_END_REASONS


def stat_db_file_identity(path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` for *path*, or None.  st_ino=0 (Windows, some network FS) would false-positive
    every replaced-file check, so it counts as unknown."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if st.st_dev and st.st_ino else None


def _ended_by_compression(row) -> bool:
    return row is not None and row["ended_at"] is not None and row["end_reason"] == "compression"


def _placeholders(items) -> str:
    """``?,?,?`` for one bound parameter per element of *items* (a sequence or an int count)."""
    return ",".join("?" for _ in range(items if isinstance(items, int) else len(items)))


# Ids per ``IN (?,...)`` list: SQLite caps bound parameters at SQLITE_MAX_VARIABLE_NUMBER (999 on builds
# < 3.32, 32766 after); a bulk prune of a cron-heavy store bound tens of thousands of ids into one list and
# died with "too many SQL variables". Every IN-list over session ids goes through ``_id_chunks``.
_SQL_IN_CHUNK = 900


def _id_chunks(ids, size: int = _SQL_IN_CHUNK):
    """Yield *ids* (any iterable) as lists of at most *size* elements."""
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


# Durable diagnostic for stale FTS recovery blocked across process restarts.
FTS_REBUILD_DEFERRAL_KEY = "fts_rebuild_deferral"


# Cross-process full-FTS-rebuild admission (single authority).  Several processes share one state.db and a
# structural rebuild (FTS5 'rebuild' or `_recover_stale_fts`'s drop/recreate) must run in ONE at a time —
# concurrent rebuilds corrupted state.db in production.  Gates `rebuild_fts()`, `_rebuild_fts_indexes()`,
# `_recover_stale_fts()`; the chunked backfill (`fts_rebuild_step`) is deliberately NOT routed through it (it
# claims progress under SQLite transaction authority).  Mirrors `repair._cross_process_repair_lock`:
# portable (msvcrt/flock), bounded wait, FAIL CLOSED; orphaned-fd holders (see `_acquire_db_flock`) are broken
# only when provably dead, indeterminate liveness defers.  `<db>.fts_rebuild.lock` is distinct from
# `<db>.repair.lock` (offline schema surgery, minutes in VACUUM).

# ── Cross-process full-FTS-rebuild admission (single authority) ────────────── Several independent Hermes
# processes routinely share one state.db (gateway service, the Desktop app's `hermes serve` backend,
# interactive CLI sessions, the TUI slash worker). A full structural FTS rebuild — the FTS5 'rebuild'
# command or the drop/recreate script in `_recover_stale_fts` — must only ever run in ONE of them at a time:
# two concurrent rebuilds collide on write and have structurally corrupted state.db in production (PR
# #93200; the 2026-08-15 / 2026-08-23 incidents and issues #89293 / #90950). This is the single admission
# authority for every full structural rebuild entry point: the search store's `rebuild_fts()`,
# `_schema._rebuild_fts_indexes()` (via `_init_schema`), and
# `_schema._recover_stale_fts()`. The chunked deferred backfill (`fts_rebuild_step`) is
# deliberately NOT routed through it — it claims progress under `_execute_write`'s SQLite transaction
# authority and is intentionally multi-process. Semantics mirror `repair._cross_process_repair_lock`
# (the schema- surgery authority): portable (msvcrt on Windows, flock elsewhere), bounded wait, and FAIL
# CLOSED — a caller that cannot acquire the lock must NOT rebuild. The kernel drops both lock types when the
# holder dies — UNLESS a forked child inherited the lock fd (flock rides the open file description, which
# fork() duplicates), in which case the orphaned descriptor holds the lock forever (issue #100108).
# `_acquire_db_flock` therefore records the holder's pid + start time under the lock and, when the recorded
# holder is provably dead, breaks the orphaned lock by unlinking and retaking it on a fresh inode;
# indeterminate liveness still defers. It lives here so the schema and the search store share it without
# an import cycle. The lock file is `<db>.fts_rebuild.lock`, distinct from
# `<db>.repair.lock`: schema surgery runs on an EXCLUSIVE offline connection and can legitimately take
# minutes in VACUUM, while runtime rebuilds run on live connections. The timeout is sized for a full
# 'rebuild' of both indexes on a large DB.
logger = logging.getLogger("hermes_state")

_FTS_REBUILD_LOCK_TIMEOUT_SECONDS = 120.0
_FTS_REBUILD_LOCK_POLL_SECONDS = 0.1
_IS_WINDOWS = sys.platform == "win32"
# Post-break re-acquire budget: the fresh inode is contended only by live processes — never the full timeout.
_LOCK_BREAK_REACQUIRE_SECONDS = 5.0

# "Another process holds the lock": flock → EWOULDBLOCK/EAGAIN, msvcrt.locking → EACCES (EDEADLK when its retry
# gives up).  Anything else (ESTALE, ENOTSUP, ENOLCK, EIO) is a persistent failure polling cannot fix.
_LOCK_CONTENTION_ERRNOS = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK}


def is_advisory_lock_contention(exc: BaseException) -> bool:
    """True when *exc* means another process holds the lock; on any other ``OSError`` fail closed at once."""
    return isinstance(exc, BlockingIOError) or (isinstance(exc, OSError) and exc.errno in _LOCK_CONTENTION_ERRNOS)


def _proc_start_ticks(pid: int):
    """Kernel start time of *pid* (field 22 of ``/proc/<pid>/stat``; with the PID it identifies a process
    uniquely).  None off Linux or on any failure — callers must treat None as unknowable and FAIL CLOSED."""
    try:
        # comm (field 2) may contain spaces/parens; split after the LAST ')'.
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return int(fh.read().rsplit(b")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_lock_holder_record(handle):
    """Best-effort parse of the holder metadata JSON in a lock file."""
    try:
        handle.seek(0)
        raw = handle.read(4096)
        record = json.loads(raw.decode("utf-8", "replace")) if raw else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _rewrite_lock_file(handle, payload: bytes) -> None:
    """Best-effort truncate-and-write of *payload* at offset 0."""
    with contextlib.suppress(OSError, ValueError):
        handle.seek(0)
        handle.truncate()
        if payload:
            handle.write(payload)
        handle.flush()


def _write_lock_holder_record(handle) -> None:
    """Record this process as holder (best effort) so timed-out contenders can tell an orphaned-fd holder
    from a live wedged one.

    Written under the flock so contenders that time out can tell an orphaned-fd holder (recorded process
    dead, flock inherited by a forked child — issue #100108) from a live wedged holder.  ``pidns``: see
    ``process_identity``.
    """
    record = {"pid": os.getpid(), "pidns": pid_namespace_id(),
              "start_ticks": _proc_start_ticks(os.getpid()), "acquired_at": time.time()}
    _rewrite_lock_file(handle, json.dumps(record, sort_keys=True).encode("utf-8"))


def _clear_lock_holder_record(handle) -> None:
    """Erase holder metadata before a normal release: a surviving record means ABNORMAL exit (break allowed)."""
    _rewrite_lock_file(handle, b"")


def _lock_holder_provably_dead(record) -> bool:
    """True ONLY when the recorded holder is provably dead or PID-recycled.  Anything indeterminate
    (no/malformed record, PID owned by another user, /proc unavailable, foreign PID namespace)
    is False: FAIL CLOSED and defer.  Unstamped records keep probing: see ``process_identity``."""
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if not persistent_record_pidns_checkable(record.get("pidns")):
        return False  # foreign namespace: a local reading is not proof — defer
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # PermissionError et al.: PID exists (or unknowable) — closed
    recorded_ticks = record.get("start_ticks")
    if recorded_ticks is None:
        return False
    current_ticks = _proc_start_ticks(pid)
    return current_ticks is not None and current_ticks != recorded_ticks  # different start time: PID recycled


def _acquire_db_flock(lock_path, handle, timeout_seconds, poll_seconds, description):
    """Bounded POSIX flock acquire with orphaned-holder break.  Returns ``(acquired, handle)``; *handle* may
    have been re-opened and the caller closes whichever comes back.  *acquired*: True, False (a holder kept
    the lock past the deadline) or None (non-contention ``OSError``, already logged: treat as not acquired
    without the held-by-another-process warning).  ``flock`` rides the open file DESCRIPTION, which ``fork()``
    duplicates, so a holder that forks then dies leaves the lock held forever; when the acquirer is provably
    dead the file is unlinked and retaken on a fresh inode (the orphan's flock excludes nobody).  Every
    acquire verifies its inode still names *lock_path*, so a racer on a dead inode retries.

    A holder that forks (multiprocessing worker, daemonized helper) and then dies leaves the flock held by a
    child that will never release it — the kernel's holder-death release never triggers, and every contender
    defers forever. Indeterminate liveness always defers (fail closed). See #100108.
    """
    import fcntl
    deadline = time.monotonic() + timeout_seconds
    broke_lock = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if not is_advisory_lock_contention(exc):
                logger.warning("Could not acquire %s %s (%s) — deferring rather than "
                               "waiting out the %.0fs holder timeout on a non-contention error.",
                               description, lock_path, exc, timeout_seconds)
                return None, handle
            if time.monotonic() < deadline:
                time.sleep(poll_seconds)
                continue
            if broke_lock:
                return False, handle
            record = _read_lock_holder_record(handle)
            if not _lock_holder_provably_dead(record):
                return False, handle
            logger.warning("%s %s is held by an orphaned file descriptor (recorded holder pid %s is dead — a "
                           "forked child inherited the lock fd); breaking the stale lock and retaking it on a "
                           "fresh file.", description, lock_path, (record or {}).get("pid"))
            try:
                os.unlink(lock_path)
                handle.close()
                handle = open(lock_path, "a+b")
            except OSError as exc:
                logger.warning("Could not break stale %s %s (%s) — deferring.", description, lock_path, exc)
                return False, handle
            broke_lock = True
            deadline = time.monotonic() + _LOCK_BREAK_REACQUIRE_SECONDS
            continue
        # A breaker may have replaced the file while we waited; a lock on a dead inode excludes nobody.
        try:
            fd_stat, path_stat = os.fstat(handle.fileno()), os.stat(lock_path)
            same_file = fd_stat.st_dev == path_stat.st_dev and fd_stat.st_ino == path_stat.st_ino
        except OSError:
            same_file = False
        if same_file:
            _write_lock_holder_record(handle)
            return True, handle
        try:
            handle.close()
            handle = open(lock_path, "a+b")
        except OSError:
            return False, handle
        if time.monotonic() >= deadline:
            return False, handle


def _describe_lock_holder(record) -> str:
    """Human-readable holder identity for deferral warnings."""
    if not isinstance(record, dict) or "pid" not in record:
        return "unknown (no holder record; pre-fix writer or non-Hermes)"
    age = ""
    with contextlib.suppress(TypeError, ValueError):
        if record.get("acquired_at") is not None:
            age = f", acquired {time.time() - float(record['acquired_at']):.0f}s ago"
    return f"pid {record.get('pid')}{age}"


def _acquire_msvcrt_lock(lock_path, handle, timeout):
    """Windows counterpart of ``_acquire_db_flock`` (no orphan break); same True / False / None contract."""
    import msvcrt
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (BlockingIOError, OSError) as exc:
            if not is_advisory_lock_contention(exc):
                logger.warning("Could not acquire FTS rebuild lock %s (%s) — deferring on a non-contention error.",
                               lock_path, exc)
                return None
            if time.monotonic() >= deadline:
                return False
            time.sleep(_FTS_REBUILD_LOCK_POLL_SECONDS)


@contextlib.contextmanager
def fts_rebuild_admission(db_path, *, timeout_seconds=None):
    """Serialize full structural FTS rebuilds on *db_path* across processes.  Yields True when this process
    holds the authority, False when the bounded acquire timed out or the lock file could not be opened: the
    caller must NOT rebuild (fail closed; the stale breadcrumb guarantees a retry).  ``db_path`` None
    (in-memory) yields True.  In-process retries pass ``timeout_seconds=0`` so a live holder never stalls a
    long-lived writer; the orphan break still applies."""
    if db_path is None:
        yield True
        return
    timeout = _FTS_REBUILD_LOCK_TIMEOUT_SECONDS if timeout_seconds is None else max(float(timeout_seconds), 0.0)
    lock_path = f"{db_path}.fts_rebuild.lock"
    try:
        handle = open(lock_path, "a+b")
    except OSError as exc:
        # Fail closed like a timed-out acquire: an unopenable lock file means the FS is out of
        # space/inodes/descriptors and a sibling that opened earlier may still be rebuilding — yielding True
        # gave every process on a full disk a concurrent rebuild.  Deferring is free: the breadcrumb retries.
        logger.warning("Could not open FTS rebuild lock %s (%s) — deferring this rebuild "
                       "rather than running it without cross-process authority.", lock_path, exc)
        yield False
        return
    acquired = False
    try:
        if _IS_WINDOWS:
            acquired = _acquire_msvcrt_lock(lock_path, handle, timeout)
        else:
            acquired, handle = _acquire_db_flock(
                lock_path, handle, timeout, _FTS_REBUILD_LOCK_POLL_SECONDS, "FTS rebuild lock")
        if acquired is None:
            # Already logged with the real errno; "held by another process" would be a lie.
            acquired = False
        elif not acquired:
            record = None if _IS_WINDOWS else _read_lock_holder_record(handle)
            if timeout <= 0:
                # Non-blocking probe from an in-process retry: keep it quiet.
                logger.info("FTS rebuild lock %s is busy — deferring this retry "
                            "(the stale-FTS breadcrumb keeps it retryable). Recorded holder: %s.",
                            lock_path, _describe_lock_holder(record))
            else:
                logger.warning("FTS rebuild lock %s held by another process for more than %.0fs — deferring "
                               "this rebuild to avoid racing the holder (the stale-FTS breadcrumb keeps it "
                               "retryable). Recorded holder: %s.", lock_path, timeout, _describe_lock_holder(record))
        yield acquired
    finally:
        try:
            with contextlib.suppress(OSError):  # best-effort release
                if acquired and _IS_WINDOWS:
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif acquired:
                    import fcntl
                    _clear_lock_holder_record(handle)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _json_or(raw: Any, fallback: Any, warning: str) -> Any:
    """``json.loads(raw)``; on failure log *warning* and return *fallback*."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(warning)
        return fallback
