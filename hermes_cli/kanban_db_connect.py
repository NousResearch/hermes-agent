"""SQLite connection lifecycle for the Kanban DB: open/configure, cross-process init and dispatch-tick locks, WAL checkpoints, corruption detection + quarantine + repair, additive migrations and the busy-retrying ``write_txn`` boundary.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
import re
import secrets
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing
from pathlib import Path
from typing import Any
from typing import Optional


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()
_INIT_LOCK = threading.RLock()
_SQLITE_HEADER = b"SQLite format 3\x00"
DEFAULT_BUSY_TIMEOUT_MS = 120_000

# Cap on ``<db>.corrupt.<hash>.bak`` quarantines per board: content-addressing
# dedupes identical bytes, but mutating corruption mints a new fingerprint each
# time (one user hit 124). Oldest-by-mtime beyond the cap are pruned after each
# new backup.
_CORRUPT_BACKUP_RETENTION = 10

# Bounded init-lock acquire: a bare blocking flock let a wedged holder block the
# dispatcher's next-tick connect forever. Poll non-blocking until the deadline,
# then proceed without the lock (in-process _INIT_LOCK + idempotent init backstop).
_INIT_LOCK_TIMEOUT_SECONDS = 10.0
_INIT_LOCK_POLL_SECONDS = 0.05


def _resolve_busy_timeout_ms() -> int:
    """Return the SQLite busy timeout for Kanban connections. Kanban is the
    shared cross-profile dispatch bus, so worker stampedes are expected; a
    long timeout lets WAL serialize writers instead of surfacing transient
    ``database is locked`` failures."""
    return _kb._env_int("HERMES_KANBAN_BUSY_TIMEOUT_MS", DEFAULT_BUSY_TIMEOUT_MS, minimum=1)


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a Kanban SQLite connection via ``connect_tracked``: while registered,
    byte-level probes of the file are refused because an ``open()``/``close()``
    would cancel this process's POSIX advisory locks (see ``sqlite_safe_read``)."""
    from hermes_cli.sqlite_safe_read import connect_tracked

    busy_timeout_ms = _resolve_busy_timeout_ms()
    conn = connect_tracked(
        path,
        connect_fn=sqlite3.connect,
        isolation_level=None,
        timeout=busy_timeout_ms / 1000.0,
    )
    try:
        # Explicit PRAGMA (besides connect(timeout=)) so it is observable and
        # survives wrapper changes; PRAGMA assignments can't bind parameters.
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    except BaseException:
        # A half-open connection would leak its fd AND leave a stale entry in the
        # connect_tracked registry (cleared only on close), permanently blocking
        # byte-level probes of this file.
        with contextlib.suppress(Exception):
            conn.close()
        raise
    return conn


def _try_lock_nb(handle) -> bool:
    """One non-blocking exclusive lock attempt on ``handle``; False when held elsewhere.
    Windows: 1-byte ``msvcrt.locking`` range at offset 0; POSIX: ``flock``."""
    if _kb._IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def _unlock(handle) -> None:
    """Release a lock taken by :func:`_try_lock_nb` (same byte range / flock)."""
    if _kb._IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _cross_process_init_lock(path: Path):
    """Serialize first-connect WAL/schema/integrity setup across processes.

    ``_INIT_LOCK`` only covers one process's threads; a dispatcher burst has
    many worker processes hit a fresh/legacy board with empty
    ``_INITIALIZED_PATHS`` caches. Post-init usage stays concurrent under WAL.

    **Bounded** acquire: a blocking ``flock`` let one stalled/stale holder hang
    every ``connect()`` (the gateway dispatcher's next tick included) with no
    traceback. After the deadline we WARN and proceed WITHOUT the lock — safe
    because ``_INIT_LOCK`` still serializes same-process threads and init is
    idempotent: two racing first-inits mean redundant work, not corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".init.lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _INIT_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                acquired = _try_lock_nb(handle)
            except OSError:
                acquired = False
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(_INIT_LOCK_POLL_SECONDS)
        if not acquired:
            _kb._log.warning(
                "kanban init lock for %s not acquired within %.0fs — proceeding "
                "without the cross-process lock (in-process lock + idempotent "
                "init are the correctness backstop). A stuck holder is no longer "
                "able to block this connect indefinitely (#36644).",
                lock_path, _INIT_LOCK_TIMEOUT_SECONDS,
            )
        yield
    finally:
        try:
            if acquired:
                _unlock(handle)
        finally:
            handle.close()


@contextlib.contextmanager
def _dispatch_tick_lock(db_path: Path):
    """Non-blocking single-writer guard around one dispatcher tick; yields
    ``True`` if this process holds the board's ``.dispatch.lock``, else
    ``False`` (caller skips the tick).

    Two dispatchers (e.g. an orphan gateway escaping its service cgroup) both
    pass ``busy_timeout`` and race on WAL frames — the root cause of
    multi-writer corruption; this is defense-in-depth behind
    ``_guard_supervised_gateway_conflict``. Non-blocking on purpose: the
    gateway's async watcher must never stall; the loser retries next interval.
    Without ``fcntl``/``msvcrt`` it degrades to a no-op (yields ``True``).

    Motivation (issue #35240): a ``hermes gateway run --replace`` / ``gateway restart`` invoked from a shell
    on a systemd/launchd host can leave an orphan gateway whose dispatcher escapes the service cgroup,
    survives ``systemctl restart``, and becomes a *second* long-lived writer on the same ``kanban.db``. The
    startup guard (``_guard_supervised_gateway_conflict``) blocks the common way an orphan is born, but this
    lock is the defense-in-depth that prevents two dispatchers from ever writing concurrently *regardless of
    how the second one got there*.
    """
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            acquired = _try_lock_nb(handle)
        except (OSError, AttributeError):
            acquired = False
    except OSError:
        # Can't even open the lock file (permissions, read-only FS): degrade to
        # a no-op so a probe failure never blocks dispatch.
        acquired = True
        handle = None
    try:
        yield acquired
    finally:
        if handle is not None:
            try:
                if acquired:
                    _unlock(handle)
            except (OSError, AttributeError):
                pass
            finally:
                handle.close()


# Periodic explicit WAL checkpoint from the dispatcher tick: a passive
# autocheckpoint can be starved on a busy multi-process board (any open reader
# snapshot blocks the WAL reset), letting -wal grow between gateway restarts.
# PASSIVE, not TRUNCATE (same fix class as state.db): the dispatch flock only
# makes this the sole *dispatcher* — CLI kanban commands in other processes
# write to the same board without it, so a TRUNCATE would race live writers.
# PASSIVE never takes the exclusive checkpoint lock; WAL size is bounded by
# ``journal_size_limit`` (set at connection init) on the writer's natural
# post-checkpoint reset. Best-effort, keyed per resolved DB path.
# Once per coarse interval the dispatcher issues an explicit ``wal_checkpoint(PASSIVE)``. Best-effort: a
# busy/locked checkpoint is logged at DEBUG and retried next interval. See #44795, #45383, #80255.
_WAL_CHECKPOINT_INTERVAL_SECONDS = 300.0
_LAST_WAL_CHECKPOINT: dict[str, float] = {}
_WAL_CHECKPOINT_LOCK = threading.Lock()


def _maybe_checkpoint_wal(conn: sqlite3.Connection, db_path: Path) -> None:
    """``PRAGMA wal_checkpoint(PASSIVE)`` at most once per interval per board,
    from the dispatcher tick under the dispatch lock. Never raises: pure
    hygiene, must not fail a tick."""
    try:
        key = str(db_path.resolve())
    except OSError:
        key = str(db_path)
    now = time.monotonic()
    with _WAL_CHECKPOINT_LOCK:
        last = _LAST_WAL_CHECKPOINT.get(key)
        if last is not None and (now - last) < _WAL_CHECKPOINT_INTERVAL_SECONDS:
            return
        # Claim the slot first so concurrent same-process ticks don't
        # double-checkpoint on the boundary.
        _LAST_WAL_CHECKPOINT[key] = now
    try:
        row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        _kb._log.debug(
            "kanban WAL checkpoint (PASSIVE) on %s -> %s "
            "(busy, wal_frames, checkpointed_frames)",
            key, tuple(row) if row is not None else None,
        )
    except sqlite3.Error as exc:
        _kb._log.debug("kanban WAL checkpoint on %s skipped: %s", key, exc)


def _looks_like_tls_record_at(data: bytes, offset: int) -> bool:
    """Return True for a TLS record header at ``data[offset:]``."""
    if len(data) < offset + 5:
        return False
    content_type, major, minor = data[offset], data[offset + 1], data[offset + 2]
    length = int.from_bytes(data[offset + 3:offset + 5], "big")
    return (
        content_type in {0x14, 0x15, 0x16, 0x17}
        and major == 0x03
        and minor in {0x00, 0x01, 0x02, 0x03, 0x04}
        and 0 < length <= 18432
    )


def _validate_sqlite_header(path: Path) -> None:
    """Fail early with an actionable error for non-SQLite Kanban DB files.
    ``sqlite3.connect()`` creates missing and zero-byte files, so those pass;
    non-empty files must carry the SQLite header, so a corrupt page 0 isn't
    collapsed into a generic PRAGMA error and the gateway's corrupt-board
    handling can identify the board by fingerprint."""
    try:
        if path.stat().st_size == 0:
            return
    except OSError:
        return
    # Byte-level probe: must run BEFORE any connection to this path exists
    # (read_header_bytes_preopen refuses once one is live, because the close()
    # would cancel this process's POSIX locks).
    from hermes_cli.sqlite_safe_read import read_header_bytes_preopen

    head = read_header_bytes_preopen(path, length=64)
    if head is None or head.startswith(_SQLITE_HEADER):
        return
    signature = ""
    if head.startswith(b"SQLit") and _looks_like_tls_record_at(head, 5):
        signature = " (TLS record header detected at byte offset 5)"
    elif _looks_like_tls_record_at(head, 0):
        signature = " (TLS record header detected at byte offset 0)"
    raise sqlite3.DatabaseError(
        "file is not a database: invalid SQLite header for "
        f"{path}{signature}; first_32={head[:32].hex(' ')}"
    )


class KanbanDbCorruptError(RuntimeError):
    """Raised when an existing kanban DB file fails integrity checks — a
    fail-closed guard against silently recreating a corrupt board (which would
    destroy the user's tasks). Carries the path and the backup made first."""

    def __init__(self, db_path: Path, backup_path: Optional[Path], reason: str):
        self.db_path = db_path
        self.backup_path = backup_path
        self.reason = reason
        super().__init__(
            f"Refusing to open corrupt kanban DB at {db_path}: {reason}. "
            f"Original preserved; backup at {_backup_label(backup_path)}."
        )


def _backup_label(backup_path: Optional[Path]) -> str:
    return str(backup_path) if backup_path is not None else "<backup failed>"


def _prune_corrupt_backups(parent: Path, base_name: str, keep: Optional[Path] = None) -> None:
    """Keep only the ``_CORRUPT_BACKUP_RETENTION`` newest (by mtime)
    ``<db>.corrupt.<hash>.bak`` files plus their ``-wal``/``-shm`` copies.
    ``keep`` (the just-created backup) is never pruned regardless of mtime —
    ``shutil.copy2`` preserves the source timestamp, which may be older than
    existing backups. Best-effort: prune failures never mask the corruption
    error the caller is about to raise."""
    try:
        backups = [
            candidate
            for candidate in parent.glob(f"{base_name}.corrupt.*.bak")
            if candidate.is_file() and candidate != keep
        ]
    except OSError:
        return
    budget = max(_CORRUPT_BACKUP_RETENTION - (1 if keep is not None else 0), 0)
    if len(backups) <= budget:
        return

    def _mtime(item: Path) -> float:
        try:
            return item.stat().st_mtime
        except OSError:
            return 0.0

    backups.sort(key=_mtime, reverse=True)
    for stale in backups[budget:]:
        for victim in (stale, stale.with_name(stale.name + "-wal"), stale.with_name(stale.name + "-shm")):
            with contextlib.suppress(OSError):
                victim.unlink(missing_ok=True)


def _backup_corrupt_db(path: Path) -> Optional[Path]:
    """Copy a corrupt DB (and WAL/SHM sidecars) to a content-addressed backup.
    The name is deterministic in the main DB's sha256, so repeated quarantines
    of the same bytes reuse one backup while changed bytes get a separate one.
    Returns the main backup path, or ``None`` if the copy failed (the caller
    still raises loudly). Writes are confined to the DB's parent directory:
    the basename derives only from ``path.name`` + content hash."""
    # Pin the resolved parent (``resolve()`` collapses ``..`` and symlinks); we
    # only ever write inside it.
    resolved = path.resolve()
    parent = resolved.parent
    base_name = resolved.name  # basename only
    # Fingerprinting reads the whole file — a close()-on-a-database-file hazard
    # (cancels this process's POSIX advisory locks; see sqlite_safe_read), so it
    # must only run once the board is out of service. Another SessionDB/kanban
    # connection in this process would still be at risk — so REFUSE rather than
    # warn-and-proceed: losing a forensic copy beats corrupting the live DB.
    from hermes_cli.sqlite_safe_read import has_live_connection

    if has_live_connection(resolved):
        _kb._log.error(
            "refusing to quarantine %s: a connection to it is still open in "
            "this process, and fingerprinting the file would cancel that "
            "connection's POSIX locks. Close all connections first.",
            resolved,
        )
        return None
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    candidate = parent / f"{base_name}.corrupt.{digest.hexdigest()[:16]}.bak"
    # Defensive: candidate must still be inside parent after construction.
    if candidate.parent != parent:
        return None
    if not candidate.exists():
        try:
            shutil.copy2(resolved, candidate)
        except OSError:
            return None
        # A NEW backup landed — enforce the retention cap so mutating-corruption
        # loops can't accumulate quarantines forever.
        _prune_corrupt_backups(parent, base_name, keep=candidate)
    for suffix in ("-wal", "-shm"):
        sidecar = parent / (base_name + suffix)
        sidecar_backup = parent / (candidate.name + suffix)
        if (
            sidecar.parent != parent or not sidecar.exists()
            or sidecar_backup.parent != parent or sidecar_backup.exists()
        ):
            continue
        with contextlib.suppress(OSError):
            shutil.copy2(sidecar, sidecar_backup)
    return candidate


# Repairable integrity_check error classes — both *index-scoped*: the table
# b-tree is intact and REINDEX rebuilds the index losslessly. The index name is
# parsed generically (no hardcoded list). Anything else (page corruption,
# "malformed", freelist damage, …) keeps the fail-closed behavior.
_REPAIRABLE_INDEX_ERROR_PATTERNS = (
    re.compile(r"^wrong # of entries in index (?P<index>.+)$"),
    re.compile(r"^row \d+ missing from index (?P<index>.+)$"),
)


def _integrity_messages_ok(messages: list[str]) -> bool:
    """True iff ``PRAGMA integrity_check`` output is the single ``ok`` row."""
    return len(messages) == 1 and messages[0].strip().lower() == "ok"


def _run_integrity_check(conn: sqlite3.Connection) -> list[str]:
    """Return all ``PRAGMA integrity_check`` message rows as strings."""
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return [str(row[0]) for row in rows if row is not None and row[0] is not None]


def _probe_integrity(path: Path) -> list[str]:
    """Open ``path`` read/write (so SQLite can recover/checkpoint a healthy WAL
    / hot-journal DB before we judge it) and return ``integrity_check``
    messages. ``OperationalError`` (locked/busy) propagates raw — not corruption."""
    probe = _sqlite_connect(path)
    try:
        return _run_integrity_check(probe)
    finally:
        probe.close()


def _repairable_index_names(messages: list[str]) -> Optional[list[str]]:
    """Distinct index names iff EVERY message is index-repairable, else ``None``
    (caller fails closed; also ``None`` for no messages). First-appearance
    order is preserved so the REINDEX pass is deterministic."""
    names: list[str] = []
    for raw in messages:
        message = (raw or "").strip()
        if not message:
            continue
        for pattern in _REPAIRABLE_INDEX_ERROR_PATTERNS:
            match = pattern.match(message)
            if match:
                break
        else:
            return None
        name = match.group("index").strip()
        if name and name not in names:
            names.append(name)
    return names or None


def _attempt_index_reindex_repair(path: Path, index_names: list[str]) -> tuple[bool, list[str]]:
    """REINDEX the named indexes (per-index first; bare ``REINDEX`` fallback if
    a parsed name is an internal/auto index), then re-run integrity_check.
    Returns ``(clean, post_repair_messages)``; never raises. Callers must hold
    the board's cross-process init flock so nothing connects mid-repair."""
    try:
        conn = _sqlite_connect(path)
    except sqlite3.Error as exc:
        return False, [f"could not reopen for REINDEX: {exc}"]
    try:
        try:
            for name in index_names:
                escaped = name.replace('"', '""')
                conn.execute(f'REINDEX "{escaped}"')
        except sqlite3.Error:
            # Per-index rebuild failed — bare REINDEX rebuilds every index.
            conn.execute("REINDEX")
        messages = _run_integrity_check(conn)
    except sqlite3.Error as exc:
        return False, [f"REINDEX failed: {exc}"]
    finally:
        conn.close()
    return _integrity_messages_ok(messages), messages


def _missing_or_empty(resolved: Path) -> bool:
    """True for a missing / zero-byte / unstat-able DB file — nothing to probe."""
    try:
        return not resolved.exists() or resolved.stat().st_size == 0
    except OSError:
        return True


def _probe_for_corruption(resolved: Path) -> tuple[Optional[list[str]], Optional[str]]:
    """``(messages, reason)`` from an integrity probe; ``reason`` is ``None``
    when healthy and ``messages`` is ``None`` when sqlite refused to open the
    file at all. ``OperationalError`` (lock/busy) is NOT corruption and
    propagates raw so a locked healthy DB is never quarantined."""
    try:
        messages = _probe_integrity(resolved)
    except sqlite3.OperationalError:
        raise
    except sqlite3.DatabaseError as exc:
        return None, f"sqlite refused to open file: {exc}"
    if _integrity_messages_ok(messages):
        return messages, None
    return messages, f"integrity_check returned {messages[0] if messages else '<no row>'!r}"


def _guard_existing_db_is_healthy(path: Path) -> None:
    """Run ``PRAGMA integrity_check`` on an existing non-empty DB file.

    Narrow auto-repair when the failure is ONLY index-scoped (table b-trees
    intact): content-addressed corrupt backup FIRST, REINDEX under the
    caller-held init flock, re-check, proceed only if clean. Anything else
    (page corruption, ``malformed``, unclean re-check) fails closed: back up
    the file + sidecars and raise :class:`KanbanDbCorruptError` so callers
    never recreate the schema on top of a damaged DB. ``OperationalError``
    (lock/busy) is NOT corruption and propagates raw (no spurious backup).
    No-op for missing / zero-byte files and paths already proven healthy this
    process. All writes are confined to the resolved path's parent.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return
    if _missing_or_empty(resolved) or str(resolved) in _INITIALIZED_PATHS:
        return
    messages, reason = _probe_for_corruption(resolved)
    if reason is None:
        return
    # Quarantine FIRST — both the repair and fail-closed paths preserve the
    # pre-touch bytes before anything mutates the file.
    backup = _backup_corrupt_db(resolved)
    index_names = _repairable_index_names(messages or [])
    if index_names:
        _kb._log.warning(
            "kanban DB %s failed integrity_check with index-only errors "
            "(%s); pre-repair backup at %s — attempting REINDEX auto-repair.",
            resolved, ", ".join(index_names), _backup_label(backup),
        )
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        if repaired:
            _kb._log.warning(
                "kanban DB %s auto-repaired via REINDEX (%s); "
                "integrity_check now clean. Pre-repair copy kept at %s.",
                resolved, ", ".join(index_names), _backup_label(backup),
            )
            return
        reason = (
            f"{reason}; REINDEX auto-repair attempted but integrity_check "
            f"still returned {post[0] if post else '<no row>'!r}"
        )
    raise KanbanDbCorruptError(resolved, backup, reason)


@dataclass
class RepairResult:
    """Outcome of :func:`repair_db`. ``status``: ``"ok"`` (already clean),
    ``"repaired"`` (index-only errors, REINDEX applied, re-check clean;
    ``backup_path`` is the pre-repair copy), ``"corrupt"`` (non-index error
    class, or re-check still dirty), ``"missing"`` (no / zero-byte file)."""

    status: str
    db_path: Path
    messages: list[str] = field(default_factory=list)
    post_repair_messages: list[str] = field(default_factory=list)
    backup_path: Optional[Path] = None
    reindexed: list[str] = field(default_factory=list)


def repair_db(db_path: Optional[Path] = None, *, board: Optional[str] = None) -> RepairResult:
    """Probe a kanban DB and apply the narrow index-REINDEX repair if needed.
    Same policy as :func:`_guard_existing_db_is_healthy` (quarantine BEFORE
    any mutation; REINDEX under the init flock; anything non-index stays
    corrupt), but returns a :class:`RepairResult` instead of raising so
    ``hermes kanban repair`` can pick its exit code. ``OperationalError``
    (locked/busy) still propagates raw: a locked healthy DB must not be
    quarantined."""
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if _missing_or_empty(resolved):
        return RepairResult(status="missing", db_path=resolved)

    with _cross_process_init_lock(resolved):
        messages, reason = _probe_for_corruption(resolved)
        if messages is None:
            # Same quarantine the connect-time guard takes when sqlite
            # refuses to open the file at all.
            return RepairResult(
                status="corrupt", db_path=resolved, messages=[str(reason)],
                backup_path=_backup_corrupt_db(resolved),
            )
        if reason is None:
            return RepairResult(status="ok", db_path=resolved, messages=messages)

        # Quarantine FIRST — identical policy to the connect-time guard.
        backup = _backup_corrupt_db(resolved)
        index_names = _repairable_index_names(messages)
        if not index_names:
            return RepairResult(status="corrupt", db_path=resolved, messages=messages, backup_path=backup)
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        # The file changed on disk; force the next connect() in this process
        # to re-probe instead of trusting the stale healthy-path cache.
        with _INIT_LOCK:
            _INITIALIZED_PATHS.discard(str(resolved))
        return RepairResult(
            status="repaired" if repaired else "corrupt",
            db_path=resolved,
            messages=messages,
            post_repair_messages=post,
            backup_path=backup,
            reindexed=index_names,
        )


def _schema_is_present(conn: sqlite3.Connection) -> bool:
    """Whether an open connection actually sees the kanban schema. ``tasks`` is
    the sentinel (SCHEMA_SQL always creates it; SQLite loses tables
    all-or-nothing), so one ``sqlite_master`` lookup on the resident page 1
    suffices — cheap by design, it runs on every steady-state connect()."""
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks' LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError:
        # Unreadable schema table is not this guard's call — the full init
        # path's header/integrity probes classify and quarantine it.
        return False
    return row is not None


def _open_configured(path: Path, under_lock) -> tuple[sqlite3.Connection, Any]:
    """Open ``path`` with the kanban PRAGMA set, then run ``under_lock(conn)``.
    WAL activation and ``under_lock`` share the ``_INIT_LOCK`` critical section:
    WAL setup can take an exclusive lock while SQLite creates sidecars for a
    fresh DB, and concurrent gateway startup threads must not race before
    ``_INITIALIZED_PATHS`` is populated. Closed if anything raises."""
    conn = _sqlite_connect(path)
    try:
        conn.row_factory = sqlite3.Row
        with _INIT_LOCK:
            # WAL doesn't work on network filesystems; the helper falls back to
            # DELETE with one ERROR log (see hermes_state_wal._WAL_INCOMPAT_MARKERS).
            from hermes_state_wal import apply_wal_with_fallback
            apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
            # FULL (not NORMAL): fsync before each checkpoint to narrow the
            # crash window that can leave a b-tree page header torn.
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA wal_autocheckpoint=100")
            # Bound the WAL file: the periodic explicit checkpoint is PASSIVE
            # (never truncates), so SQLite trims -wal to this limit on the
            # writer's natural post-checkpoint reset.
            conn.execute("PRAGMA journal_size_limit=8388608")
            conn.execute("PRAGMA foreign_keys=ON")
            # Zero freed pages so a later torn write can't expose stale cells.
            conn.execute("PRAGMA secure_delete=ON")
            # Surface corrupt cells as read errors instead of silent wrong data.
            conn.execute("PRAGMA cell_size_check=ON")
            out = under_lock(conn)
    except Exception:
        conn.close()
        raise
    return conn, out


def connect(db_path: Optional[Path] = None, *, board: Optional[str] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB. WAL is (re)enabled on
    every connection so a re-created file stays robust; the first connection
    per path auto-runs :func:`init_db`, later ones skip via
    ``_INITIALIZED_PATHS``. Path: explicit ``db_path``, else ``board``, else
    :func:`kanban_db_path` (``HERMES_KANBAN_DB`` -> ``HERMES_KANBAN_BOARD`` ->
    ``<root>/kanban/current`` -> ``default``)."""
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: once THIS process has initialized this path, skip the
    # cross-process init lock. Taking it on every connect let a single stalled
    # holder (e.g. an external `hermes kanban list` mid-integrity-probe) block
    # the gateway dispatcher's next-tick connect() forever, and steady-state has
    # nothing for it to protect (no schema/migration writes).
    resolved = str(path.resolve())
    if resolved in _INITIALIZED_PATHS:
        conn, schema_present = _open_configured(path, _schema_is_present)
        if schema_present:
            return conn
        # Cache says "initialized", file says otherwise: it was deleted or
        # replaced under a live process and the open silently recreated an empty
        # DB. Left alone, every query fails with "no such table: tasks" for the
        # rest of the process's life. Drop the stale entry and re-init.
        conn.close()
        with _INIT_LOCK:
            # Drop the stale cache entry and fall through to the full init path, which re-runs the header
            # and integrity probes and the schema script under the cross-process lock. See #83445.
            _INITIALIZED_PATHS.discard(resolved)
        _kb._log.warning(
            "kanban DB %s lost its schema after this process initialized it "
            "(deleted or replaced externally); re-initializing.",
            path,
        )

    with _cross_process_init_lock(path):
        # Read-only file/sidecar preflight first, so a stray read-only kanban.db
        # fails actionably instead of "attempt to write a readonly database".
        # See #12508.
        from hermes_state import preflight_db_writability
        preflight_db_writability(path, db_label=f"kanban.db ({path.name})")
        # Cheap byte-level header check before any sqlite connection, then the
        # full integrity probe (cached per path via _INITIALIZED_PATHS).
        _validate_sqlite_header(path)
        _guard_existing_db_is_healthy(path)
        resolved = str(path.resolve())

        def _init_if_needed(conn: sqlite3.Connection) -> None:
            # Idempotent; runs under _INIT_LOCK so same-process dispatcher
            # threads can't race the ALTER TABLE pass with stale PRAGMA snapshots.
            if resolved not in _INITIALIZED_PATHS:
                conn.executescript(_kb.SCHEMA_SQL)
                _migrate_add_optional_columns(conn)
                _INITIALIZED_PATHS.add(resolved)

        conn, _ = _open_configured(path, _init_if_needed)
    return conn


@contextlib.contextmanager
def connect_closing(db_path: Optional[Path] = None, *, board: Optional[str] = None):
    """Open a kanban DB connection and guarantee it is closed on exit. Use
    instead of ``with kb.connect() as conn:`` — sqlite3's context manager only
    commits/rolls back, it does NOT close the fd, so long-lived processes
    (gateway, dashboard) leak FDs until ``[Errno 24] Too many open files``.

    See #33159 for the production incident.
    """
    conn = connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def init_db(db_path: Optional[Path] = None, *, board: Optional[str] = None) -> Path:
    """Create the schema if it doesn't exist; return the path used. Unlike
    :func:`connect`'s cached first-time auto-init, this always re-runs the
    migration pass — callers that know the on-disk schema may have drifted
    (tests writing legacy event kinds, external upgrades) use it to force it."""
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clear the cache entry so connect() re-runs schema + migrations.
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(str(path.resolve()))
    with contextlib.closing(connect(path)):
        pass
    return path


# Additive ``tasks`` columns in the order legacy DBs receive them (= physical
# column order for ``SELECT *`` on migrated boards).
_EARLY_TASK_COLUMNS = (
    ("tenant", "tenant TEXT"),
    ("result", "result TEXT"),
    ("branch_name", "branch_name TEXT"),
    ("project_id", "project_id TEXT"),
    ("idempotency_key", "idempotency_key TEXT"),
)

# (new column, ddl, legacy source column, copy statement) — see the
# RENAME-avoidance note in ``_migrate_add_optional_columns``.
_RENAMED_TASK_COLUMNS = (
    (
        "consecutive_failures", "consecutive_failures INTEGER NOT NULL DEFAULT 0",
        "spawn_failures", "UPDATE tasks SET consecutive_failures = COALESCE(spawn_failures, 0)",
    ),
    ("worker_pid", "worker_pid INTEGER", None, None),
    (
        "last_failure_error", "last_failure_error TEXT",
        "last_spawn_error", "UPDATE tasks SET last_failure_error = last_spawn_error",
    ),
)

# NULL / 0 defaults below reproduce the behaviour existing rows had before the
# column existed.
_LATER_TASK_COLUMNS = (
    ("max_runtime_seconds", "max_runtime_seconds INTEGER"),
    ("last_heartbeat_at", "last_heartbeat_at INTEGER"),
    ("current_run_id", "current_run_id INTEGER"),
    ("workflow_template_id", "workflow_template_id TEXT"),
    ("current_step_key", "current_step_key TEXT"),
    # JSON array of skill names the dispatcher force-loads via --skills.
    ("skills", "skills TEXT"),
    # Per-task override for the consecutive-failure circuit breaker; NULL =
    # ``kanban.failure_limit`` config, then ``DEFAULT_FAILURE_LIMIT``.
    ("max_retries", "max_retries INTEGER"),
    ("model_override", "model_override TEXT"),
    ("provider_override", "provider_override TEXT"),
    ("reasoning_effort", "reasoning_effort TEXT"),
    # Ralph-style goal loop toggle; 0 = classic single-shot worker.
    ("goal_mode", "goal_mode INTEGER NOT NULL DEFAULT 0"),
    ("goal_max_turns", "goal_max_turns INTEGER"),
    ("session_id", "session_id TEXT"),
    # Typed block reason (VALID_BLOCK_KINDS); NULL = generic human blocker.
    ("block_kind", "block_kind TEXT"),
    ("block_recurrences", "block_recurrences INTEGER NOT NULL DEFAULT 0"),
)

_NOTIFY_SUB_COLUMNS = (
    ("notifier_profile", "notifier_profile TEXT"),
    ("delivery_mode", "delivery_mode TEXT NOT NULL DEFAULT 'notify'"),
    ("chat_type", "chat_type TEXT"),
    # Platform-specific stable alt ID (Signal UUID, Feishu union_id, ...)
    # so an active-wake replay reconstructs the SAME ``build_session_key``
    # (which prefers ``user_id_alt``). NULL is inert.
    ("user_id_alt", "user_id_alt TEXT"),
    ("delivery_metadata", "delivery_metadata TEXT"),
)


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
    ).fetchone() is not None


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after v1 to legacy DBs (called via ``init_db``)."""
    cols = _column_names(conn, "tasks")
    for name, ddl in _EARLY_TASK_COLUMNS:
        if name not in cols:
            _add_column_if_missing(conn, "tasks", name, ddl)

    # Re-snapshot: DBs partially migrated by older releases may already carry
    # later columns (e.g. ``consecutive_failures``), keeping the legacy-column
    # migration idempotent.
    cols = _column_names(conn, "tasks")

    # Legacy renames via ADD-then-copy rather than ``RENAME COLUMN``: very old
    # DBs may lack the legacy column entirely (RENAME raises "no such column"),
    # and RENAME reparses the whole schema, failing if views/triggers reference
    # the old name. Historical counter values are preserved when present.
    for name, ddl, legacy, copy_sql in _RENAMED_TASK_COLUMNS:
        if name not in cols:
            added = _add_column_if_missing(conn, "tasks", name, ddl)
            if added and legacy is not None and legacy in cols:
                conn.execute(copy_sql)
    for name, ddl in _LATER_TASK_COLUMNS:
        if name not in cols:
            if name == "model_override":
                conn.execute("ALTER TABLE tasks ADD COLUMN model_override TEXT")
            else:
                _add_column_if_missing(conn, "tasks", name, ddl)

    # Indexes over additive ``tasks`` columns must be created AFTER the columns
    # exist: ``executescript`` parses each statement against the live schema,
    # so a ``CREATE INDEX`` over a missing column in SCHEMA_SQL would abort
    # init on legacy boards before the ALTER TABLE pass runs. ``IF NOT EXISTS``
    # keeps re-running here cheap and correct on fresh DBs.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)")

    # task_events.run_id back-fills as NULL for historical events (they predate
    # runs and can't be attributed).
    if "run_id" not in _column_names(conn, "task_events"):
        _add_column_if_missing(conn, "task_events", "run_id", "run_id INTEGER")

    # task_comments.session_id back-fills as NULL for historical comments
    # (they predate authorship provenance and can't be attributed). Written
    # once at insert time from the calling session's env; never updated.
    # Table-guarded: migration must tolerate partial schemas (see the
    # kanban_notify_subs guard below — same concurrent/partial-migration
    # tolerance, #21708).
    if _table_exists(conn, "task_comments") and "session_id" not in _column_names(conn, "task_comments"):
        _add_column_if_missing(conn, "task_comments", "session_id", "session_id TEXT")

    # Same ordering rule as the ``tasks`` indexes above: index after column.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON task_events(run_id, id)")

    if _table_exists(conn, "kanban_notify_subs"):
        notify_cols = _column_names(conn, "kanban_notify_subs")
        for name, ddl in _NOTIFY_SUB_COLUMNS:
            if name in notify_cols:
                continue
            _add_column_if_missing(conn, "kanban_notify_subs", name, ddl)
            if name == "delivery_mode":
                # Backfill ONLY on first-add: pre-column gateway subscriptions
                # had de facto active wake; defaulting them to 'notify' would
                # silently disable that on upgrade. TUI/CLI rows keep 'notify'
                # (matches _maybe_auto_subscribe). A later explicit downgrade
                # is never overwritten.
                conn.execute(
                    "UPDATE kanban_notify_subs SET delivery_mode = 'notify+wake' "
                    "WHERE platform != 'tui'"
                )

    if _table_exists(conn, "task_runs"):
        _backfill_legacy_inflight_runs(conn)

    # One-shot event-kind rename: old names still worked but were awkward on
    # the wire. Fires once per DB — after the UPDATE no rows match.
    for old, new in (
        ("ready", "promoted"),
        ("priority", "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    ):
        conn.execute("UPDATE task_events SET kind = ? WHERE kind = ?", (new, old))

    _rebuild_drifted_tables(conn)


def _backfill_legacy_inflight_runs(conn: sqlite3.Connection) -> None:
    """One-shot backfill: tasks 'running' before runs existed carried
    claim_lock / claim_expires / worker_pid on the task row; synthesize a
    matching task_runs row so end-run / heartbeat have something to write.
    write_txn serializes against concurrent dispatchers, and the per-row
    UPDATE uses ``current_run_id IS NULL`` as a CAS guard so a racing claim
    can't produce an orphaned row."""
    with write_txn(conn):
        inflight = conn.execute(
            "SELECT id, assignee, claim_lock, claim_expires, worker_pid, "
            "       max_runtime_seconds, last_heartbeat_at, started_at "
            "FROM tasks "
            "WHERE status = 'running' AND current_run_id IS NULL"
        ).fetchall()
        for row in inflight:
            started = row["started_at"] or int(time.time())
            cur = conn.execute(
                """
                INSERT INTO task_runs (
                    task_id, profile, status,
                    claim_lock, claim_expires, worker_pid,
                    max_runtime_seconds, last_heartbeat_at,
                    started_at
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"], row["assignee"], row["claim_lock"],
                    row["claim_expires"], row["worker_pid"],
                    row["max_runtime_seconds"], row["last_heartbeat_at"],
                    started,
                ),
            )
            # CAS: only install the pointer if nothing claimed the task
            # since our SELECT (belt-and-suspenders under write_txn). On
            # failure mark the orphan run row reclaimed so it doesn't
            # look in-flight.
            upd = conn.execute(
                "UPDATE tasks SET current_run_id = ? "
                "WHERE id = ? AND current_run_id IS NULL",
                (cur.lastrowid, row["id"]),
            )
            if upd.rowcount != 1:
                conn.execute(
                    "UPDATE task_runs SET status = 'reclaimed', "
                    "    outcome = 'reclaimed', ended_at = ? "
                    "WHERE id = ?",
                    (int(time.time()), cur.lastrowid),
                )


# Legacy DBs used a ``TEXT PRIMARY KEY`` id (nullable ``TEXT last_event_id``
# for ``kanban_notify_subs``); the additive migrations can't change a column
# type, so drift requires a rebuild. Each entry pairs the canonical CREATE
# TABLE with the indexes DROP TABLE takes down with it.
# ``test_rebuilt_schema_matches_fresh`` guards this against SCHEMA_SQL drift.
# The current schema uses ``INTEGER PRIMARY KEY AUTOINCREMENT`` / ``INTEGER NOT NULL DEFAULT 0``. ``CREATE
# TABLE IF NOT EXISTS`` skips existing tables regardless of schema and ``_add_column_if_missing`` only adds
# columns, so neither can fix a drifted column type — the table must be rebuilt. See #35096. Each entry
# pairs the canonical CREATE TABLE with the CREATE INDEX statements that DROP TABLE would otherwise take
# down with it (including ``idx_events_run``, added by the additive pass above). To guard against this list
# drifting from SCHEMA_SQL, ``test_rebuilt_schema_matches_fresh`` asserts a rebuilt legacy DB is
# byte-identical to a fresh one.
_REBUILD_SPECS = {
    "task_events": (
        "CREATE TABLE task_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL,"
        " payload TEXT, created_at INTEGER NOT NULL)",
        (
            "CREATE INDEX idx_events_task ON task_events(task_id, created_at)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ),
    ),
    "task_comments": (
        "CREATE TABLE task_comments ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)",
        ("CREATE INDEX idx_comments_task ON task_comments(task_id, created_at)",),
    ),
    "task_runs": (
        "CREATE TABLE task_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, profile TEXT, step_key TEXT,"
        " status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, max_runtime_seconds INTEGER,"
        " last_heartbeat_at INTEGER, started_at INTEGER NOT NULL,"
        " ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT,"
        " error TEXT)",
        (
            "CREATE INDEX idx_runs_task ON task_runs(task_id, started_at)",
            "CREATE INDEX idx_runs_status ON task_runs(status)",
        ),
    ),
    "kanban_notify_subs": (
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, user_id_alt TEXT,"
        " chat_type TEXT,"
        " notifier_profile TEXT, delivery_mode TEXT NOT NULL DEFAULT 'notify',"
        " delivery_metadata TEXT, created_at INTEGER NOT NULL,"
        " last_event_id INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))",
        ("CREATE INDEX idx_notify_task ON kanban_notify_subs(task_id)",),
    ),
}


def _table_has_drifted(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` still carries the legacy (pre-AUTOINCREMENT) shape."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        return False  # table absent — nothing to rebuild
    if table == "kanban_notify_subs":
        lei = next((c for c in info if c["name"] == "last_event_id"), None)
        return lei is not None and (lei["type"] or "").upper() != "INTEGER"
    # task_events / task_comments / task_runs: id must be INTEGER and a PK.
    id_col = next((c for c in info if c["name"] == "id"), None)
    if id_col is None:
        return False
    return not ((id_col["type"] or "").upper() == "INTEGER" and id_col["pk"])


def _rebuild_drifted_tables(conn: sqlite3.Connection) -> None:
    """Rebuild any kanban table whose column types drifted from SCHEMA_SQL.

    Drifted boards crash the gateway notifier (``int(None)`` on a NULL id) and
    never match ``id > cursor``, silently losing every notification. Legacy
    TEXT ids are dropped (AUTOINCREMENT reassigns) and cursors reset to 0, so
    the first post-migration tick replays history once — safe for a feature
    that was already fully broken. One transaction under ``connect()``'s init
    locks so an interruption can't leave a table half-renamed. Idempotent.

    Each affected table is rebuilt with the standard SQLite pattern — CREATE new → INSERT shared columns →
    DROP old → RENAME — recreating its indexes too (DROP TABLE takes them down). See #35096.
    """
    drifted = [t for t in _REBUILD_SPECS if _table_has_drifted(conn, t)]
    if not drifted:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in drifted:
            create_sql, index_sqls = _REBUILD_SPECS[table]
            old_cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
            _kb._log.info("kanban migration: rebuilding %s to match current schema", table)
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            conn.execute(create_sql)
            new_cols = _column_names(conn, table)
            if table == "kanban_notify_subs":
                # Cast the legacy TEXT cursor to INTEGER; NULL / non-numeric → 0.
                drop, extra_cols = "last_event_id", ", last_event_id"
                extra_select = ", COALESCE(CAST(last_event_id AS INTEGER), 0)"
            else:
                # Drop the legacy TEXT id; AUTOINCREMENT reassigns it.
                drop, extra_cols, extra_select = "id", "", ""
            cols_csv = ", ".join(c for c in old_cols if c in new_cols and c != drop)
            conn.execute(
                f"INSERT INTO {table} ({cols_csv}{extra_cols}) "
                f"SELECT {cols_csv}{extra_select} FROM {table}_legacy"
            )
            conn.execute(f"DROP TABLE {table}_legacy")
            for index_sql in index_sqls:
                conn.execute(index_sql)
        conn.execute("COMMIT")
    except Exception:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ROLLBACK")
        raise


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Raise ``sqlite3.DatabaseError`` if the file is shorter than its header
    page count claims (torn-extend). Both sides are read WITHOUT opening the
    database file (``PRAGMA page_count`` on the existing connection; ``stat()``
    for disk): an earlier bare ``open(path,"rb")`` probe was wrong because
    ``close()`` cancels every POSIX advisory lock this process holds on the
    file, silently dropping concurrent writers' (and a running VACUUM's) locks
    and letting other processes write into a database a writer still believed
    it owned (sqlite.org/howtocorrupt.html §2.2)."""
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    # In WAL mode a just-committed page can still live in -wal, so the main
    # file legitimately lags its page count; only enforce under a rollback
    # journal, where every committed page must already be in the main file.
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(row[0]).lower() if row and row[0] is not None else ""
    except sqlite3.Error:
        return
    if journal_mode == "wal":
        return

    if file_length_matches_header(conn) is False:
        raise sqlite3.DatabaseError(
            "torn-extend detected: the database file is shorter than its "
            "header page count claims"
        )


# SQLite's busy_timeout backoff is near-deterministic, so stampeding writers
# re-collide in lockstep; a jittered 20-150ms retry on the transaction boundary
# breaks the convoy (mirrors state.db). Only BEGIN IMMEDIATE and COMMIT are
# retried — idempotent re-issues, so a CAS inside write_txn is never replayed.
# 5 retries (not state.db's 15): the 120s busy_timeout absorbs most waits.
_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection, *, allow_nested: bool = False):
    """IMMEDIATE write transaction; a claim CAS inside is atomic — at most one
    concurrent writer succeeds.

    Nesting is an explicit opt-in (``allow_nested=True`` → savepoint; otherwise
    a loud ``RuntimeError``). Only composition primitives (``create_task``,
    ``add_comment``) opt in — helpers with post-commit side effects
    (``complete_task`` & co.) must never run under an open outer transaction,
    since those side effects would fire while the outer txn can still roll back.
    """
    _kb._assert_not_delegated_child_mutation()
    if getattr(conn, "in_transaction", False):
        if not allow_nested:
            raise RuntimeError(
                "write_txn: already inside a transaction. Nested composition "
                "must opt in explicitly with write_txn(conn, allow_nested=True) "
                "(savepoint semantics; the inner RELEASE is not durable until "
                "the outer transaction commits)."
            )
        savepoint = f"hermes_nested_{secrets.token_hex(8)}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield conn
        except Exception:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
            raise
        else:
            conn.execute(f"RELEASE {savepoint}")
        return

    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        # SQLite may already have auto-rolled-back (EIO, contention, corruption);
        # don't let this secondary failure shadow the real one.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ROLLBACK")
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ROLLBACK")
            raise
        # Post-commit torn-extend check — raise now rather than silently corrupt.
        _check_file_length_invariant(conn)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
