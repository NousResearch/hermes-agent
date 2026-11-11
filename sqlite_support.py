"""SQLite helpers shared by the storage backend and the other SQLite stores.

Opening a connection with the journal-mode policy (WAL, its DELETE fallbacks and the never-live-downgrade rule), the configured PRAGMAs and ``sessions.*`` settings, column migration, file permissions, the writability preflight, durability barriers, and the diagnostics callers print about SQLite builds, filesystems and lock holders.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sqlite3
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from hermes_constants import get_hermes_home
from storage_errors import is_sqlite_lock_error

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]


# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")


# WAL needs mmap shared memory + fcntl byte-range locks. Network filesystems (NFS, SMB/CIFS, some FUSE, WSL1) raise
# ``locking protocol``; ZFS corrupts the -shm file under concurrent bursts (COW + mmap) -> ``disk I/O error``.
# Either would silently break everything on state.db/kanban.db, so fall back to DELETE (readers block on writes).
# "not authorized": some FUSE mounts block the WAL pragma outright.
_WAL_INCOMPAT_MARKERS = ("locking protocol", "not authorized", "disk i/o error")
# SQLite's default journal_size_limit is -1 (unlimited); see _apply_wal_size_limit.
_WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB

# Once-per-process-per-db_label dedup sets (kanban_db.connect() runs on every kanban operation, so an undeduped
# line would repeat per connection). Tests clear these sets by name.
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()

# Dedup for the both-pragmas-failed WARNING (WAL *and* DELETE rejected, e.g. APFS external SSDs under contention).
_wal_delete_fallback_failed_paths: set[str] = set()
_wal_delete_fallback_failed_lock = threading.Lock()

# Dedup for the probe-unknown WARNING (on-disk journal mode unreadable, nothing touched).
_wal_probe_unknown_paths: set[str] = set()
_wal_probe_unknown_lock = threading.Lock()
_wal_reset_bug_warned_paths: set[str] = set()
_wal_reset_bug_warned_lock = threading.Lock()
_delete_overridden_warned_paths: set[str] = set()
_delete_overridden_warned_lock = threading.Lock()
_journal_upgrade_warned_paths: set = set()
_journal_upgrade_warned_lock = threading.Lock()

_CANNOT_VERIFY_DELETE_MSG = ("could not verify journal mode before applying configured journal_mode=delete (database "
                             "is locked — possible concurrent openers); refusing to downgrade a database this process "
                             "does not exclusively own")


def _mode_from_row(row) -> str:
    """Lower-cased mode from a ``PRAGMA journal_mode`` row, ``""`` if no row."""
    return str(row[0]).strip().lower() if row and row[0] is not None else ""


def _on_disk_journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    """Read the journal mode from the DB header; ``None`` if undeterminable (new DB, or PRAGMA failed) ->
    callers take their fail-closed "refuse to downgrade" branch. ``disk i/o error`` can be transient on
    virtualized block devices (XFS on cloud hosts), so it is retried a few times first."""
    for _ in range(4):
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        except sqlite3.OperationalError as exc:
            if "disk i/o error" not in str(exc).lower():
                return None
            last_exc = exc
            time.sleep(0.05)
            continue
        mode = row[0] if row else None
        if isinstance(mode, bytes):  # defensive: sqlite3 occasionally returns bytes
            try:
                mode = mode.decode("ascii")
            except UnicodeDecodeError:
                return None
        return str(mode).strip().lower() if mode is not None else None
    logger.debug("_on_disk_journal_mode: retries exhausted on disk read (%s)", last_exc)
    return None


def _apply_wal_size_limit(conn: sqlite3.Connection) -> None:
    """Bound the WAL so it returns space after big transactions. With the default (-1) a checkpointed WAL is
    reused in place, never truncated, so ``state.db-wal`` keeps the high-water mark of the largest
    transaction ever (a 3 GB optimize left a 3 GB WAL). Best-effort: failure only costs disk slack."""
    try:
        conn.execute(f"PRAGMA journal_size_limit={_WAL_SIZE_LIMIT_BYTES}")
    except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
        logger.debug("journal_size_limit not applied: %s", exc)


def _darwin_pragma(conn: sqlite3.Connection, pragma: str) -> None:
    """Best-effort PRAGMA on macOS only (no-op elsewhere, never raises)."""
    if sys.platform == "darwin":
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(pragma)


def _apply_macos_checkpoint_barrier(conn: sqlite3.Connection) -> None:
    """Enable ``PRAGMA checkpoint_fullfsync`` on macOS. Apple's ``fsync(2)`` guarantees neither data-on-platter nor
    ordering, so without ``F_FULLFSYNC`` a launchd shutdown can turn a "durable" checkpoint into a malformed
    ``state.db``. Checkpoint boundaries only (~+0.1 ms/commit vs ~+4 ms for ``fullfsync=1``).

    During a launchd *system* shutdown/reboot the OS page cache is dropped (effectively a power-loss event
    for in-flight pages), so a WAL checkpoint whose ``fsync()`` "reported" durable may never have hit the
    platter — corrupting ``state.db`` with a malformed image. This is the trigger in issue #30636 ("SIGTERM
    during launchd shutdown under high load"), distinct from a plain in-session kill (which the page cache
    survives and SQLite recovers from).
    """
    _darwin_pragma(conn, "PRAGMA checkpoint_fullfsync=1")


def _enforce_macos_synchronous_full(conn: sqlite3.Connection) -> None:
    """Enforce ``PRAGMA synchronous=FULL`` on macOS: with NORMAL a WAL checkpoint
    racing process termination leaves half-written btree pages (``btreeInitPage
    error 11``). Called after every WAL activation so a prior NORMAL never sticks."""
    _darwin_pragma(conn, "PRAGMA synchronous=FULL")


def _apply_wal_companions(conn: sqlite3.Connection) -> None:
    """The settings every WAL activation carries: size limit + macOS barriers."""
    _apply_wal_size_limit(conn)
    _apply_macos_checkpoint_barrier(conn)
    _enforce_macos_synchronous_full(conn)


def is_sqlite_wal_reset_vulnerable(version_info: Optional[tuple] = None) -> bool:
    """True when the linked SQLite has the WAL-reset bug (3.7.0–3.51.2; fixed 3.51.3+, backports 3.50.7 /
    3.44.6). Pre-WAL libraries are safe. https://sqlite.org/wal.html#walresetbug"""
    return _is_sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info if version_info is None else version_info)


def sqlite_source_id() -> str:
    """Return ``sqlite_source_id()``, or an empty string when unavailable."""
    try:
        with contextlib.closing(sqlite3.connect(":memory:")) as conn:
            row = conn.execute("SELECT sqlite_source_id()").fetchone()
    except sqlite3.Error:
        return ""
    return str(row[0]) if row and row[0] is not None else ""


def _database_has_content(conn: sqlite3.Connection) -> bool:
    """Whether the file already holds pages (existing vs brand-new DB); lock-free header read. Fail-quiet False: the
    only caller gates a warning on this and an unknown-answer warning would fire on every fresh database."""
    try:
        row = conn.execute("PRAGMA page_count").fetchone()
        return bool(row) and row[0] is not None and int(row[0]) > 0
    except (sqlite3.Error, TypeError, ValueError):
        return False


def resolve_journal_mode() -> str:
    """The configured ``database.journal_mode`` (``wal`` default; ``delete`` for filesystems without WAL-safe
    durability: macOS virtiofs, NFS, SMB). Invalid values fail safe to ``wal``."""
    try:
        from hermes_cli.config import load_config_readonly
        database = (load_config_readonly() or {}).get("database", {})
        raw = database.get("journal_mode", "wal") if isinstance(database, dict) else "wal"
    except Exception:  # health: allow BLE001 -- an unreadable config means the WAL default
        return "wal"
    mode = raw.strip().lower() if isinstance(raw, str) else ""
    return mode if mode in ("wal", "delete") else "wal"


def routed_sessions_setting(key: str, env_var: str) -> Any:
    """``sessions.<key>`` for the profile whose state.db this process is touching.

    ``gateway/run.py`` bridges the LAUNCH profile's ``sessions.*`` into ``env_var`` (the cross-process
    carrier CLI/cron children read). Under a multiplexer a routed turn runs with a HERMES_HOME override
    and that env slot holds the default profile's value, so a served profile with different
    ``sessions.*`` settings must read its own config.yaml. Unscoped: the env bridge, as before.
    Returns ``None`` when neither source sets the key.
    """
    from hermes_constants import get_hermes_home_override

    if get_hermes_home_override():
        try:
            from hermes_cli.config import load_config_readonly
            return (load_config_readonly().get("sessions") or {}).get(key)
        except Exception:
            return None
    return os.environ.get(env_var)


class WalUnsupportedError(sqlite3.OperationalError):
    """Raised by :func:`apply_wal_with_fallback` under ``require_wal=True`` when
    the filesystem cannot provide WAL (SQLITE_PROTOCOL raised, or macOS-NFS silent
    refusal) or the on-disk mode cannot be verified (probe blocked by a concurrent
    opener). Subclasses ``OperationalError`` so DB-init handlers still catch it."""


# Docker Desktop / OrbStack / Podman expose host bind mounts to the guest as ``fuse.virtiofs`` or ``9p``. WAL's
# -shm file must be coherent shared memory for every opener; across the VM boundary it is not, and under write
# pressure the failure is SILENT (zero-filled pages), so the reactive ``_WAL_INCOMPAT_MARKERS`` fallback in
# ``_enable_wal`` never gets a signal — WAL must be refused before the pragma. Port of openclaw/openclaw#120597.
# Detection is Linux ``/proc/self/mountinfo`` only (statvfs carries no fstype); elsewhere, or when the table is
# unreadable, the answer is False and behaviour is unchanged. Nothing else (ext4/btrfs/xfs/zfs/tmpfs/overlay/nfs)
# is ever flagged here — those keep the existing reactive paths.
_CROSS_VM_FSTYPES = frozenset({"virtiofs", "fuse.virtiofs", "9p", "9p2000", "9p2000.l", "9p2000.u"})
_cross_vm_fs_cache: dict[str, bool] = {}  # per DB directory; kanban_db.connect() opens per operation
_cross_vm_fs_cache_lock = threading.Lock()
_cross_vm_warned_paths: set[str] = set()
_cross_vm_warned_lock = threading.Lock()
_cross_vm_existing_wal_warned_paths: set[str] = set()
_cross_vm_existing_wal_warned_lock = threading.Lock()


def _mountinfo_fstype(directory: str, mountinfo_path: str = "/proc/self/mountinfo") -> str:
    """fstype of the longest mount point that is a prefix of ``directory`` (``""`` if unreadable / no match)."""
    try:
        with open(mountinfo_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""
    best_len, best_fstype = -1, ""
    for line in lines:
        # proc(5): <id> <parent> <maj:min> <root> <mount point> <opts> [optional...] - <fstype> <source> <super opts>
        fields, _, tail = line.partition(" - ")
        parts = fields.split()
        if len(parts) < 5 or not tail:
            continue
        mount_point = parts[4]
        if "\\" in mount_point:  # octal escapes (\040 = space)
            mount_point = mount_point.encode("latin-1", "ignore").decode("unicode_escape")
        if directory == mount_point or directory.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best_len, best_fstype = len(mount_point), tail.split()[0]
    return best_fstype.lower()


def _detect_cross_vm_fs(directory: str, mountinfo_path: str = "/proc/self/mountinfo") -> bool:
    """True only when ``directory`` sits on a virtiofs/9p mount per ``mountinfo_path``."""
    if sys.platform != "linux":
        return False
    return _mountinfo_fstype(directory, mountinfo_path) in _CROSS_VM_FSTYPES


def path_on_cross_vm_fs(path: str) -> bool:
    """True when ``path`` resides on a virtiofs/9p (cross-VM) filesystem; cached per resolved directory."""
    try:
        directory = os.path.dirname(os.path.realpath(path)) or "/"
    except (OSError, ValueError):
        return False
    with _cross_vm_fs_cache_lock:
        cached = _cross_vm_fs_cache.get(directory)
    if cached is None:
        cached = _detect_cross_vm_fs(directory)
        with _cross_vm_fs_cache_lock:
            _cross_vm_fs_cache[directory] = cached
    return cached


def _connection_db_file(conn: sqlite3.Connection) -> str:
    """Filesystem path of the ``main`` database, ``""`` for in-memory / unknown."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        return str(row[2]) if row and row[2] else ""
    except (sqlite3.OperationalError, IndexError, TypeError):
        return ""


def _verify_configured_delete(actual: str) -> str:
    """Raise unless SQLite reported ``delete`` for an explicit operator request."""
    if actual != "delete":
        raise sqlite3.OperationalError(f"could not set configured journal_mode=delete (got {actual or 'no result'})")
    return actual


def apply_wal_with_fallback(conn: sqlite3.Connection, *, db_label: str = "state.db", require_wal: bool = False) -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the mode actually set — or, when the read-only mode probe is blocked by a concurrent opener, ``"wal"``
    as the assumed mode with nothing touched (``require_wal=True`` raises instead). Used by the state
    and kanban stores' connection setup. WAL-incompatible filesystems either raise ``OperationalError`` ("locking protocol" / "disk I/O error") or —
    macOS NFS / SMB / AgentFS — silently refuse and stay in DELETE; either way log ERROR once per process per
    ``db_label`` and fall back. ``require_wal=True`` raises :class:`WalUnsupportedError` instead. WAL-reset-bug
    builds (https://sqlite.org/wal.html#walresetbug) never enable WAL on non-WAL files; an already-WAL DB keeps WAL
    with a warning. Gate deliberately RETAINED: re-measured on the bundled 3.50.4 there is no evidence WAL is safer.

    Invariant on every path: never downgrade to DELETE if the on-disk header reports WAL or cannot be read — other
    gateway/cron/worker connections may hold the DB open, and a live downgrade destroys their uncheckpointed
    commits.

    An earlier revision of the lock-cancellation fix (#71724) reverted it on the theory that DELETE was "the
    mode that corrupts", but that comparison was confounded: the clean WAL result came from SQLite 3.53.1,
    which carries BOTH the WAL-reset fix AND 3.51.0's defenses against close()-broken POSIX locks, so it
    says nothing about 3.50.4. Re-measured on the actually-bundled 3.50.4 with the lock fix in place, WAL
    and DELETE are both clean (0/3 each) — i.e. there is no evidence that WAL is safer here, and upstream
    still documents the WAL-reset bug as real through 3.51.2 with serious consequences. Until a fixed
    runtime is delivered, keep new databases out of WAL.
    """
    config_module = sys.modules.get("hermes_cli.config")
    if getattr(getattr(config_module, "__spec__", None), "_initializing", False):
        config_module = None
    config_lock = getattr(config_module, "_CONFIG_LOCK", None)
    acquired = config_lock is not None and config_lock.acquire(blocking=False)
    try:
        if config_lock is not None and not acquired:
            # A config writer cannot hold up work on an initialized database.
            # Preserve its observed mode; defer configuration-driven transitions.
            current_mode = _on_disk_journal_mode(conn)
            if current_mode == "wal":
                if is_sqlite_wal_reset_vulnerable():
                    return _apply_delete_for_wal_reset_bug(conn, db_label=db_label)
                _apply_wal_companions(conn)
                return "wal"
            if current_mode == "delete" and not require_wal and _database_has_content(conn):
                if is_sqlite_wal_reset_vulnerable():
                    _log_wal_reset_bug_once(db_label, kept_wal=False)
                return "delete"
        configured = resolve_journal_mode()
    finally:
        if acquired:
            config_lock.release()

    # Vulnerable SQLite: never enable WAL on non-WAL files (configured mode resolved first so an explicit DELETE
    # request is still verified).
    if is_sqlite_wal_reset_vulnerable():
        return _apply_delete_for_wal_reset_bug(conn, db_label=db_label, require_delete=configured == "delete")
    # Read-only probe (no flock/checkpoint/WAL-SHM unlink): WAL-init must not unlink files other connections hold.
    current_mode = _on_disk_journal_mode(conn)
    if current_mode == "wal":
        if configured == "delete":
            # Never-live-downgrade keeps WAL; tell the operator their delete did not apply.
            _log_configured_delete_overridden_once(db_label)
        _warn_existing_wal_on_cross_vm_fs(conn)
        _apply_wal_companions(conn)
        return "wal"

    if configured == "delete":
        # #68545: honor the canonical database.journal_mode setting. Existing on-disk WAL databases were
        # returned above and are never live-downgraded.
        if current_mode is None:
            # Probe failed (locked/busy): ownership not provably exclusive. Fail loudly.
            raise sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
        return _verify_configured_delete(_set_journal_mode_no_wait(conn, "DELETE"))
    if current_mode is None:
        # Probe failed (locked/busy): ownership not provably exclusive, same as the DELETE branch above. A fresh
        # 0-page DB probes "delete" cleanly, so this is a real file some other connection holds — running WAL-init
        # would unlink its -wal/-shm sidecars. Touch nothing: the connection inherits whatever mode the header has.
        if require_wal:
            raise WalUnsupportedError("could not verify the on-disk journal mode (database is locked — possible "
                                      "concurrent openers); cannot guarantee WAL")
        _log_once("wal_probe_unknown", db_label)
        return "wal"
    # Cross-VM bind mount (virtiofs/9p): WAL corrupts silently there, so refuse to ENABLE it. On-disk WAL
    # databases were returned above (never live-downgrade); a 0-page / DELETE file just stays DELETE.
    db_file = _connection_db_file(conn)
    if db_file and path_on_cross_vm_fs(db_file):
        if require_wal:
            raise WalUnsupportedError("journal_mode=WAL refused: database is on a cross-VM filesystem (virtiofs/9p) "
                                      "where WAL shared-memory silently corrupts")
        _log_once("cross_vm_fs", db_label)
        return _set_journal_mode_no_wait(conn, "DELETE") or "delete"
    return _enable_wal(conn, db_label, require_wal, current_mode)


def _enable_wal(conn: sqlite3.Connection, db_label: str, require_wal: bool, current_mode: Optional[str]) -> str:
    """Flip a non-WAL, non-vulnerable connection to WAL, or fall back to DELETE."""
    # Decide BEFORE the flip whether it overwrites a mode somebody chose (probe and page_count are only readable
    # while the file is untouched). A 0-page DB has no prior choice, and every caller reaches this before schema.
    upgrading_existing_db = current_mode is not None and current_mode != "wal" and _database_has_content(conn)

    def _wal_activated() -> str:
        if upgrading_existing_db:
            _log_journal_mode_upgrade_once(db_label, current_mode)
        _apply_wal_companions(conn)
        return "wal"

    try:
        # ``PRAGMA journal_mode=WAL`` RETURNS the resulting mode: macOS NFS, SMB/CIFS and the AgentFS overlay
        # refuse WITHOUT raising. Trust the row, not the absence of an exception.
        mode = _mode_from_row(conn.execute("PRAGMA journal_mode=WAL").fetchone())
        if mode == "wal":
            return _wal_activated()
        silent_exc = WalUnsupportedError(f"journal_mode=WAL refused without raising (still {mode!r})")
        if require_wal:
            raise silent_exc
        _log_wal_fallback_once(db_label, silent_exc)
        return mode or "delete"
    except WalUnsupportedError:
        raise  # the require_wal silent-refusal raise above — propagate unchanged
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            raise  # unrelated OperationalError — don't silently swallow
        # ``disk i/o error`` is ambiguous: on ZFS / APFS-CoW it is a deterministic WAL-incompatibility (SHM
        # corruption under concurrent connection bursts — #55305, #71498), but it can also be a one-shot
        # transient EIO (page-cache pressure, brief lock contention). Treating a transient EIO as a
        # permanent downgrade signal produced the mixed-journal-mode corruption pattern fixed in 5c49cd0ed0
        # (process A downgrades to DELETE while sibling processes set WAL). Disambiguate by retrying the
        # pragma a couple of times: transient EIO clears and we return "wal"; the deterministic filesystem
        # cases keep failing and fall through to the guarded DELETE fallback.
        if "disk i/o error" in msg:
            # Retry twice: EIO is either deterministic WAL-incompatibility (ZFS / APFS-CoW) or a one-shot transient,
            # and treating a transient as a permanent downgrade produced mixed-mode corruption (A downgrades to
            # DELETE while siblings set WAL). A non-EIO retry error propagates.
            for _ in range(2):
                time.sleep(0.05)
                try:
                    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                except sqlite3.OperationalError as retry_exc:
                    if "disk i/o error" not in str(retry_exc).lower():
                        raise
                    exc = retry_exc
                    continue
                if _mode_from_row(row) == "wal":
                    return _wal_activated()
                break
        # Never downgrade if WAL is on disk or the mode cannot be read (probe blocked by a concurrent opener) —
        # ownership is not provably exclusive either way.
        if _on_disk_journal_mode(conn) in ("wal", None):
            raise
        if require_wal:
            raise WalUnsupportedError(str(exc)) from exc
        _log_wal_fallback_once(db_label, exc)
        try:
            _set_journal_mode_no_wait(conn, "DELETE")
        except sqlite3.OperationalError as delete_exc:
            # Filesystems that reject BOTH pragmas (APFS external SSDs under heavy contention raise
            # "disk I/O error" on DELETE too). The connection keeps its current mode — typically the
            # SQLite default DELETE — and stays usable for reads/writes, so propagating would crash
            # every DB-init caller (state, kanban and response stores) for no benefit. Log once
            # per db_label and return the mode actually in effect, read back rather than guessed.
            _log_once("wal_delete_fallback_failed", db_label, exc, delete_exc)
            try:
                read_back = _mode_from_row(conn.execute("PRAGMA journal_mode").fetchone())
            except sqlite3.OperationalError:
                read_back = ""
            return read_back or "delete"
        return "delete"


def _set_journal_mode_no_wait(conn: sqlite3.Connection, mode: str) -> str:
    """Execute ``PRAGMA journal_mode=<mode>`` without waiting on other openers.

    The ONLY place a non-WAL journal-mode switch may be issued. ``busy_timeout=0`` turns SQLite's exclusivity
    requirement into a concurrent-opener detector: leaving WAL needs exclusive access, so if ANY other connection
    holds the DB the pragma fails immediately with ``database is locked`` instead of sneaking the flip between a
    writer's transactions (how uncheckpointed WAL commits die). Callers must treat a raised ``OperationalError`` as
    "not exclusively owned: leave the mode alone", never as retryable. Returns the reported mode, ``""`` if no row."""
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        previous_timeout = int(row[0]) if row and row[0] is not None else 0
    except (sqlite3.OperationalError, TypeError, ValueError):
        previous_timeout = 0
    conn.execute("PRAGMA busy_timeout=0")
    try:
        return _mode_from_row(conn.execute(f"PRAGMA journal_mode={mode}").fetchone())
    finally:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={previous_timeout}")


def _warn_existing_wal_on_cross_vm_fs(conn: sqlite3.Connection) -> None:
    """#110848: the fresh-DB cross-VM refusal never sees a DB that was already WAL before the check existed (or was
    created on a native volume and then moved). Keeping WAL is right (never live-downgrade); staying silent is not.
    Both the vulnerable-SQLite and the regular already-WAL paths return early, so both must call this."""
    db_file = _connection_db_file(conn)
    if db_file and path_on_cross_vm_fs(db_file):
        # Keyed (and labelled) by path, not db_label: a gateway serving several profiles must hear about each one.
        _log_once("cross_vm_fs_existing_wal", db_file)


def _apply_delete_for_wal_reset_bug(conn: sqlite3.Connection, *, db_label: str, require_delete: bool = False) -> str:
    """Avoid enabling WAL when the linked SQLite has the WAL-reset bug.

    Already-WAL on disk: keep WAL (no live downgrade) and warn. Mode unreadable (probe blocked by a concurrent
    opener): not provably exclusive — leave it and warn; treating "could not read" as "not WAL" once flipped a live
    WAL state.db to DELETE under a writer, destroying its uncheckpointed commits. Otherwise set DELETE without
    waiting out openers and warn; an explicit operator request additionally verifies SQLite accepted DELETE."""
    current = _on_disk_journal_mode(conn)
    if current == "wal":
        _log_wal_reset_bug_once(db_label, kept_wal=True)
        if require_delete:
            # Upgrading SQLite doesn't help here; emit the actionable message last.
            _log_configured_delete_overridden_once(db_label)
        _warn_existing_wal_on_cross_vm_fs(conn)
        _apply_wal_companions(conn)
        return "wal"
    if current is None:
        if require_delete:
            raise sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
        _log_wal_reset_bug_once(db_label, kept_wal=True, indeterminate=True)
        return "wal"
    actual = ""
    try:
        actual = _set_journal_mode_no_wait(conn, "DELETE")
    except sqlite3.OperationalError as exc:
        if require_delete:
            raise
        if is_sqlite_lock_error(exc):
            # A concurrent opener appeared between probe and flip: leave the mode as is.
            _log_wal_reset_bug_once(db_label, kept_wal=True, indeterminate=True)
            return current or "delete"
        # Best-effort otherwise: DELETE is already the default for new file-backed DBs.
    if require_delete:
        _verify_configured_delete(actual)
    _log_wal_reset_bug_once(db_label, kept_wal=False)
    return "delete"


def wal_reset_repair_hint() -> str:
    """Repair hint matching what ``hermes update`` can actually do for this install type.

    See #75153.
    """
    try:
        from hermes_cli.config import detect_install_method, get_project_root, recommended_update_command_for_method
        method = detect_install_method(get_project_root())
        cmd = recommended_update_command_for_method(method)
        if method in {"git", "unknown"}:
            return f"Hermes-managed installs can repair the embedded runtime with `{cmd}`"
        return f"update the container image with `{cmd}`" if method == "docker" else cmd  # else nix/nixos
    except Exception:  # health: allow BLE001 -- an unknown install method gets the generic hint
        return "install a Python build bundled with SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6) and restart Hermes"


# Once-per-(process, db_label) log table. Levels are deliberate: falling back to DELETE and an ignored
# ``journal_mode: delete`` are real losses (ERROR); a non-WAL -> WAL flip is normally desirable (WARNING).
_WAL_RESET_BUG_ACTIONS = {
    "indeterminate": ("journal mode could not be verified or exclusively switched (database is locked — possible "
                      "concurrent openers); leaving the journal mode untouched (no live downgrade under concurrent "
                      "openers)"),
    "kept_wal": "is already in WAL mode — leaving WAL in place (no live downgrade under concurrent openers)",
    "delete": "using journal_mode=DELETE instead of enabling WAL",
}
_ONCE_LOGS = {
    "wal_reset_bug": (_wal_reset_bug_warned_lock, "_wal_reset_bug_warned_paths", logging.WARNING,
        # Install-type-aware so the warning never promises a repair path that doesn't exist for git/pip installs.
        "%s: linked SQLite %s (interpreter %s) is vulnerable to the WAL-reset corruption bug "
        "(https://sqlite.org/wal.html#walresetbug) — %s. Upgrade to SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6); "
        "%s. See `hermes doctor`. This warning fires once per process per database."),
    "journal_upgrade": (_journal_upgrade_warned_lock, "_journal_upgrade_warned_paths", logging.WARNING,
        # journal_mode is a property of the FILE: switching an existing DB to WAL rewrites its header and outlives
        # the process. Operators set DELETE on the file (the WAL-reset-bug mitigation) and nothing told them the
        # next open would silently put WAL back.
        "%s: on-disk journal_mode was %s and has been switched to WAL. This rewrites the database header and "
        "persists after this process exits. If %s was a deliberate choice (for example the mitigation for the SQLite "
        "WAL-reset bug, or a WAL-unsafe filesystem), setting it with PRAGMA on the file will not survive -- every "
        "open re-applies the configured mode. Set `database.journal_mode: delete` in config.yaml to make it stick. "
        "This message fires once per process per database."),
    "wal_fallback": (_wal_fallback_warned_lock, "_wal_fallback_warned_paths", logging.ERROR,
        # Under kanban dispatcher + workers a DELETE-mode write blocks readers as SQLITE_BUSY.
        "%s: WAL journal_mode unsupported on this filesystem (%s) — falling back to journal_mode=DELETE (slower "
        "rollback-journal mode; reduces concurrency but works on NFS/SMB/FUSE/ZFS). See "
        "https://www.sqlite.org/wal.html for details. This message fires once per process per database."),
    "wal_delete_fallback_failed": (_wal_delete_fallback_failed_lock, "_wal_delete_fallback_failed_paths", logging.WARNING,
        # Both pragmas rejected (observed on APFS external SSDs under heavy contention): the connection keeps
        # whatever mode it has (typically the SQLite default DELETE) and stays usable for reads/writes, so this
        # is WARNING, not ERROR — propagating would crash every DB-init caller. Deduped per db_label because
        # kanban_db.connect() runs on every kanban operation.
        "%s: both WAL and DELETE journal_mode failed (WAL: %s; DELETE: %s) — continuing with the connection's "
        "current journal mode (reads/writes still work). Typically seen on APFS external SSDs under heavy "
        "contention. This message fires once per process per database."),
    "delete_overridden": (_delete_overridden_warned_lock, "_delete_overridden_warned_paths", logging.ERROR,
        # Never-live-downgrade keeps WAL; without this the operator never learns their delete had no effect.
        "%s: database.journal_mode=delete is configured but the on-disk database is already WAL; keeping WAL (a live "
        "downgrade under open connections can corrupt the DB). To apply journal_mode=DELETE, stop every Hermes "
        "process using this database and run `hermes sessions set-journal-mode delete` (add `--db PATH` for a "
        "store other than state.db). This message fires once per process per database."),
    "wal_probe_unknown": (_wal_probe_unknown_lock, "_wal_probe_unknown_paths", logging.WARNING,
        # WARNING, not ERROR: the connection inherits the header's mode, so an already-WAL file (the common case)
        # keeps working; only a true-DELETE file stays DELETE for this connection.
        "%s: could not verify the on-disk journal mode (database is locked / busy); not issuing a journal-mode "
        "set-pragma while another connection may hold the file (it could unlink the -wal/-shm sidecars that "
        "connection still uses). Leaving the file untouched; this connection inherits the on-disk mode. This message fires once per process per database."),
    "cross_vm_fs": (_cross_vm_warned_lock, "_cross_vm_warned_paths", logging.WARNING,
        # The DB keeps working in DELETE mode; the loss is concurrency, so WARNING with the actionable fix.
        "%s: database directory is on a cross-VM filesystem (virtiofs/9p — typical for Docker Desktop / OrbStack / "
        "Podman host bind mounts). SQLite WAL shared-memory is not coherent across the VM boundary and can silently "
        "corrupt the database, so journal_mode=DELETE is used instead. To restore WAL concurrency, move the database "
        "onto a native volume (e.g. a named Docker volume) instead of a host bind mount. This message fires once per "
        "process per database."),
    "cross_vm_fs_existing_wal": (_cross_vm_existing_wal_warned_lock, "_cross_vm_existing_wal_warned_paths", logging.ERROR,
        # ERROR, unlike the fresh-DB refusal: this database IS running WAL on the corrupting filesystem and only the
        # operator can fix it (a live downgrade under other openers would destroy their uncheckpointed commits).
        "%s: existing WAL-mode database is on a cross-VM filesystem (virtiofs/9p — typical for Docker Desktop / "
        "OrbStack / Podman host bind mounts). SQLite WAL shared-memory is not coherent across the VM boundary and "
        "concurrent writers can silently corrupt the database. Hermes does not live-downgrade an on-disk WAL database. "
        "Fix one of two ways: stop every Hermes process using this database and run `hermes sessions "
        "set-journal-mode delete` (set `database.journal_mode: delete` in config.yaml to keep it), "
        "or move the database onto a native volume (e.g. a named Docker volume). This message fires once per process "
        "per database."),
}


def _log_once(kind: str, db_label: str, *args: Any) -> None:
    """Emit ``_ONCE_LOGS[kind]`` once per (process, db_label). Callable *args* are
    resolved only after the dedupe check, so install-method probes run once."""
    lock, set_name, level, message = _ONCE_LOGS[kind]
    seen = globals()[set_name]
    with lock:
        if db_label in seen:
            return
        seen.add(db_label)
    logger.log(level, message, db_label, *(a() if callable(a) else a for a in args))


def _log_wal_reset_bug_once(db_label: str, *, kept_wal: bool, indeterminate: bool = False) -> None:
    """Log once per (process, db_label) about the WAL-reset vulnerability path."""
    action = _WAL_RESET_BUG_ACTIONS["indeterminate" if indeterminate else "kept_wal" if kept_wal else "delete"]
    _log_once("wal_reset_bug", db_label, sqlite3.sqlite_version, sys.executable, action, wal_reset_repair_hint)


def _log_journal_mode_upgrade_once(db_label: str, previous_mode: str) -> None:
    """Single WARNING per (process, db_label) about a non-WAL -> WAL flip."""
    _log_once("journal_upgrade", db_label, previous_mode, previous_mode)


# Single ERROR per (process, db_label): WAL fallback / configured delete ignored (DB already WAL).
_log_wal_fallback_once = functools.partial(_log_once, "wal_fallback")
_log_configured_delete_overridden_once = functools.partial(_log_once, "delete_overridden")


# Operators write synchronous as a name; mapped so a typo becomes a warning, not a silently different level.
_SYNCHRONOUS_LEVELS: dict[str, int] = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}
_SYNCHRONOUS_NAMES: dict[int, str] = {v: k for k, v in _SYNCHRONOUS_LEVELS.items()}
_SYNCHRONOUS_FULL = 2


def resolve_synchronous_level(raw_value: Any) -> Optional[int]:
    """Map ``database.synchronous`` (``OFF``/``NORMAL``/``FULL``/``EXTRA`` any
    case, or ``0``-``3``) to its PRAGMA integer; None for anything else so the
    caller warns and leaves the level untouched (guessing at durability is worse)."""
    if isinstance(raw_value, bool):
        # bool is an int subclass and YAML turns bare `on`/`off` into one; "off" is a real choice, True meaningless.
        return 0 if raw_value is False else None
    if isinstance(raw_value, int):
        return raw_value if raw_value in _SYNCHRONOUS_NAMES else None
    text = str(raw_value).strip()
    if text.upper() in _SYNCHRONOUS_LEVELS:
        return _SYNCHRONOUS_LEVELS[text.upper()]
    with contextlib.suppress(TypeError, ValueError):
        return int(text) if int(text) in _SYNCHRONOUS_NAMES else None
    return None


def _apply_synchronous_pragma(conn: sqlite3.Connection, raw_value: Any, *, db_label: str) -> None:
    """Set ``PRAGMA synchronous`` from config, never below FULL on macOS.

    Kept out of the integer loop in :func:`apply_database_pragmas`: this PRAGMA decides whether a commit is on
    the platter, so an unrecognised value must not fall through to "SQLite default" the way a bad
    ``cache_size`` can. Darwin floor: this runs after :func:`_enforce_macos_synchronous_full`, so a configured
    ``NORMAL`` would silently undo the btree protection — raising is allowed, lowering is refused out loud."""
    if (level := resolve_synchronous_level(raw_value)) is None:
        logger.warning("%s: ignoring unrecognized database.synchronous=%r (expected OFF, NORMAL, FULL, EXTRA, or 0-3)",
                       db_label, raw_value)
        return
    if sys.platform == "darwin" and level < _SYNCHRONOUS_FULL:
        logger.warning("%s: refusing database.synchronous=%s on macOS; keeping FULL. Darwin's fsync() does not "
                       "guarantee write ordering, so a lower level readmits the half-written btree pages FULL exists "
                       "to prevent.", db_label, _SYNCHRONOUS_NAMES[level])
        return
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(f"PRAGMA synchronous={level}")


def apply_database_pragmas(conn: sqlite3.Connection, *, db_label: str = "state.db") -> None:
    """Apply optional performance and WAL-sizing PRAGMAs from ``config.yaml``.

    Journal mode is NOT handled here (owned by :func:`apply_wal_with_fallback`). ``database:`` keys: ``cache_size``
    (negative = KiB, positive = pages), ``mmap_size`` (bytes, 0 = off), ``temp_store`` (0-3), ``wal_autocheckpoint``
    (pages), ``journal_size_limit`` (bytes), ``synchronous`` (unset leaves the compile-time default, which differs
    between bundled/distro/Homebrew builds). Best-effort: failures are ignored so DB init never breaks on a
    malformed section. Applied to ALL connection types: writer, read_only, WAL readers."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly  # local: avoids a circular import
        cfg = load_config_readonly()
    except Exception:
        return
    for pragma_name in ("cache_size", "mmap_size", "temp_store", "wal_autocheckpoint", "journal_size_limit"):
        if (raw_value := cfg_get(cfg, "database", pragma_name, default=None)) is None:
            continue
        try:
            value = int(str(raw_value).strip())
        except (TypeError, ValueError):
            logger.warning("%s: ignoring non-integer database.%s=%r", db_label, pragma_name, raw_value)
            continue
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA {pragma_name}={value}")
    # Last: sizing pragmas cannot change durability (see _apply_synchronous_pragma).
    if (raw_synchronous := cfg_get(cfg, "database", "synchronous", default=None)) is not None:
        _apply_synchronous_pragma(conn, raw_synchronous, db_label=db_label)


def _version_tuple(parts: Iterable[object]) -> tuple[int, int, int]:
    values = [int(part) for part in parts]
    values.extend([0] * (3 - len(values)))
    return tuple(values[:3])


def _is_sqlite_wal_reset_vulnerable(version_info: tuple[int, ...]) -> bool:
    """Return whether *version_info* contains SQLite's WAL-reset bug."""
    info = _version_tuple(version_info)
    return not (
        info < (3, 7, 0)
        or info >= (3, 51, 3)
        or (3, 50, 7) <= info < (3, 51, 0)
        or (3, 44, 6) <= info < (3, 45, 0))


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
    path = Path(path)
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


def read_only_db_uri(db_path) -> str:
    """``file:`` URI for a ``mode=ro`` open. ``as_uri()`` percent-encodes ``?``/``#`` in the home
    path; a raw ``f"file:{path}?mode=ro"`` truncates there and opens the wrong (empty) database."""
    return Path(db_path).resolve().as_uri() + "?mode=ro"


def _read_proc_argv(pid: int) -> Optional[list[str]]:
    """Read /proc/<pid>/cmdline without losing argv boundaries."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
        if not raw:
            return None
        argv = raw.decode("utf-8", "replace").split("\x00")
        if argv[-1] == "":
            argv.pop()
        return argv or None
    except OSError:
        return None


def describe_holder_pid(pid: int) -> str:
    """``PID 123 (hermes gateway run)`` for operator-facing holder lists; /proc argv first, psutil elsewhere."""
    argv = _read_proc_argv(pid)
    if argv is None and psutil is not None:
        try:
            argv = psutil.Process(pid).cmdline() or None
        except Exception:  # health: allow BLE001 -- a process that vanished or hides its argv is described without it
            argv = None
    who = " ".join(" ".join([os.path.basename(argv[0]), *argv[1:]]).split())[:80] if argv else "command line unavailable"
    return f"PID {pid} ({who})"


def secure_db_files(db_path: Path, *, create_main: bool = False) -> None:
    """Create/tighten a writable state database and its sidecars to 0600.

    SQLite otherwise creates ``state.db``, ``-wal``, and ``-shm`` according to
    the process umask (commonly 0644 under 0022). Read-only store
    opens never call this helper and remain observational.

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


def _rw_or_vanished(p: Path) -> bool:
    """True when *p* is read-write for this user OR no longer exists.

    The preflight lists the files first and checks them after, and a live sibling connection's last ``close()``
    unlinks ``-wal``/``-shm`` in between (a concurrent quarantine can rename ``state.db`` aside the same way).
    ``os.access`` on the vanished path returns False, which is "gone", not "read-only". The final re-check covers
    a sidecar that vanished and was recreated by the next opener before ``exists()`` ran.
    """
    mode = os.R_OK | os.W_OK
    return os.access(p, mode) or not p.exists() or os.access(p, mode)


def preflight_db_writability(db_path: Path, *, db_label: str = "state.db", directory_only: bool = False) -> None:
    """Refuse-or-repair read-only DB files BEFORE the first connection opens.

    A stray read-only ``state.db`` / ``-wal`` / ``-shm`` (sudo run, restored backup, copied dotfiles) otherwise
    surfaces as an opaque "attempt to write a readonly database" inside ``_init_schema``, and the obvious wrong
    "fix" (deleting the ``-wal``) loses committed transactions. ``chmod u+rw`` repair only inside the Hermes home
    tree (Hermes owns those files; ``chmod`` fails on files the user doesn't own, bounding the repair exactly);
    otherwise fail fast naming the file and command. Never deletes/truncates a WAL sidecar — once writable, the
    normal open checkpoints it. ``:memory:``/``file:`` skipped. Shared with ``kanban_db``.

    Port of Kilo-Org/kilocode#12508's startup preflight. This preflight:
    """
    if str(db_path) == ":memory:" or str(db_path).startswith("file:"):
        return
    home: Optional[Path] = None
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        home = Path(get_hermes_home()).resolve()
    # SQLite needs a writable directory in every journal mode (WAL/SHM sidecars, or the DELETE-mode journal).
    sidecars = (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm"))
    files = () if directory_only else tuple((p, False) for p in (db_path, *sidecars) if p.is_file())
    for p, is_dir in [(db_path.parent, True), *files]:
        if (is_dir and not p.is_dir()) or _rw_or_vanished(p):
            continue
        x = "x" if is_dir else ""
        in_scope = False
        with contextlib.suppress(OSError, ValueError):
            in_scope = home is not None and p.resolve().is_relative_to(home)
            if in_scope:
                os.chmod(p, p.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if is_dir else 0))
        if in_scope and os.access(p, os.R_OK | os.W_OK):
            logger.info("%s preflight: repaired read-only %s (chmod u+rw%s)", db_label, p, x)
            continue
        wal_note = (" Do NOT delete the -wal file — it contains committed data that "
                    "will be merged into the database once it is writable." if p.name.endswith("-wal") else "")
        raise sqlite3.OperationalError(
            f"{db_label} is not writable: {'directory' if is_dir else 'file'} {p} is read-only for this user. Hermes "
            f"needs read-write access to open the database. Fix with: chmod u+rw{x} '{p}' (files owned by another "
            f"user may need sudo/chown).{wal_note}")


def _reapply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Best-effort (re)application of the macOS write barriers; True if accepted.
    Call before ``VACUUM``/``REINDEX`` once the schema parses: a connection opened
    on a malformed schema could not take them at open time. Never raises."""
    try:
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return True
    except Exception:  # health: allow BLE001 -- schema still unparseable, or anything else: pragmas cannot be set yet
        return False


def apply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Durability barriers for guest users of ``state.db`` that must inherit its owner's journal mode. Also
    applies the configured ``database.synchronous`` level, a per-connection pragma that otherwise only
    rides on the journal-mode setup path guests must not run."""
    ok = _reapply_durability_barriers(conn)
    with contextlib.suppress(Exception):
        from hermes_cli.config import cfg_get, load_config_readonly  # local: avoids an import cycle
        if (raw_synchronous := cfg_get(load_config_readonly(), "database", "synchronous", default=None)) is not None:
            _apply_synchronous_pragma(conn, raw_synchronous, db_label="state.db (guest)")
    return ok


def schema_not_built(exc: BaseException) -> bool:
    """``no such table/column``: FTS5 / core tables not created yet (brand new file mid-init)."""
    return any(m in str(exc).lower() for m in ("no such table", "no such column"))


RETIRED_GENERATION_DIR_SUFFIX = ".retired-wal-"


def is_zeroed_sqlite_file(path: Path, *, probe_bytes: int = 100, force: bool = False) -> bool:
    """True when *path* looks like the #68474 zeroed-state.db signature.

    Only regular files qualify: probing a FIFO/device/socket could block indefinitely.

    Signature: no ``SQLite format 3`` header and no data — either empty (size 0, the total-loss case,
    #97568) or first *probe_bytes* all NUL. Used at SessionDB open and for snapshot diagnostics so a silent
    all-zero file becomes a guided recovery instead of a generic failure.
    """
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
    if not force and has_live_connection(path):
        return False

    head = read_header_bytes_preopen(path, length=max(16, probe_bytes), force=force)
    # Empty or all-NUL header => zeroed; a real header (or unreadable) => not.
    return head is not None and not head.startswith(b"SQLite format 3") and not any(head)
