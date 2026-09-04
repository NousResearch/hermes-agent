"""SQLite journal-mode and PRAGMA policy for state.db (split from hermes_state).

Patchable helpers are looked up as module globals at call time, so tests patch ``hermes_state_wal.<name>``.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, Optional

from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable as _is_sqlite_wal_reset_vulnerable

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


# WAL needs mmap shared memory + fcntl byte-range locks. Network filesystems (NFS, SMB/CIFS, some FUSE, WSL1) raise
# ``locking protocol``; ZFS corrupts the -shm file under concurrent bursts (COW + mmap) -> ``disk I/O error``.
# Either would silently break everything on state.db/kanban.db, so fall back to DELETE (readers block on writes).
# "not authorized": some FUSE mounts block the WAL pragma outright.
_WAL_INCOMPAT_MARKERS = ("locking protocol", "not authorized", "disk i/o error")
# SQLite's default journal_size_limit is -1 (unlimited); see _apply_wal_size_limit.
_WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB

# Once-per-process-per-db_label dedup sets (kanban_db.connect() runs on every kanban operation, so an undeduped
# line would repeat per connection). Tests clear these via ``hermes_state_wal.<name>``.
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()
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
    except Exception:
        return "wal"
    mode = raw.strip().lower() if isinstance(raw, str) else ""
    return mode if mode in ("wal", "delete") else "wal"


class WalUnsupportedError(sqlite3.OperationalError):
    """Raised by :func:`apply_wal_with_fallback` under ``require_wal=True`` when
    the filesystem cannot provide WAL (SQLITE_PROTOCOL raised, or macOS-NFS silent
    refusal). Subclasses ``OperationalError`` so DB-init handlers still catch it."""


def _verify_configured_delete(actual: str) -> str:
    """Raise unless SQLite reported ``delete`` for an explicit operator request."""
    if actual != "delete":
        raise sqlite3.OperationalError(f"could not set configured journal_mode=delete (got {actual or 'no result'})")
    return actual


def apply_wal_with_fallback(conn: sqlite3.Connection, *, db_label: str = "state.db", require_wal: bool = False) -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the mode actually set. Shared by :class:`SessionDB` and ``hermes_cli.kanban_db_connect.connect``.
    WAL-incompatible filesystems either raise ``OperationalError`` ("locking protocol" / "disk I/O error") or —
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
        _apply_wal_companions(conn)
        return "wal"

    if configured == "delete":
        # #68545: honor the canonical database.journal_mode setting. Existing on-disk WAL databases were
        # returned above and are never live-downgraded.
        if current_mode is None:
            # Probe failed (locked/busy): ownership not provably exclusive. Fail loudly.
            raise sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
        return _verify_configured_delete(_set_journal_mode_no_wait(conn, "DELETE"))
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
        _set_journal_mode_no_wait(conn, "DELETE")
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
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            # A concurrent opener appeared between probe and flip: leave the mode as is.
            _log_wal_reset_bug_once(db_label, kept_wal=True, indeterminate=True)
            return current or "delete"
        # Best-effort otherwise: DELETE is already the default for new file-backed DBs.
    if require_delete:
        _verify_configured_delete(actual)
    _log_wal_reset_bug_once(db_label, kept_wal=False)
    return "delete"


def _wal_reset_repair_hint() -> str:
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
    except Exception:
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
    "delete_overridden": (_delete_overridden_warned_lock, "_delete_overridden_warned_paths", logging.ERROR,
        # Never-live-downgrade keeps WAL; without this the operator never learns their delete had no effect.
        "%s: database.journal_mode=delete is configured but the on-disk database is already WAL; keeping WAL (a live "
        "downgrade under open connections can corrupt the DB). To apply journal_mode=DELETE, stop all connections to "
        "this DB and run a one-time offline 'PRAGMA journal_mode=DELETE' on the file. This message fires once per "
        "process per database."),
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
    _log_once("wal_reset_bug", db_label, sqlite3.sqlite_version, sys.executable, action, _wal_reset_repair_hint)


def _log_journal_mode_upgrade_once(db_label: str, previous_mode: str) -> None:
    """Single WARNING per (process, db_label) about a non-WAL -> WAL flip."""
    _log_once("journal_upgrade", db_label, previous_mode, previous_mode)


# Single ERROR per (process, db_label): WAL fallback / configured delete ignored (DB already WAL).
_log_wal_fallback_once = functools.partial(_log_once, "wal_fallback")
_log_configured_delete_overridden_once = functools.partial(_log_once, "delete_overridden")


# Operators write synchronous as a name; mapped so a typo becomes a warning, not a silently different level.
_SYNCHRONOUS_LEVELS: Dict[str, int] = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}
_SYNCHRONOUS_NAMES: Dict[int, str] = {v: k for k, v in _SYNCHRONOUS_LEVELS.items()}
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
