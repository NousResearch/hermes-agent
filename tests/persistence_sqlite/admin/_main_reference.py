"""Main's code for the reference side of the admin parity cases whose application function now calls the admin: each block is main's, unchanged, from the module its comment names, with the imports it needs. The helpers this tree carries unchanged are imported, not copied."""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Optional

from hermes_cli.backup_restore import _restore_db_pages
from hermes_cli.backup_sqlite import _close_quietly, _safe_copy_db
from hermes_cli.doctor_report import check_info
from hermes_state_holders import read_only_db_uri

# The logger of the module the restore functions come from.
logger = logging.getLogger("hermes_cli.backup_restore")


# hermes_cli/doctor_state.py
def _session_count(state_db_path: Path):
    import sqlite3
    # mode=ro: doctor is a reader; a writable open of a gateway-held WAL DB is the second-writer class (#103339).
    conn = sqlite3.connect(read_only_db_uri(state_db_path), uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


# Above this the snapshot copy a held store needs costs more than the probe is worth; --fix still probes.
_WRITE_PROBE_SNAPSHOT_MAX_BYTES = 1 << 30


def _write_health_reason(state_db_path: Path, *, should_fix: bool):
    """FTS/write-health probe (a rolled-back BEGIN IMMEDIATE). Against a store a live writer holds,
    that probe is the second-writer class (#103339), so probe a read-only snapshot instead; a quiet
    store is probed in place. Returns the failure reason, or None when healthy or skipped."""
    from hermes_state_repair import _db_opens_cleanly, _live_writer_holds_db
    if not _live_writer_holds_db(state_db_path):
        return _db_opens_cleanly(state_db_path)
    if not should_fix and state_db_path.stat().st_size > _WRITE_PROBE_SNAPSHOT_MAX_BYTES:
        check_info("state.db write-health probe skipped: store is held by a live writer and larger than 1 GB "
                   "(run 'hermes doctor --fix' to probe it)")
        return None
    import sqlite3
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "state.db"
        src = sqlite3.connect(read_only_db_uri(state_db_path), uri=True, timeout=1.0)
        try:
            dest = sqlite3.connect(str(snapshot))
            try:
                src.backup(dest)
            finally:
                dest.close()
        finally:
            src.close()
        return _db_opens_cleanly(snapshot)


# hermes_cli/backup.py
def _query_ro_sqlite(path: Path, fn):
    """Run ``fn(conn)`` on a read-only connection to *path*; return ``(value, None)`` or ``(None, exc)``."""
    conn = None
    try:
        conn = sqlite3.connect(read_only_db_uri(path), uri=True, timeout=1.0)
        return fn(conn), None
    except Exception as exc:
        return None, exc
    finally:
        _close_quietly(conn)


_SQLITE_HEADER = b"SQLite format 3\0"


# Above this size ``PRAGMA integrity_check`` (walks every b-tree page — minutes of pegged CPU on a
# 30 GB state.db, reading as a hung ``hermes update``) is replaced by the O(1) header+schema probe.
# Default ceiling above which ``PRAGMA integrity_check`` is skipped in favour of the (O(1)) header +
# structural probe. Sessions databases in the tens of GB are normal for heavy users, so the size-unbounded
# check is never an acceptable default on the update path. See #70553.
DEFAULT_INTEGRITY_CHECK_MAX_BYTES = 2 << 30  # 2 GiB


def verify_sqlite_integrity(
    path: Path, *, check_header: bool = True, run_pragma: bool = True,
    max_bytes: int = DEFAULT_INTEGRITY_CHECK_MAX_BYTES) -> dict:
    """Verify a SQLite database: existence + minimum size, header magic, then a read-only
    ``PRAGMA integrity_check`` (or a cheap structural probe above ``max_bytes``)."""
    def _done(message: str, valid: bool = False, size: Optional[int] = None) -> dict:
        return {"valid": valid, "message": message, "size": size}
    try:
        st = path.stat()
    except FileNotFoundError:
        return _done(f"not found: {path}")
    except OSError as exc:
        return _done(f"cannot stat: {exc}")
    size = st.st_size
    if size < 100:  # SQLite minimum viable size (header + 1 page)
        return _done(f"too small ({size} bytes) to be a valid SQLite database", size=size)
    if check_header:
        # Refused when a live connection exists (close() would cancel this process's POSIX locks
        # — see sqlite_safe_read); verification targets offline snapshots/backup artifacts anyway.
        from hermes_cli.sqlite_safe_read import read_header_bytes_preopen
        head = read_header_bytes_preopen(path, length=len(_SQLITE_HEADER))
        if head is None:
            return _done("cannot read header", size=size)
        if head != _SQLITE_HEADER:
            return _done(f"missing SQLite header magic (got {head[:16].hex()!r})", size=size)
    if max_bytes > 0 and size > max_bytes:
        # O(1) probe: the header check caught the zeroed signature; reading sqlite_master + page
        # geometry catches malformed-schema and truncated-header-page classes without a data walk.
        _, exc = _query_ro_sqlite(path, lambda c: (
            c.execute("PRAGMA schema_version").fetchone(),
            c.execute("SELECT count(*) FROM sqlite_master").fetchone()))
        if exc is not None:
            kind = "failed" if isinstance(exc, sqlite3.DatabaseError) else "error"
            return _done(f"schema probe {kind}: {exc}", size=size)
        return _done(
            f"size {size:,} bytes exceeds max_bytes {max_bytes:,}; "
            "skipped PRAGMA integrity_check (header + schema probe passed)",
            valid=True, size=size)
    if run_pragma:
        rows, exc = _query_ro_sqlite(
            path, lambda c: [str(r[0]) for r in c.execute("PRAGMA integrity_check")])
        if exc is not None:
            kind = "cannot open database" if isinstance(exc, sqlite3.DatabaseError) else "integrity check error"
            return _done(f"{kind}: {exc}", size=size)
        if rows == ["ok"]:
            return _done("integrity check passed", valid=True, size=size)
        return _done(f"integrity check failed: {'; '.join(rows[:5])}", size=size)
    return _done("header check passed", valid=True, size=size)


# hermes_cli/backup_restore.py
def _restore_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(read_only_db_uri(path), uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_epoch'").fetchone():
            return 0
        row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
        return int(row[0]) if row else 0


def _restore_destination_epoch(dst: Path) -> int:
    try:
        return _restore_epoch(dst)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise OSError(
            f"Cannot read the canonical runtime epoch at {dst}; refusing in-place restore. "
            "Use `hermes sessions recover --source <snapshot> --output <separate-path>` "
            "to salvage transcripts into a separate output."
        ) from exc


def _safe_restore_db(src: Path, dst: Path) -> bool:
    """Restore only while holding the destination authority's maintenance reservation.

    Returns ``False``, without touching *dst*, when *src* fails the SQLite
    integrity check, when the runtime owns the profile, or when a canonical
    ``state.db`` epoch floor cannot be proven.
    """
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance

    # backup() copies pages without validating their contents; its fallback
    # copies bytes even when SQLite rejected the source. Neither may touch the
    # destination until the snapshot passes the existing bounded integrity policy.
    source_check = verify_sqlite_integrity(src)
    if not source_check["valid"]:
        logger.error("Refusing SQLite restore from %s: %s", src, source_check["message"])
        return False
    try:
        with exclusive_maintenance([dst.absolute().parent, dst.resolve().parent]):
            with _restore_epoch_source(src, dst) as prepared:
                return _restore_db_pages(prepared, dst)
    except (OwnershipConflict, OSError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return False


@contextmanager
def _restore_epoch_source(src: Path, dst: Path):
    """Stage an epoch floor into the image BEFORE publishing it atomically.

    Updating the destination after backup() would leave a crash window that
    reuses an older worker epoch. Never modify the user's source snapshot.
    Full restores retain all admissions/worker receipts; normal startup advances
    this floor and reconciles started work to unknown, not replayable queued work.
    An unreadable destination cannot prove its epoch floor: use transcript salvage
    into a separate output rather than silently restoring with a recycled epoch.
    """
    if dst.name != 'state.db' and dst.resolve().name != 'state.db':
        yield src
        return
    floor = _restore_destination_epoch(dst)
    if floor <= _restore_epoch(src):
        yield src
        return
    with tempfile.TemporaryDirectory(prefix='.restore-epoch-', dir=dst.parent) as work:
        prepared = Path(work) / 'state.db'
        if not _safe_copy_db(src, prepared):
            raise OSError('Unable to stage a restore with a safe runtime epoch')
        with closing(sqlite3.connect(prepared)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_epoch (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                epoch INTEGER NOT NULL CHECK (epoch > 0), instance_id TEXT NOT NULL)""")
            conn.execute("""INSERT INTO runtime_epoch VALUES(1,?, 'offline-restore')
                ON CONFLICT(singleton) DO UPDATE SET epoch=excluded.epoch,
                instance_id=excluded.instance_id""", (floor,))
            conn.commit()
        prepared.chmod(src.stat().st_mode)
        yield prepared


def _count_session_rows(path: Path) -> Optional[tuple[int, int]]:
    """Return ``(sessions, messages)`` stored in the session database *path*.

    Read-only and best effort.  ``None`` means "unknown" — a missing file, a
    database that is not a Hermes session store, or one that cannot be read.
    Callers must never read ``None`` as "zero rows": acting on an unreadable
    database would mask the very loss this count exists to surface.  Same
    contract as :func:`_count_cron_jobs`.
    """
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(read_only_db_uri(path), uri=True)
    except sqlite3.Error:
        return None
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return int(sessions), int(messages)
    except (sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        conn.close()


# hermes_cli/update_downgrade_guard.py
_LEDGER_PROBE = ("SELECT EXISTS(SELECT 1 FROM session_admissions) "
                 "OR EXISTS(SELECT 1 FROM worker_executions)")


def _has_ledger_rows(db_path: Path) -> bool:
    if not db_path.is_file():
        return False
    from hermes_state_holders import read_only_db_uri
    try:
        with closing(sqlite3.connect(read_only_db_uri(db_path), uri=True, timeout=5.0)) as conn:
            return bool(conn.execute(_LEDGER_PROBE).fetchone()[0])
    except sqlite3.Error:
        return False  # no ledger tables yet, or unreadable: nothing an older release would trip on


def _pre_runtime_snapshot(home: Path) -> str | None:
    """Newest quick snapshot of ``home`` holding a state.db from before the runtime ledger."""
    from hermes_cli.backup import _quick_snapshot_root
    from hermes_state_holders import read_only_db_uri
    root = _quick_snapshot_root(home)
    for snap in sorted(root.iterdir(), reverse=True) if root.is_dir() else ():
        db = snap / "state.db"
        if snap.name.startswith(".") or not db.is_file():
            continue
        try:
            with closing(sqlite3.connect(read_only_db_uri(db), uri=True)) as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_admissions'").fetchone() is None:
                    return snap.name
        except sqlite3.Error:
            continue
    return None
