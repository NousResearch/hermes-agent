"""SQLite integrity verification for backups, quick snapshots and restores; ``hermes_cli.backup`` re-exports it.

A store file is verified by its store admin, any other database here.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from persistence.core.admin import AdminUnsupported
from persistence_assembly import open_store_admin
from sqlite_support import read_only_db_uri

from hermes_cli.backup_restore import _is_store_file
from hermes_cli.backup_sqlite import _close_quietly


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


# --- SQLite integrity verification ---

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
    ``PRAGMA integrity_check`` (or a cheap structural probe above ``max_bytes``). A store file is
    verified by its store admin, which runs every one of those checks."""
    def _done(message: str, valid: bool = False, size: Optional[int] = None) -> dict:
        return {"valid": valid, "message": message, "size": size}
    if _is_store_file(path):
        admin = open_store_admin(path)
        if isinstance(admin, AdminUnsupported):
            return _done(f"{admin.capability} is unsupported by the {admin.backend_kind} storage backend")
        return admin.verify_file(path, max_bytes=max_bytes)
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
