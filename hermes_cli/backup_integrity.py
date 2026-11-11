"""SQLite integrity verification for backups and the post-update state.db guard.

``valid`` is tri-state: ``True`` passed, ``False`` corrupt, ``None`` indeterminate
(a lock or temporary unreadable condition prevented a firm verdict). Callers that
accuse a database must compare with ``is False``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from hermes_state_holders import read_only_db_uri

from hermes_cli.backup_sqlite import _close_quietly

# Above this size ``PRAGMA integrity_check`` (walks every b-tree page) is replaced by
# the O(1) header+schema probe. See #70553.
DEFAULT_INTEGRITY_CHECK_MAX_BYTES = 2 << 30  # 2 GiB

_SQLITE_HEADER = b"SQLite format 3\0"
_LOCK_PRIMARY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
# Open/read access failures are not corruption — even when the raw header was readable
# (Windows sharing / AV / filter drivers can allow a short raw read then refuse SQLite).
_ACCESS_PRIMARY_CODES = frozenset({sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_IOERR})
_CORRUPT_PRIMARY_CODES = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})


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


def _primary_code(exc: BaseException) -> Optional[int]:
    """SQLite primary result code (low byte of the extended code), or None."""
    code = getattr(exc, "sqlite_errorcode", None)
    return code & 0xFF if isinstance(code, int) else None


def _is_lock_contention(exc: BaseException) -> bool:
    """True when *exc* is SQLite BUSY/LOCKED."""
    return _primary_code(exc) in _LOCK_PRIMARY_CODES


def verify_sqlite_integrity(
    path: Path, *, check_header: bool = True, run_pragma: bool = True,
    max_bytes: int = DEFAULT_INTEGRITY_CHECK_MAX_BYTES) -> dict:
    """Verify a SQLite database: existence + size, header magic, then a read-only probe.

    A refused, ``OSError``-failed, or short/torn header read is not corruption: fall
    through to the read-only ``PRAGMA``/schema probe. Probe ``CANTOPEN``/``IOERR`` (and
    lock contention) stay ``valid=None`` even when the raw header was readable. Only a
    full-length wrong magic, or a probe reporting ``NOTADB``/``CORRUPT`` (or another
    non-access failure), counts as ``valid=False``. When the byte probe is unavailable
    and nothing deeper can judge, return ``valid=None``.
    """
    header_note: Optional[str] = None

    def _done(message: str, valid: Optional[bool] = False, size: Optional[int] = None) -> dict:
        if header_note and header_note not in message:
            message = f"{message} ({header_note})"
        return {"valid": valid, "message": message, "size": size}

    def _query_failure(message: str, exc: Exception, size: int) -> dict:
        code = _primary_code(exc)
        # Lock or open/IO access failure: no corruption verdict, regardless of whether
        # the raw header was readable (#147938 / kvnloo counterexample).
        if _is_lock_contention(exc) or code in _ACCESS_PRIMARY_CODES:
            return _done(message, valid=None, size=size)
        # Header unreadable AND a non-corruption probe error: still no verdict.
        if header_note and code not in _CORRUPT_PRIMARY_CODES:
            return _done(message, valid=None, size=size)
        return _done(message, size=size)

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
        # Refused when a live connection exists (close() would cancel this process's POSIX
        # locks — see sqlite_safe_read). An OSError (Windows sharing conflict included)
        # returns the same None; a short/torn read is treated the same way. None of these
        # are evidence of corruption on a live home (#147938).
        from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
        head = read_header_bytes_preopen(path, length=len(_SQLITE_HEADER))
        if head is None:
            if has_live_connection(path):
                header_note = (
                    "header check skipped: a live connection in this process owns this database"
                )
            else:
                header_note = "header check skipped: file temporarily unreadable"
        elif len(head) < len(_SQLITE_HEADER):
            # A torn/short read is not proof of bad magic — same deferral as OSError (#147938).
            header_note = "header check skipped: incomplete header read"
        elif head != _SQLITE_HEADER:
            return _done(f"missing SQLite header magic (got {head[:16].hex()!r})", size=size)
    if max_bytes > 0 and size > max_bytes:
        # O(1) probe: schema_version + sqlite_master without a full data walk.
        _, exc = _query_ro_sqlite(path, lambda c: (
            c.execute("PRAGMA schema_version").fetchone(),
            c.execute("SELECT count(*) FROM sqlite_master").fetchone()))
        if exc is not None:
            kind = "failed" if isinstance(exc, sqlite3.DatabaseError) else "error"
            return _query_failure(f"schema probe {kind}: {exc}", exc, size)
        passed = "schema probe" if header_note else "header + schema probe"
        return _done(
            f"size {size:,} bytes exceeds max_bytes {max_bytes:,}; "
            f"skipped PRAGMA integrity_check ({passed} passed)",
            valid=True, size=size)
    if run_pragma:
        rows, exc = _query_ro_sqlite(
            path, lambda c: [str(r[0]) for r in c.execute("PRAGMA integrity_check")])
        if exc is not None:
            kind = (
                "cannot open database" if isinstance(exc, sqlite3.DatabaseError)
                else "integrity check error"
            )
            return _query_failure(f"{kind}: {exc}", exc, size)
        if rows == ["ok"]:
            return _done("integrity check passed", valid=True, size=size)
        return _done(f"integrity check failed: {'; '.join(rows[:5])}", size=size)
    if header_note:
        # Byte probe was the only check asked for and it could not run: no verdict.
        return _done(header_note, valid=None, size=size)
    return _done("header check passed", valid=True, size=size)
