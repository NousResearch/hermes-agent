"""Conservative, lock-aware validation of a live state.db during updates.

Unlike offline backup verification, a live database may refuse raw header reads
while another connection holds it. Only confirmed corruption permits recovery.
"""

import sqlite3
from pathlib import Path

_SQLITE_HEADER = b"SQLite format 3\0"


def _result(valid: bool | None, message: str, size: int | None = None) -> dict:
    return {"valid": valid, "message": message, "size": size}


def _probe_failure(exc: Exception, stage: str, size: int) -> dict:
    """Busy, locked, and I/O errors are not proof of a corrupt database."""
    code = getattr(exc, "sqlite_errorcode", None)
    corrupt = isinstance(code, int) and (code & 0xFF) in (
        sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
    )
    return _result(False if corrupt else None, f"{stage}: {exc}", size)


def verify_update_state_db(path: Path) -> dict:
    """Return valid=True/False/None (healthy/corrupt/indeterminate).

    The raw header read is advisory: live SQLite holders (including Windows
    sharing conflicts) can make it unavailable, so fall through to the SQLite
    read-only probe. Do not restore a database when the verdict is unknown.
    """
    from hermes_cli.backup import DEFAULT_INTEGRITY_CHECK_MAX_BYTES, _query_ro_sqlite
    from hermes_cli.sqlite_safe_read import read_header_bytes_preopen

    try:
        if not path.is_file():
            return _result(None, f"not a regular database file: {path}")
        size = path.stat().st_size
    except OSError as exc:
        return _result(None, f"cannot inspect database: {exc}")
    if size < 100:
        return _result(False, f"too small ({size} bytes) for SQLite", size)

    header = read_header_bytes_preopen(path, length=len(_SQLITE_HEADER))
    if header is not None and header != _SQLITE_HEADER:
        return _result(False, f"invalid SQLite header ({header.hex()})", size)

    if size > DEFAULT_INTEGRITY_CHECK_MAX_BYTES:
        # Bounded structural probe rather than an unbounded full B-tree walk.
        _, exc = _query_ro_sqlite(path, lambda c: (
            c.execute("PRAGMA schema_version").fetchone(),
            c.execute("SELECT count(*) FROM sqlite_master").fetchone(),
        ))
        if exc is not None:
            return _probe_failure(exc, "schema probe unavailable", size)
        return _result(True, "SQLite schema probe passed (large database)", size)

    rows, exc = _query_ro_sqlite(
        path, lambda c: [str(row[0]) for row in c.execute("PRAGMA integrity_check")],
    )
    if exc is not None:
        return _probe_failure(exc, "integrity probe unavailable", size)
    if rows == ["ok"]:
        return _result(True, "integrity check passed", size)
    if not rows:
        return _result(None, "integrity check returned no result", size)
    return _result(False, f"integrity check failed: {'; '.join(rows[:5])}", size)
