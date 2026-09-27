"""SQLite integrity verification for backups, snapshots and the updater.

Split out of ``hermes_cli.backup`` (a facade that re-exports ``verify_sqlite_integrity``, which old
updaters import by that name — see ``tests/compat/old_updater_surface.json``)."""

import sqlite3
from pathlib import Path
from typing import Optional

from hermes_cli.backup_sqlite import _close_quietly
from hermes_cli.sqlite_safe_read import connect_tracked, has_live_connection, read_header_bytes_preopen
from hermes_state_holders import read_only_db_uri


def _query_ro_sqlite(path: Path, fn):
    """Run ``fn(conn)`` on a read-only connection to *path*; return ``(value, None)`` or ``(None, exc)``."""
    conn = None
    try:
        conn = connect_tracked(read_only_db_uri(path), uri=True, timeout=1.0, tracking_path=path)
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
    ``PRAGMA integrity_check`` (or a cheap structural probe above ``max_bytes``).

    ``valid`` is tri-state and callers must compare it with ``is False`` to accuse the
    database: ``True`` passed, ``False`` corrupt, ``None`` indeterminate. A live connection
    in this process makes the byte-level header probe unavailable, which costs the header
    answer only — the size ceiling and ``PRAGMA`` below open their own read-only connection
    and still reach a verdict, so they keep running. ``None`` is returned only when the
    header probe was skipped AND nothing deeper was asked for, i.e. there was no check left
    to run rather than a check that came back bad.
    """
    # Set only when a live connection made the byte probe unavailable. Losing the byte probe
    # is not a verdict: the checks below open their own read-only connection and are unaffected,
    # so they must still run — otherwise one tracked handle would hide real deep corruption,
    # which is exactly the class of database that admits a live connection.
    header_note: Optional[str] = None

    def _done(message: str, valid: Optional[bool] = False, size: Optional[int] = None) -> dict:
        # Every reported verdict carries the skipped byte probe, so a reader of ``message``
        # alone knows which checks actually ran behind it. The membership test is the one
        # case where the note is already the whole message, not something to append to.
        if header_note and header_note not in message:
            message = f"{message} ({header_note})"
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
        if has_live_connection(path):
            # read_header_bytes_preopen signals its refusal with None, the same value it returns
            # for an unreadable file, so reading that as "invalid" made a healthy database read
            # as corrupt — the updater then declared state.db corrupt and refused its own repair
            # with that same connection. Fall through to the deeper checks instead of judging here.
            header_note = "header check skipped: a live connection in this process owns this database"
        else:
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
        # Name only the checks that actually ran. Behind a live connection the byte probe
        # never executed, so crediting it here would assert evidence this call never gathered
        # (and would contradict the skipped-probe note _done appends to the same line).
        passed = "schema probe" if header_note else "header + schema probe"
        return _done(
            f"size {size:,} bytes exceeds max_bytes {max_bytes:,}; "
            f"skipped PRAGMA integrity_check ({passed} passed)",
            valid=True, size=size)
    if run_pragma:
        # The pragma walks every page over a fresh read-only connection, so a skipped byte
        # probe costs it nothing: this verdict is as strong with a live handle as without.
        rows, exc = _query_ro_sqlite(
            path, lambda c: [str(r[0]) for r in c.execute("PRAGMA integrity_check")])
        if exc is not None:
            kind = "cannot open database" if isinstance(exc, sqlite3.DatabaseError) else "integrity check error"
            return _done(f"{kind}: {exc}", size=size)
        if rows == ["ok"]:
            return _done("integrity check passed", valid=True, size=size)
        return _done(f"integrity check failed: {'; '.join(rows[:5])}", size=size)
    if header_note:
        # The byte probe was the only check this call asked for and it could not run: no verdict
        # was reached, which is not the same as a database that failed one. Callers compare with
        # ``is False``, so this skips rather than accusing.
        return _done(header_note, valid=None, size=size)
    return _done("header check passed", valid=True, size=size)
