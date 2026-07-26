"""Lock-safe inspection of SQLite database files.

Why this module exists
----------------------
POSIX advisory locks are cancelled **process-wide** by ``close()`` on *any*
file descriptor for that file::

    the close() system call will cancel all POSIX advisory locks on the
    same file for all threads and all file descriptors in the process
    -- https://sqlite.org/howtocorrupt.html#_posix_advisory_locks_canceled_by_a_separate_thread_doing_close_

So a bare ``open(db_path, "rb") ... close()`` on a **live** database silently
drops every lock SQLite holds on it from this process -- including the
EXCLUSIVE lock a ``VACUUM`` is holding while it rewrites the whole file, and
the RESERVED lock an in-flight ``BEGIN IMMEDIATE`` is holding. Other processes
are then free to write into a file that a writer still believes it owns, which
is the documented route to "database disk image is malformed".

Hermes is exactly the topology this hits: gateway, dispatcher, dashboard,
TUI, CLI, cron and kanban workers all open the same ``state.db`` /
``kanban.db``, and several code paths used to byte-probe those files while
connections were live.

The rules
---------
1. **Never** ``open()`` a database file that may have live connections in this
   process. Ask SQLite instead -- :func:`page_count_bytes` reads the same
   header field via ``PRAGMA``, over the existing connection, taking no new
   descriptor.
2. Byte-level probes are only safe **before any connection exists** for that
   path (first-open validation). Route those through
   :func:`read_header_bytes_preopen`, which refuses once a connection has been
   registered for the path.

:func:`track_connection` / :func:`untrack_connection` maintain the registry of
paths with live connections, so rule 2 can be enforced rather than merely
documented.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"

# Offset of the 4-byte big-endian page-count field in the SQLite header.
_HEADER_PAGE_COUNT_OFFSET = 28

_live_lock = threading.RLock()
# resolved path -> number of live connections opened by this process
_live_connections: dict[str, int] = {}


def _key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def track_connection(path: Path | str) -> None:
    """Record that this process now holds a connection to *path*."""
    key = _key(path)
    with _live_lock:
        _live_connections[key] = _live_connections.get(key, 0) + 1


class TrackedConnection(sqlite3.Connection):
    """A ``sqlite3.Connection`` that untracks its path exactly once on close.

    Counting opens is easy; counting closes reliably is not, because callers
    close connections in many places (and some hand them to
    ``contextlib.closing``). Subclassing the connection puts the decrement on
    the one method every close path must go through, so the live-connection
    registry cannot drift upward and permanently disable byte-probes.

    Note ``with conn:`` does NOT close a sqlite3 connection (it only commits or
    rolls back), so this hook is not fired spuriously by transaction scopes.
    """

    _hermes_tracked_path: str | None = None

    def close(self) -> None:
        path = getattr(self, "_hermes_tracked_path", None)
        # Untrack before the actual close: even if close() raises, the
        # descriptor is going away and holding the path open would wedge
        # every future probe.
        if path is not None:
            self._hermes_tracked_path = None
            untrack_connection(path)
        super().close()


def connect_tracked(path: Path | str, **kwargs) -> sqlite3.Connection:
    """``sqlite3.connect`` that registers the connection for the lifetime of the fd.

    Use for any connection to a database whose file might otherwise be
    byte-probed (state.db, kanban.db). The registration is released
    automatically on ``close()``.

    A caller (or a test double) that supplies its own ``factory`` wins: we
    pass ``factory`` positionally-compatible via kwargs only when it is
    absent, and simply skip tracking when the resulting object is not a
    :class:`TrackedConnection`. Tracking is an optimisation for the probe
    guard, never a correctness requirement for opening the database.
    """
    if "factory" not in kwargs:
        kwargs["factory"] = TrackedConnection
    conn = sqlite3.connect(str(path), **kwargs)
    if isinstance(conn, TrackedConnection):
        conn._hermes_tracked_path = _key(path)
        track_connection(path)
    return conn


def untrack_connection(path: Path | str) -> None:
    """Record that one connection to *path* has been closed."""
    key = _key(path)
    with _live_lock:
        remaining = _live_connections.get(key, 0) - 1
        if remaining > 0:
            _live_connections[key] = remaining
        else:
            _live_connections.pop(key, None)


def has_live_connection(path: Path | str) -> bool:
    """Whether this process currently holds any connection to *path*."""
    with _live_lock:
        return _key(path) in _live_connections


def page_count_bytes(conn: sqlite3.Connection) -> Optional[int]:
    """Logical database size in bytes, read through *conn*.

    ``page_count * page_size`` is the same quantity the 4-byte header field at
    offset 28 carries, but reading it via ``PRAGMA`` opens no new file
    descriptor and therefore cannot cancel this process's POSIX locks.

    Returns ``None`` when the pragmas cannot be read.
    """
    try:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    except (sqlite3.Error, TypeError, IndexError) as exc:
        logger.debug("page_count/page_size unavailable: %s", exc)
        return None
    try:
        return int(page_count) * int(page_size)
    except (TypeError, ValueError):
        return None


def file_length_matches_header(conn: sqlite3.Connection) -> Optional[bool]:
    """Whether the file on disk is at least as long as the header claims.

    Detects the "torn extend" shape (file shorter than its own page count)
    without ever opening the database file: the header side comes from
    ``PRAGMA page_count`` over *conn*, and the on-disk side from ``stat()``,
    which takes no descriptor and cannot break locks.

    Returns ``None`` when the check is not applicable (in-memory database,
    unreadable pragmas, or a stat failure).

    Note: in WAL mode a freshly committed page may still live in the ``-wal``
    file, so the main file legitimately lags. Callers must treat this as
    advisory unless the database is in a rollback journal mode.
    """
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    path_str = row[2] if len(row) > 2 else ""
    if not path_str:
        return None  # in-memory or unnamed database

    logical = page_count_bytes(conn)
    if not logical:
        return None
    try:
        actual = os.path.getsize(path_str)
    except OSError:
        return None
    return actual >= logical


def read_header_bytes_preopen(
    path: Path | str,
    *,
    length: int = 100,
    force: bool = False,
) -> Optional[bytes]:
    """Read the first *length* bytes of *path* -- only when no connection is live.

    This is the ONLY sanctioned byte-level read of a database file, and it is
    restricted to first-open validation (is this file a real SQLite database,
    is it zeroed, has it been overwritten by something else). Once any
    connection to *path* exists in this process, the read is refused and
    ``None`` is returned, because the ``close()`` would cancel that
    connection's POSIX locks.

    Set ``force=True`` only for genuinely offline files (quarantined copies,
    snapshot artifacts, archives) that no live connection can reference.
    """
    if not force and has_live_connection(path):
        logger.debug(
            "refusing byte-level read of %s: a live connection exists in this "
            "process and close() would cancel its POSIX locks",
            path,
        )
        return None
    try:
        with open(path, "rb") as handle:
            return handle.read(length)
    except OSError:
        return None
