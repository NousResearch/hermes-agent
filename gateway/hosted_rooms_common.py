"""Shared leaf helpers for the gateway hosted-room modules.

Each hosted-room module validates identifiers, bounded integers, exact field sets and
canonical JSON with its own error class and pinned error strings; these helpers take the
error class and message templates as parameters so failures stay byte-identical while the
logic lives once. Must stay a leaf: never import a hosted_room* module (import cycle).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
DbPath = Path | str


def identifier(
    value: Any, *, label: str, error: type[Exception], max_chars: int = 128,
    pattern: re.Pattern[str] | None = IDENTIFIER_RE, invalid: str | None = None) -> str:
    """Strip and validate a bounded string; ``pattern=None`` skips the shape check."""
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or (pattern is not None and not pattern.fullmatch(value)):
        raise error(invalid or f"invalid {label}")
    return value


def bounded_int(value: Any, *, error: type[Exception], message: str, low: int = 0, high: int | None = None) -> int:
    """Reject bools, non-ints and values outside ``[low, high]`` (``message`` is the exact text)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise error(message)
    return value


def exact_fields(
    value: Any, *, label: str, required: frozenset[str] | set[str], optional: frozenset[str] | set[str] = frozenset(),
    error: type[Exception], not_object: str | None = None, missing_fmt: str = "{label} is missing fields: {fields}",
    unknown_fmt: str = "{label} has unknown fields: {fields}") -> Mapping[str, Any]:
    """Require exactly ``required`` (+ any ``optional``) keys; formats name the offenders sorted."""
    if not isinstance(value, Mapping):
        raise error(not_object or f"{label} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise error(missing_fmt.format(label=label, fields=", ".join(sorted(missing))))
    if unknown:
        raise error(unknown_fmt.format(label=label, fields=", ".join(sorted(unknown))))
    return value


def text(value: Any, *, error: type[Exception], label: str, max_bytes: int, strip: bool = True) -> str:
    """Non-blank string bounded by ``max_bytes`` of UTF-8; ``strip=False`` keeps and measures the raw text."""
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    if not value.strip():
        raise error(f"{label} must not be empty")
    if strip:
        value = value.strip()
    if len(value.encode("utf-8")) > max_bytes:
        raise error(f"{label} is too large")
    return value


def compact_json(value: Any, *, ensure_ascii: bool = True) -> str:
    """Sorted-key, separator-free JSON (the digest/storage canonical form)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any, *, error: type[Exception], label: str, max_bytes: int, ensure_ascii: bool) -> str:
    """``compact_json`` bounded by ``max_bytes`` of UTF-8; unserializable input raises ``error``."""
    try:
        encoded = compact_json(value, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError, RecursionError) as exc:
        raise error(f"{label} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise error(f"{label} is too large")
    return encoded


def utf8_len(*parts: str) -> int:
    return len("".join(parts).encode("utf-8"))


def clock(now: float | None) -> float:
    """``now`` as a float, or the current wall clock when ``None``."""
    return time.time() if now is None else float(now)


def open_sqlite(path: DbPath, *, timeout: float = 10) -> sqlite3.Connection:
    """Row-factory connection with foreign keys on; no journal or schema work."""
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def connect(
    db_path: DbPath, *, db_label: str, ready: Callable[[sqlite3.Connection], bool],
    initialize: Callable[[sqlite3.Connection], None], lock_retries: int = 1) -> sqlite3.Connection:
    """Open the shared root store: WAL, foreign keys, then ``initialize`` in one IMMEDIATE txn if not ``ready``.

    Multiple profile gateways share this database, so every draft-schema transition
    is serialized in SQLite itself: a crash rolls back the whole DDL/data migration and
    another process can safely retry it. Only the transient "database is locked" class
    from the journal-mode pragma is retried (it may ignore the busy timeout while another
    first opener initializes the DB, especially on Windows).
    """
    from hermes_state_wal import apply_wal_with_fallback
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        for attempt in range(lock_retries):
            try:
                apply_wal_with_fallback(conn, db_label=db_label)
                break
            except sqlite3.OperationalError as exc:
                if str(exc).lower() != "database is locked" or attempt + 1 == lock_retries:
                    raise
                time.sleep(0.01 * (2**attempt))
        conn.execute("PRAGMA foreign_keys=ON")
        if not ready(conn):
            conn.execute("BEGIN IMMEDIATE")
            initialize(conn)
            conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


def fenced_update(conn: sqlite3.Connection, sql: str, params: tuple, error: Exception) -> None:
    """Run a compare-and-swap UPDATE; anything but exactly one row means the fence was lost."""
    if conn.execute(sql, params).rowcount != 1:
        raise error


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


@contextmanager
def transaction(
    connect: Callable[[DbPath], sqlite3.Connection], db_path: DbPath, *, immediate: bool
) -> Iterator[sqlite3.Connection]:
    """Open via ``connect``, optionally ``BEGIN IMMEDIATE``, commit on success, always close."""
    conn = connect(db_path)
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
