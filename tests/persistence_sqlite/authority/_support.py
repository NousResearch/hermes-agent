"""Shared helpers for the authority and worker backend tests: copy a profile store, dump the rows of its authority tables, capture a call's outcome, and count the transactions a verb opens on its holder."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

#: The tables the authority and worker transactions write.
AUTHORITY_TABLES = ("sessions", "session_admissions", "worker_executions", "worker_receipts", "state_meta",
                    "runtime_epoch", "messages")


def copy_db(src: Path, dst: Path) -> None:
    """A consistent copy of ``src`` at ``dst`` through the SQLite backup API."""
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def dump(path: Path, tables: Iterable[str] = AUTHORITY_TABLES) -> dict[str, Any]:
    """Every row of each named table that exists, in rowid order."""
    with sqlite3.connect(path) as conn:
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in tables if table in present}


def outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", value)`` or ``("error", exception)``."""
    try:
        return "ok", call()
    except Exception as exc:  # the caller inspects the exception
        return "error", exc


@contextmanager
def counted_transactions(holder: Any) -> Iterator[dict[str, int]]:
    """Count the write transactions (``_execute_write``) and read contexts (``_read_ctx``) opened on ``holder``
    while the block runs; a write's lock replays count once."""
    counts = {"write": 0, "read": 0}
    write, read = holder._execute_write, holder._read_ctx

    def counted_write(*args: Any, **kwargs: Any) -> Any:
        counts["write"] += 1
        return write(*args, **kwargs)

    def counted_read(*args: Any, **kwargs: Any) -> Any:
        counts["read"] += 1
        return read(*args, **kwargs)

    holder._execute_write, holder._read_ctx = counted_write, counted_read
    try:
        yield counts
    finally:
        del holder._execute_write, holder._read_ctx
