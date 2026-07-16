"""Away-mode kill switch for the morning brief / dev-todo pipeline.

A single global flag (there is exactly one row, id=1) stored in the
``away_state`` table of the same ``todos.db`` used by
:mod:`plugins.life_ops.todo_store` — see that module's docstring for the
broader persistent-todo-store context. When away mode is active, the caller
(a later integration step) is expected to skip the todo-nagging parts of the
brief.

No LLM/agent-client imports: stdlib + sqlite3 + hermes_cli.sqlite_util only.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

from hermes_cli.sqlite_util import write_txn
from plugins.life_ops.todo_store import todos_db_path

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS away_state (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    active  INTEGER NOT NULL DEFAULT 0,
    until   TEXT,
    set_at  TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the ``away_state`` table.

    Shares ``todos.db`` with :mod:`plugins.life_ops.todo_store` but only ever
    touches its own table — schema init here is scoped to ``away_state`` so
    the two modules stay independently owned.
    """
    path = db_path if db_path is not None else todos_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA_SQL)
    return conn


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _get_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT active, until, set_at FROM away_state WHERE id=1").fetchone()


def set_away(until: str | None) -> None:
    """Turn away mode on. ``until`` is an ISO date string, or None for indefinite."""
    conn = connect()
    try:
        with write_txn(conn):
            conn.execute(
                "INSERT INTO away_state (id, active, until, set_at) VALUES (1, 1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET active=1, until=excluded.until, "
                "set_at=excluded.set_at",
                (until, _now_iso()),
            )
    finally:
        conn.close()


def clear_away() -> None:
    """Turn away mode off."""
    conn = connect()
    try:
        with write_txn(conn):
            conn.execute(
                "INSERT INTO away_state (id, active, until, set_at) VALUES (1, 0, NULL, ?) "
                "ON CONFLICT(id) DO UPDATE SET active=0, until=NULL, set_at=excluded.set_at",
                (_now_iso(),),
            )
    finally:
        conn.close()


def is_away(today: str) -> bool:
    """True when away mode is active and not expired.

    Side effect: an active-but-expired flag (``until < today``) is
    auto-cleared via :func:`clear_away` and this returns False.
    """
    today = str(today or "")
    conn = connect()
    try:
        row = _get_row(conn)
    finally:
        conn.close()
    if row is None or not row["active"]:
        return False
    until = row["until"]
    if until is None or today <= until:
        return True
    clear_away()
    return False


def away_status(today: str) -> dict:
    """``{"active": bool, "until": str|None}`` reflecting post-expiry-check state."""
    is_away(today)  # applies auto-expiry side effect first
    conn = connect()
    try:
        row = _get_row(conn)
    finally:
        conn.close()
    if row is None:
        return {"active": False, "until": None}
    return {"active": bool(row["active"]), "until": row["until"]}
