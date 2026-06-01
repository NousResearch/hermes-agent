from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP("__SERVER_NAME__")

DATABASE_PATH = os.getenv("SQLITE_PATH", "./app.db")
MAX_ROWS = int(os.getenv("SQLITE_MAX_ROWS", "200"))
MAX_PROGRESS_CALLS = max(1, int(os.getenv("SQLITE_MAX_PROGRESS_CALLS", "20000")))
PROGRESS_GRANULARITY = max(1, int(os.getenv("SQLITE_PROGRESS_GRANULARITY", "1000")))
TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)


def _reject_mutation(sql: str) -> None:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")


def _validate_table_name(table_name: str) -> str:
    if not TABLE_NAME_RE.fullmatch(table_name):
        raise ValueError("Invalid table name")
    return table_name


def _progress_budget(max_calls: int) -> Callable[[], int]:
    state = {"remaining": max_calls}

    def _progress_handler() -> int:
        state["remaining"] -= 1
        return 1 if state["remaining"] <= 0 else 0

    return _progress_handler


@mcp.tool
def list_tables() -> list[str]:
    """List user-defined SQLite tables."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


@mcp.tool
def describe_table(table_name: str) -> list[dict[str, Any]]:
    """Describe columns for a SQLite table."""
    safe_table_name = _validate_table_name(table_name)
    with _connect() as conn:
        rows = conn.execute(f"PRAGMA table_info({safe_table_name})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


@mcp.tool
def query(sql: str, limit: int = 50) -> dict[str, Any]:
    """Run a read-only SELECT query and return rows plus column names."""
    _reject_mutation(sql)
    safe_limit = max(0, min(limit, MAX_ROWS))
    with _connect() as conn:
        progress_handler = _progress_budget(MAX_PROGRESS_CALLS)
        conn.set_progress_handler(progress_handler, PROGRESS_GRANULARITY)
        try:
            cursor = conn.execute(sql.strip())
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise ValueError("Query exceeded the execution budget") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
        columns = [column[0] for column in cursor.description or []]
        rows = [dict(zip(columns, row)) for row in cursor.fetchmany(safe_limit)]
    return {"limit": safe_limit, "columns": columns, "rows": rows}


if __name__ == "__main__":
    mcp.run()
