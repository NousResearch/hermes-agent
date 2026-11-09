"""SQLite's text spelling of a REAL, as a TEXT column stores a float bound into it, asked of the installed ``sqlite3`` library.

The spelling differs between SQLite builds (older ones keep 15 significant digits, newer ones the shortest text that reads back as the same double), so the reference stores the value in a TEXT column of an in-memory database and reads back what the library it runs on wrote.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Optional

__all__ = ["real_as_text"]


def real_as_text(value: float) -> Optional[str]:
    """``value`` as SQLite stores a REAL in a TEXT column; ``None`` for NaN, which binds as NULL."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE spelled (value TEXT)")
        conn.execute("INSERT INTO spelled (value) VALUES (?)", (value,))
        return conn.execute("SELECT value FROM spelled").fetchone()[0]
