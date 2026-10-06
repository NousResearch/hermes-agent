"""WAL-aware read-only SQLite helpers for live Hermes databases.

Rules (lane contract):
  * mode=ro — never write
  * do NOT use immutable=1 (would skip WAL and miss live writes)
  * explicit busy_timeout (default 5000ms)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_BUSY_TIMEOUT_MS = 5000


def open_readonly(
    path: Path | str,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open ``path`` read-only with WAL visibility and busy timeout."""
    p = Path(path)
    uri = f"file:{p}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=max(0, busy_timeout_ms) / 1000.0)
    conn.row_factory = sqlite3.Row
    # PRAGMA busy_timeout is advisory for this connection; timeout= above is the
    # primary wait. Set both so helpers and callers agree.
    conn.execute(f"PRAGMA busy_timeout={max(0, int(busy_timeout_ms))}")
    return conn


def busy_timeout_of(conn: sqlite3.Connection) -> Optional[int]:
    """Return current PRAGMA busy_timeout in ms, or None on failure."""
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        if row is None:
            return None
        return int(row[0])
    except Exception:
        return None
