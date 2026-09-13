"""PostgreSQL execution boundary for latency-bounded session browsing."""

from __future__ import annotations

import math


def read_bounded_sessions_postgres(conn, query, params, *, timeout_seconds: float):
    """Read with a server deadline that cannot leak to later statements.

    The caller owns SessionDB's read context, so no other operation can share
    this connection until rollback releases the read-only transaction. A zero
    PostgreSQL timeout means unlimited, so even a zero caller budget gets the
    smallest positive server deadline. Rollback also clears an aborted query
    and restores the connection's previous timeout on every exit path.
    """
    timeout_ms = max(1, math.ceil(min(timeout_seconds * 1000, 2_147_483_647)))
    conn.execute("BEGIN READ ONLY")
    try:
        conn.execute("SELECT set_config('statement_timeout', ?, true)", (f"{timeout_ms}ms",))
        return conn.execute(query, params).fetchall()
    except Exception as exc:
        if getattr(exc, "sqlstate", None) == "57014":
            raise TimeoutError(
                f"recent-session browse exceeded {timeout_seconds:g}s deadline"
            ) from exc
        raise
    finally:
        conn.rollback()
