"""Installation-wide persistent provider-quota circuit breaker.

The quota state is deliberately separate from each Kanban board so two boards
using the same normalized provider/model domain cannot stampede an exhausted
provider. The caller's board connection remains an API compatibility parameter;
state authority is the Hermes-home ``ship-crew-quota.db`` file.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class QuotaState:
    domain: str
    failure_count: int
    open_until: int
    last_error: Optional[str]

    @property
    def open(self) -> bool:
        return self.open_until > int(time.time())


def quota_db_path() -> Path:
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / "ship-crew-quota.db"


@contextmanager
def _quota_connection() -> Iterator[sqlite3.Connection]:
    path = quota_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        ensure_quota_schema(conn)
        yield conn
    finally:
        conn.close()


def ensure_quota_schema(conn=None) -> None:
    """Create the shared quota tables; ``conn`` is accepted for API parity."""
    target = conn
    if target is None:
        with _quota_connection() as owned:
            ensure_quota_schema(owned)
        return
    target.executescript(
        """CREATE TABLE IF NOT EXISTS ship_crew_quota_domains (
            domain TEXT PRIMARY KEY,
            failure_count INTEGER NOT NULL DEFAULT 0,
            open_until INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ship_crew_quota_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            event TEXT NOT NULL,
            actor TEXT,
            detail TEXT,
            created_at INTEGER NOT NULL
        );"""
    )
    target.commit()


def _row_state(row: Optional[sqlite3.Row], domain: str) -> QuotaState:
    if row is None:
        return QuotaState(domain, 0, 0, None)
    return QuotaState(row["domain"], int(row["failure_count"]), int(row["open_until"]), row["last_error"])


def quota_state(conn, domain: str) -> QuotaState:
    del conn  # shared authority is intentionally independent of board selection
    domain = str(domain)
    with _quota_connection() as qconn:
        row = qconn.execute(
            "SELECT domain, failure_count, open_until, last_error FROM ship_crew_quota_domains WHERE domain=?",
            (domain,),
        ).fetchone()
        return _row_state(row, domain)


def quota_available(conn, domain: Optional[str], *, now: Optional[int] = None) -> bool:
    del conn
    if not domain:
        return True
    now = int(time.time()) if now is None else int(now)
    with _quota_connection() as qconn:
        row = qconn.execute(
            "SELECT open_until FROM ship_crew_quota_domains WHERE domain=?", (str(domain),)
        ).fetchone()
        return row is None or int(row["open_until"]) <= now


def record_quota_failure(
    conn,
    domain: str,
    *,
    retry_after_seconds: int = 60,
    threshold: int = 3,
    error: str = "quota",
    now: Optional[int] = None,
) -> QuotaState:
    del conn
    domain = str(domain).strip()
    if not domain:
        raise ValueError("quota domain is required")
    if threshold < 1 or retry_after_seconds < 0:
        raise ValueError("threshold must be positive and retry delay non-negative")
    now = int(time.time()) if now is None else int(now)
    with _quota_connection() as qconn:
        qconn.execute("BEGIN IMMEDIATE")
        try:
            row = qconn.execute(
                "SELECT failure_count FROM ship_crew_quota_domains WHERE domain=?", (domain,)
            ).fetchone()
            failures = (int(row["failure_count"]) if row else 0) + 1
            open_until = now + int(retry_after_seconds) if failures >= threshold else 0
            qconn.execute(
                """INSERT INTO ship_crew_quota_domains(domain, failure_count, open_until, last_error, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(domain) DO UPDATE SET failure_count=excluded.failure_count,
                     open_until=excluded.open_until, last_error=excluded.last_error,
                     updated_at=excluded.updated_at""",
                (domain, failures, open_until, str(error)[:500], now),
            )
            qconn.execute(
                "INSERT INTO ship_crew_quota_events(domain, event, actor, detail, created_at) VALUES (?, 'failure', NULL, ?, ?)",
                (domain, str(error)[:500], now),
            )
            qconn.commit()
        except Exception:
            qconn.rollback()
            raise
    return quota_state(None, domain)


def record_quota_success(conn, domain: str, *, now: Optional[int] = None) -> QuotaState:
    del conn
    now = int(time.time()) if now is None else int(now)
    domain = str(domain)
    with _quota_connection() as qconn:
        qconn.execute("BEGIN IMMEDIATE")
        try:
            qconn.execute(
                """INSERT INTO ship_crew_quota_domains(domain, failure_count, open_until, last_error, updated_at)
                   VALUES (?, 0, 0, NULL, ?)
                   ON CONFLICT(domain) DO UPDATE SET failure_count=0, open_until=0,
                     last_error=NULL, updated_at=excluded.updated_at""",
                (domain, now),
            )
            qconn.execute(
                "INSERT INTO ship_crew_quota_events(domain, event, actor, detail, created_at) VALUES (?, 'success', NULL, NULL, ?)",
                (domain, now),
            )
            qconn.commit()
        except Exception:
            qconn.rollback()
            raise
    return quota_state(None, domain)


def reset_quota(conn, domain: str, *, actor: str, now: Optional[int] = None) -> QuotaState:
    """Audited manual reset for an operator-authorized quota circuit."""
    del conn
    actor = str(actor).strip()
    if not actor:
        raise ValueError("actor is required for manual quota reset")
    now = int(time.time()) if now is None else int(now)
    domain = str(domain)
    with _quota_connection() as qconn:
        qconn.execute("BEGIN IMMEDIATE")
        try:
            qconn.execute(
                """INSERT INTO ship_crew_quota_domains(domain, failure_count, open_until, last_error, updated_at)
                   VALUES (?, 0, 0, NULL, ?)
                   ON CONFLICT(domain) DO UPDATE SET failure_count=0, open_until=0,
                     last_error=NULL, updated_at=excluded.updated_at""",
                (domain, now),
            )
            qconn.execute(
                "INSERT INTO ship_crew_quota_events(domain, event, actor, detail, created_at) VALUES (?, 'manual_reset', ?, 'operator reset', ?)",
                (domain, actor, now),
            )
            qconn.commit()
        except Exception:
            qconn.rollback()
            raise
    return quota_state(None, domain)
