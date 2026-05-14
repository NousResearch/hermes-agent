"""Dead-provider registry — persistent tracking of unreachable providers.

When a provider repeatedly fails (max retries exceeded, consecutive errors,
or unresolvable credential exhaustion), it's marked as "dead" and added to
this registry. The session launch process can then skip the dead provider
and go straight to the first healthy fallback, avoiding the costly retry
delay.

A periodic health check probe tests dead providers and marks them alive
when they respond again — enabling fully automatic recovery.

Architecture:
  - SQLite-backed for persistence across agent restarts.
  - TTL-based expiry: each dead entry has a configurable TTL (default 300s).
  - Thread-safe writes via WAL mode and retry-on-lock.
  - No external dependencies beyond Python stdlib (sqlite3, threading, time).

Integration points:
  - Fallback chain traversal in run_agent.py (filter out dead entries)
  - _try_activate_fallback() — mark dead when all retries exhausted
  - Periodic health check cron (optional, in-process or via cronjob tool)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# Default TTL for a dead provider: 5 minutes.
# After this time the provider is auto-revived (not checked — just presumed
# potentially alive again).  The health-check probe can revive it sooner.
DEFAULT_DEAD_TTL_SECONDS: int = 300

# SQLite table schema
_SCHEMA = """
CREATE TABLE IF NOT EXISTS dead_providers (
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    marked_at   REAL NOT NULL,
    ttl_seconds INTEGER NOT NULL DEFAULT {ttl},
    PRIMARY KEY (provider, model)
)
""".format(ttl=DEFAULT_DEAD_TTL_SECONDS)

_COLUMNS = ("provider", "model", "reason", "marked_at", "ttl_seconds")
_DB_FILENAME = "dead_providers.db"


# ── Data class ────────────────────────────────────────────────────────────


@dataclass
class DeadProviderRecord:
    """A single entry in the dead-provider registry."""

    provider: str
    model: str
    reason: str
    marked_at: float = field(default_factory=time.monotonic)
    ttl_seconds: int = DEFAULT_DEAD_TTL_SECONDS

    def is_dead(self) -> bool:
        """Return True if this entry's TTL has not yet expired."""
        if self.ttl_seconds <= 0:
            return False
        return (time.monotonic() - self.marked_at) < self.ttl_seconds

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "marked_at": self.marked_at,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "DeadProviderRecord":
        return cls(
            provider=str(d.get("provider", "")),
            model=str(d.get("model", "")),
            reason=str(d.get("reason", "")),
            marked_at=float(float(d.get("marked_at", time.monotonic()))),  # type: ignore[arg-type]
            ttl_seconds=int(int(d.get("ttl_seconds", DEFAULT_DEAD_TTL_SECONDS))),  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return (
            f"DeadProviderRecord(provider={self.provider!r}, model={self.model!r}, "
            f"reason={self.reason!r}, ttl={self.ttl_seconds}s, "
            f"age={time.monotonic() - self.marked_at:.1f}s)"
        )


# ── Helper to resolve the default DB path ─────────────────────────────────


def _default_db_path() -> str:
    """Return the default path for the dead-providers SQLite database."""
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home() / _DB_FILENAME)
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".hermes", _DB_FILENAME)


# ── Registry ──────────────────────────────────────────────────────────────


class DeadProviderRegistry:
    """Persistent registry of dead providers, backed by SQLite.

    Thread-safe.  Uses WAL mode and retry-on-lock to handle concurrent
    access from multiple agent instances.

    Usage::

        reg = DeadProviderRegistry()
        reg.mark_provider_dead("openai", "gpt-4", "5 consecutive 500s")
        if reg.is_provider_dead("openai", "gpt-4"):
            skip_it()

        reg.mark_provider_alive("openai", "gpt-4")
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or _default_db_path())
        self._lock = threading.Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """Create the table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    # ── Core operations ───────────────────────────────────────────────────

    def mark_provider_dead(
        self,
        provider: str,
        model: str = "",
        reason: str = "",
        ttl_seconds: int = DEFAULT_DEAD_TTL_SECONDS,
    ) -> None:
        """Mark a (provider, model) pair as dead.

        If an entry already exists, it is updated (new timestamp, reason,
        and TTL).  This is an UPSERT — the primary key is (provider, model).
        """
        provider = (provider or "").strip().lower()
        model = (model or "").strip().lower()
        if not provider:
            logger.warning("mark_provider_dead called with empty provider — ignored")
            return

        marked_at = time.monotonic()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO dead_providers (provider, model, reason, marked_at, ttl_seconds)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(provider, model) DO UPDATE SET
                           reason      = excluded.reason,
                           marked_at   = excluded.marked_at,
                           ttl_seconds = excluded.ttl_seconds""",
                    (provider, model, reason, marked_at, ttl_seconds),
                )
                conn.commit()

        logger.info(
            "Provider %s/%s marked dead: %s (TTL=%ds)",
            provider, model, reason, ttl_seconds,
        )

    def mark_provider_alive(self, provider: str, model: str = "") -> None:
        """Remove a (provider, model) pair from the dead registry."""
        provider = (provider or "").strip().lower()
        model = (model or "").strip().lower()
        if not provider:
            return

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM dead_providers WHERE provider=? AND model=?",
                    (provider, model),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info("Provider %s/%s revived (marked alive)", provider, model)

    def is_provider_dead(self, provider: str, model: str = "") -> bool:
        """Return True if the (provider, model) is in the dead registry
        and its TTL has not expired."""
        entry = self.get_dead_entry(provider, model)
        return entry is not None

    def get_dead_entry(self, provider: str, model: str = "") -> Optional[DeadProviderRecord]:
        """Return the dead entry for (provider, model), or None if alive/expired."""
        provider = (provider or "").strip().lower()
        model = (model or "").strip().lower()
        if not provider:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT provider, model, reason, marked_at, ttl_seconds "
                "FROM dead_providers WHERE provider=? AND model=?",
                (provider, model),
            ).fetchone()

        if row is None:
            return None

        rec = DeadProviderRecord(
            provider=row["provider"],
            model=row["model"],
            reason=row["reason"],
            marked_at=row["marked_at"],
            ttl_seconds=row["ttl_seconds"],
        )
        if not rec.is_dead():
            return None
        return rec

    def list_dead_providers(self) -> List[DeadProviderRecord]:
        """Return all non-expired dead entries.

        Expired entries are purged from the database during this call.
        """
        now = time.monotonic()
        records: List[DeadProviderRecord] = []

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT provider, model, reason, marked_at, ttl_seconds "
                    "FROM dead_providers"
                ).fetchall()

                alive: List[DeadProviderRecord] = []
                for row in rows:
                    rec = DeadProviderRecord(
                        provider=row["provider"],
                        model=row["model"],
                        reason=row["reason"],
                        marked_at=row["marked_at"],
                        ttl_seconds=row["ttl_seconds"],
                    )
                    if rec.is_dead():
                        alive.append(rec)

                # Purge expired entries
                ttl_columns = ", ".join(
                    f"({col} + {now} - {now})" for col in ("marked_at",)
                )
                conn.execute(
                    "DELETE FROM dead_providers WHERE (marked_at + ttl_seconds) <= ?",
                    (now,),
                )
                conn.commit()

        return alive

    def filter_alive(
        self,
        candidates: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Filter a list of (provider, model) tuples to only those that are
        not dead.

        Use this when traversing the fallback chain to skip dead providers::

            filtered = registry.filter_alive(fallback_chain_entries)
            for provider, model in filtered:
                ...
        """
        if not candidates:
            return []
        dead_set = set()
        for entry in self.list_dead_providers():
            dead_set.add((entry.provider, entry.model))
        return [c for c in candidates if c not in dead_set]

    def dead_count(self) -> int:
        """Return the number of currently dead (non-expired) providers."""
        return len(self.list_dead_providers())

    def clear(self) -> int:
        """Remove all entries from the dead-provider table.
        Returns the number of rows deleted."""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM dead_providers")
                conn.commit()
                return cursor.rowcount

    def _add_raw(self, rec: DeadProviderRecord) -> None:
        """Insert a record directly, bypassing the time-based TTL check.
        Used by tests to simulate expired entries."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO dead_providers "
                    "(provider, model, reason, marked_at, ttl_seconds) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rec.provider, rec.model, rec.reason,
                     rec.marked_at, rec.ttl_seconds),
                )
                conn.commit()

    def __enter__(self) -> "DeadProviderRegistry":
        return self

    def __exit__(self, *args: object) -> None:
        pass  # SQLite connections are per-call, nothing to close


# ── Protocol for health-check checker ────────────────────────────────────


class _ProviderChecker(Protocol):
    """Minimal protocol: anything with a health_check method."""

    def health_check(self, provider: str, model: str) -> bool: ...


# ── Health check probe ────────────────────────────────────────────────────


class HealthCheckProbe:
    """Periodic health checker that tests dead providers and revives them.

    The probe needs two things:
    1. A ``DeadProviderRegistry`` to read the dead list and update status.
    2. A ``checker`` object with a ``health_check(provider, model) -> bool``
       method.

    Typical usage::

        from agent.dead_provider_registry import DeadProviderRegistry, HealthCheckProbe

        class MyProviderChecker:
            def health_check(self, provider: str, model: str) -> bool:
                # Make a cheap API call (e.g. list models, or a tiny chat)
                return True  # or False

        reg = DeadProviderRegistry()
        checker = MyProviderChecker()
        probe = HealthCheckProbe(reg, checker)
        results = probe.check_all()  # checks all dead providers

    The ``checker`` abstraction decouples the probe from any specific
    provider SDK — the agent's integration code provides the actual
    health-check logic.
    """

    def __init__(self, registry: DeadProviderRegistry, checker: _ProviderChecker):
        self.registry = registry
        self.checker = checker

    def check_once(self, provider: str, model: str) -> bool:
        """Check a single (provider, model).  Returns True if alive."""
        ok = bool(self.checker.health_check(provider, model))
        if ok:
            self.registry.mark_provider_alive(provider, model)
            logger.info("Health check: %s/%s revived — provider is responsive again", provider, model)
        else:
            logger.info(
                "Health check: %s/%s still dead — provider not responding",
                provider, model,
            )
        return ok

    def check_all(self) -> List[bool]:
        """Check all currently dead providers.

        Returns a list of booleans, one per checked provider, where True
        means the provider was revived.
        """
        dead_providers = self.registry.list_dead_providers()
        if not dead_providers:
            logger.debug("Health check: no dead providers to check")
            return []

        results: List[bool] = []
        for entry in dead_providers:
            try:
                ok = self.check_once(entry.provider, entry.model)
                results.append(ok)
            except Exception as exc:
                logger.warning(
                    "Health check error for %s/%s: %s",
                    entry.provider, entry.model, exc,
                )
                results.append(False)

        alive_count = sum(1 for r in results if r)
        if alive_count:
            logger.info(
                "Health check complete: %d/%d providers revived",
                alive_count, len(results),
            )
        else:
            logger.debug(
                "Health check complete: 0/%d providers revived",
                len(results),
            )
        return results