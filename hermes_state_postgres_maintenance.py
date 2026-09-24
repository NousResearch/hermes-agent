"""Session retention for PostgreSQL; the server owns physical storage maintenance."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO
from hermes_state_maintenance import _seconds_since


logger = logging.getLogger("hermes_state")


class SessionPostgresMaintenanceMixin:
    def logical_size_bytes(self) -> int:
        """Include this profile's tables, indexes and TOAST storage only."""
        return int(self._read_one(
            "SELECT COALESCE(SUM(pg_total_relation_size(c.oid)), 0) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = ? AND c.relkind = 'r'",
            (self.schema,))[0])

    def _try_wal_checkpoint(self) -> None:
        """PostgreSQL checkpoints are managed by the server, across all databases."""

    def _try_checkpoint(self, mode: str, fail_msg: str) -> None:
        """SQLite checkpoint requests have no per-profile PostgreSQL equivalent."""

    def vacuum(self) -> int:
        """Leave physical maintenance to PostgreSQL without claiming reclaimed space."""
        return 0

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
        """Require an explicit PostgreSQL backup before replacing SQLite's snapshot step."""
        cleanup = super().purge_stale_tool_call_markers
        if dry_run or not backup:
            return cleanup(dry_run=dry_run, backup=False)
        result = cleanup(dry_run=True, backup=False)
        if result["rows_affected"]:
            raise ValueError(
                "PostgreSQL marker cleanup requires a pg_dump backup; take the backup, "
                "then retry with backup=False.")
        result["dry_run"] = False
        return result

    def maybe_auto_prune_and_vacuum(
        self, retention_days: int = 90, min_interval_hours: int = 24, vacuum: bool = True,
        sessions_dir: Optional[Path] = None, min_vacuum_interval_days: int = 30,
        min_vacuum_freelist_ratio: float = AUTO_VACUUM_MIN_FREELIST_RATIO,
    ) -> Dict[str, Any]:
        """Coordinate retention across clients without SQLite file locks or VACUUM."""
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False}
        if self.read_only:
            result["skipped"] = True
            return result
        lock_key = f"{self.schema}:maintenance"
        try:
            with self._pool.connection() as conn:
                acquired = conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (lock_key,)).fetchone()[0]
                if not acquired:
                    result["skipped"] = True
                    return result
                try:
                    now = time.time()
                    since_prune = _seconds_since(now, self.get_meta("last_auto_prune"))
                    if since_prune is not None and since_prune < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                    result["pruned"] = self.prune_sessions(
                        older_than_days=retention_days, sessions_dir=sessions_dir, exclude_active_write_guards=True)
                    closed = self.sweep_orphaned_sessions(
                        max_idle_seconds=float(retention_days) * 86400,
                        sources=self._AUTO_PRUNE_STALE_OPEN_SOURCES, exclude_pinned=True,
                        respect_gateway_heartbeats=False)
                    result["closed"] = len(closed)
                    self.set_meta("last_auto_prune", str(now))
                finally:
                    conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (lock_key,))
        except Exception as exc:
            logger.warning("PostgreSQL session maintenance failed: %s", exc)
            result["error"] = str(exc)
        return result
