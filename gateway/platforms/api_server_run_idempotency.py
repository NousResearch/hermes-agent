"""Durable idempotency reservations for API server runs."""

import hmac
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict


# Keep the extracted store's log records on the API server logger.
logger = logging.getLogger("gateway.platforms.api_server")


class RunIdempotencyStore:
    """Durable, tenant-scoped reservations for ``POST /v1/runs``.

    A unique ``(scope, key)`` row is inserted inside ``BEGIN IMMEDIATE`` so
    separate gateway workers/processes cannot both admit the same request.
    Only request fingerprints and public run status are stored; request bodies
    and credentials are deliberately excluded.
    """

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60

    @property
    def durable(self) -> bool:
        """Whether reservations survive this process."""
        return self._db_path is not None

    def __init__(self, db_path: str = None):
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home

                db_path = str(get_hermes_home() / "runs_idempotency.db")
            except Exception:
                db_path = ":memory:"
        self._db_path = None if db_path == ":memory:" else db_path
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        except Exception as exc:
            logger.warning(
                "Run idempotency storage is unavailable; falling back to "
                "process memory, so replay will not survive a restart: %s",
                exc,
            )
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0,
                owner_started INTEGER NOT NULL DEFAULT 0,
                retention_until REAL NOT NULL DEFAULT 0,
                acknowledged_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )"""
        )
        columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(run_idempotency)")
        }
        if "owner_pid" not in columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0"
            )
        if "owner_started" not in columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN owner_started INTEGER NOT NULL DEFAULT 0"
            )
        if "retention_until" not in columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN "
                "retention_until REAL NOT NULL DEFAULT 0"
            )
        if "acknowledged_at" not in columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN acknowledged_at REAL"
            )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS run_idempotency_run_id ON run_idempotency(run_id)"
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self._tighten_permissions()

    def _tighten_permissions(self) -> None:
        if not self._db_path:
            return
        for candidate in (
            Path(self._db_path),
            Path(self._db_path + "-wal"),
            Path(self._db_path + "-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "Failed to restrict run idempotency store permissions",
                    exc_info=True,
                )

    def reserve(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        run_id: str,
        status: Dict[str, Any],
        *,
        owner_pid: int = 0,
        owner_started: int = 0,
        retention_until: float = 0,
    ):
        """Atomically reserve a key; return ``(outcome, stored_record)``."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = json.dumps(status, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._prune_stale_terminal_locked(now)
                row = self._conn.execute(
                    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
                    "FROM run_idempotency WHERE scope=? AND idempotency_key=?",
                    (scope, key),
                ).fetchone()
                if row is not None:
                    if retention_until:
                        self._conn.execute(
                            """UPDATE run_idempotency
                                  SET retention_until=MAX(retention_until, ?)
                                WHERE scope=? AND idempotency_key=?
                                  AND fingerprint=?""",
                            (retention_until, scope, key, fingerprint),
                        )
                    self._conn.commit()
                    outcome = (
                        "reused"
                        if hmac.compare_digest(row[0], fingerprint)
                        else "conflict"
                    )
                    return outcome, {
                        "run_id": row[1],
                        "status": json.loads(row[2]),
                        "owner_pid": int(row[3] or 0),
                        "owner_started": int(row[4] or 0),
                        "updated_at": float(row[5] or 0),
                    }
                self._conn.execute(
                    "INSERT INTO run_idempotency("
                    "scope,idempotency_key,fingerprint,run_id,status_json,"
                    "owner_pid,owner_started,retention_until,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        scope,
                        key,
                        fingerprint,
                        run_id,
                        encoded,
                        int(owner_pid or 0),
                        int(owner_started or 0),
                        retention_until,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
                return "created", {
                    "run_id": run_id,
                    "status": status,
                    "owner_pid": int(owner_pid or 0),
                    "owner_started": int(owner_started or 0),
                    "updated_at": now,
                }
            except Exception:
                self._conn.rollback()
                raise

    def lookup(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        retention_until: float = 0,
    ):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if retention_until:
                    self._conn.execute(
                        """UPDATE run_idempotency
                              SET retention_until=MAX(retention_until, ?)
                            WHERE scope=? AND idempotency_key=?
                              AND fingerprint=?""",
                        (retention_until, scope, key, fingerprint),
                    )
                self._prune_stale_terminal_locked(now)
                row = self._conn.execute(
                    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
                    "FROM run_idempotency WHERE scope=? AND idempotency_key=?",
                    (scope, key),
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        if row is None:
            return "missing", None
        outcome = "reused" if hmac.compare_digest(row[0], fingerprint) else "conflict"
        return outcome, {
            "run_id": row[1],
            "status": json.loads(row[2]),
            "owner_pid": int(row[3] or 0),
            "owner_started": int(row[4] or 0),
            "updated_at": float(row[5] or 0),
        }

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune replay records only after their stored run is terminal.

        The caller owns ``self._lock`` and an active transaction. Age alone
        can never release an in-flight idempotency reservation: a long or
        disconnected room turn may legitimately outlive the retention window.
        """
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json, retention_until,
                      acknowledged_at, updated_at
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (
                now - self.ACKNOWLEDGED_RETENTION_SECONDS,
                now,
                now - self.RETENTION_SECONDS,
            ),
        ).fetchall()
        for (
            stale_scope,
            stale_key,
            stale_status,
            retention_until,
            acknowledged_at,
            updated_at,
        ) in stale:
            try:
                terminal = json.loads(stale_status).get("status") in {
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                }
            except Exception:
                terminal = False
            expired = bool(
                (
                    acknowledged_at is not None
                    and float(acknowledged_at)
                    <= now - self.ACKNOWLEDGED_RETENTION_SECONDS
                )
                or (
                    float(retention_until or 0) > 0
                    and now >= float(retention_until)
                )
                or (
                    float(retention_until or 0) <= 0
                    and float(updated_at or 0) < now - self.RETENTION_SECONDS
                )
            )
            if terminal and expired:
                self._conn.execute(
                    """DELETE FROM run_idempotency
                         WHERE scope=? AND idempotency_key=?""",
                    (stale_scope, stale_key),
                )

    def status_for_run(
        self,
        scope: str,
        run_id: str,
        *,
        retention_until: float = 0,
    ) -> dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            if retention_until:
                self._conn.execute(
                    """UPDATE run_idempotency
                          SET retention_until=MAX(retention_until, ?)
                        WHERE scope=? AND run_id=?""",
                    (retention_until, scope, run_id),
                )
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": json.loads(row[0]),
            "owner_pid": int(row[1] or 0),
            "owner_started": int(row[2] or 0),
            "updated_at": float(row[3] or 0),
        }

    def acknowledge_terminal(self, scope: str, run_id: str) -> bool:
        """Allow cleanup once the room home durably imported terminal output."""
        now = time.time()
        with self._lock:
            changed = self._conn.execute(
                """UPDATE run_idempotency SET acknowledged_at=?
                     WHERE scope=? AND run_id=?""",
                (now, scope, run_id),
            ).rowcount
            self._conn.commit()
        return changed == 1

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            changed = self._conn.execute(
                """UPDATE run_idempotency
                      SET retention_until=MAX(retention_until, ?)
                    WHERE scope=? AND run_id=?""",
                (checked_until, scope, run_id),
            ).rowcount
            self._conn.commit()
        return changed == 1

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?",
                    (scope, run_id),
                ).fetchone()
                is not None
            )

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        encoded = json.dumps(status, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (encoded, time.time(), run_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
