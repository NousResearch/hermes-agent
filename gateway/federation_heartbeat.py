"""P2P federation heartbeat for Hermes multi-device coordination.

Implements the heartbeat + offline-detection + task-claim loop that
enables automatic task relay when a device in the federation goes offline.
All devices are peers; there is no primary/secondary role.

Architecture:
- A shared SQLite database (on iCloud Drive or other synced storage) holds
  device heartbeats and task state.
- Every device runs the same 3-phase loop every ``interval_s`` seconds:
  1. **Heartbeat** — update own ``last_seen`` (no lock, idempotent).
  2. **Offline detection** — check for stale peers; if found, mark them
     offline and set their tasks to ``pending_reassign`` (uses
     ``BEGIN IMMEDIATE`` so only one device executes).
  3. **Task claim** — if idle, atomically claim one ``pending_reassign``
     task (also uses ``BEGIN IMMEDIATE``).

The ``BEGIN IMMEDIATE`` pattern guarantees that when multiple devices
simultaneously detect the same offline peer, exactly one wins the lock
and performs the state transition; others see the changed data after
the lock releases and skip harmlessly.

Configuration lives in ``config.yaml`` under ``federation:`` — no new
environment variables are introduced.

Example config::

    federation:
      enabled: true
      db_path: ~/Library/Mobile Documents/com~apple~CloudDocs/hermes-federation/federation.db
      offline_threshold_s: 30
      heartbeat_interval_s: 60

See also: `docs/federation.md` (user guide), ``tests/federation/test_heartbeat.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FederationConfig:
    """Federation heartbeat configuration (config.yaml, not env vars)."""

    enabled: bool = False
    # Path to the shared SQLite database.  iCloud Drive is the default on
    # macOS because it provides automatic file sync across Apple devices.
    db_path: Optional[str] = None
    # Seconds without a heartbeat before a device is considered offline.
    offline_threshold_s: int = 30
    # Seconds between heartbeat cycles.
    heartbeat_interval_s: int = 60

    def resolve_db_path(self) -> Optional[Path]:
        """Resolve the database path, expanding ~ and env vars."""
        if not self.db_path:
            # Default to iCloud Drive on macOS.
            icloud = Path.home() / "Library" / "Mobile Documents" / \
                "com~apple~CloudDocs" / "hermes-federation"
            default = icloud / "federation.db"
            if default.parent.exists():
                return default
            return None
        raw = os.path.expandvars(os.path.expanduser(self.db_path))
        return Path(raw)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS device_heartbeats (
    device_id       TEXT PRIMARY KEY,
    hostname        TEXT,
    status          TEXT DEFAULT 'online',
    cpu_cores       INTEGER DEFAULT 0,
    memory_gb       REAL DEFAULT 0,
    load_avg        REAL DEFAULT 0,
    current_task_id TEXT,
    last_seen       REAL NOT NULL,
    ip_address      TEXT,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS federation_tasks (
    task_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT DEFAULT '',
    status          TEXT DEFAULT 'pending',
    priority        INTEGER DEFAULT 3,
    assigned_device TEXT,
    source_device   TEXT,
    created_at      REAL NOT NULL,
    started_at      REAL,
    heartbeat_at    REAL,
    completed_at    REAL,
    context_snapshot TEXT DEFAULT '{}',
    result_data     TEXT DEFAULT '{}',
    error_info      TEXT DEFAULT '',
    fail_count      INTEGER DEFAULT 0,
    max_retries     INTEGER DEFAULT 3
);
"""


def _get_connection(db_path: Path) -> Optional[sqlite3.Connection]:
    """Open a SQLite connection with WAL mode and busy timeout."""
    if not db_path.parent.exists():
        logger.debug("Federation db parent dir does not exist: %s", db_path.parent)
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except sqlite3.OperationalError as e:
        logger.debug("Federation db connection failed: %s", e)
        return None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist (idempotent)."""
    conn.executescript(_SCHEMA_SQL)


def _device_id() -> str:
    """Return a stable device identifier."""
    env_id = os.environ.get("HERMES_DEVICE_ID")
    if env_id:
        return env_id
    try:
        import subprocess
        name = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if name:
            return name.replace(" ", "-")
    except Exception:
        pass
    return socket.gethostname()


# ---------------------------------------------------------------------------
# Phase 1: Heartbeat (no lock, idempotent)
# ---------------------------------------------------------------------------


def _heartbeat(conn: sqlite3.Connection, device_id: str) -> None:
    """Update this device's heartbeat record."""
    now = time.time()
    info = _gather_device_info()
    exists = conn.execute(
        "SELECT 1 FROM device_heartbeats WHERE device_id=?",
        (device_id,),
    ).fetchone()

    if exists:
        conn.execute(
            "UPDATE device_heartbeats SET status='online', hostname=?, "
            "cpu_cores=?, memory_gb=?, load_avg=?, ip_address=?, last_seen=? "
            "WHERE device_id=?",
            (
                info["hostname"], info["cpu_cores"], info["memory_gb"],
                info["load_avg"], info["ip_address"], now, device_id,
            ),
        )
    else:
        conn.execute(
            "INSERT INTO device_heartbeats "
            "(device_id, hostname, status, cpu_cores, memory_gb, load_avg, "
            "ip_address, last_seen, created_at) "
            "VALUES (?, ?, 'online', ?, ?, ?, ?, ?, ?)",
            (
                device_id, info["hostname"], info["cpu_cores"],
                info["memory_gb"], info["load_avg"], info["ip_address"],
                now, now,
            ),
        )
    conn.commit()


def _gather_device_info() -> Dict[str, Any]:
    """Gather lightweight device metrics."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_cores": os.cpu_count() or 0,
        "memory_gb": 0.0,
        "load_avg": 0.0,
        "ip_address": "",
    }
    try:
        load = os.getloadavg()
        info["load_avg"] = round(load[0], 2)
    except (OSError, AttributeError):
        pass
    try:
        import subprocess
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        info["memory_gb"] = round(int(mem) / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["ip_address"] = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Phase 2: Offline detection (BEGIN IMMEDIATE)
# ---------------------------------------------------------------------------


def _detect_offline(
    conn: sqlite3.Connection, device_id: str, threshold: float,
) -> list[dict]:
    """Detect offline peers and mark their tasks for reassignment.

    Uses ``BEGIN IMMEDIATE`` so that when multiple devices run this
    concurrently, only one executes the state transitions; the others
    see the updated data after the lock releases and skip.
    """
    now = time.time()
    cutoff = now - threshold
    relays: list[dict] = []

    try:
        conn.execute("BEGIN IMMEDIATE")

        offline = conn.execute(
            "SELECT device_id, hostname, current_task_id, last_seen "
            "FROM device_heartbeats "
            "WHERE status='online' AND last_seen < ? AND device_id != ?",
            (cutoff, device_id),
        ).fetchall()

        if not offline:
            conn.execute("ROLLBACK")
            return []

        for row in offline:
            peer_id = row["device_id"]
            logger.warning(
                "Federation: peer '%s' (%s) offline (last_seen=%.0fs ago)",
                peer_id, row["hostname"], now - row["last_seen"] if row["last_seen"] else 0,
            )
            conn.execute(
                "UPDATE device_heartbeats SET status='offline' "
                "WHERE device_id=?",
                (peer_id,),
            )

            task_id = row["current_task_id"]
            if task_id:
                conn.execute(
                    "UPDATE federation_tasks SET status='pending_reassign', "
                    "fail_count=fail_count+1, "
                    "error_info=? "
                    "WHERE task_id=? AND status IN ('assigned','in_progress')",
                    (
                        f"Peer {peer_id} offline, triggering relay",
                        task_id,
                    ),
                )
                t = conn.execute(
                    "SELECT task_id, title FROM federation_tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if t:
                    relays.append({
                        "device": peer_id,
                        "hostname": row["hostname"],
                        "task_id": t["task_id"],
                        "task_title": t["title"],
                    })

            conn.execute(
                "UPDATE device_heartbeats SET current_task_id=NULL "
                "WHERE device_id=?",
                (peer_id,),
            )

        conn.commit()
        return relays

    except sqlite3.OperationalError:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Phase 3: Task claim (BEGIN IMMEDIATE)
# ---------------------------------------------------------------------------


def _claim_task(conn: sqlite3.Connection, device_id: str) -> Optional[dict]:
    """Atomically claim one pending_reassign task if this device is idle.

    Returns the task dict if claimed, None if no task available or device
    is already busy.
    """
    busy = conn.execute(
        "SELECT current_task_id FROM device_heartbeats WHERE device_id=?",
        (device_id,),
    ).fetchone()
    if busy and busy["current_task_id"]:
        return None

    task = conn.execute(
        "SELECT task_id, title, description, context_snapshot, priority "
        "FROM federation_tasks "
        "WHERE status='pending_reassign' AND fail_count < max_retries "
        "ORDER BY priority ASC, created_at ASC LIMIT 1",
    ).fetchone()
    if not task:
        return None

    try:
        conn.execute("BEGIN IMMEDIATE")

        # Re-check: may have been claimed while waiting for lock.
        check = conn.execute(
            "SELECT status FROM federation_tasks WHERE task_id=?",
            (task["task_id"],),
        ).fetchone()
        if not check or check["status"] != "pending_reassign":
            conn.execute("ROLLBACK")
            return None

        now = time.time()
        conn.execute(
            "UPDATE federation_tasks SET status='assigned', "
            "assigned_device=?, heartbeat_at=?, "
            "started_at=COALESCE(started_at,?) WHERE task_id=?",
            (device_id, now, now, task["task_id"]),
        )
        conn.execute(
            "UPDATE device_heartbeats SET current_task_id=? "
            "WHERE device_id=?",
            (task["task_id"], device_id),
        )
        conn.commit()

        import json
        return {
            "task_id": task["task_id"],
            "title": task["title"],
            "description": task["description"],
            "context": json.loads(task["context_snapshot"] or "{}"),
        }

    except sqlite3.OperationalError:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------


async def federation_heartbeat_loop(
    config: FederationConfig,
    interval_s: Optional[int] = None,
) -> None:
    """Run the federation heartbeat loop until cancelled.

    This is an ``asyncio``-based loop intended to be spawned as a
    background task on the gateway.  It runs the three-phase P2P cycle
    every ``interval_s`` seconds (or ``config.heartbeat_interval_s``).

    The loop is best-effort — a database connectivity error in one tick
    does not abort the loop; it logs and retries on the next tick.
    """
    if not config.enabled:
        return

    db_path = config.resolve_db_path()
    if not db_path:
        logger.info("Federation: no database path resolved, skipping")
        return

    device = _device_id()
    tick = interval_s or config.heartbeat_interval_s
    logger.info(
        "Federation heartbeat started (device=%s, db=%s, interval=%ds, offline_threshold=%ds)",
        device, db_path, tick, config.offline_threshold_s,
    )

    try:
        while True:
            try:
                conn = _get_connection(db_path)
                if conn is None:
                    await asyncio.sleep(tick)
                    continue

                try:
                    _ensure_schema(conn)
                    _heartbeat(conn, device)
                    relays = _detect_offline(conn, device, config.offline_threshold_s)
                    claimed = _claim_task(conn, device)

                    for r in relays:
                        logger.info(
                            "Federation relay: %s(%s) → task %s (%s)",
                            r["device"], r["hostname"], r["task_id"], r["task_title"],
                        )
                    if claimed:
                        logger.info(
                            "Federation: claimed task %s (%s)",
                            claimed["task_id"], claimed["title"],
                        )
                finally:
                    conn.close()

            except Exception:
                logger.debug("Federation heartbeat tick failed", exc_info=True)

            await asyncio.sleep(tick)

    except asyncio.CancelledError:
        logger.info("Federation heartbeat loop cancelled")
        raise
