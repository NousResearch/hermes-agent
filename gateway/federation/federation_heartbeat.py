"""Federation heartbeat loop — v2 WebSocket mode + v1 shared_db backward compat.

Replaces gateway/federation_heartbeat.py (v1 only) with a unified loop that
supports both modes:
- ``shared_db``: File-synced SQLite (v1 backward compat, 2-3 devices)
- ``lan``: WebSocket real-time (v2, N devices)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import FederationConfig
from gateway.federation.federation_adapter import FederationAdapter
from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
)

logger = logging.getLogger(__name__)


# ========================================================================
# Shared-db helpers (v1 backward compatibility)
# ========================================================================

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
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except sqlite3.OperationalError:
        return None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def _heartbeat(conn: sqlite3.Connection, device_id: str) -> None:
    """Update this device's heartbeat record (v1)."""
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


def _detect_offline(
    conn: sqlite3.Connection, device_id: str, threshold: float,
) -> list:
    """Detect offline peers (v1)."""
    now = time.time()
    cutoff = now - threshold
    relays: list = []

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
                peer_id, row["hostname"], now - (row["last_seen"] or now),
            )
            conn.execute(
                "UPDATE device_heartbeats SET status='offline' WHERE device_id=?",
                (peer_id,),
            )

            task_id = row["current_task_id"]
            if task_id:
                conn.execute(
                    "UPDATE federation_tasks SET status='pending_reassign', "
                    "fail_count=fail_count+1, error_info=? "
                    "WHERE task_id=? AND status IN ('assigned','in_progress')",
                    (f"Peer {peer_id} offline, triggering relay", task_id),
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
                "UPDATE device_heartbeats SET current_task_id=NULL WHERE device_id=?",
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


def _claim_task(conn: sqlite3.Connection, device_id: str) -> Optional[dict]:
    """Atomically claim one pending_reassign task (v1)."""
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
            "UPDATE device_heartbeats SET current_task_id=? WHERE device_id=?",
            (task["task_id"], device_id),
        )
        conn.commit()

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


# ========================================================================
# Unified heartbeat loop (v2)
# ========================================================================


async def federation_heartbeat_loop(
    config: FederationConfig,
    adapter: Optional[FederationAdapter] = None,
    interval_s: Optional[int] = None,
) -> None:
    """Run the federation heartbeat loop until cancelled.

    Unified loop that supports both modes:
    - ``shared_db``: v1 file-synced SQLite (backward compat)
    - ``lan``: v2 WebSocket real-time (new)
    """
    if not config.enabled:
        return

    tick = interval_s or config.heartbeat_interval_s
    device_id = _resolve_device_id(config.device_id)

    if config.mode == "shared_db":
        db_path = _resolve_db_path(config.db_path)
        if db_path:
            await _run_shared_db_loop(config, device_id, tick, db_path)
        else:
            logger.info("Federation: no database path resolved, skipping")
            return
    else:
        await _run_lan_loop(config, adapter, device_id, tick)


def _resolve_db_path(db_path_config: Optional[str]) -> Optional[Path]:
    """Resolve the database path, expanding ~ and env vars."""
    if not db_path_config:
        # Default to iCloud Drive on macOS.
        icloud = Path.home() / "Library" / "Mobile Documents" / \
            "com~apple~CloudDocs" / "hermes-federation"
        default = icloud / "federation.db"
        if default.parent.exists():
            return default
        return None
    raw = os.path.expandvars(os.path.expanduser(db_path_config))
    return Path(raw)


async def _run_shared_db_loop(
    config: FederationConfig,
    device_id: str,
    tick: int,
    db_path: Optional[Path] = None,
) -> None:
    """v1 shared database heartbeat loop."""
    if db_path is None:
        db_path = _resolve_db_path(config.db_path)
        if not db_path:
            logger.info("Federation: no database path resolved, skipping")
            return

    logger.info(
        "Federation heartbeat started (shared_db mode, device=%s, db=%s, interval=%ds)",
        device_id, db_path, tick,
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
                    _heartbeat(conn, device_id)
                    relays = _detect_offline(conn, device_id, config.offline_threshold_s)
                    claimed = _claim_task(conn, device_id)

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


async def _run_lan_loop(
    config: FederationConfig,
    adapter: Optional[FederationAdapter],
    device_id: str,
    tick: int,
) -> None:
    """v2 WebSocket-based heartbeat loop."""
    if not adapter:
        logger.warning("Federation: lan mode requires an adapter, falling back to shared_db")
        return

    logger.info(
        "Federation heartbeat started (lan mode, device=%s, port=%d, interval=%ds)",
        device_id, config.ws_port, tick,
    )

    try:
        while True:
            try:
                # Send task heartbeat for all active tasks
                for task_id, state in adapter.get_all_task_states().items():
                    if state.get("status") in ("claimed", "in_progress"):
                        await adapter.send_task_heartbeat(task_id)

                # Log federation status
                peer_count = adapter.get_peer_count()
                task_count = len(adapter.get_all_task_states())
                logger.debug(
                    "Federation: peers=%d, tasks=%d",
                    peer_count, task_count,
                )

            except Exception:
                logger.debug("Federation heartbeat tick failed", exc_info=True)

            await asyncio.sleep(tick)

    except asyncio.CancelledError:
        logger.info("Federation heartbeat loop cancelled")
        raise


def _resolve_device_id(configured: Optional[str]) -> str:
    """Resolve device ID from env, config, or auto-detection."""
    if configured and configured != "auto":
        return configured

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
