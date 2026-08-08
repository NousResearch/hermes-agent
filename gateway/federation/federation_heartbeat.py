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
    """Atomically claim one pending_reassign task (v1).

    Fixes TOCTOU race: BEGIN IMMEDIATE is now the first statement,
    so idleness check and task selection both happen inside the
    SQLite write transaction — no window for another connection to
    claim the same task between the idle check and the UPDATE.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")

        # Check if this device is already busy (inside transaction)
        busy = conn.execute(
            "SELECT current_task_id FROM device_heartbeats WHERE device_id=?",
            (device_id,),
        ).fetchone()
        if busy and busy["current_task_id"]:
            conn.execute("ROLLBACK")
            return None

        # Select the best task to claim
        task = conn.execute(
            "SELECT task_id, title, description, context_snapshot, priority "
            "FROM federation_tasks "
            "WHERE status='pending_reassign' AND fail_count < max_retries "
            "ORDER BY priority ASC, created_at ASC LIMIT 1",
        ).fetchone()
        if not task:
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
            await _run_shared_db_loop(config, device_id, tick, db_path, adapter)
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


# Lock file path suffix for single-writer election
_WRITER_LOCK_SUFFIX = ".writer.lock"


def _try_acquire_writer_lock(lock_path: Path) -> bool:
    """Try to acquire the single-writer advisory lock.

    Uses a separate lock file (not the SQLite DB) so we don't corrupt the DB.
    Returns True if this device is the writer, False otherwise.
    The writer writes heartbeats, detects offline peers, and claims tasks.
    Non-writers only update their own heartbeat row.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Use O_CREAT | O_EXCL = atomic create-if-not-exists
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        # Write device_id and timestamp
        os.write(fd, f"{socket.gethostname()}:{time.time():.0f}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Another device already holds the lock — check if it's stale (>5min)
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > 300:
                # Stale lock — take over
                fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                os.write(fd, f"{socket.gethostname()}:{time.time():.0f}".encode())
                os.close(fd)
                return True
        except Exception:
            pass
        return False


async def _run_shared_db_loop(
    config: FederationConfig,
    device_id: str,
    tick: int,
    db_path: Optional[Path] = None,
    adapter: Optional[Any] = None,
) -> None:
    """v1 shared database heartbeat loop.

    Single-writer mode: only one device writes to the SQLite.
    Other devices detect the lock and skip write operations (they still
    send WebSocket heartbeats in lan/auto modes).  This eliminates
    SQLite write conflicts (BUSY) when two devices write simultaneously.

    Election is done via an advisory lock file.  If the writer disappears
    for >5 minutes, any device can seize the lock and become the writer.
    """
    if db_path is None:
        db_path = _resolve_db_path(config.db_path)
        if not db_path:
            logger.info("Federation: no database path resolved, skipping")
            return

    writer_lock_path = Path(str(db_path) + _WRITER_LOCK_SUFFIX)
    is_writer = _try_acquire_writer_lock(writer_lock_path)

    logger.info(
        "Federation heartbeat started (shared_db mode, device=%s, db=%s, "
        "writer=%s, interval=%ds)",
        device_id, db_path, is_writer, tick,
    )

    # Track active task executions to avoid duplicate spawns.
    _active_executions: set = set()

    try:
        while True:
            try:
                # Refresh writer status on each tick (in case we became writer)
                is_writer = _try_acquire_writer_lock(writer_lock_path)

                conn = _get_connection(db_path)
                if conn is None:
                    await asyncio.sleep(tick)
                    continue

                try:
                    _ensure_schema(conn)
                    _heartbeat(conn, device_id)

                    # Only the writer does offline detection and task claiming.
                    # Non-writers skip these to avoid double-write SQLite conflicts.
                    if is_writer:
                        relays = _detect_offline(conn, device_id, config.offline_threshold_s)
                        claimed = _claim_task(conn, device_id)

                        for r in relays:
                            logger.info(
                                "Federation relay: %s(%s) → task %s (%s)",
                                r["device"], r["hostname"], r["task_id"], r["task_title"],
                            )
                        if claimed:
                            task_id = claimed["task_id"]
                            if task_id not in _active_executions:
                                _active_executions.add(task_id)
                                logger.info(
                                    "Federation: executing claimed task %s (%s)",
                                    claimed["task_id"], claimed["title"],
                                )
                                if adapter and hasattr(adapter, "claim_and_execute"):
                                    asyncio.create_task(
                                        adapter.claim_and_execute(
                                            task_id=task_id,
                                            title=claimed["title"],
                                            description=claimed.get("description", ""),
                                            context_snapshot=claimed.get("context", {}),
                                        ),
                                        name=f"fed-exec-{task_id}",
                                    )
                    else:
                        # Non-writer: just broadcast presence via adapter if available.
                        # In shared_db v1 there is no WebSocket adapter, so this is
                        # a no-op — the writer's next heartbeat tick will see us.
                        pass
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

                # Phase 22: broadcast health snapshot (ops layer)
                await _broadcast_ops_health(adapter, config, device_id)

                # Phase 22: run lost-contact SOS escalation
                await _run_sos_escalation(adapter)

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


async def _broadcast_ops_health(adapter: FederationAdapter, config: FederationConfig, device_id: str) -> None:
    """Broadcast this device's health snapshot as an OPS_HEALTH message."""
    try:
        from gateway.federation.federation_ops import collect_local_health
        health = collect_local_health()
        health["device_id"] = device_id
        health["gateway_up"] = True
        health["federation_connected"] = adapter.is_connected()
        health["level"] = "ok"
        msg = FedMessage(
            msg_type=MessageType.OPS_HEALTH.value,
            sender_id=device_id,
            payload=health,
        )
        await adapter.send(msg)
    except Exception as e:
        logger.debug("Ops health broadcast failed: %s", e)


async def _run_sos_escalation(adapter: FederationAdapter) -> None:
    """Run LostContactSOS escalation on the current health matrix."""
    try:
        sos = getattr(adapter, "_sos", None)
        if sos:
            alerts = sos.update()
            if alerts:
                logger.warning("Federation SOS: %d new escalation(s)", len(alerts))
    except Exception as e:
        logger.debug("SOS escalation failed: %s", e)


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
