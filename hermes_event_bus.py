"""
Hermes Internal Message Bus v1.0

Unified MQTT + SQLite event bus for all agent-to-agent and system events.
Replaces notification_bus.py and notification_consumer.py.

Design:
  1. SQLite is the source of truth — every event is written synchronously.
  2. MQTT is the real-time transport — best-effort fire-and-forget.
  3. Phoenix subscribes to hermes/+/+/phoenix and hermes/+/+/broadcast
     and receives callbacks in real time during active sessions.
  4. All agents publish via publish_event() which writes to SQLite FIRST,
     then publishes to MQTT.

Topic hierarchy:
    hermes/{category}/{source_agent}/{target_agent}

Categories: lifecycle, task, guardrail, capability, user, system
Target: agent name or "broadcast"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
DB_PATH = Path(HERMES_HOME) / "shared" / "hermes_event_bus.db"
BROKER = "127.0.0.1"
PORT = 1883
KEEPALIVE = 15

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_CONNECTION: sqlite3.Connection | None = None
_CONNECTION_LOCK = threading.Lock()

_MQTT_CLIENT: Any = None  # paho-mqtt client, lazy-initialised
_MQTT_LOCK = threading.Lock()
_HAS_PAHO: bool = False

try:
    import paho.mqtt.client as mqtt
    _HAS_PAHO = True
except Exception:
    logger.warning("paho-mqtt not available; MQTT transport disabled")

# ---------------------------------------------------------------------------
# Dynamic field stripping for recurrence hashing
# ---------------------------------------------------------------------------
_DYNAMIC_KEYS = frozenset({
    "timestamp", "created_at", "processed_at", "received_at", "resolved_at",
    "session_id", "task_id", "message_id", "id", "uuid", "correlation_id",
    "turn_id", "run_id", "seq", "sequence", "epoch", "ts", "time",
    "user_message", "user_text", "content", "text",  # user-specific text
    "agent", "source_agent", "target_agent", "model",  # identity — pattern is in structure
})

_DYNAMIC_SUFFIXES = ("_at", "_id", "_ts", "_time", "_uuid")


def _is_dynamic_key(key: str) -> bool:
    """Return True if a payload key is considered dynamic for recurrence hashing."""
    k = key.lower()
    if k in _DYNAMIC_KEYS:
        return True
    for suffix in _DYNAMIC_SUFFIXES:
        if k.endswith(suffix):
            return True
    return False


def _strip_dynamic(obj: Any) -> Any:
    """Recursively strip dynamic fields from a payload for recurrence hashing."""
    if isinstance(obj, dict):
        return {
            k: _strip_dynamic(v)
            for k, v in obj.items()
            if not _is_dynamic_key(k)
        }
    if isinstance(obj, list):
        return [_strip_dynamic(item) for item in obj]
    if isinstance(obj, str):
        # Normalise whitespace and lowercase
        return " ".join(obj.lower().split())
    return obj


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON representation with sorted keys."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def compute_recurrence_hash(payload: dict) -> str:
    """
    Strip dynamic values, normalise, and compute a 16-char SHA256 prefix.
    Identical recurring issues will share this hash regardless of timestamps
    or session-specific identifiers.
    """
    stripped = _strip_dynamic(payload)
    canonical = _canonical_json(stripped)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SQLite schema and connection
# ---------------------------------------------------------------------------
def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_agent TEXT NOT NULL,
            target_agent TEXT NOT NULL,
            event_type TEXT NOT NULL,
            category TEXT NOT NULL,
            priority INTEGER DEFAULT 5 NOT NULL,
            payload TEXT NOT NULL,
            task_id TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            model TEXT DEFAULT '',
            resolution_status TEXT DEFAULT 'pending',
            resolution_time_ms INTEGER DEFAULT 0,
            recurrence_hash TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP NOT NULL,
            processed_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_unresolved
        ON event(resolution_status, priority, id)
        WHERE resolution_status != 'resolved'
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_category
        ON event(category, event_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_recurrence
        ON event(recurrence_hash, created_at)
        WHERE recurrence_hash != ''
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_source_target
        ON event(source_agent, target_agent)
    """)


def _connection() -> sqlite3.Connection:
    global _CONNECTION
    if _CONNECTION is None:
        with _CONNECTION_LOCK:
            if _CONNECTION is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CONNECTION = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                _CONNECTION.row_factory = sqlite3.Row
                _init_db(_CONNECTION)
    return _CONNECTION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core public API: publish_event
# ---------------------------------------------------------------------------
def publish_event(
    category: str,
    event_type: str,
    payload: dict,
    source_agent: str | None = None,
    target_agent: str = "broadcast",
    priority: int = 5,
    task_id: str = "",
    session_id: str = "",
    model: str = "",
) -> int:
    """
    Publish an event to the Hermes Internal Message Bus.

    Steps:
      1. Write synchronously to SQLite (source of truth).
      2. Compute recurrence hash.
      3. Publish to MQTT for real-time delivery.

    Returns the SQLite event row id.
    """
    # Resolve source agent
    if source_agent is None:
        source_agent = os.environ.get("HERMES_PROFILE", "unknown").strip()

    # Validate category
    valid_categories = {"lifecycle", "task", "guardrail", "capability", "user", "system"}
    if category not in valid_categories:
        logger.warning("Invalid category %r, defaulting to 'system'", category)
        category = "system"

    # Recurrence hash
    recurrence_hash = compute_recurrence_hash(payload)

    # Serialize payload
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)

    # 1. SQLite write (synchronous, always succeeds)
    conn = _connection()
    created_at = _now_iso()
    cursor = conn.execute(
        """
        INSERT INTO event (
            source_agent, target_agent, event_type, category, priority,
            payload, task_id, session_id, model, recurrence_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_agent, target_agent, event_type, category, priority,
            payload_json, task_id, session_id, model, recurrence_hash, created_at,
        ),
    )
    conn.commit()
    event_id = cursor.lastrowid or 0

    # 2. MQTT publish (best-effort, fire-and-forget, never blocks)
    if _HAS_PAHO:
        try:
            _mqtt_publish(
                category=category,
                source_agent=source_agent,
                target_agent=target_agent,
                payload={
                    "event_id": event_id,
                    "event_type": event_type,
                    "priority": priority,
                    "payload": payload,
                    "recurrence_hash": recurrence_hash,
                    "task_id": task_id,
                    "session_id": session_id,
                    "model": model,
                    "created_at": created_at,
                },
            )
        except Exception as exc:
            logger.debug("MQTT publish failed (event still durable in SQLite): %s", exc)

    return event_id


# ---------------------------------------------------------------------------
# Resolution API (Phoenix owns this)
# ---------------------------------------------------------------------------
def resolve_event(
    event_id: int,
    status: str = "resolved",
    resolution_time_ms: int = 0,
) -> bool:
    """
    Mark an event as resolved (or other resolution status).
    Called by Phoenix when an event has been handled.
    """
    valid_statuses = {"pending", "resolved", "dropped", "escalated", "hold"}
    if status not in valid_statuses:
        logger.warning("Invalid resolution status %r", status)
        return False

    conn = _connection()
    conn.execute(
        """
        UPDATE event
        SET resolution_status = ?, resolution_time_ms = ?, processed_at = ?
        WHERE id = ?
        """,
        (status, resolution_time_ms, _now_iso(), event_id),
    )
    conn.commit()
    return True


def get_event(event_id: int) -> dict | None:
    """Retrieve a single event by id."""
    conn = _connection()
    row = conn.execute("SELECT * FROM event WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def poll_unresolved(
    target_agent: str | None = None,
    category: str | None = None,
    limit: int = 50,
) -> List[dict]:
    """
    Poll unresolved events from SQLite.
    Used by Phoenix at startup to catch anything that arrived while idle.
    """
    conn = _connection()
    query = "SELECT * FROM event WHERE resolution_status = 'pending'"
    params: List[str] = []

    if target_agent:
        query += " AND target_agent = ?"
        params.append(target_agent)

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY priority ASC, id ASC"
    if limit > 0:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, params)
    return [_row_to_dict(row) for row in cursor.fetchall()]


def get_recurrence_stats(
    since_hours: int = 24,
    min_count: int = 3,
) -> List[dict]:
    """
    Return recurrence hashes that have occurred >= min_count times
    in the last since_hours hours, with latest event details.
    """
    conn = _connection()
    cursor = conn.execute(
        """
        SELECT recurrence_hash, COUNT(*) as cnt,
               MAX(created_at) as latest,
               (SELECT event_type FROM event e2
                WHERE e2.recurrence_hash = e1.recurrence_hash
                ORDER BY created_at DESC LIMIT 1) as latest_type,
               (SELECT source_agent FROM event e3
                WHERE e3.recurrence_hash = e1.recurrence_hash
                ORDER BY created_at DESC LIMIT 1) as latest_source
        FROM event e1
        WHERE recurrence_hash != ''
          AND created_at > datetime('now', '-{} hours')
        GROUP BY recurrence_hash
        HAVING cnt >= ?
        ORDER BY cnt DESC
        """.format(since_hours),
        (min_count,),
    )
    return [
        {
            "recurrence_hash": row["recurrence_hash"],
            "count": row["cnt"],
            "latest_at": row["latest"],
            "latest_type": row["latest_type"],
            "latest_source": row["latest_source"],
        }
        for row in cursor.fetchall()
    ]


def get_stats() -> dict:
    """Return event bus statistics for diagnostics."""
    conn = _connection()
    total = conn.execute("SELECT COUNT(*) FROM event").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM event WHERE resolution_status = 'pending'"
    ).fetchone()[0]
    by_category = conn.execute(
        "SELECT category, COUNT(*) FROM event WHERE resolution_status = 'pending' GROUP BY category"
    ).fetchall()
    by_type = conn.execute(
        "SELECT event_type, COUNT(*) FROM event WHERE resolution_status = 'pending' GROUP BY event_type"
    ).fetchall()
    return {
        "total": total,
        "pending": pending,
        "by_category": {row[0]: row[1] for row in by_category},
        "by_type": {row[0]: row[1] for row in by_type},
    }


def close() -> None:
    """Close the shared SQLite connection."""
    global _CONNECTION
    if _CONNECTION:
        with _CONNECTION_LOCK:
            if _CONNECTION:
                _CONNECTION.commit()
                _CONNECTION.close()
                _CONNECTION = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row) -> dict:
    payload = {}
    try:
        payload = json.loads(row["payload"] or "{}")
    except Exception:
        pass
    return {
        "id": row["id"],
        "source_agent": row["source_agent"] or "",
        "target_agent": row["target_agent"] or "",
        "event_type": row["event_type"] or "",
        "category": row["category"] or "",
        "priority": row["priority"],
        "payload": payload,
        "task_id": row["task_id"] or "",
        "session_id": row["session_id"] or "",
        "model": row["model"] or "",
        "resolution_status": row["resolution_status"] or "pending",
        "resolution_time_ms": row["resolution_time_ms"] or 0,
        "recurrence_hash": row["recurrence_hash"] or "",
        "created_at": row["created_at"] or "",
        "processed_at": row["processed_at"] or "",
    }


# ---------------------------------------------------------------------------
# MQTT transport (private)
# ---------------------------------------------------------------------------
def _ensure_mqtt_client() -> Any:
    """Lazy-initialise the MQTT publisher client."""
    global _MQTT_CLIENT
    if _MQTT_CLIENT is not None:
        return _MQTT_CLIENT

    with _MQTT_LOCK:
        if _MQTT_CLIENT is not None:
            return _MQTT_CLIENT

        if not _HAS_PAHO:
            raise RuntimeError("paho-mqtt not available")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        try:
            client.connect(BROKER, PORT, KEEPALIVE)
            client.loop_start()
            _MQTT_CLIENT = client
            logger.info("MQTT publisher connected to %s:%s", BROKER, PORT)
        except Exception as exc:
            logger.warning("MQTT broker unreachable (%s); events remain durable in SQLite", exc)
            raise

    return _MQTT_CLIENT


def _mqtt_publish(
    category: str,
    source_agent: str,
    target_agent: str,
    payload: dict,
    qos: int = 1,
) -> None:
    """Publish a message to MQTT. Fire-and-forget in a daemon thread."""
    topic = f"hermes/{category}/{source_agent}/{target_agent}"
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)

    def _do():
        try:
            client = _ensure_mqtt_client()
            client.publish(topic, payload_json, qos=qos)
            logger.debug("Published to %s", topic)
        except Exception as exc:
            logger.debug("MQTT publish to %s failed: %s", topic, exc)

    # Always fire-and-forget so SQLite write never blocks on network
    threading.Thread(target=_do, daemon=True, name=f"mqtt-pub-{category}").start()


def _mqtt_disconnect() -> None:
    """Clean disconnect for shutdown."""
    global _MQTT_CLIENT
    if _MQTT_CLIENT is not None:
        try:
            _MQTT_CLIENT.loop_stop()
            _MQTT_CLIENT.disconnect()
        except Exception:
            pass
        _MQTT_CLIENT = None
