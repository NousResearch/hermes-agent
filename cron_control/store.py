"""Append-only control-plane persistence for cron monitoring evidence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1


def default_control_plane_db_path() -> Path:
    return get_hermes_home() / "cron" / "control-plane.db"


def _connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or default_control_plane_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def migrate_control_plane_db(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("PRAGMA user_version")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS incidents (
                   incident_id TEXT PRIMARY KEY,
                   job_id TEXT NOT NULL,
                   state TEXT NOT NULL,
                   evidence_state TEXT NOT NULL,
                   summary TEXT,
                   classifier_version TEXT,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evidence (
                   evidence_id TEXT PRIMARY KEY,
                   incident_id TEXT NOT NULL,
                   job_id TEXT NOT NULL,
                   execution_id TEXT NOT NULL,
                   kind TEXT NOT NULL,
                   source TEXT NOT NULL,
                   observed_at TEXT NOT NULL,
                   source_time TEXT NOT NULL,
                   value_json TEXT NOT NULL,
                   source_ref TEXT NOT NULL,
                   content_hash TEXT NOT NULL,
                   freshness_seconds INTEGER NOT NULL,
                   validation TEXT NOT NULL,
                   FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS verdicts (
                   verdict_id TEXT PRIMARY KEY,
                   incident_id TEXT NOT NULL,
                   job_id TEXT NOT NULL,
                   state TEXT NOT NULL,
                   evidence_state TEXT NOT NULL,
                   rule_id TEXT NOT NULL,
                   evidence_refs_json TEXT NOT NULL,
                   recommended_action TEXT NOT NULL,
                   automatic_action_allowed INTEGER NOT NULL,
                   blocked_by_json TEXT NOT NULL,
                   classified_at TEXT NOT NULL,
                   classifier_version TEXT NOT NULL,
                   FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS actions (
                   action_id TEXT PRIMARY KEY,
                   incident_id TEXT NOT NULL,
                   job_id TEXT NOT NULL,
                   action TEXT NOT NULL,
                   status TEXT NOT NULL,
                   idempotency_key TEXT NOT NULL,
                   fencing_token INTEGER NOT NULL,
                   before_state_json TEXT NOT NULL,
                   after_state_json TEXT NOT NULL,
                   result TEXT NOT NULL,
                   rollback_hint TEXT,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL,
                   FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS leases (
                   resource_key TEXT PRIMARY KEY,
                   incident_id TEXT NOT NULL,
                   holder_id TEXT NOT NULL,
                   fencing_token INTEGER NOT NULL,
                   acquired_at TEXT NOT NULL,
                   expires_at TEXT NOT NULL,
                   heartbeat_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS overrides (
                   override_id TEXT PRIMARY KEY,
                   incident_id TEXT NOT NULL,
                   job_id TEXT NOT NULL,
                   override_type TEXT NOT NULL,
                   actor_type TEXT NOT NULL,
                   actor_id TEXT NOT NULL,
                   role TEXT,
                   reason TEXT NOT NULL,
                   scope TEXT NOT NULL,
                   evidence_refs_json TEXT NOT NULL,
                   expiry TEXT NOT NULL,
                   requested_at TEXT NOT NULL,
                   approved_at TEXT
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS component_heartbeats (
                   component_id TEXT PRIMARY KEY,
                   observed_at TEXT NOT NULL,
                   status TEXT NOT NULL,
                   detail TEXT,
                   payload_json TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS audit_events (
                   audit_id TEXT PRIMARY KEY,
                   timestamp TEXT NOT NULL,
                   incident_id TEXT NOT NULL,
                   job_id TEXT NOT NULL,
                   execution_id TEXT,
                   event_type TEXT NOT NULL,
                   actor_type TEXT NOT NULL,
                   actor_id TEXT NOT NULL,
                   role TEXT,
                   evidence_refs_json TEXT NOT NULL,
                   verdict_ref TEXT,
                   action TEXT,
                   idempotency_key TEXT,
                   fencing_token INTEGER,
                   before_state_json TEXT NOT NULL,
                   after_state_json TEXT NOT NULL,
                   result TEXT NOT NULL,
                   rollback_hint TEXT
               )"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS audit_events_no_update
               BEFORE UPDATE ON audit_events
               BEGIN
                 SELECT RAISE(ABORT, 'audit_events is append-only');
               END"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
               BEFORE DELETE ON audit_events
               BEGIN
                 SELECT RAISE(ABORT, 'audit_events is append-only');
               END"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_evidence_incident
               ON evidence(incident_id, observed_at DESC)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_verdicts_incident
               ON verdicts(incident_id, classified_at DESC)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_actions_incident
               ON actions(incident_id, created_at DESC)"""
        )
        conn.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def open_control_plane_db(path: Path | None = None) -> sqlite3.Connection:
    conn = _connect(path)
    migrate_control_plane_db(conn)
    return conn


def _ensure_timestamp(value: str | None = None) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).isoformat()


def record_evidence(conn: sqlite3.Connection, evidence: dict[str, Any]) -> dict[str, Any]:
    conn.execute(
        """INSERT INTO evidence (
               evidence_id, incident_id, job_id, execution_id, kind, source,
               observed_at, source_time, value_json, source_ref, content_hash,
               freshness_seconds, validation
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            evidence["evidence_id"],
            evidence["incident_id"],
            evidence["job_id"],
            evidence["execution_id"],
            evidence["kind"],
            evidence["source"],
            evidence["observed_at"],
            evidence["source_time"],
            _json(evidence["value"]),
            evidence["source_ref"],
            evidence["content_hash"],
            int(evidence["freshness_seconds"]),
            evidence["validation"],
        ),
    )
    return evidence


def record_incident(
    conn: sqlite3.Connection,
    *,
    incident_id: str,
    job_id: str,
    state: str,
    evidence_state: str,
    summary: str | None = None,
    classifier_version: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO incidents (
               incident_id, job_id, state, evidence_state, summary,
               classifier_version, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM incidents WHERE incident_id=?), ?), ?)""",
        (
            incident_id,
            job_id,
            state,
            evidence_state,
            summary,
            classifier_version,
            incident_id,
            now,
            now,
        ),
    )
    return {"incident_id": incident_id, "job_id": job_id, "state": state, "evidence_state": evidence_state}


def record_verdict(conn: sqlite3.Connection, verdict: dict[str, Any]) -> dict[str, Any]:
    conn.execute(
        """INSERT INTO verdicts (
               verdict_id, incident_id, job_id, state, evidence_state, rule_id,
               evidence_refs_json, recommended_action, automatic_action_allowed,
               blocked_by_json, classified_at, classifier_version
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            verdict["verdict_id"],
            verdict["incident_id"],
            verdict["job_id"],
            verdict["state"],
            verdict["evidence_state"],
            verdict["rule_id"],
            _json(list(verdict.get("evidence_refs", []))),
            verdict["recommended_action"],
            1 if verdict["automatic_action_allowed"] else 0,
            _json(list(verdict.get("blocked_by", []))),
            verdict["classified_at"],
            verdict["classifier_version"],
        ),
    )
    return verdict


def record_component_heartbeat(conn: sqlite3.Connection, heartbeat: dict[str, Any]) -> dict[str, Any]:
    payload = dict(heartbeat)
    conn.execute(
        """INSERT OR REPLACE INTO component_heartbeats (
               component_id, observed_at, status, detail, payload_json
           ) VALUES (?, ?, ?, ?, ?)""",
        (
            payload["component_id"],
            payload["observed_at"],
            payload["status"],
            payload.get("detail"),
            _json(payload),
        ),
    )
    return heartbeat


def append_audit_event(conn: sqlite3.Connection, event: dict[str, Any]) -> dict[str, Any]:
    conn.execute(
        """INSERT INTO audit_events (
               audit_id, timestamp, incident_id, job_id, execution_id, event_type,
               actor_type, actor_id, role, evidence_refs_json, verdict_ref, action,
               idempotency_key, fencing_token, before_state_json, after_state_json,
               result, rollback_hint
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event["audit_id"],
            _ensure_timestamp(event.get("timestamp")),
            event["incident_id"],
            event["job_id"],
            event.get("execution_id"),
            event["event_type"],
            event["actor"]["type"],
            event["actor"]["id"],
            event["actor"].get("role"),
            _json(list(event.get("evidence_refs", []))),
            event.get("verdict_ref"),
            event.get("action"),
            event.get("idempotency_key"),
            event.get("fencing_token"),
            _json(event.get("before_state", {})),
            _json(event.get("after_state", {})),
            event["result"],
            event.get("rollback_hint"),
        ),
    )
    return event


def _action_record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "action_id": row["action_id"],
        "incident_id": row["incident_id"],
        "job_id": row["job_id"],
        "action": row["action"],
        "status": row["status"],
        "idempotency_key": row["idempotency_key"],
        "fencing_token": row["fencing_token"],
        "before_state": json.loads(row["before_state_json"]),
        "after_state": json.loads(row["after_state_json"]),
        "result": row["result"],
        "rollback_hint": row["rollback_hint"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def record_action(conn: sqlite3.Connection, action: dict[str, Any]) -> dict[str, Any]:
    conn.execute(
        """INSERT OR IGNORE INTO actions (
               action_id, incident_id, job_id, action, status, idempotency_key,
               fencing_token, before_state_json, after_state_json, result,
               rollback_hint, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            action["action_id"],
            action["incident_id"],
            action["job_id"],
            action["action"],
            action["status"],
            action["idempotency_key"],
            action["fencing_token"],
            _json(action.get("before_state", {})),
            _json(action.get("after_state", {})),
            action["result"],
            action.get("rollback_hint"),
            _ensure_timestamp(action.get("created_at")),
            _ensure_timestamp(action.get("updated_at")),
        ),
    )
    row = conn.execute(
        "SELECT * FROM actions WHERE action_id=?",
        (action["action_id"],),
    ).fetchone()
    return _action_record(row) or action


def update_action(
    conn: sqlite3.Connection,
    action_id: str,
    *,
    status: str | None = None,
    before_state: dict[str, Any] | None = None,
    after_state: dict[str, Any] | None = None,
    result: str | None = None,
    rollback_hint: str | None = None,
    fencing_token: int | None = None,
) -> dict[str, Any] | None:
    fields: list[str] = []
    values: list[Any] = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if before_state is not None:
        fields.append("before_state_json=?")
        values.append(_json(before_state))
    if after_state is not None:
        fields.append("after_state_json=?")
        values.append(_json(after_state))
    if result is not None:
        fields.append("result=?")
        values.append(result)
    if rollback_hint is not None:
        fields.append("rollback_hint=?")
        values.append(rollback_hint)
    if fencing_token is not None:
        fields.append("fencing_token=?")
        values.append(fencing_token)
    fields.append("updated_at=?")
    values.append(datetime.now(timezone.utc).isoformat())
    values.append(action_id)
    if not fields:
        return get_action(conn, action_id)
    conn.execute(
        f"UPDATE actions SET {', '.join(fields)} WHERE action_id=?",
        values,
    )
    return get_action(conn, action_id)


def get_action(conn: sqlite3.Connection, action_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM actions WHERE action_id=?",
        (action_id,),
    ).fetchone()
    return _action_record(row)


def list_actions(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM actions ORDER BY created_at DESC, action_id DESC LIMIT ?",
        (max(1, min(int(limit), 500)),),
    ).fetchall()
    return [_action_record(row) for row in rows if row is not None]


def _lease_record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "resource_key": row["resource_key"],
        "incident_id": row["incident_id"],
        "holder_id": row["holder_id"],
        "fencing_token": row["fencing_token"],
        "acquired_at": row["acquired_at"],
        "expires_at": row["expires_at"],
        "heartbeat_at": row["heartbeat_at"],
    }


def acquire_lease(
    conn: sqlite3.Connection,
    *,
    resource_key: str,
    incident_id: str,
    holder_id: str,
    ttl_seconds: int = 300,
) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(1, int(ttl_seconds)))
    current = conn.execute(
        "SELECT * FROM leases WHERE resource_key=?",
        (resource_key,),
    ).fetchone()
    if current is not None:
        try:
            current_expires = datetime.fromisoformat(str(current["expires_at"]).replace("Z", "+00:00"))
        except Exception:
            current_expires = now
        if current_expires > now and str(current["holder_id"]) != str(holder_id):
            return None
        fencing_token = int(current["fencing_token"]) + 1
    else:
        fencing_token = 1

    conn.execute(
        """INSERT OR REPLACE INTO leases (
               resource_key, incident_id, holder_id, fencing_token,
               acquired_at, expires_at, heartbeat_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            resource_key,
            incident_id,
            holder_id,
            fencing_token,
            now.isoformat(),
            expires_at.isoformat(),
            now.isoformat(),
        ),
    )
    return _lease_record(
        conn.execute(
            "SELECT * FROM leases WHERE resource_key=?",
            (resource_key,),
        ).fetchone(),
    )


def refresh_lease(
    conn: sqlite3.Connection,
    *,
    resource_key: str,
    holder_id: str,
    fencing_token: int,
    ttl_seconds: int = 300,
) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(1, int(ttl_seconds)))
    current = conn.execute(
        "SELECT * FROM leases WHERE resource_key=?",
        (resource_key,),
    ).fetchone()
    if current is None:
        return None
    if str(current["holder_id"]) != str(holder_id):
        return None
    if int(current["fencing_token"]) != int(fencing_token):
        return None
    conn.execute(
        """UPDATE leases
           SET expires_at=?, heartbeat_at=?
           WHERE resource_key=? AND holder_id=? AND fencing_token=?""",
        (expires_at.isoformat(), now.isoformat(), resource_key, holder_id, fencing_token),
    )
    return get_lease(conn, resource_key)


def release_lease(
    conn: sqlite3.Connection,
    *,
    resource_key: str,
    holder_id: str,
    fencing_token: int,
) -> bool:
    cur = conn.execute(
        """DELETE FROM leases
           WHERE resource_key=? AND holder_id=? AND fencing_token=?""",
        (resource_key, holder_id, fencing_token),
    )
    return cur.rowcount == 1


def get_lease(conn: sqlite3.Connection, resource_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM leases WHERE resource_key=?",
        (resource_key,),
    ).fetchone()
    return _lease_record(row)


def list_audit_events(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM audit_events ORDER BY timestamp DESC, audit_id DESC LIMIT ?",
        (max(1, min(int(limit), 500)),),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "audit_id": row["audit_id"],
                "timestamp": row["timestamp"],
                "incident_id": row["incident_id"],
                "job_id": row["job_id"],
                "execution_id": row["execution_id"],
                "event_type": row["event_type"],
                "actor": {"type": row["actor_type"], "id": row["actor_id"], "role": row["role"]},
                "evidence_refs": json.loads(row["evidence_refs_json"]),
                "verdict_ref": row["verdict_ref"],
                "action": row["action"],
                "idempotency_key": row["idempotency_key"],
                "fencing_token": row["fencing_token"],
                "before_state": json.loads(row["before_state_json"]),
                "after_state": json.loads(row["after_state_json"]),
                "result": row["result"],
                "rollback_hint": row["rollback_hint"],
            }
        )
    return result
