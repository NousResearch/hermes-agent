"""Read-only GHL Manager dashboard plugin API.

Mounted by the Hermes dashboard at /api/plugins/ghl-manager/.
This MVP is intentionally limited to safe status/configuration reads, local
approval-packet artifact reads, and local-only append-only approval audit events.
It does not send messages, mutate GHL, book
appointments, update CRM state, execute artifact content, or return credentials.
Dashboard plugin routes are intended for local/operator use; do not expose them
publicly without plugin-specific auth and CSRF checks.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

PLUGIN_NAME = "ghl-manager"
PLUGIN_LABEL = "GHL Manager"
PLUGIN_VERSION = "0.1.0"
SCHEMA_VERSION = "ghl-manager.approval_packet.v1"
ARTIFACT_ROOT = Path("/home/atlas/ghl-automation-handoff/workspace-ghl-draft/audits").resolve()
APPROVAL_DB_PATH = Path("/home/atlas/.hermes/blue/ghl-manager-ui.sqlite")
DECISION_VIEW_SCRIPT = Path("/home/atlas/.hermes/scripts/blue_gabriel_decision_view.py")
LOCAL_APPROVAL_EVENT_TYPES = {"approve", "deny", "edit-draft", "block-stale", "queue-reverify"}
ACTION_REQUEST_TYPES = {"send-approved", "reverify"}
ACTION_REQUEST_STATUSES = {
    "queued_for_execution",
    "queued_for_reverify",
    "verifying_live_state",
    "execution_blocked",
    "execution_failed",
    "worker_failed",
    "sent",
    "reverified",
}
CANONICAL_APPROVAL_STATES = {
    "pending_approval",
    "approved_not_sent",
    "handled_sent",
    "handled_elsewhere",
    "denied",
    "superseded",
    "expired",
    "blocked",
    "malformed",
}
CANONICAL_TERMINAL_STATES = {"handled_sent", "handled_elsewhere", "denied", "superseded", "expired", "malformed"}
LOCAL_APPROVAL_STATUS_BY_EVENT = {
    "approve": "approved_not_sent",
    "deny": "denied",
    "edit-draft": "pending_approval",
    "block-stale": "superseded",
    "queue-reverify": "pending_approval",
}
DISPLAY_STATUS_BY_EVENT = {
    "approve": "approved_not_sent",
    "deny": "denied",
    "edit-draft": "draft_edited",
    "block-stale": "blocked_stale",
    "queue-reverify": "queued_for_reverify",
}
ACTION_REQUEST_TO_APPROVAL_STATUS = {
    "queued_for_execution": "approved_not_sent",
    "queued_for_reverify": "approved_not_sent",
    "verifying_live_state": "approved_not_sent",
    "execution_blocked": "blocked",
    "execution_failed": "blocked",
    "worker_failed": "blocked",
    "sent": "handled_sent",
    "reverified": "approved_not_sent",
}
LOCAL_ONLY_WARNING = (
    "Read-only local-only operator surface. Do not expose this dashboard plugin "
    "outside localhost or a trusted private network without plugin-specific auth."
)
DISALLOWED_MVP_ACTIONS = [
    "send_customer_message",
    "mutate_crm",
    "book_appointment",
    "change_opportunity",
    "delete_anything",
]
ALLOWED_MVP_ACTIONS = ["view", "copy_draft", "copy_approval_command", "open_source", "open_kanban"]
KANBAN_CONTEXT_BOARDS = ["ghl-six-priority-cleanup", "ghl-manager-ui"]
ACTION_REQUEST_KANBAN_BOARD = "ghl-manager-ui"
RELEVANT_CRON_TERMS = (
    "ghl",
    "gohighlevel",
    "go highlevel",
    "go-highlevel",
    "ghl manager",
    "ghl-manager",
    "blue crew",
    "blue-crew",
    "blue_crew",
)

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_source_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ARTIFACT_ROOT):
        raise ValueError(f"source path outside allowlisted artifact root: {resolved}")
    return str(resolved)


def _sha256_text(text: str | None) -> str | None:
    if not text:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approval_db() -> sqlite3.Connection:
    APPROVAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APPROVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_approval_store() -> None:
    """Create the local append-only approval audit store if needed."""
    with _approval_db() as conn:
        _ensure_approval_schema(conn)


def _ensure_approval_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS approval_packets (
            approval_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            packet_status TEXT,
            source_path TEXT,
            source_type TEXT,
            draft_hash TEXT,
            packet_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN ('approve', 'deny', 'edit-draft', 'block-stale', 'queue-reverify')),
            actor TEXT NOT NULL,
            draft_hash TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            key TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approval_index (
            approval_id TEXT PRIMARY KEY,
            canonical_idempotency_key TEXT,
            dedupe_scope TEXT,
            current_status TEXT NOT NULL,
            handled_state TEXT,
            latest_request_id TEXT,
            missing_canonical_key INTEGER NOT NULL DEFAULT 0,
            missing_fields_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL,
            index_json TEXT NOT NULL,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        CREATE TABLE IF NOT EXISTS handled_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT,
            approval_id TEXT NOT NULL,
            request_id TEXT,
            final_state TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            message_id TEXT,
            handled_by TEXT NOT NULL,
            handled_at TEXT NOT NULL,
            notes TEXT,
            contract_json TEXT NOT NULL,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        CREATE TABLE IF NOT EXISTS approval_action_requests (
            request_id TEXT PRIMARY KEY,
            approval_id TEXT NOT NULL,
            action_type TEXT NOT NULL CHECK (action_type IN ('send-approved', 'reverify')),
            status TEXT NOT NULL CHECK (status IN ('queued_for_execution', 'queued_for_reverify', 'verifying_live_state', 'execution_blocked', 'execution_failed', 'worker_failed', 'sent', 'reverified')),
            draft_hash TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_json TEXT,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        """
    )
    _migrate_approval_events_constraint(conn)
    _migrate_action_requests_constraint(conn)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_events_approval_id_created
            ON approval_events (approval_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_action_requests_approval_status_created
            ON approval_action_requests (approval_id, status, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_index_status_updated
            ON approval_index (current_status, updated_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_handled_actions_key_state
            ON handled_actions (idempotency_key, final_state, handled_at)
        """
    )


def _migrate_approval_events_constraint(conn: sqlite3.Connection) -> None:
    """Allow queue-reverify in older local DBs without losing append-only history."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'approval_events'"
    ).fetchone()
    table_sql = row["sql"] if row else ""
    if "queue-reverify" in table_sql:
        return
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_approval_events_approval_id_created;
        ALTER TABLE approval_events RENAME TO approval_events_legacy;
        CREATE TABLE approval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN ('approve', 'deny', 'edit-draft', 'block-stale', 'queue-reverify')),
            actor TEXT NOT NULL,
            draft_hash TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        INSERT INTO approval_events (id, approval_id, event_type, actor, draft_hash, payload_json, created_at)
            SELECT id, approval_id, event_type, actor, draft_hash, payload_json, created_at
            FROM approval_events_legacy;
        DROP TABLE approval_events_legacy;
        """
    )


def _migrate_action_requests_constraint(conn: sqlite3.Connection) -> None:
    """Allow worker_failed in older local DBs without losing queued requests."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'approval_action_requests'"
    ).fetchone()
    table_sql = row["sql"] if row else ""
    if not table_sql or "worker_failed" in table_sql:
        return
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_action_requests_approval_status_created;
        ALTER TABLE approval_action_requests RENAME TO approval_action_requests_legacy;
        CREATE TABLE approval_action_requests (
            request_id TEXT PRIMARY KEY,
            approval_id TEXT NOT NULL,
            action_type TEXT NOT NULL CHECK (action_type IN ('send-approved', 'reverify')),
            status TEXT NOT NULL CHECK (status IN ('queued_for_execution', 'queued_for_reverify', 'verifying_live_state', 'execution_blocked', 'execution_failed', 'worker_failed', 'sent', 'reverified')),
            draft_hash TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_json TEXT,
            FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
        );
        INSERT INTO approval_action_requests (
            request_id, approval_id, action_type, status, draft_hash, payload_json, created_at, updated_at, result_json
        )
        SELECT request_id, approval_id, action_type, status, draft_hash, payload_json, created_at, updated_at, result_json
        FROM approval_action_requests_legacy;
        DROP TABLE approval_action_requests_legacy;
        """
    )


def _packet_store_values(packet: dict[str, Any], now: str) -> dict[str, Any]:
    draft_hash = packet.get("draft", {}).get("draft_hash")
    source = packet.get("source") or {}
    return {
        "approval_id": packet["approval_id"],
        "schema_version": packet.get("schema_version") or SCHEMA_VERSION,
        "packet_status": packet.get("packet_status"),
        "source_path": source.get("source_path"),
        "source_type": source.get("source_type"),
        "draft_hash": draft_hash,
        "packet_json": json.dumps(packet, sort_keys=True),
        "seen_at": now,
    }


def _record_approval_packets(packets: list[dict[str, Any]]) -> dict[str, int]:
    """Upsert normalized packets into the local store without duplicating approvals."""
    _ensure_approval_store()
    imported = inserted = updated = 0
    now = _now_iso()
    with _approval_db() as conn:
        for packet in packets:
            if not packet.get("approval_id"):
                continue
            imported += 1
            values = _packet_store_values(packet, now)
            existed = conn.execute(
                "SELECT 1 FROM approval_packets WHERE approval_id = ?",
                (values["approval_id"],),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO approval_packets (
                    approval_id, schema_version, packet_status, source_path, source_type,
                    draft_hash, packet_json, first_seen_at, last_seen_at
                ) VALUES (
                    :approval_id, :schema_version, :packet_status, :source_path, :source_type,
                    :draft_hash, :packet_json, :seen_at, :seen_at
                )
                ON CONFLICT(approval_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    packet_status = excluded.packet_status,
                    source_path = excluded.source_path,
                    source_type = excluded.source_type,
                    draft_hash = excluded.draft_hash,
                    packet_json = excluded.packet_json,
                    last_seen_at = excluded.last_seen_at
                """,
                values,
            )
            if existed:
                updated += 1
            else:
                inserted += 1
            identity = _canonical_identity(packet, draft_hash=values["draft_hash"])
            source_links = {"source_path": values.get("source_path"), "approval_id": values["approval_id"]}
            freshness = {
                "stale": packet.get("packet_status") in {"stale_obsolete", "needs_reverify"},
                "checked_at": now,
                "source_type": values.get("source_type"),
            }
            status = packet.get("packet_status") or "pending_approval"
            if identity.get("primary_key"):
                existing_by_key = conn.execute(
                    """
                    SELECT approval_id, current_status, index_json FROM approval_index
                    WHERE canonical_idempotency_key = ? AND approval_id != ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (identity["primary_key"], values["approval_id"]),
                ).fetchone()
                if existing_by_key:
                    existing_doc = json.loads(existing_by_key["index_json"] or "{}")
                    source_links["duplicate_of_approval_id"] = existing_by_key["approval_id"]
                    source_links["duplicate_canonical_key"] = identity["primary_key"]
                    freshness.update(
                        {
                            "duplicate": True,
                            "stale": True,
                            "reason": "same canonical idempotency key already exists in approval_index; preserving artifact audit but suppressing duplicate operator action",
                            "existing_status": existing_by_key["current_status"],
                            "existing_source_links": existing_doc.get("source_links") or {},
                        }
                    )
                    existing_status = _canonical_approval_status(existing_by_key["current_status"])
                    if existing_status in {"pending_approval", "approved_not_sent", "blocked"} | CANONICAL_TERMINAL_STATES:
                        status = "superseded"
                        freshness["reason"] = (
                            f"same canonical idempotency key already has {existing_status} approval_index row; "
                            "preserving duplicate artifact audit but suppressing duplicate operator action"
                        )
            _upsert_approval_index(
                conn,
                approval_id=values["approval_id"],
                current_status=status,
                canonical_identity=identity,
                handled_state=packet.get("handled_state"),
                source={"source_type": values.get("source_type"), "source_links": source_links},
                freshness=freshness,
            )
    return {"imported": imported, "inserted": inserted, "updated": updated}


def _event_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"] or "{}")
    return {
        "id": row["id"],
        "approval_id": row["approval_id"],
        "event_type": row["event_type"],
        "actor": row["actor"],
        "draft_hash": row["draft_hash"],
        "payload": payload,
        "created_at": row["created_at"],
    }


def _live_execution_enabled() -> bool:
    return str(os.getenv("GHL_MANAGER_LIVE_EXECUTION", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _action_request_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"] or "{}")
    result = json.loads(row["result_json"] or "{}") if row["result_json"] else None
    return {
        "request_id": row["request_id"],
        "approval_id": row["approval_id"],
        "action_type": row["action_type"],
        "status": row["status"],
        "draft_hash": row["draft_hash"],
        "payload": payload,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "result": result,
        "live_execution_enabled": _live_execution_enabled(),
    }


def _action_requests_for_approval(conn: sqlite3.Connection, approval_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM approval_action_requests WHERE approval_id = ? ORDER BY created_at, request_id",
        (approval_id,),
    ).fetchall()
    return [_action_request_to_dict(row) for row in rows]


def _short_identifier(value: str | None, *, prefix: str) -> str:
    text = str(value or "").strip()
    if not text:
        return f"{prefix} unknown"
    if len(text) <= 16 or re.match(r"^t_[0-9a-f]{8}(?:-\d+)?$", text):
        return f"{prefix} {text}"
    return f"{prefix} #{text[-6:]}"


def _action_type_label(action_type: str | None) -> str:
    if action_type == "reverify":
        return "Read-only reverify"
    if action_type == "send-approved":
        return "Approve + guarded executor queue"
    return str(action_type or "Action request").replace("-", " ").replace("_", " ").title()


def _action_status_label(status: str | None) -> str:
    labels = {
        "queued_for_execution": "Queued for executor",
        "queued_for_reverify": "Queued for reverify",
        "verifying_live_state": "Verifying live state",
        "execution_blocked": "Execution blocked",
        "execution_failed": "Execution failed",
        "worker_failed": "Worker failed",
        "sent": "Sent",
        "reverified": "Reverified",
    }
    return labels.get(str(status or ""), str(status or "unknown").replace("_", " ").title())


def _action_request_safety_state(action_request: dict[str, Any]) -> str:
    if action_request.get("action_type") == "reverify":
        return "Read-only follow-up required; no customer send, CRM mutation, or booking is allowed."
    if action_request.get("status") in {"execution_blocked", "execution_failed", "worker_failed"}:
        return "Guardrail stopped execution; no customer send, CRM mutation, or booking ran."
    return "Queued behind guarded executor; live sends, CRM mutations, and booking/calendar changes remain disabled by default."


def _packet_contact_summary(packet: dict[str, Any] | None) -> str:
    if not packet:
        return "Unknown contact"
    contact = packet.get("contact") or {}
    subject = packet.get("subject") or {}
    return _known_value(subject.get("contact_label"), contact.get("name"), contact.get("phone"), contact.get("contact_id")) or "Unknown contact"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _is_stale_queued_action_request(action_request: dict[str, Any], threshold: timedelta = timedelta(minutes=15)) -> bool:
    if action_request.get("status") not in {"queued_for_execution", "queued_for_reverify", "verifying_live_state"}:
        return False
    queued_at = _parse_iso_datetime(action_request.get("updated_at") or action_request.get("created_at"))
    if not queued_at:
        return False
    return datetime.now(timezone.utc) - queued_at > threshold


def _compact_action_request_notification(action_request: dict[str, Any], packet: dict[str, Any] | None = None, kanban_task_id: str | None = None) -> dict[str, Any]:
    action_label = _action_type_label(action_request.get("action_type"))
    status_label = _action_status_label(action_request.get("status"))
    contact = _packet_contact_summary(packet)
    summary = f"{action_label} for {contact} is {status_label.lower()}."
    result = action_request.get("result") or {}
    stale = _is_stale_queued_action_request(action_request)
    if result.get("blocked_reason"):
        summary += f" Guardrail: {result['blocked_reason']}"
    if stale:
        summary += " Stale queued request: processor has not recorded a terminal result within the expected cadence."
    return {
        "request_alias": _short_identifier(action_request.get("request_id"), prefix="Request"),
        "approval_alias": _short_identifier(action_request.get("approval_id"), prefix="Approval"),
        "action_type": action_request.get("action_type"),
        "action_label": action_label,
        "status": action_request.get("status"),
        "status_label": status_label,
        "stale": stale,
        "safety_state": _action_request_safety_state(action_request),
        "summary": summary,
        "created_at": action_request.get("created_at"),
        "updated_at": action_request.get("updated_at"),
        "kanban_task_id": kanban_task_id,
        "debug": {
            "request_id": action_request.get("request_id"),
            "approval_id": action_request.get("approval_id"),
            "canonical_idempotency_key": (action_request.get("payload") or {}).get("canonical_idempotency_key"),
            "draft_hash": action_request.get("draft_hash"),
        },
    }


def _recent_action_request_notifications(limit: int = 8) -> list[dict[str, Any]]:
    _ensure_approval_store()
    with _approval_db() as conn:
        rows = conn.execute(
            """
            SELECT ar.*, ap.packet_json
            FROM approval_action_requests ar
            LEFT JOIN approval_packets ap ON ap.approval_id = ar.approval_id
            ORDER BY ar.updated_at DESC, ar.created_at DESC, ar.request_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    notifications = []
    for row in rows:
        action_request = _action_request_to_dict(row)
        packet = json.loads(row["packet_json"] or "{}") if "packet_json" in row.keys() and row["packet_json"] else None
        kanban_task_id = None
        if packet:
            kanban_task_id = (packet.get("links") or {}).get("action_request_kanban_task_id") or (packet.get("kanban_context") or {}).get("action_request_task_id")
            if not kanban_task_id and action_request.get("status") in {"queued_for_execution", "queued_for_reverify", "verifying_live_state", "execution_blocked", "execution_failed", "worker_failed"}:
                synced = _sync_action_request_kanban_card(action_request, packet)
                if synced and not synced.get("error"):
                    kanban_task_id = synced.get("task_id")
        notifications.append(_compact_action_request_notification(action_request, packet, kanban_task_id))
    return notifications


def _action_request_attention_summary(notifications: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(request.get("status") for request in notifications)
    stale = [request for request in notifications if request.get("stale")]
    blocked = [request for request in notifications if request.get("status") in {"execution_blocked", "execution_failed", "worker_failed"}]
    return {
        "total": len(notifications),
        "stale_queued": len(stale),
        "blocked_or_failed": len(blocked),
        "by_status": dict(counts),
        "observable": True,
        "attention_required": bool(stale or blocked),
    }


def _action_request_kanban_idempotency_key(action_request: dict[str, Any]) -> str:
    payload = action_request.get("payload") or {}
    canonical_key = payload.get("canonical_idempotency_key")
    if canonical_key:
        return f"ghl-manager:action-request:{canonical_key}"
    fallback = ":".join(str(part or "") for part in (action_request.get("approval_id"), action_request.get("action_type"), action_request.get("draft_hash")))
    return "ghl-manager:action-request:fallback:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]


def _build_action_request_card_body(action_request: dict[str, Any], packet: dict[str, Any], notification: dict[str, Any]) -> str:
    brand = _known_value(packet.get("subject", {}).get("brand_label"), packet.get("brand", {}).get("name")) or "unknown"
    contact = packet.get("contact") or {}
    conversation = packet.get("conversation") or {}
    return "\n".join(
        [
            "Action Bridge follow-up / notification",
            "",
            f"Summary: {notification['summary']}",
            f"Brand/location: {brand}",
            f"Contact: {_packet_contact_summary(packet)} · {contact.get('phone') or 'phone unknown'} · contact {contact.get('contact_id') or 'unknown'}",
            f"Conversation: {conversation.get('conversation_id') or 'unknown'} · channel {conversation.get('channel') or 'unknown'}",
            f"Approval: {action_request.get('approval_id')}",
            f"Action: {notification['action_label']}",
            f"Status: {notification['status_label']}",
            f"Safety: no customer sends, no live CRM mutations, no booking/calendar mutations.",
            f"Operator follow-up: {notification['safety_state']}",
            "",
            "Current context:",
            f"- Latest inbound: {conversation.get('latest_customer_summary') or 'not embedded; reverify before action'}",
            f"- Last outbound: {conversation.get('last_outbound_summary') or 'not embedded'}",
            f"- Recommended action: {(packet.get('decision') or {}).get('recommended_action') or (packet.get('proposed_action') or {}).get('recommended_action') or 'review/reverify only'}",
            "",
            "Acceptance for the assignee:",
            "- Re-fetch/read-only verify latest customer/CRM state before recommending any customer-facing action.",
            "- Do not send, mutate CRM, book appointments, or change calendars from this card.",
            "- Comment the human-readable result and link/update this card instead of creating duplicates.",
            "",
            "Debug IDs (operator only; keep out of normal UI):",
            f"- request_id: {action_request.get('request_id')}",
            f"- approval_id: {action_request.get('approval_id')}",
            f"- canonical_idempotency_key: {(action_request.get('payload') or {}).get('canonical_idempotency_key') or 'missing'}",
        ]
    )


def _sync_action_request_kanban_card(action_request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from hermes_cli import kanban_db
    except Exception:  # noqa: BLE001
        return None
    notification = _compact_action_request_notification(action_request, packet)
    idempotency_key = _action_request_kanban_idempotency_key(action_request)
    title = f"Action Bridge follow-up: {notification['action_label']} for {_packet_contact_summary(packet)}"
    body = _build_action_request_card_body(action_request, packet, notification)
    try:
        db_path = kanban_db.board_dir(ACTION_REQUEST_KANBAN_BOARD) / "kanban.db"
        kanban_db.init_db(db_path=db_path)
        conn = kanban_db.connect(db_path=db_path)
        try:
            task_id = kanban_db.create_task(
                conn,
                title=title,
                body=body,
                assignee=os.getenv("GHL_MANAGER_ACTION_REQUEST_ASSIGNEE", "default"),
                created_by="ghl-manager-action-bridge",
                priority=5,
                idempotency_key=idempotency_key,
                skills=["blue-ghl-operator"],
            )
            comment_body = f"Action request status: {notification['status_label']} ({notification['request_alias']}). {notification['safety_state']}"
            existing_comments = kanban_db.list_comments(conn, task_id)
            if not any(comment_body in comment.body for comment in existing_comments):
                kanban_db.add_comment(conn, task_id, "ghl-manager-action-bridge", comment_body)
        finally:
            conn.close()
        return {"board": ACTION_REQUEST_KANBAN_BOARD, "task_id": task_id, "idempotency_key": idempotency_key, "status": notification["status"], "request_alias": notification["request_alias"]}
    except Exception as exc:  # noqa: BLE001
        return {"board": ACTION_REQUEST_KANBAN_BOARD, "error": str(exc), "idempotency_key": idempotency_key, "status": notification["status"], "request_alias": notification["request_alias"]}


def _approval_index_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "approval_id": row["approval_id"],
        "canonical_idempotency_key": row["canonical_idempotency_key"],
        "dedupe_scope": row["dedupe_scope"],
        "current_status": row["current_status"],
        "display_status": (json.loads(row["index_json"] or "{}")).get("display_status"),
        "handled_state": row["handled_state"],
        "latest_request_id": row["latest_request_id"],
        "missing_canonical_key": bool(row["missing_canonical_key"]),
        "missing_fields": json.loads(row["missing_fields_json"] or "[]"),
        "updated_at": row["updated_at"],
        "index": json.loads(row["index_json"] or "{}"),
    }


def _handled_action_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "idempotency_key": row["idempotency_key"],
        "approval_id": row["approval_id"],
        "request_id": row["request_id"],
        "final_state": row["final_state"],
        "action_taken": row["action_taken"],
        "message_id": row["message_id"],
        "handled_by": row["handled_by"],
        "handled_at": row["handled_at"],
        "notes": row["notes"],
        "contract": json.loads(row["contract_json"] or "{}"),
    }


def _handled_actions_for_approval(conn: sqlite3.Connection, approval_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM handled_actions WHERE approval_id = ? ORDER BY handled_at, id",
        (approval_id,),
    ).fetchall()
    return [_handled_action_to_dict(row) for row in rows]


def _canonical_projection_summary() -> dict[str, Any]:
    """Summarize the SQLite canonical local approval/action projection for the UI."""
    _ensure_approval_store()
    with _approval_db() as conn:
        approval_rows = conn.execute(
            "SELECT current_status, missing_canonical_key, index_json FROM approval_index"
        ).fetchall()
        packet_count = conn.execute("SELECT COUNT(*) FROM approval_packets").fetchone()[0]
        action_rows = conn.execute("SELECT status FROM approval_action_requests").fetchall()
        handled_rows = conn.execute("SELECT final_state FROM handled_actions").fetchall()

    canonical_status_counts: Counter[str] = Counter()
    display_status_counts: Counter[str] = Counter()
    missing_canonical_key = 0
    data_quality_count = 0
    active_operator_count = 0
    for row in approval_rows:
        current_status = row["current_status"] or "unknown"
        canonical_status_counts[current_status] += 1
        try:
            index_doc = json.loads(row["index_json"] or "{}")
        except json.JSONDecodeError:
            index_doc = {}
        display_status = index_doc.get("display_status") or current_status
        display_status_counts[display_status] += 1
        is_missing = bool(row["missing_canonical_key"] or index_doc.get("missing_canonical_key"))
        if is_missing:
            missing_canonical_key += 1
        if display_status in {"superseded", "malformed", "stale_obsolete", "expired", "blocked_missing_canonical_key"} or is_missing:
            data_quality_count += 1
        elif display_status in {"pending_approval", "needs_reverify", "queued_for_execution", "queued_for_reverify", "approved_not_sent", "execution_blocked", "execution_failed", "worker_failed"}:
            active_operator_count += 1

    return {
        "store_path": str(APPROVAL_DB_PATH),
        "canonical_owner": "sqlite:ghl-manager-ui.sqlite:approval_index",
        "mirror_policy": "approval-index.json and handled-actions.json are compatibility/export mirrors, not the canonical local approval/action store.",
        "approval_packets": packet_count,
        "approval_index": len(approval_rows),
        "approval_action_requests": len(action_rows),
        "handled_actions": len(handled_rows),
        "active_operator_rows": active_operator_count,
        "system_data_quality_rows": data_quality_count,
        "missing_canonical_key_rows": missing_canonical_key,
        "by_current_status": dict(canonical_status_counts),
        "by_display_status": dict(display_status_counts),
        "action_requests_by_status": dict(Counter(row["status"] or "unknown" for row in action_rows)),
        "handled_actions_by_state": dict(Counter(row["final_state"] or "unknown" for row in handled_rows)),
    }


def _write_handled_action(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    request_id: str | None,
    contract: dict[str, Any],
) -> None:
    existing = conn.execute(
        "SELECT 1 FROM handled_actions WHERE approval_id = ? AND COALESCE(request_id, '') = COALESCE(?, '') AND final_state = ?",
        (approval_id, request_id, str(contract.get("final_state") or "unknown")),
    ).fetchone()
    if existing:
        return
    conn.execute(
        """
        INSERT INTO handled_actions (
            idempotency_key, approval_id, request_id, final_state, action_taken,
            message_id, handled_by, handled_at, notes, contract_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contract.get("idempotency_key"),
            approval_id,
            request_id,
            str(contract.get("final_state") or "unknown"),
            str(contract.get("action_taken") or "none"),
            contract.get("message_id"),
            str(contract.get("handled_by") or "executor"),
            str(contract.get("handled_at") or _now_iso()),
            contract.get("notes"),
            json.dumps(contract, sort_keys=True),
        ),
    )


def _upsert_approval_index(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    current_status: str,
    canonical_identity: dict[str, Any] | None = None,
    latest_request_id: str | None = None,
    handled_state: str | None = None,
    source: dict[str, Any] | None = None,
    freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = canonical_identity or {}
    missing_fields = identity.get("missing_fields") or []
    canonical_status = _canonical_approval_status(current_status)
    index_doc = {
        "schema_version": "blue.approval.index.v2",
        "approval_id": approval_id,
        "source_type": (source or {}).get("source_type"),
        "source_links": (source or {}).get("source_links") or {},
        "canonical_idempotency_key": identity.get("primary_key"),
        "dedupe_scope": identity.get("dedupe_scope"),
        "current_status": canonical_status,
        "display_status": current_status,
        "handled_state": handled_state,
        "latest_request_id": latest_request_id,
        "missing_canonical_key": bool(missing_fields) and not identity.get("primary_key"),
        "missing_fields": missing_fields,
        "aliases": identity.get("aliases") or [],
        "lookup_fields": identity.get("lookup_fields") or {},
        "freshness": freshness or {"stale": canonical_status in {"superseded", "expired"}, "checked_at": _now_iso()},
        "updated_at": _now_iso(),
        "safety": {
            "customer_send_performed": False,
            "crm_mutation_performed": False,
            "booking_calendar_mutation_performed": False,
        },
        "blue_congruence_contract": {
            "canonical_approval_object": approval_id,
            "one_idempotency_key": identity.get("primary_key"),
            "live_state_reconciliation": "required_before_customer_send_or_crm_mutation",
            "handled_state_record": handled_state or "not_terminal",
        },
    }
    conn.execute(
        """
        INSERT INTO approval_index (
            approval_id, canonical_idempotency_key, dedupe_scope, current_status,
            handled_state, latest_request_id, missing_canonical_key,
            missing_fields_json, updated_at, index_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(approval_id) DO UPDATE SET
            canonical_idempotency_key = excluded.canonical_idempotency_key,
            dedupe_scope = excluded.dedupe_scope,
            current_status = excluded.current_status,
            handled_state = excluded.handled_state,
            latest_request_id = excluded.latest_request_id,
            missing_canonical_key = excluded.missing_canonical_key,
            missing_fields_json = excluded.missing_fields_json,
            updated_at = excluded.updated_at,
            index_json = excluded.index_json
        """,
        (
            approval_id,
            index_doc["canonical_idempotency_key"],
            index_doc["dedupe_scope"],
            index_doc["current_status"],
            handled_state,
            latest_request_id,
            1 if index_doc["missing_canonical_key"] else 0,
            json.dumps(missing_fields, sort_keys=True),
            index_doc["updated_at"],
            json.dumps(index_doc, sort_keys=True),
        ),
    )
    return index_doc


def _canonical_approval_status(status: str | None) -> str:
    text = str(status or "pending_approval")
    if text in CANONICAL_APPROVAL_STATES:
        return text
    return ACTION_REQUEST_TO_APPROVAL_STATUS.get(text) or {
        "no_local_decision": "pending_approval",
        "draft_edited": "pending_approval",
        "blocked_stale": "superseded",
        "stale_obsolete": "superseded",
        "needs_reverify": "pending_approval",
        "manual_review_only": "blocked",
        "blocked_missing_canonical_key": "blocked",
        "queued_for_reverify": "approved_not_sent",
    }.get(text, "blocked" if text.startswith("blocked") else "pending_approval")


def _approval_state_from_rows(packet: sqlite3.Row | None, event_rows: list[sqlite3.Row]) -> dict[str, Any]:
    events = [_event_to_dict(row) for row in event_rows]
    approval_id = packet["approval_id"] if packet else (events[0]["approval_id"] if events else None)
    base_draft_hash = packet["draft_hash"] if packet else None
    latest_event = events[-1] if events else None
    latest_draft_hash = base_draft_hash
    for event in events:
        if event.get("draft_hash"):
            latest_draft_hash = event["draft_hash"]
    current_status = "pending_approval"
    display_status = "no_local_decision"
    if latest_event:
        current_status = LOCAL_APPROVAL_STATUS_BY_EVENT[latest_event["event_type"]]
        display_status = DISPLAY_STATUS_BY_EVENT[latest_event["event_type"]]
    packet_json = json.loads(packet["packet_json"]) if packet else None
    return {
        "approval_id": approval_id,
        "current_status": current_status,
        "canonical_current_status": _canonical_approval_status(current_status),
        "display_status": display_status,
        "base_draft_hash": base_draft_hash,
        "latest_draft_hash": latest_draft_hash,
        "packet": packet_json,
        "events": events,
        "event_count": len(events),
        "store_path": str(APPROVAL_DB_PATH),
        "local_only": True,
        "mutations_enabled": False,
    }


def _get_approval_state(approval_id: str) -> dict[str, Any]:
    _ensure_approval_store()
    with _approval_db() as conn:
        packet = conn.execute("SELECT * FROM approval_packets WHERE approval_id = ?", (approval_id,)).fetchone()
        events = conn.execute(
            "SELECT * FROM approval_events WHERE approval_id = ? ORDER BY created_at, id",
            (approval_id,),
        ).fetchall()
        action_requests = _action_requests_for_approval(conn, approval_id)
        approval_index = _approval_index_to_dict(conn.execute("SELECT * FROM approval_index WHERE approval_id = ?", (approval_id,)).fetchone())
        handled_actions = _handled_actions_for_approval(conn, approval_id)
    if not packet and not events and not action_requests:
        raise HTTPException(status_code=404, detail="approval state not found")
    state = _approval_state_from_rows(packet, events)
    if approval_index and not events:
        index_status = approval_index.get("current_status")
        index_display = approval_index.get("display_status") or index_status
        if index_status:
            state["current_status"] = index_status
            state["canonical_current_status"] = _canonical_approval_status(index_status)
        if index_display:
            state["display_status"] = index_display
        state["status_reason"] = "canonical approval_index current_status; no local UI decision event recorded"
    state["action_requests"] = action_requests
    state["action_request_count"] = len(action_requests)
    latest_request = action_requests[-1] if action_requests else None
    state["latest_action_request"] = latest_request
    if latest_request:
        state["display_status"] = latest_request["status"]
        state["latest_action_status"] = latest_request["status"]
        state["canonical_current_status"] = _canonical_approval_status(latest_request["status"])
    state["approval_index"] = approval_index
    state["handled_actions"] = handled_actions
    state["handled_action_count"] = len(handled_actions)
    state["live_execution_enabled"] = _live_execution_enabled()
    return state


def _decision_view() -> dict[str, Any]:
    """Return the canonical Gabriel decision view generated by the Blue/GHL toolchain."""
    if not DECISION_VIEW_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="decision view script not available")
    try:
        completed = subprocess.run(
            [str(DECISION_VIEW_SCRIPT), "--json", "--live-top", "3", "--max-items", "8"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="decision view timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "decision view failed").strip().splitlines()[-1:]
        raise HTTPException(status_code=502, detail=detail[0] if detail else "decision view failed") from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="decision view returned invalid JSON") from exc
    payload["read_only"] = True
    payload["mutations_enabled"] = False
    payload["source"] = str(DECISION_VIEW_SCRIPT)
    return payload


def _approval_state_index() -> list[dict[str, Any]]:
    _ensure_approval_store()
    with _approval_db() as conn:
        packets = conn.execute("SELECT approval_id FROM approval_packets ORDER BY approval_id").fetchall()
    states = []
    for packet in packets:
        try:
            states.append(_get_approval_state(packet["approval_id"]))
        except HTTPException:
            continue
    return states


def _append_approval_event(approval_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(payload.get("event_type") or "")
    if event_type not in LOCAL_APPROVAL_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"approval event type {event_type!r} is not allowed")
    idempotency_key = payload.get("idempotency_key")
    actor = str(payload.get("actor") or "operator")
    if event_type == "edit-draft" and not payload.get("draft_text"):
        raise HTTPException(status_code=400, detail="edit-draft requires draft_text")
    draft_hash = _sha256_text(str(payload.get("draft_text"))) if payload.get("draft_text") is not None else None

    _ensure_approval_store()
    with _approval_db() as conn:
        packet = conn.execute("SELECT * FROM approval_packets WHERE approval_id = ?", (approval_id,)).fetchone()
        if not packet:
            raise HTTPException(status_code=404, detail="approval packet not imported")
        scope = f"approval-event:{approval_id}"
        if idempotency_key:
            replay = conn.execute("SELECT response_json FROM idempotency_keys WHERE key = ?", (str(idempotency_key),)).fetchone()
            if replay:
                response = json.loads(replay["response_json"])
                response["idempotent_replay"] = True
                return response
        created_at = _now_iso()
        event_payload = {
            "reason": payload.get("reason"),
            "draft_text": payload.get("draft_text"),
            "note": payload.get("note"),
        }
        cur = conn.execute(
            """
            INSERT INTO approval_events (approval_id, event_type, actor, draft_hash, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (approval_id, event_type, actor, draft_hash, json.dumps(event_payload, sort_keys=True), created_at),
        )
        events = conn.execute(
            "SELECT * FROM approval_events WHERE approval_id = ? ORDER BY created_at, id",
            (approval_id,),
        ).fetchall()
        state = _approval_state_from_rows(packet, events)
        packet_json = json.loads(packet["packet_json"] or "{}")
        identity = _canonical_identity(packet_json, draft_hash=state.get("latest_draft_hash"))
        _upsert_approval_index(
            conn,
            approval_id=approval_id,
            current_status=state["current_status"],
            canonical_identity=identity,
            handled_state=state["current_status"] if state["current_status"] in {"denied", "blocked_stale"} else None,
        )
        response = {
            "event_id": cur.lastrowid,
            "approval_id": approval_id,
            "event_type": event_type,
            "idempotent_replay": False,
            "state": state,
            "local_only": True,
            "mutations_enabled": False,
        }
        if idempotency_key:
            conn.execute(
                "INSERT INTO idempotency_keys (key, scope, response_json, created_at) VALUES (?, ?, ?, ?)",
                (str(idempotency_key), scope, json.dumps(response, sort_keys=True), created_at),
            )
        return response


def _create_action_request(approval_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    action_type = str(payload.get("action_type") or "")
    if action_type not in ACTION_REQUEST_TYPES:
        raise HTTPException(status_code=400, detail=f"action request type {action_type!r} is not allowed")
    actor = str(payload.get("actor") or "operator")
    idempotency_key = payload.get("idempotency_key")
    requested_draft_hash = payload.get("draft_hash")
    _ensure_approval_store()
    with _approval_db() as conn:
        packet = conn.execute("SELECT * FROM approval_packets WHERE approval_id = ?", (approval_id,)).fetchone()
        if not packet:
            raise HTTPException(status_code=404, detail="approval packet not imported")
        packet_json = json.loads(packet["packet_json"] or "{}")
        packet_draft_hash = packet["draft_hash"] or packet_json.get("draft", {}).get("draft_hash")
        if idempotency_key:
            replay = conn.execute("SELECT response_json FROM idempotency_keys WHERE key = ?", (str(idempotency_key),)).fetchone()
            if replay:
                response = json.loads(replay["response_json"])
                response["idempotent_replay"] = True
                return response
        events = conn.execute(
            "SELECT * FROM approval_events WHERE approval_id = ? ORDER BY created_at, id",
            (approval_id,),
        ).fetchall()
        state = _approval_state_from_rows(packet, events)
        latest_draft_hash = state["latest_draft_hash"] or packet_draft_hash
        if action_type == "send-approved":
            if not packet_json.get("draft", {}).get("customer_facing") or not packet_json.get("draft", {}).get("draft_text"):
                raise HTTPException(status_code=400, detail="send-approved requires an imported customer-facing draft")
            if requested_draft_hash and requested_draft_hash != latest_draft_hash:
                raise HTTPException(status_code=409, detail="draft hash mismatch; re-open packet and approve the latest draft")

        identity = _canonical_identity(
            packet_json,
            action_request_type=action_type,
            draft_hash=latest_draft_hash,
            supplied_key=str(idempotency_key) if idempotency_key else None,
        )
        canonical_key = identity.get("primary_key")
        if canonical_key:
            replay = conn.execute("SELECT response_json FROM idempotency_keys WHERE key = ?", (str(canonical_key),)).fetchone()
            if replay:
                response = json.loads(replay["response_json"])
                response["idempotent_replay"] = True
                response["canonical_idempotency_key"] = canonical_key
                return response
        elif action_type in {"send-approved", "reverify"}:
            created_at = _now_iso()
            request_seed = f"{approval_id}:{action_type}:{created_at}:missing-canonical"
            request_id = "ar_" + hashlib.sha256(request_seed.encode("utf-8")).hexdigest()[:16]
            result = {
                "blocked_reason": "missing canonical key; required Blue idempotency fields are absent: " + ", ".join(identity.get("missing_fields") or []),
                "missing_canonical_key": True,
                "missing_fields": identity.get("missing_fields") or [],
                "dry_run": True,
                "would_reverify_before_send": True,
                "ghl_message_id": None,
                "handled_actions_contract": {
                    "idempotency_key": None,
                    "final_state": "blocked_missing_canonical_key",
                    "handled_at": created_at,
                    "handled_by": actor,
                    "action_taken": "none",
                    "message_id": None,
                    "notes": "Action Bridge blocked instead of guessing a non-canonical idempotency key.",
                },
                "notification": {
                    "ui_attention": True,
                    "operator_message": "Action blocked: missing canonical Blue idempotency key.",
                    "kanban_checkpoint_designed": True,
                },
            }
            request_payload = {
                "actor": actor,
                "reason": payload.get("reason"),
                "note": payload.get("note"),
                "requested_draft_hash": requested_draft_hash,
                "canonical_idempotency": identity,
                "live_execution_enabled_at_request": _live_execution_enabled(),
                "local_only_warning": LOCAL_ONLY_WARNING,
            }
            conn.execute(
                """
                INSERT INTO approval_action_requests (
                    request_id, approval_id, action_type, status, draft_hash, payload_json, created_at, updated_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    approval_id,
                    action_type,
                    "execution_blocked",
                    latest_draft_hash,
                    json.dumps(request_payload, sort_keys=True),
                    created_at,
                    created_at,
                    json.dumps(result, sort_keys=True),
                ),
            )
            _write_handled_action(
                conn,
                approval_id=approval_id,
                request_id=request_id,
                contract=result["handled_actions_contract"],
            )
            _upsert_approval_index(
                conn,
                approval_id=approval_id,
                current_status="execution_blocked",
                canonical_identity=identity,
                latest_request_id=request_id,
                handled_state="blocked_missing_canonical_key",
            )
            conn.commit()
            action_request = _action_request_to_dict(conn.execute("SELECT * FROM approval_action_requests WHERE request_id = ?", (request_id,)).fetchone())
            kanban_notification = _sync_action_request_kanban_card(action_request, packet_json)
            state = _get_approval_state(approval_id)
            return {
                "request_id": request_id,
                "short_alias": request_id[-6:],
                "approval_id": approval_id,
                "action_type": action_type,
                "status": "execution_blocked",
                "canonical_idempotency_key": None,
                "canonical_idempotency": identity,
                "action_request": action_request,
                "kanban_notification": kanban_notification,
                "state": state,
                "local_only": True,
                "mutations_enabled": False,
                "live_execution_enabled": _live_execution_enabled(),
                "warning": result["blocked_reason"],
                "idempotent_replay": False,
            }

        if action_type == "send-approved":
            if state["current_status"] != "approved_not_sent":
                conn.execute(
                    """
                    INSERT INTO approval_events (approval_id, event_type, actor, draft_hash, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        "approve",
                        actor,
                        None,
                        json.dumps(
                            {"reason": "Approve + queue action request; guarded executor must reverify before any live send.", "draft_text": None, "note": None},
                            sort_keys=True,
                        ),
                        _now_iso(),
                    ),
                )
            status = "queued_for_execution"
            draft_hash = latest_draft_hash
        else:
            if state["current_status"] != "queued_for_reverify":
                conn.execute(
                    """
                    INSERT INTO approval_events (approval_id, event_type, actor, draft_hash, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        "queue-reverify",
                        actor,
                        None,
                        json.dumps({"reason": "Queued guarded read-only reverify action request.", "draft_text": None, "note": None}, sort_keys=True),
                        _now_iso(),
                    ),
                )
            status = "queued_for_reverify"
            draft_hash = latest_draft_hash

        created_at = _now_iso()
        request_seed = f"{approval_id}:{action_type}:{created_at}:{canonical_key or idempotency_key or ''}"
        request_id = "ar_" + hashlib.sha256(request_seed.encode("utf-8")).hexdigest()[:16]
        request_payload = {
            "actor": actor,
            "reason": payload.get("reason"),
            "note": payload.get("note"),
            "requested_draft_hash": requested_draft_hash,
            "canonical_idempotency_key": canonical_key,
            "canonical_idempotency": identity,
            "live_execution_enabled_at_request": _live_execution_enabled(),
            "local_only_warning": LOCAL_ONLY_WARNING,
        }
        conn.execute(
            """
            INSERT INTO approval_action_requests (
                request_id, approval_id, action_type, status, draft_hash, payload_json, created_at, updated_at, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                approval_id,
                action_type,
                status,
                draft_hash,
                json.dumps(request_payload, sort_keys=True),
                created_at,
                created_at,
                None,
            ),
        )
        _upsert_approval_index(
            conn,
            approval_id=approval_id,
            current_status=status,
            canonical_identity=identity,
            latest_request_id=request_id,
            handled_state=None,
        )
        action_request = _action_request_to_dict(
            conn.execute("SELECT * FROM approval_action_requests WHERE request_id = ?", (request_id,)).fetchone()
        )
        conn.commit()
        kanban_notification = _sync_action_request_kanban_card(action_request, packet_json)
        state = _get_approval_state(approval_id)
        response = {
            "request_id": request_id,
            "short_alias": request_id[-6:],
            "approval_id": approval_id,
            "action_type": action_type,
            "status": status,
            "canonical_idempotency_key": canonical_key,
            "canonical_idempotency": identity,
            "action_request": action_request,
            "kanban_notification": kanban_notification,
            "state": state,
            "local_only": True,
            "mutations_enabled": False,
            "live_execution_enabled": _live_execution_enabled(),
            "warning": "Action request queued for guarded server-side executor. Live sends are disabled unless GHL_MANAGER_LIVE_EXECUTION=true and executor preflight passes.",
            "idempotent_replay": False,
        }
        if canonical_key:
            conn.execute(
                "INSERT INTO idempotency_keys (key, scope, response_json, created_at) VALUES (?, ?, ?, ?)",
                (str(canonical_key), f"action-request:{approval_id}", json.dumps(response, sort_keys=True), created_at),
            )
        if idempotency_key and str(idempotency_key) != str(canonical_key):
            conn.execute(
                "INSERT INTO idempotency_keys (key, scope, response_json, created_at) VALUES (?, ?, ?, ?)",
                (str(idempotency_key), f"action-request-alias:{approval_id}", json.dumps(response, sort_keys=True), created_at),
            )
        return response


def _ghl_readonly_token() -> str:
    token = os.getenv("GHL_READONLY_API_KEY") or os.getenv("GHL_PIT") or os.getenv("GHL_API_KEY")
    if not token:
        raise RuntimeError("missing read-only GHL credentials; set GHL_READONLY_API_KEY/GHL_PIT for live read-only reverify")
    return token


def _ghl_get_json(path: str, query: dict[str, str] | None = None) -> dict[str, Any]:
    """Perform a GHL GET only; this helper must never send POST/PUT/PATCH/DELETE."""
    base_url = os.getenv("GHL_API_BASE_URL", "https://services.leadconnectorhq.com").rstrip("/")
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{urllib_parse.urlencode(query)}"
    req = urllib_request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {_ghl_readonly_token()}",
            "Accept": "application/json",
            "Version": os.getenv("GHL_API_VERSION", "2021-07-28"),
        },
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as response:  # noqa: S310 - operator configured GHL API URL.
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"GHL read-only GET failed {exc.code} for {path}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"GHL read-only GET failed for {path}: {exc.reason}") from exc
    return json.loads(raw or "{}")


def _read_only_reverify_live_state(packet_json: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Fetch latest GHL contact/conversation state only; never sends or mutates CRM/calendar state."""
    contact = packet_json.get("contact") or {}
    conversation = packet_json.get("conversation") or {}
    send_target = packet_json.get("send_target") or {}
    contact_id = _known_value(contact.get("contact_id"), send_target.get("contact_id"))
    conversation_id = _known_value(conversation.get("conversation_id"), send_target.get("conversation_id"))
    if not contact_id or not conversation_id:
        raise RuntimeError("missing contact_id or conversation_id for read-only reverify")

    contact_snapshot = _ghl_get_json(f"/contacts/{urllib_parse.quote(str(contact_id), safe='')}")
    messages_snapshot = _ghl_get_json(
        f"/conversations/{urllib_parse.quote(str(conversation_id), safe='')}/messages",
        {"limit": "20"},
    )
    messages = messages_snapshot.get("messages") or messages_snapshot.get("data") or []
    if isinstance(messages, dict):
        messages = messages.get("messages") or messages.get("items") or []
    latest_message = messages[0] if isinstance(messages, list) and messages else None
    latest_direction = str((latest_message or {}).get("direction") or "").lower()
    safe_to_send = latest_direction not in {"outbound", "sent"}
    status = "reverified" if safe_to_send else "handled_elsewhere"
    notes = "Fetched latest contact and conversation messages read-only; no mutation endpoints were called."
    if not safe_to_send:
        notes = "Latest conversation state indicates the old approval may already be handled elsewhere; no send is safe from this reverify."
    return {
        "status": status,
        "safe_to_send": safe_to_send,
        "latest_state_source": "ghl-read-only:get-contact+get-conversation-messages",
        "notes": notes,
        "contact_snapshot": contact_snapshot,
        "conversation_snapshot": {
            "conversation_id": conversation_id,
            "message_count": len(messages) if isinstance(messages, list) else None,
            "latest_message_id": (latest_message or {}).get("id") if isinstance(latest_message, dict) else None,
            "latest_direction": latest_direction or None,
        },
        "request_id": request.get("request_id"),
    }


def process_pending_action_requests(dry_run: bool = True) -> dict[str, Any]:
    """Process queued local action requests; live customer sends are blocked by default."""
    _ensure_approval_store()
    processed: list[dict[str, Any]] = []
    with _approval_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM approval_action_requests
            WHERE status IN ('queued_for_execution', 'queued_for_reverify')
            ORDER BY created_at, request_id
            """
        ).fetchall()
        for row in rows:
            request = _action_request_to_dict(row)
            now = _now_iso()
            conn.execute(
                "UPDATE approval_action_requests SET status = ?, updated_at = ? WHERE request_id = ?",
                ("verifying_live_state", now, request["request_id"]),
            )
            if request["action_type"] == "send-approved" and not _live_execution_enabled():
                result = {
                    "blocked_reason": "Live customer sends are disabled by default; set GHL_MANAGER_LIVE_EXECUTION=true only after safety review.",
                    "dry_run": dry_run,
                    "would_reverify_before_send": True,
                    "ghl_message_id": None,
                    "handled_actions_contract": {
                        "idempotency_key": request.get("payload", {}).get("canonical_idempotency_key"),
                        "final_state": "execution_blocked",
                        "handled_at": now,
                        "handled_by": request.get("payload", {}).get("actor") or "executor",
                        "action_taken": "none",
                        "message_id": None,
                        "notes": "Default-disabled Action Bridge blocked customer send before live GHL mutation.",
                    },
                    "notification": {
                        "ui_attention": True,
                        "operator_message": "Action blocked: live execution disabled.",
                        "kanban_checkpoint_designed": True,
                    },
                }
                final_status = "execution_blocked"
            elif dry_run:
                result = {
                    "blocked_reason": "Executor dry-run mode; no live GHL read/send was performed.",
                    "dry_run": True,
                    "would_fetch_latest_contact": True,
                    "would_fetch_latest_conversation": True,
                    "would_check_idempotency": True,
                    "ghl_message_id": None,
                    "mutation_endpoints_called": [],
                    "reverify_result_contract": {
                        "idempotency_key": request.get("payload", {}).get("canonical_idempotency_key"),
                        "status": "blocked_dry_run",
                        "checked_at": now,
                        "latest_state_source": None,
                        "safe_to_send": False,
                        "notes": "Dry-run/default-disabled executor did not perform a live GHL read.",
                    },
                    "notification": {
                        "ui_attention": True,
                        "operator_message": "Action blocked: executor dry run/default-disabled mode.",
                        "kanban_checkpoint_designed": True,
                    },
                }
                final_status = "execution_blocked"
            elif request["action_type"] == "reverify":
                packet_row_for_reverify = conn.execute("SELECT packet_json FROM approval_packets WHERE approval_id = ?", (request["approval_id"],)).fetchone()
                try:
                    if not packet_row_for_reverify:
                        raise RuntimeError("approval packet missing from local store; cannot reverify live state")
                    reverify_contract = _read_only_reverify_live_state(json.loads(packet_row_for_reverify["packet_json"] or "{}"), request)
                    reverify_status = str(reverify_contract.get("status") or "reverified")
                    final_status = "execution_blocked" if reverify_status == "execution_blocked" else "reverified"
                    result = {
                        "dry_run": False,
                        "ghl_message_id": None,
                        "mutation_endpoints_called": [],
                        "reverify_result_contract": {
                            "idempotency_key": request.get("payload", {}).get("canonical_idempotency_key"),
                            "status": reverify_status,
                            "checked_at": _now_iso(),
                            "latest_state_source": reverify_contract.get("latest_state_source"),
                            "safe_to_send": bool(reverify_contract.get("safe_to_send")),
                            "notes": reverify_contract.get("notes"),
                            "contact_snapshot": reverify_contract.get("contact_snapshot"),
                            "conversation_snapshot": reverify_contract.get("conversation_snapshot"),
                        },
                        "notification": {
                            "ui_attention": reverify_status in {"handled_elsewhere", "stale", "superseded", "execution_blocked"},
                            "operator_message": reverify_contract.get("notes") or "Read-only reverify completed.",
                            "kanban_checkpoint_designed": True,
                        },
                    }
                except Exception as exc:  # noqa: BLE001
                    final_status = "worker_failed"
                    result = {
                        "blocked_reason": f"Read-only reverify worker failed: {exc}",
                        "dry_run": False,
                        "ghl_message_id": None,
                        "mutation_endpoints_called": [],
                        "reverify_result_contract": {
                            "idempotency_key": request.get("payload", {}).get("canonical_idempotency_key"),
                            "status": "worker_failed",
                            "checked_at": _now_iso(),
                            "latest_state_source": None,
                            "safe_to_send": False,
                            "notes": str(exc),
                        },
                        "notification": {
                            "ui_attention": True,
                            "operator_message": "Read-only reverify worker failed; queued item needs operator attention.",
                            "kanban_checkpoint_designed": True,
                        },
                    }
            else:
                result = {
                    "blocked_reason": "Live customer send implementation is intentionally disabled in this MVP.",
                    "dry_run": False,
                    "ghl_message_id": None,
                    "mutation_endpoints_called": [],
                }
                final_status = "execution_blocked"
            updated_at = _now_iso()
            conn.execute(
                "UPDATE approval_action_requests SET status = ?, updated_at = ?, result_json = ? WHERE request_id = ?",
                (final_status, updated_at, json.dumps(result, sort_keys=True), request["request_id"]),
            )
            identity = request.get("payload", {}).get("canonical_idempotency") or {}
            handled_contract = result.get("handled_actions_contract")
            if handled_contract:
                _write_handled_action(
                    conn,
                    approval_id=request["approval_id"],
                    request_id=request["request_id"],
                    contract=handled_contract,
                )
            _upsert_approval_index(
                conn,
                approval_id=request["approval_id"],
                current_status=final_status,
                canonical_identity=identity,
                latest_request_id=request["request_id"],
                handled_state=(handled_contract or {}).get("final_state") or final_status,
            )
            refreshed = conn.execute("SELECT * FROM approval_action_requests WHERE request_id = ?", (request["request_id"],)).fetchone()
            refreshed_request = _action_request_to_dict(refreshed)
            packet_row = conn.execute("SELECT packet_json FROM approval_packets WHERE approval_id = ?", (request["approval_id"],)).fetchone()
            if packet_row:
                refreshed_request["kanban_notification"] = _sync_action_request_kanban_card(refreshed_request, json.loads(packet_row["packet_json"] or "{}"))
            processed.append(refreshed_request)
    return {
        "processed": processed,
        "processed_count": len(processed),
        "dry_run": dry_run,
        "live_execution_enabled": _live_execution_enabled(),
        "mutations_enabled": False,
    }


def _attach_approval_state(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        state = _get_approval_state(packet["approval_id"])
    except HTTPException:
        return packet
    packet["approval_state"] = {
        "current_status": state["current_status"],
        "canonical_current_status": state.get("canonical_current_status"),
        "display_status": state.get("display_status"),
        "base_draft_hash": state["base_draft_hash"],
        "latest_draft_hash": state["latest_draft_hash"],
        "event_count": state["event_count"],
        "events": state["events"],
        "action_requests": state.get("action_requests", []),
        "action_request_count": state.get("action_request_count", 0),
        "latest_action_request": state.get("latest_action_request"),
        "approval_index": state.get("approval_index"),
        "handled_actions": state.get("handled_actions", []),
        "handled_action_count": state.get("handled_action_count", 0),
        "live_execution_enabled": state.get("live_execution_enabled", False),
    }
    return packet


def _phone_from_target(target: str | None) -> str | None:
    if not target:
        return None
    match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", target)
    return match.group(1).strip() if match else None


def _channel_from_target(target: str | None) -> str:
    if not target:
        return "unknown"
    lowered = target.lower()
    # Mixed labels like "Facebook/SMS thread ..." should retain the social
    # source platform; checking SMS first makes the trust context look safer
    # and more direct than it really is.
    if "facebook" in lowered:
        return "Facebook"
    if "instagram" in lowered:
        return "Instagram"
    if "whatsapp" in lowered:
        return "WhatsApp"
    if "sms" in lowered:
        return "SMS"
    if "email" in lowered:
        return "Email"
    if "call" in lowered:
        return "Call"
    return "unknown"


def _channel_from_message_type(message_type: str | None) -> str | None:
    if not message_type:
        return None
    normalized = str(message_type).upper()
    if "FACEBOOK" in normalized:
        return "Facebook"
    if "INSTAGRAM" in normalized:
        return "Instagram"
    if "WHATSAPP" in normalized:
        return "WhatsApp"
    if "GMB" in normalized or "GOOGLE" in normalized:
        return "Google Business"
    if "EMAIL" in normalized:
        return "Email"
    if "SMS" in normalized:
        return "SMS"
    if "CALL" in normalized:
        return "Call"
    if "LIVE_CHAT" in normalized:
        return "Live Chat"
    return None


def _slug_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9:+.-]+", "-", text).strip("-")
    return text or fallback


def _stable_hash_component(prefix: str, value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "unknown":
        return None
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def _known_value(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {"unknown", "not embedded", "none", "null"}:
            return text
    return None


def _canonical_action_type(packet: dict[str, Any], action_type: str | None = None) -> str:
    proposed = packet.get("proposed_action") or {}
    if proposed.get("action_type"):
        return _slug_component(proposed["action_type"], "other").replace("-", "_")
    if action_type == "reverify":
        return "request_review"
    if action_type == "send-approved" or packet.get("draft", {}).get("customer_facing"):
        channel = _known_value(
            packet.get("send_target", {}).get("channel"),
            packet.get("conversation", {}).get("channel"),
            packet.get("subject", {}).get("channel"),
        )
        return f"send_{_slug_component(channel, 'message').replace('-', '_')}"
    if packet.get("booking_context", {}).get("is_booking_related"):
        return "offer_slot"
    return _slug_component(packet.get("approval_type") or "review", "review").replace("-", "_")


def _canonical_dedupe_scope(packet: dict[str, Any], canonical_action: str) -> str:
    if packet.get("idempotency", {}).get("dedupe_scope"):
        return _slug_component(packet["idempotency"]["dedupe_scope"], "admin_decision").replace("-", "_")
    if packet.get("booking_context", {}).get("is_booking_related") or canonical_action in {"offer_slot", "book_slot"}:
        return "booking_slot"
    if packet.get("draft", {}).get("customer_facing") or canonical_action.startswith("send_"):
        return "customer_message"
    proposed = packet.get("proposed_action") or {}
    if proposed.get("crm_mutation") or canonical_action in {"update_opportunity", "add_tag", "create_task"}:
        return "crm_mutation"
    return "admin_decision"


def _canonical_date_bucket(packet: dict[str, Any]) -> str | None:
    booking = packet.get("booking_context") or {}
    if booking.get("is_booking_related"):
        slots = booking.get("proposed_slots") or []
        if slots and isinstance(slots[0], dict):
            return _slug_component(slots[0].get("slot_key") or slots[0].get("display_text"), "unknown-slot")[:96]
    for value in (
        packet.get("state_snapshot", {}).get("latest_inbound_at"),
        packet.get("conversation", {}).get("latest_customer_at"),
        packet.get("verification", {}).get("latest_verified_at"),
        packet.get("source", {}).get("source_created_at"),
        packet.get("source", {}).get("artifact_checked_at"),
        packet.get("source", {}).get("imported_at"),
    ):
        if value:
            return str(value)[:10]
    return None


def _canonical_aliases(packet: dict[str, Any], supplied_key: str | None = None) -> list[str]:
    aliases: list[str] = []
    for value in (
        supplied_key,
        packet.get("approval_id"),
        packet.get("source", {}).get("source_id"),
        packet.get("source", {}).get("source_task_id"),
        packet.get("source", {}).get("source_path"),
        packet.get("source", {}).get("linked_kanban_task_id"),
        packet.get("kanban_context", {}).get("task_id"),
        packet.get("kanban_context", {}).get("parent_task_id"),
        packet.get("draft", {}).get("draft_hash"),
        packet.get("contact", {}).get("contact_id"),
        packet.get("conversation", {}).get("conversation_id"),
    ):
        if value and str(value) not in aliases:
            aliases.append(str(value))
    for key in ("linked_source_ids",):
        for value in packet.get("source", {}).get(key) or []:
            if value and str(value) not in aliases:
                aliases.append(str(value))
    for key in ("kanban_task_ids", "approval_packet_ids", "source_message_ids", "appointment_ids", "opportunity_ids"):
        for value in packet.get("links", {}).get(key) or []:
            if value and str(value) not in aliases:
                aliases.append(str(value))
    return aliases


def _canonical_identity(
    packet: dict[str, Any],
    *,
    action_request_type: str | None = None,
    draft_hash: str | None = None,
    supplied_key: str | None = None,
) -> dict[str, Any]:
    existing = packet.get("idempotency") or {}
    if existing.get("primary_key") and str(existing["primary_key"]).startswith("blue:v1:"):
        primary_key = str(existing["primary_key"])
        missing: list[str] = []
        lookup_fields = dict(existing.get("lookup_fields") or {})
        scope = str(existing.get("dedupe_scope") or primary_key.split(":", 4)[2])
    else:
        missing = []
        canonical_action = _canonical_action_type(packet, action_request_type)
        scope = _canonical_dedupe_scope(packet, canonical_action)
        brand_slug = _slug_component(
            _known_value(packet.get("subject", {}).get("brand"), packet.get("brand", {}).get("name")),
            "unknown",
        )
        if brand_slug == "unknown":
            missing.append("brand")
        contact = _known_value(
            packet.get("subject", {}).get("contact_id"),
            packet.get("contact", {}).get("contact_id"),
            packet.get("send_target", {}).get("contact_id"),
        )
        contact_component = _slug_component(contact, "") if contact else _stable_hash_component(
            "targethash", _known_value(packet.get("send_target", {}).get("target_label"), packet.get("conversation", {}).get("thread_ref"))
        )
        if not contact_component:
            missing.append("contact_or_target")
        conversation = _known_value(
            packet.get("subject", {}).get("conversation_id"),
            packet.get("conversation", {}).get("conversation_id"),
            packet.get("send_target", {}).get("conversation_id"),
        )
        thread = _known_value(packet.get("subject", {}).get("thread_ref"), packet.get("conversation", {}).get("thread_ref"))
        conversation_component = _slug_component(conversation, "") if conversation else _stable_hash_component("threadhash", thread)
        if not conversation_component:
            if scope in {"admin_decision", "crm_mutation"}:
                conversation_component = "no-conversation"
            else:
                missing.append("conversation_or_thread")
        hash_component = draft_hash or packet.get("draft", {}).get("draft_hash") or packet.get("approval_state", {}).get("latest_draft_hash")
        if not hash_component:
            missing.append("draft_or_action_hash")
        date_bucket = _canonical_date_bucket(packet)
        if not date_bucket:
            missing.append("date_or_slot_bucket")
        lookup_fields = {
            "brand_slug": brand_slug,
            "contact_or_target": contact_component,
            "conversation_or_thread": conversation_component,
            "action_type": canonical_action,
            "draft_or_action_hash": hash_component,
            "date_or_slot_bucket": date_bucket,
        }
        primary_key = None if missing else ":".join(
            [
                "blue",
                "v1",
                scope,
                brand_slug,
                str(contact_component),
                str(conversation_component),
                canonical_action,
                str(hash_component),
                str(date_bucket),
            ]
        )
    return {
        "primary_key": primary_key,
        "dedupe_scope": scope,
        "aliases": _canonical_aliases(packet, supplied_key),
        "lookup_fields": lookup_fields,
        "missing_fields": missing,
        "canonical": bool(primary_key),
    }


def _contact_name_from_target(target: str | None) -> str | None:
    if not target:
        return None
    match = re.search(r"\bto\s+(.+?)/contact\b", target, re.IGNORECASE)
    if not match:
        return None
    name = match.group(1).strip()
    if not name or name.lower() in {"contact", "unknown"}:
        return None
    return name


def _contains_specific_datetime(text: str | None) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\b\d{1,2}:\d{2}\b", text)
        or re.search(r"\b\d{1,2}\s*(?:am|pm)\b", text, re.IGNORECASE)
        or re.search(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            text,
            re.IGNORECASE,
        )
    )


def _contains_price(text: str | None) -> bool:
    return bool(text and re.search(r"\$\s*\d", text))


def _contains_link(text: str | None) -> bool:
    return bool(text and re.search(r"https?://", text, re.IGNORECASE))


def _first_match(pattern: str, text: str, default: str | None = None, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else default


def _section_after_heading(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^## ", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def _fenced_block_after_heading(text: str, heading: str) -> str | None:
    section = _section_after_heading(text, heading)
    if not section:
        return None
    match = re.search(r"```(?:text|yaml)?\s*\n(.*?)\n```", section, re.DOTALL)
    return match.group(1).strip() if match else section.strip() or None


def _base_packet(
    *,
    approval_id: str,
    source_type: str,
    source_path: Path,
    artifact_checked_at: str | None = None,
) -> dict[str, Any]:
    return {
        "approval_id": approval_id,
        "schema_version": SCHEMA_VERSION,
        "packet_status": "malformed",
        "status_reason": "Packet has not been normalized yet.",
        "source": {
            "source_type": source_type,
            "source_path": _safe_source_path(source_path),
            "source_task_id": None,
            "source_row": None,
            "source_artifacts": [],
            "imported_at": _now_iso(),
            "artifact_checked_at": artifact_checked_at,
        },
        "brand": {
            "name": "unknown",
            "location_id": None,
            "calendar_ids": [],
            "timezone": "Australia/Sydney",
        },
        "contact": {
            "contact_id": "unknown",
            "name": "unknown",
            "phone": None,
            "email": None,
            "source_confidence": "artifact_parse",
        },
        "conversation": {
            "conversation_id": "unknown",
            "channel": "unknown",
            "channel_source": "target_label_or_source_packet",
            "thread_ref": "unknown",
            "latest_customer_summary": "Not embedded; local source packet omitted latest customer message.",
            "latest_customer_at": None,
            "latest_customer_source": "source_packet_omitted_message",
            "latest_customer_message_type": None,
            "last_outbound_summary": "No outbound context found in local source packet.",
            "last_outbound_at": None,
            "last_outbound_source": "source_packet_omitted_outbound",
        },
        "crm_context": {
            "classification": "unknown",
            "opportunity": None,
            "appointments": [],
            "tasks": [],
            "existing_tags": [],
            "payments_or_invoices": None,
            "crm_context_summary": "Not embedded; live re-fetch required before action.",
        },
        "booking_context": {
            "is_booking_related": False,
            "job_location": "not relevant",
            "live_calendar_check": {"checked_at": None, "source": None, "result_summary": "not relevant"},
            "weather_check": {"checked_at": None, "result_summary": "not relevant"},
            "ledger_status": "not relevant",
            "proposed_slots": [],
        },
        "decision": {
            "why_it_matters": "Approval packet imported from local artifact for operator review.",
            "recommended_action": "Review packet details; MVP is read-only and cannot send or mutate CRM.",
            "approval_options": [],
            "risk_level": "unknown",
            "risk_summary": "unknown",
        },
        "draft": {
            "customer_facing": False,
            "draft_text": None,
            "draft_hash": None,
            "draft_hash_basis": "none",
            "draft_language": "en-AU",
            "contains_link": False,
            "contains_specific_datetime": False,
            "contains_price": False,
            "manual_review_only": False,
        },
        "idempotency": {
            "primary_key": None,
            "dedupe_scope": None,
            "aliases": [],
            "lookup_fields": {},
            "missing_fields": [],
        },
        "congruence": {
            "canonical_object": "blue.approval.v1-compatible projection",
            "live_state_reconciliation_rule": "Before any customer-facing send or CRM mutation, re-fetch live GHL state and verify the canonical idempotency key is still unhandled.",
            "handled_state_record": "approval_action_requests.result mirrors the future handled-actions contract; live writes remain disabled in this build.",
        },
        "send_target": {
            "channel": "unknown",
            "target_label": "unknown",
            "contact_id": "unknown",
            "conversation_id": "unknown",
            "brand_name": "unknown",
        },
        "evidence": [],
        "verification": {
            "required_before_send": False,
            "verification_steps": [],
            "stale_after_seconds": 600,
            "latest_verified_at": None,
        },
        "kanban_context": {
            "task_id": None,
            "parent_task_id": None,
            "status": None,
            "assignee": None,
            "block_reason": None,
            "comments_summary": None,
            "title": None,
            "board": None,
            "source": None,
        },
        "safety": {
            "customer_text_untrusted": True,
            "allowed_mvp_actions": ALLOWED_MVP_ACTIONS,
            "disallowed_mvp_actions": DISALLOWED_MVP_ACTIONS,
        },
        "parse_errors": [],
        "data_contract": {
            "required_fields": [
                "contact.name",
                "conversation.channel",
                "conversation.latest_customer_summary",
                "conversation.latest_customer_at",
                "conversation.last_outbound_summary",
                "conversation.last_outbound_at",
            ],
            "missing_data": [],
        },
        "ui_indicators": {
            "stale": False,
            "malformed": False,
            "booking_related": False,
            "manual_review_only": False,
        },
    }


def _finalize_packet(packet: dict[str, Any]) -> dict[str, Any]:
    draft_text = packet["draft"].get("draft_text")
    packet["draft"]["draft_hash"] = _sha256_text(draft_text)
    packet["draft"]["draft_hash_basis"] = "normalized_exact_text" if draft_text else "none"
    packet["draft"]["contains_link"] = _contains_link(draft_text)
    packet["draft"]["contains_price"] = _contains_price(draft_text)
    packet["draft"]["contains_specific_datetime"] = _contains_specific_datetime(draft_text)

    customer_facing = bool(packet["draft"].get("customer_facing"))
    if customer_facing:
        packet["decision"]["approval_options"] = ["approve/send", "edit: ...", "reject"]
        packet["verification"]["required_before_send"] = True
        if not packet["verification"]["verification_steps"]:
            packet["verification"]["verification_steps"] = [
                "Re-fetch latest conversation and verify no newer inbound/outbound changed the context.",
                "Confirm the exact draft hash matches the approved draft before any send.",
            ]

    classification = (packet["crm_context"].get("classification") or "").lower()
    risk_summary = (packet["decision"].get("risk_summary") or "").lower()
    booking_related = classification == "booking-needed" or packet["draft"].get("contains_specific_datetime")
    packet["booking_context"]["is_booking_related"] = bool(packet["booking_context"].get("is_booking_related") or booking_related)
    packet["ui_indicators"]["booking_related"] = bool(packet["booking_context"]["is_booking_related"])
    packet["idempotency"] = _canonical_identity(packet)

    errors: list[str] = packet["parse_errors"]
    if not packet.get("approval_id"):
        errors.append("missing approval_id")
    if customer_facing and not draft_text:
        errors.append("customer_facing packet missing draft_text")
    if customer_facing and packet["contact"].get("contact_id") == "unknown" and packet["conversation"].get("thread_ref") == "unknown":
        errors.append("customer_facing packet missing contact_id/thread_ref")
    if customer_facing and not packet["decision"].get("approval_options"):
        errors.append("customer_facing packet missing approval options")

    if errors:
        packet["packet_status"] = "malformed"
        packet["status_reason"] = "; ".join(errors)
    elif packet["packet_status"] in {"blocked", "stale_obsolete"}:
        packet["status_reason"] = packet["status_reason"] or "Imported status overlay requires operator review."
    elif packet["draft"].get("manual_review_only"):
        packet["packet_status"] = "manual_review_only"
        packet["status_reason"] = "Manual review only; MVP cannot send or mutate CRM."
    elif customer_facing and (booking_related or "stale" in risk_summary or "recheck" in risk_summary):
        packet["packet_status"] = "needs_reverify"
        packet["status_reason"] = "Customer-facing packet requires fresh live-state reverify before any action."
    elif customer_facing:
        packet["packet_status"] = "pending_approval"
        packet["status_reason"] = "Customer-facing draft requires Gabriel approval; MVP is read-only."
    else:
        packet["packet_status"] = "blocked"
        packet["status_reason"] = "Non-send guard or missing approval artifact imported for operator review."

    packet["ui_indicators"]["malformed"] = packet["packet_status"] == "malformed"
    packet["ui_indicators"]["manual_review_only"] = bool(packet["draft"].get("manual_review_only"))
    packet["ui_indicators"]["stale"] = packet["packet_status"] in {"needs_reverify", "stale_obsolete", "blocked"} or "stale" in (
        packet.get("status_reason") or ""
    ).lower()
    return packet


def _normalize_approval_item(item: dict[str, Any], root: dict[str, Any], path: Path) -> dict[str, Any]:
    approval_id = str(item.get("approval_id") or f"{root.get('task_id', path.stem)}-row-{item.get('source_row', 'unknown')}")
    packet = _base_packet(
        approval_id=approval_id,
        source_type="json_approval_item",
        source_path=path,
        artifact_checked_at=root.get("created_at"),
    )
    proposed_action = item.get("proposed_action") or {}
    target = item.get("send_target") or proposed_action.get("channel_target")
    draft_text = proposed_action.get("draft")
    evidence = item.get("evidence")
    risk = item.get("risk") or {}

    packet["source"].update(
        {
            "source_task_id": item.get("source_task_id") or root.get("task_id"),
            "source_row": item.get("source_row"),
            "source_artifacts": root.get("source_artifacts") or [],
        }
    )
    packet["brand"]["name"] = item.get("brand") or "unknown"
    packet["contact"].update(
        {
            "contact_id": item.get("contactId") or "unknown",
            "name": item.get("contactName") or _contact_name_from_target(target) or "unknown",
            "phone": _phone_from_target(target),
        }
    )
    packet["conversation"].update(
        {
            "conversation_id": item.get("conversationId") or "unknown",
            "channel": _channel_from_target(target),
            "channel_source": "send_target",
            "thread_ref": target or item.get("conversationId") or "unknown",
            "latest_customer_summary": evidence if isinstance(evidence, str) else "Not embedded; live re-fetch required before action.",
        }
    )
    packet["crm_context"]["classification"] = item.get("classification") or "unknown"
    packet["decision"].update(
        {
            "why_it_matters": "Customer-facing action is waiting in an approval artifact; wrong/stale replies can harm trust.",
            "recommended_action": f"Review {proposed_action.get('type', 'proposed action')}; re-fetch live state before any send.",
            "risk_level": risk.get("level") or "unknown",
            "risk_summary": risk.get("summary") or "unknown",
        }
    )
    packet["draft"].update(
        {
            "customer_facing": bool(proposed_action.get("customer_facing_send")),
            "draft_text": draft_text,
            "manual_review_only": proposed_action.get("type") == "manual_review_only" or (draft_text or "").lstrip().upper().startswith("FLAG"),
        }
    )
    packet["send_target"].update(
        {
            "channel": _channel_from_target(target),
            "target_label": target or "unknown",
            "contact_id": item.get("contactId") or "unknown",
            "conversation_id": item.get("conversationId") or "unknown",
            "brand_name": item.get("brand") or "unknown",
        }
    )
    packet["evidence"] = [{"kind": "artifact_text", "summary": evidence}] if isinstance(evidence, str) else []
    for source_artifact in root.get("source_artifacts") or []:
        packet["evidence"].append({"kind": "source_file", "path": source_artifact, "summary": "Referenced source artifact."})
    packet["verification"]["verification_steps"] = item.get("verification_step") or root.get("global_verification_before_any_action") or []
    packet["kanban_context"]["parent_task_id"] = root.get("task_id")
    return _finalize_packet(packet)


def _normalize_record(record: dict[str, Any], root: dict[str, Any], path: Path) -> dict[str, Any]:
    approval_id = f"{root.get('task_id', path.stem)}-row-{record.get('source_row', 'unknown')}"
    packet = _base_packet(
        approval_id=approval_id,
        source_type="json_record",
        source_path=path,
        artifact_checked_at=root.get("checked_at"),
    )
    target = record.get("send_target")
    draft_text = record.get("draft")
    evidence = record.get("evidence")
    classification = record.get("classification") or "unknown"

    packet["source"].update({"source_task_id": root.get("task_id"), "source_row": record.get("source_row")})
    packet["brand"]["name"] = record.get("brand") or "unknown"
    packet["contact"].update(
        {
            "contact_id": record.get("contactId") or "unknown",
            "name": record.get("contactName") or _contact_name_from_target(target) or "unknown",
            "phone": _phone_from_target(target),
        }
    )
    packet["conversation"].update(
        {
            "conversation_id": record.get("conversationId") or "unknown",
            "channel": _channel_from_target(target),
            "channel_source": "send_target",
            "thread_ref": target or record.get("conversationId") or "unknown",
            "latest_customer_summary": evidence if isinstance(evidence, str) else "Not embedded; live re-fetch required before action.",
        }
    )
    packet["crm_context"]["classification"] = classification
    packet["decision"].update(
        {
            "why_it_matters": "Draft-only record needs operator approval before any customer-facing action.",
            "recommended_action": "Review draft; re-fetch live state before any send.",
            "risk_level": "medium" if classification == "booking-needed" else "low",
            "risk_summary": "Derived from draft-only approval record.",
        }
    )
    packet["draft"].update(
        {
            "customer_facing": str(record.get("needs_approval", "")).lower() == "yes" and bool(draft_text),
            "draft_text": draft_text,
            "manual_review_only": (draft_text or "").lstrip().upper().startswith("FLAG"),
        }
    )
    packet["send_target"].update(
        {
            "channel": _channel_from_target(target),
            "target_label": target or "unknown",
            "contact_id": record.get("contactId") or "unknown",
            "conversation_id": record.get("conversationId") or "unknown",
            "brand_name": record.get("brand") or "unknown",
        }
    )
    packet["evidence"] = [{"kind": "artifact_text", "summary": evidence}] if isinstance(evidence, str) else []
    return _finalize_packet(packet)


def _normalize_missing_approval(root: dict[str, Any], path: Path) -> dict[str, Any]:
    approval_id = root.get("task_id") or path.stem
    packet = _base_packet(
        approval_id=str(approval_id),
        source_type="missing_approval_guard",
        source_path=path,
        artifact_checked_at=root.get("checked_at"),
    )
    packet["packet_status"] = "blocked"
    packet["status_reason"] = "Source artifact records blocked_missing_approval."
    packet["source"].update({"source_task_id": root.get("task_id")})
    packet["decision"].update(
        {
            "why_it_matters": "This guard prevents CRM mutation without Gabriel approval.",
            "recommended_action": root.get("needs_approval") or "Gabriel approval required before action.",
            "risk_level": "high",
            "risk_summary": "Missing approval guard; no CRM mutation is allowed.",
        }
    )
    packet["draft"].update({"customer_facing": False, "manual_review_only": True})
    packet["evidence"] = [
        {"kind": "artifact_text", "summary": str(item)} for item in (root.get("evidence") or [])
    ] + [
        {"kind": "search_performed", "summary": str(item)} for item in (root.get("searches_performed") or [])
    ]
    packet["kanban_context"]["task_id"] = root.get("task_id")
    return _finalize_packet(packet)


def _parse_recheck_overlays(path: Path) -> dict[int, dict[str, str | None]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    overlays: dict[int, dict[str, str | None]] = {}
    section_status: str | None = None
    current_row: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## Current / still approval-ready"):
            section_status = None
            current_row = None
            continue
        if line.startswith("## Already handled / obsolete"):
            section_status = "stale_obsolete"
            current_row = None
            continue
        if line.startswith("## Context changed"):
            section_status = "blocked"
            current_row = None
            continue
        match = re.match(r"\d+\.\s+(?:([^—/]+?)\s*/\s*)?Source row (\d+)(?:\s*/\s*(t_[0-9a-f]+))?.*", line)
        if match:
            current_row = int(match.group(2))
            task_id = match.group(1) or match.group(3)
            if section_status:
                overlays[current_row] = {"packet_status": section_status, "status_reason": "", "task_id": task_id.strip() if task_id else None}
            continue
        if current_row is not None and current_row in overlays and (line.startswith("Reason:") or line.startswith("Change:")):
            overlays[current_row]["status_reason"] = line.split(":", 1)[1].strip()
    return overlays


def _load_latest_inbound_context() -> dict[str, dict[str, Any]]:
    """Load local, read-only latest-inbound snapshots keyed by conversation ID/source row."""
    contexts: dict[str, dict[str, Any]] = {}
    for path in (
        ARTIFACT_ROOT / "t_42bcdc5a_latest_inbound_conversations.json",
        ARTIFACT_ROOT / "t_c317a5f5_classified_inbox_candidates.json",
    ):
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - stale/malformed enrichment source should not break packet import.
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            conversation_id = row.get("conversationId")
            preview = row.get("preview")
            message_at = row.get("lastMessageDate")
            direction = str(row.get("lastMessageDirection") or "").lower()
            message_type = row.get("lastMessageType")
            if not conversation_id or not preview or direction != "inbound":
                continue
            context = {
                "summary": str(preview),
                "at": message_at,
                "source": path.name,
                "channel": _channel_from_message_type(message_type),
                "message_type": message_type,
                "contact_name": row.get("contactName"),
                "contact_id": row.get("contactId"),
                "phone": row.get("phone"),
                "email": row.get("email"),
                "source_row": str(row.get("source_row")) if row.get("source_row") is not None else None,
            }
            contexts[str(conversation_id)] = context
            if context["contact_id"]:
                contexts[f"contact:{context['contact_id']}"] = context
            if context["source_row"]:
                contexts[f"row:{context['source_row']}"] = context
    return contexts


def _parse_recheck_message_context(path: Path) -> dict[str, dict[str, Any]]:
    """Parse local refreshed triage prose for latest ask and last-outbound context."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    contexts: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(
            r"\d+\.\s+(?:(t_[0-9a-f]+)\s*/\s*)?(?:Source row\s+)?(\d+)?(?:\s*/\s*(t_[0-9a-f]+))?.*?conversation\s+([A-Za-z0-9]+)",
            line,
            re.IGNORECASE,
        )
        if match:
            current = {
                "source": path.name,
                "source_row": match.group(2),
                "task_id": match.group(1) or match.group(3),
                "conversation_id": match.group(4),
            }
            if current["source_row"]:
                contexts[f"row:{current['source_row']}"] = current
            contexts[f"conversation:{current['conversation_id']}"] = current
            continue
        if current is None:
            continue
        inbound_match = re.match(r"Latest (?:customer ask|customer action|state):\s*(.+)$", line)
        if inbound_match:
            current["latest_customer_summary"] = inbound_match.group(1).strip()
            continue
        outbound_match = re.match(r"Last outbound:\s*(.+)$", line)
        if outbound_match:
            outbound = outbound_match.group(1).strip()
            current["last_outbound_summary"] = outbound
            # The refreshed prose sometimes embeds only a date, not an exact send timestamp.
            date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b", outbound)
            if date_match:
                current["last_outbound_at"] = date_match.group(0)
            current["last_outbound_source"] = path.name
    return contexts


def _apply_message_context(packets: list[dict[str, Any]]) -> None:
    inbound_contexts = _load_latest_inbound_context()
    recheck_contexts = _parse_recheck_message_context(ARTIFACT_ROOT / "t_bb320de1_refreshed_pending_drafts.md")
    for packet in packets:
        source = packet.get("source", {})
        conversation = packet.get("conversation", {})
        row_key = f"row:{source.get('source_row')}" if source.get("source_row") is not None else None
        conversation_key = f"conversation:{conversation.get('conversation_id')}" if conversation.get("conversation_id") else None
        contact_key = f"contact:{packet.get('contact', {}).get('contact_id')}" if packet.get("contact", {}).get("contact_id") else None
        inbound = (
            inbound_contexts.get(str(conversation.get("conversation_id")))
            or (inbound_contexts.get(row_key) if row_key else None)
            or (inbound_contexts.get(contact_key) if contact_key else None)
        )
        recheck = (recheck_contexts.get(row_key) if row_key else None) or (recheck_contexts.get(conversation_key) if conversation_key else None)

        if inbound:
            conversation["latest_customer_summary"] = inbound["summary"]
            conversation["latest_customer_at"] = inbound.get("at")
            conversation["latest_customer_source"] = inbound.get("source")
            conversation["latest_customer_message_type"] = inbound.get("message_type")
            if inbound.get("channel"):
                conversation["channel"] = inbound["channel"]
                conversation["channel_source"] = inbound.get("message_type") or inbound.get("source")
            if inbound.get("contact_name") and packet.get("contact", {}).get("name") == "unknown":
                packet["contact"]["name"] = inbound["contact_name"]
            if inbound.get("phone") and not packet.get("contact", {}).get("phone"):
                packet["contact"]["phone"] = inbound["phone"]
            if inbound.get("email") and not packet.get("contact", {}).get("email"):
                packet["contact"]["email"] = inbound["email"]
        elif recheck and recheck.get("latest_customer_summary"):
            conversation["latest_customer_summary"] = recheck["latest_customer_summary"]
            conversation["latest_customer_source"] = recheck.get("source")

        if recheck and recheck.get("last_outbound_summary"):
            conversation["last_outbound_summary"] = recheck["last_outbound_summary"]
            conversation["last_outbound_at"] = recheck.get("last_outbound_at")
            conversation["last_outbound_source"] = recheck.get("last_outbound_source")
        elif not conversation.get("last_outbound_summary"):
            conversation["last_outbound_summary"] = "No outbound found in local packet/source data."
            conversation["last_outbound_source"] = "no_outbound_found_in_local_sources"


def _apply_data_contract_reasons(packets: list[dict[str, Any]]) -> None:
    for packet in packets:
        missing: list[dict[str, str]] = []
        contact = packet.get("contact", {})
        conversation = packet.get("conversation", {})
        if contact.get("name") in {None, "", "unknown"}:
            missing.append(
                {
                    "field": "contact.name",
                    "reason": "source packet and local inbound snapshots omitted contact display name; use read-only GHL contact/conversation fetch before approval.",
                }
            )
        if conversation.get("channel") in {None, "", "unknown"}:
            missing.append(
                {
                    "field": "conversation.channel",
                    "reason": "source packet did not include channel and no local lastMessageType snapshot matched this packet.",
                }
            )
        latest_summary = str(conversation.get("latest_customer_summary") or "")
        if not latest_summary or "omitted latest customer message" in latest_summary.lower():
            missing.append(
                {
                    "field": "conversation.latest_customer_summary",
                    "reason": str(conversation.get("latest_customer_source") or "source_packet_omitted_message"),
                }
            )
        if not conversation.get("latest_customer_at"):
            missing.append(
                {
                    "field": "conversation.latest_customer_at",
                    "reason": str(conversation.get("latest_customer_source") or "source_packet_omitted_message"),
                }
            )
        if not conversation.get("last_outbound_summary") or "no outbound" in str(conversation.get("last_outbound_summary") or "").lower():
            missing.append(
                {
                    "field": "conversation.last_outbound_summary",
                    "reason": str(conversation.get("last_outbound_source") or "source_packet_omitted_outbound"),
                }
            )
        if not conversation.get("last_outbound_at"):
            missing.append(
                {
                    "field": "conversation.last_outbound_at",
                    "reason": str(conversation.get("last_outbound_source") or "source_packet_omitted_outbound"),
                }
            )
        packet.setdefault("data_contract", {})["missing_data"] = missing
        packet.setdefault("ui_indicators", {})["data_incomplete"] = bool(missing)


def _apply_recheck_overlays(packets: list[dict[str, Any]]) -> None:
    overlays = _parse_recheck_overlays(ARTIFACT_ROOT / "t_bb320de1_refreshed_pending_drafts.md")
    for packet in packets:
        raw_row = packet.get("source", {}).get("source_row")
        try:
            row = int(raw_row)
        except (TypeError, ValueError):
            row = raw_row
        overlay = overlays.get(row)
        if not overlay:
            continue
        packet["packet_status"] = overlay["packet_status"]
        packet["status_reason"] = overlay["status_reason"] or "Status overlaid from latest refreshed pending draft triage report."
        if overlay.get("task_id"):
            packet["kanban_context"]["task_id"] = overlay["task_id"]
        _finalize_packet(packet)


def _normalize_booking_markdown(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    task_id = _first_match(r"^Task:\s*(\S+)", text, path.stem, re.MULTILINE) or path.stem
    created_at = _first_match(r"^Created:\s*(.+)$", text, None, re.MULTILINE)
    packet = _base_packet(
        approval_id=task_id,
        source_type="markdown_booking_packet",
        source_path=path,
        artifact_checked_at=created_at,
    )
    customer_section = _section_after_heading(text, "Customer / location")
    conversation_section = _section_after_heading(text, "Conversation recheck")
    calendar_section = _section_after_heading(text, "Calendar / ledger recheck")
    weather_section = _section_after_heading(text, "Weather")
    slots_section = _section_after_heading(text, "Safe alternative Saturday options")
    ledger_block = _fenced_block_after_heading(text, "Booking-slot ledger for this draft") or ""
    draft_text = _section_after_heading(text, "Revised draft needing Gabriel approval").split("Needs Gabriel approval before send.", 1)[0].strip()
    target = _first_match(r"^- Channel target if approved:\s*(.+)$", customer_section, None, re.MULTILINE)

    proposed_slots = []
    for slot_match in re.finditer(r"^\d+\.\s*(Saturday .+?) — (.+)$", slots_section, re.MULTILINE):
        proposed_slots.append(
            {
                "starts_at": None,
                "display_text": slot_match.group(1).strip(),
                "source_status": slot_match.group(2).strip(),
            }
        )
    if not proposed_slots:
        for slot_match in re.finditer(r"display_text:\s*\"([^\"]+)\"[\s\S]*?source_status:\s*\"([^\"]+)\"", ledger_block):
            proposed_slots.append(
                {"starts_at": None, "display_text": slot_match.group(1), "source_status": slot_match.group(2)}
            )

    packet["source"].update({"source_task_id": task_id})
    packet["brand"]["name"] = _first_match(r"^- Brand/location:\s*(.+)$", customer_section, "unknown", re.MULTILINE) or "unknown"
    packet["contact"].update(
        {
            "contact_id": _first_match(r"^- Contact ID:\s*(.+)$", customer_section, "unknown", re.MULTILINE) or "unknown",
            "name": _first_match(r"^- Customer:\s*(.+)$", customer_section, "unknown", re.MULTILINE) or "unknown",
            "phone": _phone_from_target(target),
        }
    )
    packet["conversation"].update(
        {
            "conversation_id": _first_match(r"^- Conversation ID:\s*(.+)$", customer_section, "unknown", re.MULTILINE) or "unknown",
            "channel": _channel_from_target(target),
            "thread_ref": target or "unknown",
            "latest_customer_summary": _first_match(r"Latest customer message remains .+?:\s*\n\"(.+?)\"", conversation_section, None, re.DOTALL)
            or "Not embedded; live re-fetch required before action.",
            "last_outbound_summary": "No newer GHL outbound/inbound message was present in the fetched conversation page.",
        }
    )
    packet["crm_context"]["classification"] = "booking-needed"
    packet["booking_context"].update(
        {
            "is_booking_related": True,
            "job_location": _first_match(r"^- Service location:\s*(.+)$", customer_section, "location not found", re.MULTILINE) or "location not found",
            "live_calendar_check": {
                "checked_at": created_at,
                "source": _first_match(r"Calendar source checked live in this run:\s*\n-\s*(.+)$", calendar_section, "markdown_booking_packet", re.MULTILINE),
                "result_summary": _first_match(r"result_summary:\s*\"([^\"]+)\"", ledger_block, None)
                or "Calendar and ledger recheck embedded in booking packet.",
            },
            "weather_check": {"checked_at": created_at, "result_summary": weather_section or "not checked"},
            "ledger_status": _first_match(r"send_status:\s*\"([^\"]+)\"", ledger_block, "draft_only_not_sent") or "draft_only_not_sent",
            "proposed_slots": proposed_slots,
        }
    )
    packet["decision"].update(
        {
            "why_it_matters": "Customer is waiting for alternative booking availability; stale slots risk double-booking.",
            "recommended_action": "Gabriel approve, edit, choose one alternative, or reject the revised SMS; re-check live state before any send.",
            "risk_level": "medium",
            "risk_summary": "Booking date options require fresh calendar, ledger, and conversation recheck before send.",
        }
    )
    packet["draft"].update({"customer_facing": True, "draft_text": draft_text})
    packet["send_target"].update(
        {
            "channel": _channel_from_target(target),
            "target_label": target or "unknown",
            "contact_id": packet["contact"]["contact_id"],
            "conversation_id": packet["conversation"]["conversation_id"],
            "brand_name": packet["brand"]["name"],
        }
    )
    packet["verification"]["verification_steps"] = [
        "Re-fetch the latest conversation before any send.",
        "Re-check Solar Renew calendar and booking-slot ledger before offering a date.",
        "Confirm the approved draft hash still matches the customer-facing text.",
    ]
    packet["kanban_context"]["task_id"] = task_id
    packet["evidence"] = [
        {"kind": "source_file", "path": str(path), "summary": "Booking-specific markdown approval packet."},
        {"kind": "artifact_text", "summary": "Alternative booking options and ledger evidence imported as inert text."},
    ]
    return _finalize_packet(packet)


def _malformed_artifact_packet(path: Path, error: str) -> dict[str, Any]:
    packet = _base_packet(approval_id=f"malformed:{path.name}", source_type="unknown", source_path=path)
    packet["parse_errors"].append(error)
    packet["status_reason"] = error
    return _finalize_packet(packet)


def _status_counts(tasks: list[Any]) -> dict[str, int]:
    counts = Counter(getattr(task, "status", "unknown") for task in tasks)
    return {
        "total": len(tasks),
        "triage": counts.get("triage", 0),
        "todo": counts.get("todo", 0),
        "ready": counts.get("ready", 0),
        "running": counts.get("running", 0),
        "blocked": counts.get("blocked", 0),
        "done": counts.get("done", 0),
    }


def _compact_task(task: Any, *, board: str) -> dict[str, Any]:
    return {
        "id": getattr(task, "id", None),
        "title": getattr(task, "title", None),
        "status": getattr(task, "status", None),
        "assignee": getattr(task, "assignee", None),
        "priority": getattr(task, "priority", None),
        "tenant": getattr(task, "tenant", None),
        "board": board,
    }


def _load_board_summary(board: str) -> dict[str, Any]:
    """Return a compact read-only summary for one Kanban board."""
    try:
        from hermes_cli import kanban_db
    except Exception as exc:  # noqa: BLE001 - UI should degrade instead of crashing.
        return {"board": board, "available": False, "counts": {"total": 0}, "recent_tasks": [], "source": "kanban_db", "error": str(exc)}

    try:
        if not kanban_db.board_exists(board):
            return {
                "board": board,
                "available": False,
                "counts": {"total": 0},
                "recent_tasks": [],
                "source": "kanban_db",
                "error": f"board {board!r} does not exist",
            }
        db_path = kanban_db.board_dir(board) / "kanban.db"
        kanban_db.init_db(db_path=db_path)
        conn = kanban_db.connect(db_path=db_path)
        try:
            tasks = kanban_db.list_tasks(conn, include_archived=False)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        return {"board": board, "available": False, "counts": {"total": 0}, "recent_tasks": [], "source": "kanban_db", "error": str(exc)}

    status_order = {"blocked": 0, "running": 1, "ready": 2, "todo": 3, "triage": 4, "done": 5}
    notable = sorted(
        tasks,
        key=lambda task: (status_order.get(getattr(task, "status", ""), 9), -int(getattr(task, "priority", 0) or 0), int(getattr(task, "created_at", 0) or 0)),
    )[:8]
    return {
        "board": board,
        "available": True,
        "counts": _status_counts(tasks),
        "recent_tasks": [_compact_task(task, board=board) for task in notable],
        "source": "kanban_db",
    }


def _load_task_context(task_id: str | None) -> dict[str, Any] | None:
    """Find task status for a packet-linked Kanban task across the relevant boards."""
    if not task_id:
        return None
    try:
        from hermes_cli import kanban_db
    except Exception:  # noqa: BLE001
        return None

    for board in KANBAN_CONTEXT_BOARDS:
        try:
            if not kanban_db.board_exists(board):
                continue
            db_path = kanban_db.board_dir(board) / "kanban.db"
            kanban_db.init_db(db_path=db_path)
            conn = kanban_db.connect(db_path=db_path)
            try:
                task = kanban_db.get_task(conn, task_id)
                if task is None:
                    continue
                summary = None
                try:
                    summary = kanban_db.latest_summary(conn, task_id)
                except Exception:  # noqa: BLE001
                    summary = None
                return {
                    "task_id": task.id,
                    "status": task.status,
                    "assignee": task.assignee,
                    "title": task.title,
                    "board": board,
                    "block_reason": getattr(task, "result", None) if task.status == "blocked" else None,
                    "comments_summary": summary,
                    "source": "kanban_db",
                }
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            continue
    return {"task_id": task_id, "status": "missing", "source": "kanban_db"}


def _infer_packet_task_id(packet: dict[str, Any]) -> str | None:
    context = packet.get("kanban_context") or {}
    for value in (
        context.get("task_id"),
        packet.get("approval_id"),
        packet.get("source", {}).get("source_task_id"),
        context.get("parent_task_id"),
    ):
        if isinstance(value, str):
            match = re.search(r"\bt_[0-9a-f]+\b", value)
            if match:
                return match.group(0)
    return None


def _hydrate_kanban_context(packet: dict[str, Any]) -> dict[str, Any]:
    task_id = _infer_packet_task_id(packet)
    if task_id:
        packet["kanban_context"]["task_id"] = task_id
        task_context = _load_task_context(task_id)
        if task_context:
            for key, value in task_context.items():
                if value is not None or key not in packet["kanban_context"]:
                    packet["kanban_context"][key] = value
    return packet


def _cron_schedule_display(job: dict[str, Any]) -> str | None:
    if job.get("schedule_display"):
        return str(job["schedule_display"])
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        return schedule.get("display") or schedule.get("expr") or schedule.get("run_at")
    if schedule is not None:
        return str(schedule)
    return None


def _load_relevant_cron_jobs() -> list[dict[str, Any]]:
    """Return compact metadata for cron jobs relevant to GHL Manager work."""
    try:
        from cron import jobs as cron_jobs
    except Exception:  # noqa: BLE001
        return []

    relevant: list[dict[str, Any]] = []
    try:
        jobs = cron_jobs.list_jobs(include_disabled=True)
    except Exception:  # noqa: BLE001
        return []
    for job in jobs:
        haystack = " ".join(
            str(value or "")
            for value in (
                job.get("id"),
                job.get("name"),
                job.get("prompt"),
                job.get("schedule_display"),
                _cron_schedule_display(job),
            )
        ).lower()
        if not any(term in haystack for term in RELEVANT_CRON_TERMS):
            continue
        enabled = bool(job.get("enabled", True))
        state = job.get("state") or ("active" if enabled else "paused")
        relevant.append(
            {
                "id": job.get("id"),
                "name": job.get("name") or job.get("id"),
                "schedule": _cron_schedule_display(job),
                "enabled": enabled,
                "state": state,
                "next_run_at": job.get("next_run_at"),
                "last_run_at": job.get("last_run_at"),
                "last_status": job.get("last_status"),
            }
        )
    return relevant


def _load_packets() -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    json_paths = [
        ARTIFACT_ROOT / "pending-approval-t_c50747c6.json",
        ARTIFACT_ROOT / "t_a1c85349_pending_lead_reply_drafts.json",
        ARTIFACT_ROOT / "t_627dfbf5_blocked_missing_approval.json",
    ]
    for path in json_paths:
        if not path.exists():
            packets.append(_malformed_artifact_packet(path, "expected approval artifact file is missing"))
            continue
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - malformed artifacts must not crash the inbox.
            packets.append(_malformed_artifact_packet(path, f"invalid JSON: {exc.__class__.__name__}"))
            continue

        if isinstance(root, dict) and isinstance(root.get("approval_items"), list):
            packets.extend(_normalize_approval_item(item, root, path) for item in root["approval_items"] if isinstance(item, dict))
        elif isinstance(root, dict) and isinstance(root.get("records"), list):
            packets.extend(_normalize_record(record, root, path) for record in root["records"] if isinstance(record, dict))
        elif isinstance(root, dict) and root.get("status") == "blocked_missing_approval":
            packets.append(_normalize_missing_approval(root, path))
        else:
            packets.append(_malformed_artifact_packet(path, "unrecognized approval artifact shape"))

    booking_packet = _normalize_booking_markdown(ARTIFACT_ROOT / "t_ba5b8b47-vanessa-alternative-booking-options.md")
    if booking_packet:
        packets.append(booking_packet)

    _apply_message_context(packets)
    _apply_recheck_overlays(packets)
    _apply_data_contract_reasons(packets)
    packets = [_hydrate_kanban_context(packet) for packet in packets]
    packets.sort(key=lambda packet: (packet["packet_status"], packet["approval_id"]))
    return packets


@router.get("/health")
async def health() -> dict[str, object]:
    """Return plugin status without touching GHL or customer data."""
    return {
        "plugin": PLUGIN_NAME,
        "label": PLUGIN_LABEL,
        "version": PLUGIN_VERSION,
        "status": "ok",
        "read_only": True,
        "warning": LOCAL_ONLY_WARNING,
    }


@router.get("/config")
async def config() -> dict[str, object]:
    """Return safe static UI capabilities for the read-only plugin."""
    return {
        "plugin": PLUGIN_NAME,
        "label": PLUGIN_LABEL,
        "version": PLUGIN_VERSION,
        "read_only": True,
        "mutations_enabled": False,
        "live_execution_enabled": _live_execution_enabled(),
        "artifact_root": str(ARTIFACT_ROOT),
        "endpoints": [
            "GET /health",
            "GET /config",
            "GET /context",
            "GET /decision-view",
            "GET /packets",
            "GET /packets/{approval_id}",
            "GET /approval-state",
            "GET /approval-state/{approval_id}",
            "GET /action-requests",
            "POST /approval-state/{approval_id}/events",
            "POST /approval-state/{approval_id}/action-requests",
            "POST /action-requests/process",
        ],
        "capabilities": [
            "display operator-facing GHL Manager approval inbox",
            "read local approval packet artifacts from the allowlisted audit directory",
            "show staleness, reverify, malformed, and manual-review indicators",
            "render customer and CRM artifact text as inert data in the frontend",
            "surface queued/blocked Action Bridge requests as compact notifications",
            "surface stale queued Action Bridge requests when processor cadence is missed",
            "surface Gabriel's canonical Blue/GHL decision view and live recheck summary",
            "queue auditable guarded action requests for server-side executor review",
            "perform explicit GET-only read-only GHL reverify for queued reverify requests when operator invokes processor with dry_run=false",
        ],
        "disabled_capabilities": [
            "send customer messages",
            "update contacts or opportunities",
            "book or modify appointments",
            "send or execute approved actions",
            "live customer send execution even when read-only reverify is available",
            "live action execution unless GHL_MANAGER_LIVE_EXECUTION=true and executor preflight passes",
        ],
        "warning": LOCAL_ONLY_WARNING,
    }


@router.get("/decision-view")
async def decision_view() -> dict[str, object]:
    """Return Gabriel's current Blue/GHL decision view without mutating GHL or local state."""
    return _decision_view()


@router.get("/context")
async def context() -> dict[str, object]:
    """Return read-only Kanban, action-request, and cron context for the GHL Manager UI."""
    boards = [_load_board_summary(board) for board in KANBAN_CONTEXT_BOARDS]
    cron_jobs = _load_relevant_cron_jobs()
    action_requests = _recent_action_request_notifications()
    missing_data_sources = []
    for board in boards:
        if not board.get("available"):
            missing_data_sources.append(
                {
                    "kind": "kanban_board",
                    "name": board.get("board"),
                    "reason": board.get("error") or "board unavailable",
                }
            )
    return {
        "plugin": PLUGIN_NAME,
        "read_only": True,
        "mutations_enabled": False,
        "boards": boards,
        "action_requests": action_requests,
        "action_request_count": len(action_requests),
        "action_request_attention": _action_request_attention_summary(action_requests),
        "cron_jobs": cron_jobs,
        "missing_data_sources": missing_data_sources,
    }


@router.get("/packets")
async def packets() -> dict[str, object]:
    """Return normalized approval packets imported from local artifacts only."""
    imported = _load_packets()
    approval_store = _record_approval_packets(imported)
    imported = [_attach_approval_state(packet) for packet in imported]
    status_counts = Counter(packet["packet_status"] for packet in imported)
    return {
        "plugin": PLUGIN_NAME,
        "read_only": True,
        "mutations_enabled": False,
        "artifact_root": str(ARTIFACT_ROOT),
        "approval_store": approval_store,
        "canonical_projection": _canonical_projection_summary(),
        "counts": {"total": len(imported), **dict(status_counts)},
        "packets": imported,
    }


@router.get("/packets/{approval_id}")
async def packet_detail(approval_id: str) -> dict[str, object]:
    """Return a single normalized approval packet by stable approval ID."""
    imported = _load_packets()
    _record_approval_packets(imported)
    for packet in imported:
        if packet["approval_id"] == approval_id:
            return _attach_approval_state(packet)
    raise HTTPException(status_code=404, detail="approval packet not found")


@router.get("/approval-state")
async def approval_state_index() -> dict[str, object]:
    """Return local-only approval states from the append-only SQLite event store."""
    states = _approval_state_index()
    canonical_counts = Counter(
        (state.get("approval_index") or {}).get("current_status") or state.get("current_status") for state in states
    )
    return {
        "plugin": PLUGIN_NAME,
        "read_only": True,
        "mutations_enabled": False,
        "store_path": str(APPROVAL_DB_PATH),
        "states": states,
        "counts": {"total": len(states), **dict(canonical_counts)},
    }


@router.get("/approval-state/{approval_id}")
async def approval_state_detail(approval_id: str) -> dict[str, object]:
    """Return local-only append-only approval state for a single packet."""
    return _get_approval_state(approval_id)


@router.get("/action-requests")
async def action_requests() -> dict[str, object]:
    """Return compact operator notifications for queued/blocked Action Bridge requests."""
    requests = _recent_action_request_notifications()
    return {
        "plugin": PLUGIN_NAME,
        "read_only": True,
        "mutations_enabled": False,
        "store_path": str(APPROVAL_DB_PATH),
        "action_requests": requests,
        "action_request_count": len(requests),
        "attention": _action_request_attention_summary(requests),
        "counts": {"total": len(requests), **dict(Counter(request["status"] for request in requests))},
    }


@router.post("/approval-state/{approval_id}/events")
async def approval_state_event(approval_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, object]:
    """Append a local approval decision event; never sends or mutates live CRM."""
    return _append_approval_event(approval_id, payload)


@router.post("/approval-state/{approval_id}/action-requests")
async def approval_action_request(approval_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, object]:
    """Queue a guarded action request for the server-side executor; never calls GHL directly."""
    return _create_action_request(approval_id, payload)


@router.post("/action-requests/process")
async def action_requests_process(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, object]:
    """Process queued requests; dry-run by default, GET-only live reverify when explicitly requested."""
    dry_run = bool(payload.get("dry_run", True))
    return process_pending_action_requests(dry_run=dry_run)
