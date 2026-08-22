"""ORCH V4 SQLite digest UDFs + additive trigger guards.

Server recomputes digests inside the DB connection. Caller-supplied digest
columns are assertions only; mismatch aborts the statement.

Covered:
- orch_origins.route_json/route_digest
- orch_requests.request_json/request_digest
- orch_results.result_json/result_digest
- orch_events.payload_json/payload_digest + event_key
- orch_delivery_receipts.receipt_json/receipt_digest
- orch_delivery_attempt_events.event_json/event_digest
- orch_delivery_attempts.adapter_evidence_json/adapter_evidence_digest (when json present)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from hermes_cli.kanban_orch_canonical import (
    CanonicalError,
    digest,
    event_key,
    request_digest,
    result_digest,
    route_digest,
    strict_json_loads,
)


DIGEST_GUARD_SQL = """
CREATE TRIGGER IF NOT EXISTS orch_v4_origins_route_digest_guard
BEFORE INSERT ON orch_origins
BEGIN
  SELECT CASE WHEN orch_route_digest_eq(NEW.route_json, NEW.route_digest) != 1
    THEN RAISE(ABORT, 'route_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_origins_route_digest_update_guard
BEFORE UPDATE OF route_json, route_digest ON orch_origins
BEGIN
  SELECT CASE WHEN orch_route_digest_eq(NEW.route_json, NEW.route_digest) != 1
    THEN RAISE(ABORT, 'route_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_requests_request_digest_guard
BEFORE INSERT ON orch_requests
BEGIN
  SELECT CASE WHEN orch_request_digest_eq(NEW.request_json, NEW.request_digest) != 1
    THEN RAISE(ABORT, 'request_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_requests_request_digest_update_guard
BEFORE UPDATE OF request_json, request_digest ON orch_requests
BEGIN
  SELECT CASE WHEN orch_request_digest_eq(NEW.request_json, NEW.request_digest) != 1
    THEN RAISE(ABORT, 'request_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_results_result_digest_guard
BEFORE INSERT ON orch_results
BEGIN
  SELECT CASE WHEN orch_result_digest_eq(NEW.result_json, NEW.result_digest) != 1
    THEN RAISE(ABORT, 'result_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_results_result_digest_update_guard
BEFORE UPDATE OF result_json, result_digest ON orch_results
BEGIN
  SELECT CASE WHEN orch_result_digest_eq(NEW.result_json, NEW.result_digest) != 1
    THEN RAISE(ABORT, 'result_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_events_payload_digest_guard
BEFORE INSERT ON orch_events
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.payload_json, NEW.payload_digest) != 1
    THEN RAISE(ABORT, 'payload_digest_mismatch') END;
  SELECT CASE WHEN orch_event_key_eq(
      NEW.board_instance_id, NEW.tenant_scope, NEW.orch_id,
      NEW.lifecycle_revision, NEW.cancel_epoch, NEW.event_kind,
      NEW.target_key, NEW.payload_digest, NEW.event_key
    ) != 1
    THEN RAISE(ABORT, 'event_key_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_events_payload_digest_update_guard
BEFORE UPDATE OF payload_json, payload_digest, event_key, event_kind, target_key,
                 lifecycle_revision, cancel_epoch, orch_id, tenant_scope, board_instance_id
ON orch_events
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.payload_json, NEW.payload_digest) != 1
    THEN RAISE(ABORT, 'payload_digest_mismatch') END;
  SELECT CASE WHEN orch_event_key_eq(
      NEW.board_instance_id, NEW.tenant_scope, NEW.orch_id,
      NEW.lifecycle_revision, NEW.cancel_epoch, NEW.event_kind,
      NEW.target_key, NEW.payload_digest, NEW.event_key
    ) != 1
    THEN RAISE(ABORT, 'event_key_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_receipts_receipt_digest_guard
BEFORE INSERT ON orch_delivery_receipts
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.receipt_json, NEW.receipt_digest) != 1
    THEN RAISE(ABORT, 'receipt_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_receipts_receipt_digest_update_guard
BEFORE UPDATE OF receipt_json, receipt_digest ON orch_delivery_receipts
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.receipt_json, NEW.receipt_digest) != 1
    THEN RAISE(ABORT, 'receipt_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_attempt_events_event_digest_guard
BEFORE INSERT ON orch_delivery_attempt_events
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.event_json, NEW.event_digest) != 1
    THEN RAISE(ABORT, 'attempt_event_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_attempt_events_event_digest_update_guard
BEFORE UPDATE OF event_json, event_digest ON orch_delivery_attempt_events
BEGIN
  SELECT CASE WHEN orch_canonical_digest_eq(NEW.event_json, NEW.event_digest) != 1
    THEN RAISE(ABORT, 'attempt_event_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_attempts_evidence_digest_guard
BEFORE INSERT ON orch_delivery_attempts
WHEN NEW.adapter_evidence_json IS NOT NULL
BEGIN
  SELECT CASE WHEN NEW.adapter_evidence_digest IS NULL
      OR orch_canonical_digest_eq(NEW.adapter_evidence_json, NEW.adapter_evidence_digest) != 1
    THEN RAISE(ABORT, 'adapter_evidence_digest_mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS orch_v4_attempts_evidence_digest_update_guard
BEFORE UPDATE OF adapter_evidence_json, adapter_evidence_digest ON orch_delivery_attempts
WHEN NEW.adapter_evidence_json IS NOT NULL
BEGIN
  SELECT CASE WHEN NEW.adapter_evidence_digest IS NULL
      OR orch_canonical_digest_eq(NEW.adapter_evidence_json, NEW.adapter_evidence_digest) != 1
    THEN RAISE(ABORT, 'adapter_evidence_digest_mismatch') END;
END;
"""


def _parse_json_text(raw: Any) -> Any:
    if raw is None:
        raise CanonicalError("invalid_json")
    if type(raw) is bytes:
        return strict_json_loads(raw)
    if type(raw) is str:
        return strict_json_loads(raw.encode("utf-8"))
    raise CanonicalError("invalid_json")


def _claimed_ok(claimed: Any) -> str:
    if type(claimed) is not str or len(claimed) != 64:
        raise CanonicalError("invalid_digest")
    if any(ch not in "0123456789abcdef" for ch in claimed):
        raise CanonicalError("invalid_digest")
    return claimed


def _route_digest_eq(route_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(route_json)
        if type(obj) is not dict:
            return 0
        return 1 if route_digest(obj) == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _request_digest_eq(request_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(request_json)
        if type(obj) is not dict:
            return 0
        return 1 if request_digest(obj) == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _result_digest_eq(result_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(result_json)
        if type(obj) is not dict:
            return 0
        return 1 if result_digest(obj) == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _canonical_digest_eq(payload_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(payload_json)
        return 1 if digest(obj) == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _json_digest(payload_json: Any) -> str:
    try:
        return digest(_parse_json_text(payload_json))
    except Exception:
        return ""


def _event_key_eq(
    board: Any,
    tenant: Any,
    orch_id: Any,
    lifecycle_revision: Any,
    cancel_epoch: Any,
    event_kind: Any,
    target_key: Any,
    payload_digest: Any,
    claimed_event_key: Any,
) -> int:
    try:
        actual = event_key(
            {
                "board_instance_id": board,
                "tenant_scope": "" if tenant is None else tenant,
                "orch_id": orch_id,
                "lifecycle_revision": int(lifecycle_revision),
                "cancel_epoch": int(cancel_epoch),
                "event_kind": event_kind,
                "target_key": target_key,
                "payload_digest": _claimed_ok(payload_digest),
            }
        )
        return 1 if actual == _claimed_ok(claimed_event_key) else 0
    except Exception:
        return 0


def install_digest_udfs(conn: sqlite3.Connection) -> None:
    """Register fail-closed digest UDFs on this connection."""
    specs = [
        ("orch_route_digest_eq", 2, _route_digest_eq),
        ("orch_request_digest_eq", 2, _request_digest_eq),
        ("orch_result_digest_eq", 2, _result_digest_eq),
        ("orch_canonical_digest_eq", 2, _canonical_digest_eq),
        ("orch_json_digest", 1, _json_digest),
        ("orch_event_key_eq", 9, _event_key_eq),
    ]
    try:
        for name, n, fn in specs:
            conn.create_function(name, n, fn, deterministic=True)
    except TypeError:
        for name, n, fn in specs:
            conn.create_function(name, n, fn)


def apply_digest_guards(conn: sqlite3.Connection) -> None:
    """Install additive BEFORE INSERT/UPDATE digest guards."""
    install_digest_udfs(conn)
    conn.executescript(DIGEST_GUARD_SQL)
    conn.commit()


def build_route_json_and_digest(
    *,
    origin_kind: str,
    platform: str,
    adapter_instance_id: str,
    account_id: str,
    conversation_id: str,
    thread_id: str = "",
    reply_to_id: str = "",
    session_id: str = "",
    notifier_profile: str = "",
    required_ack_family: str,
    required_ack_strength: str,
    route_revision: int,
) -> tuple[str, str]:
    """Return (route_json, route_digest) with server-side formula authority."""
    route_obj = {
        "schema_version": 4,
        "kind": "orch_route",
        "origin_kind": origin_kind,
        "platform": platform,
        "adapter_instance_id": adapter_instance_id,
        "account_id": account_id,
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "reply_to_id": reply_to_id,
        "session_id": session_id,
        "notifier_profile": notifier_profile,
        "required_ack_family": required_ack_family,
        "required_ack_strength": required_ack_strength,
        "route_revision": route_revision,
    }
    route_d = route_digest(route_obj)
    route_json = json.dumps(route_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return route_json, route_d


def build_result_json_and_digest(
    *,
    request_digest_hex: str,
    plan_digest_hex: str,
    accepted_lane_set: list[Any],
    synthesis: Any,
) -> tuple[str, str]:
    obj = {
        "schema_version": 4,
        "kind": "orch_result",
        "request_digest": request_digest_hex,
        "plan_digest": plan_digest_hex,
        "accepted_lane_set": accepted_lane_set,
        "synthesis": synthesis,
    }
    d = result_digest(obj)
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return raw, d


def build_payload_json_and_digest(payload: Any) -> tuple[str, str]:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return raw, digest(payload)


__all__ = [
    "DIGEST_GUARD_SQL",
    "install_digest_udfs",
    "apply_digest_guards",
    "build_route_json_and_digest",
    "build_result_json_and_digest",
    "build_payload_json_and_digest",
]
