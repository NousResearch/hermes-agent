"""ORCH V4 SQLite digest UDFs + additive trigger guards.

Server recomputes digests inside the DB connection. Caller-supplied digest
columns are assertions only; mismatch aborts the statement.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from hermes_cli.kanban_orch_canonical import (
    CanonicalError,
    digest,
    request_digest,
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
        actual = route_digest(obj)
        return 1 if actual == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _request_digest_eq(request_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(request_json)
        if type(obj) is not dict:
            return 0
        actual = request_digest(obj)
        return 1 if actual == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _canonical_digest_eq(payload_json: Any, claimed: Any) -> int:
    try:
        obj = _parse_json_text(payload_json)
        actual = digest(obj)
        return 1 if actual == _claimed_ok(claimed) else 0
    except Exception:
        return 0


def _json_digest(payload_json: Any) -> str:
    """Return recomputed canonical digest or empty string on failure."""
    try:
        obj = _parse_json_text(payload_json)
        return digest(obj)
    except Exception:
        return ""


def install_digest_udfs(conn: sqlite3.Connection) -> None:
    """Register fail-closed digest UDFs on this connection."""
    # deterministic=True lets SQLite cache safely within a statement.
    kwargs = {"deterministic": True}
    try:
        conn.create_function("orch_route_digest_eq", 2, _route_digest_eq, **kwargs)
        conn.create_function("orch_request_digest_eq", 2, _request_digest_eq, **kwargs)
        conn.create_function("orch_canonical_digest_eq", 2, _canonical_digest_eq, **kwargs)
        conn.create_function("orch_json_digest", 1, _json_digest, **kwargs)
    except TypeError:
        # Older SQLite/Python without deterministic kw.
        conn.create_function("orch_route_digest_eq", 2, _route_digest_eq)
        conn.create_function("orch_request_digest_eq", 2, _request_digest_eq)
        conn.create_function("orch_canonical_digest_eq", 2, _canonical_digest_eq)
        conn.create_function("orch_json_digest", 1, _json_digest)


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


__all__ = [
    "DIGEST_GUARD_SQL",
    "install_digest_udfs",
    "apply_digest_guards",
    "build_route_json_and_digest",
]
