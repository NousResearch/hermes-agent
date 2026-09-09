"""Atomic native notification admission; host lifecycle and board ack stay outside.

Identity is supplied by a trusted host reading a versioned board subscription,
never by formatted text. This module does not establish board-side provenance or
interpret missing receipts as proof of non-delivery. Only future versioned events
may use it. All identity and lease checks share the message/receipt transaction.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from hermes_state_errors import _STATE_DB_GENERATION_KEY, SessionTurnLeaseLostError
from hermes_state_messages import _INSERT_MESSAGE_SQL


_FIELDS = frozenset({"store_path", "store_generation", "board_path", "board_generation",
                     "task_id", "event_id", "origin_session_id", "platform", "thread_id",
                     "subscription_generation"})


def _canonical_identity(db, identity):
    if not isinstance(identity, dict) or set(identity) != _FIELDS:
        raise ValueError("Incomplete or unknown notification identity")
    identity = dict(identity)
    for field in _FIELDS - {"event_id", "thread_id"}:
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"Unknown notification identity: {field}")
    if type(identity["event_id"]) is not int or identity["event_id"] <= 0:
        raise ValueError("Invalid notification event id")
    if identity["thread_id"] is not None and (
            not isinstance(identity["thread_id"], str) or not identity["thread_id"]):
        raise ValueError("Unknown notification thread")
    for field in ("store_path", "board_path"):
        path = Path(identity[field])
        if not path.is_absolute():
            raise ValueError("Notification store paths must be absolute")
        identity[field] = str(path.resolve())
    if identity["store_path"] != str(Path(db.db_path).resolve()):
        raise ValueError("Notification belongs to another profile store")
    if identity["platform"] != "tui":
        raise ValueError("Only TUI admission is supported")
    return identity


def append_notification_once(db, session_id, identity, content, turn_lease_holder):
    identity = _canonical_identity(db, identity)
    if not isinstance(turn_lease_holder, str) or not turn_lease_holder:
        raise SessionTurnLeaseLostError("Notification admission requires known ownership")
    if not isinstance(content, str) or not content:
        raise ValueError("Notification content must be nonempty text")
    key = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    params = db._message_row_params(session_id, "system", {
        "content": content, "display_kind": "notification", "display_metadata": identity,
    }, None, time.time(), keep_reasoning=False)

    def _do(conn):
        generation = conn.execute("SELECT value FROM state_meta WHERE key = ?",
                                  (_STATE_DB_GENERATION_KEY,)).fetchone()
        if generation is None or generation[0] != identity["store_generation"]:
            raise ValueError("Unknown or replaced notification store generation")
        origin = conn.execute("SELECT source, thread_id FROM sessions WHERE id = ?",
                              (identity["origin_session_id"],)).fetchone()
        if origin is None or origin["source"] != identity["platform"] or origin["thread_id"] != identity["thread_id"]:
            raise ValueError("Notification origin does not match persisted session")
        conversation_id = db._session_turn_lease_key_on_conn(conn, session_id)
        origin_conversation = db._session_turn_lease_key_on_conn(conn, identity["origin_session_id"])
        if conversation_id != origin_conversation:
            raise ValueError("Notification belongs to another conversation")
        lease = conn.execute("SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?",
                             (conversation_id,)).fetchone()
        # Unlike ordinary turn flushes, notification admission never renews an
        # expired holder or steals/releases a competing holder's lease.
        if lease is None or lease["holder"] != turn_lease_holder or not float(lease["expires_at"]) > time.time():
            raise SessionTurnLeaseLostError("Notification turn lease missing, competing or stale")
        receipt = conn.execute("SELECT message_id FROM notification_receipts WHERE identity_json = ?",
                               (key,)).fetchone()
        if receipt is not None:
            return int(receipt[0])
        db._check_transcript_write_guards(conn, session_id, None,
            turn_lease_holder=turn_lease_holder, strict_turn_lease=True)
        message_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
        db._bump_session_counters(conn, session_id, 1, 0, unit=True)
        conn.execute("INSERT INTO notification_receipts "
                     "(identity_json, session_id, message_id, admitted_at, continuation_text) "
                     "VALUES (?, ?, ?, ?, ?)", (key, session_id, message_id, time.time(), content))
        return int(message_id)

    return db._execute_write(_do, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
