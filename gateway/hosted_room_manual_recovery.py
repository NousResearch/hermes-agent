"""Explicit manual-continuation preflight, not a takeover or a liveness detector.

The preview never creates approval requests, runs, notices, or authority claims.
Its digest detects stale selection; it is not a credential or fencing proof.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gateway import hosted_room_replicas as replicas
from gateway.hosted_rooms import MAX_ACTOR_ID_CHARS, MAX_ROOM_ID_CHARS, _validate_identifier
from gateway.hosted_room_work_records import encode


def _history_digest(conn, room_id: str) -> str:
    digest = hashlib.sha256()
    for row in conn.execute(
        """SELECT seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at
           FROM hosted_room_replica_events WHERE room_id=? ORDER BY seq""", (room_id,),
    ):
        event = [row["seq"], row["event_id"], row["kind"], json.loads(row["actor_json"]),
                 row["authority_epoch"], json.loads(row["payload_json"]), row["created_at"]]
        digest.update(encode(event).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _member_origins(state: dict[str, Any], target_gateway_id: str) -> tuple[list[str], list[str]]:
    local_candidates, previous_host_members = [], []
    if not isinstance(state["members"], list):
        raise replicas.ReplicaError("Saved membership is unavailable.")
    for member in state["members"]:
        if not isinstance(member, dict) or not isinstance(member.get("member_id"), str):
            raise replicas.ReplicaError("Saved member identity is unavailable.")
        target = member.get("target") or {}
        if not isinstance(target, dict) or target.get("kind") not in (None, "local", "peer"):
            raise replicas.ReplicaError("Saved member target is unsupported.")
        if target.get("kind") == "peer":
            if target.get("installation_id") == target_gateway_id:
                local_candidates.append(member["member_id"])
        else:
            previous_host_members.append(member["member_id"])
    return local_candidates, previous_host_members


def prepare_recovery(
    db_path: Path | str, *, room_id: str, target_gateway_id: str,
) -> dict[str, Any]:
    """Inspect one exact saved state only when an operator asks to continue here.

    All source facts come from one audited transaction. Normal reconnect/status
    polling does not call this function or manufacture a pending confirmation.
    """
    room_id = _validate_identifier(room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS)
    target_gateway_id = _validate_identifier(
        target_gateway_id, label="target_gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    with replicas._replica_transaction(db_path) as conn:
        return prepare_recovery_locked(conn, room_id=room_id, target_gateway_id=target_gateway_id)


def prepare_recovery_locked(conn, *, room_id: str, target_gateway_id: str, evidence=False) -> dict[str, Any]:
    """Use the caller's audited transaction for the preview and later decision."""
    state = replicas._replica_state_locked(conn, room_id)
    blockers = []
    if state["safety_status"] != "passive":
        blockers.append("copy_" + state["safety_status"])
    if state["disbanded_at"] is not None:
        blockers.append("group_disbanded")
    if state["last_seq"] != state["latest_seq"]:
        blockers.append("copy_incomplete")
    if state["authority"]["gateway_id"] == target_gateway_id:
        blockers.append("target_is_previous_host")
    candidates, host_members = _member_origins(state, target_gateway_id)
    if not candidates:
        blockers.append("target_not_a_participant")
    from gateway import hosted_room_recovery_evidence as retained
    rows, enrollment, origins, evidence_blockers = retained.selection_locked(conn, state, target_gateway_id)
    blockers.extend(evidence_blockers)
    records = state["work_records"]
    if records.get("availability") != "available":
        blockers.append("work_records_unavailable")
    if any(scope.get("stop", {}).get("closing") for scope in records.get("scopes", [records])):
        blockers.append("group_closing")
    history_digest = _history_digest(conn, room_id) if state["safety_status"] == "passive" else None
    # Retransmission timestamps are not a new decision. Bind actual content,
    # authority, roster and work records instead of the latest poll time.
    binding = {
        "room_id": room_id, "target_gateway_id": target_gateway_id,
        "source_authority": state["authority"], "members": state["members"],
        "saved_through_seq": state["last_seq"], "advertised_latest_seq": state["latest_seq"],
        "history_digest": history_digest, "inventory_sha256": retained.fingerprint(rows),
        "enrollment": enrollment, "member_origins": origins,
        "lineage": {k: state.get(k) for k in ("replica_version", "lineage_sha256", "authority_history", "source_authority", "lineage_status")},
        "safety_status": state["safety_status"], "disbanded_at": state["disbanded_at"],
        "copy_retired_at": state.get("copy_retired_at"),
    }
    snapshot_id = hashlib.sha256(encode(binding).encode("utf-8")).hexdigest()
    return {
        "object": "hermes.group_recovery.preview", "room_id": room_id, "name": state["name"],
        "snapshot_id": snapshot_id, "source_authority": state["authority"],
        "target_gateway_id": target_gateway_id, "saved_through_seq": state["last_seq"],
        "advertised_latest_seq": state["latest_seq"], "copy_updated_at": state["updated_at"],
        "work_records": records, "blockers": list(dict.fromkeys(blockers)),
        "member_origins": origins,
        **({"evidence": retained.envelope(binding, rows)} if evidence else {}),
        "candidate_member_ids": candidates, "previous_host_member_ids": host_members,
        "reconciliation_required": records.get("availability") != "available" or any(
            scope.get("availability") != "available" or scope.get("tasks") or scope.get("receipts") or scope.get("incompleteness")
            for scope in records.get("scopes", [records])),
        "accepted_tail": "unverified", "execution_authorized": False,
        "requirements": ["operator_confirms_previous_host_fenced", "reconcile_recorded_work",
                         "confirm_saved_recovery_point", "preserve_bot_installation_identity"],
    }
