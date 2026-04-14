#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from orchestrator_contract_v1_mediator import (
    OrchestratorRuntimeContext,
    OrchestratorRuntimeMediator,
    OrchestratorRuntimeRequest,
    OrchestratorRuntimeResult,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = ROOT / "state" / "continuity" / "orchestrator_contract_v1" / "runtime.sqlite"
DEFAULT_RETENTION_MAX_EVENTS = 500
SUPPORTED_COMMANDS = ("plan", "run", "emit-event", "replay-resync", "emit-execution-supervisor-health-event")
EXECUTION_SUPERVISOR_HEALTH_EVENT_TYPES = {
    "execution_supervisor.worker_health.canary_updated",
    "execution_supervisor.worker_health.status_changed",
    "execution_supervisor.probe_execution.scheduled",
    "execution_supervisor.probe_execution.completed",
}
EVENT_SEVERITIES = {"info", "warn", "error"}
REPLAY_RESYNC_REASONS = {"cursor_gap", "consumer_restart", "ledger_mismatch", "manual_recovery"}
ENTITY_REF_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9:_\.-]{3,127}$")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_after_minutes(minutes: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Any) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = _canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _slug(seed: str, n: int = 24) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:n]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
CREATE TABLE IF NOT EXISTS orchestrator_runs (
  idempotency_key TEXT PRIMARY KEY,
  canonical_request_hash TEXT NOT NULL,
  run_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS orchestrator_events (
  event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  emitted_at TEXT NOT NULL,
  available_at TEXT NOT NULL,
  available_seq INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  entity_ref TEXT,
  payload_ref TEXT,
  correlation_id TEXT,
  idempotency_key TEXT,
  parent_event_id TEXT,
  dedupe_key TEXT NOT NULL,
  stream_id TEXT NOT NULL,
  event_json TEXT NOT NULL
)
"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_orchestrator_events_stream_seq ON orchestrator_events(stream_id, event_seq)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_orchestrator_events_dedupe ON orchestrator_events(dedupe_key)")
    con.commit()
    return con


def _retention_bounds(con: sqlite3.Connection, retention_max_events: int) -> tuple[int, int]:
    row = con.execute("SELECT MIN(event_seq) AS min_seq, MAX(event_seq) AS max_seq FROM orchestrator_events").fetchone()
    max_seq = int(row["max_seq"] or 0)
    min_seq_actual = int(row["min_seq"] or 0)
    if max_seq <= 0:
        return 0, 0
    computed_min = max(0, max_seq - max(1, retention_max_events) + 1)
    return max(min_seq_actual, computed_min), max_seq


def _normalize_admission(raw: Any, *, default_mode: str = "normal") -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"mode": default_mode, "decision": "accepted", "reason_codes": []}
    mode = str(raw.get("mode") or default_mode)
    decision = str(raw.get("decision") or "accepted")
    reason_codes = raw.get("reason_codes") if isinstance(raw.get("reason_codes"), list) else []
    normalized = {
        "mode": mode,
        "decision": decision,
        "reason_codes": [str(x) for x in reason_codes if str(x).strip()],
    }
    return normalized


def _coerce_replay_reason(raw: Any, *, errors: list[str]) -> str:
    reason = str(raw or "").strip()
    if not reason:
        errors.append("missing_reason")
        return "manual_recovery"
    if reason not in REPLAY_RESYNC_REASONS:
        errors.append("invalid_reason")
        return "manual_recovery"
    return reason


def _coerce_bounded_int(
    raw: Any,
    *,
    field: str,
    minimum: int,
    maximum: int | None,
    default: int,
    required: bool,
    errors: list[str],
) -> int:
    if raw is None:
        if required:
            errors.append(f"missing_{field}")
        return default
    if isinstance(raw, bool):
        errors.append(f"invalid_{field}")
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"invalid_{field}")
        return default
    if value < minimum or (maximum is not None and value > maximum):
        errors.append(f"invalid_{field}")
        return default
    return value


def _coerce_optional_non_negative_int(raw: Any, *, field: str, errors: list[str]) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        errors.append(f"invalid_{field}")
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"invalid_{field}")
        return None
    if value < 0:
        errors.append(f"invalid_{field}")
        return None
    return value


def _build_plan_packet(request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    snapshot_ref = request.get("snapshot_ref") if isinstance(request.get("snapshot_ref"), dict) else {}
    artifacts = request.get("artifacts") if isinstance(request.get("artifacts"), list) else []
    determinism = {"canonical_json": True, "hash_algorithm": "sha256"}

    reason_codes: list[str] = []
    rejected = False

    # Validate snapshot_ref
    if not isinstance(request.get("snapshot_ref"), dict):
        rejected = True
        reason_codes.append("missing_snapshot_ref")
    else:
        snapshot_id = snapshot_ref.get("snapshot_id")
        if not snapshot_id or not isinstance(snapshot_id, str) or not snapshot_id.strip():
            rejected = True
            reason_codes.append("missing_snapshot_id")
        elif snapshot_id == "snapshot:missing":
            rejected = True
            reason_codes.append("invalid_snapshot_id")
    
    # Validate artifacts
    if not isinstance(request.get("artifacts"), list):
        rejected = True
        reason_codes.append("missing_artifacts")
    elif len(artifacts) == 0:
        rejected = True
        reason_codes.append("empty_artifacts")
    else:
        for idx, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                rejected = True
                reason_codes.append(f"invalid_artifact_{idx+1}_not_dict")
                continue
            kind = artifact.get("artifact_kind")
            version = artifact.get("artifact_version")
            if not kind or not isinstance(kind, str) or not kind.strip():
                rejected = True
                reason_codes.append(f"missing_artifact_kind_{idx+1}")
            if not version or not isinstance(version, str) or not version.strip():
                rejected = True
                reason_codes.append(f"missing_artifact_version_{idx+1}")

    declared_inputs = {
        "snapshot_ref": snapshot_ref,
        "artifacts": artifacts,
        "determinism": determinism,
    }
    declared_inputs_hash = _sha256(declared_inputs)
    plan_seed = f"{snapshot_ref.get('snapshot_id', '')}|{snapshot_ref.get('manifest_id', '')}|{declared_inputs_hash}"
    plan_id = str(request.get("plan_id") or f"plan:{_slug(plan_seed)}")

    expected_outputs: list[dict[str, Any]] = []
    for idx, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        kind = str(artifact.get("artifact_kind") or f"artifact_{idx}")
        version = str(artifact.get("artifact_version") or "v1")
        expected_outputs.append(
            {
                "artifact_kind": kind,
                "artifact_version": version,
                "output_slot": f"{kind}.{idx}",
            }
        )

    plan_hash = _sha256(
        {
            "plan_id": plan_id,
            "declared_inputs_hash": declared_inputs_hash,
            "expected_outputs": expected_outputs,
        }
    )

    admission = _normalize_admission(request.get("admission"), default_mode="normal")
    if rejected:
        admission["decision"] = "rejected"
        admission["reason_codes"] = sorted(set((admission.get("reason_codes") or []) + reason_codes))
    else:
        admission["decision"] = "accepted"
        admission["reason_codes"] = sorted(set(admission.get("reason_codes") or []))

    packet = {
        "schema_version": "clawd.orchestrator.plan.v1",
        "contract_version": "orchestrator_api_contract_v1",
        "request_id": str(request.get("request_id") or f"req:orchestrator:plan:{_slug(plan_id, 16)}"),
        "generated_at": str(request.get("generated_at") or _now_iso()),
        "snapshot_ref": {
            "snapshot_id": str(snapshot_ref.get("snapshot_id") or "snapshot:missing"),
            "manifest_id": str(snapshot_ref.get("manifest_id") or "manifest:missing"),
        },
        "artifacts": artifacts,
        "determinism": determinism,
        "admission": admission,
        "response": {
            "plan_id": plan_id,
            "declared_inputs_hash": declared_inputs_hash,
            "expected_outputs": expected_outputs,
            "plan_hash": plan_hash,
            "expires_in_sec": int(request.get("expires_in_sec") or 1800),
        },
    }
    exit_code = 2 if rejected else 0
    return packet, exit_code


def _compute_run_request_hash(request: dict[str, Any]) -> str:
    stable = {
        "plan_id": request.get("plan_id"),
        "idempotency_key": request.get("idempotency_key"),
        "dry_run": bool(request.get("dry_run", True)),
        "dispatch": request.get("dispatch") if isinstance(request.get("dispatch"), dict) else {"enabled": False, "target_profile": None},
        "evaluation_gate": request.get("evaluation_gate") if isinstance(request.get("evaluation_gate"), dict) else {
            "required": False,
            "canary_mode": "off",
            "attestation_refs": [],
        },
        "output_artifacts": request.get("output_artifacts") if isinstance(request.get("output_artifacts"), list) else [],
    }
    return _sha256(stable)


def _coerce_evaluation_gate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"required": False, "gate_policy_ref": None, "canary_mode": "off", "attestation_refs": []}
    return {
        "required": bool(raw.get("required", False)),
        "gate_policy_ref": raw.get("gate_policy_ref"),
        "canary_mode": str(raw.get("canary_mode") or "off"),
        "attestation_refs": [str(x) for x in (raw.get("attestation_refs") or []) if str(x).strip()],
    }


def _run_outputs(request: dict[str, Any]) -> list[dict[str, Any]]:
    output_artifacts = request.get("output_artifacts") if isinstance(request.get("output_artifacts"), list) else []
    outputs: list[dict[str, Any]] = []
    for idx, artifact in enumerate(output_artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        artifact_kind = str(artifact.get("artifact_kind") or f"artifact_{idx}")
        snapshot_id = str(artifact.get("snapshot_id") or "snapshot:runtime:missing")
        manifest_id = str(artifact.get("manifest_id") or "manifest:runtime:missing")
        output_hash = _sha256({"artifact_kind": artifact_kind, "snapshot_id": snapshot_id, "manifest_id": manifest_id})
        outputs.append(
            {
                "artifact_kind": artifact_kind,
                "output_hash": output_hash,
                "snapshot_id": snapshot_id,
                "manifest_id": manifest_id,
            }
        )
    return outputs


def _load_existing_run(con: sqlite3.Connection, idempotency_key: str) -> dict[str, Any] | None:
    row = con.execute("SELECT payload_json FROM orchestrator_runs WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
    if row is None:
        return None
    return json.loads(str(row["payload_json"]))


def _build_run_packet(request: dict[str, Any], *, con: sqlite3.Connection) -> tuple[dict[str, Any], int]:
    now = _now_iso()
    idempotency_key = str(request.get("idempotency_key") or "")
    canonical_request_hash = str(request.get("canonical_request_hash") or _compute_run_request_hash(request))
    dry_run = bool(request.get("dry_run", True))
    dispatch = request.get("dispatch") if isinstance(request.get("dispatch"), dict) else {"enabled": False, "target_profile": None}
    dispatch = {"enabled": bool(dispatch.get("enabled", False)), "target_profile": dispatch.get("target_profile")}
    evaluation_gate = _coerce_evaluation_gate(request.get("evaluation_gate"))
    admission = _normalize_admission(request.get("admission"), default_mode="normal")

    reason_codes: list[str] = []
    rejected = False

    if not idempotency_key:
        rejected = True
        reason_codes.append("missing_idempotency_key")
    if not canonical_request_hash.startswith("sha256:"):
        rejected = True
        reason_codes.append("invalid_canonical_request_hash")

    if admission.get("mode") == "read_only" and not dry_run:
        rejected = True
        reason_codes.append("read_only_side_effect_run_rejected")

    if evaluation_gate.get("required"):
        if not evaluation_gate.get("attestation_refs"):
            rejected = True
            reason_codes.append("missing_evaluation_attestation_refs")
        if not evaluation_gate.get("gate_policy_ref"):
            rejected = True
            reason_codes.append("missing_gate_policy_ref")

    existing = _load_existing_run(con, idempotency_key) if idempotency_key else None
    duplicate_suppressed = False
    duplicate_of_run_id: str | None = None
    status = "accepted"
    run_id = str(request.get("run_id") or f"run:{_slug(f'{request.get('plan_id', '')}|{idempotency_key}|{canonical_request_hash}')}")

    outputs = _run_outputs(request)

    if existing:
        existing_hash = str(existing.get("canonical_request_hash") or "")
        existing_run_id = str((existing.get("response") or {}).get("run_id") or "")
        if existing_hash == canonical_request_hash:
            duplicate_suppressed = True
            duplicate_of_run_id = existing_run_id or None
            status = "duplicate_suppressed"
            run_id = existing_run_id or run_id
            outputs = (existing.get("response") or {}).get("outputs") or outputs
        else:
            rejected = True
            reason_codes.append("idempotency_conflict")

    if rejected:
        admission["decision"] = "rejected"
        admission["reason_codes"] = sorted(set((admission.get("reason_codes") or []) + reason_codes))
        status = "rejected"
        duplicate_suppressed = False
        duplicate_of_run_id = None
        outputs = []
    else:
        if admission.get("decision") not in {"accepted", "deferred"}:
            admission["decision"] = "accepted"
        admission["reason_codes"] = sorted(set(admission.get("reason_codes") or []))

    packet = {
        "schema_version": "clawd.orchestrator.run.v1",
        "contract_version": "orchestrator_api_contract_v1",
        "request_id": str(request.get("request_id") or f"req:orchestrator:run:{_slug(idempotency_key or now, 16)}"),
        "generated_at": str(request.get("generated_at") or now),
        "plan_id": str(request.get("plan_id") or "plan:missing"),
        "idempotency_key": idempotency_key,
        "canonical_request_hash": canonical_request_hash,
        "dry_run": dry_run,
        "dispatch": dispatch,
        "evaluation_gate": evaluation_gate,
        "admission": admission,
        "response": {
            "run_id": run_id,
            "status": status,
            "duplicate_suppressed": duplicate_suppressed,
            "duplicate_of_run_id": duplicate_of_run_id,
            "outputs": outputs,
        },
    }

    if not rejected and not existing:
        con.execute(
            """
INSERT INTO orchestrator_runs(idempotency_key, canonical_request_hash, run_id, payload_json, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
""",
            (idempotency_key, canonical_request_hash, run_id, json.dumps(packet, ensure_ascii=False, sort_keys=True), now, now),
        )
        con.commit()

    exit_code = 0 if not rejected else 2
    return packet, exit_code


def _event_packet_from_row(
    row: sqlite3.Row,
    *,
    retention_min: int,
    retention_max: int,
    since_seq: int,
    gap_detected: bool,
    resync_hint: str | None,
) -> dict[str, Any]:
    event_payload = json.loads(str(row["event_json"]))
    packet = {
        "schema_version": "clawd.orchestrator.event_stream.v1",
        "contract_version": "orchestrator_api_contract_v1",
        "stream_id": str(row["stream_id"]),
        "emitted_at": str(row["emitted_at"]),
        "event": event_payload,
        "cursor": {
            "since_seq": int(max(0, since_seq)),
            "next_after_seq": int(row["event_seq"]),
            "retention": {"min_seq": int(max(0, retention_min)), "max_seq": int(max(0, retention_max))},
            "gap_detected": bool(gap_detected),
            "resync_hint": resync_hint,
        },
    }
    return packet


def _build_event_packet(request: dict[str, Any], *, con: sqlite3.Connection, retention_max_events: int) -> dict[str, Any]:
    stream_id = str(request.get("stream_id") or "stream:default")
    event = request.get("event") if isinstance(request.get("event"), dict) else {}

    requested_since_seq = int(request.get("since_seq") or 0)
    event_id = str(event.get("event_id") or f"event:{_slug(_canonical_json(event) + '|' + stream_id)}")

    existing = con.execute(
        "SELECT * FROM orchestrator_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()

    if existing is None:
        now = _now_iso()
        available_at = str(event.get("available_at") or now)
        available_seq_raw = event.get("available_seq")
        if available_seq_raw is None:
            row = con.execute("SELECT MAX(event_seq) AS max_seq FROM orchestrator_events").fetchone()
            available_seq = int(row["max_seq"] or 0)
        else:
            available_seq = int(available_seq_raw)

        event_payload = {
            "event_id": event_id,
            "event_seq": -1,
            "available_at": available_at,
            "available_seq": int(max(0, available_seq)),
            "type": str(event.get("type") or "artifacts.run.accepted"),
            "severity": str(event.get("severity") or "info"),
            "entity_ref": event.get("entity_ref"),
            "payload_ref": event.get("payload_ref"),
            "correlation_id": event.get("correlation_id"),
            "idempotency_key": event.get("idempotency_key"),
            "parent_event_id": event.get("parent_event_id"),
            "dedupe_key": str(event.get("dedupe_key") or f"dedupe:{stream_id}:{event_id}"),
        }
        con.execute(
            """
INSERT INTO orchestrator_events(
  event_id, emitted_at, available_at, available_seq, event_type, severity,
  entity_ref, payload_ref, correlation_id, idempotency_key, parent_event_id,
  dedupe_key, stream_id, event_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                event_id,
                now,
                available_at,
                int(max(0, available_seq)),
                event_payload["type"],
                event_payload["severity"],
                event_payload.get("entity_ref"),
                event_payload.get("payload_ref"),
                event_payload.get("correlation_id"),
                event_payload.get("idempotency_key"),
                event_payload.get("parent_event_id"),
                event_payload["dedupe_key"],
                stream_id,
                json.dumps(event_payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        seq = int(con.execute("SELECT event_seq FROM orchestrator_events WHERE event_id = ?", (event_id,)).fetchone()["event_seq"])
        event_payload["event_seq"] = seq
        if event_payload["available_seq"] > seq:
            event_payload["available_seq"] = seq
        con.execute(
            "UPDATE orchestrator_events SET available_seq = ?, event_json = ? WHERE event_id = ?",
            (event_payload["available_seq"], json.dumps(event_payload, ensure_ascii=False, sort_keys=True), event_id),
        )
        con.commit()
        row = con.execute("SELECT * FROM orchestrator_events WHERE event_id = ?", (event_id,)).fetchone()
    else:
        row = existing

    retention_min, retention_max = _retention_bounds(con, retention_max_events)
    event_seq = int(row["event_seq"])
    gap_detected = requested_since_seq < max(0, event_seq - 1)
    resync_hint = "replay_resync_required" if gap_detected else None
    return _event_packet_from_row(
        row,
        retention_min=retention_min,
        retention_max=retention_max,
        since_seq=requested_since_seq,
        gap_detected=gap_detected,
        resync_hint=resync_hint,
    )


def _build_execution_supervisor_health_event_packet(
    request: dict[str, Any],
    *,
    con: sqlite3.Connection,
    retention_max_events: int,
) -> dict[str, Any]:
    """Build orchestrator event packet for execution supervisor health events."""
    event_type = str(request.get("event_type") or "").strip()
    severity = str(request.get("severity") or "").strip()
    entity_ref = str(request.get("entity_ref") or "").strip()
    dedupe_key = str(request.get("dedupe_key") or "").strip()
    payload_ref_raw = request.get("payload_ref")
    payload_ref: str | None

    errors: list[str] = []
    if not event_type:
        errors.append("missing_event_type")
    elif event_type not in EXECUTION_SUPERVISOR_HEALTH_EVENT_TYPES:
        errors.append(f"invalid_event_type:{event_type}")

    if not severity:
        errors.append("missing_severity")
    elif severity not in EVENT_SEVERITIES:
        errors.append(f"invalid_severity:{severity}")

    if not entity_ref:
        errors.append("missing_entity_ref")
    elif not ENTITY_REF_PATTERN.fullmatch(entity_ref):
        errors.append(f"invalid_entity_ref:{entity_ref}")

    if not dedupe_key:
        errors.append("missing_dedupe_key")
    elif len(dedupe_key) < 8:
        errors.append("dedupe_key_too_short")
    elif len(dedupe_key) > 256:
        errors.append("dedupe_key_too_long")

    if payload_ref_raw is None:
        payload_ref = None
    elif isinstance(payload_ref_raw, str):
        payload_ref = payload_ref_raw.strip() or None
    else:
        payload_ref = None
        errors.append("invalid_payload_ref")

    if errors:
        raise ValueError(";".join(errors))

    event_payload = {
        "stream_id": "stream:execution_supervisor:health",
        "event": {
            "event_id": f"event:execution_supervisor:{_slug(dedupe_key)}",
            "type": event_type,
            "severity": severity,
            "entity_ref": entity_ref,
            "payload_ref": payload_ref,
            "dedupe_key": dedupe_key,
        },
        "since_seq": 0,
    }
    
    # Use existing event emission logic
    return _build_event_packet(
        event_payload,
        con=con,
        retention_max_events=retention_max_events,
    )


def _build_replay_packet(
    request_packet: dict[str, Any],
    *,
    con: sqlite3.Connection,
    retention_max_events: int,
) -> tuple[dict[str, Any], int]:
    now = _now_iso()
    request = request_packet.get("request") if isinstance(request_packet.get("request"), dict) else {}
    admission = _normalize_admission(request_packet.get("admission"), default_mode="normal")

    errors: list[str] = []
    reason = _coerce_replay_reason(request.get("reason"), errors=errors)
    last_applied = _coerce_bounded_int(
        request.get("last_applied_event_seq"),
        field="last_applied_event_seq",
        minimum=0,
        maximum=None,
        default=0,
        required=True,
        errors=errors,
    )
    from_event_seq = _coerce_optional_non_negative_int(request.get("from_event_seq"), field="from_event_seq", errors=errors)
    to_event_seq = _coerce_optional_non_negative_int(request.get("to_event_seq"), field="to_event_seq", errors=errors)
    max_events = _coerce_bounded_int(
        request.get("max_events"),
        field="max_events",
        minimum=1,
        maximum=100000,
        default=max(1, retention_max_events),
        required=False,
        errors=errors,
    )

    if from_event_seq is not None and to_event_seq is not None and from_event_seq > to_event_seq:
        errors.append("invalid_replay_window")

    rejected = len(errors) > 0

    if rejected:
        admission["decision"] = "rejected"
        admission["reason_codes"] = sorted(set((admission.get("reason_codes") or []) + errors))
    else:
        if admission.get("decision") not in {"accepted", "deferred"}:
            admission["decision"] = "accepted"
        admission["reason_codes"] = sorted(set(admission.get("reason_codes") or []))

    retention_min, retention_max = _retention_bounds(con, retention_max_events)
    start_seq = from_event_seq if from_event_seq is not None else max(0, last_applied + 1)
    requested_end_seq = to_event_seq if to_event_seq is not None else retention_max
    requested_end_seq = max(start_seq, requested_end_seq)

    if rejected:
        replay_start = int(max(0, last_applied + 1))
        replay_end = replay_start
        available_count = 0
        status = "blocked"
        actions: list[str] = ["operator_review_required"]
        notes = "Replay/resync request rejected due to invalid request fields."
    else:
        status = "ready"
        actions = []
        notes = ""

        if retention_max == 0:
            replay_start = start_seq
            replay_end = max(start_seq, requested_end_seq)
            available_count = 0
            status = "ready"
            notes = "No retained events available yet; caller may continue streaming from next cursor."
        elif start_seq < retention_min:
            replay_start = retention_min
            replay_end = min(requested_end_seq, retention_max)
            available_count = max(0, replay_end - replay_start + 1)
            status = "snapshot_reseed_required"
            actions = ["resolve_snapshot", "rebuild_plan", "operator_review_required"]
            notes = "Retention cliff detected; snapshot reseed required before mutating resume state."
        else:
            replay_start = start_seq
            replay_end = min(requested_end_seq, retention_max)
            available_count = max(0, replay_end - replay_start + 1)
            if available_count > max_events:
                replay_end = replay_start + max_events - 1
                available_count = max_events
                status = "partial"
                actions = ["replay_events"]
                notes = "Replay window truncated to request.max_events ceiling."
            else:
                status = "ready"
                actions = ["replay_events"] if available_count > 0 else []
                notes = "Replay window available within retention bounds."

        if reason == "ledger_mismatch" and status in {"ready", "partial"}:
            if "rerun_idempotent" not in actions:
                actions.append("rerun_idempotent")

        if reason == "manual_recovery" and status == "snapshot_reseed_required":
            if "operator_review_required" not in actions:
                actions.append("operator_review_required")

    next_since_seq = replay_end if available_count > 0 else last_applied
    token_seed = {
        "reason": reason,
        "last_applied_event_seq": last_applied,
        "start_seq": replay_start,
        "end_seq": replay_end,
        "status": status,
    }

    packet = {
        "schema_version": "clawd.orchestrator.replay_resync.v1",
        "contract_version": "orchestrator_api_contract_v1",
        "request_id": str(request_packet.get("request_id") or f"req:orchestrator:replay:{_slug(_canonical_json(token_seed), 16)}"),
        "generated_at": str(request_packet.get("generated_at") or now),
        "request": {
            "reason": reason,
            "from_event_seq": from_event_seq,
            "to_event_seq": to_event_seq,
            "last_known_snapshot_id": request.get("last_known_snapshot_id"),
            "last_applied_event_seq": last_applied,
            "max_events": max_events,
        },
        "admission": admission,
        "response": {
            "status": status,
            "replay_window": {
                "start_seq": int(max(0, replay_start)),
                "end_seq": int(max(0, replay_end)),
                "available_count": int(max(0, available_count)),
            },
            "resume_cursor": {
                "next_since_seq": int(max(0, next_since_seq)),
                "resync_token": f"resync_{_slug(_canonical_json(token_seed), 20)}",
                "expires_at": _iso_after_minutes(30),
            },
            "actions": actions,
            "notes": notes,
        },
    }
    exit_code = 2 if rejected else 0
    return packet, exit_code


def _handle_plan(context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult:
    packet, exit_code = _build_plan_packet(context.envelope.request)
    return OrchestratorRuntimeResult(packet=packet, exit_code=exit_code)


def _require_connection(context: OrchestratorRuntimeContext) -> sqlite3.Connection:
    if context.connection is None:
        raise RuntimeError("runtime_connection_missing")
    return context.connection


def _handle_run(context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult:
    packet, exit_code = _build_run_packet(context.envelope.request, con=_require_connection(context))
    return OrchestratorRuntimeResult(packet=packet, exit_code=exit_code)


def _handle_emit_event(context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult:
    packet = _build_event_packet(
        context.envelope.request,
        con=_require_connection(context),
        retention_max_events=context.envelope.retention_max_events,
    )
    return OrchestratorRuntimeResult(packet=packet, exit_code=0)


def _handle_replay_resync(context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult:
    packet, exit_code = _build_replay_packet(
        context.envelope.request,
        con=_require_connection(context),
        retention_max_events=context.envelope.retention_max_events,
    )
    return OrchestratorRuntimeResult(packet=packet, exit_code=exit_code)


def _handle_emit_execution_supervisor_health_event(context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult:
    """Handle execution supervisor health event emission."""
    try:
        packet = _build_execution_supervisor_health_event_packet(
            context.envelope.request,
            con=_require_connection(context),
            retention_max_events=context.envelope.retention_max_events,
        )
        return OrchestratorRuntimeResult(packet=packet, exit_code=0)
    except ValueError as exc:
        # Fail-closed on validation errors
        error_msg = str(exc)
        packet = {
            "schema_version": "clawd.orchestrator.event_stream.v1",
            "contract_version": "orchestrator_api_contract_v1",
            "error": {
                "code": "validation_failed",
                "message": f"Execution supervisor health event validation failed: {error_msg}",
                "retryable": False,
                "details": {"validation_error": error_msg},
            },
        }
        return OrchestratorRuntimeResult(packet=packet, exit_code=2)


def _build_runtime_mediator() -> OrchestratorRuntimeMediator:
    mediator = OrchestratorRuntimeMediator(connect_db=_connect_db)
    mediator.register("plan", _handle_plan, requires_db=False)
    mediator.register("run", _handle_run, requires_db=True)
    mediator.register("emit-event", _handle_emit_event, requires_db=True)
    mediator.register("replay-resync", _handle_replay_resync, requires_db=True)
    mediator.register("emit-execution-supervisor-health-event", _handle_emit_execution_supervisor_health_event, requires_db=True)
    return mediator


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bounded EX-06 orchestrator contract v1 integration surface")
    ap.add_argument("command", choices=list(SUPPORTED_COMMANDS), help="Surface to execute")
    ap.add_argument("--request", required=True, help="Path to JSON request payload")
    ap.add_argument("--state-db", default=str(DEFAULT_DB_PATH), help="SQLite state path for idempotency/event ordering")
    ap.add_argument(
        "--retention-max-events",
        type=int,
        default=DEFAULT_RETENTION_MAX_EVENTS,
        help=f"Retention ceiling for replay windows (default: {DEFAULT_RETENTION_MAX_EVENTS})",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    envelope = OrchestratorRuntimeRequest(
        command=args.command,
        request=_read_json(Path(args.request)),
        state_db=Path(args.state_db),
        retention_max_events=max(1, int(args.retention_max_events)),
    )
    result = _build_runtime_mediator().dispatch(envelope)
    _print(result.packet)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
