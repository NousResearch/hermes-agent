#!/usr/bin/env python3
"""Successor-safe handover proof helpers (Slice 10).

Deterministic producer/consumer helpers that reuse existing continuity surfaces.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from typing import Any, Mapping

PROOF_STATUS_ENUM: tuple[str, ...] = ("ACTIVE", "EXPIRED", "INVALIDATED", "SUPERSEDED", "REFUSED")
PROOF_STATE_ENUM: tuple[str, ...] = (
    "PROOF_MISSING",
    "PROOF_REFUSED",
    "PROOF_INVALID",
    "PROOF_EXPIRED",
    "PROOF_VALID_BLOCKED",
    "PROOF_VALID_PASS",
)

PROOF_BLOCKER_CODES: tuple[str, ...] = (
    "BLK_PROOF_MISSING",
    "BLK_PROOF_SCHEMA_INVALID",
    "BLK_PROOF_EXPIRED",
    "BLK_PROOF_INVALIDATED",
    "BLK_PROOF_REFUSED",
    "BLK_PROOF_GENERATION_MISMATCH",
    "BLK_PROOF_READ_POINTER_MISMATCH",
    "BLK_PROOF_VERIFY_GATE_NOT_PASS",
    "BLK_PROOF_MUTATION_UNSAFE",
    "BLK_PROOF_QUEUE_AUTHORITY_AMBIGUOUS",
    "BLK_PROOF_CONNECTOR_STALE",
)

PROOF_INVALIDATION_REASON_CODES: tuple[str, ...] = (
    "INV_POST_PROOF_MUTATION",
    "INV_CONTINUITY_POINTER_CHANGED",
    "INV_BUILD_GENERATION_CHANGED",
    "INV_COHERENCE_EXPIRED",
    "INV_CONNECTOR_FRESHNESS_DROPPED",
    "INV_QUEUE_AUTHORITY_AMBIGUOUS",
    "INV_MANUAL_SUPERSEDED",
)

RESET_BLOCKER_MAP: dict[str, str] = {
    "continuity_fresh": "BLK_CONTINUITY_STALE",
    "proof_valid": "BLK_PROOF_STALE_OR_INVALID",
    "mutation_safe": "BLK_MUTATION_IN_FLIGHT_UNSAFE",
    "coherence_valid": "BLK_COHERENCE_INVALID",
    "queue_authority_clear": "BLK_QUEUE_AUTHORITY_AMBIGUOUS",
    "connector_fresh": "BLK_CONNECTOR_SNAPSHOT_STALE",
}


REQUIRED_PROOF_KEYS = {
    "object_type",
    "proof_id",
    "proof_generation_id",
    "produced_at",
    "expires_at",
    "status",
    "source_refs",
    "coherence_tuple_ref",
    "safety_inputs",
    "verdicts",
    "invalidation",
}


_DEFAULT_PROOF_PATH = "state/continuity/latest/successor_safe_handover_proof.json"


def _now_dt(now: str | None = None) -> dt.datetime:
    if isinstance(now, str) and now.strip():
        parsed = _parse_iso(now)
        if parsed is not None:
            return parsed
    return dt.datetime.now(dt.timezone.utc)


def _now_iso(now: str | None = None) -> str:
    return _now_dt(now).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(raw: Any) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _non_empty(value: Any) -> str | None:
    txt = str(value or "").strip()
    return txt if txt else None


def _pointer_generation_id(pointer: Mapping[str, Any] | None) -> str | None:
    if not isinstance(pointer, Mapping):
        return None
    contract = pointer.get("continuity_read_contract") if isinstance(pointer.get("continuity_read_contract"), Mapping) else {}
    return (
        _non_empty(pointer.get("build_generation_id"))
        or _non_empty(pointer.get("coherence_build_generation_id"))
        or _non_empty(contract.get("coherence_build_generation_id"))
    )


def _pointer_current_sha256(pointer: Mapping[str, Any] | None) -> str | None:
    if not isinstance(pointer, Mapping):
        return None
    contract = pointer.get("continuity_read_contract") if isinstance(pointer.get("continuity_read_contract"), Mapping) else {}
    source_current = pointer.get("source_current") if isinstance(pointer.get("source_current"), Mapping) else {}
    return (
        _non_empty(pointer.get("current_sha256"))
        or _non_empty(contract.get("continuity_current_sha256"))
        or _non_empty(source_current.get("sha256"))
    )


def _connector_reason_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_connector_freshness(current: Mapping[str, Any] | None) -> tuple[str | None, list[str]]:
    if not isinstance(current, Mapping):
        return (None, [])

    coherence = current.get("coherence") if isinstance(current.get("coherence"), Mapping) else {}
    direct = coherence.get("connector_freshness") if isinstance(coherence.get("connector_freshness"), Mapping) else {}
    direct_status = _non_empty(direct.get("status"))
    direct_stale = _connector_reason_list(direct.get("stale_connectors"))
    if direct_status:
        return (direct_status, direct_stale)

    connector_blocking = _connector_reason_list(coherence.get("connector_blocking_reasons"))
    if connector_blocking:
        return ("STALE", connector_blocking)

    connectors_obj = coherence.get("connectors") if isinstance(coherence.get("connectors"), Mapping) else {}
    connectors_blocking = _connector_reason_list(connectors_obj.get("blocking_reasons"))
    if connectors_blocking:
        return ("STALE", connectors_blocking)

    connector_warning = _connector_reason_list(coherence.get("connector_warning_reasons"))
    connectors_warning = _connector_reason_list(connectors_obj.get("warning_reasons"))
    if connector_warning or connectors_warning:
        return ("FRESH", [])

    if "connector_blocking_reasons" in coherence or "connectors" in coherence:
        return ("FRESH", [])

    return (None, [])


def _status_is_pass(value: Any) -> bool:
    status = str(value or "").strip().upper()
    return status in {"PASS", "OK", "READY", "SUCCESS"}


def _unique(rows: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        txt = str(row or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


def build_successor_safe_handover_proof(
    *,
    root: str | pathlib.Path,
    now: str | None = None,
    ttl_seconds: int = 10 * 60,
    trigger: str = "manual_refresh",
) -> dict[str, Any]:
    root_path = pathlib.Path(root).resolve()

    current_path = root_path / "state" / "continuity" / "current.json"
    handover_path = root_path / "state" / "handover" / "latest.json"
    pointer_path = root_path / "state" / "continuity" / "latest" / "continuity_read_pointer.json"
    verify_path = root_path / "state" / "continuity" / "latest" / "verify_last.json"
    latest_pointer_path = root_path / "state" / "continuity" / "latest" / "latest_pointer.json"

    current = _read_json(current_path)
    handover = _read_json(handover_path)
    pointer = _read_json(pointer_path)
    verify = _read_json(verify_path)
    latest_pointer = _read_json(latest_pointer_path)

    blockers: list[str] = []

    if current is None or handover is None or pointer is None:
        blockers.append("BLK_PROOF_MISSING")

    now_dt = _now_dt(now)
    produced_at = now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    current_sha = _sha256_file(current_path)
    handover_sha = _sha256_file(handover_path)
    pointer_sha = _sha256_file(pointer_path)

    current_generation = _non_empty((((current or {}).get("coherence") or {}).get("build_generation_id")))
    pointer_generation = _pointer_generation_id(pointer)

    if current_sha and pointer is not None:
        pointer_current_sha = _pointer_current_sha256(pointer)
        if not pointer_current_sha or pointer_current_sha != current_sha:
            blockers.append("BLK_PROOF_READ_POINTER_MISMATCH")

    if current_generation and pointer_generation and current_generation != pointer_generation:
        blockers.append("BLK_PROOF_GENERATION_MISMATCH")

    verify_status = _non_empty((verify or {}).get("status"))
    if verify is None or not _status_is_pass(verify_status):
        blockers.append("BLK_PROOF_VERIFY_GATE_NOT_PASS")

    mutation_gate_status = _non_empty((((current or {}).get("mutation_gate") or {}).get("status")))
    in_flight = bool((((current or {}).get("in_flight") or {}).get("value") is True))
    if mutation_gate_status != "allowed" or in_flight:
        blockers.append("BLK_PROOF_MUTATION_UNSAFE")

    queue_ambiguity = handover.get("queue_authority") if isinstance(handover, Mapping) else None
    queue_conflicts = []
    if isinstance(queue_ambiguity, Mapping):
        queue_conflicts = queue_ambiguity.get("unresolved_conflicts") or []
    if queue_conflicts:
        blockers.append("BLK_PROOF_QUEUE_AUTHORITY_AMBIGUOUS")

    connector_status, stale_connectors = _resolve_connector_freshness(current)
    if connector_status and connector_status.upper() not in {"FRESH", "PASS", "OK"}:
        blockers.append("BLK_PROOF_CONNECTOR_STALE")

    coherence_valid_until_txt = _non_empty((((current or {}).get("coherence") or {}).get("valid_until")))
    coherence_valid_until_dt = _parse_iso(coherence_valid_until_txt)

    expires_at_dt = now_dt + dt.timedelta(seconds=max(1, int(ttl_seconds)))
    if coherence_valid_until_dt is not None and coherence_valid_until_dt < expires_at_dt:
        expires_at_dt = coherence_valid_until_dt
    expires_at = expires_at_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    blockers = _unique(blockers)
    status = "ACTIVE" if not blockers else "REFUSED"
    reset_safety = "PASS" if not blockers else "FAIL_BLOCKED"
    resume_safety = "PASS" if not blockers else "FAIL_BLOCKED"

    proof_id_seed = f"{produced_at}|{current_sha or 'missing'}|{pointer_generation or 'unknown'}"
    proof_id = f"proof_{hashlib.sha256(proof_id_seed.encode('utf-8')).hexdigest()[:16]}"

    proof = {
        "object_type": "clawd.successor_safe_handover_proof.v1",
        "proof_id": proof_id,
        "proof_generation_id": pointer_generation or "proofgen_unknown",
        "produced_at": produced_at,
        "expires_at": expires_at,
        "status": status,
        "producer": {
            "component": "continuity.successor_safe_handover_proof",
            "version": "v1",
            "trigger": str(trigger or "manual_refresh"),
        },
        "source_refs": {
            "continuity_current": {
                "path": "state/continuity/current.json",
                "sha256": current_sha,
                "generated_at": _non_empty((current or {}).get("generated_at")),
                "build_generation_id": current_generation,
            },
            "handover_latest": {
                "path": "state/handover/latest.json",
                "sha256": handover_sha,
                "generated_at": _non_empty((handover or {}).get("generated_at")),
            },
            "checkpoint": {
                "checkpoint_id": _non_empty((latest_pointer or {}).get("checkpoint_id")),
                "sha256": _non_empty((latest_pointer or {}).get("json_sha256")),
            },
            "continuity_read_pointer": {
                "path": "state/continuity/latest/continuity_read_pointer.json",
                "sha256": pointer_sha,
                "current_sha256": _pointer_current_sha256(pointer),
                "continuity_current_sha256": _pointer_current_sha256(pointer),
                "generated_at": _non_empty((pointer or {}).get("generated_at")),
                "build_generation_id": pointer_generation,
                "coherence_build_generation_id": pointer_generation,
            },
        },
        "coherence_tuple_ref": {
            "world_anchor_id": _non_empty((((current or {}).get("coherence") or {}).get("world_anchor_id"))),
            "policy_epoch_id": _non_empty((((current or {}).get("coherence") or {}).get("policy_epoch_id"))),
            "connector_snapshot_id": _non_empty((((current or {}).get("coherence") or {}).get("connector_snapshot_id"))),
            "valid_until": coherence_valid_until_txt,
            "build_generation_id": current_generation,
        },
        "safety_inputs": {
            "verify_gate": {
                "status": verify_status or "UNKNOWN",
                "report_path": "state/continuity/latest/verify_last.json",
                "generated_at": _non_empty((verify or {}).get("timestamp")),
            },
            "mutation_safety": {
                "unsafe_in_flight_mutation": bool(in_flight),
                "mutation_gate_posture": _non_empty((((current or {}).get("mutation_gate") or {}).get("posture")))
                or "unknown",
                "action_token_epoch": _non_empty((((current or {}).get("mutation_gate") or {}).get("action_token_epoch"))),
            },
            "queue_authority": {
                "status": "AMBIGUOUS" if queue_conflicts else "CLEAR",
                "unresolved_conflicts": list(queue_conflicts or []),
            },
            "connector_freshness": {
                "status": connector_status or "UNKNOWN",
                "stale_connectors": stale_connectors,
            },
        },
        "verdicts": {
            "reset_safety": reset_safety,
            "resume_safety": resume_safety,
            "blockers": blockers,
            "required_actions": [
                "refresh_successor_safe_proof" if blockers else "none"
            ],
        },
        "invalidation": {
            "invalidated_at": None,
            "reason_codes": [],
            "superseded_by_proof_id": None,
        },
    }

    return proof


def evaluate_proof_consumability(
    proof: Mapping[str, Any] | None,
    *,
    mode: str,
    now: str | None = None,
    expected_generation_id: str | None = None,
    expected_pointer_sha256: str | None = None,
    require_no_post_proof_invalidation: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    proof_state = "PROOF_VALID_BLOCKED"

    if proof is None:
        blockers.append("BLK_PROOF_MISSING")
        proof_state = "PROOF_MISSING"
    elif not isinstance(proof, Mapping):
        blockers.append("BLK_PROOF_SCHEMA_INVALID")
        proof_state = "PROOF_INVALID"
    else:
        missing = sorted(REQUIRED_PROOF_KEYS - set(proof.keys()))
        if missing or str(proof.get("object_type") or "") != "clawd.successor_safe_handover_proof.v1":
            blockers.append("BLK_PROOF_SCHEMA_INVALID")
            proof_state = "PROOF_INVALID"

    if proof is None or not isinstance(proof, Mapping):
        blockers = _unique(blockers)
        return {
            "ok": False,
            "proof_state": proof_state,
            "blockers": blockers,
        }

    status = str(proof.get("status") or "").strip().upper()
    verdicts = proof.get("verdicts") if isinstance(proof.get("verdicts"), Mapping) else {}
    producer_blockers = [str(item).strip() for item in (verdicts.get("blockers") or []) if str(item).strip()]

    if status == "REFUSED":
        blockers.append("BLK_PROOF_REFUSED")
        blockers.extend(producer_blockers)
        proof_state = "PROOF_REFUSED"
    elif status == "INVALIDATED":
        blockers.append("BLK_PROOF_INVALIDATED")
        blockers.extend(producer_blockers)
        proof_state = "PROOF_INVALID"
    elif status == "EXPIRED":
        blockers.append("BLK_PROOF_EXPIRED")
        proof_state = "PROOF_EXPIRED"
    elif status != "ACTIVE":
        blockers.append("BLK_PROOF_SCHEMA_INVALID")
        proof_state = "PROOF_INVALID"

    expires_at = _parse_iso(proof.get("expires_at"))
    now_dt = _now_dt(now)
    if expires_at is None or now_dt > expires_at:
        blockers.append("BLK_PROOF_EXPIRED")
        proof_state = "PROOF_EXPIRED"

    source_refs = proof.get("source_refs") if isinstance(proof.get("source_refs"), Mapping) else {}
    pointer_ref = source_refs.get("continuity_read_pointer") if isinstance(source_refs.get("continuity_read_pointer"), Mapping) else {}

    if expected_generation_id:
        pointer_generation = _non_empty(pointer_ref.get("build_generation_id")) or _non_empty(
            pointer_ref.get("coherence_build_generation_id")
        )
        if pointer_generation != expected_generation_id:
            blockers.append("BLK_PROOF_GENERATION_MISMATCH")

    if expected_pointer_sha256:
        pointer_sha_candidates = {
            _non_empty(pointer_ref.get("sha256")),
            _non_empty(pointer_ref.get("current_sha256")),
            _non_empty(pointer_ref.get("continuity_current_sha256")),
        }
        pointer_sha_candidates.discard(None)
        if expected_pointer_sha256 not in pointer_sha_candidates:
            blockers.append("BLK_PROOF_READ_POINTER_MISMATCH")

    safety_inputs = proof.get("safety_inputs") if isinstance(proof.get("safety_inputs"), Mapping) else {}
    verify_gate = safety_inputs.get("verify_gate") if isinstance(safety_inputs.get("verify_gate"), Mapping) else {}
    if not _status_is_pass(verify_gate.get("status")):
        blockers.append("BLK_PROOF_VERIFY_GATE_NOT_PASS")

    mutation_safety = safety_inputs.get("mutation_safety") if isinstance(safety_inputs.get("mutation_safety"), Mapping) else {}
    mutation_gate_posture = str(mutation_safety.get("mutation_gate_posture") or "").strip().lower()
    if bool(mutation_safety.get("unsafe_in_flight_mutation")) or mutation_gate_posture not in {"", "open", "allowed"}:
        blockers.append("BLK_PROOF_MUTATION_UNSAFE")

    queue_authority = safety_inputs.get("queue_authority") if isinstance(safety_inputs.get("queue_authority"), Mapping) else {}
    if str(queue_authority.get("status") or "").strip().upper() not in {"CLEAR", "PASS", "OK"}:
        blockers.append("BLK_PROOF_QUEUE_AUTHORITY_AMBIGUOUS")

    connector = safety_inputs.get("connector_freshness") if isinstance(safety_inputs.get("connector_freshness"), Mapping) else {}
    if str(connector.get("status") or "").strip().upper() not in {"FRESH", "PASS", "OK"}:
        blockers.append("BLK_PROOF_CONNECTOR_STALE")

    if status == "ACTIVE":
        if mode == "reset":
            if str(verdicts.get("reset_safety") or "").strip().upper() != "PASS":
                blockers.extend(producer_blockers or ["BLK_PROOF_INVALIDATED"])
        elif mode == "resume":
            if str(verdicts.get("resume_safety") or "").strip().upper() != "PASS":
                blockers.extend(producer_blockers or ["BLK_PROOF_INVALIDATED"])

    if require_no_post_proof_invalidation:
        invalidation = proof.get("invalidation") if isinstance(proof.get("invalidation"), Mapping) else {}
        if _non_empty(invalidation.get("invalidated_at")):
            blockers.append("BLK_PROOF_INVALIDATED")

    blockers = _unique(blockers)

    if not blockers:
        proof_state = "PROOF_VALID_PASS"
    elif proof_state not in {"PROOF_MISSING", "PROOF_REFUSED", "PROOF_EXPIRED", "PROOF_INVALID"}:
        proof_state = "PROOF_VALID_BLOCKED"

    return {
        "ok": not blockers,
        "proof_state": proof_state,
        "blockers": blockers,
        "proof_id": _non_empty(proof.get("proof_id")),
        "expires_at": _non_empty(proof.get("expires_at")),
    }


def evaluate_reset_readiness_with_proof(
    *,
    proof: Mapping[str, Any] | None,
    evaluated_at: str | None = None,
    expected_generation_id: str | None = None,
    expected_pointer_sha256: str | None = None,
    baseline_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    proof_eval = evaluate_proof_consumability(
        proof,
        mode="reset",
        now=evaluated_at,
        expected_generation_id=expected_generation_id,
        expected_pointer_sha256=expected_pointer_sha256,
    )

    checks_bool = {
        "continuity_fresh": True,
        "proof_valid": bool(proof_eval.get("ok")),
        "mutation_safe": True,
        "coherence_valid": True,
        "queue_authority_clear": True,
        "connector_fresh": True,
    }

    for key, value in (baseline_checks or {}).items():
        if key in checks_bool:
            checks_bool[key] = bool(value)

    blockers: list[str] = list(proof_eval.get("blockers") or [])
    for key, passed in checks_bool.items():
        if passed:
            continue
        mapped = RESET_BLOCKER_MAP.get(key)
        if mapped:
            blockers.append(mapped)

    blockers = _unique(blockers)
    verdict = "PASS" if not blockers else "FAIL_BLOCKED"

    checks = {key: ("pass" if passed else "fail") for key, passed in checks_bool.items()}

    required_actions = []
    if blockers:
        if any(code.startswith("BLK_PROOF_") for code in blockers):
            required_actions.append("refresh_successor_safe_proof")
        if "BLK_MUTATION_IN_FLIGHT_UNSAFE" in blockers:
            required_actions.append("drain_or_abort_unsafe_mutation")

    return {
        "object_type": "clawd.failover_fsm.reset_readiness_report.v1",
        "evaluated_at": _now_iso(evaluated_at),
        "verdict": verdict,
        "checks": checks,
        "blockers": blockers,
        "next_required_actions": required_actions,
        "proof": {
            "proof_state": proof_eval.get("proof_state"),
            "proof_id": proof_eval.get("proof_id"),
            "expires_at": proof_eval.get("expires_at"),
        },
    }


def evaluate_successor_resume_validation_with_proof(
    *,
    proof: Mapping[str, Any] | None,
    successor_generation_id: str | None,
    expected_pointer_sha256: str | None = None,
    post_proof_invalidation_absent: bool = True,
    queue_reconcile_clear: bool = True,
    lease_transfer_or_reclaim: bool = True,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    proof_eval = evaluate_proof_consumability(
        proof,
        mode="resume",
        now=evaluated_at,
        expected_generation_id=successor_generation_id,
        expected_pointer_sha256=expected_pointer_sha256,
        require_no_post_proof_invalidation=post_proof_invalidation_absent,
    )

    proof_blockers = list(proof_eval.get("blockers") or [])
    generation_parity_ok = "BLK_PROOF_GENERATION_MISMATCH" not in proof_blockers
    read_pointer_parity_ok = "BLK_PROOF_READ_POINTER_MISMATCH" not in proof_blockers
    takeover_stability_ok = generation_parity_ok and read_pointer_parity_ok and bool(proof_eval.get("ok"))

    checks = {
        "generation_parity": "pass" if generation_parity_ok else "fail",
        "read_pointer_parity": "pass" if read_pointer_parity_ok else "fail",
        "takeover_stability": "pass" if takeover_stability_ok else "fail",
        "resume_safety_proof": "pass" if bool(proof_eval.get("ok")) else "fail",
        "queue_reconcile": "pass" if queue_reconcile_clear else "fail",
        "lease_transfer_or_reclaim": "pass" if lease_transfer_or_reclaim else "fail",
        "post_proof_invalidation_absent": "pass" if post_proof_invalidation_absent else "fail",
    }

    blockers: list[str] = list(proof_eval.get("blockers") or [])
    if not queue_reconcile_clear:
        blockers.append("BLK_QUEUE_AUTHORITY_AMBIGUOUS")
    if not lease_transfer_or_reclaim:
        blockers.append("BLK_LEASE_OWNERSHIP_AMBIGUOUS")
    if not post_proof_invalidation_absent:
        blockers.append("BLK_PROOF_INVALIDATED")

    blockers = _unique(blockers)
    verdict = "PASS" if not blockers else "FAIL_BLOCKED"

    return {
        "object_type": "clawd.failover_fsm.successor_resume_validation_report.v1",
        "evaluated_at": _now_iso(evaluated_at),
        "verdict": verdict,
        "checks": checks,
        "blockers": blockers,
        "proof": {
            "proof_state": proof_eval.get("proof_state"),
            "proof_id": proof_eval.get("proof_id"),
            "expires_at": proof_eval.get("expires_at"),
        },
    }


def reset_readiness_report_to_trigger(report: Mapping[str, Any]) -> str:
    verdict = str(report.get("verdict") or "").strip().upper()
    return "TR_RESET_READINESS_PASS" if verdict == "PASS" else "TR_RESET_READINESS_FAIL"


def successor_validation_report_to_trigger(report: Mapping[str, Any]) -> str:
    verdict = str(report.get("verdict") or "").strip().upper()
    return "TR_SUCCESSOR_VALIDATION_PASS" if verdict == "PASS" else "TR_SUCCESSOR_VALIDATION_FAIL"


def project_proof_status(
    *,
    proof: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    proof_eval = evaluation if isinstance(evaluation, Mapping) else {}
    proof_map = proof if isinstance(proof, Mapping) else {}

    blockers = [str(item).strip() for item in (proof_eval.get("blockers") or []) if str(item).strip()]
    producer_verdicts = proof_map.get("verdicts") if isinstance(proof_map.get("verdicts"), Mapping) else {}
    producer_blockers = [str(item).strip() for item in (producer_verdicts.get("blockers") or []) if str(item).strip()]

    effective_top_blocker = producer_blockers[0] if producer_blockers else (blockers[1] if len(blockers) > 1 else (blockers[0] if blockers else None))

    return {
        "$id": "clawd.successor_safe_handover_proof.status.v1",
        "proof_state": str(proof_eval.get("proof_state") or "PROOF_MISSING"),
        "proof_id": _non_empty(proof_map.get("proof_id")),
        "evaluated_at": _now_iso(evaluated_at),
        "reset_allowed": bool(proof_eval.get("ok")),
        "resume_allowed": bool(proof_eval.get("ok")),
        "top_blocker": blockers[0] if blockers else None,
        "effective_top_blocker": effective_top_blocker,
        "blockers": blockers,
    }


def write_proof_artifact(
    *,
    root: str | pathlib.Path,
    proof: Mapping[str, Any],
    rel_path: str = _DEFAULT_PROOF_PATH,
) -> pathlib.Path:
    root_path = pathlib.Path(root).resolve()
    out_path = (root_path / rel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(proof), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
