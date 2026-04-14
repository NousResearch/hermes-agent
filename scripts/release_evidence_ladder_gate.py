#!/usr/bin/env python3
"""Release evidence ladder gate (Wave 6, v1)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "release_evidence_bundle.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "release_governance" / "release_evidence_ladder_decisions.jsonl"
DEFAULT_ROLLBACK_TRIGGER_LATEST = Path(
    "state/continuity/latest/release_error_budget_rollback_trigger_latest.json"
)
DEFAULT_ROLLBACK_TRIGGER_HISTORY = Path(
    "state/continuity/release_governance/release_error_budget_rollback_trigger_history.jsonl"
)
DEFAULT_ROLLBACK_COMMAND = "bash ops/openclaw/continuity/verify_then_resume.sh --run-rollback"
DEFAULT_XD_GATE_RUNTIME_LATEST = Path("state/continuity/latest/xd_design_gate_runtime_latest.json")
DEFAULT_XD_GATE_RUNTIME_HISTORY = Path("state/continuity/design_governance/xd_design_gate_runtime_history.jsonl")

STAGE_ORDER = [
    "local_determinism",
    "presubmit",
    "integration_replay",
    "shadow",
    "canary",
    "progressive",
    "broad_activation",
]

REQUIRED_STAGE_DEPTH_BY_MODE = {
    "shadow": 4,
    "canary": 5,
    "progressive": 6,
    "broad_activation": 7,
}

XD_GATE_ORDER = [
    "G1_SCHEMA",
    "G2_STRUCTURE",
    "G3_A11Y",
    "G4_VISUAL",
    "G5_RUNTIME",
    "G6_ALIGNMENT",
]

DEFAULT_REQUIRED_HEALTH_LANES = [
    "A1_CONTROL_PLANE",
    "A2_RUNTIME_CONTINUITY",
    "A3_MODEL_ROUTING",
    "A6_OPS_OBSERVABILITY",
    "C1_OPERATOR_SURFACE",
    "C2_RELEASE_SUBSTRATE",
]

HEALTH_LAYER_RANK = {
    "alive": 1,
    "ready": 2,
    "safe-to-act": 3,
    "truthful": 4,
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_iso(raw: str) -> dt.datetime:
    token = str(raw or "").strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(token)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def gate_schema(bundle: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(bundle),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        },
    )


def gate_stage_order(bundle: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    stages = bundle.get("stages") if isinstance(bundle.get("stages"), list) else []
    observed = [str((row or {}).get("stage") or "") for row in stages if isinstance(row, dict)]
    if not observed:
        return False, "stage_order_failed", {"error": "stages_missing"}

    indexes: List[int] = []
    for stage in observed:
        if stage not in STAGE_ORDER:
            return False, "stage_order_failed", {"error": "stage_unknown", "stage": stage}
        indexes.append(STAGE_ORDER.index(stage))

    if indexes != sorted(indexes):
        return False, "stage_order_failed", {"error": "stage_order_invalid", "observed": observed}

    if len(set(observed)) != len(observed):
        return False, "stage_order_failed", {"error": "stage_duplicate", "observed": observed}

    return True, None, {"observed": observed}


def gate_stage_coverage(bundle: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    mode = str(bundle.get("activation_mode") or "")
    required_depth = REQUIRED_STAGE_DEPTH_BY_MODE.get(mode)
    if required_depth is None:
        return False, "stage_coverage_failed", {"error": "activation_mode_unknown", "activation_mode": mode}

    required = STAGE_ORDER[:required_depth]
    stages = bundle.get("stages") if isinstance(bundle.get("stages"), list) else []
    stage_map = {str((row or {}).get("stage") or ""): row for row in stages if isinstance(row, dict)}

    missing = [name for name in required if name not in stage_map]
    if missing:
        return False, "stage_coverage_failed", {
            "error": "required_stage_missing",
            "activation_mode": mode,
            "required_stages": required,
            "missing": missing,
        }

    blocked = [name for name in required if str((stage_map.get(name) or {}).get("status") or "") != "pass"]
    if blocked:
        return False, "stage_coverage_failed", {
            "error": "required_stage_not_pass",
            "activation_mode": mode,
            "blocked_stages": blocked,
        }

    return True, None, {"activation_mode": mode, "required_stages": required}


def gate_evidence_refs(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    stages = bundle.get("stages") if isinstance(bundle.get("stages"), list) else []
    checked = 0
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        refs = stage.get("evidence_refs") if isinstance(stage.get("evidence_refs"), list) else []
        for ref in refs:
            raw = str(ref or "").strip()
            if not raw:
                return False, "evidence_refs_failed", {"error": "evidence_ref_missing"}
            resolved = resolve_repo_path(repo_root, raw)
            if not is_within(repo_root, resolved):
                return False, "evidence_refs_failed", {"error": "evidence_ref_outside_repo", "path": raw}
            if not resolved.exists() or not resolved.is_file():
                return False, "evidence_refs_failed", {"error": "evidence_ref_unresolved", "path": raw}
            checked += 1

    if checked == 0:
        return False, "evidence_refs_failed", {"error": "evidence_refs_empty"}

    return True, None, {"checked_refs": checked}


def _design_gate_required(bundle: Dict[str, Any]) -> bool:
    lane_context = bundle.get("lane_context") if isinstance(bundle.get("lane_context"), dict) else {}
    lane_id = str(lane_context.get("lane_id") or "").strip().lower()
    if not lane_id:
        return False

    if lane_id in {"xd", "lane.xd", "lane.designops", "lane.designops.runtime"}:
        return True

    if lane_id.startswith("xd"):
        return True

    return "designops" in lane_id


def gate_design_gate_stack(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    required = _design_gate_required(bundle)
    row = bundle.get("design_gate_stack") if isinstance(bundle.get("design_gate_stack"), dict) else None

    if row is None:
        if required:
            return False, "design_gate_stack_missing", {"required": True}
        return True, None, {"applicable": False, "required": False}

    ordered_results = row.get("ordered_results") if isinstance(row.get("ordered_results"), list) else []
    if len(ordered_results) != len(XD_GATE_ORDER):
        return False, "design_gate_stack_invalid", {
            "error": "ordered_results_length_invalid",
            "expected": len(XD_GATE_ORDER),
            "observed": len(ordered_results),
        }

    observed_order: List[str] = []
    checked_refs = 0
    failing: List[str] = []

    for idx, item in enumerate(ordered_results):
        if not isinstance(item, dict):
            return False, "design_gate_stack_invalid", {"error": "gate_row_invalid", "index": idx}

        gate_id = str(item.get("gate_id") or "").strip()
        expected = XD_GATE_ORDER[idx]
        if gate_id != expected:
            return False, "design_gate_stack_order_invalid", {
                "error": "gate_order_mismatch",
                "index": idx,
                "expected": expected,
                "observed": gate_id,
            }
        observed_order.append(gate_id)

        status = str(item.get("status") or "").strip()
        if status not in {"pass", "block"}:
            return False, "design_gate_stack_invalid", {
                "error": "gate_status_invalid",
                "gate_id": gate_id,
                "status": status,
            }

        evidence_refs = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
        if not evidence_refs:
            return False, "design_gate_stack_evidence_missing", {
                "error": "gate_evidence_refs_missing",
                "gate_id": gate_id,
            }

        for ref in evidence_refs:
            raw = str(ref or "").strip()
            if not raw:
                return False, "design_gate_stack_evidence_missing", {
                    "error": "gate_evidence_ref_blank",
                    "gate_id": gate_id,
                }
            resolved = resolve_repo_path(repo_root, raw)
            if not is_within(repo_root, resolved):
                return False, "design_gate_stack_evidence_missing", {
                    "error": "gate_evidence_ref_outside_repo",
                    "gate_id": gate_id,
                    "path": raw,
                }
            if not resolved.exists() or not resolved.is_file():
                return False, "design_gate_stack_evidence_missing", {
                    "error": "gate_evidence_ref_unresolved",
                    "gate_id": gate_id,
                    "path": raw,
                }
            checked_refs += 1

        if status != "pass":
            failing.append(gate_id)

    if observed_order != XD_GATE_ORDER:
        return False, "design_gate_stack_order_invalid", {
            "error": "gate_order_invalid",
            "expected": XD_GATE_ORDER,
            "observed": observed_order,
        }

    if failing:
        return False, "design_gate_stack_not_pass", {
            "failing_gates": failing,
            "required": required,
            "observed_order": observed_order,
            "checked_refs": checked_refs,
        }

    return True, None, {
        "applicable": True,
        "required": required,
        "observed_order": observed_order,
        "checked_refs": checked_refs,
    }


def gate_rollback_recency(bundle: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    rollback = bundle.get("rollback_proof") if isinstance(bundle.get("rollback_proof"), dict) else {}
    drilled_at_raw = str(rollback.get("drilled_at") or "")
    max_age_hours = int(rollback.get("max_age_hours") or 0)

    if not drilled_at_raw:
        return False, "rollback_recency_failed", {"error": "drilled_at_missing"}
    if max_age_hours <= 0:
        return False, "rollback_recency_failed", {"error": "max_age_hours_invalid", "max_age_hours": max_age_hours}

    try:
        drilled_at = parse_iso(drilled_at_raw)
    except Exception as exc:
        return False, "rollback_recency_failed", {"error": "drilled_at_invalid", "detail": str(exc)}

    age_hours = (dt.datetime.now(dt.timezone.utc) - drilled_at).total_seconds() / 3600.0
    if age_hours > float(max_age_hours):
        return False, "rollback_recency_failed", {
            "error": "rollback_drill_stale",
            "age_hours": round(age_hours, 2),
            "max_age_hours": max_age_hours,
            "drilled_at": drilled_at_raw,
        }

    return True, None, {
        "age_hours": round(age_hours, 2),
        "max_age_hours": max_age_hours,
        "drilled_at": drilled_at_raw,
    }


def gate_a6_observability(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    health_path = repo_root / "state" / "continuity" / "latest" / "layered_health_snapshot.json"
    slo_path = repo_root / "state" / "continuity" / "latest" / "slo_snapshot.json"

    if not health_path.exists() or not health_path.is_file():
        return False, "a6_layered_health_missing", {"path": str(health_path)}
    if not slo_path.exists() or not slo_path.is_file():
        return False, "a6_slo_snapshot_missing", {"path": str(slo_path)}

    try:
        health_data = load_json_file(health_path)
    except Exception as exc:
        return False, "a6_layered_health_unreadable", {"path": str(health_path), "detail": str(exc)}

    try:
        slo_data = load_json_file(slo_path)
    except Exception as exc:
        return False, "a6_slo_snapshot_unreadable", {"path": str(slo_path), "detail": str(exc)}

    if not isinstance(health_data, dict):
        return False, "a6_layered_health_invalid", {"error": "not_object", "path": str(health_path)}
    if not isinstance(slo_data, dict):
        return False, "a6_slo_snapshot_invalid", {"error": "not_object", "path": str(slo_path)}

    health_status = str(health_data.get("status") or "unknown")
    health_layer = str(health_data.get("health_layer") or "unknown")
    slo_status = str(slo_data.get("status") or "unknown")
    evaluations = slo_data.get("evaluations") if isinstance(slo_data.get("evaluations"), list) else []
    failing_slos = [
        str((row or {}).get("id") or "unknown")
        for row in evaluations
        if isinstance(row, dict) and str((row or {}).get("status") or "").strip().lower() != "pass"
    ]

    requirement = bundle.get("health_requirement") if isinstance(bundle.get("health_requirement"), dict) else {}
    requirement_mode = str(requirement.get("mode") or "strict").strip().lower() or "strict"
    required_lanes = [
        str(token).strip()
        for token in (
            requirement.get("required_lanes")
            if isinstance(requirement.get("required_lanes"), list)
            else DEFAULT_REQUIRED_HEALTH_LANES
        )
        if str(token).strip()
    ]
    if not required_lanes:
        required_lanes = list(DEFAULT_REQUIRED_HEALTH_LANES)

    min_health_layer = str(requirement.get("min_health_layer") or "truthful").strip().lower() or "truthful"
    if min_health_layer not in HEALTH_LAYER_RANK:
        min_health_layer = "truthful"

    require_restore_evidence = requirement.get("require_restore_evidence")
    if not isinstance(require_restore_evidence, bool):
        require_restore_evidence = True

    lane_rows = health_data.get("lanes") if isinstance(health_data.get("lanes"), list) else []
    lane_index: Dict[str, Dict[str, Any]] = {}
    for row in lane_rows:
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane") or "").strip()
        if not lane_id:
            continue
        lane_index[lane_id] = row

    missing_required_lanes = [lane for lane in required_lanes if lane not in lane_index]
    failing_required_lanes: List[str] = []
    layer_insufficient_required_lanes: List[str] = []
    for lane in required_lanes:
        row = lane_index.get(lane)
        if not isinstance(row, dict):
            continue
        lane_status = str(row.get("status") or "unknown").strip().lower()
        lane_layer = str(row.get("health_layer") or "unknown").strip().lower()
        if lane_status != "pass":
            failing_required_lanes.append(lane)
        if HEALTH_LAYER_RANK.get(lane_layer, 0) < HEALTH_LAYER_RANK[min_health_layer]:
            layer_insufficient_required_lanes.append(lane)

    slo_status_by_id = {
        str((row or {}).get("id") or "").strip(): str((row or {}).get("status") or "").strip().lower()
        for row in evaluations
        if isinstance(row, dict) and str((row or {}).get("id") or "").strip()
    }
    restore_slo_status = slo_status_by_id.get("SLO-4_RESTORE_DRILL_FRESHNESS")

    # Release ladder promotion is blocked unless A6 observability is fully green.
    if health_status != "pass":
        return False, "a6_layered_health_not_pass", {
            "health_status": health_status,
            "health_layer": health_layer,
        }
    if health_layer not in {"safe-to-act", "truthful"}:
        return False, "a6_layered_health_layer_insufficient", {
            "health_status": health_status,
            "health_layer": health_layer,
        }
    if requirement_mode == "strict":
        if missing_required_lanes:
            return False, "a6_layered_health_required_lanes_missing", {
                "required_lanes": required_lanes,
                "missing_required_lanes": missing_required_lanes,
                "min_health_layer": min_health_layer,
            }
        if failing_required_lanes:
            return False, "a6_layered_health_required_lanes_not_pass", {
                "required_lanes": required_lanes,
                "failing_required_lanes": failing_required_lanes,
                "min_health_layer": min_health_layer,
            }
        if layer_insufficient_required_lanes:
            return False, "a6_layered_health_required_lanes_layer_insufficient", {
                "required_lanes": required_lanes,
                "layer_insufficient_required_lanes": layer_insufficient_required_lanes,
                "min_health_layer": min_health_layer,
            }
    if slo_status != "pass":
        return False, "a6_slo_budget_not_pass", {"slo_status": slo_status}
    if failing_slos:
        return False, "a6_slo_evaluations_not_pass", {
            "slo_status": slo_status,
            "failing_slos": failing_slos,
        }
    if require_restore_evidence and restore_slo_status != "pass":
        return False, "a6_restore_evidence_not_pass", {
            "restore_slo_id": "SLO-4_RESTORE_DRILL_FRESHNESS",
            "restore_slo_status": restore_slo_status or "missing",
            "require_restore_evidence": require_restore_evidence,
        }

    return True, None, {
        "health_status": health_status,
        "health_layer": health_layer,
        "slo_status": slo_status,
        "evaluated_slos": len(evaluations),
        "requirement_mode": requirement_mode,
        "required_lanes": required_lanes,
        "min_health_layer": min_health_layer,
        "missing_required_lanes": missing_required_lanes,
        "failing_required_lanes": failing_required_lanes,
        "layer_insufficient_required_lanes": layer_insufficient_required_lanes,
        "restore_slo_status": restore_slo_status,
        "require_restore_evidence": require_restore_evidence,
    }


def gate_compatibility_lifecycle(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    row = bundle.get("compatibility_lifecycle") if isinstance(bundle.get("compatibility_lifecycle"), dict) else {}
    register_ref = str(row.get("register_ref") or "").strip()
    active_exceptions = int(row.get("active_exceptions") or 0)
    rfc_refs = row.get("removal_rfc_refs") if isinstance(row.get("removal_rfc_refs"), list) else []

    if not register_ref:
        return False, "compatibility_lifecycle_failed", {"error": "register_ref_missing"}

    register_path = resolve_repo_path(repo_root, register_ref)
    if not is_within(repo_root, register_path):
        return False, "compatibility_lifecycle_failed", {"error": "register_ref_outside_repo", "path": register_ref}
    if not register_path.exists() or not register_path.is_file():
        return False, "compatibility_lifecycle_failed", {"error": "register_ref_unresolved", "path": register_ref}

    if active_exceptions > 0 and not rfc_refs:
        return False, "compatibility_lifecycle_failed", {
            "error": "removal_rfc_refs_required_for_active_exceptions",
            "active_exceptions": active_exceptions,
        }

    checked_rfcs: List[str] = []
    for ref in rfc_refs:
        token = str(ref or "").strip()
        if not token:
            return False, "compatibility_lifecycle_failed", {"error": "removal_rfc_ref_missing"}
        rfc_path = resolve_repo_path(repo_root, token)
        if not is_within(repo_root, rfc_path):
            return False, "compatibility_lifecycle_failed", {"error": "removal_rfc_ref_outside_repo", "path": token}
        if not rfc_path.exists() or not rfc_path.is_file():
            return False, "compatibility_lifecycle_failed", {"error": "removal_rfc_ref_unresolved", "path": token}
        checked_rfcs.append(token)

    return True, None, {
        "register_ref": register_ref,
        "active_exceptions": active_exceptions,
        "checked_rfc_refs": checked_rfcs,
    }


def append_jsonl_record(repo_root: Path, target_path: Optional[Path], payload: Dict[str, Any]) -> Dict[str, Any]:
    if target_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = target_path if target_path.is_absolute() else (repo_root / target_path).resolve()
    if not is_within(repo_root, path):
        return {"enabled": True, "appended": False, "reason": "unsafe_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"enabled": True, "appended": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(payload) + "\n")
        return {"enabled": True, "appended": True, "path": str(path)}
    except Exception as exc:
        return {"enabled": True, "appended": False, "reason": "append_failed", "path": str(path), "detail": str(exc)}


def append_decision_record(repo_root: Path, decision_log: Optional[Path], payload: Dict[str, Any]) -> Dict[str, Any]:
    return append_jsonl_record(repo_root, decision_log, payload)


def _gate_entry(result: Dict[str, Any], gate_name: str) -> Optional[Dict[str, Any]]:
    for row in result.get("gates") or []:
        if isinstance(row, dict) and str(row.get("gate") or "") == gate_name:
            return row
    return None


def _write_json_payload(repo_root: Path, path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    resolved = path if path.is_absolute() else (repo_root / path).resolve()
    if not is_within(repo_root, resolved):
        return {"enabled": True, "written": False, "reason": "unsafe_path", "path": str(resolved)}

    try:
        if resolved.exists() and not resolved.is_file():
            return {"enabled": True, "written": False, "reason": "path_not_file", "path": str(resolved)}
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"enabled": True, "written": True, "path": str(resolved)}
    except Exception as exc:
        return {"enabled": True, "written": False, "reason": "write_failed", "path": str(resolved), "detail": str(exc)}


def write_rollback_trigger_artifacts(
    repo_root: Path,
    result: Dict[str, Any],
    *,
    latest_path: Path = DEFAULT_ROLLBACK_TRIGGER_LATEST,
    history_path: Path = DEFAULT_ROLLBACK_TRIGGER_HISTORY,
) -> Dict[str, Any]:
    a6_gate = _gate_entry(result, "a6_observability") or {}
    a6_blocked = result.get("decision") == "BLOCK" and str(result.get("block_gate") or "") == "a6_observability"

    observability_details = (a6_gate.get("details") if isinstance(a6_gate, dict) else {}) or {}
    if not isinstance(observability_details, dict):
        observability_details = {"raw": observability_details}

    bundle_row = result.get("bundle") if isinstance(result.get("bundle"), dict) else {}
    trigger_basis = {
        "release_id": result.get("release_id"),
        "bundle_sha256": bundle_row.get("sha256"),
        "decision": result.get("decision"),
        "block_gate": result.get("block_gate"),
        "block_reason": result.get("block_reason"),
        "a6_observability": observability_details,
    }
    trigger_hash = hashlib.sha256(stable_json_dumps(trigger_basis).encode("utf-8")).hexdigest()

    payload = {
        "schema": "clawd.release_error_budget_rollback_trigger.v1",
        "generated_at": result.get("evaluated_at") or now_iso(),
        "trigger_hash": trigger_hash,
        "trigger_active": bool(a6_blocked),
        "trigger_source": "release_evidence_ladder_gate",
        "rollback_command": DEFAULT_ROLLBACK_COMMAND,
        "release_id": result.get("release_id"),
        "decision": result.get("decision"),
        "block_gate": result.get("block_gate"),
        "block_reason": result.get("block_reason"),
        "observability_gate": {
            "status": a6_gate.get("status") if isinstance(a6_gate, dict) else None,
            "reason": a6_gate.get("reason") if isinstance(a6_gate, dict) else None,
            "details": observability_details,
        },
        "evidence_refs": {
            "decision_log": (result.get("decision_record") or {}).get("path"),
            "bundle_path": bundle_row.get("path"),
            "health_snapshot": "state/continuity/latest/layered_health_snapshot.json",
            "slo_snapshot": "state/continuity/latest/slo_snapshot.json",
        },
    }

    latest_write = _write_json_payload(repo_root, latest_path, payload)
    history_append = append_jsonl_record(repo_root, history_path, payload)

    return {
        "latest": latest_write,
        "history": history_append,
        "trigger_active": bool(a6_blocked),
        "trigger_hash": trigger_hash,
        "path": str((latest_path if latest_path.is_absolute() else (repo_root / latest_path).resolve())),
    }


def write_design_gate_runtime_artifacts(
    repo_root: Path,
    result: Dict[str, Any],
    bundle: Optional[Dict[str, Any]],
    *,
    latest_path: Path = DEFAULT_XD_GATE_RUNTIME_LATEST,
    history_path: Path = DEFAULT_XD_GATE_RUNTIME_HISTORY,
) -> Dict[str, Any]:
    bundle_row = bundle if isinstance(bundle, dict) else {}
    required = _design_gate_required(bundle_row)
    gate_entry = _gate_entry(result, "design_gate_stack") or {}
    details = gate_entry.get("details") if isinstance(gate_entry, dict) and isinstance(gate_entry.get("details"), dict) else {}
    stack_payload = bundle_row.get("design_gate_stack") if isinstance(bundle_row.get("design_gate_stack"), dict) else None

    applicable = bool(stack_payload) or required
    if not applicable:
        return {"enabled": False, "written": False, "reason": "not_applicable"}

    payload = {
        "schema": "clawd.xd_design_gate_runtime_snapshot.v1",
        "generated_at": result.get("evaluated_at") or now_iso(),
        "release_id": result.get("release_id"),
        "lane_context": bundle_row.get("lane_context") if isinstance(bundle_row.get("lane_context"), dict) else {},
        "required": required,
        "release_gate_decision": result.get("decision"),
        "release_gate_block_gate": result.get("block_gate"),
        "release_gate_block_reason": result.get("block_reason"),
        "design_gate_stack": stack_payload or {},
        "design_gate_gate_result": {
            "status": gate_entry.get("status") if isinstance(gate_entry, dict) else "unknown",
            "reason": gate_entry.get("reason") if isinstance(gate_entry, dict) else None,
            "details": details,
        },
    }

    latest_write = _write_json_payload(repo_root, latest_path, payload)
    history_append = append_jsonl_record(repo_root, history_path, payload)
    return {
        "enabled": True,
        "written": bool(latest_write.get("written")),
        "latest": latest_write,
        "history": history_append,
        "path": str((latest_path if latest_path.is_absolute() else (repo_root / latest_path).resolve())),
    }


def evaluate_bundle(bundle: Any, bundle_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    bundle_dict = bundle if isinstance(bundle, dict) else {}

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(bundle, schema_path)),
        ("stage_order", lambda: gate_stage_order(bundle_dict)),
        ("a6_observability", lambda: gate_a6_observability(bundle_dict, repo_root)),
        ("stage_coverage", lambda: gate_stage_coverage(bundle_dict)),
        ("evidence_refs", lambda: gate_evidence_refs(bundle_dict, repo_root)),
        ("design_gate_stack", lambda: gate_design_gate_stack(bundle_dict, repo_root)),
        ("rollback_recency", lambda: gate_rollback_recency(bundle_dict)),
        ("compatibility_lifecycle", lambda: gate_compatibility_lifecycle(bundle_dict, repo_root)),
    ]

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gates.append({"gate": gate_name, "status": "skipped", "reason": "blocked_by_previous_gate"})
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gates.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_gate = gate_name
        block_reason = reason or "gate_unavailable"
        gates.append({"gate": gate_name, "status": "fail", "reason": block_reason, "details": details})

    try:
        bundle_sha = file_sha256(bundle_path)
    except Exception:
        bundle_sha = None

    return {
        "schema": "clawd.release_evidence_ladder.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "release_id": bundle_dict.get("release_id"),
        "bundle": {"path": str(bundle_path), "sha256": bundle_sha},
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Release evidence ladder gate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Release bundle schema path")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Decision log path")
    ap.add_argument("--no-decision-log", action="store_true", help="Disable decision log append")
    ap.add_argument("--bundle", required=True, help="Release evidence bundle path")
    ap.add_argument("--json", action="store_true", help="Pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    bundle_path = Path(args.bundle).expanduser().resolve()
    decision_log = None if args.no_decision_log else Path(args.decision_log).expanduser()

    bundle: Optional[Any] = None
    try:
        bundle = load_json_file(bundle_path)
    except Exception as exc:
        result = {
            "schema": "clawd.release_evidence_ladder.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "release_id": None,
            "bundle": {"path": str(bundle_path), "sha256": None},
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {"error": "bundle_json_unreadable", "detail": str(exc)},
                },
                {"gate": "stage_order", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "a6_observability", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "stage_coverage", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "evidence_refs", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "design_gate_stack", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "rollback_recency", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "compatibility_lifecycle", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_bundle(bundle, bundle_path, repo_root, schema_path)

    result["decision_record"] = append_decision_record(repo_root, decision_log, result)
    result["rollback_trigger"] = write_rollback_trigger_artifacts(repo_root, result)
    result["design_gate_runtime"] = write_design_gate_runtime_artifacts(
        repo_root,
        result,
        bundle if isinstance(bundle, dict) else None,
    )

    rc = 0 if result.get("decision") == "PASS" else 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
