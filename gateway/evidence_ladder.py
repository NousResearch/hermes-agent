"""Hermes-native release evidence ladder generation and gating."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in some environments
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

from gateway.operator_surfaces import build_operator_mission_surface, build_operator_triage_surface
from gateway.status import read_runtime_status, validate_runtime_artifacts
from hermes_constants import get_hermes_home

SCHEMA_VERSION = "clawd.release_evidence_bundle.v1"
DECISION_SCHEMA = "hermes.release_evidence_ladder_decision.v1"
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _governance_root() -> Path:
    return get_hermes_home() / "release_governance"


def _release_root(release_id: str) -> Path:
    return _governance_root() / "releases" / release_id


def _decision_log_path() -> Path:
    return _governance_root() / "release_evidence_ladder_decisions.jsonl"


def _schema_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "ops" / "schemas" / "release_evidence_bundle.schema.json"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ref(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_ref(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _bundle_path(release_id: str) -> Path:
    return _release_root(release_id) / "release_evidence_bundle.json"


def _stage_status(*conditions: bool) -> str:
    return "pass" if all(conditions) else "block"


def _collect_release_artifacts(release_id: str, repo_root: Path) -> dict[str, str]:
    mission = build_operator_mission_surface()
    triage = build_operator_triage_surface()
    runtime = read_runtime_status() or {}
    validation = validate_runtime_artifacts()
    now = _utc_now_iso()
    release_root = _release_root(release_id)

    mission_path = release_root / "operator_mission_surface.json"
    triage_path = release_root / "operator_triage_surface.json"
    rollback_path = release_root / "rollback_posture.json"
    compatibility_path = release_root / "compatibility_lifecycle.json"

    _json_dump(mission_path, mission)
    _json_dump(triage_path, triage)
    _json_dump(
        rollback_path,
        {
            "schema": "hermes.release_rollback_posture.v1",
            "generated_at": now,
            "release_id": release_id,
            "gateway_state": runtime.get("gateway_state") or "unknown",
            "restart_requested": bool(runtime.get("restart_requested")),
            "runtime_evidence_ref": str(get_hermes_home() / "gateway_runtime_events.jsonl"),
            "runtime_status_ref": str(get_hermes_home() / "gateway_state.json"),
            "validation": validation,
        },
    )
    _json_dump(
        compatibility_path,
        {
            "schema": "hermes.release_compatibility_lifecycle.v1",
            "generated_at": now,
            "release_id": release_id,
            "register_ref": str(_schema_path(repo_root)),
            "active_exceptions": 0,
            "removal_rfc_refs": [],
            "notes": "Wave 3 Hermes-native evidence ladder baseline",
        },
    )

    return {
        "mission_path": _ref(mission_path, repo_root),
        "triage_path": _ref(triage_path, repo_root),
        "rollback_path": _ref(rollback_path, repo_root),
        "compatibility_path": _ref(compatibility_path, repo_root),
        "runtime_status": str((get_hermes_home() / "gateway_state.json").resolve()),
        "runtime_events": str((get_hermes_home() / "gateway_runtime_events.jsonl").resolve()),
        "gateway_pid": str((get_hermes_home() / "gateway.pid").resolve()),
    }


def build_release_evidence_bundle(
    *,
    release_id: str,
    activation_mode: str,
    repo_root: Path,
    lane_id: str = "C2_RELEASE_SUBSTRATE",
    wave: str = "wave_3",
) -> dict[str, Any]:
    if activation_mode not in REQUIRED_STAGE_DEPTH_BY_MODE:
        raise ValueError("activation_mode must be one of shadow, canary, progressive, broad_activation")
    if not release_id.startswith("rel_"):
        raise ValueError("release_id must start with 'rel_'")

    mission = build_operator_mission_surface()
    triage = build_operator_triage_surface()
    validation = validate_runtime_artifacts()
    refs = _collect_release_artifacts(release_id, repo_root)
    now = _utc_now_iso()

    runtime_valid = bool(validation.get("runtime_status", {}).get("valid", False))
    pid_valid = bool(validation.get("pid", {}).get("valid", False))
    evidence_exists = bool(validation.get("evidence", {}).get("exists", False))
    no_operator_issues = int(triage.get("issue_count") or 0) == 0
    gateway_running = mission.get("gateway", {}).get("state") == "running"

    stage_rows = [
        {
            "stage": "local_determinism",
            "status": _stage_status(runtime_valid, pid_valid),
            "evidence_refs": [refs["runtime_status"], refs["gateway_pid"]],
            "evaluated_at": now,
            "notes": "Gateway runtime artifacts validated",
        },
        {
            "stage": "presubmit",
            "status": _stage_status(runtime_valid, pid_valid, no_operator_issues),
            "evidence_refs": [refs["mission_path"], refs["triage_path"]],
            "evaluated_at": now,
            "notes": mission.get("headline") or "Operator mission surface captured",
        },
        {
            "stage": "integration_replay",
            "status": _stage_status(runtime_valid, evidence_exists),
            "evidence_refs": [refs["runtime_events"], refs["rollback_path"]],
            "evaluated_at": now,
            "notes": "Runtime evidence history and rollback posture captured",
        },
        {
            "stage": "shadow",
            "status": _stage_status(runtime_valid, pid_valid, evidence_exists, no_operator_issues),
            "evidence_refs": [refs["mission_path"], refs["runtime_events"]],
            "evaluated_at": now,
            "notes": "Shadow rollout requires healthy runtime posture",
        },
        {
            "stage": "canary",
            "status": _stage_status(runtime_valid, pid_valid, evidence_exists, no_operator_issues, gateway_running),
            "evidence_refs": [refs["triage_path"], refs["rollback_path"]],
            "evaluated_at": now,
            "notes": "Canary rollout requires clean triage and running gateway",
        },
        {
            "stage": "progressive",
            "status": _stage_status(runtime_valid, pid_valid, evidence_exists, no_operator_issues, gateway_running),
            "evidence_refs": [refs["mission_path"], refs["triage_path"], refs["runtime_events"]],
            "evaluated_at": now,
            "notes": "Progressive rollout inherits canary health requirements",
        },
        {
            "stage": "broad_activation",
            "status": _stage_status(runtime_valid, pid_valid, evidence_exists, no_operator_issues, gateway_running),
            "evidence_refs": [refs["mission_path"], refs["triage_path"], refs["rollback_path"]],
            "evaluated_at": now,
            "notes": "Broad activation allowed only from a clean runtime evidence posture",
        },
    ]

    required_depth = REQUIRED_STAGE_DEPTH_BY_MODE[activation_mode]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "generated_at": now,
        "activation_mode": activation_mode,
        "lane_context": {"lane_id": lane_id, "wave": wave},
        "health_requirement": {
            "mode": "strict",
            "required_lanes": ["C1_OPERATOR_SURFACE", "C2_RELEASE_SUBSTRATE"],
            "min_health_layer": "safe-to-act",
            "require_restore_evidence": True,
        },
        "stages": stage_rows[:required_depth],
        "rollback_proof": {
            "artifact_rollback_ref": refs["runtime_status"],
            "state_rollback_ref": refs["runtime_events"],
            "drill_ref": refs["rollback_path"],
            "drilled_at": now,
            "max_age_hours": 24,
        },
        "compatibility_lifecycle": {
            "register_ref": refs["compatibility_path"],
            "active_exceptions": 0,
            "removal_rfc_refs": [],
        },
    }
    _json_dump(_bundle_path(release_id), bundle)
    return bundle


def _basic_schema_validation(bundle: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    required_fields = [
        "schema_version",
        "release_id",
        "generated_at",
        "activation_mode",
        "stages",
        "rollback_proof",
        "compatibility_lifecycle",
    ]
    missing = [field for field in required_fields if field not in bundle]
    if missing:
        return False, {"error": "required_fields_missing", "missing": missing}
    if bundle.get("schema_version") != SCHEMA_VERSION:
        return False, {"error": "schema_version_invalid", "observed": bundle.get("schema_version")}
    if bundle.get("activation_mode") not in REQUIRED_STAGE_DEPTH_BY_MODE:
        return False, {"error": "activation_mode_invalid", "observed": bundle.get("activation_mode")}
    if not str(bundle.get("release_id") or "").startswith("rel_"):
        return False, {"error": "release_id_invalid", "observed": bundle.get("release_id")}
    stages = bundle.get("stages")
    if not isinstance(stages, list) or not stages:
        return False, {"error": "stages_invalid"}
    return True, {"checked_fields": required_fields}


def _schema_gate(bundle: dict[str, Any], repo_root: Path) -> tuple[bool, dict[str, Any]]:
    schema_path = _schema_path(repo_root)
    fallback_ok, fallback_details = _basic_schema_validation(bundle)
    if Draft202012Validator is None or FormatChecker is None or not schema_path.exists():
        return fallback_ok, {
            "mode": "basic",
            "schema_path": str(schema_path),
            **fallback_details,
        }

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(bundle),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, {"mode": "jsonschema", "schema_path": str(schema_path)}

    err = errors[0]
    return False, {
        "mode": "jsonschema",
        "schema_path": str(schema_path),
        "error": str(err.message),
        "data_path": "/".join(str(part) for part in err.absolute_path),
    }


def _stage_order_gate(bundle: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    observed = [str((row or {}).get("stage") or "") for row in bundle.get("stages", []) if isinstance(row, dict)]
    if not observed:
        return False, {"error": "stages_missing"}
    if len(set(observed)) != len(observed):
        return False, {"error": "stage_duplicate", "observed": observed}
    indexes: list[int] = []
    for stage in observed:
        if stage not in STAGE_ORDER:
            return False, {"error": "stage_unknown", "stage": stage}
        indexes.append(STAGE_ORDER.index(stage))
    if indexes != sorted(indexes):
        return False, {"error": "stage_order_invalid", "observed": observed}
    return True, {"observed": observed}


def _stage_coverage_gate(bundle: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    activation_mode = str(bundle.get("activation_mode") or "")
    required = STAGE_ORDER[: REQUIRED_STAGE_DEPTH_BY_MODE[activation_mode]]
    stage_map = {str((row or {}).get("stage") or ""): row for row in bundle.get("stages", []) if isinstance(row, dict)}
    missing = [stage for stage in required if stage not in stage_map]
    if missing:
        return False, {"error": "required_stage_missing", "required_stages": required, "missing": missing}
    blocked = [stage for stage in required if str((stage_map[stage] or {}).get("status") or "") != "pass"]
    if blocked:
        return False, {"error": "required_stage_blocked", "blocked_stages": blocked}
    return True, {"required_stages": required}


def _artifact_refs(bundle: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for stage in bundle.get("stages", []):
        if not isinstance(stage, dict):
            continue
        refs.extend(str(ref) for ref in stage.get("evidence_refs", []) if str(ref or "").strip())
    rollback = bundle.get("rollback_proof") or {}
    refs.extend(
        str(rollback.get(key) or "")
        for key in ("artifact_rollback_ref", "state_rollback_ref", "drill_ref")
        if str(rollback.get(key) or "").strip()
    )
    compatibility = bundle.get("compatibility_lifecycle") or {}
    register_ref = str(compatibility.get("register_ref") or "").strip()
    if register_ref:
        refs.append(register_ref)
    return refs


def _evidence_refs_gate(bundle: dict[str, Any], repo_root: Path) -> tuple[bool, dict[str, Any]]:
    refs = _artifact_refs(bundle)
    if not refs:
        return False, {"error": "evidence_refs_empty"}
    missing: list[str] = []
    for ref in refs:
        if not _resolve_ref(ref, repo_root).exists():
            missing.append(ref)
    if missing:
        return False, {"error": "evidence_ref_missing", "missing": missing}
    return True, {"checked_refs": len(refs)}


def evaluate_release_evidence_ladder(*, bundle: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    gate_results = []
    for gate_id, evaluator in (
        ("schema", lambda: _schema_gate(bundle, repo_root)),
        ("stage_order", lambda: _stage_order_gate(bundle)),
        ("stage_coverage", lambda: _stage_coverage_gate(bundle)),
        ("evidence_refs", lambda: _evidence_refs_gate(bundle, repo_root)),
    ):
        passed, details = evaluator()
        gate_results.append({"gate_id": gate_id, "status": "pass" if passed else "block", "details": details})

    verdict = "pass" if all(row["status"] == "pass" for row in gate_results) else "block"
    decision = {
        "schema": DECISION_SCHEMA,
        "generated_at": _utc_now_iso(),
        "release_id": bundle.get("release_id"),
        "activation_mode": bundle.get("activation_mode"),
        "verdict": verdict,
        "gate_results": gate_results,
        "bundle_path": str(_bundle_path(str(bundle.get("release_id"))).resolve()),
    }
    decision_log = _decision_log_path()
    decision_log.parent.mkdir(parents=True, exist_ok=True)
    with decision_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return decision
