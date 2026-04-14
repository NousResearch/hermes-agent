#!/usr/bin/env python3
"""Deterministic model qualification + rollout gate runner (v1).

Evaluates a model rollout candidate packet against fail-closed governance gates:
1) schema
2) source_refs
3) qualification_checklist
4) benchmark_thresholds
5) replay_evidence
6) lane_authority
7) rollout_transition
8) rollback_killswitch

Design goals:
- deterministic decisions
- explicit block reasons
- append-only gate decision log
- fail-closed when validators are unavailable
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

try:  # pragma: no cover (environment wiring)
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

from model_pool_policy_contract import load_pool_policy


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "model_qualification_packet.schema.json"
DEFAULT_POOL_POLICY_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "model_pool_policy_v1.json"
DEFAULT_POOL_POLICY_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "model_pool_policy.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "model_rollout_gate_runner" / "decisions.jsonl"

REQUIRED_CHECKLIST_IDS = [
    "schema_contract_valid",
    "tool_compatibility_validated",
    "evidence_pointer_resolution",
    "abstention_behavior_validated",
    "fallback_route_defined",
    "cost_budget_guard_defined",
    "failure_mode_reviewed",
    "rollback_playbook_linked",
]

BENCHMARK_THRESHOLDS: Dict[str, float] = {
    "json_valid_rate": 0.995,
    "evidence_pointer_resolution_rate": 1.0,
    "abstain_f1": 0.95,
    "non_abstain_recall": 0.95,
}
MIN_EVAL_SAMPLE_SIZE = 30

ROLLOUT_ORDER: Dict[str, int] = {
    "DRAFT": 0,
    "QUALIFIED": 1,
    "SHADOW": 2,
    "CANARY": 3,
    "RING_1": 4,
    "RING_2": 5,
    "FULL": 6,
    "KILLED": 7,
}

FORWARD_TRANSITIONS: Dict[str, str] = {
    "DRAFT": "QUALIFIED",
    "QUALIFIED": "SHADOW",
    "SHADOW": "CANARY",
    "CANARY": "RING_1",
    "RING_1": "RING_2",
    "RING_2": "FULL",
}

AUTHORITY_MATRIX: Dict[str, Dict[str, Any]] = {
    "lane.column_c.upgrade_substrate": {
        "promote_max_state": "SHADOW",
        "can_rollback": True,
        "can_kill": False,
    },
    "lane.column_b.swarm_orchestration": {
        "promote_max_state": "RING_1",
        "can_rollback": True,
        "can_kill": False,
    },
    "lane.column_a.no_nudge_autonomy": {
        "promote_max_state": "FULL",
        "can_rollback": True,
        "can_kill": True,
    },
}

ACTION_ROLE_ALLOWLIST = {
    "promote": {"VALIDATOR", "LIBRARIAN", "SRE", "EXECUTOR"},
    "rollback": {"VALIDATOR", "LIBRARIAN", "SRE"},
    "kill": {"VALIDATOR", "LIBRARIAN", "SRE"},
}

CANARY_EXPOSURE_MAX = 0.20
RING_EXPOSURE_MAX = {
    1: 0.35,
    2: 0.75,
}
MAX_ROLLBACK_MINUTES = 30

LIVE_ROLLOUT_STATES = {"CANARY", "RING_1", "RING_2", "FULL"}
DEFAULT_REPLAY_EVIDENCE_INDEX = "state/continuity/latest/wave2_replay_evidence_index.json"
REPLAY_EVIDENCE_MAX_AGE_SECONDS = 6 * 60 * 60
REPLAY_REQUIRED_SOURCE_PATHS = (
    "ops/openclaw/continuity/failover_fsm.py",
    "ops/openclaw/continuity/successor_safe_handover_proof.py",
)
REPLAY_REQUIRED_ARTIFACT_KEYS = (
    "run_dir",
    "fixture_manifest_ref",
    "run_request_ref",
    "gate_trace_ref",
    "decision_ref",
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def normalize_sha256(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _validate_qualification_packet_timestamps(packet: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate qualification packet timestamp completeness.
    
    Args:
        packet: Qualification packet dictionary
    
    Returns:
        Tuple of (is_valid, missing_fields, invalid_fields)
        - is_valid: True if all required timestamps are present and valid
        - missing_fields: List of missing timestamp field names
        - invalid_fields: List of timestamp fields with invalid format
    """
    missing = []
    invalid = []
    
    # Check evaluated_at
    evaluated_at = packet.get("evaluated_at")
    if not evaluated_at:
        missing.append("evaluated_at")
    elif not parse_iso(evaluated_at):
        invalid.append("evaluated_at")
    
    # Check scorecard.scored_at
    scorecard = packet.get("scorecard", {})
    if isinstance(scorecard, dict):
        scored_at = scorecard.get("scored_at")
        if not scored_at:
            missing.append("scorecard.scored_at")
        elif not parse_iso(scored_at):
            invalid.append("scorecard.scored_at")
    else:
        # No scorecard at all
        missing.append("scorecard.scored_at")
    
    # Check scorecard.cost.provider_evidence_updated_at
    cost = scorecard.get("cost", {}) if isinstance(scorecard, dict) else {}
    if isinstance(cost, dict):
        provider_updated = cost.get("provider_evidence_updated_at")
        if not provider_updated:
            missing.append("scorecard.cost.provider_evidence_updated_at")
        elif not parse_iso(provider_updated):
            invalid.append("scorecard.cost.provider_evidence_updated_at")
    else:
        # No cost section
        missing.append("scorecard.cost.provider_evidence_updated_at")
    
    is_valid = len(missing) == 0 and len(invalid) == 0
    return is_valid, missing, invalid


def _ensure_qualification_packet_timestamps(packet: Dict[str, Any], allow_backfill: bool = False) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Ensure qualification packet has required timestamps.
    
    Args:
        packet: Qualification packet dictionary
        allow_backfill: If True, backfill missing timestamps
    
    Returns:
        Tuple of (updated_packet, was_backfilled, backfilled_fields)
        - updated_packet: Packet with timestamps (backfilled if needed)
        - was_backfilled: True if timestamps were backfilled
        - backfilled_fields: List of field names that were backfilled
    
    Raises:
        ValueError: If timestamps are missing and backfill not allowed
    """
    is_valid, missing, invalid = _validate_qualification_packet_timestamps(packet)
    
    if is_valid:
        return packet, False, []
    
    if not allow_backfill:
        raise ValueError(f"Missing required timestamp fields: {', '.join(missing)}")
    
    # Try to import and use the backfill utility
    try:
        from qualification_packet_timestamp_backfill import backfill_qualification_packet
        backfilled = backfill_qualification_packet(packet, force_fresh=False)
    except ImportError:
        # Fallback implementation if import fails
        backfilled = packet.copy()
        now_iso_str = now_iso()
        
        # Backfill evaluated_at if missing
        if not packet.get("evaluated_at"):
            backfilled["evaluated_at"] = now_iso_str
        
        # Ensure scorecard exists
        if "scorecard" not in backfilled:
            backfilled["scorecard"] = {}
        
        scorecard = backfilled["scorecard"]
        if not isinstance(scorecard, dict):
            scorecard = {}
            backfilled["scorecard"] = scorecard
        
        # Backfill scored_at if missing
        if not scorecard.get("scored_at"):
            scorecard["scored_at"] = now_iso_str
        
        # Ensure cost section exists
        if "cost" not in scorecard:
            scorecard["cost"] = {}
        
        cost = scorecard["cost"]
        if not isinstance(cost, dict):
            cost = {}
            scorecard["cost"] = cost
        
        # Backfill provider_evidence_updated_at if missing
        if not cost.get("provider_evidence_updated_at"):
            cost["provider_evidence_updated_at"] = now_iso_str
    
    was_backfilled = backfilled != packet
    
    # Determine which fields were backfilled
    backfilled_fields = []
    
    # Check evaluated_at
    if packet.get("evaluated_at") != backfilled.get("evaluated_at"):
        backfilled_fields.append("evaluated_at")
    
    # Check scored_at
    packet_scorecard = packet.get("scorecard", {})
    backfilled_scorecard = backfilled.get("scorecard", {})
    if isinstance(packet_scorecard, dict) and isinstance(backfilled_scorecard, dict):
        if packet_scorecard.get("scored_at") != backfilled_scorecard.get("scored_at"):
            backfilled_fields.append("scored_at")
    
    # Check provider_evidence_updated_at
    packet_cost = packet_scorecard.get("cost", {}) if isinstance(packet_scorecard, dict) else {}
    backfilled_cost = backfilled_scorecard.get("cost", {}) if isinstance(backfilled_scorecard, dict) else {}
    if isinstance(packet_cost, dict) and isinstance(backfilled_cost, dict):
        if packet_cost.get("provider_evidence_updated_at") != backfilled_cost.get("provider_evidence_updated_at"):
            backfilled_fields.append("provider_evidence_updated_at")
    
    return backfilled, was_backfilled, backfilled_fields


def _build_timestamp_validation_failure_result(candidate: Any, error: str) -> Dict[str, Any]:
    """Build failure result for timestamp validation.
    
    Args:
        candidate: Qualification packet candidate
        error: Error message from timestamp validation
    
    Returns:
        Failure result dictionary
    """
    candidate_obj = candidate if isinstance(candidate, dict) else {}
    qualification_id = candidate_obj.get("qualification_id") or candidate_obj.get("rollout_id")
    
    gate_rows = [
        {"gate": "timestamp_validation", "status": "fail", "reason": "missing_required_timestamps", "details": {"error": error}},
        {"gate": "schema", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "source_refs", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "pool_policy", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "qualification_checklist", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "benchmark_thresholds", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "replay_evidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "lane_authority", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "rollout_transition", "status": "skipped", "reason": "blocked_by_previous_gate"},
        {"gate": "rollback_killswitch", "status": "skipped", "reason": "blocked_by_previous_gate"},
    ]
    
    return {
        "schema": "clawd.model_rollout_gate_decision.v1",
        "timestamp": now_iso(),
        "qualification_id": qualification_id,
        "decision": "FAIL",
        "decision_reason": "timestamp_validation_failed",
        "timestamp_validation": {
            "enabled": True,
            "allow_backfill": False,
            "error": error,
        },
        "gates": gate_rows,
        "candidate_path": None,
        "candidate_sha256": None,
    }


def _extract_model_identity(candidate: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    model = candidate.get("model") if isinstance(candidate.get("model"), Mapping) else {}
    model_key = model.get("model_key") or model.get("model_ref")
    route_class = model.get("route_class") or model.get("model_family")
    provider = model.get("provider")

    out_model = str(model_key).strip() if isinstance(model_key, str) and str(model_key).strip() else None
    out_route = str(route_class).strip() if isinstance(route_class, str) and str(route_class).strip() else None
    out_provider = str(provider).strip() if isinstance(provider, str) and str(provider).strip() else None
    return out_model, out_route, out_provider


def _policy_required_checklist_ids(pool_policy: Mapping[str, Any]) -> List[str]:
    section = pool_policy.get("qualification_policy") if isinstance(pool_policy.get("qualification_policy"), Mapping) else {}
    values = section.get("required_checklist_ids") if isinstance(section.get("required_checklist_ids"), list) else []
    out = [str(x) for x in values if isinstance(x, str) and str(x).strip()]
    return out or list(REQUIRED_CHECKLIST_IDS)


def _policy_benchmark_thresholds(pool_policy: Mapping[str, Any]) -> Dict[str, float]:
    section = pool_policy.get("qualification_policy") if isinstance(pool_policy.get("qualification_policy"), Mapping) else {}
    raw = section.get("benchmark_thresholds") if isinstance(section.get("benchmark_thresholds"), Mapping) else {}
    out: Dict[str, float] = {}
    for key, default in BENCHMARK_THRESHOLDS.items():
        value = raw.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            out[key] = default
    return out


def _policy_min_eval_sample_size(pool_policy: Mapping[str, Any]) -> int:
    section = pool_policy.get("qualification_policy") if isinstance(pool_policy.get("qualification_policy"), Mapping) else {}
    raw = section.get("min_eval_sample_size")
    if isinstance(raw, int) and raw > 0:
        return raw
    return MIN_EVAL_SAMPLE_SIZE


def _policy_lane_authority(pool_policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    section = pool_policy.get("lane_authority") if isinstance(pool_policy.get("lane_authority"), Mapping) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for lane_id, lane_policy in section.items():
        if not isinstance(lane_id, str) or not isinstance(lane_policy, Mapping):
            continue
        out[lane_id] = {
            "promote_max_state": lane_policy.get("promote_max_state"),
            "can_rollback": lane_policy.get("can_rollback"),
            "can_kill": lane_policy.get("can_kill"),
        }
    return out or dict(AUTHORITY_MATRIX)


def _policy_action_role_allowlist(pool_policy: Mapping[str, Any]) -> Dict[str, Set[str]]:
    section = pool_policy.get("action_role_allowlist") if isinstance(pool_policy.get("action_role_allowlist"), Mapping) else {}
    out: Dict[str, Set[str]] = {}
    for action in ("promote", "rollback", "kill"):
        values = section.get(action)
        if isinstance(values, list):
            out[action] = {str(x) for x in values if isinstance(x, str) and str(x).strip()}
    if all(out.get(action) for action in ("promote", "rollback", "kill")):
        return out
    return {key: set(vals) for key, vals in ACTION_ROLE_ALLOWLIST.items()}


def _policy_rollout_limits(pool_policy: Mapping[str, Any]) -> Tuple[float, Dict[int, float], int]:
    section = pool_policy.get("rollout_policy") if isinstance(pool_policy.get("rollout_policy"), Mapping) else {}
    canary_limit = section.get("canary_exposure_max")
    canary = float(canary_limit) if isinstance(canary_limit, (int, float)) else CANARY_EXPOSURE_MAX

    ring_cfg = section.get("ring_exposure_max") if isinstance(section.get("ring_exposure_max"), Mapping) else {}
    ring_1 = ring_cfg.get("1")
    ring_2 = ring_cfg.get("2")
    ring = {
        1: float(ring_1) if isinstance(ring_1, (int, float)) else RING_EXPOSURE_MAX[1],
        2: float(ring_2) if isinstance(ring_2, (int, float)) else RING_EXPOSURE_MAX[2],
    }

    rollback_raw = section.get("max_rollback_minutes")
    max_rollback = rollback_raw if isinstance(rollback_raw, int) and rollback_raw > 0 else MAX_ROLLBACK_MINUTES
    return canary, ring, max_rollback


def _rollout_state_to_stage(pool_policy: Mapping[str, Any], state: Optional[str]) -> Optional[str]:
    if not isinstance(state, str) or not state:
        return None
    stage_map = pool_policy.get("state_stage_map") if isinstance(pool_policy.get("state_stage_map"), Mapping) else {}
    canary_states = {str(x) for x in stage_map.get("canary_states", []) if isinstance(x, str)}
    active_states = {str(x) for x in stage_map.get("active_states", []) if isinstance(x, str)}

    if canary_states and state in canary_states:
        return "canary"
    if active_states and state in active_states:
        return "active"

    if state == "CANARY":
        return "canary"
    if state in {"RING_1", "RING_2", "FULL"}:
        return "active"
    return None


def _allowed_stages_for_state(pool_policy: Mapping[str, Any], state: Optional[str]) -> List[str]:
    stage = _rollout_state_to_stage(pool_policy, state)
    if stage == "active":
        return ["canary", "active"]
    if stage == "canary":
        return ["canary"]
    return []


def gate_schema(candidate: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(candidate),
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


def gate_source_refs(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    refs = candidate.get("source_refs")
    if not isinstance(refs, list) or not refs:
        return False, "source_refs_unresolved", {"issues": [{"reason": "source_refs_missing"}]}

    issues: List[Dict[str, Any]] = []
    checked = 0
    for idx, ref in enumerate(refs):
        checked += 1
        if not isinstance(ref, dict):
            issues.append({"ref_index": idx, "reason": "source_ref_not_object"})
            continue

        ref_id = ref.get("ref_id")
        raw_path = ref.get("path")
        raw_hash = ref.get("content_hash")

        if not isinstance(raw_path, str) or not raw_path.strip():
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "path_missing"})
            continue

        if not isinstance(raw_hash, str) or not raw_hash.strip():
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "content_hash_missing"})
            continue

        resolved = resolve_repo_path(repo_root, raw_path)
        if not is_within(repo_root, resolved):
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "path_outside_repo", "path": raw_path})
            continue

        if not resolved.exists() or not resolved.is_file():
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "path_unresolved", "path": raw_path})
            continue

        declared = normalize_sha256(raw_hash)
        try:
            actual = file_sha256(resolved)
        except Exception as exc:
            issues.append({
                "ref_index": idx,
                "ref_id": ref_id,
                "reason": "hash_compute_failed",
                "detail": str(exc),
            })
            continue

        if declared != actual:
            issues.append({
                "ref_index": idx,
                "ref_id": ref_id,
                "reason": "content_hash_mismatch",
                "path": raw_path,
                "declared": declared,
                "actual": actual,
            })

    if issues:
        return False, "source_refs_unresolved", {"checked": checked, "issues": issues}

    return True, None, {"checked": checked}


def gate_pool_policy(candidate: Dict[str, Any], pool_policy: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    model_key, route_class, _provider = _extract_model_identity(candidate)

    if not route_class:
        return False, "pool_policy_violation", {"error": "candidate_route_class_missing"}

    route_classes = pool_policy.get("route_classes") if isinstance(pool_policy.get("route_classes"), Mapping) else {}
    route_entry = route_classes.get(route_class)
    if not isinstance(route_entry, Mapping):
        return (
            False,
            "pool_policy_violation",
            {"error": "route_class_not_defined", "route_class": route_class, "policy_id": pool_policy.get("policy_id")},
        )

    allowed_models = [
        str(x) for x in (route_entry.get("allowed_models") if isinstance(route_entry.get("allowed_models"), list) else [])
        if isinstance(x, str) and str(x).strip()
    ]
    if route_class != "NO_LLM":
        if not model_key:
            return (
                False,
                "pool_policy_violation",
                {"error": "candidate_model_key_missing", "route_class": route_class},
            )
        if model_key not in allowed_models:
            return (
                False,
                "pool_policy_violation",
                {
                    "error": "model_not_in_route_pool",
                    "route_class": route_class,
                    "model_key": model_key,
                    "allowed_models": allowed_models,
                },
            )

    return (
        True,
        None,
        {
            "policy_id": pool_policy.get("policy_id"),
            "route_class": route_class,
            "model_key": model_key,
            "route_owner_lane_id": route_entry.get("owner_lane_id"),
            "default_required_rollout_stage": route_entry.get("default_required_rollout_stage"),
        },
    )


def gate_qualification_checklist(candidate: Dict[str, Any], required_checklist_ids: List[str]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    qualification = candidate.get("qualification")
    checklist = qualification.get("checklist") if isinstance(qualification, dict) else None
    if not isinstance(checklist, list) or not checklist:
        return False, "qualification_checklist_incomplete", {"error": "checklist_missing"}

    observed: Dict[str, str] = {}
    for row in checklist:
        if not isinstance(row, dict):
            continue
        check_id = row.get("check_id")
        status = row.get("status")
        if isinstance(check_id, str) and isinstance(status, str):
            observed[check_id] = status

    required_ids = [str(x) for x in required_checklist_ids if isinstance(x, str) and str(x).strip()]
    if not required_ids:
        required_ids = list(REQUIRED_CHECKLIST_IDS)

    missing = [cid for cid in required_ids if cid not in observed]
    if missing:
        return (
            False,
            "qualification_checklist_incomplete",
            {"missing_check_ids": sorted(missing), "required": required_ids},
        )

    non_pass = {cid: observed.get(cid) for cid in required_ids if observed.get(cid) != "pass"}
    if non_pass:
        return (
            False,
            "qualification_checklist_failed",
            {"non_pass_checks": non_pass},
        )

    return True, None, {"required_checks": len(required_ids), "observed_checks": len(observed)}


def gate_benchmark_thresholds(
    candidate: Dict[str, Any],
    thresholds: Mapping[str, float],
    min_eval_sample_size: int,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    qualification = candidate.get("qualification")
    summary = qualification.get("benchmark_summary") if isinstance(qualification, dict) else None
    if not isinstance(summary, dict):
        return False, "benchmark_below_threshold", {"error": "benchmark_summary_missing"}

    violations: List[Dict[str, Any]] = []

    effective_thresholds: Dict[str, float] = {}
    for metric, default in BENCHMARK_THRESHOLDS.items():
        raw_threshold = thresholds.get(metric)
        effective_thresholds[metric] = float(raw_threshold) if isinstance(raw_threshold, (int, float)) else float(default)

    effective_min_eval = min_eval_sample_size if isinstance(min_eval_sample_size, int) and min_eval_sample_size > 0 else MIN_EVAL_SAMPLE_SIZE

    for metric, threshold in effective_thresholds.items():
        raw = summary.get(metric)
        if not isinstance(raw, (int, float)):
            violations.append({"metric": metric, "reason": "missing_or_invalid", "threshold": threshold})
            continue
        value = float(raw)
        if value < threshold:
            violations.append({"metric": metric, "value": value, "threshold": threshold})

    sample = summary.get("eval_sample_size")
    if not isinstance(sample, int) or sample < effective_min_eval:
        violations.append(
            {
                "metric": "eval_sample_size",
                "value": sample,
                "threshold": effective_min_eval,
            }
        )

    if violations:
        return (
            False,
            "benchmark_below_threshold",
            {"violations": violations, "thresholds": effective_thresholds, "min_eval_sample_size": effective_min_eval},
        )

    return True, None, {
        "thresholds": effective_thresholds,
        "min_eval_sample_size": effective_min_eval,
    }


def gate_replay_evidence(candidate: Dict[str, Any], repo_root: Path, *, now: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    rollout = candidate.get("rollout")
    action = rollout.get("requested_action") if isinstance(rollout, dict) else None
    requested_state = rollout.get("requested_state") if isinstance(rollout, dict) else None

    if action != "promote" or requested_state not in LIVE_ROLLOUT_STATES:
        return (
            True,
            None,
            {
                "required": False,
                "reason": "not_live_promotion",
                "requested_action": action,
                "requested_state": requested_state,
            },
        )

    replay_cfg = candidate.get("replay_evidence") if isinstance(candidate.get("replay_evidence"), dict) else {}

    raw_path = replay_cfg.get("index_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raw_path = DEFAULT_REPLAY_EVIDENCE_INDEX

    max_age_raw = replay_cfg.get("max_age_seconds", REPLAY_EVIDENCE_MAX_AGE_SECONDS)
    if not isinstance(max_age_raw, int) or max_age_raw <= 0:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "invalid_max_age_seconds",
                "value": max_age_raw,
                "default": REPLAY_EVIDENCE_MAX_AGE_SECONDS,
            },
        )

    evidence_path = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, evidence_path):
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "path_outside_repo",
                "path": str(evidence_path),
            },
        )

    if not evidence_path.exists() or not evidence_path.is_file():
        return (
            False,
            "replay_evidence_missing",
            {
                "error": "evidence_index_missing",
                "path": str(evidence_path),
            },
        )

    try:
        evidence = load_json_file(evidence_path)
    except Exception as exc:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "evidence_index_unreadable",
                "path": str(evidence_path),
                "detail": str(exc),
            },
        )

    if not isinstance(evidence, dict):
        return False, "replay_evidence_invalid", {"error": "evidence_not_object", "path": str(evidence_path)}

    if str(evidence.get("object_type") or "") != "clawd.wave2_replay.evidence_index.v1":
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "object_type_mismatch",
                "path": str(evidence_path),
                "object_type": evidence.get("object_type"),
            },
        )

    summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
    overall_verdict = str(summary.get("overall_verdict") or "").upper()
    if overall_verdict != "PASS":
        return (
            False,
            "replay_evidence_not_pass",
            {
                "error": "overall_verdict_not_pass",
                "path": str(evidence_path),
                "overall_verdict": overall_verdict or None,
            },
        )

    scenario_count = summary.get("scenario_count")
    if not isinstance(scenario_count, int) or scenario_count < 1:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "scenario_count_invalid",
                "path": str(evidence_path),
                "scenario_count": scenario_count,
            },
        )

    generated_at = parse_iso(evidence.get("generated_at"))
    now_dt = parse_iso(now) if now else None
    if now_dt is None:
        now_dt = dt.datetime.now(dt.timezone.utc)
    if generated_at is None:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "generated_at_invalid",
                "path": str(evidence_path),
                "generated_at": evidence.get("generated_at"),
            },
        )

    age_sec = int(max(0.0, (now_dt - generated_at).total_seconds()))
    if age_sec > max_age_raw:
        return (
            False,
            "replay_evidence_stale",
            {
                "error": "evidence_too_old",
                "path": str(evidence_path),
                "age_sec": age_sec,
                "max_age_sec": max_age_raw,
                "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            },
        )

    fixture_family = str(evidence.get("fixture_family") or "")
    if fixture_family and fixture_family != "A3_CRITICAL":
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "fixture_family_unexpected",
                "path": str(evidence_path),
                "fixture_family": fixture_family,
                "expected": "A3_CRITICAL",
            },
        )

    run_id = str(evidence.get("run_id") or "").strip()
    if not run_id:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "run_id_missing",
                "path": str(evidence_path),
            },
        )

    artifacts = evidence.get("artifacts") if isinstance(evidence.get("artifacts"), dict) else {}
    resolved_artifacts: Dict[str, str] = {}
    for key in REPLAY_REQUIRED_ARTIFACT_KEYS:
        raw_ref = artifacts.get(key)
        if not isinstance(raw_ref, str) or not raw_ref.strip():
            return (
                False,
                "replay_evidence_invalid",
                {
                    "error": "artifact_ref_missing",
                    "path": str(evidence_path),
                    "artifact_key": key,
                },
            )
        resolved_ref = resolve_repo_path(repo_root, raw_ref)
        if not is_within(repo_root, resolved_ref):
            return (
                False,
                "replay_evidence_invalid",
                {
                    "error": "artifact_ref_outside_repo",
                    "path": str(evidence_path),
                    "artifact_key": key,
                    "artifact_ref": raw_ref,
                },
            )
        if key == "run_dir":
            if not resolved_ref.exists() or not resolved_ref.is_dir():
                return (
                    False,
                    "replay_evidence_invalid",
                    {
                        "error": "run_dir_missing",
                        "path": str(evidence_path),
                        "artifact_ref": raw_ref,
                    },
                )
        else:
            if not resolved_ref.exists() or not resolved_ref.is_file():
                return (
                    False,
                    "replay_evidence_invalid",
                    {
                        "error": "artifact_ref_missing",
                        "path": str(evidence_path),
                        "artifact_key": key,
                        "artifact_ref": raw_ref,
                    },
                )
        resolved_artifacts[key] = str(resolved_ref)

    try:
        decision_obj = load_json_file(Path(resolved_artifacts["decision_ref"]))
    except Exception as exc:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "decision_artifact_unreadable",
                "path": str(evidence_path),
                "decision_ref": resolved_artifacts.get("decision_ref"),
                "detail": str(exc),
            },
        )

    if not isinstance(decision_obj, dict):
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "decision_artifact_not_object",
                "path": str(evidence_path),
                "decision_ref": resolved_artifacts.get("decision_ref"),
            },
        )

    decision_verdict = str(decision_obj.get("verdict") or "").upper()
    decision_run_id = str(decision_obj.get("run_id") or "").strip()
    if decision_verdict != overall_verdict:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "decision_verdict_mismatch",
                "path": str(evidence_path),
                "summary_verdict": overall_verdict,
                "decision_verdict": decision_verdict or None,
            },
        )
    if decision_run_id != run_id:
        return (
            False,
            "replay_evidence_invalid",
            {
                "error": "decision_run_id_mismatch",
                "path": str(evidence_path),
                "summary_run_id": run_id,
                "decision_run_id": decision_run_id or None,
            },
        )

    source_refs = evidence.get("source_refs") if isinstance(evidence.get("source_refs"), list) else []
    source_map: Dict[str, str] = {}
    for row in source_refs:
        if not isinstance(row, dict):
            continue
        src_path = str(row.get("path") or "").strip()
        src_sha = str(row.get("sha256") or "").strip()
        if src_path and src_sha:
            source_map[src_path] = normalize_sha256(src_sha)

    for required_path in REPLAY_REQUIRED_SOURCE_PATHS:
        declared_sha = source_map.get(required_path)
        if not declared_sha:
            return (
                False,
                "replay_evidence_invalid",
                {
                    "error": "source_ref_missing",
                    "path": str(evidence_path),
                    "required_path": required_path,
                },
            )
        source_abs = resolve_repo_path(repo_root, required_path)
        if not is_within(repo_root, source_abs) or not source_abs.exists() or not source_abs.is_file():
            return (
                False,
                "replay_evidence_invalid",
                {
                    "error": "source_ref_path_unresolved",
                    "path": str(evidence_path),
                    "required_path": required_path,
                },
            )
        actual_sha = file_sha256(source_abs)
        if normalize_sha256(declared_sha) != normalize_sha256(actual_sha):
            return (
                False,
                "replay_evidence_invalid",
                {
                    "error": "source_ref_hash_mismatch",
                    "path": str(evidence_path),
                    "required_path": required_path,
                    "declared": declared_sha,
                    "actual": actual_sha,
                },
            )

    return (
        True,
        None,
        {
            "required": True,
            "path": str(evidence_path),
            "overall_verdict": overall_verdict,
            "scenario_count": scenario_count,
            "age_sec": age_sec,
            "max_age_sec": max_age_raw,
            "run_id": run_id,
            "decision_ref": resolved_artifacts.get("decision_ref"),
        },
    )


def _rank(state: str) -> Optional[int]:
    return ROLLOUT_ORDER.get(state)


def gate_lane_authority(
    candidate: Dict[str, Any],
    authority_matrix: Mapping[str, Mapping[str, Any]],
    action_role_allowlist: Mapping[str, Set[str]],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    lane_context = candidate.get("lane_context")
    rollout = candidate.get("rollout")

    lane_id = lane_context.get("requested_by_lane_id") if isinstance(lane_context, dict) else None
    actor_role = lane_context.get("actor_role") if isinstance(lane_context, dict) else None
    action = rollout.get("requested_action") if isinstance(rollout, dict) else None
    requested_state = rollout.get("requested_state") if isinstance(rollout, dict) else None

    lane_policy = authority_matrix.get(str(lane_id))
    if not lane_policy:
        return False, "lane_authority_denied", {"error": "unknown_lane", "lane_id": lane_id}

    allowed_roles = action_role_allowlist.get(str(action))
    if not allowed_roles:
        return False, "lane_authority_denied", {"error": "unknown_action", "action": action}

    if actor_role not in allowed_roles:
        return (
            False,
            "lane_authority_denied",
            {
                "error": "actor_role_not_allowed",
                "action": action,
                "actor_role": actor_role,
                "allowed_roles": sorted(list(allowed_roles)),
            },
        )

    if action == "promote":
        max_state = str(lane_policy.get("promote_max_state") or "")
        max_rank = _rank(max_state)
        req_rank = _rank(str(requested_state))
        if max_rank is None or req_rank is None or req_rank > max_rank:
            return (
                False,
                "lane_authority_denied",
                {
                    "error": "promote_state_out_of_authority",
                    "lane_id": lane_id,
                    "requested_state": requested_state,
                    "promote_max_state": max_state,
                },
            )

    elif action == "rollback":
        if lane_policy.get("can_rollback") is not True:
            return False, "lane_authority_denied", {"error": "rollback_not_allowed", "lane_id": lane_id}

    elif action == "kill":
        if lane_policy.get("can_kill") is not True:
            return False, "lane_authority_denied", {"error": "kill_not_allowed", "lane_id": lane_id}
        if requested_state != "KILLED":
            return (
                False,
                "lane_authority_denied",
                {"error": "kill_requires_killed_requested_state", "requested_state": requested_state},
            )

    return (
        True,
        None,
        {
            "lane_id": lane_id,
            "actor_role": actor_role,
            "action": action,
            "requested_state": requested_state,
            "lane_policy": lane_policy,
        },
    )


def gate_rollout_transition(
    candidate: Dict[str, Any],
    *,
    canary_exposure_max: float,
    ring_exposure_max: Mapping[int, float],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    rollout = candidate.get("rollout")
    if not isinstance(rollout, dict):
        return False, "transition_invalid", {"error": "rollout_missing"}

    current_state = rollout.get("current_state")
    requested_state = rollout.get("requested_state")
    action = rollout.get("requested_action")

    current_rank = _rank(str(current_state))
    requested_rank = _rank(str(requested_state))
    if current_rank is None or requested_rank is None:
        return (
            False,
            "transition_invalid",
            {"error": "unknown_state", "current_state": current_state, "requested_state": requested_state},
        )

    if current_state == "KILLED" and action != "kill":
        return False, "transition_invalid", {"error": "killed_state_terminal"}

    if action == "promote":
        expected = FORWARD_TRANSITIONS.get(str(current_state))
        if not expected:
            return (
                False,
                "transition_invalid",
                {"error": "no_forward_transition_defined", "current_state": current_state},
            )
        if requested_state != expected:
            return (
                False,
                "transition_invalid",
                {
                    "error": "non_stepwise_promotion",
                    "current_state": current_state,
                    "requested_state": requested_state,
                    "expected_state": expected,
                },
            )

    elif action == "rollback":
        if requested_state == "KILLED":
            return (
                False,
                "transition_invalid",
                {"error": "rollback_cannot_target_killed", "requested_state": requested_state},
            )
        if requested_rank >= current_rank:
            return (
                False,
                "transition_invalid",
                {
                    "error": "rollback_must_go_to_lower_state",
                    "current_state": current_state,
                    "requested_state": requested_state,
                },
            )

    elif action == "kill":
        if requested_state != "KILLED":
            return (
                False,
                "transition_invalid",
                {"error": "kill_requires_killed_requested_state", "requested_state": requested_state},
            )

    else:
        return False, "transition_invalid", {"error": "unknown_action", "action": action}

    # Stage-specific envelope checks
    if requested_state == "CANARY":
        canary = rollout.get("canary") if isinstance(rollout.get("canary"), dict) else None
        if not canary:
            return False, "transition_invalid", {"error": "canary_config_missing"}
        exposure = canary.get("exposure_percent")
        if not isinstance(exposure, (int, float)) or float(exposure) > float(canary_exposure_max):
            return (
                False,
                "transition_invalid",
                {
                    "error": "canary_exposure_too_high",
                    "value": exposure,
                    "max": canary_exposure_max,
                },
            )

    if requested_state in {"RING_1", "RING_2"}:
        ring = rollout.get("ring") if isinstance(rollout.get("ring"), dict) else None
        if not ring:
            return False, "transition_invalid", {"error": "ring_config_missing"}
        ring_index = ring.get("ring_index")
        expected_index = 1 if requested_state == "RING_1" else 2
        if ring_index != expected_index:
            return (
                False,
                "transition_invalid",
                {
                    "error": "ring_index_mismatch",
                    "ring_index": ring_index,
                    "expected_index": expected_index,
                },
            )
        exposure = ring.get("exposure_percent")
        max_exposure = float(ring_exposure_max.get(expected_index, RING_EXPOSURE_MAX.get(expected_index, 1.0)))
        if not isinstance(exposure, (int, float)) or float(exposure) > max_exposure:
            return (
                False,
                "transition_invalid",
                {
                    "error": "ring_exposure_too_high",
                    "ring_index": expected_index,
                    "value": exposure,
                    "max": max_exposure,
                },
            )

    return True, None, {"action": action, "current_state": current_state, "requested_state": requested_state}


def gate_rollback_killswitch(candidate: Dict[str, Any], max_rollback_minutes: int) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    rollout = candidate.get("rollout")
    rollback = candidate.get("rollback")
    kill_switch = candidate.get("kill_switch")

    if not isinstance(rollout, dict):
        return False, "safety_controls_incomplete", {"error": "rollout_missing"}
    if not isinstance(rollback, dict):
        return False, "safety_controls_incomplete", {"error": "rollback_missing"}
    if not isinstance(kill_switch, dict):
        return False, "safety_controls_incomplete", {"error": "kill_switch_missing"}

    action = rollout.get("requested_action")
    requested_state = rollout.get("requested_state")

    can_auto_rollback = rollback.get("can_auto_rollback")
    max_minutes = rollback.get("max_rollback_minutes")

    if can_auto_rollback is not True and requested_state in {"CANARY", "RING_1", "RING_2", "FULL"}:
        return (
            False,
            "safety_controls_incomplete",
            {"error": "auto_rollback_required_for_live_rollout", "requested_state": requested_state},
        )

    limit = max_rollback_minutes if isinstance(max_rollback_minutes, int) and max_rollback_minutes > 0 else MAX_ROLLBACK_MINUTES

    if not isinstance(max_minutes, int) or max_minutes > limit:
        return (
            False,
            "safety_controls_incomplete",
            {
                "error": "rollback_window_invalid",
                "value": max_minutes,
                "max_allowed": limit,
            },
        )

    armed = kill_switch.get("armed")
    engage_requested = kill_switch.get("engage_requested")

    if requested_state in {"CANARY", "RING_1", "RING_2", "FULL"} and armed is not True:
        return (
            False,
            "safety_controls_incomplete",
            {"error": "killswitch_must_be_armed_for_live_rollout", "requested_state": requested_state},
        )

    if action == "kill":
        reason = kill_switch.get("reason")
        if engage_requested is not True:
            return False, "safety_controls_incomplete", {"error": "kill_action_requires_engage_requested"}
        if armed is not True:
            return False, "safety_controls_incomplete", {"error": "kill_action_requires_armed_killswitch"}
        if not isinstance(reason, str) or not reason.strip():
            return False, "safety_controls_incomplete", {"error": "kill_action_requires_reason"}
    else:
        if engage_requested is True:
            return False, "safety_controls_incomplete", {"error": "engage_requested_requires_kill_action"}

    return (
        True,
        None,
        {
            "action": action,
            "requested_state": requested_state,
            "max_rollback_minutes": max_minutes,
            "kill_switch_armed": armed,
        },
    )


def append_decision_record(
    *,
    decision_log_path: Optional[Path],
    repo_root: Path,
    decision_row: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not is_within(repo_root, path):
        return {"enabled": True, "appended": False, "reason": "unsafe_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"enabled": True, "appended": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(decision_row) + "\n")
        return {"enabled": True, "appended": True, "path": str(path)}
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(path),
            "error": str(exc),
        }


def write_decision_artifact(*, decision_out_path: Optional[Path], repo_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    if decision_out_path is None:
        return {"enabled": False, "written": False, "reason": "disabled"}

    path = decision_out_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not is_within(repo_root, path):
        return {"enabled": True, "written": False, "reason": "unsafe_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"enabled": True, "written": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"enabled": True, "written": True, "path": str(path)}
    except Exception as exc:
        return {
            "enabled": True,
            "written": False,
            "reason": "write_failed",
            "path": str(path),
            "error": str(exc),
        }


def _decision_model_projection(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    model_key, route_class, provider = _extract_model_identity(candidate)
    return {
        "model_key": model_key,
        "provider": provider,
        "route_class": route_class,
    }


def _decision_rollout_projection(candidate: Mapping[str, Any], pool_policy: Mapping[str, Any]) -> Dict[str, Any]:
    rollout = candidate.get("rollout") if isinstance(candidate.get("rollout"), Mapping) else {}
    requested_action = rollout.get("requested_action")
    requested_state = rollout.get("requested_state")
    current_state = rollout.get("current_state")

    stage = _rollout_state_to_stage(pool_policy, requested_state if isinstance(requested_state, str) else None)
    allowed_stages = _allowed_stages_for_state(pool_policy, requested_state if isinstance(requested_state, str) else None)

    return {
        "current_state": current_state,
        "requested_action": requested_action,
        "requested_state": requested_state,
        "target_stage": stage,
        "approved_stage": stage,
        "allowed_stages": allowed_stages,
    }


def _coerce_float(raw: Any) -> Optional[float]:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return None
        try:
            return float(token)
        except Exception:
            return None
    return None


def _normalize_unit_rate(raw: Any) -> Optional[float]:
    value = _coerce_float(raw)
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 100.0:
        return value / 100.0
    return None


def _score_0_100(raw: Any) -> Optional[float]:
    value = _coerce_float(raw)
    if value is None:
        return None
    if 0.0 <= value <= 100.0:
        return value
    return None


def _decision_qualification_signal(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    qualification = candidate.get("qualification") if isinstance(candidate.get("qualification"), Mapping) else {}
    benchmark = qualification.get("benchmark_summary") if isinstance(qualification.get("benchmark_summary"), Mapping) else {}

    benchmark_metrics: Dict[str, Optional[float]] = {
        "json_valid_rate": _normalize_unit_rate(benchmark.get("json_valid_rate")),
        "evidence_pointer_resolution_rate": _normalize_unit_rate(benchmark.get("evidence_pointer_resolution_rate")),
        "abstain_f1": _normalize_unit_rate(benchmark.get("abstain_f1")),
        "non_abstain_recall": _normalize_unit_rate(benchmark.get("non_abstain_recall")),
    }
    benchmark_values = [value for value in benchmark_metrics.values() if isinstance(value, float)]
    benchmark_composite = round((sum(benchmark_values) / len(benchmark_values)) * 100.0, 2) if benchmark_values else None

    eval_sample_size: Optional[int] = None
    benchmark_eval_size = benchmark.get("eval_sample_size")
    if isinstance(benchmark_eval_size, int) and benchmark_eval_size > 0:
        eval_sample_size = benchmark_eval_size

    scorecard = candidate.get("scorecard") if isinstance(candidate.get("scorecard"), Mapping) else {}
    scorecard_cost = scorecard.get("cost") if isinstance(scorecard.get("cost"), Mapping) else {}
    scorecard_summary = scorecard.get("summary") if isinstance(scorecard.get("summary"), Mapping) else {}
    scorecard_window = scorecard.get("window") if isinstance(scorecard.get("window"), Mapping) else {}

    weighted_score = _score_0_100(scorecard_summary.get("weighted_score_0_100"))
    promotion_recommendation_raw = scorecard_summary.get("promotion_recommendation")
    promotion_recommendation = (
        str(promotion_recommendation_raw).strip().lower()
        if isinstance(promotion_recommendation_raw, str) and str(promotion_recommendation_raw).strip()
        else None
    )
    provider_cost_coverage_rate = _normalize_unit_rate(scorecard_cost.get("provider_cost_coverage_rate"))

    guardrail_violations = scorecard_summary.get("guardrail_violations") if isinstance(scorecard_summary.get("guardrail_violations"), list) else []
    guardrail_violation_count = len([row for row in guardrail_violations if isinstance(row, (str, int, float)) or isinstance(row, Mapping)])

    if eval_sample_size is None:
        scorecard_sample_count = scorecard_window.get("sample_count")
        if isinstance(scorecard_sample_count, int) and scorecard_sample_count > 0:
            eval_sample_size = scorecard_sample_count

    effective_score = weighted_score if weighted_score is not None else benchmark_composite
    if weighted_score is not None:
        score_source = "scorecard_summary"
    elif benchmark_composite is not None:
        score_source = "benchmark_summary"
    else:
        score_source = "none"

    readiness_state = "unknown"
    if effective_score is not None:
        coverage_ok = provider_cost_coverage_rate is None or provider_cost_coverage_rate >= 0.90
        promotion_block = promotion_recommendation in {"hold", "blocked", "block", "reject"}
        if effective_score >= 90.0 and guardrail_violation_count == 0 and coverage_ok and not promotion_block:
            readiness_state = "qualified"
        elif effective_score >= 80.0 and guardrail_violation_count <= 1:
            readiness_state = "provisional"
        else:
            readiness_state = "hold"

    return {
        "schema": "clawd.model_rollout_gate.qualification_signal.v1",
        "readiness_state": readiness_state,
        "score_source": score_source,
        "weighted_score_0_100": weighted_score,
        "benchmark_composite_0_100": benchmark_composite,
        "effective_score_0_100": effective_score,
        "provider_cost_coverage_rate": provider_cost_coverage_rate,
        "guardrail_violation_count": guardrail_violation_count,
        "promotion_recommendation": promotion_recommendation,
        "eval_sample_size": eval_sample_size,
        "benchmark_metrics": benchmark_metrics,
    }


def evaluate_candidate(
    *,
    candidate: Any,
    candidate_path: Path,
    repo_root: Path,
    schema_path: Path,
    pool_policy: Mapping[str, Any],
    pool_policy_meta: Mapping[str, Any],
    validate_timestamps: bool = True,
    allow_timestamp_backfill: bool = False,
) -> Dict[str, Any]:
    decision_at = now_iso()

    qualification_id: Optional[str] = None
    action: Optional[str] = None
    requested_state: Optional[str] = None
    candidate_obj = candidate if isinstance(candidate, dict) else {}
    if isinstance(candidate, dict):
        raw_id = candidate.get("qualification_id")
        if not isinstance(raw_id, str):
            raw_id = candidate.get("rollout_id")
        if isinstance(raw_id, str):
            qualification_id = raw_id
        rollout = candidate.get("rollout")
        if isinstance(rollout, dict):
            raw_action = rollout.get("requested_action")
            raw_state = rollout.get("requested_state")
            if isinstance(raw_action, str):
                action = raw_action
            if isinstance(raw_state, str):
                requested_state = raw_state

    # Timestamp validation and backfill
    timestamp_validation = {
        "enabled": validate_timestamps,
        "allow_backfill": allow_timestamp_backfill,
        "was_backfilled": False,
        "backfilled_fields": [],
        "error": None,
    }
    
    if validate_timestamps and isinstance(candidate, dict):
        try:
            candidate_dict = dict(candidate)
            updated_candidate, was_backfilled, backfilled_fields = _ensure_qualification_packet_timestamps(
                candidate_dict, allow_backfill=allow_timestamp_backfill
            )
            timestamp_validation.update({
                "was_backfilled": was_backfilled,
                "backfilled_fields": backfilled_fields,
            })
            if was_backfilled:
                candidate = updated_candidate
                candidate_obj = candidate if isinstance(candidate, dict) else {}
        except Exception as exc:
            timestamp_validation["error"] = str(exc)
            # Fail closed if timestamp validation fails and backfill not allowed
            return _build_timestamp_validation_failure_result(candidate, str(exc))

    required_checklist_ids = _policy_required_checklist_ids(pool_policy)
    benchmark_thresholds = _policy_benchmark_thresholds(pool_policy)
    min_eval_sample_size = _policy_min_eval_sample_size(pool_policy)
    authority_matrix = _policy_lane_authority(pool_policy)
    action_role_allowlist = _policy_action_role_allowlist(pool_policy)
    canary_exposure_max, ring_exposure_max, max_rollback_minutes = _policy_rollout_limits(pool_policy)

    gate_rows: List[Dict[str, Any]] = []
    blocked = False
    block_reason: Optional[str] = None
    block_gate: Optional[str] = None

    # Add timestamp validation gate if enabled
    gate_specs = []
    if validate_timestamps:
        gate_specs.append(("timestamp_validation", lambda: (True, "timestamp_validation_passed", timestamp_validation)))
    
    gate_specs.extend([
        ("schema", lambda: gate_schema(candidate, schema_path)),
        ("source_refs", lambda: gate_source_refs(candidate_obj, repo_root)),
        ("pool_policy", lambda: gate_pool_policy(candidate_obj, pool_policy)),
        (
            "qualification_checklist",
            lambda: gate_qualification_checklist(candidate_obj, required_checklist_ids=required_checklist_ids),
        ),
        (
            "benchmark_thresholds",
            lambda: gate_benchmark_thresholds(
                candidate_obj,
                thresholds=benchmark_thresholds,
                min_eval_sample_size=min_eval_sample_size,
            ),
        ),
        ("replay_evidence", lambda: gate_replay_evidence(candidate_obj, repo_root, now=decision_at)),
        (
            "lane_authority",
            lambda: gate_lane_authority(
                candidate_obj,
                authority_matrix=authority_matrix,
                action_role_allowlist=action_role_allowlist,
            ),
        ),
        (
            "rollout_transition",
            lambda: gate_rollout_transition(
                candidate_obj,
                canary_exposure_max=canary_exposure_max,
                ring_exposure_max=ring_exposure_max,
            ),
        ),
        (
            "rollback_killswitch",
            lambda: gate_rollback_killswitch(candidate_obj, max_rollback_minutes=max_rollback_minutes),
        ),
    ])

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gate_rows.append({"gate": gate_name, "status": "skipped", "reason": "blocked_by_previous_gate"})
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gate_rows.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_reason = reason or "gate_unavailable"
        block_gate = gate_name
        gate_rows.append({"gate": gate_name, "status": "fail", "reason": block_reason, "details": details})

    decision = "BLOCK" if blocked else "PASS"
    if blocked:
        final_state = "BLOCKED"
    elif action == "rollback":
        final_state = "ROLLED_BACK"
    elif action == "kill":
        final_state = "KILLED"
    else:
        final_state = str(requested_state or "APPROVED")

    try:
        candidate_sha = file_sha256(candidate_path)
    except Exception:
        candidate_sha = None

    return {
        "schema": "clawd.model_rollout_gate.decision.v1",
        "evaluated_at": decision_at,
        "decision": decision,
        "final_state": final_state,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "qualification_id": qualification_id,
        "rollout_id": qualification_id,
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
        },
        "model": _decision_model_projection(candidate_obj),
        "rollout": _decision_rollout_projection(candidate_obj, pool_policy),
        "qualification_signal": _decision_qualification_signal(candidate_obj),
        "policy": {
            "policy_id": pool_policy.get("policy_id"),
            "policy_path": pool_policy_meta.get("path"),
            "policy_schema_path": pool_policy_meta.get("schema_path"),
            "required_checklist_ids": required_checklist_ids,
            "benchmark_thresholds": benchmark_thresholds,
            "min_eval_sample_size": min_eval_sample_size,
            "replay_evidence": {
                "required_for_live_promote_states": sorted(LIVE_ROLLOUT_STATES),
                "default_index_path": DEFAULT_REPLAY_EVIDENCE_INDEX,
                "default_max_age_seconds": REPLAY_EVIDENCE_MAX_AGE_SECONDS,
            },
            "authority_matrix": authority_matrix,
            "forward_transitions": FORWARD_TRANSITIONS,
            "canary_exposure_max": canary_exposure_max,
            "ring_exposure_max": ring_exposure_max,
            "max_rollback_minutes": max_rollback_minutes,
        },
        "gates": gate_rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic model qualification + rollout gate runner (v1)")
    ap.add_argument("--candidate", default=None, help="Path to qualification packet/candidate JSON")
    ap.add_argument("--packet", default=None, help="Alias for --candidate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root for relative path resolution")
    ap.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to model qualification packet JSON schema",
    )
    ap.add_argument(
        "--pool-policy",
        default=str(DEFAULT_POOL_POLICY_PATH),
        help="Unified model pool policy JSON path",
    )
    ap.add_argument(
        "--pool-policy-schema",
        default=str(DEFAULT_POOL_POLICY_SCHEMA_PATH),
        help="Unified model pool policy schema path",
    )
    ap.add_argument(
        "--decision-log",
        default=str(DEFAULT_DECISION_LOG),
        help="Append-only decision log path (relative to repo root unless absolute)",
    )
    ap.add_argument("--no-decision-log", action="store_true", help="Disable append-only decision recording")
    ap.add_argument(
        "--decision-out",
        default=None,
        help="Optional write path for decision artifact JSON (relative to repo root unless absolute)",
    )
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON output")
    ap.add_argument(
        "--validate-timestamps",
        action="store_true",
        default=True,
        help="Validate qualification packet timestamp completeness (default: True)",
    )
    ap.add_argument(
        "--no-validate-timestamps",
        action="store_false",
        dest="validate_timestamps",
        help="Disable timestamp validation",
    )
    ap.add_argument(
        "--allow-timestamp-backfill",
        action="store_true",
        default=False,
        help="Allow automatic backfill of missing timestamps (default: False)",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()

    pool_policy_path_raw = Path(args.pool_policy).expanduser()
    pool_policy_path = pool_policy_path_raw if pool_policy_path_raw.is_absolute() else (repo_root / pool_policy_path_raw).resolve()
    pool_policy_schema_raw = Path(args.pool_policy_schema).expanduser()
    pool_policy_schema_path = (
        pool_policy_schema_raw if pool_policy_schema_raw.is_absolute() else (repo_root / pool_policy_schema_raw).resolve()
    )

    candidate_arg = args.candidate or args.packet
    if not candidate_arg:
        print("error: one of --candidate or --packet is required", file=sys.stderr)
        return 2
    candidate_path = Path(candidate_arg).expanduser().resolve()

    pool_policy_ok, pool_policy_reason, pool_policy_details, pool_policy_doc = load_pool_policy(
        policy_path=pool_policy_path,
        policy_schema_path=pool_policy_schema_path,
    )
    if not pool_policy_ok:
        result = {
            "schema": "clawd.model_rollout_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "pool_policy",
            "block_reason": pool_policy_reason,
            "qualification_id": None,
            "rollout_id": None,
            "candidate": {
                "path": str(candidate_path),
                "sha256": None,
            },
            "policy": {
                "policy_path": str(pool_policy_path),
                "policy_schema_path": str(pool_policy_schema_path),
            },
            "gates": [
                {
                    "gate": "schema",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "source_refs",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "pool_policy",
                    "status": "fail",
                    "reason": pool_policy_reason,
                    "details": pool_policy_details,
                },
                {
                    "gate": "qualification_checklist",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "benchmark_thresholds",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "replay_evidence",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "lane_authority",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "rollout_transition",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
                {
                    "gate": "rollback_killswitch",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
            ],
        }
    else:
        try:
            candidate_doc = load_json_file(candidate_path)
        except Exception as exc:
            result = {
                "schema": "clawd.model_rollout_gate.decision.v1",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "final_state": "BLOCKED",
                "block_gate": "schema",
                "block_reason": "schema_invalid",
                "qualification_id": None,
                "rollout_id": None,
                "candidate": {
                    "path": str(candidate_path),
                    "sha256": None,
                },
                "policy": {
                    "policy_id": pool_policy_doc.get("policy_id") if isinstance(pool_policy_doc, dict) else None,
                    "policy_path": str(pool_policy_path),
                    "policy_schema_path": str(pool_policy_schema_path),
                },
                "gates": [
                    {
                        "gate": "schema",
                        "status": "fail",
                        "reason": "schema_invalid",
                        "details": {
                            "error": "candidate_json_unreadable",
                            "detail": str(exc),
                        },
                    },
                    {"gate": "source_refs", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "pool_policy", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "qualification_checklist", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "benchmark_thresholds", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "replay_evidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "lane_authority", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "rollout_transition", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "rollback_killswitch", "status": "skipped", "reason": "blocked_by_previous_gate"},
                ],
            }
        else:
            result = evaluate_candidate(
                candidate=candidate_doc,
                candidate_path=candidate_path,
                repo_root=repo_root,
                schema_path=schema_path,
                pool_policy=pool_policy_doc if isinstance(pool_policy_doc, dict) else {},
                pool_policy_meta=pool_policy_details,
                validate_timestamps=args.validate_timestamps,
                allow_timestamp_backfill=args.allow_timestamp_backfill,
            )

    decision_log_path: Optional[Path] = None
    if not args.no_decision_log:
        decision_log_path = Path(args.decision_log).expanduser()

    record = append_decision_record(decision_log_path=decision_log_path, repo_root=repo_root, decision_row=result)
    result["decision_record"] = record

    decision_out_path: Optional[Path] = None
    if args.decision_out:
        decision_out_path = Path(args.decision_out).expanduser()
    decision_artifact = write_decision_artifact(decision_out_path=decision_out_path, repo_root=repo_root, payload=result)
    result["decision_artifact"] = decision_artifact

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
