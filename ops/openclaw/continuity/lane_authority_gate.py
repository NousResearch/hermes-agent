#!/usr/bin/env python3
"""Deterministic lane-authority mutation gate evaluator.

This helper consumes lane_topology_authority_contract.v1 and evaluates whether
one mutation attempt should be allowed under staged enforcement policy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ALLOWED_RISK_TIERS = {"low", "medium", "high", "critical"}
ATTESTATION_SCHEMA_VERSION = "lane.mutation_attestation.v1"
ACTION_INTENT_SCHEMA_VERSION = "lane.action_intent.v1"
ALLOWED_OVERRIDE_MODES = {"normal", "stop", "read_only", "drain"}
ALLOWED_MUTATION_CLASSES = {"none", "read_only", "drain", "mutating"}

DEFAULT_ATTESTATION_OBJECT_POLICY: Dict[str, Any] = {
    "schema_version": ATTESTATION_SCHEMA_VERSION,
    "required_fields": [
        "attestation_name",
        "status",
        "issued_at",
        "evidence_ref",
        "integrity_hash",
        "operation_id",
    ],
    "require_pass_status": True,
    "require_operation_match": True,
    "reject_expired": True,
}

DEFAULT_OVERRIDE_GOVERNANCE_POLICY: Dict[str, Any] = {
    "mode": "normal",
    "enforce_at_actuation_boundary": True,
    "require_action_intent": True,
    "intent_schema_version": ACTION_INTENT_SCHEMA_VERSION,
    "allowed_mutation_classes_by_mode": {
        "normal": ["none", "read_only", "drain", "mutating"],
        "stop": ["none"],
        "read_only": ["none", "read_only"],
        "drain": ["none", "read_only", "drain"],
    },
}


def _parse_iso(raw: str) -> Optional[dt.datetime]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _now_utc(now_override: str) -> dt.datetime:
    if now_override:
        parsed = _parse_iso(now_override)
        if parsed is None:
            raise ValueError(f"invalid_now:{now_override}")
        return parsed
    return dt.datetime.now(dt.timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("contract_top_level_not_object")
    return data


def _parse_json_object_input(
    raw: str,
    *,
    missing_code: str,
    path_missing_prefix: str,
    parse_failed_prefix: str,
    not_object_code: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    txt = str(raw or "").strip()
    if not txt:
        return None, missing_code, None

    source: Optional[str] = "inline_json"
    if txt.startswith("@"):
        p = Path(txt[1:]).expanduser()
        if not p.exists():
            return None, f"{path_missing_prefix}:{p}", None
        txt = p.read_text(encoding="utf-8")
        source = str(p)
    else:
        p = Path(txt).expanduser()
        if p.exists() and p.is_file():
            txt = p.read_text(encoding="utf-8")
            source = str(p)

    try:
        obj = json.loads(txt)
    except Exception as exc:
        return None, f"{parse_failed_prefix}:{exc}", source

    if not isinstance(obj, dict):
        return None, not_object_code, source
    return obj, None, source


def _parse_ticket(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    obj, err, _ = _parse_json_object_input(
        raw,
        missing_code="ticket_missing",
        path_missing_prefix="ticket_path_missing",
        parse_failed_prefix="ticket_parse_failed",
        not_object_code="ticket_not_object",
    )
    return obj, err


def _parse_attestation_object(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    return _parse_json_object_input(
        raw,
        missing_code="attestation_object_missing",
        path_missing_prefix="attestation_object_path_missing",
        parse_failed_prefix="attestation_object_parse_failed",
        not_object_code="attestation_object_not_object",
    )


def _parse_action_intent(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    return _parse_json_object_input(
        raw,
        missing_code="action_intent_missing",
        path_missing_prefix="action_intent_path_missing",
        parse_failed_prefix="action_intent_parse_failed",
        not_object_code="action_intent_not_object",
    )


def _collect_attestations(args_attestations: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for value in args_attestations:
        token = str(value or "").strip()
        if token:
            out.add(token)

    env_raw = str(os.environ.get("OPENCLAW_MUTATION_ATTESTATIONS", "") or "").strip()
    if env_raw:
        normalized = env_raw.replace(";", ",").replace("|", ",")
        for part in normalized.split(","):
            token = part.strip()
            if token:
                out.add(token)
    return out


def _collect_attestation_objects(args_attestation_objects: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in args_attestation_objects:
        token = str(raw or "").strip()
        if token:
            out.append(token)

    env_raw = str(os.environ.get("OPENCLAW_MUTATION_ATTESTATION_OBJECTS", "") or "").strip()
    if not env_raw:
        return out

    # Allow JSON list envelope for env usage.
    if env_raw.startswith("["):
        try:
            arr = json.loads(env_raw)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, str):
                        token = item.strip()
                        if token:
                            out.append(token)
                    elif isinstance(item, dict):
                        out.append(json.dumps(item, ensure_ascii=False))
                return out
        except Exception:
            pass

    normalized = env_raw.replace("\n", ",").replace(";", ",").replace("|", ",")
    for part in normalized.split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _normalize_override_governance_policy(raw_policy: Any) -> Dict[str, Any]:
    policy = dict(DEFAULT_OVERRIDE_GOVERNANCE_POLICY)
    allowed_map = {
        str(k): list(v)
        for k, v in (DEFAULT_OVERRIDE_GOVERNANCE_POLICY.get("allowed_mutation_classes_by_mode") or {}).items()
    }

    if isinstance(raw_policy, dict):
        mode = str(raw_policy.get("mode") or "").strip()
        if mode in ALLOWED_OVERRIDE_MODES:
            policy["mode"] = mode

        for key in ("enforce_at_actuation_boundary", "require_action_intent"):
            if key in raw_policy:
                policy[key] = bool(raw_policy.get(key) is True)

        intent_schema_version = str(raw_policy.get("intent_schema_version") or "").strip()
        if intent_schema_version:
            policy["intent_schema_version"] = intent_schema_version

        raw_allowed = raw_policy.get("allowed_mutation_classes_by_mode")
        if isinstance(raw_allowed, dict):
            for mode_key, values in raw_allowed.items():
                mode_name = str(mode_key or "").strip()
                if mode_name not in ALLOWED_OVERRIDE_MODES:
                    continue
                if not isinstance(values, list):
                    continue
                cleaned = [
                    str(v or "").strip()
                    for v in values
                    if str(v or "").strip() in ALLOWED_MUTATION_CLASSES
                ]
                if cleaned:
                    # Preserve deterministic order by mode-specific baseline ordering.
                    baseline = [
                        item
                        for item in ["none", "read_only", "drain", "mutating"]
                        if item in set(cleaned)
                    ]
                    allowed_map[mode_name] = baseline

    policy["allowed_mutation_classes_by_mode"] = allowed_map
    return policy


def _evaluate_override_posture(
    *,
    override_policy: Dict[str, Any],
    action_intent_raw: str,
    mutation_operation: str,
    now_utc: dt.datetime,
) -> Dict[str, Any]:
    mode = str(override_policy.get("mode") or "normal").strip()
    if mode not in ALLOWED_OVERRIDE_MODES:
        mode = "normal"

    enforce = bool(override_policy.get("enforce_at_actuation_boundary") is True)
    require_action_intent = bool(override_policy.get("require_action_intent") is True)
    expected_schema = str(override_policy.get("intent_schema_version") or ACTION_INTENT_SCHEMA_VERSION).strip()
    allowed_map = override_policy.get("allowed_mutation_classes_by_mode")
    if not isinstance(allowed_map, dict):
        allowed_map = DEFAULT_OVERRIDE_GOVERNANCE_POLICY["allowed_mutation_classes_by_mode"]
    action_intent_required_now = require_action_intent and mode != "normal"

    result: Dict[str, Any] = {
        "mode": mode,
        "enforce_at_actuation_boundary": enforce,
        "require_action_intent": require_action_intent,
        "action_intent_required_now": action_intent_required_now,
        "expected_intent_schema_version": expected_schema,
        "allowed_mutation_classes": list(allowed_map.get(mode) or []),
        "action_intent_provided": bool(str(action_intent_raw or "").strip()),
        "status": "PASS",
    }

    if not enforce:
        result["status"] = "BYPASS"
        result["decision"] = "override_boundary_not_enforced"
        return result

    action_intent_obj: Optional[Dict[str, Any]] = None
    action_intent_source: Optional[str] = None
    action_intent_error: Optional[str] = None

    if action_intent_required_now:
        action_intent_obj, action_intent_error, action_intent_source = _parse_action_intent(action_intent_raw)
        if action_intent_error is not None or action_intent_obj is None:
            result.update(
                {
                    "status": "BLOCK",
                    "decision": "missing_or_invalid_action_intent",
                    "rejection_code": "action_intent_missing" if action_intent_error == "action_intent_missing" else "action_intent_invalid",
                    "error": action_intent_error,
                }
            )
            return result
    elif str(action_intent_raw or "").strip():
        action_intent_obj, action_intent_error, action_intent_source = _parse_action_intent(action_intent_raw)
        if action_intent_error is not None or action_intent_obj is None:
            result.update(
                {
                    "status": "BLOCK",
                    "decision": "invalid_action_intent",
                    "rejection_code": "action_intent_invalid",
                    "error": action_intent_error,
                }
            )
            return result

    if action_intent_obj is None:
        # Allowed only when policy explicitly does not require an action-intent.
        result["decision"] = "action_intent_optional_and_absent"
        return result

    schema_version = str(action_intent_obj.get("schema_version") or "").strip()
    intent_id = str(action_intent_obj.get("intent_id") or "").strip()
    operation_id = str(action_intent_obj.get("operation_id") or "").strip()
    mutation_class = str(action_intent_obj.get("mutation_class") or "").strip()
    issued_at = _parse_iso(str(action_intent_obj.get("issued_at") or ""))
    expires_at = _parse_iso(str(action_intent_obj.get("expires_at") or "")) if str(action_intent_obj.get("expires_at") or "").strip() else None

    result["action_intent"] = {
        "source": action_intent_source,
        "intent_id": intent_id or None,
        "operation_id": operation_id or None,
        "mutation_class": mutation_class or None,
        "issued_at": issued_at.replace(microsecond=0).isoformat().replace("+00:00", "Z") if issued_at is not None else None,
        "expires_at": expires_at.replace(microsecond=0).isoformat().replace("+00:00", "Z") if expires_at is not None else None,
    }

    if schema_version != expected_schema:
        result.update(
            {
                "status": "BLOCK",
                "decision": "action_intent_schema_mismatch",
                "rejection_code": "action_intent_invalid",
                "error": "action_intent_schema_mismatch",
            }
        )
        return result

    if not intent_id or not operation_id or mutation_class not in ALLOWED_MUTATION_CLASSES or issued_at is None:
        result.update(
            {
                "status": "BLOCK",
                "decision": "action_intent_missing_required_fields",
                "rejection_code": "action_intent_invalid",
                "error": "action_intent_required_fields_invalid",
            }
        )
        return result

    if expires_at is not None and now_utc > expires_at:
        result.update(
            {
                "status": "BLOCK",
                "decision": "action_intent_expired",
                "rejection_code": "action_intent_invalid",
                "error": "action_intent_expired",
            }
        )
        return result

    if mutation_operation and operation_id != mutation_operation:
        result.update(
            {
                "status": "BLOCK",
                "decision": "action_intent_operation_mismatch",
                "rejection_code": "action_intent_mismatch",
                "error": "action_intent_operation_mismatch",
                "required_operation_id": mutation_operation,
                "action_intent_operation_id": operation_id,
            }
        )
        return result

    allowed_mutation_classes = {
        str(value or "").strip()
        for value in (allowed_map.get(mode) or [])
        if str(value or "").strip()
    }
    if mutation_class not in allowed_mutation_classes:
        rejection_by_mode = {
            "stop": "override_stop_active",
            "read_only": "override_read_only_violation",
            "drain": "override_drain_violation",
        }
        result.update(
            {
                "status": "BLOCK",
                "decision": "override_mode_violation",
                "rejection_code": rejection_by_mode.get(mode, "override_read_only_violation"),
                "error": f"override_mode_{mode}_disallows_{mutation_class}",
            }
        )
        return result

    result["decision"] = "override_posture_pass"
    return result


def _normalize_attestation_object_policy(raw_policy: Any) -> Dict[str, Any]:
    policy = dict(DEFAULT_ATTESTATION_OBJECT_POLICY)
    if isinstance(raw_policy, dict):
        schema_version = str(raw_policy.get("schema_version") or "").strip()
        if schema_version:
            policy["schema_version"] = schema_version

        required_fields = [
            str(x or "").strip()
            for x in (raw_policy.get("required_fields") or [])
            if str(x or "").strip()
        ]
        if required_fields:
            policy["required_fields"] = required_fields

        for key in ("require_pass_status", "require_operation_match", "reject_expired"):
            if key in raw_policy:
                policy[key] = bool(raw_policy.get(key) is True)
    return policy


def _evaluate_attestation_objects(
    *,
    attestation_object_raws: Sequence[str],
    policy: Dict[str, Any],
    now_utc: dt.datetime,
    mutation_operation: str,
) -> Dict[str, Any]:
    required_fields = {
        str(x or "").strip()
        for x in (policy.get("required_fields") or [])
        if str(x or "").strip()
    }
    require_pass_status = bool(policy.get("require_pass_status") is True)
    require_operation_match = bool(policy.get("require_operation_match") is True)
    reject_expired = bool(policy.get("reject_expired") is True)
    expected_schema = str(policy.get("schema_version") or "").strip() or ATTESTATION_SCHEMA_VERSION

    satisfied_attestations: Set[str] = set()
    accepted_objects: List[Dict[str, Any]] = []
    rejected_objects: List[Dict[str, Any]] = []

    for raw in attestation_object_raws:
        obj, err, source = _parse_attestation_object(raw)
        if err is not None or obj is None:
            rejected_objects.append(
                {
                    "source": source,
                    "error": err or "attestation_object_parse_failed",
                }
            )
            continue

        att_name = str(obj.get("attestation_name") or "").strip()
        status = str(obj.get("status") or "").strip().lower()
        operation_id = str(obj.get("operation_id") or "").strip()
        schema_version = str(obj.get("schema_version") or "").strip()

        missing_fields = [
            field
            for field in sorted(required_fields)
            if not str(obj.get(field) or "").strip()
        ]
        if missing_fields:
            rejected_objects.append(
                {
                    "source": source,
                    "attestation_name": att_name or None,
                    "error": "attestation_object_missing_fields",
                    "missing_fields": missing_fields,
                }
            )
            continue

        if expected_schema and schema_version != expected_schema:
            rejected_objects.append(
                {
                    "source": source,
                    "attestation_name": att_name or None,
                    "error": "attestation_object_schema_mismatch",
                    "expected_schema_version": expected_schema,
                    "actual_schema_version": schema_version or None,
                }
            )
            continue

        issued = _parse_iso(str(obj.get("issued_at") or ""))
        if issued is None:
            rejected_objects.append(
                {
                    "source": source,
                    "attestation_name": att_name or None,
                    "error": "attestation_object_time_invalid",
                    "field": "issued_at",
                }
            )
            continue

        expires_raw = str(obj.get("expires_at") or "").strip()
        expires: Optional[dt.datetime] = None
        if expires_raw:
            expires = _parse_iso(expires_raw)
            if expires is None:
                rejected_objects.append(
                    {
                        "source": source,
                        "attestation_name": att_name or None,
                        "error": "attestation_object_time_invalid",
                        "field": "expires_at",
                    }
                )
                continue

        if reject_expired and expires is not None and now_utc > expires:
            rejected_objects.append(
                {
                    "source": source,
                    "attestation_name": att_name or None,
                    "error": "attestation_object_expired",
                    "expires_at": expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                }
            )
            continue

        if require_pass_status and status != "pass":
            rejected_objects.append(
                {
                    "source": source,
                    "attestation_name": att_name or None,
                    "error": "attestation_object_not_pass",
                    "status": status or None,
                }
            )
            continue

        if require_operation_match and mutation_operation:
            if operation_id != mutation_operation:
                rejected_objects.append(
                    {
                        "source": source,
                        "attestation_name": att_name or None,
                        "error": "attestation_object_operation_mismatch",
                        "required_operation_id": mutation_operation,
                        "attestation_operation_id": operation_id or None,
                    }
                )
                continue

        if not att_name:
            rejected_objects.append(
                {
                    "source": source,
                    "error": "attestation_object_name_missing",
                }
            )
            continue

        satisfied_attestations.add(att_name)
        accepted_objects.append(
            {
                "source": source,
                "attestation_name": att_name,
                "evidence_ref": str(obj.get("evidence_ref") or "").strip() or None,
                "operation_id": operation_id or None,
                "issued_at": issued.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "expires_at": (
                    expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    if expires is not None
                    else None
                ),
            }
        )

    return {
        "satisfied_attestations": sorted(satisfied_attestations),
        "accepted_objects": accepted_objects,
        "rejected_objects": rejected_objects,
    }


def evaluate_gate(
    contract: Dict[str, Any],
    *,
    risk_tier: str,
    mutation_ticket_raw: str,
    action_intent_raw: str,
    mutation_operation: str,
    attestation_names: Set[str],
    attestation_object_raws: Sequence[str],
    now_utc: dt.datetime,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "schema": "clawd.lane_authority_gate.decision.v1",
        "risk_tier": risk_tier,
        "evaluated_at": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "accepted": False,
        "rejection_codes": [],
    }

    try:
        policy = contract.get("mutation_gateway_policy") or {}
        ticket_policy = policy.get("ticket_policy") or {}
        override_governance = _normalize_override_governance_policy(contract.get("override_governance"))
        jeopardy_policy = contract.get("jeopardy_policy") or {}
        authority_leases = contract.get("authority_leases") or {}
        control_lease = authority_leases.get("control_lease") or {}
        workflow_lease = authority_leases.get("workflow_lease") or {}
        attestation_object_policy = _normalize_attestation_object_policy(policy.get("attestation_object_policy"))
    except Exception:
        out["rejection_codes"] = ["lease_state_unknown"]
        out["error"] = "contract_shape_invalid"
        return out

    mode = str(policy.get("mode") or "audit").strip()
    high_risk_posture = str(policy.get("high_risk_posture") or "audit_only").strip()
    required_risk_tiers = {
        str(x or "").strip()
        for x in (ticket_policy.get("required_for_risk_tiers") or [])
        if str(x or "").strip()
    }

    if mode == "enforce_all":
        enforcement_active = True
    elif mode == "enforce_high_risk":
        enforcement_active = risk_tier in required_risk_tiers
    else:
        enforcement_active = False

    out["enforcement"] = {
        "mode": mode,
        "active": enforcement_active,
        "high_risk_posture": high_risk_posture,
    }

    override_eval = _evaluate_override_posture(
        override_policy=override_governance,
        action_intent_raw=action_intent_raw,
        mutation_operation=mutation_operation,
        now_utc=now_utc,
    )
    out["override_posture"] = override_eval
    if str(override_eval.get("status") or "") == "BLOCK":
        rejection_code = str(override_eval.get("rejection_code") or "action_intent_invalid")
        out["rejection_codes"] = [rejection_code]
        out["error"] = str(override_eval.get("error") or "override_posture_blocked")
        if override_eval.get("required_operation_id") is not None:
            out["required_operation_id"] = override_eval.get("required_operation_id")
        if override_eval.get("action_intent_operation_id") is not None:
            out["action_intent_operation_id"] = override_eval.get("action_intent_operation_id")
        return out

    # In audit mode or non-targeted risk tier, do not block.
    if not enforcement_active:
        out["ok"] = True
        out["accepted"] = True
        out["decision_reason"] = "enforcement_inactive"
        out["ticket_required"] = False
        return out

    fail_closed = bool(jeopardy_policy.get("fail_closed_on_lease_uncertainty") is True)

    def _lease_state(lease_obj: Dict[str, Any]) -> str:
        return str(lease_obj.get("status") or "").strip()

    lease_states = {
        "control_lease_status": _lease_state(control_lease),
        "workflow_lease_status": _lease_state(workflow_lease),
    }
    out["lease_states"] = lease_states

    def _lease_detail(lease_obj: Dict[str, Any], label: str) -> Dict[str, Any]:
        status = str(lease_obj.get("status") or "").strip()
        jeopardy_reason = str(lease_obj.get("jeopardy_reason") or "").strip() or None
        try:
            fencing_term = int(lease_obj.get("fencing_term"))
        except Exception:
            fencing_term = None
        return {
            "label": label,
            "lease_id": str(lease_obj.get("lease_id") or "").strip() or None,
            "status": status or None,
            "jeopardy_reason": jeopardy_reason,
            "fencing_term": fencing_term,
        }

    control_detail = _lease_detail(control_lease, "control_lease")
    workflow_detail = _lease_detail(workflow_lease, "workflow_lease")
    out["lease_evaluation"] = {
        "control_lease": control_detail,
        "workflow_lease": workflow_detail,
    }

    lease_uncertain = (
        not lease_states["control_lease_status"]
        or not lease_states["workflow_lease_status"]
        or lease_states["control_lease_status"] != "active"
        or lease_states["workflow_lease_status"] != "active"
    )

    if fail_closed and lease_uncertain:
        lease_block_reason = "lease_not_active"
        jeopardy_notes: List[Dict[str, Any]] = []
        for detail in (control_detail, workflow_detail):
            if str(detail.get("status") or "") == "jeopardy":
                jeopardy_notes.append(
                    {
                        "lease": detail.get("label"),
                        "reason": detail.get("jeopardy_reason") or "unspecified",
                    }
                )
        if jeopardy_notes:
            lease_block_reason = "lease_jeopardy"
            out["lease_jeopardy"] = jeopardy_notes
        out["lease_block_reason"] = lease_block_reason
        out["rejection_codes"] = ["lease_state_unknown"]
        out["error"] = "lease_fail_close"
        return out

    ticket_required = high_risk_posture == "ticket_required"
    out["ticket_required"] = ticket_required

    required_attestations = {
        str(x or "").strip()
        for x in (policy.get("required_attestations") or [])
        if str(x or "").strip()
    }

    attestation_obj_eval = _evaluate_attestation_objects(
        attestation_object_raws=attestation_object_raws,
        policy=attestation_object_policy,
        now_utc=now_utc,
        mutation_operation=mutation_operation,
    )

    satisfied_attestations = set(attestation_names)
    satisfied_attestations.update(attestation_obj_eval.get("satisfied_attestations") or [])

    out["attestation_summary"] = {
        "legacy_attestation_count": len(attestation_names),
        "attestation_object_count": len(attestation_object_raws),
        "accepted_attestation_object_count": len(attestation_obj_eval.get("accepted_objects") or []),
        "rejected_attestation_object_count": len(attestation_obj_eval.get("rejected_objects") or []),
    }

    if attestation_obj_eval.get("accepted_objects"):
        out["accepted_attestation_objects"] = attestation_obj_eval["accepted_objects"]
    if attestation_obj_eval.get("rejected_objects"):
        out["rejected_attestation_objects"] = attestation_obj_eval["rejected_objects"]

    missing_attestations = sorted(required_attestations.difference(satisfied_attestations))
    if missing_attestations:
        out["rejection_codes"] = ["attestation_missing"]
        out["missing_attestations"] = missing_attestations
        return out

    if not ticket_required:
        out["ok"] = True
        out["accepted"] = True
        out["decision_reason"] = "ticket_not_required"
        return out

    ticket_obj, ticket_err = _parse_ticket(mutation_ticket_raw)
    if ticket_err is not None or ticket_obj is None:
        out["rejection_codes"] = ["ticket_missing"]
        out["ticket_error"] = ticket_err
        return out

    required_ticket_fields = [
        str(x or "").strip()
        for x in (ticket_policy.get("required_ticket_fields") or [])
        if str(x or "").strip()
    ]
    missing_ticket_fields = [k for k in required_ticket_fields if not str(ticket_obj.get(k) or "").strip()]
    if missing_ticket_fields:
        out["rejection_codes"] = ["ticket_missing"]
        out["missing_ticket_fields"] = missing_ticket_fields
        return out

    if mutation_operation:
        ticket_operation = str(ticket_obj.get("operation_id") or "").strip()
        if ticket_operation != mutation_operation:
            out["rejection_codes"] = ["ticket_missing"]
            out["ticket_error"] = "operation_mismatch"
            out["ticket_operation_id"] = ticket_operation or None
            out["required_operation_id"] = mutation_operation
            return out

    issued = _parse_iso(str(ticket_obj.get("issued_at") or ""))
    expires = _parse_iso(str(ticket_obj.get("expires_at") or ""))
    if issued is None or expires is None:
        out["rejection_codes"] = ["ticket_missing"]
        out["ticket_error"] = "ticket_time_invalid"
        return out

    if now_utc > expires:
        out["rejection_codes"] = ["ticket_expired"]
        out["ticket_error"] = "ticket_expired"
        return out

    ttl_seconds_max = int(ticket_policy.get("ttl_seconds_max") or 0)
    ttl_seconds = int((expires - issued).total_seconds())
    if ttl_seconds_max > 0 and ttl_seconds > ttl_seconds_max:
        out["rejection_codes"] = ["ticket_expired"]
        out["ticket_error"] = "ticket_ttl_exceeded"
        out["ticket_ttl_seconds"] = ttl_seconds
        out["ticket_ttl_seconds_max"] = ttl_seconds_max
        return out

    reject_stale_fencing_term = bool(ticket_policy.get("reject_stale_fencing_term") is True)
    if reject_stale_fencing_term:
        try:
            ticket_term = int(ticket_obj.get("fencing_term"))
            control_term = int(control_lease.get("fencing_term"))
            workflow_term = int(workflow_lease.get("fencing_term"))
        except Exception:
            out["rejection_codes"] = ["fencing_term_stale"]
            out["ticket_error"] = "fencing_term_parse_failed"
            return out

        lease_term_floor = max(control_term, workflow_term)
        if ticket_term < lease_term_floor:
            out["rejection_codes"] = ["fencing_term_stale"]
            out["ticket_fencing_term"] = ticket_term
            out["required_fencing_term_min"] = lease_term_floor
            return out

    out["ok"] = True
    out["accepted"] = True
    out["decision_reason"] = "ticket_and_attestations_verified"
    out["ticket_id"] = str(ticket_obj.get("ticket_id") or "").strip() or None
    return out


def _render(payload: Dict[str, Any], *, json_out: bool) -> None:
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    status = "PASS" if payload.get("accepted") else "BLOCK"
    reason = str(payload.get("decision_reason") or payload.get("error") or "").strip()
    suffix = f" ({reason})" if reason else ""
    print(f"{status}: lane_authority_gate{suffix}")


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate lane-topology authority gate for one mutation attempt")
    ap.add_argument(
        "--contract",
        default=os.environ.get(
            "OPENCLAW_LANE_AUTHORITY_CONTRACT_PATH",
            "/home/yeqiuqiu/clawd-architect/docs/ops/templates/lane_topology_authority_contract.template.json",
        ),
        help="Path to lane authority contract JSON",
    )
    ap.add_argument("--risk-tier", default="low", choices=sorted(ALLOWED_RISK_TIERS))
    ap.add_argument("--mutation-ticket", default="", help="Ticket JSON string, @path, or path")
    ap.add_argument("--action-intent", default="", help="Action-intent JSON string, @path, or path")
    ap.add_argument("--mutation-operation", default="", help="Expected operation_id (optional)")
    ap.add_argument("--attestation", action="append", default=[], help="Satisfied attestation name (repeatable)")
    ap.add_argument(
        "--attestation-object",
        action="append",
        default=[],
        help="Structured attestation JSON string, @path, or path (repeatable)",
    )
    ap.add_argument("--now", default="", help="Override current time (ISO8601)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    contract_path = Path(str(args.contract or "").strip()).expanduser()
    if not contract_path.exists():
        payload = {
            "ok": False,
            "schema": "clawd.lane_authority_gate.decision.v1",
            "accepted": False,
            "rejection_codes": ["lease_state_unknown"],
            "error": "authority_contract_missing",
            "contract": str(contract_path),
        }
        _render(payload, json_out=args.json)
        return 1

    try:
        now_utc = _now_utc(str(args.now or ""))
    except Exception as exc:
        payload = {
            "ok": False,
            "schema": "clawd.lane_authority_gate.decision.v1",
            "accepted": False,
            "rejection_codes": ["lease_state_unknown"],
            "error": str(exc),
        }
        _render(payload, json_out=args.json)
        return 2

    try:
        contract = _load_json(contract_path)
    except Exception as exc:
        payload = {
            "ok": False,
            "schema": "clawd.lane_authority_gate.decision.v1",
            "accepted": False,
            "rejection_codes": ["lease_state_unknown"],
            "error": f"authority_contract_parse_failed:{exc}",
            "contract": str(contract_path),
        }
        _render(payload, json_out=args.json)
        return 1

    payload = evaluate_gate(
        contract,
        risk_tier=str(args.risk_tier or "").strip(),
        mutation_ticket_raw=str(args.mutation_ticket or "").strip() or str(os.environ.get("OPENCLAW_MUTATION_TICKET", "") or "").strip(),
        action_intent_raw=str(args.action_intent or "").strip() or str(os.environ.get("OPENCLAW_ACTION_INTENT", "") or "").strip(),
        mutation_operation=str(args.mutation_operation or "").strip(),
        attestation_names=_collect_attestations(args.attestation),
        attestation_object_raws=_collect_attestation_objects(args.attestation_object),
        now_utc=now_utc,
    )
    _render(payload, json_out=args.json)
    return 0 if payload.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
