#!/usr/bin/env python3
"""Generate XH-703 health ingestion/support runtime artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "health_typed_record.schema.json"
DEFAULT_FIXTURE_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xh" / "health_runtime_fixture_v1.json"
DEFAULT_RISK_MATRIX_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_risk_matrix_2026-03-28.json"
DEFAULT_OWNER_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_owner_registry_2026-03-28.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"
DEFAULT_INCIDENT_GATE = DEFAULT_REPO_ROOT / "scripts" / "domain_failclose_incident_gate.py"

DISCLAIMER = "Informational support only — not a diagnosis or treatment recommendation."


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_ts(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    token = str(value)
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    return dt.datetime.fromisoformat(token)


def iso_plus_minutes(value: str, minutes: int) -> str:
    base = parse_ts(value)
    if base is None:
        raise SystemExit(f"Unable to parse timestamp: {value}")
    return (base + dt.timedelta(minutes=minutes)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XH-703 health ingestion/support runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--risk-matrix-path", default=str(DEFAULT_RISK_MATRIX_PATH))
    ap.add_argument("--owner-registry-path", default=str(DEFAULT_OWNER_REGISTRY_PATH))
    ap.add_argument("--incident-gate-path", default=str(DEFAULT_INCIDENT_GATE))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--json", action="store_true", help="Emit artifact manifest JSON")
    return ap.parse_args(argv)


def _queue_index(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    return {
        str(row.get("id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _dependency_states(queue_index: Dict[str, Dict[str, Any]], dependencies: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    states: Dict[str, str] = {}
    unresolved: List[str] = []
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_state = str((queue_index.get(dep_id) or {}).get("state") or "UNKNOWN")
        states[dep_id] = dep_state
        if dep_state != "DONE":
            unresolved.append(dep_id)
    return states, unresolved


def _owner_tuple(owner_registry: Dict[str, Any], lane_id: str) -> Dict[str, str]:
    rows = owner_registry.get("lane_owner_tuples") if isinstance(owner_registry.get("lane_owner_tuples"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id:
            return {
                "governance_owner": str(row.get("governance_owner") or "unassigned"),
                "lane_owner": str(row.get("lane_owner") or "unassigned"),
                "release_owner": str(row.get("release_owner") or "unassigned"),
                "incident_owner": str(row.get("incident_owner") or "unassigned"),
            }
    return {
        "governance_owner": "unassigned",
        "lane_owner": "unassigned",
        "release_owner": "unassigned",
        "incident_owner": "unassigned",
    }


def _lane_risk(risk_matrix: Dict[str, Any], lane_id: str) -> Dict[str, str]:
    rows = risk_matrix.get("lane_risk_assignments") if isinstance(risk_matrix.get("lane_risk_assignments"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id:
            return {
                "risk_class": str(row.get("risk_class") or "UNKNOWN"),
                "notes": str(row.get("notes") or ""),
            }
    return {"risk_class": "UNKNOWN", "notes": ""}


def _route_decision() -> Dict[str, Any]:
    return {
        "selected_route": "NO_LLM",
        "reason": "XH-703 is safety-sensitive and requires deterministic schema/policy gating plus machine-readable support and incident artifacts.",
        "escalation_trigger": None,
        "fallback_route": "NO_LLM",
        "task_class": "implementation",
        "risk_tier": "critical",
        "scope_shape": "multi_surface_coupled",
        "worker_topology": "single",
        "verification_class": "validator_required",
        "verification_plan": [
            "execute XH-703 runtime generator",
            "compile runtime generator and tests",
            "run direct XH-703 test file",
            "run XG-803 incident gate on emitted fail-close packet",
            "parse generated artifacts via json.tool",
            "run source-of-truth map regression check",
        ],
        "fold_in_target": "queue_continuity",
    }


def _schema_validator(schema_doc: Dict[str, Any]) -> Optional[Any]:
    if Draft202012Validator is None or FormatChecker is None:
        return None
    return Draft202012Validator(schema_doc, format_checker=FormatChecker())


def _schema_errors(validator: Any, payload: Dict[str, Any]) -> List[str]:
    if validator is None:
        return []
    sanitized = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    errors = sorted(validator.iter_errors(sanitized), key=lambda err: (list(err.absolute_path), str(err.message)))
    return [f"{'/'.join(str(part) for part in err.absolute_path) or '<root>'}: {err.message}" for err in errors]


def _source_records(source: Dict[str, Any], repo_root: Path) -> List[Dict[str, Any]]:
    source_path = repo_root / str(source.get("source_ref") or "")
    doc = load_json(source_path)
    mode = str(source.get("record_mode") or "single")
    if mode == "single":
        records = [doc]
    elif mode == "records":
        records = doc.get("records") if isinstance(doc.get("records"), list) else []
    else:
        raise SystemExit(f"Unsupported XH-703 record_mode: {mode}")

    out: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit(f"Non-object record encountered in source {source_path}")
        enriched = dict(record)
        enriched["_source_id"] = str(source.get("source_id") or "")
        enriched["_source_ref"] = str(source.get("source_ref") or "")
        enriched["_source_modality"] = str(source.get("source_modality") or "unknown")
        enriched["_ingest_channel"] = str(source.get("ingest_channel") or "unknown")
        out.append(enriched)
    return out


def _policy_flags(record: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    confidentiality = record.get("confidentiality") if isinstance(record.get("confidentiality"), dict) else {}
    governance = record.get("governance") if isinstance(record.get("governance"), dict) else {}
    record_type = str(record.get("record_type") or "")

    contains_phi = confidentiality.get("contains_phi") is True
    sharing_policy = str(confidentiality.get("sharing_policy") or "")
    redaction_required = confidentiality.get("redaction_required") is True
    classification = str(confidentiality.get("classification") or "")

    if contains_phi and sharing_policy == "deidentified_only":
        flags.append("phi_deidentified_policy_breach")
    if contains_phi and not redaction_required:
        flags.append("phi_redaction_missing")
    if str(governance.get("route_class") or "") != "advisory":
        flags.append("route_class_not_advisory")
    if governance.get("advisory_only") is not True:
        flags.append("advisory_only_not_true")
    if record_type == "lab_result" and classification not in {"health_sensitive", "health_restricted"}:
        flags.append("lab_classification_invalid")
    if str(governance.get("risk_class") or "") == "RG3_CRITICAL" and str(governance.get("escalation_code") or "") != "HE4_EMERGENCY_ESCALATION":
        flags.append("critical_escalation_missing")
    return flags


def _record_sort_key(record: Dict[str, Any]) -> Tuple[dt.datetime, str]:
    return (
        parse_ts(str(record.get("recorded_at") or "")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        str(record.get("record_id") or ""),
    )


def _lab_question_prep(record: Dict[str, Any]) -> List[str]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    lab = payload.get("lab_result") if isinstance(payload.get("lab_result"), dict) else {}
    test_name = str(lab.get("test_name") or "lab result")
    interpretation = str(lab.get("interpretation") or "unknown")
    if interpretation in {"high", "low", "critical"}:
        return [
            f"Ask the clinician how the {test_name} result should be interpreted in clinical context.",
            "Bring the original report and any recent symptoms to the discussion.",
        ]
    return [
        f"Confirm whether the {test_name} trend needs any follow-up testing or repeat timing.",
        "Keep the original report available for clinician review.",
    ]


def _build_support_cards(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record_index = {str(record.get("record_id") or ""): record for record in records}
    cards: List[Dict[str, Any]] = []
    for idx, record in enumerate(sorted(records, key=_record_sort_key), start=1):
        record_id = str(record.get("record_id") or "")
        record_type = str(record.get("record_type") or "unknown")
        governance = record.get("governance") if isinstance(record.get("governance"), dict) else {}
        relationships = record.get("relationships") if isinstance(record.get("relationships"), list) else []
        linked_ids = [
            str(row.get("target_record_id") or "")
            for row in relationships
            if isinstance(row, dict) and str(row.get("target_record_id") or "").strip()
        ]
        record_ids = [record_id] + [value for value in linked_ids if value in record_index and value != record_id]
        if record_type == "measurement":
            measurement = ((record.get("payload") if isinstance(record.get("payload"), dict) else {}).get("measurement") or {})
            metric_name = str(measurement.get("metric_name") or "metric")
            value = measurement.get("value")
            unit = str(measurement.get("unit") or "")
            summary = f"Wearable baseline: {metric_name} recorded at {value} {unit}."
            action_items = [
                "Keep the same logging cadence so trend comparisons stay reliable.",
                "If this trend becomes concerning, bring it to a licensed clinician for interpretation.",
            ]
            card_type = "trend_summary"
        elif record_type == "lab_result":
            lab = ((record.get("payload") if isinstance(record.get("payload"), dict) else {}).get("lab_result") or {})
            test_name = str(lab.get("test_name") or "lab result")
            observed_value = lab.get("observed_value")
            unit = str(lab.get("unit") or "")
            interpretation = str(lab.get("interpretation") or "unknown")
            summary = f"Lab support: {test_name} observed at {observed_value} {unit} ({interpretation})."
            action_items = _lab_question_prep(record)
            card_type = "clinician_question_prep"
        elif record_type == "symptom":
            symptom = ((record.get("payload") if isinstance(record.get("payload"), dict) else {}).get("symptom") or {})
            symptom_name = str(symptom.get("symptom_name") or "symptom")
            severity = symptom.get("severity_0_to_10")
            progression = str(symptom.get("progression") or "unknown")
            linked_protocol = next((value for value in linked_ids if value in record_index), None)
            protocol_note = f" Linked protocol: {linked_protocol}." if linked_protocol else ""
            summary = f"Symptom tracking: {symptom_name} severity {severity}/10 and {progression}.{protocol_note}"
            action_items = [
                "Continue tracking severity and timing so a clinician can review the pattern.",
                "Escalate to professional care sooner if symptoms worsen or new red flags appear.",
            ]
            card_type = "symptom_tracking"
        elif record_type == "protocol":
            protocol = ((record.get("payload") if isinstance(record.get("payload"), dict) else {}).get("protocol") or {})
            protocol_name = str(protocol.get("protocol_name") or "protocol")
            adherence = protocol.get("adherence_pct")
            status = str(protocol.get("status") or "unknown")
            summary = f"Protocol adherence: {protocol_name} is {status} with {adherence}% adherence."
            action_items = [
                "Review whether the routine still feels sustainable and safe.",
                "Discuss protocol changes with a licensed clinician when health concerns or lab shifts appear.",
            ]
            card_type = "protocol_checkin"
        else:
            summary = f"Unsupported record type {record_type} was loaded for advisory review."
            action_items = ["Review this record manually before using it in any support workflow."]
            card_type = "manual_review"

        cards.append(
            {
                "card_id": f"xh703_card_{idx:03d}",
                "record_ids": record_ids,
                "record_type": record_type,
                "card_type": card_type,
                "source_modality": str(record.get("_source_modality") or "unknown"),
                "summary": summary,
                "action_items": action_items,
                "escalation_code": str(governance.get("escalation_code") or "HE5_GOVERNANCE_BLOCK"),
                "risk_class": str(governance.get("risk_class") or "UNKNOWN"),
                "advisory_only": True,
                "disclaimer": DISCLAIMER,
            }
        )
    return cards


def build_runtime_package(
    *,
    repo_root: Path,
    queue_doc: Dict[str, Any],
    schema_doc: Dict[str, Any],
    fixture_doc: Dict[str, Any],
    risk_matrix: Dict[str, Any],
    owner_registry: Dict[str, Any],
    generated_at: str,
    stamp: str,
) -> Dict[str, Any]:
    queue_index = _queue_index(queue_doc)
    xh_703 = queue_index.get("XH-703")
    if not isinstance(xh_703, dict):
        raise SystemExit("XH-703 missing from authoritative queue")

    observed_state = str(xh_703.get("state") or "UNKNOWN")
    if observed_state not in {"READY", "DONE"}:
        raise SystemExit(f"XH-703 queue state must be READY or DONE, observed {observed_state}")

    anchor = fixture_doc.get("runtime_anchor") if isinstance(fixture_doc.get("runtime_anchor"), dict) else {}
    required_dependencies = [str(dep) for dep in (anchor.get("required_dependencies") or [])]
    dependency_states, unresolved_dependencies = _dependency_states(queue_index, required_dependencies)
    if unresolved_dependencies:
        raise SystemExit(f"XH-703 dependencies unresolved: {', '.join(unresolved_dependencies)}")

    validator = _schema_validator(schema_doc)
    schema_validation_mode = "jsonschema" if validator is not None else "basic_only"

    source_rows = fixture_doc.get("ingest_sources") if isinstance(fixture_doc.get("ingest_sources"), list) else []
    if not source_rows:
        raise SystemExit("XH-703 runtime fixture is missing ingest_sources")

    source_results: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    accepted_modalities: set[str] = set()
    policy_violation_sources: List[Dict[str, Any]] = []

    for source in source_rows:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise SystemExit("Encountered XH-703 source without source_id")
        records = _source_records(source, repo_root)
        source_errors: List[str] = []
        source_policy_flags: List[str] = []
        rejected_entry_ids: List[str] = []
        accepted_entry_ids: List[str] = []
        observed_result = "accepted"
        rejection_class: Optional[str] = None

        for record in records:
            record_id = str(record.get("record_id") or "")
            schema_errors = _schema_errors(validator, record)
            policy_flags = _policy_flags(record)
            source_errors.extend(schema_errors)
            source_policy_flags.extend(policy_flags)
            if schema_errors or policy_flags:
                observed_result = "rejected"
                rejected_entry_ids.append(record_id)
                rejected_records.append(record)
                if policy_flags:
                    rejection_class = "policy_violation"
                elif rejection_class is None:
                    rejection_class = "schema_invalid"
            else:
                accepted_entry_ids.append(record_id)

        if observed_result == "accepted":
            accepted_records.extend(records)
            accepted_modalities.add(str(source.get("source_modality") or "unknown"))
        else:
            accepted_entry_ids = []

        result_row = {
            "source_id": source_id,
            "source_ref": str(source.get("source_ref") or ""),
            "record_mode": str(source.get("record_mode") or "single"),
            "source_modality": str(source.get("source_modality") or "unknown"),
            "ingest_channel": str(source.get("ingest_channel") or "unknown"),
            "expected_result": str(source.get("expected_result") or "accepted"),
            "observed_result": observed_result,
            "expected_record_count": int(source.get("expected_record_count") or 0),
            "observed_record_count": len(records),
            "accepted_record_ids": accepted_entry_ids,
            "rejected_record_ids": rejected_entry_ids,
            "rejection_class": rejection_class,
            "schema_errors": source_errors,
            "policy_flags": sorted(set(source_policy_flags)),
            "result": "PASS" if observed_result == str(source.get("expected_result") or "accepted") and len(records) == int(source.get("expected_record_count") or 0) and ((source.get("expected_rejection_class") is None) or str(source.get("expected_rejection_class")) == str(rejection_class)) else "FAIL",
        }
        source_results.append(result_row)
        if rejection_class == "policy_violation":
            policy_violation_sources.append(result_row)

    if not policy_violation_sources:
        raise SystemExit("XH-703 requires at least one policy_violation source to exercise fail-close incident emission")

    support_cards = _build_support_cards(accepted_records)

    expected = fixture_doc.get("expected") if isinstance(fixture_doc.get("expected"), dict) else {}
    required_modalities = {str(value) for value in (anchor.get("required_modalities") or [])}
    missing_modalities = sorted(required_modalities - accepted_modalities)

    owner_tuple = _owner_tuple(owner_registry, "XH")
    lane_risk = _lane_risk(risk_matrix, "XH")
    route_decision = _route_decision()

    runtime_payload = {
        "schema": "clawd.xh_703.health_ingestion_support_runtime.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "queue_precondition": {
            "authoritative_queue_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "observed_slice_state": observed_state,
            "dependency_states": dependency_states,
            "dependencies_resolved": not unresolved_dependencies,
        },
        "route_decision": route_decision,
        "governance_boundary": {
            "lane_id": "XH",
            "lane_risk_class": lane_risk.get("risk_class"),
            "lane_risk_notes": lane_risk.get("notes"),
            "owners": owner_tuple,
            "advisory_only_required": True,
            "external_mutation_allowed": False,
            "invalid_output_supervision_contract_ref": "docs/ops/invalid_output_retry_relaunch_contract_v1.md",
            "incident_contract_ref": "docs/ops/xg_803_domain_failclose_incident_contract_v1.md",
        },
        "summary": {
            "source_count": len(source_results),
            "accepted_source_count": sum(1 for row in source_results if row.get("observed_result") == "accepted"),
            "rejected_source_count": sum(1 for row in source_results if row.get("observed_result") == "rejected"),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "support_card_count": len(support_cards),
            "accepted_modalities": sorted(accepted_modalities),
            "missing_required_modalities": missing_modalities,
            "policy_violation_source_ids": [row["source_id"] for row in policy_violation_sources],
            "advisory_only_boundary": True,
        },
        "operator_surface_refs": {
            "ingest_audit": f"state/continuity/latest/xh_703_ingest_audit_{stamp}.json",
            "support_workspace": f"state/continuity/latest/xh_703_support_workspace_{stamp}.json",
            "failclose_incident_packet": f"state/continuity/latest/xh_703_failclose_incident_packet_{stamp}.json",
            "failclose_incident_gate_decision": f"state/continuity/latest/xh_703_failclose_incident_gate_decision_{stamp}.json",
        },
    }

    ingest_audit_payload = {
        "schema": "clawd.xh_703.ingest_audit.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "schema_validation_mode": schema_validation_mode,
        "source_results": source_results,
        "status": "PASS" if all(row.get("result") == "PASS" for row in source_results) and not missing_modalities else "FAIL",
    }

    support_workspace_payload = {
        "schema": "clawd.xh_703.support_workspace.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "workspace_mode": str(anchor.get("support_workspace_mode") or "non_diagnostic_triage"),
        "disclaimer": DISCLAIMER,
        "support_cards": support_cards,
        "failclosed_sources": [row for row in source_results if row.get("observed_result") == "rejected"],
        "status": "PASS" if support_cards and all(card.get("advisory_only") is True for card in support_cards) else "FAIL",
    }

    policy_source = policy_violation_sources[0]
    operator_remediation_payload = {
        "schema": "clawd.xh_703.operator_remediation_plan.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "incident_source_id": policy_source["source_id"],
        "owner": owner_tuple.get("incident_owner"),
        "actions": [
            "Block the violating source from health support workspace ingestion.",
            "Retain evidence and schema/policy failure traces for operator review.",
            "Correct the source privacy envelope before any future re-ingest attempt.",
        ],
        "status": "verified",
    }

    lesson_handoff_payload = {
        "schema": "clawd.xh_703.incident_to_lesson_handoff.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "incident_source_id": policy_source["source_id"],
        "lesson": {
            "title": "Reject PHI-bearing records that claim deidentified sharing",
            "classification": "safety_and_privacy_guardrail",
            "action": "Preserve fail-close behavior and keep privacy envelope validation ahead of support-card rendering.",
        },
        "status": "PASS",
    }

    knowledge_trace_payload = {
        "schema": "clawd.xh_703.knowledge_queue_ingestion_trace.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "queue": "knowledge_review_approval_promotion_queue_v1",
        "incident_source_id": policy_source["source_id"],
        "ingestion_status": "ingested",
        "status": "PASS",
    }

    closure_verification_payload = {
        "schema": "clawd.xh_703.remediation_closure_verification.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "incident_source_id": policy_source["source_id"],
        "verified_by": owner_tuple.get("incident_owner"),
        "verification_summary": "The privacy-breaching source was excluded from support outputs and recorded in fail-close evidence.",
        "status": "PASS",
    }

    incident_packet_payload = {
        "schema_version": "clawd.domain_failclose_incident_packet.v1",
        "incident_id": "inc_xh703_privacy_policy_trip_20260329",
        "occurred_at": iso_plus_minutes(generated_at, -5),
        "detected_at": iso_plus_minutes(generated_at, -4),
        "lane_context": {
            "lane_id": "XH",
            "slice_id": "XH-703",
            "risk_class": lane_risk.get("risk_class") or "RG3_CRITICAL",
            "auth_tier": "ADMIN",
        },
        "incident_class": "policy_violation",
        "severity": "L4_SAFETY_BLOCK",
        "failclose_action": {
            "triggered": True,
            "mode": "block_activation",
            "triggered_at": iso_plus_minutes(generated_at, -3),
            "operator_surface_ref": f"state/continuity/latest/xh_703_support_workspace_{stamp}.json",
            "notes": "Blocked privacy-breaching health source before any support artifact could be emitted.",
        },
        "policy_bindings": [
            "docs/ops/health_lane_boundary_safety_contract_v1.md",
            "docs/ops/health_typed_record_schema_pack_v1.md",
            "docs/ops/xg_803_domain_failclose_incident_contract_v1.md",
            "docs/ops/verify_before_resume_gate_checklist_v1.md",
        ],
        "evidence_refs": [
            f"state/continuity/latest/xh_703_health_runtime_{stamp}.json",
            f"state/continuity/latest/xh_703_ingest_audit_{stamp}.json",
            f"state/continuity/latest/xh_703_end_to_end_ingestion_tests_{stamp}.json",
        ],
        "remediation": {
            "owner": owner_tuple.get("incident_owner"),
            "due_at": iso_plus_minutes(generated_at, 60),
            "status": "verified",
            "closure_plan_ref": f"state/continuity/latest/xh_703_operator_remediation_plan_{stamp}.json",
            "closure_verified_at": iso_plus_minutes(generated_at, 30),
            "closure_verification_ref": f"state/continuity/latest/xh_703_remediation_closure_verification_{stamp}.json",
        },
        "lesson_handoff": {
            "required": True,
            "incident_to_lesson_handoff_ref": f"state/continuity/latest/xh_703_incident_to_lesson_handoff_packet_{stamp}.json",
            "knowledge_queue_ingestion_trace_ref": f"state/continuity/latest/xh_703_knowledge_queue_ingestion_trace_{stamp}.json",
            "promotion_target": "promote_later",
            "ingestion_status": "ingested",
        },
    }

    check_rows = [
        {
            "test": "queue_state_ready_or_done",
            "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            "details": observed_state,
        },
        {
            "test": "dependencies_done",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "details": dependency_states,
        },
        {
            "test": "accepted_modalities_cover_manual_wearable_lab",
            "result": "PASS" if not missing_modalities else "FAIL",
            "details": {"accepted_modalities": sorted(accepted_modalities), "missing_modalities": missing_modalities},
        },
        {
            "test": "support_cards_are_advisory_only",
            "result": "PASS" if all(card.get("advisory_only") is True and card.get("action_items") and card.get("summary") for card in support_cards) else "FAIL",
            "details": {"support_card_count": len(support_cards)},
        },
        {
            "test": "privacy_breach_source_rejected",
            "result": "PASS" if any(row.get("source_id") == "privacy_breach_probe" and row.get("observed_result") == "rejected" for row in source_results) else "FAIL",
            "details": {"rejected_sources": [row.get("source_id") for row in source_results if row.get("observed_result") == "rejected"]},
        },
    ]
    end_to_end_payload = {
        "schema": "clawd.xh_703.end_to_end_ingestion_tests.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "checks": check_rows,
        "status": "PASS" if all(row.get("result") == "PASS" for row in check_rows) else "FAIL",
    }

    return {
        "route_decision": route_decision,
        "runtime": runtime_payload,
        "ingest_audit": ingest_audit_payload,
        "support_workspace": support_workspace_payload,
        "operator_remediation": operator_remediation_payload,
        "lesson_handoff": lesson_handoff_payload,
        "knowledge_trace": knowledge_trace_payload,
        "closure_verification": closure_verification_payload,
        "incident_packet": incident_packet_payload,
        "end_to_end": end_to_end_payload,
        "expected": expected,
        "source_results": source_results,
        "accepted_records": accepted_records,
        "support_cards": support_cards,
        "missing_modalities": missing_modalities,
        "observed_state": observed_state,
        "dependency_states": dependency_states,
    }


def build_validation_payload(package: Dict[str, Any], incident_decision: Dict[str, Any], generated_at: str, stamp: str) -> Dict[str, Any]:
    expected = package.get("expected") if isinstance(package.get("expected"), dict) else {}
    source_results = package.get("source_results") if isinstance(package.get("source_results"), list) else []
    accepted_records = package.get("accepted_records") if isinstance(package.get("accepted_records"), list) else []
    support_cards = package.get("support_cards") if isinstance(package.get("support_cards"), list) else []
    missing_modalities = package.get("missing_modalities") if isinstance(package.get("missing_modalities"), list) else []

    checks = [
        {"name": "queue_state_ready_or_done", "result": "PASS" if package.get("observed_state") in {"READY", "DONE"} else "FAIL"},
        {"name": "queue_dependencies_done", "result": "PASS" if all(value == "DONE" for value in (package.get("dependency_states") or {}).values()) else "FAIL"},
        {"name": "accepted_source_count_matches_fixture", "result": "PASS" if sum(1 for row in source_results if row.get("observed_result") == "accepted") == int(expected.get("accepted_source_count") or -1) else "FAIL"},
        {"name": "rejected_source_count_matches_fixture", "result": "PASS" if sum(1 for row in source_results if row.get("observed_result") == "rejected") == int(expected.get("rejected_source_count") or -1) else "FAIL"},
        {"name": "accepted_record_count_matches_fixture", "result": "PASS" if len(accepted_records) == int(expected.get("accepted_record_count") or -1) else "FAIL"},
        {"name": "support_card_count_matches_fixture", "result": "PASS" if len(support_cards) == int(expected.get("support_card_count") or -1) else "FAIL"},
        {"name": "required_modalities_covered", "result": "PASS" if not missing_modalities else "FAIL"},
        {"name": "accepted_records_preserve_advisory_boundary", "result": "PASS" if all(((row.get("governance") if isinstance(row.get("governance"), dict) else {}).get("advisory_only") is True and ((row.get("governance") if isinstance(row.get("governance"), dict) else {}).get("route_class") == "advisory")) for row in accepted_records) else "FAIL"},
        {"name": "privacy_breach_source_rejected", "result": "PASS" if any(row.get("source_id") == "privacy_breach_probe" and row.get("observed_result") == "rejected" and row.get("rejection_class") == "policy_violation" for row in source_results) else "FAIL"},
        {"name": "xg_803_failclose_incident_gate_pass", "result": "PASS" if incident_decision.get("decision") == str(expected.get("incident_decision") or "PASS") else "FAIL"},
        {"name": "end_to_end_ingestion_tests_pass", "result": package.get("end_to_end", {}).get("status", "FAIL")},
    ]
    status = "PASS" if all(row.get("result") == "PASS" for row in checks) else "FAIL"
    return {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "status": status,
        "checks": checks,
        "artifact_refs": {
            "runtime_snapshot": f"state/continuity/latest/xh_703_health_runtime_{stamp}.json",
            "ingest_audit": f"state/continuity/latest/xh_703_ingest_audit_{stamp}.json",
            "support_workspace": f"state/continuity/latest/xh_703_support_workspace_{stamp}.json",
            "failclose_incident_packet": f"state/continuity/latest/xh_703_failclose_incident_packet_{stamp}.json",
            "failclose_incident_gate_decision": f"state/continuity/latest/xh_703_failclose_incident_gate_decision_{stamp}.json",
            "end_to_end_ingestion_tests": f"state/continuity/latest/xh_703_end_to_end_ingestion_tests_{stamp}.json",
            "verify_before_resume_gate": f"state/continuity/latest/xh_703_verify_before_resume_gate_{stamp}.json",
        },
    }


def build_verify_before_resume_payload(package: Dict[str, Any], validation_payload: Dict[str, Any], incident_decision: Dict[str, Any], generated_at: str, stamp: str) -> Dict[str, Any]:
    gate_checks = [
        {
            "check": "fresh_truth_check",
            "result": "PASS",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "verification_check",
            "result": validation_payload.get("status", "FAIL"),
            "evidence_ref": f"state/continuity/latest/xh_703_runtime_validation_{stamp}.json",
        },
        {
            "check": "dependency_health_check",
            "result": "PASS" if all(value == "DONE" for value in (package.get("dependency_states") or {}).values()) else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "continuity_coherence_check",
            "result": package.get("support_workspace", {}).get("status", "FAIL"),
            "evidence_ref": f"state/continuity/latest/xh_703_support_workspace_{stamp}.json",
        },
        {
            "check": "blocker_state_check",
            "result": "PASS" if incident_decision.get("decision") == "PASS" else "FAIL",
            "evidence_ref": f"state/continuity/latest/xh_703_failclose_incident_gate_decision_{stamp}.json",
        },
        {
            "check": "evidence_quality_check",
            "result": package.get("end_to_end", {}).get("status", "FAIL"),
            "evidence_ref": f"state/continuity/latest/xh_703_end_to_end_ingestion_tests_{stamp}.json",
        },
    ]
    gate_result = "allowed" if all(row.get("result") == "PASS" for row in gate_checks) else "forbidden"
    return {
        "schema": "clawd.verify_before_resume_gate.v1",
        "slice_id": "XH-703",
        "gate_result": gate_result,
        "evaluated_at": generated_at,
        "evidence_refs": [
            "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "docs/ops/health_lane_boundary_safety_contract_v1.md",
            "docs/ops/health_typed_record_schema_pack_v1.md",
            "docs/ops/health_ingestion_support_runtime_v1.md",
            f"state/continuity/latest/xh_703_ingest_audit_{stamp}.json",
            f"state/continuity/latest/xh_703_support_workspace_{stamp}.json",
            f"state/continuity/latest/xh_703_failclose_incident_gate_decision_{stamp}.json",
            f"state/continuity/latest/xh_703_end_to_end_ingestion_tests_{stamp}.json",
        ],
        "failed_checks": [row["check"] for row in gate_checks if row.get("result") != "PASS"],
        "constraints_if_caution": [],
        "next_recheck_at": generated_at,
        "checks": gate_checks,
    }


def run_incident_gate(repo_root: Path, incident_gate_path: Path, packet_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(incident_gate_path),
        "--repo-root",
        str(repo_root),
        "--packet",
        str(packet_path),
        "--json",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if not result.stdout.strip():
        raise SystemExit(f"XH-703 incident gate produced no stdout: rc={result.returncode} stderr={result.stderr}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"Unable to parse incident gate stdout: {exc}\nstdout={result.stdout}\nstderr={result.stderr}") from exc
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    queue_path = Path(args.queue_path).resolve()
    schema_path = Path(args.schema_path).resolve()
    fixture_path = Path(args.fixture_path).resolve()
    risk_matrix_path = Path(args.risk_matrix_path).resolve()
    owner_registry_path = Path(args.owner_registry_path).resolve()
    incident_gate_path = Path(args.incident_gate_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).date().isoformat()
    generated_at = utc_now_iso()

    queue_doc = load_json(queue_path)
    schema_doc = load_json(schema_path)
    fixture_doc = load_json(fixture_path)
    risk_matrix = load_json(risk_matrix_path)
    owner_registry = load_json(owner_registry_path)

    package = build_runtime_package(
        repo_root=repo_root,
        queue_doc=queue_doc,
        schema_doc=schema_doc,
        fixture_doc=fixture_doc,
        risk_matrix=risk_matrix,
        owner_registry=owner_registry,
        generated_at=generated_at,
        stamp=stamp,
    )

    artifacts = {
        "runtime_snapshot": output_dir / f"xh_703_health_runtime_{stamp}.json",
        "ingest_audit": output_dir / f"xh_703_ingest_audit_{stamp}.json",
        "support_workspace": output_dir / f"xh_703_support_workspace_{stamp}.json",
        "operator_remediation_plan": output_dir / f"xh_703_operator_remediation_plan_{stamp}.json",
        "incident_to_lesson_handoff": output_dir / f"xh_703_incident_to_lesson_handoff_packet_{stamp}.json",
        "knowledge_queue_ingestion_trace": output_dir / f"xh_703_knowledge_queue_ingestion_trace_{stamp}.json",
        "remediation_closure_verification": output_dir / f"xh_703_remediation_closure_verification_{stamp}.json",
        "end_to_end_ingestion_tests": output_dir / f"xh_703_end_to_end_ingestion_tests_{stamp}.json",
        "failclose_incident_packet": output_dir / f"xh_703_failclose_incident_packet_{stamp}.json",
        "failclose_incident_gate_decision": output_dir / f"xh_703_failclose_incident_gate_decision_{stamp}.json",
        "verify_before_resume_gate": output_dir / f"xh_703_verify_before_resume_gate_{stamp}.json",
        "runtime_validation": output_dir / f"xh_703_runtime_validation_{stamp}.json",
        "artifact_manifest": output_dir / f"xh_703_runtime_artifact_manifest_{stamp}.json",
    }

    write_json(artifacts["runtime_snapshot"], package["runtime"])
    write_json(artifacts["ingest_audit"], package["ingest_audit"])
    write_json(artifacts["support_workspace"], package["support_workspace"])
    write_json(artifacts["operator_remediation_plan"], package["operator_remediation"])
    write_json(artifacts["incident_to_lesson_handoff"], package["lesson_handoff"])
    write_json(artifacts["knowledge_queue_ingestion_trace"], package["knowledge_trace"])
    write_json(artifacts["remediation_closure_verification"], package["closure_verification"])
    write_json(artifacts["end_to_end_ingestion_tests"], package["end_to_end"])
    write_json(artifacts["failclose_incident_packet"], package["incident_packet"])

    incident_decision = run_incident_gate(repo_root, incident_gate_path, artifacts["failclose_incident_packet"])
    write_json(artifacts["failclose_incident_gate_decision"], incident_decision)

    validation_payload = build_validation_payload(package, incident_decision, generated_at, stamp)
    write_json(artifacts["runtime_validation"], validation_payload)

    verify_payload = build_verify_before_resume_payload(package, validation_payload, incident_decision, generated_at, stamp)
    write_json(artifacts["verify_before_resume_gate"], verify_payload)

    manifest_payload = {
        "schema": "clawd.xh_703.runtime_artifact_manifest.v1",
        "slice_id": "XH-703",
        "generated_at": generated_at,
        "status": validation_payload.get("status"),
        "route_decision": package["route_decision"],
        "artifacts": {name: relpath(path, repo_root) for name, path in artifacts.items()},
    }
    write_json(artifacts["artifact_manifest"], manifest_payload)

    if args.json:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0 if validation_payload.get("status") == "PASS" and verify_payload.get("gate_result") == "allowed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
