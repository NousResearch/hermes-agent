#!/usr/bin/env python3
"""Domain fail-close incident gate (XG-803)."""

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
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "domain_failclose_incident_packet.schema.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(raw: str) -> Optional[dt.datetime]:
    token = (raw or "").strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(token)
    except ValueError:
        return None


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _gate_result(gate: str, ok: bool, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {"gate": gate, "status": "pass" if ok else "fail"}
    if reason:
        row["reason"] = reason
    if details is not None:
        row["details"] = details
    return row


def gate_schema(packet: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "incident_schema_gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "incident_schema_gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "incident_schema_gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(packet), key=lambda e: (list(e.absolute_path), str(e.message)))
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return False, "incident_schema_invalid", {
        "error": "schema_validation_failed",
        "path": "/".join(str(p) for p in err.absolute_path),
        "message": str(err.message),
    }


def _check_resolved_file(repo_root: Path, ref: str, missing_reason: str, key: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    token = str(ref or "").strip()
    if not token:
        return False, missing_reason, {"error": f"{key}_missing"}
    path = resolve_repo_path(repo_root, token)
    if not is_within(repo_root, path) or not path.exists() or not path.is_file():
        return False, missing_reason, {key: token}
    return True, None, {key: token}


def gate_failclose(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    incident_class = str(packet.get("incident_class") or "")
    failclose = packet.get("failclose_action") if isinstance(packet.get("failclose_action"), dict) else {}
    triggered = bool(failclose.get("triggered"))

    must_trigger_classes = {"policy_violation", "boundary_breach", "safety_guard_trip"}
    if incident_class in must_trigger_classes and not triggered:
        return False, "failclose_action_missing", {"incident_class": incident_class, "triggered": triggered}

    ok, reason, details = _check_resolved_file(
        repo_root,
        str(failclose.get("operator_surface_ref") or ""),
        "operator_surface_ref_unresolved",
        "operator_surface_ref",
    )
    if not ok:
        return ok, reason, details

    return True, None, {"incident_class": incident_class, "triggered": triggered, **details}


def gate_evidence_refs(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    refs = packet.get("evidence_refs") if isinstance(packet.get("evidence_refs"), list) else []
    if not refs:
        return False, "evidence_ref_unresolved", {"error": "evidence_refs_missing"}

    checked = 0
    for ref in refs:
        token = str(ref or "").strip()
        path = resolve_repo_path(repo_root, token)
        if not token or not is_within(repo_root, path) or not path.exists() or not path.is_file():
            return False, "evidence_ref_unresolved", {"path": token}
        checked += 1

    return True, None, {"evidence_refs_checked": checked}


def gate_remediation(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    remediation = packet.get("remediation") if isinstance(packet.get("remediation"), dict) else {}
    owner = str(remediation.get("owner") or "").strip()
    if not owner:
        return False, "remediation_owner_missing", {"owner": owner}

    detected_at = parse_iso(str(packet.get("detected_at") or ""))
    due_at_raw = str(remediation.get("due_at") or "")
    due_at = parse_iso(due_at_raw)
    if due_at is None:
        return False, "remediation_due_missing_or_invalid", {"due_at": due_at_raw}

    if detected_at is not None and due_at <= detected_at:
        return False, "remediation_not_timebound", {"detected_at": packet.get("detected_at"), "due_at": due_at_raw}

    status = str(remediation.get("status") or "")
    if status == "verified":
        verified_at = parse_iso(str(remediation.get("closure_verified_at") or ""))
        if verified_at is None:
            return False, "remediation_closure_unverified", {"closure_verified_at": remediation.get("closure_verified_at")}

        ok, reason, details = _check_resolved_file(
            repo_root,
            str(remediation.get("closure_verification_ref") or ""),
            "remediation_verification_ref_unresolved",
            "closure_verification_ref",
        )
        if not ok:
            return ok, reason, details

    return True, None, {"owner": owner, "status": status, "due_at": due_at_raw}


def gate_learning_handoff(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    handoff = packet.get("lesson_handoff") if isinstance(packet.get("lesson_handoff"), dict) else {}
    if bool(handoff.get("required")) is not True:
        return False, "lesson_handoff_missing", {"required": handoff.get("required")}

    ok, reason, details = _check_resolved_file(
        repo_root,
        str(handoff.get("incident_to_lesson_handoff_ref") or ""),
        "incident_to_lesson_ref_unresolved",
        "incident_to_lesson_handoff_ref",
    )
    if not ok:
        return ok, reason, details

    ok2, reason2, details2 = _check_resolved_file(
        repo_root,
        str(handoff.get("knowledge_queue_ingestion_trace_ref") or ""),
        "knowledge_queue_trace_missing_or_unresolved",
        "knowledge_queue_ingestion_trace_ref",
    )
    if not ok2:
        return ok2, reason2, details2

    ingestion_status = str(handoff.get("ingestion_status") or "")
    if ingestion_status == "blocked":
        return False, "ingestion_trace_status_blocked", {"ingestion_status": ingestion_status}

    return True, None, {
        "promotion_target": handoff.get("promotion_target"),
        "ingestion_status": ingestion_status,
        **details,
        **details2,
    }


def evaluate(packet: Any, packet_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    packet_dict = packet if isinstance(packet, dict) else {}
    gates: List[Dict[str, Any]] = []

    gate_specs = [
        ("incident_schema", lambda: gate_schema(packet, schema_path)),
        ("failclose_action", lambda: gate_failclose(packet_dict, repo_root)),
        ("evidence_refs", lambda: gate_evidence_refs(packet_dict, repo_root)),
        ("remediation_contract", lambda: gate_remediation(packet_dict, repo_root)),
        ("incident_learning_handoff", lambda: gate_learning_handoff(packet_dict, repo_root)),
    ]

    blocked = False
    block_gate = None
    block_reason = None

    for name, fn in gate_specs:
        if blocked:
            gates.append({"gate": name, "status": "skipped", "reason": "blocked_by_previous_gate"})
            continue
        ok, reason, details = fn()
        gates.append(_gate_result(name, ok, reason, details))
        if not ok:
            blocked = True
            block_gate = name
            block_reason = reason

    raw = json.dumps(packet_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "schema": "clawd.xg_803_domain_failclose_incident_gate.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "incident_id": packet_dict.get("incident_id"),
        "packet": {
            "path": str(packet_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Domain fail-close incident gate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--packet", required=True)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    packet_path = Path(args.packet).expanduser().resolve()

    try:
        packet = load_json_file(packet_path)
    except Exception as exc:
        result = {
            "schema": "clawd.xg_803_domain_failclose_incident_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "incident_schema",
            "block_reason": "incident_schema_invalid",
            "incident_id": None,
            "packet": {"path": str(packet_path), "sha256": None},
            "gates": [_gate_result("incident_schema", False, "incident_schema_invalid", {"error": str(exc)})],
        }
    else:
        result = evaluate(packet, packet_path, repo_root, schema_path)

    rc = 0 if result.get("decision") == "PASS" else 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
