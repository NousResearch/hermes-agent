#!/usr/bin/env python3
"""Validate B8 operator task-state critique packets.

Performs:
1) JSON Schema validation
2) Semantic cross-reference validation for evidence/finding/recommendation linkage
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


@dataclass
class Issue:
    code: str
    path: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def json_ptr(parts: Iterable[Any]) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _build_issue(code: str, path: str, message: str) -> Issue:
    return Issue(code=code, path=path, message=message)


def semantic_checks(packet: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    evidence_items = packet.get("evidence_bundle", {}).get("evidence_items", [])
    evidence_ids = [item.get("evidence_id") for item in evidence_items if isinstance(item, dict)]
    evidence_id_set: Set[str] = {eid for eid in evidence_ids if isinstance(eid, str)}

    if len(evidence_ids) != len(evidence_id_set):
        issues.append(_build_issue(
            "duplicate_evidence_id",
            "$/evidence_bundle/evidence_items",
            "evidence_items contains duplicate evidence_id values",
        ))

    minimum_required_refs = packet.get("evidence_bundle", {}).get("minimum_required_refs", {})
    for key, value in minimum_required_refs.items():
        if value not in evidence_id_set:
            issues.append(_build_issue(
                "minimum_required_ref_unresolved",
                f"$/evidence_bundle/minimum_required_refs/{key}",
                f"minimum_required_refs.{key} points to missing evidence_id '{value}'",
            ))

    dimensions = packet.get("scoring", {}).get("dimensions", {})
    for dim_name, dim_payload in dimensions.items():
        refs = dim_payload.get("evidence_refs", []) if isinstance(dim_payload, dict) else []
        for idx, ref in enumerate(refs):
            if ref not in evidence_id_set:
                issues.append(_build_issue(
                    "dimension_evidence_ref_unresolved",
                    f"$/scoring/dimensions/{dim_name}/evidence_refs/{idx}",
                    f"dimension '{dim_name}' references missing evidence_id '{ref}'",
                ))

    findings = packet.get("findings", [])
    finding_ids = [f.get("finding_id") for f in findings if isinstance(f, dict)]
    finding_id_set: Set[str] = {fid for fid in finding_ids if isinstance(fid, str)}
    if len(finding_ids) != len(finding_id_set):
        issues.append(_build_issue(
            "duplicate_finding_id",
            "$/findings",
            "findings contains duplicate finding_id values",
        ))

    recommendations = packet.get("recommendations", [])
    recommendation_ids = [r.get("recommendation_id") for r in recommendations if isinstance(r, dict)]
    recommendation_id_set: Set[str] = {rid for rid in recommendation_ids if isinstance(rid, str)}
    if len(recommendation_ids) != len(recommendation_id_set):
        issues.append(_build_issue(
            "duplicate_recommendation_id",
            "$/recommendations",
            "recommendations contains duplicate recommendation_id values",
        ))

    for fidx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        for eidx, ref in enumerate(finding.get("evidence_refs", [])):
            if ref not in evidence_id_set:
                issues.append(_build_issue(
                    "finding_evidence_ref_unresolved",
                    f"$/findings/{fidx}/evidence_refs/{eidx}",
                    f"finding '{finding.get('finding_id')}' references missing evidence_id '{ref}'",
                ))

        for ridx, ref in enumerate(finding.get("recommendation_refs", [])):
            if ref not in recommendation_id_set:
                issues.append(_build_issue(
                    "finding_recommendation_ref_unresolved",
                    f"$/findings/{fidx}/recommendation_refs/{ridx}",
                    f"finding '{finding.get('finding_id')}' references missing recommendation_id '{ref}'",
                ))

    for ridx, recommendation in enumerate(recommendations):
        if not isinstance(recommendation, dict):
            continue
        for fidx, fid in enumerate(recommendation.get("linked_finding_ids", [])):
            if fid not in finding_id_set:
                issues.append(_build_issue(
                    "recommendation_finding_ref_unresolved",
                    f"$/recommendations/{ridx}/linked_finding_ids/{fidx}",
                    f"recommendation '{recommendation.get('recommendation_id')}' links missing finding_id '{fid}'",
                ))

    execution_links = packet.get("evidence_bundle", {}).get("execution_evidence_links", [])
    context_contracts = packet.get("execution_context", {}).get("relevant_contracts", [])
    context_contract_set = set(context_contracts) if isinstance(context_contracts, list) else set()

    if execution_links and "execution_context" not in packet:
        issues.append(_build_issue(
            "execution_context_missing",
            "$/execution_context",
            "execution_evidence_links provided but execution_context is missing",
        ))

    for lidx, link in enumerate(execution_links):
        if not isinstance(link, dict):
            continue
        evidence_id = link.get("evidence_id")
        if evidence_id not in evidence_id_set:
            issues.append(_build_issue(
                "execution_link_evidence_unresolved",
                f"$/evidence_bundle/execution_evidence_links/{lidx}/evidence_id",
                f"execution link references missing evidence_id '{evidence_id}'",
            ))

        context_ref = link.get("execution_context_ref")
        if context_contract_set and context_ref not in context_contract_set:
            issues.append(_build_issue(
                "execution_link_context_unresolved",
                f"$/evidence_bundle/execution_evidence_links/{lidx}/execution_context_ref",
                f"execution_context_ref '{context_ref}' not present in execution_context.relevant_contracts",
            ))

    weights = packet.get("scoring", {}).get("weights", {})
    if isinstance(weights, dict) and weights:
        weight_sum = sum(float(v) for v in weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            issues.append(_build_issue(
                "weights_sum_invalid",
                "$/scoring/weights",
                f"weights must sum to 1.0; got {weight_sum:.6f}",
            ))

    generated_at = parse_dt(packet.get("generated_at"))
    snapshot_at = parse_dt(packet.get("surface", {}).get("snapshot_at"))
    state_from = parse_dt(packet.get("task_state_scope", {}).get("state_window", {}).get("from"))
    state_to = parse_dt(packet.get("task_state_scope", {}).get("state_window", {}).get("to"))

    if state_from and state_to and state_from > state_to:
        issues.append(_build_issue(
            "state_window_invalid",
            "$/task_state_scope/state_window",
            "state_window.from must be <= state_window.to",
        ))

    if snapshot_at and state_from and snapshot_at < state_from:
        issues.append(_build_issue(
            "snapshot_outside_state_window",
            "$/surface/snapshot_at",
            "surface.snapshot_at is earlier than task_state_scope.state_window.from",
        ))

    if snapshot_at and state_to and snapshot_at > state_to:
        issues.append(_build_issue(
            "snapshot_outside_state_window",
            "$/surface/snapshot_at",
            "surface.snapshot_at is later than task_state_scope.state_window.to",
        ))

    if generated_at and snapshot_at and generated_at < snapshot_at:
        issues.append(_build_issue(
            "generated_before_snapshot",
            "$/generated_at",
            "generated_at must be >= surface.snapshot_at",
        ))

    return issues


def validate_packet(packet_path: Path, validator: Draft202012Validator) -> Dict[str, Any]:
    try:
        packet = load_json_file(packet_path)
    except Exception as exc:
        return {
            "packet_path": str(packet_path),
            "ok": False,
            "error": "packet_unreadable",
            "detail": str(exc),
        }

    schema_errors = sorted(
        validator.iter_errors(packet),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if schema_errors:
        err = schema_errors[0]
        return {
            "packet_path": str(packet_path),
            "ok": False,
            "packet_id": packet.get("packet_id"),
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        }

    issues = semantic_checks(packet)
    if issues:
        return {
            "packet_path": str(packet_path),
            "ok": False,
            "packet_id": packet.get("packet_id"),
            "error": "semantic_validation_failed",
            "issues": [issue.as_dict() for issue in issues],
        }

    return {
        "packet_path": str(packet_path),
        "ok": True,
        "packet_id": packet.get("packet_id"),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate B8 operator task-state critique packets")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "state/contracts/schemas/b8_operator_task_state_critique_packet.schema.json"),
        help="Path to schema JSON",
    )
    parser.add_argument(
        "--packet",
        action="append",
        required=True,
        help="Path to packet JSON. Repeat --packet for multiple files.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if Draft202012Validator is None or FormatChecker is None:
        payload = {"ok": False, "error": "jsonschema_validator_unavailable"}
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 2

    schema_path = Path(args.schema).expanduser().resolve()
    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        payload = {
            "ok": False,
            "error": "schema_unreadable",
            "schema_path": str(schema_path),
            "detail": str(exc),
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 2

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    packets = [Path(path).expanduser().resolve() for path in args.packet]
    results = [validate_packet(path, validator) for path in packets]

    failed = [item for item in results if not item.get("ok")]
    payload = {
        "ok": len(failed) == 0,
        "schema_path": str(schema_path),
        "checked": len(results),
        "failed": len(failed),
        "results": results,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
