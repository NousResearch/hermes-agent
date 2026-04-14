#!/usr/bin/env python3
"""Validate COD-06 regression/refactor/code-health packet bridge contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


TIER_ORDER: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _json_ptr(parts: Sequence[Any]) -> str:
    if not parts:
        return "$"
    return "$/" + "/".join(str(part) for part in parts)


def _issue(code: str, path: str, message: str) -> Dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tier_value(value: Any) -> int:
    token = str(value or "").strip().lower()
    return TIER_ORDER.get(token, -1)


def _schema_issues(payload: Any, validator: Optional[Draft202012Validator], path_prefix: Sequence[Any]) -> List[Dict[str, str]]:
    if validator is None:
        return []
    issues: List[Dict[str, str]] = []
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    for err in errors:
        full_path = list(path_prefix) + list(err.path)
        issues.append(_issue("schema_validation_failed", _json_ptr(full_path), err.message))
    return issues


def _semantic_checks(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []

    proposal_id = str(payload.get("proposal_id") or "").strip()
    packets = payload.get("packets") if isinstance(payload.get("packets"), Mapping) else {}

    regression = packets.get("regression_risk_packet") if isinstance(packets.get("regression_risk_packet"), Mapping) else {}
    refactor = packets.get("refactor_risk_packet") if isinstance(packets.get("refactor_risk_packet"), Mapping) else {}
    code_health = (
        packets.get("code_health_rule_governance_packet")
        if isinstance(packets.get("code_health_rule_governance_packet"), Mapping)
        else {}
    )

    for packet_name, packet in (
        ("regression", regression),
        ("refactor", refactor),
        ("code_health", code_health),
    ):
        packet_proposal_id = str(packet.get("proposal_id") or "").strip()
        if proposal_id and packet_proposal_id and packet_proposal_id != proposal_id:
            issues.append(
                _issue(
                    f"{packet_name}_proposal_id_mismatch",
                    _json_ptr(["packets", f"{packet_name}_risk_packet" if packet_name != "code_health" else "code_health_rule_governance_packet", "proposal_id"]),
                    f"{packet_name} proposal_id '{packet_proposal_id}' must match bridge proposal_id '{proposal_id}'",
                )
            )

    regression_tier = ((regression.get("risk_assessment") or {}).get("overall_tier") if isinstance(regression.get("risk_assessment"), Mapping) else None)
    refactor_tier = ((refactor.get("risk_assessment") or {}).get("overall_tier") if isinstance(refactor.get("risk_assessment"), Mapping) else None)
    max_tier = regression_tier
    if _tier_value(refactor_tier) > _tier_value(max_tier):
        max_tier = refactor_tier

    change_scope = code_health.get("change_scope") if isinstance(code_health.get("change_scope"), Mapping) else {}
    health_tier = change_scope.get("risk_tier")
    if _tier_value(max_tier) >= 0 and _tier_value(health_tier) >= 0 and _tier_value(health_tier) < _tier_value(max_tier):
        issues.append(
            _issue(
                "code_health_risk_tier_understated",
                _json_ptr(["packets", "code_health_rule_governance_packet", "change_scope", "risk_tier"]),
                f"code-health risk tier '{health_tier}' must be >= max(regression/refactor) tier '{max_tier}'",
            )
        )

    regression_required_approvals = (
        (regression.get("validation") or {}).get("required_approvals") if isinstance(regression.get("validation"), Mapping) else None
    )
    refactor_required_approvals = (
        (refactor.get("validation") or {}).get("required_approvals") if isinstance(refactor.get("validation"), Mapping) else None
    )
    code_health_required_approvals = (
        (code_health.get("validation") or {}).get("required_approvals") if isinstance(code_health.get("validation"), Mapping) else None
    )

    max_required = max(
        int(regression_required_approvals) if isinstance(regression_required_approvals, int) else 0,
        int(refactor_required_approvals) if isinstance(refactor_required_approvals, int) else 0,
    )
    health_required = int(code_health_required_approvals) if isinstance(code_health_required_approvals, int) else 0
    if health_required < max_required:
        issues.append(
            _issue(
                "code_health_required_approvals_understated",
                _json_ptr(["packets", "code_health_rule_governance_packet", "validation", "required_approvals"]),
                f"code-health required_approvals ({health_required}) must be >= max(regression/refactor) approvals ({max_required})",
            )
        )

    regression_blocking = str(regression.get("blocking_classification") or "").strip().lower() == "blocking"
    merged_analytics = (
        (code_health.get("rule_findings") or {}).get("merged_violation_analytics")
        if isinstance(code_health.get("rule_findings"), Mapping)
        else {}
    )
    governance_signal = str((merged_analytics or {}).get("governance_signal") or "").strip().lower()

    recommendations = code_health.get("governance_recommendations") if isinstance(code_health.get("governance_recommendations"), list) else []
    has_p0_recommendation = any(str((rec or {}).get("priority") or "").strip().lower() == "p0" for rec in recommendations if isinstance(rec, Mapping))
    if (regression_blocking or governance_signal == "action_required") and not has_p0_recommendation:
        issues.append(
            _issue(
                "action_required_without_p0_recommendation",
                _json_ptr(["packets", "code_health_rule_governance_packet", "governance_recommendations"]),
                "blocking/action_required posture requires at least one p0 governance recommendation",
            )
        )

    metric_evidence = (
        (code_health.get("code_health_metrics") or {}).get("metric_evidence")
        if isinstance(code_health.get("code_health_metrics"), Mapping)
        else []
    )
    summary = (
        (code_health.get("code_health_metrics") or {}).get("summary")
        if isinstance(code_health.get("code_health_metrics"), Mapping)
        else {}
    )
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for entry in metric_evidence if isinstance(metric_evidence, list) else []:
        if not isinstance(entry, Mapping):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status in status_counts:
            status_counts[status] += 1

    summary_map = {
        "pass": summary.get("pass_metric_count") if isinstance(summary, Mapping) else None,
        "warn": summary.get("warn_metric_count") if isinstance(summary, Mapping) else None,
        "fail": summary.get("failed_metric_count") if isinstance(summary, Mapping) else None,
    }
    for key, expected in summary_map.items():
        if isinstance(expected, int) and expected != status_counts[key]:
            issues.append(
                _issue(
                    "code_health_metric_summary_mismatch",
                    _json_ptr([
                        "packets",
                        "code_health_rule_governance_packet",
                        "code_health_metrics",
                        "summary",
                        f"{key}_metric_count" if key != "fail" else "failed_metric_count",
                    ]),
                    f"summary count for '{key}' is {expected}, but metric evidence count is {status_counts[key]}",
                )
            )

    rule_findings = code_health.get("rule_findings") if isinstance(code_health.get("rule_findings"), Mapping) else {}
    conflicts = rule_findings.get("conflicts") if isinstance(rule_findings.get("conflicts"), list) else []
    duplicates = rule_findings.get("duplicates") if isinstance(rule_findings.get("duplicates"), list) else []
    if isinstance(merged_analytics, Mapping):
        conflict_count = merged_analytics.get("conflict_count")
        duplicate_count = merged_analytics.get("duplicate_count")
        if isinstance(conflict_count, int) and conflict_count != len(conflicts):
            issues.append(
                _issue(
                    "conflict_count_mismatch",
                    _json_ptr([
                        "packets",
                        "code_health_rule_governance_packet",
                        "rule_findings",
                        "merged_violation_analytics",
                        "conflict_count",
                    ]),
                    f"merged_violation_analytics.conflict_count={conflict_count} must equal len(conflicts)={len(conflicts)}",
                )
            )
        if isinstance(duplicate_count, int) and duplicate_count != len(duplicates):
            issues.append(
                _issue(
                    "duplicate_count_mismatch",
                    _json_ptr([
                        "packets",
                        "code_health_rule_governance_packet",
                        "rule_findings",
                        "merged_violation_analytics",
                        "duplicate_count",
                    ]),
                    f"merged_violation_analytics.duplicate_count={duplicate_count} must equal len(duplicates)={len(duplicates)}",
                )
            )

    refactor_decomposition = refactor.get("decomposition_plan") if isinstance(refactor.get("decomposition_plan"), list) else []
    requires_review_chunk = _tier_value(refactor_tier) >= TIER_ORDER["high"] or len(refactor_decomposition) > 1
    if requires_review_chunk:
        has_review_chunk = any(
            isinstance(chunk, Mapping) and str(chunk.get("task_class") or "").strip() == "code:review"
            for chunk in refactor_decomposition
        )
        if not has_review_chunk:
            issues.append(
                _issue(
                    "refactor_requires_review_chunk",
                    _json_ptr(["packets", "refactor_risk_packet", "decomposition_plan"]),
                    "high/critical or multi-chunk refactor packet must include at least one code:review chunk",
                )
            )

    return issues


def _build_validator(schema_path: Path) -> Optional[Draft202012Validator]:
    if Draft202012Validator is None:
        return None
    schema_obj = _load_json(schema_path)
    return Draft202012Validator(schema_obj, format_checker=FormatChecker())


def _validate_bridge(
    bridge_path: Path,
    bridge_validator: Optional[Draft202012Validator],
    regression_validator: Optional[Draft202012Validator],
    refactor_validator: Optional[Draft202012Validator],
    code_health_validator: Optional[Draft202012Validator],
) -> Dict[str, Any]:
    try:
        payload = _load_json(bridge_path)
    except Exception as exc:
        return {
            "bridge_path": str(bridge_path),
            "ok": False,
            "error": "load_failed",
            "message": str(exc),
            "issues": [],
        }

    if not isinstance(payload, Mapping):
        return {
            "bridge_path": str(bridge_path),
            "ok": False,
            "error": "invalid_payload",
            "message": "bridge payload must be a JSON object",
            "issues": [],
        }

    issues: List[Dict[str, str]] = []
    issues.extend(_schema_issues(payload, bridge_validator, []))

    packets = payload.get("packets") if isinstance(payload.get("packets"), Mapping) else {}
    regression = packets.get("regression_risk_packet") if isinstance(packets.get("regression_risk_packet"), Mapping) else {}
    refactor = packets.get("refactor_risk_packet") if isinstance(packets.get("refactor_risk_packet"), Mapping) else {}
    code_health = (
        packets.get("code_health_rule_governance_packet")
        if isinstance(packets.get("code_health_rule_governance_packet"), Mapping)
        else {}
    )

    issues.extend(_schema_issues(regression, regression_validator, ["packets", "regression_risk_packet"]))
    issues.extend(_schema_issues(refactor, refactor_validator, ["packets", "refactor_risk_packet"]))
    issues.extend(
        _schema_issues(code_health, code_health_validator, ["packets", "code_health_rule_governance_packet"])
    )

    if not any(issue["code"] == "schema_validation_failed" for issue in issues):
        issues.extend(_semantic_checks(payload))

    if issues:
        return {
            "bridge_path": str(bridge_path),
            "bridge_id": payload.get("bridge_id"),
            "proposal_id": payload.get("proposal_id"),
            "ok": False,
            "error": "validation_failed",
            "issues": issues,
        }

    return {
        "bridge_path": str(bridge_path),
        "bridge_id": payload.get("bridge_id"),
        "proposal_id": payload.get("proposal_id"),
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate COD-06 coding risk packet bridge objects")
    parser.add_argument(
        "--bridge-schema",
        type=Path,
        default=Path("docs/ops/schemas/coding_risk_packet_bridge.v1.schema.json"),
        help="Path to bridge schema",
    )
    parser.add_argument(
        "--regression-schema",
        type=Path,
        default=Path("docs/ops/schemas/regression_risk_packet.v2.schema.json"),
        help="Path to regression-risk schema",
    )
    parser.add_argument(
        "--refactor-schema",
        type=Path,
        default=Path("docs/ops/schemas/refactor_risk_packet.v1.schema.json"),
        help="Path to refactor-risk schema",
    )
    parser.add_argument(
        "--code-health-schema",
        type=Path,
        default=Path("docs/ops/schemas/code_health_rule_governance_packet.v1.schema.json"),
        help="Path to code-health rule-governance schema",
    )
    parser.add_argument(
        "--bridge",
        action="append",
        required=True,
        type=Path,
        help="Bridge packet JSON path (repeatable)",
    )
    args = parser.parse_args()

    if Draft202012Validator is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "jsonschema_unavailable",
                    "message": "jsonschema is required for bridge validation",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    bridge_validator = _build_validator(args.bridge_schema)
    regression_validator = _build_validator(args.regression_schema)
    refactor_validator = _build_validator(args.refactor_schema)
    code_health_validator = _build_validator(args.code_health_schema)

    results = [
        _validate_bridge(
            bridge_path=path,
            bridge_validator=bridge_validator,
            regression_validator=regression_validator,
            refactor_validator=refactor_validator,
            code_health_validator=code_health_validator,
        )
        for path in args.bridge
    ]
    failed = sum(1 for row in results if not row.get("ok"))
    payload = {
        "ok": failed == 0,
        "validated": len(results),
        "failed": failed,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
