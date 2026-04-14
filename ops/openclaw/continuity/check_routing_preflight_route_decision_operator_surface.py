#!/usr/bin/env python3
"""Fail-closed EX-10 operator-surface parity checker for session-route decisions.

This validator ensures route-decision readability fields stay coherent across:
- route
- route.provider_route_decision
- routing_audit

Primary purpose: catch regressions where provider-selection evidence is emitted in one
surface but drifts/mutates/missing in others, which breaks operator diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "clawd.routing_preflight_route_decision_operator_surface_check.v1"
DEFAULT_ROUTE_DECISION_PATH = Path("state/continuity/latest/routing_preflight_refresh_route_decision.json")
EXPECTED_ROUTE_SCHEMA = "clawd.session_topology_routing.decision.v1"
EXPECTED_PROVIDER_SCHEMA = "clawd.session_topology.provider_route_decision.v1"
EXPECTED_PROVIDER_DOCTRINE = "ex10_provider_doctrine_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"parse_error:{exc}"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def run_check(*, route_decision_payload: dict[str, Any], route_decision_path: Path) -> dict[str, Any]:
    issues: list[str] = []

    if str(route_decision_payload.get("schema") or "").strip() != EXPECTED_ROUTE_SCHEMA:
        issues.append("route_decision_schema_invalid")

    route = _as_dict(route_decision_payload.get("route"))
    provider = _as_dict(route.get("provider_route_decision"))
    audit = _as_dict(route_decision_payload.get("routing_audit"))

    if not route:
        issues.append("route_missing")
    if not provider:
        issues.append("provider_route_decision_missing")
    if not audit:
        issues.append("routing_audit_missing")

    if str(provider.get("schema") or "").strip() != EXPECTED_PROVIDER_SCHEMA:
        issues.append("provider_route_decision_schema_invalid")
    if str(provider.get("doctrine_id") or "").strip() != EXPECTED_PROVIDER_DOCTRINE:
        issues.append("provider_route_decision_doctrine_invalid")

    route_signal = _as_dict(route.get("selected_qualification_signal"))
    provider_signal = _as_dict(provider.get("selected_qualification_signal"))
    audit_signal = _as_dict(audit.get("selected_qualification_signal"))

    if not route_signal:
        issues.append("route_selected_qualification_signal_missing")
    if not provider_signal:
        issues.append("provider_selected_qualification_signal_missing")
    if not audit_signal:
        issues.append("audit_selected_qualification_signal_missing")

    if route_signal and provider_signal and _canonical(route_signal) != _canonical(provider_signal):
        issues.append("selected_qualification_signal_route_provider_mismatch")
    if route_signal and audit_signal and _canonical(route_signal) != _canonical(audit_signal):
        issues.append("selected_qualification_signal_route_audit_mismatch")

    route_rubric = _as_dict(route.get("selection_rubric"))
    provider_rubric = _as_dict(provider.get("selection_rubric"))
    audit_rubric = _as_dict(audit.get("selection_rubric"))

    if not route_rubric:
        issues.append("route_selection_rubric_missing")
    if not provider_rubric:
        issues.append("provider_selection_rubric_missing")
    if not audit_rubric:
        issues.append("audit_selection_rubric_missing")

    if route_rubric and provider_rubric and _canonical(route_rubric) != _canonical(provider_rubric):
        issues.append("selection_rubric_route_provider_mismatch")
    if route_rubric and audit_rubric and _canonical(route_rubric) != _canonical(audit_rubric):
        issues.append("selection_rubric_route_audit_mismatch")

    # Rubric IDs are mandatory operator-readable anchors.
    rubric_rule_id_values = {
        "route.rubric_rule_id": route.get("rubric_rule_id"),
        "provider.rubric_rule_id": provider.get("rubric_rule_id"),
        "audit.rubric_rule_id": audit.get("rubric_rule_id"),
        "route.selection_rubric.rubric_rule_id": route_rubric.get("rubric_rule_id"),
    }
    if not all(_is_non_empty_string(value) for value in rubric_rule_id_values.values()):
        issues.append("rubric_rule_id_missing")
    else:
        canonical_rule = str(route.get("rubric_rule_id")).strip()
        if any(str(value).strip() != canonical_rule for value in rubric_rule_id_values.values()):
            issues.append("rubric_rule_id_surface_mismatch")

    # v2 IDs may be absent globally, but if present on any surface they must align.
    rubric_rule_id_v2_values = {
        "route.rubric_rule_id_v2": route.get("rubric_rule_id_v2"),
        "provider.rubric_rule_id_v2": provider.get("rubric_rule_id_v2"),
        "audit.rubric_rule_id_v2": audit.get("rubric_rule_id_v2"),
        "route.selection_rubric.rubric_rule_id_v2": route_rubric.get("rubric_rule_id_v2"),
    }
    present_v2 = [str(v).strip() for v in rubric_rule_id_v2_values.values() if _is_non_empty_string(v)]
    if present_v2 and len(set(present_v2)) != 1:
        issues.append("rubric_rule_id_v2_surface_mismatch")

    route_model = str(route.get("selected_model") or "").strip()
    provider_model = str(provider.get("selected_model") or "").strip()
    if route_model != provider_model:
        issues.append("selected_model_route_provider_mismatch")

    route_task_class = str(route.get("task_class") or "").strip()
    provider_task_class = str(provider.get("task_class") or "").strip()
    audit_task_class = str(audit.get("task_class") or "").strip()
    present_task_values = [v for v in (route_task_class, provider_task_class, audit_task_class) if v]
    if present_task_values and len(set(present_task_values)) != 1:
        issues.append("task_class_surface_mismatch")

    route_risk = str(route.get("effective_risk_tier") or route.get("request_risk_tier") or "").strip()
    provider_risk = str(provider.get("risk_tier") or "").strip()
    audit_risk = str(audit.get("effective_risk_tier") or audit.get("request_risk_tier") or "").strip()
    present_risk_values = [v for v in (route_risk, provider_risk, audit_risk) if v]
    if present_risk_values and len(set(present_risk_values)) != 1:
        issues.append("risk_tier_surface_mismatch")

    if route_signal.get("is_stale") is True and not _is_non_empty_string(route_signal.get("freshness_reason")):
        issues.append("stale_signal_missing_freshness_reason")

    if route_signal.get("provider_evidence_stale") is True and not _is_non_empty_string(
        route_signal.get("provider_freshness_reason")
    ):
        issues.append("provider_stale_signal_missing_reason")

    if route_signal.get("is_legacy_packet") is True:
        missing_fields = route_signal.get("legacy_missing_fields")
        guidance = route_signal.get("legacy_migration_guidance")
        if isinstance(missing_fields, list) and missing_fields and not (isinstance(guidance, list) and guidance):
            issues.append("legacy_signal_missing_migration_guidance")

    unique_issues = sorted(set(issues))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": len(unique_issues) == 0,
        "issues": unique_issues,
        "issue_count": len(unique_issues),
        "inputs": {
            "route_decision_path": str(route_decision_path),
        },
        "surface_summary": {
            "decision": route_decision_payload.get("decision"),
            "final_state": route_decision_payload.get("final_state"),
            "route_class": route.get("route_class"),
            "selected_model": route.get("selected_model"),
            "rubric_rule_id": route.get("rubric_rule_id"),
            "rubric_rule_id_v2": route.get("rubric_rule_id_v2"),
            "legacy_signal": bool(route_signal.get("is_legacy_packet") is True),
            "stale_signal": bool(route_signal.get("is_stale") is True),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check route/provider/audit operator-surface parity for routing-preflight decision artifacts."
    )
    parser.add_argument("--root", default=None, help="OpenClaw root (defaults to OPENCLAW_ROOT or repo root)")
    parser.add_argument(
        "--route-decision",
        default=str(DEFAULT_ROUTE_DECISION_PATH),
        help="Path to session_topology_router decision artifact JSON",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[3]
    if args.root:
        root = _resolve_path(Path.cwd(), args.root)
    else:
        root = Path(os.environ.get("OPENCLAW_ROOT", str(default_root))).resolve()

    route_decision_path = _resolve_path(root, args.route_decision)
    route_payload, route_error = _load_json(route_decision_path)

    if route_error is not None:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "ok": False,
            "issues": [f"route_decision_load_error:{route_error}"],
            "issue_count": 1,
            "inputs": {
                "route_decision_path": str(route_decision_path),
            },
            "surface_summary": {},
        }
    else:
        summary = run_check(route_decision_payload=route_payload or {}, route_decision_path=route_decision_path)

    if not args.json:
        status = "PASS" if summary.get("ok") else "FAIL"
        print(f"{status}: routing preflight route decision operator-surface parity")
        surface = summary.get("surface_summary") if isinstance(summary.get("surface_summary"), dict) else {}
        print(
            f"selected_model={surface.get('selected_model') or 'unknown'} "
            f"rubric_rule_id={surface.get('rubric_rule_id') or 'missing'} "
            f"issues={summary.get('issue_count', 0)}"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
