#!/usr/bin/env python3
"""Validate B8 UI evidence bundles against the contract schema."""

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


def json_ptr(parts: Iterable[Any]) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class Issue:
    code: str
    path: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def _build_issue(code: str, path: str, message: str) -> Issue:
    return Issue(code=code, path=path, message=message)


def semantic_checks(bundle: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    evidence = bundle.get("evidence", {}) if isinstance(bundle.get("evidence"), dict) else {}
    screenshot = evidence.get("screenshot", {}) if isinstance(evidence.get("screenshot"), dict) else {}
    state_snapshot = evidence.get("state_snapshot", {}) if isinstance(evidence.get("state_snapshot"), dict) else {}

    regions = screenshot.get("regions", []) if isinstance(screenshot.get("regions"), list) else []
    region_ids = [region.get("region_id") for region in regions if isinstance(region, dict)]
    region_id_set: Set[str] = {rid for rid in region_ids if isinstance(rid, str)}
    if len(region_ids) != len(region_id_set):
        issues.append(_build_issue(
            "duplicate_region_id",
            "$/evidence/screenshot/regions",
            "screenshot.regions contains duplicate region_id values",
        ))

    dom_refs = evidence.get("dom_component_refs", []) if isinstance(evidence.get("dom_component_refs"), list) else []
    dom_ref_ids = [dom_ref.get("dom_ref_id") for dom_ref in dom_refs if isinstance(dom_ref, dict)]
    dom_ref_id_set: Set[str] = {did for did in dom_ref_ids if isinstance(did, str)}
    if len(dom_ref_ids) != len(dom_ref_id_set):
        issues.append(_build_issue(
            "duplicate_dom_ref_id",
            "$/evidence/dom_component_refs",
            "dom_component_refs contains duplicate dom_ref_id values",
        ))

    state_facts = state_snapshot.get("state_fact_refs", []) if isinstance(state_snapshot.get("state_fact_refs"), list) else []
    state_fact_ids = [fact.get("fact_id") for fact in state_facts if isinstance(fact, dict)]
    state_fact_id_set: Set[str] = {fid for fid in state_fact_ids if isinstance(fid, str)}
    if len(state_fact_ids) != len(state_fact_id_set):
        issues.append(_build_issue(
            "duplicate_state_fact_id",
            "$/evidence/state_snapshot/state_fact_refs",
            "state_snapshot.state_fact_refs contains duplicate fact_id values",
        ))

    findings = bundle.get("findings", []) if isinstance(bundle.get("findings"), list) else []
    finding_ids = [finding.get("finding_id") for finding in findings if isinstance(finding, dict)]
    finding_id_set: Set[str] = {fid for fid in finding_ids if isinstance(fid, str)}
    if len(finding_ids) != len(finding_id_set):
        issues.append(_build_issue(
            "duplicate_finding_id",
            "$/findings",
            "findings contains duplicate finding_id values",
        ))

    recommendation_set = bundle.get("recommendation_set", []) if isinstance(bundle.get("recommendation_set"), list) else []
    recommendation_ids = [item.get("recommendation_id") for item in recommendation_set if isinstance(item, dict)]
    recommendation_id_set: Set[str] = {rid for rid in recommendation_ids if isinstance(rid, str)}
    if len(recommendation_ids) != len(recommendation_id_set):
        issues.append(_build_issue(
            "duplicate_recommendation_id",
            "$/recommendation_set",
            "recommendation_set contains duplicate recommendation_id values",
        ))

    for fidx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue

        links = finding.get("evidence_links", {}) if isinstance(finding.get("evidence_links"), dict) else {}
        for ridx, ref in enumerate(links.get("screenshot_region_refs", [])):
            if ref not in region_id_set:
                issues.append(_build_issue(
                    "finding_screenshot_region_ref_unresolved",
                    f"$/findings/{fidx}/evidence_links/screenshot_region_refs/{ridx}",
                    f"finding '{finding.get('finding_id')}' references missing screenshot region '{ref}'",
                ))

        for didx, ref in enumerate(links.get("dom_component_ref_ids", [])):
            if ref not in dom_ref_id_set:
                issues.append(_build_issue(
                    "finding_dom_ref_unresolved",
                    f"$/findings/{fidx}/evidence_links/dom_component_ref_ids/{didx}",
                    f"finding '{finding.get('finding_id')}' references missing dom ref '{ref}'",
                ))

        for sidx, ref in enumerate(links.get("state_fact_refs", [])):
            if ref not in state_fact_id_set:
                issues.append(_build_issue(
                    "finding_state_fact_ref_unresolved",
                    f"$/findings/{fidx}/evidence_links/state_fact_refs/{sidx}",
                    f"finding '{finding.get('finding_id')}' references missing state fact '{ref}'",
                ))

        finding_recommendations = finding.get("recommendations", []) if isinstance(finding.get("recommendations"), list) else []
        for ridx, recommendation in enumerate(finding_recommendations):
            if not isinstance(recommendation, dict):
                continue
            recommendation_id = recommendation.get("recommendation_id")
            if recommendation_id not in recommendation_id_set:
                issues.append(_build_issue(
                    "finding_recommendation_not_promoted",
                    f"$/findings/{fidx}/recommendations/{ridx}/recommendation_id",
                    f"finding recommendation_id '{recommendation_id}' is not present in recommendation_set",
                ))

    for ridx, recommendation in enumerate(recommendation_set):
        if not isinstance(recommendation, dict):
            continue
        for fidx, finding_id in enumerate(recommendation.get("finding_ids", [])):
            if finding_id not in finding_id_set:
                issues.append(_build_issue(
                    "recommendation_finding_ref_unresolved",
                    f"$/recommendation_set/{ridx}/finding_ids/{fidx}",
                    f"recommendation '{recommendation.get('recommendation_id')}' references missing finding_id '{finding_id}'",
                ))

    provenance = bundle.get("provenance", {}) if isinstance(bundle.get("provenance"), dict) else {}
    source_records = provenance.get("source_records", []) if isinstance(provenance.get("source_records"), list) else []
    source_ref_uris = {
        src.get("ref_uri")
        for src in source_records
        if isinstance(src, dict) and isinstance(src.get("ref_uri"), str)
    }
    source_kinds = {
        src.get("source_kind")
        for src in source_records
        if isinstance(src, dict) and isinstance(src.get("source_kind"), str)
    }

    screenshot_uri = screenshot.get("artifact_ref", {}).get("ref_uri") if isinstance(screenshot.get("artifact_ref"), dict) else None
    if isinstance(screenshot_uri, str) and screenshot_uri not in source_ref_uris:
        issues.append(_build_issue(
            "artifact_ref_not_in_provenance_source_records",
            "$/evidence/screenshot/artifact_ref/ref_uri",
            "screenshot artifact_ref.ref_uri must exist in provenance.source_records",
        ))

    state_uri = state_snapshot.get("artifact_ref", {}).get("ref_uri") if isinstance(state_snapshot.get("artifact_ref"), dict) else None
    if isinstance(state_uri, str) and state_uri not in source_ref_uris:
        issues.append(_build_issue(
            "artifact_ref_not_in_provenance_source_records",
            "$/evidence/state_snapshot/artifact_ref/ref_uri",
            "state_snapshot artifact_ref.ref_uri must exist in provenance.source_records",
        ))

    if dom_ref_id_set and "dom_component_tree" not in source_kinds:
        issues.append(_build_issue(
            "dom_source_record_missing",
            "$/provenance/source_records",
            "dom_component_refs are present but provenance.source_records has no source_kind='dom_component_tree'",
        ))

    generated_at = parse_dt(provenance.get("generated_at"))
    capture_at = parse_dt(bundle.get("capture", {}).get("captured_at") if isinstance(bundle.get("capture"), dict) else None)
    screenshot_at = parse_dt(screenshot.get("captured_at"))
    state_at = parse_dt(state_snapshot.get("captured_at"))

    if generated_at and capture_at and generated_at < capture_at:
        issues.append(_build_issue(
            "generated_before_capture",
            "$/provenance/generated_at",
            "provenance.generated_at must be >= capture.captured_at",
        ))

    if capture_at and screenshot_at and screenshot_at < capture_at:
        issues.append(_build_issue(
            "screenshot_before_capture",
            "$/evidence/screenshot/captured_at",
            "evidence.screenshot.captured_at must be >= capture.captured_at",
        ))

    if capture_at and state_at and state_at < capture_at:
        issues.append(_build_issue(
            "state_snapshot_before_capture",
            "$/evidence/state_snapshot/captured_at",
            "evidence.state_snapshot.captured_at must be >= capture.captured_at",
        ))

    if generated_at:
        if screenshot_at and screenshot_at > generated_at:
            issues.append(_build_issue(
                "screenshot_after_generated_at",
                "$/evidence/screenshot/captured_at",
                "evidence.screenshot.captured_at must be <= provenance.generated_at",
            ))

        if state_at and state_at > generated_at:
            issues.append(_build_issue(
                "state_snapshot_after_generated_at",
                "$/evidence/state_snapshot/captured_at",
                "evidence.state_snapshot.captured_at must be <= provenance.generated_at",
            ))

        for sidx, source_record in enumerate(source_records):
            if not isinstance(source_record, dict):
                continue
            source_at = parse_dt(source_record.get("captured_at"))
            if source_at and source_at > generated_at:
                issues.append(_build_issue(
                    "source_record_after_generated_at",
                    f"$/provenance/source_records/{sidx}/captured_at",
                    "provenance.source_records[].captured_at must be <= provenance.generated_at",
                ))

    return issues


def validate_bundle(bundle_path: Path, validator: Draft202012Validator) -> Dict[str, Any]:
    try:
        payload = load_json_file(bundle_path)
    except Exception as exc:
        return {
            "bundle_path": str(bundle_path),
            "ok": False,
            "error": "bundle_unreadable",
            "detail": str(exc),
        }

    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return {
            "bundle_path": str(bundle_path),
            "ok": False,
            "bundle_id": payload.get("bundle_id"),
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        }

    issues = semantic_checks(payload)
    if issues:
        return {
            "bundle_path": str(bundle_path),
            "ok": False,
            "bundle_id": payload.get("bundle_id"),
            "error": "semantic_validation_failed",
            "issues": [issue.as_dict() for issue in issues],
        }

    return {
        "bundle_path": str(bundle_path),
        "ok": True,
        "bundle_id": payload.get("bundle_id"),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate B8 UI evidence bundles")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "docs/ops/schemas/b8_ui_evidence_bundle.schema.json"),
        help="Path to schema JSON (default: docs/ops/schemas/b8_ui_evidence_bundle.schema.json)",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Path to bundle JSON. Repeat --bundle for multiple files.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if Draft202012Validator is None or FormatChecker is None:
        payload = {
            "ok": False,
            "error": "jsonschema_validator_unavailable",
        }
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
    bundles: List[Path] = [Path(p).expanduser().resolve() for p in args.bundle]
    results = [validate_bundle(path, validator) for path in bundles]

    failed = [r for r in results if not r.get("ok")]
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
