#!/usr/bin/env python3
"""Validate B8 component consistency audit overlay packets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

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


@dataclass
class Issue:
    code: str
    path: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def _issue(code: str, path: str, message: str) -> Issue:
    return Issue(code=code, path=path, message=message)


def semantic_checks(payload: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    finding_count = int(payload.get("finding_count") or 0)
    if finding_count != len(findings):
        issues.append(
            _issue(
                "finding_count_mismatch",
                "$/finding_count",
                f"finding_count={finding_count} does not match len(findings)={len(findings)}",
            )
        )

    critical_finding_count = int(payload.get("critical_finding_count") or 0)
    computed_critical = len(
        [
            row
            for row in findings
            if isinstance(row, dict)
            and str(row.get("severity") or "").strip().lower() in {"critical", "high"}
        ]
    )
    if critical_finding_count != computed_critical:
        issues.append(
            _issue(
                "critical_finding_count_mismatch",
                "$/critical_finding_count",
                (
                    "critical_finding_count must equal number of findings with severity "
                    f"critical|high; got {critical_finding_count}, expected {computed_critical}"
                ),
            )
        )

    status = str(payload.get("status") or "").strip().lower()
    if status == "clean" and finding_count != 0:
        issues.append(
            _issue(
                "clean_status_has_findings",
                "$/status",
                "status=clean requires finding_count=0",
            )
        )
    if status == "applied" and finding_count <= 0:
        issues.append(
            _issue(
                "applied_status_without_findings",
                "$/status",
                "status=applied requires finding_count>0",
            )
        )
    if status in {"skipped", "invalid"} and finding_count > 0:
        issues.append(
            _issue(
                "non_applied_status_has_findings",
                "$/status",
                "status=skipped|invalid should not carry findings",
            )
        )

    finding_ids = [row.get("finding_id") for row in findings if isinstance(row, dict)]
    finding_id_set: Set[str] = {fid for fid in finding_ids if isinstance(fid, str)}
    if len(finding_ids) != len(finding_id_set):
        issues.append(
            _issue(
                "duplicate_finding_id",
                "$/findings",
                "findings contains duplicate finding_id values",
            )
        )

    rule_ids = payload.get("rule_ids_evaluated") if isinstance(payload.get("rule_ids_evaluated"), list) else []
    rule_id_set: Set[str] = {str(rule_id).strip() for rule_id in rule_ids if str(rule_id).strip()}

    for fidx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        finding_rule_id = str(finding.get("audit_rule_id") or "").strip()
        if finding_rule_id and finding_rule_id not in rule_id_set:
            issues.append(
                _issue(
                    "finding_rule_not_evaluated",
                    f"$/findings/{fidx}/audit_rule_id",
                    f"finding audit_rule_id '{finding_rule_id}' is missing from rule_ids_evaluated",
                )
            )

        inconsistent_paths = finding.get("inconsistent_paths") if isinstance(finding.get("inconsistent_paths"), list) else []
        for pidx, ptr in enumerate(inconsistent_paths):
            if not isinstance(ptr, str) or not ptr.startswith("/"):
                issues.append(
                    _issue(
                        "inconsistent_path_not_json_pointer",
                        f"$/findings/{fidx}/inconsistent_paths/{pidx}",
                        "inconsistent_paths entries must be absolute JSON pointers",
                    )
                )

    runtime_overlay = payload.get("runtime_execution_snapshot_overlay")
    if isinstance(runtime_overlay, dict):
        runtime_status = str(runtime_overlay.get("status") or "").strip().lower()
        runtime_finding_count = int(runtime_overlay.get("finding_count") or 0)
        runtime_critical_count = int(runtime_overlay.get("critical_finding_count") or 0)

        if runtime_critical_count > runtime_finding_count:
            issues.append(
                _issue(
                    "runtime_critical_count_exceeds_total",
                    "$/runtime_execution_snapshot_overlay/critical_finding_count",
                    "runtime critical_finding_count cannot exceed finding_count",
                )
            )

        if runtime_status == "clean" and runtime_finding_count != 0:
            issues.append(
                _issue(
                    "runtime_clean_status_has_findings",
                    "$/runtime_execution_snapshot_overlay/status",
                    "runtime status=clean requires finding_count=0",
                )
            )

    return issues


def validate_overlay(overlay_path: Path, validator: Draft202012Validator) -> Dict[str, Any]:
    try:
        payload = load_json_file(overlay_path)
    except Exception as exc:
        return {
            "overlay_path": str(overlay_path),
            "ok": False,
            "error": "overlay_unreadable",
            "detail": str(exc),
        }

    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return {
            "overlay_path": str(overlay_path),
            "ok": False,
            "overlay_id": payload.get("overlay_id"),
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        }

    issues = semantic_checks(payload)
    if issues:
        return {
            "overlay_path": str(overlay_path),
            "ok": False,
            "overlay_id": payload.get("overlay_id"),
            "error": "semantic_validation_failed",
            "issues": [issue.as_dict() for issue in issues],
        }

    return {
        "overlay_path": str(overlay_path),
        "ok": True,
        "overlay_id": payload.get("overlay_id"),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate B8 component consistency audit overlays")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "docs/ops/schemas/b8_component_consistency_audit_overlay.schema.json"),
        help="Path to schema JSON",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        required=True,
        help="Path to overlay JSON. Repeat --overlay for multiple files.",
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
    overlays = [Path(p).expanduser().resolve() for p in args.overlay]
    results = [validate_overlay(path, validator) for path in overlays]

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
