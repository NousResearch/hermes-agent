#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(candidate: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(candidate.get("audit_rule_id") or "").strip().lower(),
        str(candidate.get("title") or "").strip().lower(),
    )


def validate_guardrail_pack(pack: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    errors: List[str] = []

    if pack.get("schema") != "clawd.sys02_candidate_opportunity_guardrail_pack.v1":
        errors.append("pack.schema must be 'clawd.sys02_candidate_opportunity_guardrail_pack.v1'")

    surface = pack.get("candidate_surface")
    if not isinstance(surface, dict):
        errors.append("candidate_surface must be an object")
        return {
            "ok": False,
            "checked_at": _utc_now_iso(),
            "checks": {},
            "errors": errors,
        }

    try:
        import jsonschema  # type: ignore

        schema_path = repo_root / "docs" / "ops" / "schemas" / "b7_candidate_opportunity_surface.v1.schema.json"
        surface_schema = _load_json(schema_path)
        jsonschema.validate(surface, surface_schema)
    except ImportError:
        errors.append("jsonschema is required for schema validation")
    except Exception as exc:  # pragma: no cover - surfaced in tests/CLI
        errors.append(f"candidate_surface schema validation failed: {exc}")

    policy = surface.get("policy") if isinstance(surface.get("policy"), dict) else {}
    summary = surface.get("summary") if isinstance(surface.get("summary"), dict) else {}
    candidates = [row for row in (surface.get("candidates") or []) if isinstance(row, dict)]

    now_count = sum(1 for row in candidates if row.get("verdict") == "now")
    later_count = sum(1 for row in candidates if row.get("verdict") == "later")
    reject_count = sum(1 for row in candidates if row.get("verdict") == "reject")

    candidate_count = len(candidates)

    verdict_summary_consistent = (
        summary.get("candidate_count") == candidate_count
        and summary.get("now_count") == now_count
        and summary.get("later_count") == later_count
        and summary.get("reject_count") == reject_count
        and (now_count + later_count + reject_count) == candidate_count
    )

    max_total = int(policy.get("max_total_candidates", -1))
    max_now = int(policy.get("max_now_candidates", -1))
    max_later = int(policy.get("max_later_candidates", -1))
    policy_caps_respected = candidate_count <= max_total and now_count <= max_now and later_count <= max_later

    no_unmapped_promotions = True
    if bool(policy.get("reject_unmapped_rules", False)):
        for row in candidates:
            lane = row.get("canonical_lane")
            verdict = row.get("verdict")
            if lane in (None, "") and verdict in ("now", "later"):
                no_unmapped_promotions = False
                break

    no_low_signal_promotions = True
    if bool(policy.get("reject_low_signal_candidates", False)):
        for row in candidates:
            reason = str(row.get("verdict_reason") or "")
            verdict = row.get("verdict")
            if "low_signal" in reason and verdict != "reject":
                no_low_signal_promotions = False
                break

    dedupe_respected = True
    if bool(policy.get("deduplicate_by_rule_and_text", False)):
        seen = set()
        for row in candidates:
            key = _candidate_key(row)
            if key in seen:
                dedupe_respected = False
                break
            seen.add(key)

    checks = {
        "verdict_summary_consistent": verdict_summary_consistent,
        "policy_caps_respected": policy_caps_respected,
        "no_unmapped_promotions": no_unmapped_promotions,
        "no_low_signal_promotions": no_low_signal_promotions,
        "dedupe_respected": dedupe_respected,
    }

    expected = pack.get("guardrail_assertions")
    if isinstance(expected, dict):
        for key, actual in checks.items():
            if key in expected and bool(expected[key]) != bool(actual):
                errors.append(
                    f"guardrail_assertions.{key}={expected[key]!r} does not match computed value {actual!r}"
                )

    for key, ok in checks.items():
        if not ok:
            errors.append(f"guardrail check failed: {key}")

    return {
        "ok": not errors,
        "checked_at": _utc_now_iso(),
        "checks": checks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SYS-02 candidate opportunity anti-idea-spam guardrail pack."
    )
    parser.add_argument("--pack", required=True, help="Path to guardrail pack JSON file")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root (default: inferred from script location)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    pack_path = Path(args.pack).resolve()
    repo_root = Path(args.repo_root).resolve()

    try:
        pack_obj = _load_json(pack_path)
    except Exception as exc:
        result = {
            "ok": False,
            "checked_at": _utc_now_iso(),
            "checks": {},
            "errors": [f"failed to load pack JSON: {exc}"],
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"ERROR: {result['errors'][0]}")
        return 2

    result = validate_guardrail_pack(pack_obj, repo_root)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        status = "PASS" if result.get("ok") else "FAIL"
        print(f"{status} sys02-candidate-guardrail")
        if result.get("errors"):
            for error in result["errors"]:
                print(f"- {error}")

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
