#!/usr/bin/env python3
"""Domain release evidence extension gate (XG-802)."""

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
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "domain_release_evidence_bundle.schema.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def gate_schema(bundle: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "domain_schema_gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "domain_schema_gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "domain_schema_gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(bundle), key=lambda e: (list(e.absolute_path), str(e.message)))
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return False, "domain_schema_invalid", {
        "error": "schema_validation_failed",
        "path": "/".join(str(p) for p in err.absolute_path),
        "message": str(err.message),
    }


def gate_internal_only(bundle: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    profile = bundle.get("internal_release_profile") if isinstance(bundle.get("internal_release_profile"), dict) else {}
    in_scope = bool(profile.get("public_packaging_in_scope"))
    if in_scope:
        return False, "public_packaging_scope_violation", {"public_packaging_in_scope": True}
    return True, None, {"public_packaging_in_scope": False}


def gate_auth_and_risk(bundle: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    gov = bundle.get("domain_governance") if isinstance(bundle.get("domain_governance"), dict) else {}
    auth_tier = str(gov.get("auth_tier") or "")
    risk = str(gov.get("risk_class") or "")
    allowed_auth = {"ADMIN", "OBSERVABILITY", "INTERNAL", "PUBLIC"}
    allowed_risk = {"RG0_LOW", "RG1_MODERATE", "RG2_HIGH", "RG3_CRITICAL"}
    if auth_tier not in allowed_auth:
        return False, "auth_tier_missing_or_invalid", {"auth_tier": auth_tier}
    if risk not in allowed_risk:
        return False, "risk_class_missing_or_invalid", {"risk_class": risk}
    return True, None, {"auth_tier": auth_tier, "risk_class": risk}


def gate_approval_ladder(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    gov = bundle.get("domain_governance") if isinstance(bundle.get("domain_governance"), dict) else {}
    sensitive = bool(gov.get("sensitive_action"))
    ladder = bundle.get("approval_ladder") if isinstance(bundle.get("approval_ladder"), dict) else {}
    required = bool(ladder.get("required"))
    steps = ladder.get("steps") if isinstance(ladder.get("steps"), list) else []

    if sensitive and not required:
        return False, "approval_ladder_missing_for_sensitive_action", {"required": required, "sensitive_action": sensitive}

    pass_steps = [s for s in steps if isinstance(s, dict) and str(s.get("status") or "") == "pass"]
    if sensitive and not pass_steps:
        return False, "approval_ladder_no_pass_step", {"sensitive_action": sensitive}

    checked = 0
    for step in pass_steps:
        approver = str(step.get("approver_id") or "").strip()
        if not approver:
            return False, "approval_step_approver_missing", {"step": step.get("step")}
        ref = str(step.get("evidence_ref") or "").strip()
        if not ref:
            return False, "proof_ref_unresolved", {"error": "evidence_ref_missing"}
        path = resolve_repo_path(repo_root, ref)
        if not is_within(repo_root, path) or not path.exists() or not path.is_file():
            return False, "proof_ref_unresolved", {"path": ref}
        checked += 1

    return True, None, {"sensitive_action": sensitive, "pass_steps_checked": checked}


def gate_attribution(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    row = bundle.get("attribution") if isinstance(bundle.get("attribution"), dict) else {}
    required_fields = ["requested_by", "approved_by", "executed_by", "decision_log_ref"]
    missing = [f for f in required_fields if not str(row.get(f) or "").strip()]
    if missing:
        return False, "attribution_incomplete", {"missing": missing}

    ref = str(row.get("decision_log_ref") or "")
    path = resolve_repo_path(repo_root, ref)
    if not is_within(repo_root, path) or not path.exists() or not path.is_file():
        return False, "proof_ref_unresolved", {"decision_log_ref": ref}

    return True, None, {"decision_log_ref": ref}


def gate_quality_and_rollback(bundle: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    profile = bundle.get("internal_release_profile") if isinstance(bundle.get("internal_release_profile"), dict) else {}
    thresholds = profile.get("benchmark_thresholds") if isinstance(profile.get("benchmark_thresholds"), list) else []
    if not thresholds:
        return False, "benchmark_threshold_failed", {"error": "benchmark_thresholds_missing"}

    for item in thresholds:
        if not isinstance(item, dict):
            return False, "benchmark_threshold_failed", {"error": "benchmark_threshold_row_invalid"}
        if str(item.get("status") or "") != "pass":
            return False, "benchmark_threshold_failed", {"metric_id": item.get("metric_id"), "status": item.get("status")}
        ref = str(item.get("evidence_ref") or "").strip()
        if not ref:
            return False, "proof_ref_unresolved", {"error": "benchmark_evidence_ref_missing"}
        path = resolve_repo_path(repo_root, ref)
        if not is_within(repo_root, path) or not path.exists() or not path.is_file():
            return False, "proof_ref_unresolved", {"path": ref}

    rollback = profile.get("rollback_readiness") if isinstance(profile.get("rollback_readiness"), dict) else {}
    if str(rollback.get("status") or "") != "pass":
        return False, "rollback_readiness_not_pass", {"status": rollback.get("status")}

    refs = rollback.get("proof_refs") if isinstance(rollback.get("proof_refs"), list) else []
    if not refs:
        return False, "rollback_readiness_not_pass", {"error": "rollback_proof_refs_missing"}

    for ref in refs:
        token = str(ref or "").strip()
        if not token:
            return False, "proof_ref_unresolved", {"error": "rollback_ref_missing"}
        path = resolve_repo_path(repo_root, token)
        if not is_within(repo_root, path) or not path.exists() or not path.is_file():
            return False, "proof_ref_unresolved", {"path": token}

    return True, None, {"threshold_count": len(thresholds), "rollback_ref_count": len(refs)}


def evaluate(bundle: Any, bundle_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    bundle_dict = bundle if isinstance(bundle, dict) else {}
    gates: List[Dict[str, Any]] = []

    gate_specs = [
        ("domain_schema", lambda: gate_schema(bundle, schema_path)),
        ("internal_only_scope", lambda: gate_internal_only(bundle_dict)),
        ("auth_and_risk", lambda: gate_auth_and_risk(bundle_dict)),
        ("approval_ladder", lambda: gate_approval_ladder(bundle_dict, repo_root)),
        ("attribution", lambda: gate_attribution(bundle_dict, repo_root)),
        ("quality_and_rollback", lambda: gate_quality_and_rollback(bundle_dict, repo_root)),
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

    raw = json.dumps(bundle_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
      "schema": "clawd.xg_802_domain_release_extension_gate.decision.v1",
      "evaluated_at": now_iso(),
      "decision": "BLOCK" if blocked else "PASS",
      "block_gate": block_gate,
      "block_reason": block_reason,
      "release_id": bundle_dict.get("release_id"),
      "bundle": {
        "path": str(bundle_path),
        "sha256": hashlib.sha256(raw).hexdigest()
      },
      "gates": gates
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Domain release evidence extension gate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    bundle_path = Path(args.bundle).expanduser().resolve()

    try:
        bundle = load_json_file(bundle_path)
    except Exception as exc:
        result = {
            "schema": "clawd.xg_802_domain_release_extension_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "domain_schema",
            "block_reason": "domain_schema_invalid",
            "release_id": None,
            "bundle": {"path": str(bundle_path), "sha256": None},
            "gates": [_gate_result("domain_schema", False, "domain_schema_invalid", {"error": str(exc)})],
        }
    else:
        result = evaluate(bundle, bundle_path, repo_root, schema_path)

    rc = 0 if result.get("decision") == "PASS" else 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
