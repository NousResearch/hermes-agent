#!/usr/bin/env python3
"""Validate web-capture login-wall assist contracts (WEB-03 support)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - environment dependency
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "web_capture_login_wall_contract.schema.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$/"
    return "$/" + "/".join(str(part) for part in seq)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_command(cmd: str) -> str:
    return " ".join(str(cmd).strip().split())


def gate_schema(contract: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(contract),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    first = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": json_ptr(first.absolute_path),
            "schema_path": json_ptr(first.absolute_schema_path),
            "message": str(first.message),
            "error_count": len(errors),
        },
    )


def gate_policy_invariants(contract: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    status = str(contract.get("status") or "").strip()
    incident = contract.get("incident_actionability") if isinstance(contract.get("incident_actionability"), dict) else None
    issues: List[Dict[str, Any]] = []

    if status == "open":
        resume_command = str(contract.get("resume_command") or "").strip()
        if not resume_command:
            issues.append({"code": "open_resume_command_missing"})

        if incident is None:
            issues.append({"code": "open_incident_actionability_missing"})
        else:
            if incident.get("status") != "open":
                issues.append(
                    {
                        "code": "open_incident_status_invalid",
                        "expected": "open",
                        "actual": incident.get("status"),
                    }
                )
            if incident.get("action_required") is not True:
                issues.append(
                    {
                        "code": "open_incident_action_required_invalid",
                        "expected": True,
                        "actual": incident.get("action_required"),
                    }
                )

            commands = incident.get("recommended_commands")
            normalized_commands = set()
            if not isinstance(commands, list) or not commands:
                issues.append({"code": "open_recommended_commands_missing"})
            else:
                normalized_commands = {
                    normalize_command(cmd)
                    for cmd in commands
                    if isinstance(cmd, str) and cmd.strip()
                }
                if resume_command and normalize_command(resume_command) not in normalized_commands:
                    issues.append(
                        {
                            "code": "open_resume_command_missing_from_recommended_commands",
                            "resume_command": resume_command,
                        }
                    )

            normalized_step_commands = set()
            steps = incident.get("recommended_steps")
            if isinstance(steps, list):
                for idx, step in enumerate(steps):
                    if not isinstance(step, dict):
                        issues.append({"code": "open_recommended_step_not_object", "index": idx})
                        continue
                    step_id = str(step.get("step_id") or "").strip()
                    command = str(step.get("command") or "").strip()
                    if not step_id:
                        issues.append({"code": "open_recommended_step_missing_step_id", "index": idx})
                    if not command:
                        issues.append({"code": "open_recommended_step_missing_command", "index": idx})
                        continue
                    normalized_step_commands.add(normalize_command(command))

                if resume_command and normalized_step_commands and normalize_command(resume_command) not in normalized_step_commands:
                    issues.append(
                        {
                            "code": "open_resume_command_missing_from_recommended_steps",
                            "resume_command": resume_command,
                        }
                    )

    elif status == "resolved":
        resolved_at = str(contract.get("resolved_at") or "").strip()
        if not resolved_at:
            issues.append({"code": "resolved_resolved_at_missing"})

        if incident is not None:
            if incident.get("status") != "resolved":
                issues.append(
                    {
                        "code": "resolved_incident_status_invalid",
                        "expected": "resolved",
                        "actual": incident.get("status"),
                    }
                )
            if incident.get("action_required") is not False:
                issues.append(
                    {
                        "code": "resolved_incident_action_required_invalid",
                        "expected": False,
                        "actual": incident.get("action_required"),
                    }
                )

    if issues:
        return False, "policy_invariants_invalid", {"issues": issues}

    return True, None, {"status": status}


def evaluate_contract(contract_path: Path, schema_path: Path) -> Dict[str, Any]:
    evaluated_at = now_iso()
    checks: List[Dict[str, Any]] = []

    try:
        contract = load_json(contract_path)
    except Exception as exc:
        return {
            "schema": "clawd.web_capture_login_wall_contract_validation.v1",
            "evaluated_at": evaluated_at,
            "contract_path": str(contract_path),
            "schema_path": str(schema_path),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "checks": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {
                        "error": "candidate_json_unreadable",
                        "detail": str(exc),
                    },
                },
                {
                    "gate": "policy_invariants",
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                },
            ],
        }

    schema_ok, schema_reason, schema_details = gate_schema(contract, schema_path)
    if schema_ok:
        checks.append({"gate": "schema", "status": "pass", "details": schema_details})
    else:
        checks.append({"gate": "schema", "status": "fail", "reason": schema_reason, "details": schema_details})
        checks.append({"gate": "policy_invariants", "status": "skipped", "reason": "blocked_by_previous_gate"})
        return {
            "schema": "clawd.web_capture_login_wall_contract_validation.v1",
            "evaluated_at": evaluated_at,
            "contract_path": str(contract_path),
            "schema_path": str(schema_path),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": schema_reason,
            "checks": checks,
        }

    policy_ok, policy_reason, policy_details = gate_policy_invariants(contract if isinstance(contract, dict) else {})
    if policy_ok:
        checks.append({"gate": "policy_invariants", "status": "pass", "details": policy_details})
        decision = "PASS"
        final_state = "PASS"
        block_gate = None
        block_reason = None
    else:
        checks.append({"gate": "policy_invariants", "status": "fail", "reason": policy_reason, "details": policy_details})
        decision = "BLOCK"
        final_state = "BLOCKED"
        block_gate = "policy_invariants"
        block_reason = policy_reason

    return {
        "schema": "clawd.web_capture_login_wall_contract_validation.v1",
        "evaluated_at": evaluated_at,
        "contract_path": str(contract_path),
        "schema_path": str(schema_path),
        "decision": decision,
        "final_state": final_state,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "checks": checks,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate web capture login-wall assist contract")
    ap.add_argument("--contract", required=True, help="Path to login-wall contract JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Schema path")
    ap.add_argument("--json", action="store_true", help="Pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()

    contract_path = Path(args.contract).expanduser()
    if not contract_path.is_absolute():
        contract_path = (repo_root / contract_path).resolve()
    else:
        contract_path = contract_path.resolve()

    schema_path = Path(args.schema_path).expanduser()
    if not schema_path.is_absolute():
        schema_path = (repo_root / schema_path).resolve()
    else:
        schema_path = schema_path.resolve()

    result = evaluate_contract(contract_path, schema_path)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
