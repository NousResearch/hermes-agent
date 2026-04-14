#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: check_swarm_operability.sh [options]

Executable operability checks for swarm role contracts + runbook command wiring.

Options:
  --json    JSON output
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$JSON_OUT" "$0" <<'PY'
import json
import pathlib
import re
import sys
from typing import Any, Dict, List

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
script_path = pathlib.Path(sys.argv[3]).resolve()

SCHEMA_REL = "ops/openclaw/architecture/schemas/swarm_operability_check.schema.json"
SCHEMA_VERSION = "swarm.operability.check.v1"
FAILURE_TAXONOMY_VERSION = "swarm_operability.check_failure.v1"
CONTRACT_PREFIX = "swarm_operability_check_contract"


def maybe_rel(path: pathlib.Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def resolve_schema_path() -> pathlib.Path:
    candidates = [
        (root / SCHEMA_REL).resolve(),
        (script_path.parent / "schemas" / "swarm_operability_check.schema.json").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


def validate_payload_schema(payload: Dict[str, Any], schema_path: pathlib.Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except Exception:
        raise RuntimeError(f"{CONTRACT_PREFIX}_validator_unavailable")

    if not schema_path.exists():
        raise RuntimeError(f"{CONTRACT_PREFIX}_schema_missing:{schema_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_doc, dict):
        raise RuntimeError(f"{CONTRACT_PREFIX}_schema_not_object")

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return

    err = errors[0]
    data_ptr = _json_ptr(err.absolute_path)
    schema_ptr = _json_ptr(err.absolute_schema_path)
    raise RuntimeError(
        f"{CONTRACT_PREFIX}_schema_validation_failed:"
        f"data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )


try:
    import yaml
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"pyyaml_missing:{exc}"}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

contract_rel = "ops/openclaw/architecture/swarm_role_contracts.v1.yaml"
runbook_rel = "docs/ops/swarm_operating_contract_runbook_v1.md"
continuity_dispatcher_rel = "ops/openclaw/continuity.sh"
check_script_rel = "ops/openclaw/architecture/check_swarm_operability.sh"

contract_path = root / contract_rel
runbook_path = root / runbook_rel
continuity_dispatcher = root / continuity_dispatcher_rel
schema_path = resolve_schema_path()

checks: List[Dict[str, Any]] = []


def add_check(name: str, ok: bool, severity: str = "critical", details: Any = None):
    row: Dict[str, Any] = {"name": name, "ok": bool(ok), "severity": severity}
    if details is not None:
        row["details"] = details
    checks.append(row)


contract = {}
if not contract_path.exists():
    add_check("swarm_contract_exists", False, "critical", str(contract_path))
else:
    add_check("swarm_contract_exists", True, "info", str(contract_path))
    try:
        parsed = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            contract = parsed
            add_check("swarm_contract_parse", True, "info")
        else:
            add_check("swarm_contract_parse", False, "critical", "top_level_not_object")
    except Exception as exc:
        add_check("swarm_contract_parse", False, "critical", str(exc))

required_roles = ["planner", "executor", "validator", "sre_watchdog", "librarian"]
roles = contract.get("roles") if isinstance(contract, dict) else None
if isinstance(roles, dict):
    missing_roles = [r for r in required_roles if r not in roles]
    add_check("swarm_roles_present", len(missing_roles) == 0, "critical", {"missing": missing_roles})

    required_role_keys = ["mandate", "must_not", "allowed_writes", "required_inputs", "required_outputs"]
    malformed = []
    for role_name in required_roles:
        role_obj = roles.get(role_name)
        if not isinstance(role_obj, dict):
            malformed.append({"role": role_name, "reason": "missing_or_not_object"})
            continue
        for key in required_role_keys:
            val = role_obj.get(key)
            if key == "mandate":
                if not isinstance(val, str) or not val.strip():
                    malformed.append({"role": role_name, "key": key, "reason": "empty"})
            else:
                if not isinstance(val, list) or len([x for x in val if str(x).strip()]) == 0:
                    malformed.append({"role": role_name, "key": key, "reason": "missing_or_empty_list"})
    add_check("swarm_role_shape", len(malformed) == 0, "critical", malformed)
else:
    add_check("swarm_roles_present", False, "critical", "roles_missing")
    add_check("swarm_role_shape", False, "critical", "roles_missing")

handoff = contract.get("handoff_packet") if isinstance(contract, dict) else None
required_handoff = [
    "task_id",
    "parent_task_id",
    "from_role",
    "to_role",
    "evidence_refs",
    "next_gate",
    "created_at",
    "budget_tokens_used",
    "model_tier",
]
if isinstance(handoff, dict):
    req = handoff.get("required_fields")
    req_set = set(req) if isinstance(req, list) else set()
    missing = [f for f in required_handoff if f not in req_set]
    add_check("handoff_required_fields", len(missing) == 0, "critical", {"missing": missing})
else:
    add_check("handoff_required_fields", False, "critical", "handoff_packet_missing")

gating = contract.get("gating_policy") if isinstance(contract, dict) else None
if isinstance(gating, dict):
    gating_expectations = {
        "no_merge_without_validator": True,
        "require_evidence_refs_on_block": True,
        "lock_release_on_done_or_rollback": True,
    }
    mismatches = []
    for key, expected in gating_expectations.items():
        if gating.get(key) is not expected:
            mismatches.append({"key": key, "expected": expected, "actual": gating.get(key)})
    add_check("gating_policy_invariants", len(mismatches) == 0, "critical", mismatches)
else:
    add_check("gating_policy_invariants", False, "critical", "gating_policy_missing")

runbook_text = ""
if not runbook_path.exists():
    add_check("runbook_exists", False, "critical", str(runbook_path))
else:
    runbook_text = runbook_path.read_text(encoding="utf-8")
    add_check("runbook_exists", True, "info", str(runbook_path))
    add_check(
        "runbook_canonical_contract_ref",
        contract_rel in runbook_text,
        "warn",
    )

cmd_paths = []
for match in re.finditer(r"bash\s+([\w./_-]+\.sh)", runbook_text):
    rel = match.group(1)
    p = (root / rel).resolve()
    cmd_paths.append({"rel": rel, "exists": p.exists(), "path": str(p)})
missing_cmds = [row["rel"] for row in cmd_paths if not row["exists"]]
add_check("runbook_command_paths_exist", len(missing_cmds) == 0, "critical", {"missing": missing_cmds, "count": len(cmd_paths)})

required_runbook_snippets = [
    "queue_arbitrator.sh claim",
    "queue_arbitrator.sh transition",
    "queue_arbitrator.sh handoffs --json",
    "queue_arbitrator.sh locks --active-only --json",
    "queue_arbitrator.sh remediate",
    "db_integrity_check.sh --strict --json",
    "swarm_runtime_check.sh --strict --json",
]
missing_snippets = [s for s in required_runbook_snippets if s not in runbook_text]
add_check("runbook_required_command_snippets", len(missing_snippets) == 0, "warn", {"missing": missing_snippets})

if continuity_dispatcher.exists():
    dispatcher_text = continuity_dispatcher.read_text(encoding="utf-8")
    add_check("continuity_dispatcher_queue_arb", "queue-arb" in dispatcher_text, "warn")
    add_check("continuity_dispatcher_swarm_check", "swarm-check" in dispatcher_text, "warn")
    add_check("continuity_dispatcher_slot_fill_check", "slot-fill-check" in dispatcher_text, "warn")
    add_check("continuity_dispatcher_gtc_sync", "gtc-sync" in dispatcher_text, "warn")
else:
    add_check("continuity_dispatcher_queue_arb", False, "warn", "continuity.sh_missing")
    add_check("continuity_dispatcher_swarm_check", False, "warn", "continuity.sh_missing")
    add_check("continuity_dispatcher_slot_fill_check", False, "warn", "continuity.sh_missing")
    add_check("continuity_dispatcher_gtc_sync", False, "warn", "continuity.sh_missing")


def _taxonomy_code(name: Any, severity: str) -> str:
    raw = str(name or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    if not cleaned:
        cleaned = "unknown_check"
    return f"{severity}:{cleaned}"


def build_summary(contract_state_valid: bool, contract_errors: List[str]) -> Dict[str, Any]:
    critical_failures = [c for c in checks if c.get("severity") == "critical" and not c.get("ok")]
    warn_failures = [c for c in checks if c.get("severity") == "warn" and not c.get("ok")]

    fail_close_codes = [_taxonomy_code(row.get("name"), "critical") for row in critical_failures]
    warn_codes = [_taxonomy_code(row.get("name"), "warn") for row in warn_failures]

    failure_category = None
    failure_code = None
    failure_retryable = None
    if critical_failures:
        failure_category = "operability_contract_fail_close"
        failure_code = fail_close_codes[0]
        failure_retryable = False
    elif warn_failures:
        failure_category = "operability_warning_boundary"
        failure_code = warn_codes[0]
        failure_retryable = True

    return {
        "ok": len(critical_failures) == 0,
        "schema_version": SCHEMA_VERSION,
        "check_count": len(checks),
        "critical_failures": len(critical_failures),
        "warn_failures": len(warn_failures),
        "checks": checks,
        "evidence_refs": [
            contract_rel,
            runbook_rel,
            continuity_dispatcher_rel,
            check_script_rel,
        ],
        "failure_taxonomy_version": FAILURE_TAXONOMY_VERSION,
        "fail_close_triggered": bool(critical_failures),
        "failure_category": failure_category,
        "failure_code": failure_code,
        "failure_retryable": failure_retryable,
        "failure_codes": {
            "critical": fail_close_codes,
            "warn": warn_codes,
        },
        "contract": {
            "schema_path": maybe_rel(schema_path),
            "state_valid": bool(contract_state_valid),
            "validation_errors": [str(item) for item in contract_errors if str(item or "").strip()],
        },
    }


contract_errors: List[str] = []
summary = build_summary(contract_state_valid=True, contract_errors=contract_errors)
try:
    validate_payload_schema(summary, schema_path)
except Exception as exc:
    contract_errors = [str(exc)]
    add_check("summary_contract_schema", False, "critical", str(exc))
    summary = build_summary(contract_state_valid=False, contract_errors=contract_errors)

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("SWARM OPERABILITY CHECK")
    print(f"- ok: {summary['ok']}")
    print(f"- critical_failures: {summary['critical_failures']}")
    print(f"- warn_failures: {summary['warn_failures']}")
    print(f"- fail_close_triggered: {summary['fail_close_triggered']}")
    if summary.get("failure_code"):
        print(f"- failure_code: {summary['failure_code']}")
    for row in checks:
        status = "PASS" if row.get("ok") else "FAIL"
        print(f"- [{status}] {row.get('name')} severity={row.get('severity')}")

if not summary["ok"]:
    raise SystemExit(1)
PY
