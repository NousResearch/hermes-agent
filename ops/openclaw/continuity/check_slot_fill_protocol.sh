#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: check_slot_fill_protocol.sh [options]

Validates the child-slot refill protocol artifact and its runbook wiring.

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
import sys
from typing import Any, Dict, List

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
script_path = pathlib.Path(sys.argv[3]).resolve()

SCHEMA_REL = "ops/openclaw/architecture/schemas/slot_fill_protocol_check.schema.json"
SCHEMA_VERSION = "slot_fill_protocol.check.v1"
FAILURE_TAXONOMY_VERSION = "slot_fill_protocol.check_failure.v1"
CONTRACT_PREFIX = "slot_fill_protocol_check_contract"


def maybe_rel(path: pathlib.Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def resolve_schema_path() -> pathlib.Path:
    candidates = [
        (root / SCHEMA_REL).resolve(),
        (script_path.parent.parent / "architecture" / "schemas" / "slot_fill_protocol_check.schema.json").resolve(),
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


protocol_rel = "docs/ops/subagent_slot_fill_protocol_v1.md"
workflow_rel = "WORKFLOW_AUTO.md"

protocol_path = root / protocol_rel
workflow_path = root / workflow_rel
schema_path = resolve_schema_path()

checks: List[Dict[str, Any]] = []


def add(name: str, ok: bool, details: Any = None, severity: str = "critical"):
    row: Dict[str, Any] = {"name": name, "ok": bool(ok), "severity": severity}
    if details is not None:
        row["details"] = details
    checks.append(row)


protocol_text = ""
if protocol_path.exists():
    protocol_text = protocol_path.read_text(encoding="utf-8")
    add("protocol_exists", True, protocol_rel, "info")
else:
    add("protocol_exists", False, protocol_rel)

required_protocol_snippets = [
    "main lane orchestration-only for non-trivial slices",
    "Spawn-before-speak invariant",
    "call `sessions_spawn` first",
    "Narration-only acknowledgment",
    "If spawn blocked",
    "blocker with explicit reason",
    "Delegation trigger rules (explicit)",
    "main_session_tiny_exception",
    "delegation_basis",
    "Stale-worker / closeout-bundle discipline",
    "stale_worker_decision",
    "closeout_bundle_ref",
    "Quick checklist",
    "execution_mode",
    "worker_lane",
    "model_selection",
]

if protocol_text:
    missing = [s for s in required_protocol_snippets if s not in protocol_text]
    add("protocol_required_snippets", len(missing) == 0, {"missing": missing})
else:
    add("protocol_required_snippets", False, {"missing": required_protocol_snippets})

workflow_text = ""
if workflow_path.exists():
    workflow_text = workflow_path.read_text(encoding="utf-8")
    add("workflow_exists", True, workflow_rel, "info")
else:
    add("workflow_exists", False, workflow_rel)

if workflow_text:
    workflow_text_lower = workflow_text.lower()
    add(
        "workflow_references_slot_fill_protocol",
        protocol_rel in workflow_text,
        {"expected_ref": protocol_rel},
        "warn",
    )
    add(
        "workflow_mentions_spawn_before_speak",
        "spawn-before-speak" in workflow_text_lower,
        None,
        "warn",
    )
    add(
        "workflow_declares_execute_now_plan_only",
        "execute_now" in workflow_text_lower and "plan_only" in workflow_text_lower,
    )
    add(
        "workflow_declares_execution_tuple_fields",
        all(
            token in workflow_text
            for token in (
                "execution_mode",
                "worker_lane",
                "model_selection",
                "delegation_basis",
                "stale_worker_decision",
                "closeout_bundle_ref",
            )
        ),
    )


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
        failure_category = "protocol_contract_fail_close"
        failure_code = fail_close_codes[0]
        failure_retryable = False
    elif warn_failures:
        failure_category = "protocol_warning_boundary"
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
            protocol_rel,
            workflow_rel,
            "ops/openclaw/continuity/check_slot_fill_protocol.sh",
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
out = build_summary(contract_state_valid=True, contract_errors=contract_errors)
try:
    validate_payload_schema(out, schema_path)
except Exception as exc:
    contract_errors = [str(exc)]
    add("summary_contract_schema", False, str(exc), "critical")
    out = build_summary(contract_state_valid=False, contract_errors=contract_errors)

if json_out:
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("SLOT FILL PROTOCOL CHECK")
    print(f"- ok: {out['ok']}")
    print(f"- critical_failures: {out['critical_failures']}")
    print(f"- warn_failures: {out['warn_failures']}")
    print(f"- fail_close_triggered: {out['fail_close_triggered']}")
    if out.get("failure_code"):
        print(f"- failure_code: {out['failure_code']}")

if not out["ok"]:
    raise SystemExit(1)
PY
