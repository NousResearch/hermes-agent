#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CRON_JSON_PATH=""
JSON_OUT=0
STRICT_EXIT=0
REMINDER_NAMES_CSV="${OPENCLAW_NO_NUDGE_REMINDER_NAMES:-continuity:backup-checkpoint-90m,continuity:stale-progress-45m}"
GUARD_SCHEMA_VERSION="openclaw.no_nudge_continuity_cron_guard.summary.v2"
GUARD_FAILURE_TAXONOMY_VERSION="openclaw.no_nudge_continuity_cron_guard.failure_taxonomy.v1"
GUARD_CLASSIFICATION_TAXONOMY_VERSION="openclaw.no_nudge_continuity_cron_guard.classification.v1"
GUARD_SCHEMA_REL="ops/openclaw/architecture/schemas/no_nudge_continuity_cron_guard_summary.schema.json"
GUARD_SCHEMA_CONTRACT_PREFIX="no_nudge_cron_guard_summary_contract"

usage() {
  cat <<'EOF'
Usage: no_nudge_continuity_cron_guard.sh [options]

Validate that continuity reminder cron rails are internal-only and silent-by-default.

Rules for enabled reminder jobs (default names: continuity:backup-checkpoint-90m, continuity:stale-progress-45m):
- Exactly one enabled job MUST exist per expected reminder name
- sessionTarget MUST be "isolated"
- payload.kind MUST be "agentTurn"
- delivery.mode MUST be "none"
- payload.message MUST include "BLOCKER:" and "NO_REPLY" contract markers
- payload.message MUST NOT forward "READY:", "PROGRESS:", or "BLOCKER_JSON:" lines

Options:
  --cron-json <path>   Use a saved `openclaw cron list --json` payload instead of live CLI
  --json               Emit machine-readable JSON after first protocol line
  --strict             Exit non-zero when violations are found
  -h, --help
EOF
}

_sanitize_inline_text() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

_validate_and_reclassify_guard_summary_json() {
  local raw_json="${1:-{}}"

  python3 - "$ROOT" "$GUARD_SCHEMA_REL" "$0" "$GUARD_SCHEMA_VERSION" "$GUARD_FAILURE_TAXONOMY_VERSION" "$GUARD_CLASSIFICATION_TAXONOMY_VERSION" "$GUARD_SCHEMA_CONTRACT_PREFIX" 3<<<"$raw_json" <<'PY'
import json
import os
import pathlib
import sys
from typing import Any

with os.fdopen(3, "r", encoding="utf-8") as raw_fp:
    raw_json = raw_fp.read()
if not str(raw_json or "").strip():
    raw_json = "{}"
root = pathlib.Path(sys.argv[1]).resolve()
schema_rel = str(sys.argv[2] or "").strip() or "ops/openclaw/architecture/schemas/no_nudge_continuity_cron_guard_summary.schema.json"
script_path = pathlib.Path(sys.argv[3]).resolve()
schema_version = str(sys.argv[4] or "openclaw.no_nudge_continuity_cron_guard.summary.v2")
failure_taxonomy_version = str(sys.argv[5] or "openclaw.no_nudge_continuity_cron_guard.failure_taxonomy.v1")
classification_taxonomy_version = str(sys.argv[6] or "openclaw.no_nudge_continuity_cron_guard.classification.v1")
contract_prefix = str(sys.argv[7] or "no_nudge_cron_guard_summary_contract").strip() or "no_nudge_cron_guard_summary_contract"


def dedupe(items):
    out = []
    seen = set()
    for item in items:
        txt = str(item or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def _json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


def resolve_schema_path() -> pathlib.Path:
    filename = pathlib.Path(schema_rel).name
    candidates = [
        (root / schema_rel).resolve(),
        (script_path.parent / "architecture" / "schemas" / filename).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_payload_schema(payload: dict, schema_path: pathlib.Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except Exception:
        raise RuntimeError(f"{contract_prefix}_validator_unavailable")

    if not schema_path.exists():
        raise RuntimeError(f"{contract_prefix}_schema_missing:{schema_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_doc, dict):
        raise RuntimeError(f"{contract_prefix}_schema_not_object")

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
        f"{contract_prefix}_schema_validation_failed:"
        f"data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )


obj = None
parse_error_detail = ""
text = str(raw_json or "")
try:
    obj = json.loads(text)
except Exception as exc:
    parse_error_detail = str(exc)

    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip())
        parse_error_detail = ""
    except Exception:
        pass

    if obj is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                parse_error_detail = ""
            except Exception:
                pass

    if obj is None:
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                parse_error_detail = ""
                break
            except Exception:
                continue

if obj is None:
    obj = {
        "ok": False,
        "classification": "cron_contract_drift",
        "classification_bucket": "contract",
        "error": "guard_summary_json_parse_failed",
        "detail": f"guard summary json parse failed before schema validation: {parse_error_detail}",
        "checked": 0,
        "checked_jobs": [],
        "coverage": {
            "expected_enabled_reminder_names": [],
            "missing_enabled_reminder_names": [],
            "duplicate_enabled_reminder_names": [],
            "enabled_counts": {},
        },
        "violations": [],
        "evidence_refs": ["ops/openclaw/no_nudge_continuity_cron_guard.sh"],
        "fail_close_triggered": True,
        "failure_category": "cron_contract_fail_close",
        "failure_code": "contract:guard_summary_json_parse_failed",
        "failure_retryable": False,
        "failure_codes": {
            "all": ["contract:guard_summary_json_parse_failed"],
            "critical": ["contract:guard_summary_json_parse_failed"],
            "policy": [],
            "contract": ["contract:guard_summary_json_parse_failed"],
            "connectivity": [],
        },
    }

if not isinstance(obj, dict):
    obj = {
        "ok": False,
        "classification": "cron_contract_drift",
        "classification_bucket": "contract",
        "error": "guard_summary_not_object",
        "detail": f"guard summary payload type={type(obj).__name__}",
        "checked": 0,
        "checked_jobs": [],
        "coverage": {
            "expected_enabled_reminder_names": [],
            "missing_enabled_reminder_names": [],
            "duplicate_enabled_reminder_names": [],
            "enabled_counts": {},
        },
        "violations": [],
        "evidence_refs": ["ops/openclaw/no_nudge_continuity_cron_guard.sh"],
        "fail_close_triggered": True,
        "failure_category": "cron_contract_fail_close",
        "failure_code": "contract:guard_summary_not_object",
        "failure_retryable": False,
        "failure_codes": {
            "all": ["contract:guard_summary_not_object"],
            "critical": ["contract:guard_summary_not_object"],
            "policy": [],
            "contract": ["contract:guard_summary_not_object"],
            "connectivity": [],
        },
    }

obj.setdefault("schema_version", schema_version)
obj.setdefault("classification_taxonomy_version", classification_taxonomy_version)
obj.setdefault("failure_taxonomy_version", failure_taxonomy_version)
obj.setdefault("checked", 0)
obj.setdefault("checked_jobs", [])
obj.setdefault(
    "coverage",
    {
        "expected_enabled_reminder_names": [],
        "missing_enabled_reminder_names": [],
        "duplicate_enabled_reminder_names": [],
        "enabled_counts": {},
    },
)
obj.setdefault("violations", [])
obj.setdefault("evidence_refs", ["ops/openclaw/no_nudge_continuity_cron_guard.sh"])
obj.setdefault("fail_close_triggered", True)
obj.setdefault("failure_category", "cron_contract_fail_close")
obj.setdefault("failure_code", "contract:unknown_failure")
obj.setdefault("failure_retryable", False)
obj.setdefault(
    "failure_codes",
    {
        "all": ["contract:unknown_failure"],
        "critical": ["contract:unknown_failure"],
        "policy": [],
        "contract": ["contract:unknown_failure"],
        "connectivity": [],
    },
)

schema_path = resolve_schema_path()
schema_path_str = str(schema_path)
try:
    schema_path_str = str(schema_path.relative_to(root))
except Exception:
    schema_path_str = str(schema_path)

validation_errors = []
try:
    validate_payload_schema(obj, schema_path)
except Exception as exc:
    validation_errors = [str(exc)]

if validation_errors:
    contract_code = "contract:guard_summary_schema_invalid"
    obj["ok"] = False
    obj["classification"] = "cron_contract_drift"
    obj["classification_bucket"] = "contract"
    obj["error"] = "guard_summary_schema_invalid"
    obj["detail"] = validation_errors[0]
    obj["fail_close_triggered"] = True
    obj["failure_category"] = "cron_contract_fail_close"
    obj["failure_code"] = contract_code
    obj["failure_retryable"] = False
    obj["failure_codes"] = {
        "all": [contract_code],
        "critical": [contract_code],
        "policy": [],
        "contract": [contract_code],
        "connectivity": [],
    }
    violations = obj.get("violations") if isinstance(obj.get("violations"), list) else []
    violations.append(
        {
            "id": "",
            "name": "guard_summary_contract",
            "reasons": [validation_errors[0]],
            "codes": [contract_code],
            "bucket": "contract",
        }
    )
    obj["violations"] = violations

obj["contract"] = {
    "schema_path": schema_path_str,
    "state_valid": len(validation_errors) == 0,
    "validation_errors": dedupe(validation_errors),
}

print(json.dumps(obj, ensure_ascii=False))
PY
}

_emit_failure_json_if_requested() {
  local classification="${1:-cron_contract_drift}"
  local error="${2:-unknown_error}"
  local detail="${3:-}"
  local failure_code="${4:-contract:unknown_failure}"
  local retryable_raw="${5:-0}"
  local evidence_ref="${6:-}"

  if [[ "$JSON_OUT" -ne 1 ]]; then
    return 0
  fi

  local payload_json
  payload_json="$(python3 - "$classification" "$error" "$detail" "$failure_code" "$retryable_raw" "$evidence_ref" "$REMINDER_NAMES_CSV" "$GUARD_SCHEMA_VERSION" "$GUARD_FAILURE_TAXONOMY_VERSION" "$GUARD_CLASSIFICATION_TAXONOMY_VERSION" <<'PY'
import json
import sys

classification = str(sys.argv[1] or "cron_contract_drift")
error = str(sys.argv[2] or "unknown_error")
detail = str(sys.argv[3] or "")
failure_code = str(sys.argv[4] or "").strip()
retryable = str(sys.argv[5] or "0").strip() in {"1", "true", "TRUE", "yes", "on"}
evidence_ref = str(sys.argv[6] or "").strip()
reminder_names_csv = str(sys.argv[7] or "")
schema_version = str(sys.argv[8] or "openclaw.no_nudge_continuity_cron_guard.summary.v2")
failure_taxonomy_version = str(sys.argv[9] or "openclaw.no_nudge_continuity_cron_guard.failure_taxonomy.v1")
classification_taxonomy_version = str(sys.argv[10] or "openclaw.no_nudge_continuity_cron_guard.classification.v1")

reminder_names = []
for token in reminder_names_csv.split(","):
    name = token.strip()
    if not name or name in reminder_names:
        continue
    reminder_names.append(name)


def dedupe(items):
    out = []
    seen = set()
    for item in items:
        txt = str(item or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


classification_bucket = "contract"
failure_category = "cron_contract_fail_close"
if classification == "gateway_connectivity_failure":
    classification_bucket = "connectivity"
    failure_category = "external_dependency_unreachable"
elif classification == "cron_policy_failed":
    classification_bucket = "policy"
    failure_category = "policy_contract_fail_close"

all_codes = [failure_code] if failure_code else []
policy_codes = all_codes if classification_bucket == "policy" else []
contract_codes = all_codes if classification_bucket == "contract" else []
connectivity_codes = all_codes if classification_bucket == "connectivity" else []

violations = []
if failure_code:
    violations.append(
        {
            "id": "",
            "name": "guard_preflight",
            "reasons": [error if not detail else f"{error}; {detail}"],
            "codes": [failure_code],
            "bucket": classification_bucket,
        }
    )

payload = {
    "ok": False,
    "schema_version": schema_version,
    "classification_taxonomy_version": classification_taxonomy_version,
    "failure_taxonomy_version": failure_taxonomy_version,
    "classification": classification,
    "classification_bucket": classification_bucket,
    "error": error,
    "detail": detail,
    "checked": 0,
    "checked_jobs": [],
    "coverage": {
        "expected_enabled_reminder_names": reminder_names,
        "missing_enabled_reminder_names": reminder_names,
        "duplicate_enabled_reminder_names": [],
        "enabled_counts": {name: 0 for name in reminder_names},
    },
    "violations": violations,
    "evidence_refs": dedupe(
        [
            "ops/openclaw/no_nudge_continuity_cron_guard.sh",
            evidence_ref,
        ]
    ),
    "fail_close_triggered": True,
    "failure_category": failure_category,
    "failure_code": failure_code or None,
    "failure_retryable": bool(retryable),
    "failure_codes": {
        "all": all_codes,
        "critical": all_codes,
        "policy": policy_codes,
        "contract": contract_codes,
        "connectivity": connectivity_codes,
    },
}

print(json.dumps(payload, ensure_ascii=False))
PY
)"

  payload_json="$(_validate_and_reclassify_guard_summary_json "$payload_json")"
  echo "$payload_json"
}

_is_gateway_connectivity_failure() {
  local raw="${1:-}"
  local lower
  lower="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"

  local -a tokens=(
    "gateway unavailable"
    "connection refused"
    "connection reset"
    "connect econn"
    "econnrefused"
    "ehostunreach"
    "enotfound"
    "network is unreachable"
    "temporary failure in name resolution"
    "timed out"
    "timeout"
    "socket hang up"
    "fetch failed"
    "service unavailable"
    "status 502"
    "status 503"
    "http 502"
    "http 503"
  )

  local token
  for token in "${tokens[@]}"; do
    if [[ "$lower" == *"$token"* ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cron-json)
      CRON_JSON_PATH="${2:-}"
      shift 2
      ;;
    --json)
      JSON_OUT=1
      shift
      ;;
    --strict)
      STRICT_EXIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

source_evidence_ref="openclaw cron list --json"
payload_json=""
if [[ -n "$CRON_JSON_PATH" ]]; then
  source_evidence_ref="$CRON_JSON_PATH"
  if [[ ! -f "$CRON_JSON_PATH" ]]; then
    detail="path=$CRON_JSON_PATH"
    echo "BLOCKER: no_nudge_cron_guard_cron_contract_drift; reason=cron_json_input_missing; ${detail}"
    _emit_failure_json_if_requested \
      "cron_contract_drift" \
      "cron_json_input_missing" \
      "$detail" \
      "contract:cron_json_input_missing" \
      "0" \
      "$CRON_JSON_PATH"
    [[ "$STRICT_EXIT" -eq 1 ]] && exit 1 || exit 0
  fi
  payload_json="$(cat "$CRON_JSON_PATH")"
else
  set +e
  payload_json="$(openclaw cron list --json 2>/tmp/no_nudge_continuity_cron_guard.err)"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    err_raw="$(cat /tmp/no_nudge_continuity_cron_guard.err 2>/dev/null || true)"
    err="$(_sanitize_inline_text "$err_raw")"
    if _is_gateway_connectivity_failure "$err"; then
      detail="rc=$rc; err=${err:0:180}"
      echo "BLOCKER: no_nudge_cron_guard_gateway_connectivity_failure; reason=cron_list_unreachable; ${detail}"
      _emit_failure_json_if_requested \
        "gateway_connectivity_failure" \
        "cron_list_unreachable" \
        "$detail" \
        "connectivity:cron_list_unreachable" \
        "1" \
        "openclaw cron list --json"
    else
      detail="rc=$rc; err=${err:0:180}"
      echo "BLOCKER: no_nudge_cron_guard_cron_contract_drift; reason=cron_list_failed; ${detail}"
      _emit_failure_json_if_requested \
        "cron_contract_drift" \
        "cron_list_failed" \
        "$detail" \
        "contract:cron_list_failed" \
        "0" \
        "openclaw cron list --json"
    fi
    [[ "$STRICT_EXIT" -eq 1 ]] && exit 1 || exit 0
  fi
fi

summary_json="$(python3 - "$payload_json" "$REMINDER_NAMES_CSV" "$source_evidence_ref" "$GUARD_SCHEMA_VERSION" "$GUARD_FAILURE_TAXONOMY_VERSION" "$GUARD_CLASSIFICATION_TAXONOMY_VERSION" <<'PY'
import json
import sys

raw = sys.argv[1]
reminder_names_csv = sys.argv[2]
source_evidence_ref = str(sys.argv[3] or "").strip()
schema_version = str(sys.argv[4] or "openclaw.no_nudge_continuity_cron_guard.summary.v2")
failure_taxonomy_version = str(sys.argv[5] or "openclaw.no_nudge_continuity_cron_guard.failure_taxonomy.v1")
classification_taxonomy_version = str(sys.argv[6] or "openclaw.no_nudge_continuity_cron_guard.classification.v1")

reminder_names = []
for token in reminder_names_csv.split(","):
    name = token.strip()
    if not name or name in reminder_names:
        continue
    reminder_names.append(name)
reminder_name_set = set(reminder_names)


def dedupe(items):
    out = []
    seen = set()
    for item in items:
        txt = str(item or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def reason_to_policy_codes(reason: str):
    lower = str(reason or "").strip().lower()
    if lower.startswith("sessiontarget="):
        return ["policy:session_target_not_isolated"]
    if lower.startswith("payload.kind="):
        return ["policy:payload_kind_not_agent_turn"]
    if lower.startswith("delivery.mode="):
        return ["policy:delivery_mode_not_none"]
    if "missing blocker contract" in lower:
        return ["policy:message_missing_blocker_marker"]
    if "missing no_reply contract" in lower:
        return ["policy:message_missing_no_reply_marker"]
    if "forwards non-blocker protocol lines" in lower:
        codes = ["policy:message_forwards_non_blocker_protocol"]
        if "ready:" in lower:
            codes.append("policy:message_forwards_ready")
        if "progress:" in lower:
            codes.append("policy:message_forwards_progress")
        if "blocker_json:" in lower:
            codes.append("policy:message_forwards_blocker_json")
        return codes
    if "missing enabled expected reminder job" in lower:
        return ["policy:missing_enabled_expected_reminder_job"]
    if "duplicate enabled expected reminder jobs" in lower:
        return ["policy:duplicate_enabled_expected_reminder_jobs"]
    return ["policy:unknown_violation"]


def failure_codes_payload(*, policy_codes=None, contract_codes=None, connectivity_codes=None):
    policy_codes = dedupe(policy_codes or [])
    contract_codes = dedupe(contract_codes or [])
    connectivity_codes = dedupe(connectivity_codes or [])
    all_codes = dedupe(policy_codes + contract_codes + connectivity_codes)
    return {
        "all": all_codes,
        "critical": all_codes,
        "policy": policy_codes,
        "contract": contract_codes,
        "connectivity": connectivity_codes,
    }


def base_payload() -> dict:
    return {
        "schema_version": schema_version,
        "classification_taxonomy_version": classification_taxonomy_version,
        "failure_taxonomy_version": failure_taxonomy_version,
        "checked": 0,
        "checked_jobs": [],
        "coverage": {
            "expected_enabled_reminder_names": reminder_names,
            "missing_enabled_reminder_names": reminder_names,
            "duplicate_enabled_reminder_names": [],
            "enabled_counts": {name: 0 for name in reminder_names},
        },
        "violations": [],
        "evidence_refs": dedupe(
            [
                "ops/openclaw/no_nudge_continuity_cron_guard.sh",
                source_evidence_ref,
            ]
        ),
        "classification_bucket": "contract",
        "fail_close_triggered": True,
        "failure_category": "cron_contract_fail_close",
        "failure_code": "contract:unknown_failure",
        "failure_retryable": False,
        "failure_codes": failure_codes_payload(contract_codes=["contract:unknown_failure"]),
    }


try:
    obj = json.loads(raw)
except Exception as exc:
    contract_codes = ["contract:invalid_cron_json"]
    payload = base_payload()
    payload.update(
        {
            "ok": False,
            "classification": "cron_contract_drift",
            "classification_bucket": "contract",
            "error": "invalid_cron_json",
            "detail": str(exc),
            "fail_close_triggered": True,
            "failure_category": "cron_contract_fail_close",
            "failure_code": contract_codes[0],
            "failure_retryable": False,
            "failure_codes": failure_codes_payload(contract_codes=contract_codes),
            "violations": [
                {
                    "id": "",
                    "name": "payload",
                    "reasons": ["invalid cron json payload"],
                    "codes": contract_codes,
                    "bucket": "contract",
                }
            ],
        }
    )
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

jobs = obj.get("jobs") if isinstance(obj, dict) else []
if not isinstance(jobs, list):
    jobs = []

checked = []
violations = []
policy_codes = []
enabled_counts = {name: 0 for name in reminder_names}
enabled_ids = {name: [] for name in reminder_names}

for job in jobs:
    if not isinstance(job, dict):
        continue

    name = str(job.get("name") or "").strip()
    if reminder_name_set:
        if name not in reminder_name_set:
            continue
    elif not name.startswith("continuity:"):
        continue

    if not bool(job.get("enabled", False)):
        continue

    payload = job.get("payload") or {}
    delivery = job.get("delivery") or {}

    session_target = str(job.get("sessionTarget") or "").strip()
    payload_kind = str(payload.get("kind") or "").strip()
    delivery_mode = str(delivery.get("mode") or "").strip()
    message = str(payload.get("message") or "")
    job_id = str(job.get("id") or "")

    if reminder_name_set and name in enabled_counts:
        enabled_counts[name] = int(enabled_counts.get(name) or 0) + 1
        enabled_ids.setdefault(name, []).append(job_id)

    checked.append(
        {
            "id": job_id,
            "name": name,
            "sessionTarget": session_target,
            "payloadKind": payload_kind,
            "deliveryMode": delivery_mode,
        }
    )

    reasons = []
    if session_target != "isolated":
        reasons.append(f"sessionTarget={session_target or '<empty>'}")
    if payload_kind != "agentTurn":
        reasons.append(f"payload.kind={payload_kind or '<empty>'}")
    if delivery_mode != "none":
        reasons.append(f"delivery.mode={delivery_mode or '<empty>'}")
    if "BLOCKER:" not in message:
        reasons.append("payload.message missing BLOCKER contract")
    if "NO_REPLY" not in message:
        reasons.append("payload.message missing NO_REPLY contract")

    noisy_tokens = [tok for tok in ("READY:", "PROGRESS:", "BLOCKER_JSON:") if tok in message]
    if noisy_tokens:
        reasons.append("payload.message forwards non-blocker protocol lines: " + ",".join(noisy_tokens))

    if reasons:
        codes = dedupe(code for reason in reasons for code in reason_to_policy_codes(reason))
        policy_codes.extend(codes)
        violations.append(
            {
                "id": job_id,
                "name": name,
                "reasons": reasons,
                "codes": codes,
                "bucket": "policy",
            }
        )

missing_names = []
duplicate_names = []
if reminder_names:
    for name in reminder_names:
        count = int(enabled_counts.get(name) or 0)
        if count <= 0:
            missing_names.append(name)
            code = "policy:missing_enabled_expected_reminder_job"
            policy_codes.append(code)
            violations.append(
                {
                    "id": "",
                    "name": name,
                    "reasons": ["missing enabled expected reminder job"],
                    "codes": [code],
                    "bucket": "policy",
                }
            )
        elif count > 1:
            duplicate_names.append(name)
            ids = [row for row in (enabled_ids.get(name) or []) if row]
            code = "policy:duplicate_enabled_expected_reminder_jobs"
            policy_codes.append(code)
            violations.append(
                {
                    "id": ",".join(ids[:5]),
                    "name": name,
                    "reasons": [
                        f"duplicate enabled expected reminder jobs; count={count}; ids={','.join(ids[:5]) or 'none'}"
                    ],
                    "codes": [code],
                    "bucket": "policy",
                }
            )

policy_codes = dedupe(policy_codes)
ok = len(violations) == 0
failure_code = None if ok else (policy_codes[0] if policy_codes else "policy:violations_present")
failure_codes = failure_codes_payload(policy_codes=policy_codes)

print(
    json.dumps(
        {
            "ok": ok,
            "schema_version": schema_version,
            "classification_taxonomy_version": classification_taxonomy_version,
            "failure_taxonomy_version": failure_taxonomy_version,
            "classification": "ok" if ok else "cron_policy_failed",
            "classification_bucket": "none" if ok else "policy",
            "error": "" if ok else "policy_violations",
            "checked": len(checked),
            "checked_jobs": checked,
            "coverage": {
                "expected_enabled_reminder_names": reminder_names,
                "missing_enabled_reminder_names": missing_names,
                "duplicate_enabled_reminder_names": duplicate_names,
                "enabled_counts": enabled_counts,
            },
            "violations": violations,
            "evidence_refs": dedupe(
                [
                    "ops/openclaw/no_nudge_continuity_cron_guard.sh",
                    source_evidence_ref,
                ]
            ),
            "fail_close_triggered": bool(not ok),
            "failure_category": None if ok else "policy_contract_fail_close",
            "failure_code": failure_code,
            "failure_retryable": None if ok else False,
            "failure_codes": {
                "all": [] if ok else failure_codes.get("all") or [],
                "critical": [] if ok else failure_codes.get("critical") or [],
                "policy": [] if ok else failure_codes.get("policy") or [],
                "contract": [] if ok else failure_codes.get("contract") or [],
                "connectivity": [] if ok else failure_codes.get("connectivity") or [],
            },
        },
        ensure_ascii=False,
    )
)
PY
)"

summary_json="$(_validate_and_reclassify_guard_summary_json "$summary_json")"

readarray -t summary_fields < <(python3 - 3<<<"$summary_json" <<'PY'
import json
import os
import sys

with os.fdopen(3, "r", encoding="utf-8") as raw_fp:
    raw = raw_fp.read()
obj = json.loads(raw)
rows = obj.get("violations") or []
ids = [str(r.get("id") or "") for r in rows if isinstance(r, dict)]
ids = [x for x in ids if x]

print("1" if obj.get("ok") else "0")
print(int(obj.get("checked") or 0))
print(str(obj.get("classification") or ("ok" if obj.get("ok") else "cron_policy_failed")))
print(str(obj.get("error") or ""))
print(str(obj.get("detail") or ""))
print(len(rows))
print(",".join(ids[:5]))
PY
)

ok_flag="${summary_fields[0]:-0}"
checked_count="${summary_fields[1]:-0}"
classification="${summary_fields[2]:-cron_policy_failed}"
summary_error="${summary_fields[3]:-}"
summary_detail_raw="${summary_fields[4]:-}"
viol_count="${summary_fields[5]:-0}"
viol_ids="${summary_fields[6]:-}"

if [[ "$ok_flag" == "1" ]]; then
  echo "READY: no-nudge continuity reminder rails are internal-only; checked=${checked_count}"
elif [[ "$classification" == "gateway_connectivity_failure" ]]; then
  summary_detail="$(_sanitize_inline_text "$summary_detail_raw")"
  echo "BLOCKER: no_nudge_cron_guard_gateway_connectivity_failure; reason=${summary_error:-cron_list_unreachable}; detail=${summary_detail:0:180}"
elif [[ "$classification" == "cron_contract_drift" ]]; then
  summary_detail="$(_sanitize_inline_text "$summary_detail_raw")"
  echo "BLOCKER: no_nudge_cron_guard_cron_contract_drift; reason=${summary_error:-invalid_cron_json}; detail=${summary_detail:0:180}"
else
  echo "BLOCKER: no_nudge_cron_guard_policy_failed; violations=${viol_count}; ids=${viol_ids:-none}"
fi

if [[ "$JSON_OUT" -eq 1 ]]; then
  echo "$summary_json"
fi

if [[ "$STRICT_EXIT" -eq 1 && "$ok_flag" != "1" ]]; then
  exit 1
fi

exit 0
