#!/usr/bin/env bash
# OpenClaw SLO Evaluator (A6 Ops Lane)
# Evaluates critical SLO targets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LATEST_JSON="${WORKSPACE_DIR}/state/continuity/latest/slo_snapshot.json"

mkdir -p "$(dirname "$LATEST_JSON")"

read_nonnegative_int_env() {
    local raw="$1"
    local default="$2"
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        echo "$raw"
    else
        echo "$default"
    fi
}

now_iso_utc() {
    python3 - <<'PY'
import datetime as dt
import os

fixed = os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS", "").strip()
if fixed and fixed.lstrip("-").isdigit():
    now_dt = dt.datetime.fromtimestamp(int(fixed), tz=dt.timezone.utc)
else:
    now_dt = dt.datetime.now(dt.timezone.utc)
print(now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"))
PY
}

artifact_age_meta() {
    local path="$1"
    local timestamp_fields_csv="$2"
    python3 - "$path" "$timestamp_fields_csv" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys
from typing import Any, Dict

path = pathlib.Path(sys.argv[1]).resolve()
field_tokens = [token.strip() for token in str(sys.argv[2] or "").split(",") if token.strip()]


def parse_iso(raw: Any):
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def now_ts() -> int:
    fixed = os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS", "").strip()
    if fixed and fixed.lstrip("-").isdigit():
        return int(fixed)
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


payload: Dict[str, Any] = {}
if path.exists():
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            payload = obj
    except Exception:
        payload = {}

age_source = "mtime"
payload_timestamp_present = False
reference_dt = None

for field in field_tokens:
    raw_value = payload.get(field)
    if raw_value is None:
        continue
    payload_timestamp_present = True
    parsed = parse_iso(raw_value)
    if parsed is not None:
        reference_dt = parsed
        age_source = f"payload:{field}"
        break

if reference_dt is not None:
    ref_ts = int(reference_dt.timestamp())
else:
    ref_ts = int(path.stat().st_mtime)
    if payload_timestamp_present and field_tokens:
        age_source = "mtime_fallback_invalid_payload_timestamp"

age_sec = max(0, now_ts() - ref_ts)
print(json.dumps({"age_sec": age_sec, "source": age_source}, ensure_ascii=False))
PY
}

SLOS="[]"
OVERALL_STATUS="pass"
NOW="$(now_iso_utc)"

VERIFY_MAX_AGE_SEC="$(read_nonnegative_int_env "${OPENCLAW_SLO_VERIFY_MAX_AGE_SEC:-1800}" 1800)"
GROUND_TRUTH_MAX_AGE_SEC="$(read_nonnegative_int_env "${OPENCLAW_SLO_GROUND_TRUTH_MAX_AGE_SEC:-3600}" 3600)"
RESTORE_DRILL_MAX_AGE_SEC="$(read_nonnegative_int_env "${OPENCLAW_SLO_RESTORE_DRILL_MAX_AGE_SEC:-604800}" 604800)"
RESTORE_DRILL_EVIDENCE_PATH="${OPENCLAW_RESTORE_DRILL_EVIDENCE_PATH:-${WORKSPACE_DIR}/state/continuity/latest/restore_drill_latest.json}"

# Helper to add SLO result
add_slo() {
    local id="$1"
    local passed="$2"
    local detail="$3"

    local status="pass"
    if [ "$passed" = "false" ]; then
        status="fail"
        OVERALL_STATUS="fail"
    fi

    SLOS=$(jq --arg id "$id" --arg status "$status" --arg detail "$detail" \
      '. += [{"id": $id, "status": $status, "detail": $detail}]' <<< "$SLOS")
}

# SLO-1: Verification evidence freshness (runtime-enforced)
VERIFY_REPORT="${WORKSPACE_DIR}/state/continuity/latest/verify_last.json"
if [ -f "$VERIFY_REPORT" ]; then
    VERIFY_AGE_META="$(artifact_age_meta "$VERIFY_REPORT" "timestamp,generated_at,updated_at")"
    AGE_SEC="$(jq -r '.age_sec' <<< "$VERIFY_AGE_META")"
    AGE_SOURCE="$(jq -r '.source' <<< "$VERIFY_AGE_META")"

    if [ "$VERIFY_MAX_AGE_SEC" -le 0 ]; then
        add_slo "SLO-1_VERIFY_FRESHNESS" "true" "Verification freshness budget disabled (age=$AGE_SEC sec; source=${AGE_SOURCE})"
    elif [ "$AGE_SEC" -le "$VERIFY_MAX_AGE_SEC" ]; then
        add_slo "SLO-1_VERIFY_FRESHNESS" "true" "Verification evidence age=$AGE_SEC sec (budget ${VERIFY_MAX_AGE_SEC}s; source=${AGE_SOURCE})"
    else
        add_slo "SLO-1_VERIFY_FRESHNESS" "false" "Verification evidence stale (age=$AGE_SEC sec; budget ${VERIFY_MAX_AGE_SEC}s; source=${AGE_SOURCE})"
    fi
else
    add_slo "SLO-1_VERIFY_FRESHNESS" "false" "Verification evidence missing (verify_last.json)"
fi

# SLO-2: Continuity freshness (runtime-enforced)
GT_LATEST="${WORKSPACE_DIR}/state/ground_truth/latest.json"
if [ -f "$GT_LATEST" ]; then
    GT_AGE_META="$(artifact_age_meta "$GT_LATEST" "updated_at,snapshot_ts_utc,timestamp")"
    AGE_SEC="$(jq -r '.age_sec' <<< "$GT_AGE_META")"
    AGE_SOURCE="$(jq -r '.source' <<< "$GT_AGE_META")"

    if [ "$GROUND_TRUTH_MAX_AGE_SEC" -le 0 ]; then
        add_slo "SLO-2_CONTINUITY_FRESHNESS" "true" "Ground truth freshness budget disabled (age=$AGE_SEC sec; source=${AGE_SOURCE})"
    elif [ "$AGE_SEC" -le "$GROUND_TRUTH_MAX_AGE_SEC" ]; then
        add_slo "SLO-2_CONTINUITY_FRESHNESS" "true" "Ground truth age=$AGE_SEC sec (budget ${GROUND_TRUTH_MAX_AGE_SEC}s; source=${AGE_SOURCE})"
    else
        add_slo "SLO-2_CONTINUITY_FRESHNESS" "false" "Ground truth stale (age=$AGE_SEC sec; budget ${GROUND_TRUTH_MAX_AGE_SEC}s; source=${AGE_SOURCE})"
    fi
else
    add_slo "SLO-2_CONTINUITY_FRESHNESS" "false" "Ground truth latest.json missing"
fi

# SLO-4: Restore drill freshness (runtime-enforced)
if [ -f "$RESTORE_DRILL_EVIDENCE_PATH" ]; then
    RESTORE_AGE_META="$(artifact_age_meta "$RESTORE_DRILL_EVIDENCE_PATH" "drilled_at,executed_at,timestamp,generated_at,updated_at")"
    AGE_SEC="$(jq -r '.age_sec' <<< "$RESTORE_AGE_META")"
    AGE_SOURCE="$(jq -r '.source' <<< "$RESTORE_AGE_META")"
    RESTORE_STATUS="$(python3 - "$RESTORE_DRILL_EVIDENCE_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = ""
try:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        status = str(obj.get("status") or "").strip().lower()
except Exception:
    status = ""
print(status)
PY
)"

    if [ -z "$RESTORE_STATUS" ]; then
        add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "false" "Restore drill evidence status missing (age=$AGE_SEC sec; source=${AGE_SOURCE}; path=${RESTORE_DRILL_EVIDENCE_PATH})"
    elif [ "$RESTORE_STATUS" != "pass" ]; then
        add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "false" "Restore drill evidence status not pass (status=${RESTORE_STATUS}; age=$AGE_SEC sec; source=${AGE_SOURCE}; path=${RESTORE_DRILL_EVIDENCE_PATH})"
    elif [ "$RESTORE_DRILL_MAX_AGE_SEC" -le 0 ]; then
        add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "true" "Restore drill freshness budget disabled (age=$AGE_SEC sec; source=${AGE_SOURCE}; path=${RESTORE_DRILL_EVIDENCE_PATH})"
    elif [ "$AGE_SEC" -le "$RESTORE_DRILL_MAX_AGE_SEC" ]; then
        add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "true" "Restore drill evidence age=$AGE_SEC sec (budget ${RESTORE_DRILL_MAX_AGE_SEC}s; source=${AGE_SOURCE}; path=${RESTORE_DRILL_EVIDENCE_PATH})"
    else
        add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "false" "Restore drill evidence stale (age=$AGE_SEC sec; budget ${RESTORE_DRILL_MAX_AGE_SEC}s; source=${AGE_SOURCE}; path=${RESTORE_DRILL_EVIDENCE_PATH})"
    fi
else
    add_slo "SLO-4_RESTORE_DRILL_FRESHNESS" "false" "Restore drill evidence missing (${RESTORE_DRILL_EVIDENCE_PATH})"
fi

# Write output
jq -n \
  --arg ts "$NOW" \
  --arg status "$OVERALL_STATUS" \
  --argjson slos "$SLOS" \
  '{
    timestamp: $ts,
    status: $status,
    enforced_targets: ["SLO-1_VERIFY_FRESHNESS", "SLO-2_CONTINUITY_FRESHNESS", "SLO-4_RESTORE_DRILL_FRESHNESS"],
    deferred_targets: ["SLO-3_QUEUE_ARBITRATION_CYCLE", "SLO-5_WEB_CAPTURE_RUNTIME_RELIABILITY"],
    evaluations: $slos
  }' > "$LATEST_JSON"

echo "Generated SLO snapshot at $LATEST_JSON"
cat "$LATEST_JSON"

if [ "$OVERALL_STATUS" = "fail" ]; then
    exit 2
fi
