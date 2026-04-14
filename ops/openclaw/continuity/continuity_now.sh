#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
REFRESH=0
JSON_OUT=0
STRICT=0

usage() {
  cat <<'EOF'
Usage: continuity_now.sh [options]

Compact operator-facing continuity status surface.
Builds status from latest checkpoint + ground-truth snapshot + verify report + bridge alignment.

Options:
  --refresh    Refresh ground-truth snapshot and continuity bridge before rendering status.
  --json       Print machine JSON output.
  --strict     Exit non-zero when status is not ready (verify BLOCKER, checkpoint BLOCKER,
               critical anomalies, or pointer drift).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --strict)
      STRICT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

read_positive_int_env() {
  local raw="$1"
  local fallback="$2"
  if [[ "$raw" =~ ^[0-9]+$ ]] && (( raw > 0 )); then
    echo "$raw"
  else
    echo "$fallback"
  fi
}

read_nonnegative_int_env() {
  local raw="$1"
  local fallback="$2"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
  else
    echo "$fallback"
  fi
}

REFRESH_PREFLIGHT_TIMEOUT_SEC="$(read_positive_int_env "${OPENCLAW_CONTINUITY_REFRESH_PREFLIGHT_TIMEOUT_SEC:-60}" 60)"
REFRESH_HOOK_TIMEOUT_SEC="$(read_positive_int_env "${OPENCLAW_CONTINUITY_REFRESH_HOOK_TIMEOUT_SEC:-60}" 60)"

REFRESH_STORM_GUARD_WINDOW_SEC="$(read_nonnegative_int_env "${OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_WINDOW_SEC:-60}" 60)"
REFRESH_STORM_GUARD_MAX_RUNS="$(read_nonnegative_int_env "${OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_MAX_RUNS:-4}" 4)"
REFRESH_STORM_GUARD_STATE_PATH_DEFAULT="$ROOT/state/continuity/latest/refresh_storm_guard_state.json"
REFRESH_STORM_GUARD_STATE_PATH="${OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_STATE_PATH:-$REFRESH_STORM_GUARD_STATE_PATH_DEFAULT}"
REFRESH_STORM_GUARD_ENABLED_RAW="${OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_ENABLED:-1}"

REFRESH_STORM_GUARD_DECISION="run"
REFRESH_STORM_GUARD_REASON="not_requested"
REFRESH_STORM_GUARD_WINDOW_COUNT=0
REFRESH_STORM_GUARD_BUDGET_REMAINING=0
REFRESH_STORM_GUARD_COOLDOWN_REMAINING_SEC=0
REFRESH_STORM_GUARD_JSON='{"requested":false,"enabled":true,"decision":"run","reason":"not_requested","window_sec":60,"max_runs":4,"window_count":0,"budget_remaining":0,"cooldown_remaining_sec":0}'

continuity_refresh_storm_guard_decide() {
  local _decision_out
  _decision_out="$(python3 - "$ROOT" "$REFRESH_STORM_GUARD_STATE_PATH" "$REFRESH_STORM_GUARD_ENABLED_RAW" "$REFRESH_STORM_GUARD_WINDOW_SEC" "$REFRESH_STORM_GUARD_MAX_RUNS" <<'PY'
import json
import os
import pathlib
import tempfile
import time


def parse_bool(raw: str, *, default: bool = True) -> bool:
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_nonnegative_int(raw: str, *, default: int) -> int:
    try:
        return max(0, int(str(raw or "").strip()))
    except Exception:
        return int(default)


root = pathlib.Path(os.path.abspath(os.path.expanduser(str(os.sys.argv[1])))).resolve()
state_path_raw = str(os.sys.argv[2] if len(os.sys.argv) > 2 else "").strip()
enabled = parse_bool(os.sys.argv[3] if len(os.sys.argv) > 3 else "1", default=True)
window_sec = parse_nonnegative_int(os.sys.argv[4] if len(os.sys.argv) > 4 else "60", default=60)
max_runs = parse_nonnegative_int(os.sys.argv[5] if len(os.sys.argv) > 5 else "4", default=4)

state_path = pathlib.Path(state_path_raw) if state_path_raw else (root / "state" / "continuity" / "latest" / "refresh_storm_guard_state.json")
if not state_path.is_absolute():
    state_path = (root / state_path).resolve()
else:
    state_path = state_path.resolve()

now_ts = int(time.time())
decision = "run"
reason = "under_budget"
window_count = 0
budget_remaining = max(0, max_runs)
cooldown_remaining_sec = 0

if not enabled:
    decision = "run"
    reason = "guard_disabled"
elif window_sec <= 0 or max_runs <= 0:
    decision = "run"
    reason = "guard_budget_disabled"
else:
    import fcntl

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pathlib.Path(f"{state_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        state = {}
        if state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}

        runs = []
        for raw_ts in state.get("window_runs", []) if isinstance(state.get("window_runs"), list) else []:
            try:
                ts = int(raw_ts)
            except Exception:
                continue
            if ts <= now_ts and (now_ts - ts) <= window_sec:
                runs.append(ts)

        runs.sort()
        prospective_window_count = len(runs) + 1

        if prospective_window_count > max_runs:
            decision = "skip"
            reason = "over_budget_skip"
            oldest = runs[0] if runs else now_ts
            cooldown_remaining_sec = max(0, window_sec - max(0, now_ts - oldest))
        else:
            runs.append(now_ts)
            runs.sort()

        window_count = len(runs)
        budget_remaining = max(0, max_runs - window_count)

        prev_total = 0
        prev_skipped = 0
        try:
            prev_total = max(0, int(state.get("total_invocations") or 0))
        except Exception:
            prev_total = 0
        try:
            prev_skipped = max(0, int(state.get("total_skipped") or 0))
        except Exception:
            prev_skipped = 0

        state_out = {
            "schema_version": "continuity.refresh_storm_guard_state.v1",
            "updated_at": now_ts,
            "window_sec": int(window_sec),
            "max_runs": int(max_runs),
            "window_runs": runs,
            "window_count": int(window_count),
            "decision": decision,
            "reason": reason,
            "budget_remaining": int(budget_remaining),
            "cooldown_remaining_sec": int(cooldown_remaining_sec),
            "total_invocations": int(prev_total + 1),
            "total_skipped": int(prev_skipped + (1 if decision == "skip" else 0)),
        }

        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".refresh_storm_guard.",
            suffix=".tmp",
            dir=str(state_path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(state_out, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            os.replace(tmp_path, state_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

print(f"decision={decision}")
print(f"reason={reason}")
print(f"window_count={window_count}")
print(f"budget_remaining={budget_remaining}")
print(f"cooldown_remaining_sec={cooldown_remaining_sec}")
print(f"window_sec={window_sec}")
print(f"max_runs={max_runs}")
print(f"enabled={'1' if enabled else '0'}")
print(f"state_path={state_path}")
PY
)"

  while IFS='=' read -r key value; do
    case "$key" in
      decision) REFRESH_STORM_GUARD_DECISION="$value" ;;
      reason) REFRESH_STORM_GUARD_REASON="$value" ;;
      window_count) REFRESH_STORM_GUARD_WINDOW_COUNT="$value" ;;
      budget_remaining) REFRESH_STORM_GUARD_BUDGET_REMAINING="$value" ;;
      cooldown_remaining_sec) REFRESH_STORM_GUARD_COOLDOWN_REMAINING_SEC="$value" ;;
      window_sec) REFRESH_STORM_GUARD_WINDOW_SEC="$value" ;;
      max_runs) REFRESH_STORM_GUARD_MAX_RUNS="$value" ;;
      enabled) REFRESH_STORM_GUARD_ENABLED_RAW="$value" ;;
      state_path) REFRESH_STORM_GUARD_STATE_PATH="$value" ;;
    esac
  done <<< "$_decision_out"

  REFRESH_STORM_GUARD_JSON="$(python3 - "$REFRESH_STORM_GUARD_ENABLED_RAW" "$REFRESH_STORM_GUARD_DECISION" "$REFRESH_STORM_GUARD_REASON" "$REFRESH_STORM_GUARD_WINDOW_SEC" "$REFRESH_STORM_GUARD_MAX_RUNS" "$REFRESH_STORM_GUARD_WINDOW_COUNT" "$REFRESH_STORM_GUARD_BUDGET_REMAINING" "$REFRESH_STORM_GUARD_COOLDOWN_REMAINING_SEC" "$REFRESH_STORM_GUARD_STATE_PATH" <<'PY'
import json
import sys


def to_int(raw: str, default: int = 0) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return int(default)


enabled = str(sys.argv[1] if len(sys.argv) > 1 else "1").strip().lower() in {"1", "true", "yes", "on"}
payload = {
    "requested": True,
    "enabled": bool(enabled),
    "decision": str(sys.argv[2] if len(sys.argv) > 2 else "run").strip() or "run",
    "reason": str(sys.argv[3] if len(sys.argv) > 3 else "").strip() or "unknown",
    "window_sec": max(0, to_int(sys.argv[4] if len(sys.argv) > 4 else "60", 60)),
    "max_runs": max(0, to_int(sys.argv[5] if len(sys.argv) > 5 else "4", 4)),
    "window_count": max(0, to_int(sys.argv[6] if len(sys.argv) > 6 else "0", 0)),
    "budget_remaining": max(0, to_int(sys.argv[7] if len(sys.argv) > 7 else "0", 0)),
    "cooldown_remaining_sec": max(0, to_int(sys.argv[8] if len(sys.argv) > 8 else "0", 0)),
    "state_path": str(sys.argv[9] if len(sys.argv) > 9 else "").strip() or None,
}
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)"
}

attach_refresh_storm_guard() {
  local payload_raw="$1"
  python3 - "$payload_raw" "$REFRESH_STORM_GUARD_JSON" <<'PY'
import json
import sys

payload_raw = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
guard_raw = str(sys.argv[2] if len(sys.argv) > 2 else "").strip()

try:
    payload = json.loads(payload_raw) if payload_raw else {}
except Exception:
    payload = {}
if not isinstance(payload, dict):
    payload = {}

try:
    guard = json.loads(guard_raw) if guard_raw else {}
except Exception:
    guard = {}
if not isinstance(guard, dict):
    guard = {}

payload["storm_guard"] = guard
payload["skipped_due_to_storm_guard"] = bool(guard.get("decision") == "skip")

print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
}

REFRESH_HOOKS_JSON='{"requested": false, "hooks": [], "failure_count": 0, "failed_hooks": []}'
REFRESH_PREFLIGHT_JSON='{"requested": false, "ok": true, "stage": "snapshot_ground_truth", "returncode": 0}'

detect_verify_status_evidence_stale() {
  python3 - "$ROOT" "${OPENCLAW_VERIFY_GATE_STATUS_VERIFY_MAX_AGE_SEC:-1800}" <<'PY'
import datetime as dt
import json
import pathlib
import sys


def parse_iso(raw):
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def parse_nonnegative_int(raw, default):
    try:
        return max(0, int(str(raw or "").strip()))
    except Exception:
        return int(default)


root = pathlib.Path(sys.argv[1]).resolve()
verify_max_age_sec = parse_nonnegative_int(sys.argv[2] if len(sys.argv) > 2 else "1800", 1800)
verify_report = root / "state" / "continuity" / "latest" / "verify_last.json"

status = ""
timestamp_raw = ""
failure_reason = ""
age_sec = None

if not verify_report.exists():
    failure_reason = "verify_report_missing"
else:
    try:
        obj = json.loads(verify_report.read_text(encoding="utf-8"))
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        failure_reason = "verify_report_unreadable"
    else:
        status = str(obj.get("status") or "").strip().upper()
        timestamp_raw = str(obj.get("timestamp") or "").strip()
        if not timestamp_raw:
            failure_reason = "verify_report_timestamp_missing"
        else:
            ts = parse_iso(timestamp_raw)
            if ts is None:
                failure_reason = "verify_report_timestamp_invalid"
            else:
                age_sec = max(0, int((dt.datetime.now(dt.timezone.utc) - ts).total_seconds()))
                if verify_max_age_sec > 0 and age_sec > verify_max_age_sec:
                    failure_reason = "verify_report_stale"

repair_needed = bool(status == "READY" and failure_reason == "verify_report_stale")

print(f"repair_needed={'1' if repair_needed else '0'}")
print(f"status={status}")
print(f"failure_reason={failure_reason}")
print(f"age_sec={'' if age_sec is None else age_sec}")
print(f"max_age_sec={verify_max_age_sec}")
PY
}

if [[ "$REFRESH" -eq 1 ]]; then
  continuity_refresh_storm_guard_decide

  if [[ "$REFRESH_STORM_GUARD_DECISION" == "skip" ]]; then
    REFRESH_PREFLIGHT_JSON='{"requested": true, "ok": true, "stage": "snapshot_ground_truth", "returncode": 0}'
    REFRESH_HOOKS_JSON='{"requested": true, "hooks": [], "failure_count": 0, "failed_hooks": []}'

    if [[ "${OPENCLAW_VERIFY_THEN_RESUME_ACTIVE:-0}" != "1" ]]; then
      _stale_repair_needed="0"
      _stale_status=""
      _stale_failure_reason=""
      _stale_age_sec=""
      _stale_max_age_sec=""

      while IFS='=' read -r key value; do
        case "$key" in
          repair_needed) _stale_repair_needed="$value" ;;
          status) _stale_status="$value" ;;
          failure_reason) _stale_failure_reason="$value" ;;
          age_sec) _stale_age_sec="$value" ;;
          max_age_sec) _stale_max_age_sec="$value" ;;
        esac
      done <<< "$(detect_verify_status_evidence_stale)"

      if [[ "$_stale_repair_needed" == "1" ]]; then
        _verify_repair_rc=0
        _verify_repair_timed_out=0
        set +e
        timeout "${REFRESH_HOOK_TIMEOUT_SEC}s" "$ROOT/ops/openclaw/continuity/verify_then_resume.sh" --skip-baseline-checks --status-evidence-repair >/dev/null 2>&1
        _verify_repair_rc=$?
        set -e
        if [[ "$_verify_repair_rc" -eq 124 ]]; then
          _verify_repair_timed_out=1
        fi

        REFRESH_HOOKS_JSON="$(python3 - "$_verify_repair_rc" "$_verify_repair_timed_out" "$REFRESH_HOOK_TIMEOUT_SEC" "$ROOT" "$_stale_status" "$_stale_failure_reason" "$_stale_age_sec" "$_stale_max_age_sec" <<'PY'
import json
import pathlib
import sys


def to_int(raw, default=0):
    try:
        return int(str(raw or "").strip())
    except Exception:
        return int(default)


rc = to_int(sys.argv[1] if len(sys.argv) > 1 else "1", 1)
timed_out = str(sys.argv[2] if len(sys.argv) > 2 else "0").strip() == "1"
timeout_sec = to_int(sys.argv[3] if len(sys.argv) > 3 else "0", 0)
root = pathlib.Path(sys.argv[4] if len(sys.argv) > 4 else "").resolve()
verify_status = str(sys.argv[5] if len(sys.argv) > 5 else "").strip()
failure_reason = str(sys.argv[6] if len(sys.argv) > 6 else "").strip()
age_sec_raw = str(sys.argv[7] if len(sys.argv) > 7 else "").strip()
max_age_sec_raw = str(sys.argv[8] if len(sys.argv) > 8 else "").strip()

age_sec = to_int(age_sec_raw, 0) if age_sec_raw else None
max_age_sec = to_int(max_age_sec_raw, 0) if max_age_sec_raw else None

hook = {
    "name": "verify_then_resume",
    "command": f"{root / 'ops' / 'openclaw' / 'continuity' / 'verify_then_resume.sh'} --skip-baseline-checks --status-evidence-repair",
    "ok": rc == 0,
    "returncode": rc,
    "fail_soft": True,
    "timed_out": timed_out,
    "timeout_sec": timeout_sec,
    "storm_guard_escape_hatch": "verify_status_evidence_stale",
}

payload = {
    "requested": True,
    "hooks": [hook],
    "failure_count": 0 if rc == 0 else 1,
    "failed_hooks": [] if rc == 0 else ["verify_then_resume"],
    "storm_guard_skip_repair_attempted": True,
    "storm_guard_skip_repair_reason": "verify_status_evidence_stale",
    "status_evidence": {
        "verify_status": verify_status or None,
        "failure_reason": failure_reason or None,
        "age_sec": age_sec,
        "verify_max_age_sec": max_age_sec,
    },
}

print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)"
      fi
    fi
  else
    _snapshot_err_file="$(mktemp)"
    set +e
    timeout "${REFRESH_PREFLIGHT_TIMEOUT_SEC}s" "$ROOT/ops/openclaw/snapshot_ground_truth.sh" >/dev/null 2>"$_snapshot_err_file"
    _snapshot_rc=$?
    set -e
    _snapshot_timed_out=0
    if [[ "$_snapshot_rc" -eq 124 ]]; then
      _snapshot_timed_out=1
    fi
    _snapshot_err="$(tail -c 2000 "$_snapshot_err_file" 2>/dev/null || true)"
    rm -f "$_snapshot_err_file"

    REFRESH_PREFLIGHT_JSON="$(python3 - "$_snapshot_rc" "$_snapshot_err" "$_snapshot_timed_out" "$REFRESH_PREFLIGHT_TIMEOUT_SEC" <<'PY'
import json
import sys

rc = 1
try:
    rc = int(sys.argv[1])
except Exception:
    rc = 1
stderr_tail = (sys.argv[2] if len(sys.argv) > 2 else "").strip()
timed_out = str(sys.argv[3] if len(sys.argv) > 3 else "0").strip() == "1"
try:
    timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 else None
except Exception:
    timeout_sec = None

if timed_out and not stderr_tail and timeout_sec is not None:
    stderr_tail = f"snapshot_ground_truth timeout after {timeout_sec}s"

payload = {
    "requested": True,
    "ok": rc == 0,
    "stage": "snapshot_ground_truth",
    "returncode": rc,
    "timed_out": timed_out,
}
if timeout_sec is not None:
    payload["timeout_sec"] = timeout_sec
if stderr_tail:
    payload["stderr_tail"] = stderr_tail[-500:]
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)"

    if [[ "$_snapshot_rc" -eq 0 ]]; then
      declare -a _refresh_hook_rows=()
      run_refresh_hook() {
        local hook_name="$1"
        shift
        local rc=0
        local timed_out=0
        set +e
        timeout "${REFRESH_HOOK_TIMEOUT_SEC}s" "$@" >/dev/null 2>&1
        rc=$?
        set -e
        if [[ "$rc" -eq 124 ]]; then
          timed_out=1
        fi
        _refresh_hook_rows+=("${hook_name}:${rc}:${timed_out}:${REFRESH_HOOK_TIMEOUT_SEC}:$*")
      }

      run_refresh_hook "sync_latest_artifacts" env OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="continuity_now.sh:refresh_hook:sync_latest_artifacts" "$ROOT/ops/openclaw/continuity/sync_latest_artifacts.sh" --skip-render
      # Avoid recursive contract-probe loops during continuity refresh:
      # verify_then_resume baseline includes validate_contracts.sh, which probes
      # handover_latest/operator_mission_control (both call continuity_current -> continuity_now).
      # During refresh we always use the narrow status-evidence-repair path so
      # freshness repair stays bounded and does not stall on full strict-autonomy/A6 runs.
      # When already inside verify_then_resume, skip the nested fail-soft verify refresh hook entirely.
      if [[ "${OPENCLAW_VERIFY_THEN_RESUME_ACTIVE:-0}" != "1" ]]; then
        run_refresh_hook "verify_then_resume" "$ROOT/ops/openclaw/continuity/verify_then_resume.sh" --skip-baseline-checks --status-evidence-repair
      else
        _refresh_hook_rows+=("verify_then_resume:0:0:${REFRESH_HOOK_TIMEOUT_SEC}:skipped_nested_verify_then_resume")
      fi
      run_refresh_hook "gtc_v2_sync" env OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="continuity_now.sh:refresh_hook:gtc_v2_sync" "$ROOT/ops/openclaw/continuity/gtc_v2_sync.sh"

      REFRESH_HOOKS_JSON="$(python3 - "${_refresh_hook_rows[@]}" <<'PY'
import json
import sys

rows = list(sys.argv[1:])
hooks = []
failed = []
for row in rows:
    parts = row.split(":", 4)
    if len(parts) != 5:
        continue
    name, rc_txt, timed_out_txt, timeout_txt, cmd = parts
    try:
        rc = int(rc_txt)
    except Exception:
        rc = 1
    timed_out = str(timed_out_txt).strip() == "1"
    try:
        timeout_sec = int(timeout_txt)
    except Exception:
        timeout_sec = None
    ok = rc == 0
    entry = {
        "name": name,
        "command": cmd,
        "ok": ok,
        "returncode": rc,
        "fail_soft": True,
        "timed_out": timed_out,
    }
    if timeout_sec is not None:
        entry["timeout_sec"] = timeout_sec
    hooks.append(entry)
    if not ok:
        failed.append(name)

payload = {
    "requested": True,
    "hooks": hooks,
    "failure_count": len(failed),
    "failed_hooks": failed,
}
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)"
    else
      REFRESH_HOOKS_JSON='{"requested": true, "hooks": [], "failure_count": 0, "failed_hooks": [], "skipped_due_to_preflight_failure": true}'
    fi
  fi
fi

if [[ "$REFRESH" -eq 1 ]]; then
  REFRESH_PREFLIGHT_JSON="$(attach_refresh_storm_guard "$REFRESH_PREFLIGHT_JSON")"
  REFRESH_HOOKS_JSON="$(attach_refresh_storm_guard "$REFRESH_HOOKS_JSON")"
fi

export OPENCLAW_CONTINUITY_REFRESH_HOOKS_JSON="$REFRESH_HOOKS_JSON"
export OPENCLAW_CONTINUITY_REFRESH_PREFLIGHT_JSON="$REFRESH_PREFLIGHT_JSON"
export OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_JSON="$REFRESH_STORM_GUARD_JSON"

python3 - "$ROOT" "$JSON_OUT" "$STRICT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
strict = bool(int(sys.argv[3]))

LEGACY_ROOT_LITERAL = "/home/yeqiuqiu/clawd-architect"


def shell_cmd_for(rel_path: str, *args: str) -> str:
    script_path = (root / rel_path).resolve()
    base = f"bash {shlex.quote(str(script_path))}"
    return f"{base} {' '.join(args)}".strip()


def cat_cmd_for(path_value: Any) -> str:
    path = pathlib.Path(str(path_value or "").strip())
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return f"cat {shlex.quote(str(path))}"


def normalize_operator_command(raw: Any) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return txt
    return txt.replace(LEGACY_ROOT_LITERAL, str(root))


def summarize_a6_observability_failures(results: Any) -> Dict[str, Any]:
    failed_components: List[str] = []
    failed_commands: List[str] = []
    if not isinstance(results, list):
        return {
            "failed_components": failed_components,
            "failed_commands": failed_commands,
            "failed_count": 0,
            "only_layered_health_snapshot": False,
        }

    for row in results:
        if not isinstance(row, dict):
            continue
        ok_raw = row.get("ok")
        if ok_raw is True:
            continue
        command = str(row.get("command") or "").strip()
        if not command:
            continue
        failed_commands.append(command)
        if "layered_health_snapshot.sh" in command:
            component = "layered_health_snapshot"
        elif "slo_evaluator_snapshot.sh" in command:
            component = "slo_evaluator_snapshot"
        elif "a6_multi_host_jitter_harness.py" in command:
            component = "a6_multi_host_jitter_harness"
        else:
            component = "unknown"
        if component not in failed_components:
            failed_components.append(component)

    return {
        "failed_components": failed_components,
        "failed_commands": failed_commands,
        "failed_count": len(failed_commands),
        "only_layered_health_snapshot": bool(failed_components and set(failed_components).issubset({"layered_health_snapshot"})),
    }


cmd_queue_ready_list_json = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "ready-list", "--json")
cmd_cont_queue_sync_json = shell_cmd_for("ops/openclaw/continuity.sh", "queue-sync", "--json")
cmd_queue_sync_internal = shell_cmd_for("ops/openclaw/continuity/queue_sync_from_autopilot_json.sh")
cmd_queue_requeue_orphaned_apply_json = shell_cmd_for(
    "ops/openclaw/continuity/queue_arbitrator.sh",
    "remediate",
    "--requeue-orphaned-running",
    "--apply",
    "--json",
)
cmd_snapshot_ground_truth = shell_cmd_for("ops/openclaw/snapshot_ground_truth.sh")
cmd_queue_trace_prefix = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "trace")
cmd_cont_history_json = shell_cmd_for("ops/openclaw/continuity/history.sh", "--source-preset", "control-plane", "--since-checkpoint", "latest", "--hours", "24", "--json")
cmd_idle_lane_watchdog_history_json = shell_cmd_for(
    "ops/openclaw/continuity/history.sh",
    "--source-preset",
    "watchdogs",
    "--source",
    "watchdog.no_nudge_continuity",
    "--hours",
    "24",
    "--include-suppressed",
    "--json",
)
cmd_cont_verify_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify", "--json")
cmd_cont_verify_gate_status_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify-gate-status", "--json")
cmd_cont_worker_health_canary_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "worker-health-canary", "--json")
cmd_cont_failover_stress_runtime_evidence_refresh_json = shell_cmd_for(
    "ops/openclaw/continuity.sh",
    "failover-stress-runtime-evidence",
    "--json",
)
cmd_read_execution_supervisor_worker_health_canary_json = cat_cmd_for(
    "state/continuity/latest/execution_supervisor_worker_health_canary_latest.json"
)
cmd_read_failover_stress_runtime_evidence_json = cat_cmd_for(
    "state/continuity/latest/failover_stress_runtime_evidence.json"
)
cmd_parity_force = shell_cmd_for("ops/openclaw/run_competitive_parity_harness.sh", "--force")
cmd_db_integrity_strict_json = shell_cmd_for("ops/openclaw/continuity/db_integrity_check.sh", "--strict", "--json")
cmd_queue_locks_active_json = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "locks", "--active-only", "--json")
cmd_queue_remediate_extended_json = shell_cmd_for(
    "ops/openclaw/continuity/queue_arbitrator.sh",
    "remediate",
    "--expire-overdue-locks",
    "--release-terminal-locks",
    "--requeue-resolved-blocked",
    "--requeue-orphaned-running",
    "--json",
)
cmd_read_orphaned_auto_json = cat_cmd_for("state/continuity/latest/orphaned_running_auto_remediation_now_latest.json")
cmd_read_stale_wave_auto_json = cat_cmd_for("state/continuity/latest/queue_stale_wave_auto_remediation_now_latest.json")
reset_ready_refresh_latest_rel = "state/continuity/latest/reset_ready_refresh_latest.json"
cmd_cont_reset_ready_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "reset-ready-refresh", "--json")
cmd_read_reset_ready_refresh_latest_json = cat_cmd_for(reset_ready_refresh_latest_rel)
cmd_watchdog = shell_cmd_for("ops/openclaw/run_no_nudge_continuity_watchdog.sh")
cmd_read_idle_lane_trace_json = cat_cmd_for("state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json")
cmd_read_execution_frontier_controller_trace_json = cat_cmd_for(
    "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"
)
cmd_read_execution_frontier_controller_history_json = cat_cmd_for(
    "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl"
)
cmd_read_execution_frontier_enforcement_latch_json = cat_cmd_for(
    "state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json"
)
cmd_read_execution_frontier_enforcement_latch_history_json = cat_cmd_for(
    "state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl"
)
cmd_read_autonomous_execution_intent_json = cat_cmd_for(
    "state/continuity/latest/autonomous_execution_intent_latest.json"
)
cmd_read_autonomous_execution_intent_history_json = cat_cmd_for(
    "state/continuity/history/autonomous_execution_intent_history.jsonl"
)
cmd_web_capture_auto_dry_json = shell_cmd_for("ops/openclaw/run_web_capture_macro.sh", "--mode", "auto", "--dry-run", "--json")
cmd_web_capture_scheduler_dry_json = shell_cmd_for("ops/openclaw/run_web_capture_scheduler.sh", "--dry-run", "--json")
cmd_reconcile_with_token = f"{shell_cmd_for('ops/openclaw/continuity.sh')} --action-token <action_token> reconcile"

refresh_hooks: Dict[str, Any] = {
    "requested": False,
    "hooks": [],
    "failure_count": 0,
    "failed_hooks": [],
}
refresh_hooks_raw = os.environ.get("OPENCLAW_CONTINUITY_REFRESH_HOOKS_JSON", "").strip()
if refresh_hooks_raw:
    try:
        parsed_refresh_hooks = json.loads(refresh_hooks_raw)
        if isinstance(parsed_refresh_hooks, dict):
            refresh_hooks = parsed_refresh_hooks
    except Exception:
        refresh_hooks = {
            "requested": True,
            "hooks": [],
            "failure_count": 1,
            "failed_hooks": ["refresh_hooks_parse_failed"],
            "parse_error": "OPENCLAW_CONTINUITY_REFRESH_HOOKS_JSON invalid",
        }

refresh_preflight: Dict[str, Any] = {
    "requested": False,
    "ok": True,
    "stage": "snapshot_ground_truth",
    "returncode": 0,
}
refresh_preflight_raw = os.environ.get("OPENCLAW_CONTINUITY_REFRESH_PREFLIGHT_JSON", "").strip()
if refresh_preflight_raw:
    try:
        parsed_refresh_preflight = json.loads(refresh_preflight_raw)
        if isinstance(parsed_refresh_preflight, dict):
            refresh_preflight = parsed_refresh_preflight
    except Exception:
        refresh_preflight = {
            "requested": True,
            "ok": False,
            "stage": "snapshot_ground_truth",
            "returncode": 1,
            "parse_error": "OPENCLAW_CONTINUITY_REFRESH_PREFLIGHT_JSON invalid",
        }

_refresh_preflight_requested_raw = refresh_preflight.get("requested")
if isinstance(_refresh_preflight_requested_raw, str):
    refresh_preflight_requested = _refresh_preflight_requested_raw.strip().lower() in {"1", "true", "yes"}
else:
    refresh_preflight_requested = bool(_refresh_preflight_requested_raw)

_refresh_preflight_ok_raw = refresh_preflight.get("ok", True)
if isinstance(_refresh_preflight_ok_raw, str):
    refresh_preflight_ok = _refresh_preflight_ok_raw.strip().lower() in {"1", "true", "yes"}
else:
    refresh_preflight_ok = bool(_refresh_preflight_ok_raw)

refresh_preflight["requested"] = refresh_preflight_requested
refresh_preflight["ok"] = refresh_preflight_ok
refresh_preflight_failed = refresh_preflight_requested and not refresh_preflight_ok

refresh_storm_guard: Dict[str, Any] = {
    "requested": False,
    "enabled": True,
    "decision": "run",
    "reason": "not_requested",
    "window_sec": 60,
    "max_runs": 4,
    "window_count": 0,
    "budget_remaining": 0,
    "cooldown_remaining_sec": 0,
    "state_path": None,
}
refresh_storm_guard_raw = os.environ.get("OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_JSON", "").strip()
if refresh_storm_guard_raw:
    try:
        parsed_refresh_storm_guard = json.loads(refresh_storm_guard_raw)
        if isinstance(parsed_refresh_storm_guard, dict):
            refresh_storm_guard = parsed_refresh_storm_guard
    except Exception:
        refresh_storm_guard = {
            "requested": True,
            "enabled": True,
            "decision": "skip",
            "reason": "parse_error",
            "window_sec": 60,
            "max_runs": 4,
            "window_count": 0,
            "budget_remaining": 0,
            "cooldown_remaining_sec": 0,
            "state_path": None,
            "parse_error": "OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_JSON invalid",
        }

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from coherence_tuple import build_coherence_tuple
except Exception:  # pragma: no cover
    build_coherence_tuple = None

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from continuity_policy import (
        DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC as _DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC,
        DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC as _DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC,
        DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC as _DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC,
        DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC as _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
        DRIFT_REASON_SET as _DRIFT_REASON_SET,
        project_reset_ready_refresh_escalation_reason as _project_reset_ready_refresh_escalation_reason,
        project_reset_ready_refresh_posture as _project_reset_ready_refresh_posture,
        read_nonnegative_int_env as _read_nonnegative_int_env,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC = 1800
    _DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC = 1800
    _DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC = 21600
    _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC = 21600
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }

    def _read_nonnegative_int_env(name: str, *, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(int(default)))))
        except Exception:
            return int(default)

    def _project_reset_ready_refresh_posture(
        *,
        surface: Any = None,
        latest_payload: Any = None,
        path: Any = None,
        sha256: Any = None,
        present: Any = None,
        now_ts: Any = None,
        freshness_max_age_sec: Any = None,
    ) -> Dict[str, Any]:
        surface_map = surface if isinstance(surface, dict) else {}
        latest_map = latest_payload if isinstance(latest_payload, dict) else {}

        path_text = str(path or surface_map.get("path") or "").strip()
        sha_text = str(sha256 or surface_map.get("sha256") or "").strip() or None

        if isinstance(present, bool):
            present_value = present
        else:
            present_value = bool(surface_map.get("present") is True or bool(latest_map))

        ok = surface_map.get("ok") if isinstance(surface_map.get("ok"), bool) else None
        if ok is None and isinstance(latest_map.get("ok"), bool):
            ok = latest_map.get("ok")

        phase = str(surface_map.get("phase") or latest_map.get("phase") or "").strip() or None
        if phase is None and ok is True:
            phase = "complete"

        partial_refresh = surface_map.get("partial_refresh") if isinstance(surface_map.get("partial_refresh"), dict) else {}
        if not partial_refresh and isinstance(latest_map.get("partial_refresh"), dict):
            partial_refresh = latest_map.get("partial_refresh")

        def _partial_flag(name: str) -> Optional[bool]:
            raw_value = partial_refresh.get(name)
            return raw_value if isinstance(raw_value, bool) else None

        partial_current = _partial_flag("current_refreshed")
        partial_proof = _partial_flag("proof_refreshed")
        partial_handover = _partial_flag("handover_refreshed")

        explicit_partial_failure = surface_map.get("partial_failure")
        if isinstance(explicit_partial_failure, bool):
            partial_failure = explicit_partial_failure
        else:
            partial_failure = bool(
                present_value
                and any(value is False for value in [partial_current, partial_proof, partial_handover])
            )

        error_code = str(
            surface_map.get("error_code")
            or (((latest_map.get("error") or {}).get("code")) if isinstance(latest_map.get("error"), dict) else "")
            or ""
        ).strip() or None

        explicit_degraded = surface_map.get("degraded")
        if isinstance(explicit_degraded, bool):
            degraded = explicit_degraded
        else:
            degraded = bool(present_value and (ok is False or partial_failure))

        generated_at = str(surface_map.get("generated_at") or latest_map.get("generated_at") or "").strip() or None

        def _coerce_nonnegative_int(raw: Any) -> Optional[int]:
            if isinstance(raw, bool):
                return None
            try:
                return max(0, int(raw))
            except Exception:
                return None

        freshness_limit_sec = _coerce_nonnegative_int(surface_map.get("freshness_limit_sec"))
        if freshness_limit_sec is None:
            freshness_limit_sec = _coerce_nonnegative_int(freshness_max_age_sec)
        if freshness_limit_sec is None:
            freshness_limit_sec = int(_DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC)

        age_sec = _coerce_nonnegative_int(surface_map.get("age_sec"))
        fresh = surface_map.get("fresh") if isinstance(surface_map.get("fresh"), bool) else None
        stale = surface_map.get("stale") if isinstance(surface_map.get("stale"), bool) else None

        if fresh is None and isinstance(stale, bool):
            fresh = not stale
        if stale is None and isinstance(fresh, bool):
            stale = not fresh

        if freshness_limit_sec > 0 and (age_sec is None or fresh is None):
            generated_dt = None
            if generated_at:
                generated_txt = generated_at[:-1] + "+00:00" if generated_at.endswith("Z") else generated_at
                try:
                    generated_dt = dt.datetime.fromisoformat(generated_txt)
                    if generated_dt.tzinfo is None:
                        generated_dt = generated_dt.replace(tzinfo=dt.timezone.utc)
                except Exception:
                    generated_dt = None
            now_ts_int = _coerce_nonnegative_int(now_ts)
            if generated_dt is not None and now_ts_int is not None:
                derived_age_sec = max(0, int(now_ts_int - int(generated_dt.timestamp())))
                age_sec = derived_age_sec
                fresh = derived_age_sec <= freshness_limit_sec
                stale = not fresh

        if stale is None:
            stale = fresh is False

        status = "missing"
        if present_value:
            if degraded:
                status = "degraded"
            elif ok is True:
                status = "ok"
            else:
                status = "present"

        recommended_action = None
        if degraded or stale:
            recommended_action = "rerun_reset_ready_refresh"
        elif present_value:
            recommended_action = "inspect_reset_ready_refresh_result"

        return {
            "path": path_text,
            "sha256": sha_text,
            "generated_at": generated_at,
            "present": present_value,
            "status": status,
            "ok": ok,
            "phase": phase,
            "error_code": error_code,
            "freshness_limit_sec": freshness_limit_sec,
            "age_sec": age_sec,
            "fresh": fresh,
            "stale": stale,
            "partial_refresh": {
                "current_refreshed": partial_current,
                "proof_refreshed": partial_proof,
                "handover_refreshed": partial_handover,
            },
            "degraded": degraded,
            "partial_failure": partial_failure,
            "action_required": bool(degraded or stale),
            "recommended_action": recommended_action,
        }

    def _project_reset_ready_refresh_escalation_reason(
        *,
        posture: Any = None,
        degraded: Any = None,
        phase: Any = None,
        error_code: Any = None,
    ) -> Optional[str]:
        posture_map = posture if isinstance(posture, dict) else {}

        if isinstance(degraded, bool):
            degraded_value = degraded
        else:
            degraded_value = posture_map.get("degraded") is True

        if not degraded_value:
            return None

        phase_value = str(phase if phase is not None else posture_map.get("phase") or "").strip()
        error_code_value = str(
            error_code if error_code is not None else posture_map.get("error_code") or ""
        ).strip()

        if phase_value == "alignment_check" and error_code_value == "proof_alignment_mismatch":
            return "reset_ready_refresh_alignment_mismatch"
        return None

try:
    from schema_contract_validation import validate_contract_payload_schema
except Exception:  # pragma: no cover
    validate_contract_payload_schema = None


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def clock_now_dt() -> dt.datetime:
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc)


def clock_now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_iso(s: str):
    if not s:
        return None
    raw = s.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def iso_to_ts(value: Any) -> Optional[int]:
    parsed = parse_iso(str(value or ""))
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    try:
        return int(parsed.timestamp())
    except Exception:
        return None


def age_sec(ts: str):
    d = parse_iso(ts)
    if d is None:
        return None
    now = clock_now_dt()
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return max(0, int((now - d).total_seconds()))


def age_compact(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


def maybe_rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = pathlib.Path(fh.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def run_json_command(
    cmd: List[str],
    env_overrides: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    run_env = None
    if isinstance(env_overrides, dict) and env_overrides:
        run_env = os.environ.copy()
        for key, value in env_overrides.items():
            run_env[str(key)] = str(value)
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, check=False, env=run_env)
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    try:
        payload = json.loads(cp.stdout or "{}")
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def run_tag_from_ts(value: Any):
    try:
        n = int(value)
        if n <= 0:
            return None
        return dt.datetime.fromtimestamp(n, tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def autopilot_step_evidence_refs(ap_state: Dict[str, Any]):
    refs = []
    steps = ap_state.get("steps") or []
    runs_dir = root / "ops" / "autopilot" / "runs"
    repo_path_raw = str(((ap_state.get("repo") or {}).get("path") or "")).strip()
    repo_path = pathlib.Path(repo_path_raw).resolve() if repo_path_raw else None

    artifact_map = {
        "sync_spec_context": ["autopilot_artifacts/spec/spec_backlog_summary.md"],
        "audit_alignment": ["autopilot_artifacts/audit_alignment.md"],
        "audit_runtime_probes": ["autopilot_artifacts/audit_runtime_probes.md"],
        "audit_breaktests": ["autopilot_artifacts/audit_breaktests.md"],
        "synth_fix_plan": ["autopilot_artifacts/fix_plan.md"],
        "apply_fixes": ["autopilot_artifacts/apply_fixes.md", "autopilot_artifacts/p0_progress.md"],
        "quality_gate": ["autopilot_artifacts/quality_gate.md"],
    }

    scored = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            continue
        ts = int(step.get("last_started_ts") or 0)
        scored.append((ts, step_id, step))
    scored.sort(reverse=True)

    for _, step_id, step in scored:
        tag = run_tag_from_ts(step.get("last_started_ts"))
        if tag and runs_dir.exists():
            base = runs_dir / f"{tag}_{step_id}"
            for ext in (".log", ".exit"):
                p = pathlib.Path(str(base) + ext)
                if p.exists():
                    refs.append(maybe_rel(p))

        if repo_path is not None:
            for rel in artifact_map.get(step_id, []):
                p = (repo_path / rel).resolve()
                if p.exists():
                    refs.append(str(p))

        if len(refs) >= 8:
            break

    out = []
    seen = set()
    for ref in refs:
        val = str(ref or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
        if len(out) >= 8:
            break
    return out


latest_dir = root / "state" / "continuity" / "latest"
continuity_now_latest_path = latest_dir / "continuity_now_latest.json"
coherence_stamp_path = latest_dir / "coherence_stamp.json"
coherence_bundle_path = latest_dir / "coherence_bundle_latest.json"
orphaned_running_auto_remediation_state_rel = "state/continuity/latest/orphaned_running_auto_remediation_now_latest.json"
orphaned_running_auto_remediation_state_path = latest_dir / "orphaned_running_auto_remediation_now_latest.json"
queue_stale_wave_auto_remediation_state_rel = "state/continuity/latest/queue_stale_wave_auto_remediation_now_latest.json"
queue_stale_wave_auto_remediation_state_path = latest_dir / "queue_stale_wave_auto_remediation_now_latest.json"
queue_stale_wave_auto_remediation_schema_rel = "ops/openclaw/architecture/schemas/queue_stale_wave_auto_remediation.schema.json"
queue_stale_wave_auto_remediation_schema_path = root / "ops" / "openclaw" / "architecture" / "schemas" / "queue_stale_wave_auto_remediation.schema.json"
latest_pointer = load_json(latest_dir / "latest_pointer.json")
verify_last = load_json(latest_dir / "verify_last.json")
bridge = load_json(latest_dir / "runtime_truth_bridge.json")
reset_ready_refresh_freshness_max_age_sec = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_RESET_READY_REFRESH_MAX_AGE_SEC",
    default=_DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC,
)
reset_ready_refresh_path = latest_dir / "reset_ready_refresh_latest.json"
reset_ready_refresh_latest = load_json(reset_ready_refresh_path)
if not isinstance(reset_ready_refresh_latest, dict):
    reset_ready_refresh_latest = {}
reset_ready_refresh_projection = _project_reset_ready_refresh_posture(
    latest_payload=reset_ready_refresh_latest,
    path=reset_ready_refresh_latest_rel,
    present=reset_ready_refresh_path.exists() and bool(reset_ready_refresh_latest),
    now_ts=clock_now_ts(),
    freshness_max_age_sec=reset_ready_refresh_freshness_max_age_sec,
)
reset_ready_refresh_present = bool(reset_ready_refresh_projection.get("present") is True)
reset_ready_refresh_ok = reset_ready_refresh_projection.get("ok") if isinstance(reset_ready_refresh_projection.get("ok"), bool) else None
reset_ready_refresh_phase = str(reset_ready_refresh_projection.get("phase") or "").strip() or None
reset_ready_refresh_partial = (
    reset_ready_refresh_projection.get("partial_refresh")
    if isinstance(reset_ready_refresh_projection.get("partial_refresh"), dict)
    else {}
)
reset_ready_refresh_partial_current = (
    reset_ready_refresh_partial.get("current_refreshed")
    if isinstance(reset_ready_refresh_partial.get("current_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_proof = (
    reset_ready_refresh_partial.get("proof_refreshed")
    if isinstance(reset_ready_refresh_partial.get("proof_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_handover = (
    reset_ready_refresh_partial.get("handover_refreshed")
    if isinstance(reset_ready_refresh_partial.get("handover_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_failure = bool(reset_ready_refresh_projection.get("partial_failure") is True)
reset_ready_refresh_error_code = str(reset_ready_refresh_projection.get("error_code") or "").strip() or None
reset_ready_refresh_degraded = bool(reset_ready_refresh_projection.get("degraded") is True)
reset_ready_refresh_freshness_limit_sec = (
    int(reset_ready_refresh_projection.get("freshness_limit_sec"))
    if isinstance(reset_ready_refresh_projection.get("freshness_limit_sec"), int)
    else reset_ready_refresh_freshness_max_age_sec
)
reset_ready_refresh_age_sec = (
    int(reset_ready_refresh_projection.get("age_sec"))
    if isinstance(reset_ready_refresh_projection.get("age_sec"), int)
    else None
)
reset_ready_refresh_fresh = (
    reset_ready_refresh_projection.get("fresh")
    if isinstance(reset_ready_refresh_projection.get("fresh"), bool)
    else None
)
reset_ready_refresh_stale = bool(reset_ready_refresh_projection.get("stale") is True)

try:
    coherence_hard_ttl_sec = max(0, int(os.environ.get("OPENCLAW_COHERENCE_HARD_TTL_SEC", "300")))
except Exception:
    coherence_hard_ttl_sec = 300

orphaned_running_min_sec = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_ORPHANED_RUNNING_MIN_SEC",
    default=_DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC,
)

orphaned_running_auto_remediate_enabled = str(
    os.environ.get("OPENCLAW_CONTINUITY_ORPHANED_RUNNING_AUTO_REMEDIATE", "1")
).strip().lower() in {"1", "true", "yes", "y", "on"}

try:
    orphaned_running_auto_remediate_cooldown_sec = max(
        0,
        int(os.environ.get("OPENCLAW_CONTINUITY_ORPHANED_RUNNING_AUTO_REMEDIATE_COOLDOWN_SEC", "900")),
    )
except Exception:
    orphaned_running_auto_remediate_cooldown_sec = 900

queue_stale_wave_ready_idle_sec = max(
    60,
    _read_nonnegative_int_env(
        "OPENCLAW_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC",
        default=_DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC,
    ),
)

queue_stale_wave_auto_remediate_enabled = str(
    os.environ.get("OPENCLAW_CONTINUITY_QUEUE_STALE_WAVE_AUTO_REMEDIATE", "1")
).strip().lower() in {"1", "true", "yes", "y", "on"}

try:
    queue_stale_wave_auto_remediate_cooldown_sec = max(
        0,
        int(os.environ.get("OPENCLAW_CONTINUITY_QUEUE_STALE_WAVE_AUTO_REMEDIATE_COOLDOWN_SEC", "900")),
    )
except Exception:
    queue_stale_wave_auto_remediate_cooldown_sec = 900

try:
    idle_lane_autospawn_max_age_sec = max(
        0,
        int(os.environ.get("OPENCLAW_NO_NUDGE_IDLE_LANE_AUTOSPAWN_MAX_AGE_SEC", "10800")),
    )
except Exception:
    idle_lane_autospawn_max_age_sec = 10800

try:
    execution_frontier_controller_tick_max_age_sec = max(
        0,
        int(os.environ.get("OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_TICK_MAX_AGE_SEC", "10800")),
    )
except Exception:
    execution_frontier_controller_tick_max_age_sec = 10800

checkpoint = {}
checkpoint_path = ""
cp_rel = str(latest_pointer.get("json_path") or "")
if cp_rel:
    cp_abs = (root / cp_rel).resolve()
    if cp_abs.exists():
        checkpoint = load_json(cp_abs)
        checkpoint_path = cp_rel

checkpoint_meta = checkpoint.get("metadata") or {}
checkpoint_obj = checkpoint.get("objective") or {}

checkpoint_status = str(checkpoint_obj.get("status") or "unknown")
checkpoint_id = str(checkpoint_meta.get("checkpoint_id") or latest_pointer.get("checkpoint_id") or "n/a")
checkpoint_created = str(checkpoint_meta.get("created_at") or "")
checkpoint_age = age_sec(checkpoint_created)

verify_status = str(verify_last.get("status") or "unknown")
verify_reason = str(verify_last.get("reason") or "")
verify_ts = str(verify_last.get("timestamp") or "")
verify_age = age_sec(verify_ts)
verify_a6_observability = summarize_a6_observability_failures(
    verify_last.get("a6_observability_results") if isinstance(verify_last, dict) else None
)
verify_a6_failed_components = [
    str(x).strip()
    for x in (verify_a6_observability.get("failed_components") or [])
    if str(x).strip()
]
verify_a6_layered_health_only_failure = bool(verify_a6_observability.get("only_layered_health_snapshot") is True)
verify_strict_payload = (verify_last.get("strict_autonomy_regressions") or {}) if isinstance(verify_last, dict) else {}
if not isinstance(verify_strict_payload, dict):
    verify_strict_payload = {}
verify_strict_enabled = bool(verify_strict_payload.get("enabled"))
verify_strict_source = str(verify_strict_payload.get("source") or "disabled")
verify_strict_effective_source = str(verify_strict_payload.get("effective_source") or verify_strict_source or "disabled")
verify_strict_result = verify_strict_payload.get("result") if isinstance(verify_strict_payload.get("result"), dict) else None
verify_strict_result_ok = None
if isinstance(verify_strict_result, dict):
    verify_strict_result_ok = verify_strict_result.get("ok")
verify_strict_wrapper_effective = verify_strict_payload.get("wrapper_effective") if isinstance(verify_strict_payload.get("wrapper_effective"), dict) else None
verify_strict_wrapper_hint_mismatch = bool(verify_strict_payload.get("wrapper_hint_mismatch") is True)

verify_gate_preflight_required = None
if isinstance(verify_strict_wrapper_effective, dict):
    wrapper_required_raw = verify_strict_wrapper_effective.get("required")
    if isinstance(wrapper_required_raw, bool):
        verify_gate_preflight_required = wrapper_required_raw

verify_gate_preflight: Dict[str, Any] = {
    "available": False,
    "status_source": "verify_last_fallback",
    "strict_autonomy": {
        "enabled": verify_strict_enabled,
        "source": verify_strict_effective_source,
        "required": verify_gate_preflight_required,
        "override": None,
        "override_denied_if_run": None,
    },
    "predicted_gate": {
        "ready_to_run": None,
        "predicted_blocker_reason": None,
    },
    "status_evidence_gate": {
        "failure_reason": None,
        "fresh": None,
        "age_sec": verify_age,
        "verify_max_age_sec": None,
        "verify_max_age_enforced": None,
        "ready_claim_supported": None,
        "run_verify_command": cmd_cont_verify_json,
    },
    "internal_bypass_stage_b": {
        "closeout_ready": None,
        "closeout_failure_reason": None,
        "unknown_callsite_total": None,
        "break_glass_allow_count": None,
        "break_glass_denied_count": None,
        "inspect_audit_command": None,
    },
    "layered_health_gate": {
        "closeout_ready": None,
        "failure_reason": None,
        "layered_health_snapshot_path": None,
        "slo_snapshot_path": None,
        "health_status": None,
        "health_layer": None,
        "restore_slo_status": None,
        "missing_required_lanes": [],
        "failing_required_lanes": [],
        "layer_insufficient_required_lanes": [],
        "inspect_layered_health_command": None,
        "inspect_slo_snapshot_command": None,
        "run_layered_health_command": None,
        "run_slo_snapshot_command": None,
    },
    "failover_stress_runtime_evidence_gate": {
        "active_blocker": None,
        "blocker_reason": None,
        "failure_reason": None,
        "failover_stress_runtime_evidence_path": None,
        "generated_at": None,
        "age_sec": None,
        "fresh": None,
        "max_age_sec": None,
        "overall_verdict": None,
        "publish_chain_verdict": None,
        "publish_assertions_failed": None,
        "repeatability_status": None,
        "repeatability_match": None,
        "repeatability_mismatch_fields": [],
        "active_top_blocker": None,
        "effective_top_blocker": None,
        "inspect_failover_stress_runtime_evidence_command": None,
        "refresh_failover_stress_runtime_evidence_command": None,
        "auto_refresh": None,
    },
    "launch_readiness_severity_gate": {
        "active_blocker": None,
        "blocker_reason": None,
        "failure_reason": None,
        "severity_state": None,
        "severity_reason": None,
        "severity_active": None,
        "non_ready_ticks_consecutive": None,
        "threshold_ticks": None,
        "cohort_worker_count": None,
        "dispatch_qualification_path": None,
        "inspect_dispatch_qualification_command": None,
        "refresh_dispatch_qualification_command": None,
    },
    "launch_readiness_worker_health_canary_gate": {
        "active_blocker": None,
        "blocker_reason": None,
        "failure_reason": None,
        "dispatch_qualification_present": None,
        "dispatch_qualification_failure_reason": None,
        "gate_required": None,
        "artifact_required": None,
        "resource_preflight_required": None,
        "resource_preflight_status": None,
        "resource_preflight_reason": None,
        "resource_preflight_blocking_candidate_count": None,
        "resource_preflight_blocking_task_ids": [],
        "resource_preflight_telemetry_complete": None,
        "resource_preflight_lowest_headroom_pct": None,
        "uncertainty_confidence_score": None,
        "uncertainty_confidence_label": None,
        "uncertainty_confidence_quantiles": {},
        "uncertainty_reasons": [],
        "uncertainty_requires_operator_review": None,
        "worker_health_canary_source": None,
        "worker_health_canary_present": None,
        "worker_health_canary_path": None,
        "worker_health_canary_generated_at": None,
        "worker_health_canary_age_sec": None,
        "worker_health_canary_fresh": None,
        "dispatch_qualification_path": None,
        "inspect_dispatch_qualification_command": None,
        "inspect_worker_health_canary_command": None,
        "refresh_worker_health_canary_command": None,
        "refresh_dispatch_qualification_command": None,
        "first_actionable_command": None,
        "action_priority": None,
    },
    "launch_readiness_probe_execution_gate": {
        "active_blocker": None,
        "blocker_reason": None,
        "failure_reason": None,
        "dispatch_qualification_present": None,
        "dispatch_qualification_failure_reason": None,
        "probe_execution_source": None,
        "probe_execution_status": None,
        "probe_execution_reason": None,
        "pending_worker_count": None,
        "due_now_worker_count": None,
        "overdue_worker_count": None,
        "oldest_due_now_started_at": None,
        "oldest_due_now_worker": None,
        "oldest_due_now_age_sec": None,
        "oldest_overdue_started_at": None,
        "oldest_overdue_worker": None,
        "oldest_overdue_age_sec": None,
        "due_now_active": None,
        "overdue_active": None,
        "probe_execution_plan_path": None,
        "probe_execution_plan_present": None,
        "dispatch_qualification_path": None,
        "inspect_dispatch_qualification_command": None,
        "inspect_probe_execution_plan_command": None,
        "refresh_dispatch_qualification_command": None,
        "first_actionable_command": None,
        "action_priority": None,
    },
}

verify_gate_status_script = root / "ops" / "openclaw" / "continuity" / "verify_gate_status.sh"
if verify_gate_status_script.exists() and os.access(verify_gate_status_script, os.X_OK):
    preflight_payload = run_json_command(
        [str(verify_gate_status_script), "--task", "continuity_now", "--json"],
        env_overrides={
            "OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH": "1",
        },
    )
    if isinstance(preflight_payload, dict):
        strict_preflight = preflight_payload.get("strict_autonomy_regressions") if isinstance(preflight_payload.get("strict_autonomy_regressions"), dict) else {}
        predicted_preflight = preflight_payload.get("predicted_gate") if isinstance(preflight_payload.get("predicted_gate"), dict) else {}
        status_evidence_preflight = preflight_payload.get("status_evidence_gate") if isinstance(preflight_payload.get("status_evidence_gate"), dict) else {}
        internal_bypass_stage_b_preflight = preflight_payload.get("internal_bypass_stage_b") if isinstance(preflight_payload.get("internal_bypass_stage_b"), dict) else {}
        layered_health_preflight = preflight_payload.get("layered_health_gate") if isinstance(preflight_payload.get("layered_health_gate"), dict) else {}
        failover_stress_runtime_preflight = preflight_payload.get("failover_stress_runtime_evidence_gate") if isinstance(preflight_payload.get("failover_stress_runtime_evidence_gate"), dict) else {}
        launch_readiness_severity_preflight = preflight_payload.get("launch_readiness_severity_gate") if isinstance(preflight_payload.get("launch_readiness_severity_gate"), dict) else {}
        launch_readiness_worker_health_canary_preflight = preflight_payload.get("launch_readiness_worker_health_canary_gate") if isinstance(preflight_payload.get("launch_readiness_worker_health_canary_gate"), dict) else {}
        launch_readiness_probe_execution_preflight = preflight_payload.get("launch_readiness_probe_execution_gate") if isinstance(preflight_payload.get("launch_readiness_probe_execution_gate"), dict) else {}
        verify_gate_preflight = {
            "available": True,
            "generated_at": preflight_payload.get("generated_at"),
            "status_source": "verify_gate_status",
            "strict_autonomy": {
                "enabled": bool(strict_preflight.get("enabled") is True),
                "source": str(strict_preflight.get("source") or "disabled"),
                "required": strict_preflight.get("required") if isinstance(strict_preflight.get("required"), bool) else None,
                "override": strict_preflight.get("override"),
                "override_denied_if_run": bool(strict_preflight.get("override_denied_if_run") is True),
            },
            "predicted_gate": {
                "ready_to_run": predicted_preflight.get("ready_to_run") if isinstance(predicted_preflight.get("ready_to_run"), bool) else None,
                "predicted_blocker_reason": predicted_preflight.get("predicted_blocker_reason"),
            },
            "status_evidence_gate": {
                "failure_reason": status_evidence_preflight.get("failure_reason"),
                "fresh": status_evidence_preflight.get("fresh") if isinstance(status_evidence_preflight.get("fresh"), bool) else None,
                "age_sec": status_evidence_preflight.get("age_sec") if isinstance(status_evidence_preflight.get("age_sec"), int) else None,
                "verify_max_age_sec": status_evidence_preflight.get("verify_max_age_sec") if isinstance(status_evidence_preflight.get("verify_max_age_sec"), int) else None,
                "verify_max_age_enforced": status_evidence_preflight.get("verify_max_age_enforced") if isinstance(status_evidence_preflight.get("verify_max_age_enforced"), bool) else None,
                "ready_claim_supported": status_evidence_preflight.get("ready_claim_supported") if isinstance(status_evidence_preflight.get("ready_claim_supported"), bool) else None,
                "run_verify_command": normalize_operator_command(status_evidence_preflight.get("run_verify_command") or cmd_cont_verify_json) or cmd_cont_verify_json,
            },
            "internal_bypass_stage_b": {
                "closeout_ready": internal_bypass_stage_b_preflight.get("closeout_ready") if isinstance(internal_bypass_stage_b_preflight.get("closeout_ready"), bool) else None,
                "closeout_failure_reason": internal_bypass_stage_b_preflight.get("closeout_failure_reason"),
                "unknown_callsite_total": internal_bypass_stage_b_preflight.get("unknown_callsite_total") if isinstance(internal_bypass_stage_b_preflight.get("unknown_callsite_total"), int) else None,
                "break_glass_allow_count": internal_bypass_stage_b_preflight.get("break_glass_allow_count") if isinstance(internal_bypass_stage_b_preflight.get("break_glass_allow_count"), int) else None,
                "break_glass_denied_count": internal_bypass_stage_b_preflight.get("break_glass_denied_count") if isinstance(internal_bypass_stage_b_preflight.get("break_glass_denied_count"), int) else None,
                "inspect_audit_command": normalize_operator_command(internal_bypass_stage_b_preflight.get("inspect_audit_command") or cmd_cont_verify_gate_status_json) or cmd_cont_verify_gate_status_json,
            },
            "layered_health_gate": {
                "closeout_ready": layered_health_preflight.get("closeout_ready") if isinstance(layered_health_preflight.get("closeout_ready"), bool) else None,
                "failure_reason": layered_health_preflight.get("failure_reason"),
                "layered_health_snapshot_path": layered_health_preflight.get("layered_health_snapshot_path"),
                "slo_snapshot_path": layered_health_preflight.get("slo_snapshot_path"),
                "health_status": layered_health_preflight.get("health_status"),
                "health_layer": layered_health_preflight.get("health_layer"),
                "restore_slo_status": layered_health_preflight.get("restore_slo_status"),
                "required_lanes": [
                    str(x).strip()
                    for x in (layered_health_preflight.get("required_lanes") or [])
                    if str(x).strip()
                ],
                "missing_required_lanes": [
                    str(x).strip()
                    for x in (layered_health_preflight.get("missing_required_lanes") or [])
                    if str(x).strip()
                ],
                "failing_required_lanes": [
                    str(x).strip()
                    for x in (layered_health_preflight.get("failing_required_lanes") or [])
                    if str(x).strip()
                ],
                "layer_insufficient_required_lanes": [
                    str(x).strip()
                    for x in (layered_health_preflight.get("layer_insufficient_required_lanes") or [])
                    if str(x).strip()
                ],
                "inspect_layered_health_command": normalize_operator_command(layered_health_preflight.get("inspect_layered_health_command")) if layered_health_preflight.get("inspect_layered_health_command") else None,
                "inspect_slo_snapshot_command": normalize_operator_command(layered_health_preflight.get("inspect_slo_snapshot_command")) if layered_health_preflight.get("inspect_slo_snapshot_command") else None,
                "run_layered_health_command": normalize_operator_command(layered_health_preflight.get("run_layered_health_command")) if layered_health_preflight.get("run_layered_health_command") else None,
                "run_slo_snapshot_command": normalize_operator_command(layered_health_preflight.get("run_slo_snapshot_command")) if layered_health_preflight.get("run_slo_snapshot_command") else None,
            },
            "failover_stress_runtime_evidence_gate": {
                "active_blocker": failover_stress_runtime_preflight.get("active_blocker") if isinstance(failover_stress_runtime_preflight.get("active_blocker"), bool) else None,
                "blocker_reason": failover_stress_runtime_preflight.get("blocker_reason"),
                "failure_reason": failover_stress_runtime_preflight.get("failure_reason"),
                "failover_stress_runtime_evidence_path": failover_stress_runtime_preflight.get("failover_stress_runtime_evidence_path"),
                "generated_at": failover_stress_runtime_preflight.get("generated_at"),
                "age_sec": failover_stress_runtime_preflight.get("age_sec") if isinstance(failover_stress_runtime_preflight.get("age_sec"), int) else None,
                "fresh": failover_stress_runtime_preflight.get("fresh") if isinstance(failover_stress_runtime_preflight.get("fresh"), bool) else None,
                "max_age_sec": failover_stress_runtime_preflight.get("max_age_sec") if isinstance(failover_stress_runtime_preflight.get("max_age_sec"), int) else None,
                "overall_verdict": failover_stress_runtime_preflight.get("overall_verdict"),
                "publish_chain_verdict": failover_stress_runtime_preflight.get("publish_chain_verdict"),
                "publish_assertions_failed": failover_stress_runtime_preflight.get("publish_assertions_failed") if isinstance(failover_stress_runtime_preflight.get("publish_assertions_failed"), int) else None,
                "repeatability_status": failover_stress_runtime_preflight.get("repeatability_status"),
                "repeatability_match": failover_stress_runtime_preflight.get("repeatability_match"),
                "repeatability_mismatch_fields": [
                    str(x).strip()
                    for x in (failover_stress_runtime_preflight.get("repeatability_mismatch_fields") or [])
                    if str(x).strip()
                ],
                "active_top_blocker": failover_stress_runtime_preflight.get("active_top_blocker"),
                "effective_top_blocker": failover_stress_runtime_preflight.get("effective_top_blocker"),
                "inspect_failover_stress_runtime_evidence_command": normalize_operator_command(failover_stress_runtime_preflight.get("inspect_failover_stress_runtime_evidence_command")) if failover_stress_runtime_preflight.get("inspect_failover_stress_runtime_evidence_command") else None,
                "refresh_failover_stress_runtime_evidence_command": normalize_operator_command(failover_stress_runtime_preflight.get("refresh_failover_stress_runtime_evidence_command")) if failover_stress_runtime_preflight.get("refresh_failover_stress_runtime_evidence_command") else None,
                "auto_refresh": dict(failover_stress_runtime_preflight.get("auto_refresh") or {}) if isinstance(failover_stress_runtime_preflight.get("auto_refresh"), dict) else None,
            },
            "launch_readiness_severity_gate": {
                "active_blocker": launch_readiness_severity_preflight.get("active_blocker") if isinstance(launch_readiness_severity_preflight.get("active_blocker"), bool) else None,
                "blocker_reason": launch_readiness_severity_preflight.get("blocker_reason"),
                "failure_reason": launch_readiness_severity_preflight.get("failure_reason"),
                "severity_state": launch_readiness_severity_preflight.get("severity_state"),
                "severity_reason": launch_readiness_severity_preflight.get("severity_reason"),
                "severity_active": launch_readiness_severity_preflight.get("severity_active") if isinstance(launch_readiness_severity_preflight.get("severity_active"), bool) else None,
                "non_ready_ticks_consecutive": launch_readiness_severity_preflight.get("non_ready_ticks_consecutive") if isinstance(launch_readiness_severity_preflight.get("non_ready_ticks_consecutive"), int) else None,
                "threshold_ticks": launch_readiness_severity_preflight.get("threshold_ticks") if isinstance(launch_readiness_severity_preflight.get("threshold_ticks"), int) else None,
                "cohort_worker_count": launch_readiness_severity_preflight.get("cohort_worker_count") if isinstance(launch_readiness_severity_preflight.get("cohort_worker_count"), int) else None,
                "dispatch_qualification_path": launch_readiness_severity_preflight.get("dispatch_qualification_path"),
                "inspect_dispatch_qualification_command": normalize_operator_command(launch_readiness_severity_preflight.get("inspect_dispatch_qualification_command")) if launch_readiness_severity_preflight.get("inspect_dispatch_qualification_command") else None,
                "refresh_dispatch_qualification_command": normalize_operator_command(launch_readiness_severity_preflight.get("refresh_dispatch_qualification_command")) if launch_readiness_severity_preflight.get("refresh_dispatch_qualification_command") else None,
            },
            "launch_readiness_worker_health_canary_gate": {
                "active_blocker": launch_readiness_worker_health_canary_preflight.get("active_blocker") if isinstance(launch_readiness_worker_health_canary_preflight.get("active_blocker"), bool) else None,
                "blocker_reason": launch_readiness_worker_health_canary_preflight.get("blocker_reason"),
                "failure_reason": launch_readiness_worker_health_canary_preflight.get("failure_reason"),
                "dispatch_qualification_present": launch_readiness_worker_health_canary_preflight.get("dispatch_qualification_present") if isinstance(launch_readiness_worker_health_canary_preflight.get("dispatch_qualification_present"), bool) else None,
                "dispatch_qualification_failure_reason": launch_readiness_worker_health_canary_preflight.get("dispatch_qualification_failure_reason"),
                "gate_required": launch_readiness_worker_health_canary_preflight.get("gate_required") if isinstance(launch_readiness_worker_health_canary_preflight.get("gate_required"), bool) else None,
                "artifact_required": launch_readiness_worker_health_canary_preflight.get("artifact_required") if isinstance(launch_readiness_worker_health_canary_preflight.get("artifact_required"), bool) else None,
                "resource_preflight_required": launch_readiness_worker_health_canary_preflight.get("resource_preflight_required") if isinstance(launch_readiness_worker_health_canary_preflight.get("resource_preflight_required"), bool) else None,
                "resource_preflight_status": str(launch_readiness_worker_health_canary_preflight.get("resource_preflight_status") or "").strip() or None,
                "resource_preflight_reason": launch_readiness_worker_health_canary_preflight.get("resource_preflight_reason"),
                "resource_preflight_blocking_candidate_count": launch_readiness_worker_health_canary_preflight.get("resource_preflight_blocking_candidate_count") if isinstance(launch_readiness_worker_health_canary_preflight.get("resource_preflight_blocking_candidate_count"), int) else None,
                "resource_preflight_blocking_task_ids": [
                    str(x).strip()
                    for x in (launch_readiness_worker_health_canary_preflight.get("resource_preflight_blocking_task_ids") or [])
                    if str(x).strip()
                ],
                "resource_preflight_telemetry_complete": launch_readiness_worker_health_canary_preflight.get("resource_preflight_telemetry_complete") if isinstance(launch_readiness_worker_health_canary_preflight.get("resource_preflight_telemetry_complete"), bool) else None,
                "resource_preflight_lowest_headroom_pct": launch_readiness_worker_health_canary_preflight.get("resource_preflight_lowest_headroom_pct") if isinstance(launch_readiness_worker_health_canary_preflight.get("resource_preflight_lowest_headroom_pct"), int) else None,
                "uncertainty_confidence_score": launch_readiness_worker_health_canary_preflight.get("uncertainty_confidence_score") if isinstance(launch_readiness_worker_health_canary_preflight.get("uncertainty_confidence_score"), (int, float)) else None,
                "uncertainty_confidence_label": str(launch_readiness_worker_health_canary_preflight.get("uncertainty_confidence_label") or "").strip() or None,
                "uncertainty_confidence_quantiles": dict(launch_readiness_worker_health_canary_preflight.get("uncertainty_confidence_quantiles") or {}) if isinstance(launch_readiness_worker_health_canary_preflight.get("uncertainty_confidence_quantiles"), dict) else {},
                "uncertainty_reasons": [
                    str(x).strip()
                    for x in (launch_readiness_worker_health_canary_preflight.get("uncertainty_reasons") or [])
                    if str(x).strip()
                ],
                "uncertainty_requires_operator_review": launch_readiness_worker_health_canary_preflight.get("uncertainty_requires_operator_review") if isinstance(launch_readiness_worker_health_canary_preflight.get("uncertainty_requires_operator_review"), bool) else None,
                "worker_health_canary_source": str(launch_readiness_worker_health_canary_preflight.get("worker_health_canary_source") or "").strip() or None,
                "worker_health_canary_present": launch_readiness_worker_health_canary_preflight.get("worker_health_canary_present") if isinstance(launch_readiness_worker_health_canary_preflight.get("worker_health_canary_present"), bool) else None,
                "worker_health_canary_path": launch_readiness_worker_health_canary_preflight.get("worker_health_canary_path"),
                "worker_health_canary_generated_at": launch_readiness_worker_health_canary_preflight.get("worker_health_canary_generated_at"),
                "worker_health_canary_age_sec": launch_readiness_worker_health_canary_preflight.get("worker_health_canary_age_sec") if isinstance(launch_readiness_worker_health_canary_preflight.get("worker_health_canary_age_sec"), int) else None,
                "worker_health_canary_fresh": launch_readiness_worker_health_canary_preflight.get("worker_health_canary_fresh") if isinstance(launch_readiness_worker_health_canary_preflight.get("worker_health_canary_fresh"), bool) else None,
                "dispatch_qualification_path": launch_readiness_worker_health_canary_preflight.get("dispatch_qualification_path"),
                "inspect_dispatch_qualification_command": normalize_operator_command(launch_readiness_worker_health_canary_preflight.get("inspect_dispatch_qualification_command")) if launch_readiness_worker_health_canary_preflight.get("inspect_dispatch_qualification_command") else None,
                "inspect_worker_health_canary_command": normalize_operator_command(launch_readiness_worker_health_canary_preflight.get("inspect_worker_health_canary_command")) if launch_readiness_worker_health_canary_preflight.get("inspect_worker_health_canary_command") else None,
                "refresh_worker_health_canary_command": normalize_operator_command(launch_readiness_worker_health_canary_preflight.get("refresh_worker_health_canary_command")) if launch_readiness_worker_health_canary_preflight.get("refresh_worker_health_canary_command") else None,
                "refresh_dispatch_qualification_command": normalize_operator_command(launch_readiness_worker_health_canary_preflight.get("refresh_dispatch_qualification_command")) if launch_readiness_worker_health_canary_preflight.get("refresh_dispatch_qualification_command") else None,
                "first_actionable_command": normalize_operator_command(launch_readiness_worker_health_canary_preflight.get("first_actionable_command")) if launch_readiness_worker_health_canary_preflight.get("first_actionable_command") else None,
                "action_priority": str(launch_readiness_worker_health_canary_preflight.get("action_priority") or "").strip() or None,
            },
            "launch_readiness_probe_execution_gate": {
                "active_blocker": launch_readiness_probe_execution_preflight.get("active_blocker") if isinstance(launch_readiness_probe_execution_preflight.get("active_blocker"), bool) else None,
                "blocker_reason": launch_readiness_probe_execution_preflight.get("blocker_reason"),
                "failure_reason": launch_readiness_probe_execution_preflight.get("failure_reason"),
                "dispatch_qualification_present": launch_readiness_probe_execution_preflight.get("dispatch_qualification_present") if isinstance(launch_readiness_probe_execution_preflight.get("dispatch_qualification_present"), bool) else None,
                "dispatch_qualification_failure_reason": launch_readiness_probe_execution_preflight.get("dispatch_qualification_failure_reason"),
                "probe_execution_source": str(launch_readiness_probe_execution_preflight.get("probe_execution_source") or "").strip() or None,
                "probe_execution_status": launch_readiness_probe_execution_preflight.get("probe_execution_status"),
                "probe_execution_reason": launch_readiness_probe_execution_preflight.get("probe_execution_reason"),
                "pending_worker_count": launch_readiness_probe_execution_preflight.get("pending_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("pending_worker_count"), int) else None,
                "due_now_worker_count": launch_readiness_probe_execution_preflight.get("due_now_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("due_now_worker_count"), int) else None,
                "overdue_worker_count": launch_readiness_probe_execution_preflight.get("overdue_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("overdue_worker_count"), int) else None,
                "oldest_due_now_started_at": launch_readiness_probe_execution_preflight.get("oldest_due_now_started_at"),
                "oldest_due_now_worker": launch_readiness_probe_execution_preflight.get("oldest_due_now_worker"),
                "oldest_due_now_age_sec": launch_readiness_probe_execution_preflight.get("oldest_due_now_age_sec") if isinstance(launch_readiness_probe_execution_preflight.get("oldest_due_now_age_sec"), int) else None,
                "oldest_overdue_started_at": launch_readiness_probe_execution_preflight.get("oldest_overdue_started_at"),
                "oldest_overdue_worker": launch_readiness_probe_execution_preflight.get("oldest_overdue_worker"),
                "oldest_overdue_age_sec": launch_readiness_probe_execution_preflight.get("oldest_overdue_age_sec") if isinstance(launch_readiness_probe_execution_preflight.get("oldest_overdue_age_sec"), int) else None,
                "demotion_restore_pending_worker_count": launch_readiness_probe_execution_preflight.get("demotion_restore_pending_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_restore_pending_worker_count"), int) else None,
                "demotion_demoted_worker_count": launch_readiness_probe_execution_preflight.get("demotion_demoted_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_demoted_worker_count"), int) else None,
                "demotion_restored_worker_count": launch_readiness_probe_execution_preflight.get("demotion_restored_worker_count") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_restored_worker_count"), int) else None,
                "demotion_oldest_restore_pending_since": launch_readiness_probe_execution_preflight.get("demotion_oldest_restore_pending_since"),
                "demotion_oldest_restore_pending_worker": launch_readiness_probe_execution_preflight.get("demotion_oldest_restore_pending_worker"),
                "demotion_oldest_restore_pending_age_sec": launch_readiness_probe_execution_preflight.get("demotion_oldest_restore_pending_age_sec") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_oldest_restore_pending_age_sec"), int) else None,
                "demotion_oldest_demoted_at": launch_readiness_probe_execution_preflight.get("demotion_oldest_demoted_at"),
                "demotion_oldest_demoted_worker": launch_readiness_probe_execution_preflight.get("demotion_oldest_demoted_worker"),
                "demotion_oldest_demoted_age_sec": launch_readiness_probe_execution_preflight.get("demotion_oldest_demoted_age_sec") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_oldest_demoted_age_sec"), int) else None,
                "demotion_latest_restored_at": launch_readiness_probe_execution_preflight.get("demotion_latest_restored_at"),
                "demotion_latest_restored_worker": launch_readiness_probe_execution_preflight.get("demotion_latest_restored_worker"),
                "demotion_latest_restored_age_sec": launch_readiness_probe_execution_preflight.get("demotion_latest_restored_age_sec") if isinstance(launch_readiness_probe_execution_preflight.get("demotion_latest_restored_age_sec"), int) else None,
                "demotion_action_priority": str(launch_readiness_probe_execution_preflight.get("demotion_action_priority") or "").strip() or None,
                "due_now_active": launch_readiness_probe_execution_preflight.get("due_now_active") if isinstance(launch_readiness_probe_execution_preflight.get("due_now_active"), bool) else None,
                "overdue_active": launch_readiness_probe_execution_preflight.get("overdue_active") if isinstance(launch_readiness_probe_execution_preflight.get("overdue_active"), bool) else None,
                "launch_readiness_state": launch_readiness_probe_execution_preflight.get("launch_readiness_state"),
                "launch_readiness_reason": launch_readiness_probe_execution_preflight.get("launch_readiness_reason"),
                "due_now_idle_no_dispatch_candidate": launch_readiness_probe_execution_preflight.get("due_now_idle_no_dispatch_candidate") if isinstance(launch_readiness_probe_execution_preflight.get("due_now_idle_no_dispatch_candidate"), bool) else None,
                "probe_execution_plan_path": launch_readiness_probe_execution_preflight.get("probe_execution_plan_path"),
                "probe_execution_plan_present": launch_readiness_probe_execution_preflight.get("probe_execution_plan_present") if isinstance(launch_readiness_probe_execution_preflight.get("probe_execution_plan_present"), bool) else None,
                "dispatch_qualification_path": launch_readiness_probe_execution_preflight.get("dispatch_qualification_path"),
                "inspect_dispatch_qualification_command": normalize_operator_command(launch_readiness_probe_execution_preflight.get("inspect_dispatch_qualification_command")) if launch_readiness_probe_execution_preflight.get("inspect_dispatch_qualification_command") else None,
                "inspect_probe_execution_plan_command": normalize_operator_command(launch_readiness_probe_execution_preflight.get("inspect_probe_execution_plan_command")) if launch_readiness_probe_execution_preflight.get("inspect_probe_execution_plan_command") else None,
                "refresh_dispatch_qualification_command": normalize_operator_command(launch_readiness_probe_execution_preflight.get("refresh_dispatch_qualification_command")) if launch_readiness_probe_execution_preflight.get("refresh_dispatch_qualification_command") else None,
                "first_actionable_command": normalize_operator_command(launch_readiness_probe_execution_preflight.get("first_actionable_command")) if launch_readiness_probe_execution_preflight.get("first_actionable_command") else None,
                "action_priority": str(launch_readiness_probe_execution_preflight.get("action_priority") or "").strip() or None,
                "action_priority_source": str(launch_readiness_probe_execution_preflight.get("action_priority_source") or "").strip() or None,
            },
        }

# ground truth

gt_latest = load_json(root / "state" / "ground_truth" / "latest.json")
gt_snapshot = {}
gt_snapshot_path = str(gt_latest.get("snapshot_path") or "")
if gt_snapshot_path:
    p = (root / gt_snapshot_path).resolve()
    if p.exists():
        gt_snapshot = load_json(p)

gt_snapshot_id = str(gt_latest.get("snapshot_id") or gt_snapshot.get("snapshot_id") or "n/a")
gt_ts = str(gt_latest.get("snapshot_ts_utc") or gt_snapshot.get("snapshot_ts_utc") or "")
gt_age = age_sec(gt_ts)

anomalies = gt_snapshot.get("anomalies") or []
critical_keys = [a.get("key") for a in anomalies if isinstance(a, dict) and str(a.get("severity") or "") == "critical"]
warn_keys = [a.get("key") for a in anomalies if isinstance(a, dict) and str(a.get("severity") or "") == "warn"]

bridge_lp = bridge.get("latest_pointer") or {}
bridge_env = bridge.get("env_snapshot_latest") or {}
bridge_gt = bridge.get("ground_truth_latest") or {}

captured_env = str(((checkpoint.get("state_capture") or {}).get("env_snapshot_path") or "")).strip()
captured_gt = str(((checkpoint.get("state_capture") or {}).get("ground_truth_snapshot_path") or "")).strip()

# Compute live alignment directly from latest artifacts (bridge file may be stale between syncs).
def file_sha256(path: pathlib.Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

pointer_ok = None
pointer_sha_ok = None
if latest_pointer:
    lp_id = str(latest_pointer.get("checkpoint_id") or "").strip()
    lp_path = str(latest_pointer.get("json_path") or "").strip()
    if checkpoint_id != "n/a" and checkpoint_path:
        pointer_ok = lp_id == checkpoint_id and lp_path == checkpoint_path
    elif lp_id or lp_path:
        pointer_ok = False

    lp_sha = str(latest_pointer.get("json_sha256") or "").strip()
    if pointer_ok is True:
        cp_abs = (root / checkpoint_path).resolve() if checkpoint_path else None
        cp_sha = file_sha256(cp_abs) if cp_abs is not None else None
        if lp_sha and cp_sha:
            pointer_sha_ok = lp_sha == cp_sha
        elif lp_sha and not cp_sha:
            pointer_sha_ok = False

env_latest = load_json(latest_dir / "env_snapshot_latest.json")
env_latest_path = str(env_latest.get("env_snapshot_path") or "").strip()
env_capture_ok = None
if captured_env or env_latest_path:
    env_capture_ok = bool(captured_env and env_latest_path and captured_env == env_latest_path)

gt_capture_ok = None
if captured_gt or gt_snapshot_path:
    gt_capture_ok = bool(captured_gt and gt_snapshot_path and captured_gt == gt_snapshot_path)

bridge_pointer_reported = bridge_lp.get("matches_checkpoint") if bridge_lp else None
bridge_pointer_sha_reported = bridge_lp.get("sha_match") if bridge_lp else None
bridge_env_capture_reported = bridge_env.get("matches_checkpoint_capture") if bridge_env else None
bridge_gt_capture_reported = bridge_gt.get("matches_checkpoint_capture") if bridge_gt else None
bridge_stale = any(
    reported is not None and live is not None and reported != live
    for reported, live in [
        (bridge_pointer_reported, pointer_ok),
        (bridge_pointer_sha_reported, pointer_sha_ok),
        (bridge_env_capture_reported, env_capture_ok),
        (bridge_gt_capture_reported, gt_capture_ok),
    ]
)

# autopilot snapshot summary
ap_state = ((gt_snapshot.get("autopilot_state") or {}).get("state") or {})
ap_steps = ap_state.get("steps") or []
step_counts = {}
for st in ap_steps:
    if not isinstance(st, dict):
        continue
    key = str(st.get("status") or "unknown")
    step_counts[key] = int(step_counts.get(key) or 0) + 1

active = ap_state.get("active") if isinstance(ap_state, dict) else None
active_step = active.get("step_id") if isinstance(active, dict) else None
active_evidence_refs = []
if isinstance(active, dict):
    for field in ("log_path", "exit_code_path"):
        p_raw = str(active.get(field) or "").strip()
        if not p_raw:
            continue
        p = pathlib.Path(p_raw)
        if p.exists():
            active_evidence_refs.append(maybe_rel(p))

autopilot_recent_evidence_refs = autopilot_step_evidence_refs(ap_state if isinstance(ap_state, dict) else {})
queue_degraded_meta = ap_state.get("queue_infra_degraded") if isinstance(ap_state, dict) else {}
if not isinstance(queue_degraded_meta, dict):
    queue_degraded_meta = {}

pending_stale_signal_raw = queue_degraded_meta.get("degraded_pending_stale_signal")
pending_stale_signal = pending_stale_signal_raw if isinstance(pending_stale_signal_raw, dict) else {}

def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

autopilot_degraded_pending_signal = {
    "active": bool(pending_stale_signal.get("active") is True),
    "stale_ticks_consecutive": _to_int(pending_stale_signal.get("stale_ticks_consecutive"), 0),
    "activate_after_ticks": _to_int(pending_stale_signal.get("activate_after_ticks"), 0),
    "cooldown_sec": _to_int(pending_stale_signal.get("cooldown_sec"), 0),
    "pending_stale_count": _to_int(pending_stale_signal.get("pending_stale_count"), _to_int(queue_degraded_meta.get("degraded_local_runs_pending_stale_count"), 0)),
    "pending_total": _to_int(pending_stale_signal.get("pending_total"), _to_int(queue_degraded_meta.get("degraded_local_runs_pending"), 0)),
    "pending_oldest_age_sec": _to_int(pending_stale_signal.get("pending_oldest_age_sec"), _to_int(queue_degraded_meta.get("degraded_local_runs_pending_oldest_age_sec"), 0)),
    "pending_oldest_run_id": str(pending_stale_signal.get("pending_oldest_run_id") or queue_degraded_meta.get("degraded_local_runs_pending_oldest_run_id") or ""),
    "pending_oldest_task_id": str(pending_stale_signal.get("pending_oldest_task_id") or ""),
    "last_emit_iso": pending_stale_signal.get("last_emit_iso"),
    "active_since_iso": pending_stale_signal.get("active_since_iso"),
    "next_emit_after_iso": pending_stale_signal.get("next_emit_after_iso"),
    "inspect_command": normalize_operator_command(pending_stale_signal.get("inspect_command") or "") or None,
    "recovery_command": normalize_operator_command(pending_stale_signal.get("recovery_command") or "") or None,
}

stale_task_recovery_raw = queue_degraded_meta.get("stale_task_recovery_counters")
stale_task_recovery = stale_task_recovery_raw if isinstance(stale_task_recovery_raw, dict) else {}
autopilot_stale_task_recovery_counters = {
    "schema_version": str(stale_task_recovery.get("schema_version") or "autopilot.degraded_stale_task_recovery_counters.v1"),
    "attempts_total": _to_int(stale_task_recovery.get("attempts_total"), 0),
    "recovered_total": _to_int(stale_task_recovery.get("recovered_total"), 0),
    "failed_total": _to_int(stale_task_recovery.get("failed_total"), 0),
    "stale_processing_recovered_total": _to_int(stale_task_recovery.get("stale_processing_recovered_total"), 0),
    "stale_running_recovered_total": _to_int(stale_task_recovery.get("stale_running_recovered_total"), 0),
    "attempts_last_tick": _to_int(stale_task_recovery.get("attempts_last_tick"), 0),
    "recovered_last_tick": _to_int(stale_task_recovery.get("recovered_last_tick"), 0),
    "failed_last_tick": _to_int(stale_task_recovery.get("failed_last_tick"), 0),
    "updated_at": stale_task_recovery.get("updated_at"),
    "last_recovery": stale_task_recovery.get("last_recovery") if isinstance(stale_task_recovery.get("last_recovery"), dict) else None,
    "last_recovery_at": stale_task_recovery.get("last_recovery_at"),
    "last_failure_at": stale_task_recovery.get("last_failure_at"),
}

idle_lane_autospawn_state_rel = "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json"
idle_lane_autospawn_state_path = latest_dir / "no_nudge_idle_lane_autospawn_latest.json"
idle_lane_autospawn_state_exists = idle_lane_autospawn_state_path.exists()
idle_lane_autospawn_raw = load_json(idle_lane_autospawn_state_path)
if not isinstance(idle_lane_autospawn_raw, dict):
    idle_lane_autospawn_raw = {}

idle_lane_eval = idle_lane_autospawn_raw.get("evaluation") if isinstance(idle_lane_autospawn_raw.get("evaluation"), dict) else {}
idle_lane_updated_at = str(idle_lane_autospawn_raw.get("updated_at") or "").strip()
idle_lane_updated_age_sec = age_sec(idle_lane_updated_at) if idle_lane_updated_at else None
idle_lane_updated_valid = parse_iso(idle_lane_updated_at) is not None if idle_lane_updated_at else False
idle_lane_issues: List[str] = []
if idle_lane_autospawn_state_exists:
    if not idle_lane_autospawn_raw:
        idle_lane_issues.append("trace_invalid")
    if not idle_lane_updated_at:
        idle_lane_issues.append("updated_at_missing")
    elif not idle_lane_updated_valid:
        idle_lane_issues.append("updated_at_invalid")
    elif idle_lane_autospawn_max_age_sec > 0 and idle_lane_updated_age_sec is not None and idle_lane_updated_age_sec > idle_lane_autospawn_max_age_sec:
        idle_lane_issues.append("trace_stale")

idle_lane_contract_healthy = len(idle_lane_issues) == 0
idle_lane_contract_degraded_reason = idle_lane_issues[0] if idle_lane_issues else None
idle_lane_contract_fresh = None
if idle_lane_updated_age_sec is not None:
    idle_lane_contract_fresh = True if idle_lane_autospawn_max_age_sec <= 0 else idle_lane_updated_age_sec <= idle_lane_autospawn_max_age_sec

idle_lane_autospawn_contract = {
    "schema_version": "continuity.no_nudge_idle_lane_autospawn_contract.v1",
    "status": "ok" if idle_lane_contract_healthy else "degraded",
    "healthy": idle_lane_contract_healthy,
    "issues": idle_lane_issues,
    "state_exists": idle_lane_autospawn_state_exists,
    "state_path": idle_lane_autospawn_state_rel,
    "updated_at": idle_lane_updated_at or None,
    "updated_age_sec": idle_lane_updated_age_sec,
    "max_age_sec": idle_lane_autospawn_max_age_sec,
    "fresh": idle_lane_contract_fresh,
}

idle_lane_autospawn = {
    "status": str(idle_lane_autospawn_raw.get("status") or "missing"),
    "updated_at": idle_lane_updated_at or None,
    "updated_age_sec": idle_lane_updated_age_sec,
    "max_age_sec": idle_lane_autospawn_max_age_sec,
    "fresh": idle_lane_contract_fresh,
    "enabled": bool(idle_lane_autospawn_raw.get("enabled") is True),
    "trace_path": str(idle_lane_autospawn_raw.get("trace_path") or idle_lane_autospawn_state_rel),
    "target_step_id": str(idle_lane_autospawn_raw.get("target_step_id") or ""),
    "launched": bool(idle_lane_autospawn_raw.get("launched") is True),
    "launched_step_id": str(idle_lane_autospawn_raw.get("launched_step_id") or ""),
    "last_attempt_at": idle_lane_autospawn_raw.get("last_attempt_at"),
    "last_launch_at": idle_lane_autospawn_raw.get("last_launch_at"),
    "ready_work_exists": bool(idle_lane_autospawn_raw.get("ready_work_exists") is True),
    "idle_threshold_exceeded": bool(idle_lane_autospawn_raw.get("idle_threshold_exceeded") is True),
    "idle_sec": _to_int(idle_lane_autospawn_raw.get("idle_sec"), 0),
    "idle_threshold_sec": _to_int(idle_lane_autospawn_raw.get("idle_threshold_sec"), 0),
    "skip_reason": str(idle_lane_autospawn_raw.get("skip_reason") or ""),
    "contradiction_abort_active": bool(idle_lane_autospawn_raw.get("contradiction_abort_active") is True),
    "contradiction_abort_remaining_sec": _to_int(
        idle_lane_autospawn_raw.get("contradiction_abort_remaining_sec"),
        _to_int(idle_lane_eval.get("contradiction_abort_remaining_sec"), 0),
    ),
    "contradiction_latch_repaired": bool(idle_lane_autospawn_raw.get("contradiction_latch_repaired") is True),
    "contradiction_latch_repair_reason": str(idle_lane_autospawn_raw.get("contradiction_latch_repair_reason") or ""),
    "tick_first_line": str(idle_lane_autospawn_raw.get("tick_first_line") or ""),
    "cooldown_remaining_sec": _to_int(idle_lane_eval.get("cooldown_remaining_sec"), 0),
    "active_impl_running": bool(idle_lane_eval.get("active_impl_running") is True),
    "error": str(idle_lane_autospawn_raw.get("error") or ""),
    "contract_source_degraded": not idle_lane_contract_healthy,
    "contract_source_degraded_reason": idle_lane_contract_degraded_reason,
    "contract_source_degraded_path": idle_lane_autospawn_state_rel if not idle_lane_contract_healthy else None,
}


def _dedupe_nonempty_strings(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in (values or []):
        txt = str(raw or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


execution_frontier_controller_state_rel = "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"
execution_frontier_controller_state_path = latest_dir / "no_nudge_execution_frontier_controller_tick_latest.json"
execution_frontier_controller_state_exists = execution_frontier_controller_state_path.exists()
execution_frontier_controller_raw = load_json(execution_frontier_controller_state_path)
if not isinstance(execution_frontier_controller_raw, dict):
    execution_frontier_controller_raw = {}

execution_frontier_latch_state_rel = "state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json"
execution_frontier_latch_state_path = latest_dir / "execution_frontier_post_completion_enforcement_latch.json"
execution_frontier_latch_state_exists = execution_frontier_latch_state_path.exists()
execution_frontier_latch_raw = load_json(execution_frontier_latch_state_path)
if not isinstance(execution_frontier_latch_raw, dict):
    execution_frontier_latch_raw = {}

execution_frontier_intent_state_rel = "state/continuity/latest/autonomous_execution_intent_latest.json"
execution_frontier_intent_state_path = latest_dir / "autonomous_execution_intent_latest.json"
execution_frontier_intent_state_exists = execution_frontier_intent_state_path.exists()
execution_frontier_intent_raw = load_json(execution_frontier_intent_state_path)
if not isinstance(execution_frontier_intent_raw, dict):
    execution_frontier_intent_raw = {}

execution_frontier_controller_updated_at = str(execution_frontier_controller_raw.get("recorded_at") or "").strip()
execution_frontier_controller_updated_age_sec = (
    age_sec(execution_frontier_controller_updated_at) if execution_frontier_controller_updated_at else None
)
execution_frontier_controller_updated_valid = (
    parse_iso(execution_frontier_controller_updated_at) is not None if execution_frontier_controller_updated_at else False
)
execution_frontier_controller_issues: List[str] = []
if execution_frontier_controller_state_exists:
    if not execution_frontier_controller_raw:
        execution_frontier_controller_issues.append("trace_invalid")
    if not execution_frontier_controller_updated_at:
        execution_frontier_controller_issues.append("recorded_at_missing")
    elif not execution_frontier_controller_updated_valid:
        execution_frontier_controller_issues.append("recorded_at_invalid")
    elif (
        execution_frontier_controller_tick_max_age_sec > 0
        and execution_frontier_controller_updated_age_sec is not None
        and execution_frontier_controller_updated_age_sec > execution_frontier_controller_tick_max_age_sec
    ):
        execution_frontier_controller_issues.append("trace_stale")

execution_frontier_controller_contract_healthy = len(execution_frontier_controller_issues) == 0
execution_frontier_controller_contract_degraded_reason = (
    execution_frontier_controller_issues[0] if execution_frontier_controller_issues else None
)
execution_frontier_controller_contract_fresh = None
if execution_frontier_controller_updated_age_sec is not None:
    execution_frontier_controller_contract_fresh = (
        True
        if execution_frontier_controller_tick_max_age_sec <= 0
        else execution_frontier_controller_updated_age_sec <= execution_frontier_controller_tick_max_age_sec
    )

execution_frontier_controller_contract = {
    "schema_version": "continuity.no_nudge_execution_frontier_controller_tick_contract.v1",
    "status": "ok" if execution_frontier_controller_contract_healthy else "degraded",
    "healthy": execution_frontier_controller_contract_healthy,
    "issues": execution_frontier_controller_issues,
    "state_exists": execution_frontier_controller_state_exists,
    "state_path": execution_frontier_controller_state_rel,
    "recorded_at": execution_frontier_controller_updated_at or None,
    "updated_age_sec": execution_frontier_controller_updated_age_sec,
    "max_age_sec": execution_frontier_controller_tick_max_age_sec,
    "fresh": execution_frontier_controller_contract_fresh,
}

execution_frontier_controller_execution_frontier_obj = (
    execution_frontier_controller_raw.get("execution_frontier")
    if isinstance(execution_frontier_controller_raw.get("execution_frontier"), dict)
    else {}
)
execution_frontier_controller_dispatch_attempt_obj = (
    execution_frontier_controller_raw.get("dispatch_attempt")
    if isinstance(execution_frontier_controller_raw.get("dispatch_attempt"), dict)
    else {}
)
execution_frontier_controller_trace_path = (
    str(execution_frontier_controller_raw.get("trace_path") or execution_frontier_controller_state_rel)
)
execution_frontier_controller_history_path = str(
    execution_frontier_controller_raw.get("history_path")
    or "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl"
)
execution_frontier_controller_status = str(execution_frontier_controller_raw.get("status") or "missing")
execution_frontier_controller_decision = str(execution_frontier_controller_raw.get("decision") or "SKIP")
execution_frontier_controller_block_reasons = _dedupe_nonempty_strings(
    execution_frontier_controller_raw.get("block_reasons")
)
post_completion_selector_states = {"ready_for_dispatch", "closed_blocked", "idle_no_candidate"}
execution_frontier_controller_close_condition_met = (
    execution_frontier_controller_execution_frontier_obj.get("close_condition_met")
    if isinstance(execution_frontier_controller_execution_frontier_obj.get("close_condition_met"), bool)
    else None
)
execution_frontier_controller_selector_state = str(
    execution_frontier_controller_execution_frontier_obj.get("selector_state") or ""
)
execution_frontier_controller_post_completion_enforcement_required = (
    execution_frontier_controller_execution_frontier_obj.get("post_completion_enforcement_required")
)
if not isinstance(execution_frontier_controller_post_completion_enforcement_required, bool):
    execution_frontier_controller_post_completion_enforcement_required = bool(
        execution_frontier_controller_close_condition_met is True
        and execution_frontier_controller_selector_state in post_completion_selector_states
    )
execution_frontier_controller_post_completion_enforcement_latched = bool(
    execution_frontier_controller_post_completion_enforcement_required
    and execution_frontier_controller_status in {"blocked", "error", "missing", "skipped"}
)
if isinstance(execution_frontier_latch_raw.get("latched"), bool):
    execution_frontier_controller_post_completion_enforcement_latched = bool(
        execution_frontier_latch_raw.get("latched")
    )

execution_frontier_loop_state = str(execution_frontier_latch_raw.get("loop_state") or "").strip()
execution_frontier_latch_first_seen_at = str(execution_frontier_latch_raw.get("first_seen_at") or "").strip()
execution_frontier_latch_history_path = str(
    execution_frontier_latch_raw.get("latch_history_path")
    or "state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl"
)
execution_frontier_latch_blocked_streak = max(0, _to_int(execution_frontier_latch_raw.get("blocked_streak"), 0))
execution_frontier_latch_error_streak = max(0, _to_int(execution_frontier_latch_raw.get("error_streak"), 0))
execution_frontier_latch_retry_contract = (
    execution_frontier_latch_raw.get("retry_contract")
    if isinstance(execution_frontier_latch_raw.get("retry_contract"), dict)
    else {}
)
execution_frontier_latch_cooldown_policy = (
    execution_frontier_latch_raw.get("cooldown_policy")
    if isinstance(execution_frontier_latch_raw.get("cooldown_policy"), dict)
    else {}
)
execution_frontier_latch_parity = (
    execution_frontier_latch_raw.get("queue_truth_vs_narrative_parity")
    if isinstance(execution_frontier_latch_raw.get("queue_truth_vs_narrative_parity"), dict)
    else {}
)
execution_frontier_intent_active = bool(execution_frontier_intent_raw.get("active") is True)
execution_frontier_intent_path = str(
    execution_frontier_intent_raw.get("intent_path") or execution_frontier_intent_state_rel
)
execution_frontier_intent_history_path = str(
    execution_frontier_intent_raw.get("intent_history_path")
    or "state/continuity/history/autonomous_execution_intent_history.jsonl"
)
if execution_frontier_controller_status in {"missing", "skipped"} and execution_frontier_intent_active:
    execution_frontier_controller_status = str(execution_frontier_intent_raw.get("status") or execution_frontier_controller_status)

execution_frontier_controller = {
    "status": execution_frontier_controller_status,
    "decision": execution_frontier_controller_decision,
    "recorded_at": execution_frontier_controller_updated_at or None,
    "updated_age_sec": execution_frontier_controller_updated_age_sec,
    "max_age_sec": execution_frontier_controller_tick_max_age_sec,
    "fresh": execution_frontier_controller_contract_fresh,
    "controller_enabled": bool(execution_frontier_controller_raw.get("controller_enabled") is True),
    "controller_reason": str(execution_frontier_controller_raw.get("controller_reason") or ""),
    "skip_reason": str(execution_frontier_controller_raw.get("skip_reason") or ""),
    "block_reason": str(execution_frontier_controller_raw.get("block_reason") or ""),
    "block_reasons": execution_frontier_controller_block_reasons,
    "error": str(execution_frontier_controller_raw.get("error") or ""),
    "action_token_present": bool(execution_frontier_controller_raw.get("action_token_present") is True),
    "trace_path": execution_frontier_controller_trace_path,
    "history_path": execution_frontier_controller_history_path,
    "selector_state": execution_frontier_controller_selector_state,
    "close_condition_met": execution_frontier_controller_close_condition_met,
    "post_completion_enforcement_required": execution_frontier_controller_post_completion_enforcement_required,
    "post_completion_enforcement_latched": execution_frontier_controller_post_completion_enforcement_latched,
    "post_completion_loop_state": execution_frontier_loop_state or None,
    "post_completion_first_seen_at": execution_frontier_latch_first_seen_at or None,
    "post_completion_latch_path": str(
        execution_frontier_latch_raw.get("latch_path") or execution_frontier_latch_state_rel
    ),
    "post_completion_latch_history_path": execution_frontier_latch_history_path,
    "post_completion_blocked_streak": execution_frontier_latch_blocked_streak,
    "post_completion_error_streak": execution_frontier_latch_error_streak,
    "post_completion_retry_contract": execution_frontier_latch_retry_contract,
    "post_completion_cooldown_policy": execution_frontier_latch_cooldown_policy,
    "queue_truth_vs_narrative_parity": execution_frontier_latch_parity,
    "autonomous_execution_intent_active": execution_frontier_intent_active,
    "autonomous_execution_intent_path": execution_frontier_intent_path,
    "autonomous_execution_intent_history_path": execution_frontier_intent_history_path,
    "next_candidate": str(execution_frontier_controller_execution_frontier_obj.get("next_candidate") or ""),
    "next_candidate_wave": execution_frontier_controller_execution_frontier_obj.get("next_candidate_wave")
    if isinstance(execution_frontier_controller_execution_frontier_obj.get("next_candidate_wave"), int)
    else None,
    "supervisor_state": str(execution_frontier_controller_execution_frontier_obj.get("supervisor_state") or ""),
    "autonomous_dispatch_eligible": execution_frontier_controller_execution_frontier_obj.get(
        "autonomous_dispatch_eligible"
    )
    if isinstance(execution_frontier_controller_execution_frontier_obj.get("autonomous_dispatch_eligible"), bool)
    else None,
    "autonomous_dispatch_block_reasons": _dedupe_nonempty_strings(
        execution_frontier_controller_execution_frontier_obj.get("autonomous_dispatch_block_reasons")
    ),
    "dispatch_executed": bool(execution_frontier_controller_dispatch_attempt_obj.get("executed") is True),
    "dispatch_returncode": execution_frontier_controller_dispatch_attempt_obj.get("returncode")
    if isinstance(execution_frontier_controller_dispatch_attempt_obj.get("returncode"), int)
    else None,
    "dispatch_decision": str(execution_frontier_controller_dispatch_attempt_obj.get("decision") or ""),
    "dispatch_advance_applied": bool(execution_frontier_controller_dispatch_attempt_obj.get("advance_applied") is True),
    "dispatch_block_reason": str(execution_frontier_controller_dispatch_attempt_obj.get("block_reason") or ""),
    "dispatch_block_reasons": _dedupe_nonempty_strings(
        execution_frontier_controller_dispatch_attempt_obj.get("block_reasons")
    ),
    "dispatch_error": str(execution_frontier_controller_dispatch_attempt_obj.get("error") or ""),
    "contract_source_degraded": not execution_frontier_controller_contract_healthy,
    "contract_source_degraded_reason": execution_frontier_controller_contract_degraded_reason,
    "contract_source_degraded_path": (
        execution_frontier_controller_state_rel if not execution_frontier_controller_contract_healthy else None
    ),
}

current_now_iso = clock_now_iso()

# continuity DB queue summary
queue_counts = {}
queue_role_required_counts = {}
queue_role_missing_count = 0
queue_review_role_mismatch_count = 0
queue_ready_count = 0
queue_dependency_blocked_count = 0
queue_dependency_blocked_examples: List[Dict[str, Any]] = []
active_file_lock_count = 0
stale_active_file_lock_count = 0
effective_active_file_lock_count = 0
active_lock_examples: List[Dict[str, Any]] = []
orphaned_running_without_locks_count = 0
orphaned_running_without_locks_examples: List[Dict[str, Any]] = []
effective_running_count = 0
in_flight_effective = False
queue_ready_oldest_updated_at = None
queue_ready_oldest_age_sec = None
queue_stale_wave_signal: Dict[str, Any] = {
    "schema_version": "continuity.queue_stale_wave_signal.v1",
    "active": False,
    "reason": "queue_flow_recent",
    "ready_count": 0,
    "ready_oldest_updated_at": None,
    "ready_oldest_age_sec": None,
    "ready_idle_threshold_sec": int(queue_stale_wave_ready_idle_sec),
    "in_flight_effective": False,
    "last_transition_at": None,
    "last_transition_age_sec": None,
    "last_handoff_at": None,
    "last_handoff_age_sec": None,
    "inspect_command": cmd_queue_ready_list_json,
    "recovery_command": cmd_cont_queue_sync_json,
}
queue_stale_wave_auto_remediation: Dict[str, Any] = {
    "schema_version": "continuity.queue_stale_wave_auto_remediation.v1",
    "enabled": bool(queue_stale_wave_auto_remediate_enabled),
    "state_path": queue_stale_wave_auto_remediation_state_rel,
    "status": "disabled" if not queue_stale_wave_auto_remediate_enabled else "idle",
    "reason": "disabled_by_env" if not queue_stale_wave_auto_remediate_enabled else "not_evaluated",
    "eligible": False,
    "triggered": False,
    "attempted": False,
    "recovered": False,
    "cooldown_sec": int(queue_stale_wave_auto_remediate_cooldown_sec),
    "cooldown_remaining_sec": 0,
    "next_attempt_after_iso": None,
    "last_eval_at": None,
    "last_attempt_at": None,
    "last_attempt_status": None,
    "last_success_at": None,
    "last_failure_at": None,
    "queue_stale_wave_active_before": False,
    "queue_stale_wave_active_after": False,
    "ready_count_before": 0,
    "ready_count_after": 0,
    "ready_oldest_age_sec_before": None,
    "ready_oldest_age_sec_after": None,
    "in_flight_effective_before": False,
    "in_flight_effective_after": False,
    "ready_idle_threshold_sec": int(queue_stale_wave_ready_idle_sec),
    "command": cmd_queue_sync_internal,
    "command_rc": None,
    "command_ok": None,
    "payload_ok": None,
    "payload_parse_error": None,
    "progress_detected": None,
    "progress_reasons": [],
    "last_transition_at_before": None,
    "last_transition_at_after": None,
    "last_handoff_at_before": None,
    "last_handoff_at_after": None,
    "failure_taxonomy_version": "continuity.queue_stale_wave_auto_remediation_failure.v1",
    "failure_category": None,
    "failure_code": None,
    "failure_retryable": None,
    "attempt_sequence": 0,
    "consecutive_failures": 0,
    "retry_contract": {
        "schema_version": "continuity.queue_stale_wave_auto_remediation_retry_contract.v1",
        "policy": "fixed_cooldown",
        "deterministic_key": "queue_stale_wave_auto_remediation",
        "eligible_now": False,
        "retry_scheduled": False,
        "retry_due": False,
        "retry_blocked_reason": None,
        "cooldown_sec": int(queue_stale_wave_auto_remediate_cooldown_sec),
        "cooldown_remaining_sec": 0,
        "next_attempt_after_ts": 0,
        "next_attempt_after_iso": None,
    },
    "failure_evidence": {
        "schema_version": "continuity.queue_stale_wave_auto_remediation_failure_evidence.v1",
        "present": False,
        "captured_at": None,
        "attempt_at": None,
        "attempt_sequence": 0,
        "status": None,
        "reason": None,
        "category": None,
        "code": None,
        "retryable": None,
        "command": cmd_queue_sync_internal,
        "command_rc": None,
        "command_ok": None,
        "payload_ok": None,
        "payload_parse_error": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "queue_stale_wave_active_before": False,
        "queue_stale_wave_active_after": False,
        "ready_count_before": 0,
        "ready_count_after": 0,
        "ready_oldest_age_sec_before": None,
        "ready_oldest_age_sec_after": None,
        "in_flight_effective_before": False,
        "in_flight_effective_after": False,
        "progress_detected": None,
        "progress_reasons": [],
        "next_retry_after_ts": 0,
        "next_retry_after_iso": None,
        "retry_cooldown_sec": int(queue_stale_wave_auto_remediate_cooldown_sec),
    },
}
orphaned_running_auto_remediation: Dict[str, Any] = {
    "schema_version": "continuity.queue_orphaned_running_auto_remediation.v1",
    "enabled": bool(orphaned_running_auto_remediate_enabled),
    "state_path": orphaned_running_auto_remediation_state_rel,
    "status": "disabled" if not orphaned_running_auto_remediate_enabled else "idle",
    "reason": "disabled_by_env" if not orphaned_running_auto_remediate_enabled else "not_evaluated",
    "eligible": False,
    "triggered": False,
    "attempted": False,
    "cooldown_sec": int(orphaned_running_auto_remediate_cooldown_sec),
    "cooldown_remaining_sec": 0,
    "next_attempt_after_iso": None,
    "last_eval_at": None,
    "last_attempt_at": None,
    "last_success_at": None,
    "last_failure_at": None,
    "orphaned_running_without_locks_count": 0,
    "effective_running_count": 0,
    "effective_active_file_lock_count": 0,
    "in_flight_effective": None,
    "orphaned_running_min_sec": int(orphaned_running_min_sec),
    "command": cmd_queue_requeue_orphaned_apply_json,
    "command_rc": None,
    "applied_requeued_orphaned_running": 0,
}
transition_last_at = None
transition_last_event = None
recent_evidence_refs = []
transition_history_24h = {
    "window_hours": 24,
    "total": 0,
    "to_status": {},
    "actor_role": {},
}
handoff_history_24h = {
    "window_hours": 24,
    "total": 0,
    "role_edges": {},
}
latest_handoff_packet = None
parity_summary = {
    "task_id": "parity:weekly_harness",
    "status": None,
    "last_done_at": None,
    "last_done_age_sec": None,
    "freshness_limit_sec": 9 * 24 * 3600,
    "fresh": None,
    "due": None,
}
web_capture_summary = {
    "tracked_domains": 0,
    "blocked_domains": 0,
    "cooldown_active_domains": 0,
    "operator_action_required_domains": 0,
    "actionable_incident_domains": 0,
    "latest_updated_at": None,
    "domains": [],
    "scheduler": {
        "state_path": "state/continuity/latest/web_capture_scheduler_state.json",
        "selection_status": None,
        "updated_at": None,
        "state_age_sec": None,
        "freshness_limit_sec": None,
        "fresh": None,
        "state_exists": False,
        "schema_version": None,
        "contract_schema_path": None,
        "contract_state_valid": None,
        "contract_validation_errors": [],
        "contract_previous_state_valid": None,
        "eligible_macros": None,
        "total_macros": None,
        "last_selected_domain": None,
        "last_selected_macro_slug": None,
    },
}
gtc_summary = {
    "enabled": False,
    "mutate_allowed": None,
    "status": None,
    "blocking_reasons": [],
    "warning_reasons": [],
    "latest_path": "state/gtc-v2/latest/gateboard.json",
    "generated_at": None,
    "open_incident_count": None,
    "verify_status": None,
    "incident_replay_path": "state/gtc-v2/latest/incident_replay.json",
    "incident_replay_commands": [],
}
reconcile_history = {
    "window_hours": 24,
    "checkpoint_count": 0,
    "latest_checkpoint_at": None,
    "latest_checkpoint_id": None,
    "event_emitted_count": 0,
    "event_suppressed_count": 0,
    "latest_event_at": None,
    "latest_event_key": None,
}
incident_replay = {
    "recommended_commands": [],
}
db_path = root / "state" / "continuity" / "continuity_os.sqlite"
if db_path.exists():
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        for status, cnt in cur.execute("SELECT status, COUNT(*) FROM work_queue GROUP BY status"):
            queue_counts[str(status)] = int(cnt)

        role_rows = cur.execute(
            """
SELECT COALESCE(NULLIF(TRIM(role_required), ''), 'UNSET') AS role_key, COUNT(*)
FROM work_queue
GROUP BY role_key
ORDER BY role_key
"""
        ).fetchall()
        for role_key, cnt in role_rows:
            queue_role_required_counts[str(role_key)] = int(cnt or 0)

        queue_role_missing_count = int(queue_role_required_counts.get("UNSET") or 0)

        review_role_row = cur.execute(
            """
SELECT COUNT(*)
FROM work_queue
WHERE status = 'REVIEW'
  AND LOWER(TRIM(COALESCE(role_required, ''))) <> 'validator'
"""
        ).fetchone()
        if review_role_row:
            queue_review_role_mismatch_count = int(review_role_row[0] or 0)

        ready_row = cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if ready_row:
            queue_ready_count = int(ready_row[0] or 0)

        ready_oldest_row = cur.execute(
            """
SELECT MIN(w.updated_at)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if ready_oldest_row:
            queue_ready_oldest_updated_at = ready_oldest_row[0]
            queue_ready_oldest_age_sec = age_sec(queue_ready_oldest_updated_at)

        dep_blocked_row = cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if dep_blocked_row:
            queue_dependency_blocked_count = int(dep_blocked_row[0] or 0)

        dep_example_rows = cur.execute(
            """
SELECT
  w.task_id,
  GROUP_CONCAT(
    d.depends_on_task_id || ':' || COALESCE(dep.status, 'MISSING'),
    ' | '
  ) AS blockers
FROM work_queue w
JOIN task_dependencies d ON d.task_id = w.task_id AND d.relation = 'blocks'
LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
WHERE w.status = 'QUEUED'
  AND COALESCE(dep.status, 'MISSING') <> 'DONE'
GROUP BY w.task_id
ORDER BY w.updated_at DESC, w.task_id ASC
LIMIT 6
"""
        ).fetchall()
        for dep_row in dep_example_rows:
            queue_dependency_blocked_examples.append(
                {
                    "task_id": dep_row[0],
                    "blocked_by": [
                        p.strip()
                        for p in str(dep_row[1] or "").split("|")
                        if str(p).strip()
                    ],
                }
            )

        lock_row = cur.execute("SELECT COUNT(*) FROM file_locks WHERE lock_state = 'ACTIVE'").fetchone()
        if lock_row:
            active_file_lock_count = int(lock_row[0] or 0)

        stale_lock_rows = cur.execute(
            """
SELECT lock_id, file_path, locked_by_task_id, acquired_at, lock_expires_at
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
ORDER BY lock_expires_at ASC
LIMIT 6
""",
            (current_now_iso,),
        ).fetchall()
        stale_active_file_lock_count = len(stale_lock_rows)
        for lrow in stale_lock_rows:
            active_lock_examples.append(
                {
                    "lock_id": lrow[0],
                    "file_path": lrow[1],
                    "locked_by_task_id": lrow[2],
                    "acquired_at": lrow[3],
                    "lock_expires_at": lrow[4],
                }
            )

        orphaned_running_cutoff_iso = (
            clock_now_dt() - dt.timedelta(seconds=int(orphaned_running_min_sec))
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        orphaned_running_count_row = cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
""",
            (orphaned_running_cutoff_iso,),
        ).fetchone()
        if orphaned_running_count_row:
            orphaned_running_without_locks_count = int(orphaned_running_count_row[0] or 0)

        orphaned_running_rows = cur.execute(
            """
SELECT w.task_id, w.assigned_agent, w.role_required, w.updated_at
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
ORDER BY w.updated_at ASC
LIMIT 6
""",
            (orphaned_running_cutoff_iso,),
        ).fetchall()
        for orow in orphaned_running_rows:
            orphaned_running_without_locks_examples.append(
                {
                    "task_id": orow[0],
                    "assigned_agent": orow[1],
                    "role_required": orow[2],
                    "updated_at": orow[3],
                    "stale_cutoff": orphaned_running_cutoff_iso,
                }
            )

        running_status_count = int(queue_counts.get("RUNNING") or 0)
        effective_active_file_lock_count = max(0, active_file_lock_count - stale_active_file_lock_count)
        effective_running_count = max(0, running_status_count - orphaned_running_without_locks_count)
        in_flight_effective = bool(effective_running_count > 0 or effective_active_file_lock_count > 0)

        row = cur.execute(
            "SELECT task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at FROM task_transitions ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            transition_last_at = row[6]
            transition_last_event = {
                "task_id": row[0],
                "from_status": row[1],
                "to_status": row[2],
                "actor_role": row[3],
                "reason": row[4],
                "evidence_ref": row[5],
                "created_at": row[6],
            }

        ev_rows = cur.execute(
            "SELECT evidence_ref FROM task_transitions WHERE evidence_ref IS NOT NULL AND evidence_ref != '' ORDER BY created_at DESC LIMIT 12"
        ).fetchall()
        seen = set()
        for ev_row in ev_rows:
            raw = str(ev_row[0] or "").strip()
            if not raw:
                continue
            for piece in [p.strip() for p in raw.split("|") if str(p).strip()]:
                if piece in seen:
                    continue
                seen.add(piece)
                recent_evidence_refs.append(piece)
                if len(recent_evidence_refs) >= 6:
                    break
            if len(recent_evidence_refs) >= 6:
                break

        transition_cutoff_iso = (
            clock_now_dt() - dt.timedelta(hours=int(transition_history_24h["window_hours"]))
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        tr_total_row = cur.execute(
            "SELECT COUNT(*) FROM task_transitions WHERE created_at >= ?",
            (transition_cutoff_iso,),
        ).fetchone()
        if tr_total_row:
            transition_history_24h["total"] = int(tr_total_row[0] or 0)

        tr_to_rows = cur.execute(
            "SELECT to_status, COUNT(*) FROM task_transitions WHERE created_at >= ? GROUP BY to_status",
            (transition_cutoff_iso,),
        ).fetchall()
        for tr_row in tr_to_rows:
            transition_history_24h["to_status"][str(tr_row[0] or "")] = int(tr_row[1] or 0)

        tr_actor_rows = cur.execute(
            "SELECT actor_role, COUNT(*) FROM task_transitions WHERE created_at >= ? GROUP BY actor_role",
            (transition_cutoff_iso,),
        ).fetchall()
        for tr_row in tr_actor_rows:
            transition_history_24h["actor_role"][str(tr_row[0] or "")] = int(tr_row[1] or 0)

        handoff_total_row = cur.execute(
            "SELECT COUNT(*) FROM task_handoff_packets WHERE created_at >= ?",
            (transition_cutoff_iso,),
        ).fetchone()
        if handoff_total_row:
            handoff_history_24h["total"] = int(handoff_total_row[0] or 0)

        for hrow in cur.execute(
            """
SELECT from_role || '->' || to_role AS edge, COUNT(*)
FROM task_handoff_packets
WHERE created_at >= ?
GROUP BY edge
ORDER BY COUNT(*) DESC, edge ASC
""",
            (transition_cutoff_iso,),
        ).fetchall():
            handoff_history_24h["role_edges"][str(hrow[0] or "")] = int(hrow[1] or 0)

        latest_handoff_row = cur.execute(
            """
SELECT packet_id, task_id, from_role, to_role, from_status, to_status, created_at, transition_event_id
FROM task_handoff_packets
ORDER BY created_at DESC
LIMIT 1
"""
        ).fetchone()
        if latest_handoff_row:
            latest_handoff_packet = {
                "packet_id": latest_handoff_row[0],
                "task_id": latest_handoff_row[1],
                "from_role": latest_handoff_row[2],
                "to_role": latest_handoff_row[3],
                "from_status": latest_handoff_row[4],
                "to_status": latest_handoff_row[5],
                "created_at": latest_handoff_row[6],
                "transition_event_id": latest_handoff_row[7],
            }

        cutoff_iso = (
            clock_now_dt() - dt.timedelta(hours=int(reconcile_history["window_hours"]))
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        cp_count_row = cur.execute(
            "SELECT COUNT(*), MAX(created_at) FROM checkpoints WHERE trigger = 'drift_reconcile' AND created_at >= ?",
            (cutoff_iso,),
        ).fetchone()
        if cp_count_row:
            reconcile_history["checkpoint_count"] = int(cp_count_row[0] or 0)
            reconcile_history["latest_checkpoint_at"] = cp_count_row[1]

        cp_last_row = cur.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE trigger = 'drift_reconcile' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if cp_last_row:
            reconcile_history["latest_checkpoint_id"] = cp_last_row[0]

        ev_count_row = cur.execute(
            """
            SELECT
              SUM(CASE WHEN emitted = 1 THEN 1 ELSE 0 END) AS emitted_count,
              SUM(CASE WHEN emitted = 0 THEN 1 ELSE 0 END) AS suppressed_count,
              MAX(created_at) AS latest_event_at
            FROM continuity_events
            WHERE source = 'continuity.reconcile' AND created_at >= ?
            """,
            (cutoff_iso,),
        ).fetchone()
        if ev_count_row:
            reconcile_history["event_emitted_count"] = int(ev_count_row[0] or 0)
            reconcile_history["event_suppressed_count"] = int(ev_count_row[1] or 0)
            reconcile_history["latest_event_at"] = ev_count_row[2]

        ev_last_row = cur.execute(
            "SELECT event_key FROM continuity_events WHERE source = 'continuity.reconcile' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if ev_last_row:
            reconcile_history["latest_event_key"] = ev_last_row[0]

        parity_row = cur.execute(
            "SELECT status FROM work_queue WHERE task_id = 'parity:weekly_harness'"
        ).fetchone()
        if parity_row:
            parity_summary["status"] = parity_row[0]

        parity_done_row = cur.execute(
            """
            SELECT created_at
            FROM task_transitions
            WHERE task_id = 'parity:weekly_harness' AND to_status = 'DONE'
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if parity_done_row and parity_done_row[0]:
            last_done_at = str(parity_done_row[0])
            parity_summary["last_done_at"] = last_done_at
            parsed = parse_iso(last_done_at)
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.timezone.utc)
                age = max(0, int((clock_now_dt() - parsed).total_seconds()))
                parity_summary["last_done_age_sec"] = age
                parity_summary["fresh"] = age <= int(parity_summary["freshness_limit_sec"])
                parity_summary["due"] = age > int(parity_summary["freshness_limit_sec"])

        con.close()
    except Exception:
        pass

# Proactive ghost RUNNING remediation:
# when orphaned RUNNING exists but effective in-flight is false, attempt bounded requeue.
current_eval_ts = clock_now_ts()
current_eval_iso = clock_now_iso()
prior_auto_state = load_json(orphaned_running_auto_remediation_state_path)
if isinstance(prior_auto_state, dict):
    orphaned_running_auto_remediation["last_attempt_at"] = prior_auto_state.get("last_attempt_at")
    orphaned_running_auto_remediation["last_success_at"] = prior_auto_state.get("last_success_at")
    orphaned_running_auto_remediation["last_failure_at"] = prior_auto_state.get("last_failure_at")
    orphaned_running_auto_remediation["next_attempt_after_iso"] = prior_auto_state.get("next_attempt_after_iso")

next_attempt_after_ts = None
if isinstance(prior_auto_state, dict):
    try:
        next_attempt_after_ts = int(prior_auto_state.get("next_attempt_after_ts"))
    except Exception:
        next_attempt_after_ts = iso_to_ts(prior_auto_state.get("next_attempt_after_iso"))

if next_attempt_after_ts is None:
    next_attempt_after_ts = 0

orphaned_running_auto_remediation["last_eval_at"] = current_eval_iso
orphaned_running_auto_remediation["orphaned_running_without_locks_count"] = int(orphaned_running_without_locks_count)
orphaned_running_auto_remediation["effective_running_count"] = int(effective_running_count)
orphaned_running_auto_remediation["effective_active_file_lock_count"] = int(effective_active_file_lock_count)
orphaned_running_auto_remediation["in_flight_effective"] = bool(in_flight_effective)

remediate_eligible = bool(
    orphaned_running_auto_remediate_enabled
    and orphaned_running_without_locks_count > 0
    and not in_flight_effective
)
orphaned_running_auto_remediation["eligible"] = remediate_eligible

if not orphaned_running_auto_remediate_enabled:
    orphaned_running_auto_remediation["status"] = "disabled"
    orphaned_running_auto_remediation["reason"] = "disabled_by_env"
elif not remediate_eligible:
    orphaned_running_auto_remediation["status"] = "not_eligible"
    if orphaned_running_without_locks_count <= 0:
        orphaned_running_auto_remediation["reason"] = "no_orphaned_running"
    elif in_flight_effective:
        orphaned_running_auto_remediation["reason"] = "in_flight_effective_true"
    else:
        orphaned_running_auto_remediation["reason"] = "safety_guard"
    orphaned_running_auto_remediation["cooldown_remaining_sec"] = 0
    next_attempt_after_ts = 0
    orphaned_running_auto_remediation["next_attempt_after_iso"] = None
else:
    cooldown_remaining_sec = max(0, int(next_attempt_after_ts - current_eval_ts)) if next_attempt_after_ts else 0
    if orphaned_running_auto_remediate_cooldown_sec > 0 and cooldown_remaining_sec > 0:
        orphaned_running_auto_remediation["status"] = "cooldown_active"
        orphaned_running_auto_remediation["reason"] = "cooldown_active"
        orphaned_running_auto_remediation["cooldown_remaining_sec"] = int(cooldown_remaining_sec)
    else:
        queue_arb_path = root / "ops" / "openclaw" / "continuity" / "queue_arbitrator.sh"
        orphaned_running_auto_remediation["attempted"] = True
        orphaned_running_auto_remediation["triggered"] = True
        orphaned_running_auto_remediation["last_attempt_at"] = current_eval_iso

        if not (queue_arb_path.exists() and os.access(queue_arb_path, os.X_OK)):
            orphaned_running_auto_remediation["status"] = "remediator_missing"
            orphaned_running_auto_remediation["reason"] = "queue_arbitrator_missing"
        else:
            remediate_cmd = [
                str(queue_arb_path),
                "remediate",
                "--requeue-orphaned-running",
                "--orphaned-running-min-sec",
                str(orphaned_running_min_sec),
                "--apply",
                "--json",
            ]
            run_env = os.environ.copy()
            run_env["OPENCLAW_ROOT"] = str(root)
            run_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
            run_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "continuity_now.sh:auto_orphaned_running_remediation"
            cp = subprocess.run(
                remediate_cmd,
                text=True,
                capture_output=True,
                check=False,
                env=run_env,
            )
            orphaned_running_auto_remediation["command_rc"] = int(cp.returncode)
            stdout_tail = str(cp.stdout or "").strip()[-240:]
            stderr_tail = str(cp.stderr or "").strip()[-240:]
            if stdout_tail:
                orphaned_running_auto_remediation["stdout_tail"] = stdout_tail
            if stderr_tail:
                orphaned_running_auto_remediation["stderr_tail"] = stderr_tail

            payload = {}
            if cp.returncode == 0:
                try:
                    maybe_payload = json.loads(cp.stdout or "{}")
                    if isinstance(maybe_payload, dict):
                        payload = maybe_payload
                except Exception:
                    payload = {}

            if cp.returncode != 0:
                orphaned_running_auto_remediation["status"] = "command_failed"
                orphaned_running_auto_remediation["reason"] = "queue_arbitrator_nonzero"
                orphaned_running_auto_remediation["last_failure_at"] = current_eval_iso
            elif not bool(payload.get("ok")):
                orphaned_running_auto_remediation["status"] = "payload_not_ok"
                orphaned_running_auto_remediation["reason"] = "queue_arbitrator_payload_not_ok"
                orphaned_running_auto_remediation["last_failure_at"] = current_eval_iso
            else:
                try:
                    requeued_count = int(((payload.get("applied") or {}).get("requeued_orphaned_running") or 0))
                except Exception:
                    requeued_count = 0
                orphaned_running_auto_remediation["applied_requeued_orphaned_running"] = int(requeued_count)
                if requeued_count > 0:
                    orphaned_running_auto_remediation["status"] = "remediated"
                    orphaned_running_auto_remediation["reason"] = "orphaned_running_requeued"
                    orphaned_running_auto_remediation["last_success_at"] = current_eval_iso
                else:
                    orphaned_running_auto_remediation["status"] = "noop"
                    orphaned_running_auto_remediation["reason"] = "no_orphaned_rows_applied"

        if orphaned_running_auto_remediate_cooldown_sec > 0:
            next_attempt_after_ts = int(current_eval_ts) + int(orphaned_running_auto_remediate_cooldown_sec)
        else:
            next_attempt_after_ts = 0
        orphaned_running_auto_remediation["cooldown_remaining_sec"] = max(
            0,
            int(next_attempt_after_ts - current_eval_ts),
        ) if next_attempt_after_ts else 0

if next_attempt_after_ts and next_attempt_after_ts > 0:
    orphaned_running_auto_remediation["next_attempt_after_ts"] = int(next_attempt_after_ts)
    orphaned_running_auto_remediation["next_attempt_after_iso"] = (
        dt.datetime.fromtimestamp(int(next_attempt_after_ts), tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
else:
    orphaned_running_auto_remediation["next_attempt_after_ts"] = 0
    orphaned_running_auto_remediation["next_attempt_after_iso"] = None

# If remediation attempted, refresh queue in-flight truth from DB so downstream surfaces see settled state.
if orphaned_running_auto_remediation.get("attempted") and db_path.exists():
    refresh_con = None
    try:
        refresh_con = sqlite3.connect(db_path)
        refresh_cur = refresh_con.cursor()

        running_row = refresh_cur.execute(
            "SELECT COUNT(*) FROM work_queue WHERE status = 'RUNNING'"
        ).fetchone()
        if running_row:
            queue_counts["RUNNING"] = int(running_row[0] or 0)

        active_row = refresh_cur.execute(
            "SELECT COUNT(*) FROM file_locks WHERE lock_state = 'ACTIVE'"
        ).fetchone()
        if active_row:
            active_file_lock_count = int(active_row[0] or 0)

        stale_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
""",
            (current_eval_iso,),
        ).fetchone()
        if stale_row:
            stale_active_file_lock_count = int(stale_row[0] or 0)
            if stale_active_file_lock_count <= 0:
                active_lock_examples = []

        orphaned_cutoff_refresh = (
            clock_now_dt() - dt.timedelta(seconds=int(orphaned_running_min_sec))
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        orphaned_refresh_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
""",
            (orphaned_cutoff_refresh,),
        ).fetchone()
        if orphaned_refresh_row:
            orphaned_running_without_locks_count = int(orphaned_refresh_row[0] or 0)
            if orphaned_running_without_locks_count <= 0:
                orphaned_running_without_locks_examples = []

        running_status_count = int(queue_counts.get("RUNNING") or 0)
        effective_active_file_lock_count = max(0, active_file_lock_count - stale_active_file_lock_count)
        effective_running_count = max(0, running_status_count - orphaned_running_without_locks_count)
        in_flight_effective = bool(effective_running_count > 0 or effective_active_file_lock_count > 0)

        orphaned_running_auto_remediation["orphaned_running_without_locks_count"] = int(orphaned_running_without_locks_count)
        orphaned_running_auto_remediation["effective_running_count"] = int(effective_running_count)
        orphaned_running_auto_remediation["effective_active_file_lock_count"] = int(effective_active_file_lock_count)
        orphaned_running_auto_remediation["in_flight_effective"] = bool(in_flight_effective)
    except Exception:
        pass
    finally:
        if refresh_con is not None:
            try:
                refresh_con.close()
            except Exception:
                pass

atomic_write(orphaned_running_auto_remediation_state_path, orphaned_running_auto_remediation)

orphaned_running_auto_remediation_contract_issues: List[str] = []
orphaned_running_auto_remediation_contract_state = load_json(orphaned_running_auto_remediation_state_path)
orphaned_running_contract_expected_schema = "continuity.queue_orphaned_running_auto_remediation.v1"
orphaned_running_projection_present = bool(
    isinstance(orphaned_running_auto_remediation, dict) and orphaned_running_auto_remediation
)
orphaned_running_projection_schema_version = str(
    ((orphaned_running_auto_remediation or {}).get("schema_version") if isinstance(orphaned_running_auto_remediation, dict) else "") or ""
).strip()
orphaned_running_projection_state_path = str(
    ((orphaned_running_auto_remediation or {}).get("state_path") if isinstance(orphaned_running_auto_remediation, dict) else "") or ""
).strip()

orphaned_running_state_exists = orphaned_running_auto_remediation_state_path.exists()
orphaned_running_state_valid = bool(
    isinstance(orphaned_running_auto_remediation_contract_state, dict)
    and orphaned_running_auto_remediation_contract_state
)
orphaned_running_state_schema_version = str(
    ((orphaned_running_auto_remediation_contract_state or {}).get("schema_version") if isinstance(orphaned_running_auto_remediation_contract_state, dict) else "") or ""
).strip()
orphaned_running_state_state_path = str(
    ((orphaned_running_auto_remediation_contract_state or {}).get("state_path") if isinstance(orphaned_running_auto_remediation_contract_state, dict) else "") or ""
).strip()

if not orphaned_running_projection_present:
    orphaned_running_auto_remediation_contract_issues.append("projection_missing")
if orphaned_running_projection_schema_version != orphaned_running_contract_expected_schema:
    orphaned_running_auto_remediation_contract_issues.append("projection_schema_invalid")
if orphaned_running_projection_state_path != orphaned_running_auto_remediation_state_rel:
    orphaned_running_auto_remediation_contract_issues.append("projection_state_path_mismatch")

if not orphaned_running_state_exists:
    orphaned_running_auto_remediation_contract_issues.append("state_missing")
elif not orphaned_running_state_valid:
    orphaned_running_auto_remediation_contract_issues.append("state_invalid")

if orphaned_running_state_valid:
    if orphaned_running_state_schema_version != orphaned_running_contract_expected_schema:
        orphaned_running_auto_remediation_contract_issues.append("state_schema_invalid")
    if orphaned_running_state_state_path != orphaned_running_auto_remediation_state_rel:
        orphaned_running_auto_remediation_contract_issues.append("state_state_path_mismatch")

orphaned_running_projection_state_match: Optional[bool] = None
if orphaned_running_projection_present and orphaned_running_state_valid:
    try:
        projection_payload_canonical = json.dumps(
            orphaned_running_auto_remediation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        state_payload_canonical = json.dumps(
            orphaned_running_auto_remediation_contract_state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        orphaned_running_projection_state_match = projection_payload_canonical == state_payload_canonical
    except Exception:
        orphaned_running_projection_state_match = None
        orphaned_running_auto_remediation_contract_issues.append("projection_state_compare_failed")

if orphaned_running_projection_state_match is False:
    orphaned_running_auto_remediation_contract_issues.append("projection_state_mismatch")

orphaned_running_auto_remediation_contract_healthy = len(orphaned_running_auto_remediation_contract_issues) == 0
orphaned_running_auto_remediation_contract = {
    "schema_version": "continuity.queue_orphaned_running_auto_remediation_contract.v1",
    "status": "ok" if orphaned_running_auto_remediation_contract_healthy else "degraded",
    "healthy": orphaned_running_auto_remediation_contract_healthy,
    "checked_at": clock_now_iso(),
    "projection_path": "queue.orphaned_running_auto_remediation",
    "state_path": orphaned_running_auto_remediation_state_rel,
    "projection_present": orphaned_running_projection_present,
    "projection_schema_version": orphaned_running_projection_schema_version or None,
    "projection_state_path": orphaned_running_projection_state_path or None,
    "state_exists": orphaned_running_state_exists,
    "state_valid": orphaned_running_state_valid,
    "state_schema_version": orphaned_running_state_schema_version or None,
    "state_state_path": orphaned_running_state_state_path or None,
    "projection_state_match": orphaned_running_projection_state_match,
    "issues": orphaned_running_auto_remediation_contract_issues,
}

def derive_queue_stale_wave_state(
    ready_count: int,
    in_flight: bool,
    ready_oldest_age: Optional[int],
    idle_threshold_sec: int,
) -> Any:
    is_active = bool(
        ready_count > 0
        and not in_flight
        and ready_oldest_age is not None
        and ready_oldest_age >= int(idle_threshold_sec)
    )
    if is_active:
        return True, "ready_backlog_idle"
    if ready_count <= 0:
        return False, "no_ready_backlog"
    if in_flight:
        return False, "in_flight_active"
    if ready_oldest_age is None:
        return False, "ready_backlog_age_unknown"
    return False, "ready_backlog_within_idle_window"


QUEUE_STALE_WAVE_AUTO_FAILURE_STATUSES = {
    "command_failed",
    "payload_not_ok",
    "remediator_missing",
    "remediation_degraded",
}

QUEUE_STALE_WAVE_AUTO_FAILURE_TAXONOMY = {
    "remediator_missing": {
        "category": "dependency_missing",
        "code": "continuity_sh_missing",
        "retryable": True,
    },
    "command_failed": {
        "category": "command_execution_failed",
        "code": "queue_sync_nonzero",
        "retryable": True,
    },
    "payload_not_ok": {
        "category": "command_payload_invalid",
        "code": "queue_sync_payload_not_ok",
        "retryable": True,
    },
    "remediation_degraded": {
        "category": "post_check_failed",
        "code": "queue_sync_no_recovery",
        "retryable": True,
    },
}


def derive_queue_stale_wave_failure_details(status: Any, reason: Any) -> Dict[str, Any]:
    status_txt = str(status or "").strip()
    reason_txt = str(reason or "").strip()
    base = dict(QUEUE_STALE_WAVE_AUTO_FAILURE_TAXONOMY.get(status_txt) or {
        "category": "unknown_failure",
        "code": "unknown_failure",
        "retryable": True,
    })
    if status_txt == "payload_not_ok" and reason_txt == "queue_sync_payload_invalid_json":
        base["code"] = "queue_sync_payload_invalid_json"
    elif status_txt == "remediation_degraded" and reason_txt:
        base["code"] = reason_txt
    return {
        "category": str(base.get("category") or "unknown_failure"),
        "code": str(base.get("code") or "unknown_failure"),
        "retryable": bool(base.get("retryable") is not False),
    }


ready_backlog_stale_before, ready_backlog_stale_reason_before = derive_queue_stale_wave_state(
    int(queue_ready_count),
    bool(in_flight_effective),
    queue_ready_oldest_age_sec,
    int(queue_stale_wave_ready_idle_sec),
)

queue_stale_wave_auto_remediation["last_eval_at"] = current_eval_iso
queue_stale_wave_auto_remediation["queue_stale_wave_active_before"] = bool(ready_backlog_stale_before)
queue_stale_wave_auto_remediation["ready_count_before"] = int(queue_ready_count)
queue_stale_wave_auto_remediation["ready_oldest_age_sec_before"] = queue_ready_oldest_age_sec
queue_stale_wave_auto_remediation["in_flight_effective_before"] = bool(in_flight_effective)
queue_stale_wave_auto_remediation["last_transition_at_before"] = transition_last_at
queue_stale_wave_auto_remediation["last_handoff_at_before"] = (
    latest_handoff_packet.get("created_at") if isinstance(latest_handoff_packet, dict) else None
)

prior_queue_stale_wave_auto_state = load_json(queue_stale_wave_auto_remediation_state_path)
prior_queue_wave_attempt_sequence = 0
prior_queue_wave_consecutive_failures = 0
prior_queue_wave_reason = str((prior_queue_stale_wave_auto_state.get("reason") if isinstance(prior_queue_stale_wave_auto_state, dict) else "") or "").strip()
if isinstance(prior_queue_stale_wave_auto_state, dict):
    queue_stale_wave_auto_remediation["last_attempt_at"] = prior_queue_stale_wave_auto_state.get("last_attempt_at")
    queue_stale_wave_auto_remediation["last_attempt_status"] = prior_queue_stale_wave_auto_state.get("last_attempt_status")
    queue_stale_wave_auto_remediation["last_success_at"] = prior_queue_stale_wave_auto_state.get("last_success_at")
    queue_stale_wave_auto_remediation["last_failure_at"] = prior_queue_stale_wave_auto_state.get("last_failure_at")
    queue_stale_wave_auto_remediation["next_attempt_after_iso"] = prior_queue_stale_wave_auto_state.get("next_attempt_after_iso")
    queue_stale_wave_auto_remediation["command_rc"] = prior_queue_stale_wave_auto_state.get("command_rc")
    queue_stale_wave_auto_remediation["command_ok"] = prior_queue_stale_wave_auto_state.get("command_ok")
    queue_stale_wave_auto_remediation["payload_ok"] = prior_queue_stale_wave_auto_state.get("payload_ok")
    queue_stale_wave_auto_remediation["payload_parse_error"] = prior_queue_stale_wave_auto_state.get("payload_parse_error")
    if isinstance(prior_queue_stale_wave_auto_state.get("retry_contract"), dict):
        queue_stale_wave_auto_remediation["retry_contract"] = dict(prior_queue_stale_wave_auto_state.get("retry_contract") or {})
    if isinstance(prior_queue_stale_wave_auto_state.get("failure_evidence"), dict):
        queue_stale_wave_auto_remediation["failure_evidence"] = dict(prior_queue_stale_wave_auto_state.get("failure_evidence") or {})
    try:
        prior_queue_wave_attempt_sequence = max(0, int(prior_queue_stale_wave_auto_state.get("attempt_sequence") or 0))
    except Exception:
        prior_queue_wave_attempt_sequence = 0
    try:
        prior_queue_wave_consecutive_failures = max(0, int(prior_queue_stale_wave_auto_state.get("consecutive_failures") or 0))
    except Exception:
        prior_queue_wave_consecutive_failures = 0

queue_stale_wave_auto_remediation["attempt_sequence"] = int(prior_queue_wave_attempt_sequence)
queue_stale_wave_auto_remediation["consecutive_failures"] = int(prior_queue_wave_consecutive_failures)

queue_wave_next_attempt_after_ts = None
if isinstance(prior_queue_stale_wave_auto_state, dict):
    try:
        queue_wave_next_attempt_after_ts = int(prior_queue_stale_wave_auto_state.get("next_attempt_after_ts"))
    except Exception:
        queue_wave_next_attempt_after_ts = iso_to_ts(prior_queue_stale_wave_auto_state.get("next_attempt_after_iso"))

if queue_wave_next_attempt_after_ts is None:
    queue_wave_next_attempt_after_ts = 0

queue_wave_remediate_eligible = bool(
    queue_stale_wave_auto_remediate_enabled
    and int(queue_ready_count) > 0
    and not bool(in_flight_effective)
    and queue_ready_oldest_age_sec is not None
    and int(queue_ready_oldest_age_sec) >= int(queue_stale_wave_ready_idle_sec)
)
queue_stale_wave_auto_remediation["eligible"] = queue_wave_remediate_eligible

if not queue_stale_wave_auto_remediate_enabled:
    queue_stale_wave_auto_remediation["status"] = "disabled"
    queue_stale_wave_auto_remediation["reason"] = "disabled_by_env"
elif not queue_wave_remediate_eligible:
    queue_stale_wave_auto_remediation["status"] = "not_eligible"
    queue_stale_wave_auto_remediation["reason"] = str(ready_backlog_stale_reason_before)
    queue_stale_wave_auto_remediation["cooldown_remaining_sec"] = 0
    queue_wave_next_attempt_after_ts = 0
    queue_stale_wave_auto_remediation["next_attempt_after_iso"] = None
else:
    queue_wave_cooldown_remaining_sec = max(0, int(queue_wave_next_attempt_after_ts - current_eval_ts)) if queue_wave_next_attempt_after_ts else 0
    if queue_stale_wave_auto_remediate_cooldown_sec > 0 and queue_wave_cooldown_remaining_sec > 0:
        queue_stale_wave_auto_remediation["status"] = "cooldown_active"
        queue_stale_wave_auto_remediation["reason"] = "cooldown_active"
        queue_stale_wave_auto_remediation["cooldown_remaining_sec"] = int(queue_wave_cooldown_remaining_sec)
    else:
        queue_stale_wave_auto_remediation["attempted"] = True
        queue_stale_wave_auto_remediation["triggered"] = True
        queue_stale_wave_auto_remediation["last_attempt_at"] = current_eval_iso

        queue_sync_path = root / "ops" / "openclaw" / "continuity" / "queue_sync_from_autopilot_json.sh"
        if not queue_sync_path.exists():
            queue_stale_wave_auto_remediation["status"] = "remediator_missing"
            queue_stale_wave_auto_remediation["reason"] = "queue_sync_script_missing"
            queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
            queue_stale_wave_auto_remediation["command_ok"] = False
            queue_stale_wave_auto_remediation["payload_ok"] = None
            queue_stale_wave_auto_remediation["payload_parse_error"] = None
        else:
            remediate_cmd = [
                "bash",
                str(queue_sync_path),
            ]
            run_env = os.environ.copy()
            run_env["OPENCLAW_ROOT"] = str(root)
            run_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
            run_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "continuity_now.sh:auto_queue_stale_wave_remediation"
            cp = subprocess.run(
                remediate_cmd,
                text=True,
                capture_output=True,
                check=False,
                env=run_env,
            )
            queue_stale_wave_auto_remediation["command_rc"] = int(cp.returncode)

            stdout_tail = str(cp.stdout or "").strip()[-240:]
            stderr_tail = str(cp.stderr or "").strip()[-240:]
            if stdout_tail:
                queue_stale_wave_auto_remediation["stdout_tail"] = stdout_tail
            if stderr_tail:
                queue_stale_wave_auto_remediation["stderr_tail"] = stderr_tail

            payload = {}
            payload_parse_error = None
            if cp.returncode == 0:
                try:
                    maybe_payload = json.loads(cp.stdout or "{}")
                    if isinstance(maybe_payload, dict):
                        payload = maybe_payload
                    else:
                        payload_parse_error = "payload_not_object"
                except Exception as exc:
                    payload_parse_error = f"json_decode_error:{exc.__class__.__name__}"
                    payload = {}

            payload_ok = bool(payload.get("ok")) if isinstance(payload, dict) and payload else False
            queue_stale_wave_auto_remediation["payload_ok"] = payload_ok if cp.returncode == 0 else None
            queue_stale_wave_auto_remediation["payload_parse_error"] = payload_parse_error

            if cp.returncode != 0:
                queue_stale_wave_auto_remediation["status"] = "command_failed"
                queue_stale_wave_auto_remediation["reason"] = "queue_sync_nonzero"
                queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
                queue_stale_wave_auto_remediation["command_ok"] = False
            elif payload_parse_error:
                queue_stale_wave_auto_remediation["status"] = "payload_not_ok"
                queue_stale_wave_auto_remediation["reason"] = "queue_sync_payload_invalid_json"
                queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
                queue_stale_wave_auto_remediation["command_ok"] = False
            elif not payload_ok:
                queue_stale_wave_auto_remediation["status"] = "payload_not_ok"
                queue_stale_wave_auto_remediation["reason"] = "queue_sync_payload_not_ok"
                queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
                queue_stale_wave_auto_remediation["command_ok"] = False
            else:
                queue_stale_wave_auto_remediation["status"] = "queue_sync_applied"
                queue_stale_wave_auto_remediation["reason"] = "queue_sync_ok"
                queue_stale_wave_auto_remediation["command_ok"] = True

        if queue_stale_wave_auto_remediate_cooldown_sec > 0:
            queue_wave_next_attempt_after_ts = int(current_eval_ts) + int(queue_stale_wave_auto_remediate_cooldown_sec)
        else:
            queue_wave_next_attempt_after_ts = 0
        queue_stale_wave_auto_remediation["cooldown_remaining_sec"] = max(
            0,
            int(queue_wave_next_attempt_after_ts - current_eval_ts),
        ) if queue_wave_next_attempt_after_ts else 0

if queue_wave_next_attempt_after_ts and queue_wave_next_attempt_after_ts > 0:
    queue_stale_wave_auto_remediation["next_attempt_after_ts"] = int(queue_wave_next_attempt_after_ts)
    queue_stale_wave_auto_remediation["next_attempt_after_iso"] = (
        dt.datetime.fromtimestamp(int(queue_wave_next_attempt_after_ts), tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
else:
    queue_stale_wave_auto_remediation["next_attempt_after_ts"] = 0
    queue_stale_wave_auto_remediation["next_attempt_after_iso"] = None

# If stale-wave remediation attempted, refresh queue truth from DB so stale-wave signal reflects settled state.
if queue_stale_wave_auto_remediation.get("attempted") and db_path.exists():
    refresh_con = None
    try:
        refresh_con = sqlite3.connect(db_path)
        refresh_cur = refresh_con.cursor()

        queue_counts = {}
        for status, cnt in refresh_cur.execute("SELECT status, COUNT(*) FROM work_queue GROUP BY status"):
            queue_counts[str(status)] = int(cnt)

        queue_role_required_counts = {}
        role_rows = refresh_cur.execute(
            """
SELECT COALESCE(NULLIF(TRIM(role_required), ''), 'UNSET') AS role_key, COUNT(*)
FROM work_queue
GROUP BY role_key
ORDER BY role_key
"""
        ).fetchall()
        for role_key, cnt in role_rows:
            queue_role_required_counts[str(role_key)] = int(cnt or 0)
        queue_role_missing_count = int(queue_role_required_counts.get("UNSET") or 0)

        review_role_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM work_queue
WHERE status = 'REVIEW'
  AND LOWER(TRIM(COALESCE(role_required, ''))) <> 'validator'
"""
        ).fetchone()
        if review_role_row:
            queue_review_role_mismatch_count = int(review_role_row[0] or 0)

        ready_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if ready_row:
            queue_ready_count = int(ready_row[0] or 0)

        ready_oldest_row = refresh_cur.execute(
            """
SELECT MIN(w.updated_at)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if ready_oldest_row:
            queue_ready_oldest_updated_at = ready_oldest_row[0]
            queue_ready_oldest_age_sec = age_sec(queue_ready_oldest_updated_at)

        dep_blocked_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        if dep_blocked_row:
            queue_dependency_blocked_count = int(dep_blocked_row[0] or 0)

        running_status_count = int(queue_counts.get("RUNNING") or 0)
        effective_running_count = max(0, running_status_count - orphaned_running_without_locks_count)
        in_flight_effective = bool(effective_running_count > 0 or effective_active_file_lock_count > 0)

        orphaned_cutoff_refresh = (
            clock_now_dt() - dt.timedelta(seconds=int(orphaned_running_min_sec))
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        orphaned_refresh_row = refresh_cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
""",
            (orphaned_cutoff_refresh,),
        ).fetchone()
        if orphaned_refresh_row:
            orphaned_running_without_locks_count = int(orphaned_refresh_row[0] or 0)
            if orphaned_running_without_locks_count <= 0:
                orphaned_running_without_locks_examples = []

        running_status_count = int(queue_counts.get("RUNNING") or 0)
        effective_running_count = max(0, running_status_count - orphaned_running_without_locks_count)
        in_flight_effective = bool(effective_running_count > 0 or effective_active_file_lock_count > 0)

        refreshed_transition_row = refresh_cur.execute(
            "SELECT task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at FROM task_transitions ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if refreshed_transition_row:
            transition_last_at = refreshed_transition_row[6]
            transition_last_event = {
                "task_id": refreshed_transition_row[0],
                "from_status": refreshed_transition_row[1],
                "to_status": refreshed_transition_row[2],
                "actor_role": refreshed_transition_row[3],
                "reason": refreshed_transition_row[4],
                "evidence_ref": refreshed_transition_row[5],
                "created_at": refreshed_transition_row[6],
            }

        refreshed_handoff_row = refresh_cur.execute(
            """
SELECT packet_id, task_id, from_role, to_role, from_status, to_status, created_at, transition_event_id
FROM task_handoff_packets
ORDER BY created_at DESC
LIMIT 1
"""
        ).fetchone()
        if refreshed_handoff_row:
            latest_handoff_packet = {
                "packet_id": refreshed_handoff_row[0],
                "task_id": refreshed_handoff_row[1],
                "from_role": refreshed_handoff_row[2],
                "to_role": refreshed_handoff_row[3],
                "from_status": refreshed_handoff_row[4],
                "to_status": refreshed_handoff_row[5],
                "created_at": refreshed_handoff_row[6],
                "transition_event_id": refreshed_handoff_row[7],
            }
    except Exception:
        pass
    finally:
        if refresh_con is not None:
            try:
                refresh_con.close()
            except Exception:
                pass

ready_backlog_stale, ready_backlog_stale_reason = derive_queue_stale_wave_state(
    int(queue_ready_count),
    bool(in_flight_effective),
    queue_ready_oldest_age_sec,
    int(queue_stale_wave_ready_idle_sec),
)

queue_stale_wave_signal["active"] = bool(ready_backlog_stale)
queue_stale_wave_signal["reason"] = str(ready_backlog_stale_reason)
queue_stale_wave_signal["ready_count"] = int(queue_ready_count)
queue_stale_wave_signal["ready_oldest_updated_at"] = queue_ready_oldest_updated_at
queue_stale_wave_signal["ready_oldest_age_sec"] = queue_ready_oldest_age_sec
queue_stale_wave_signal["in_flight_effective"] = bool(in_flight_effective)
queue_stale_wave_signal["last_transition_at"] = transition_last_at
queue_stale_wave_signal["last_transition_age_sec"] = age_sec(transition_last_at)
latest_handoff_at = latest_handoff_packet.get("created_at") if isinstance(latest_handoff_packet, dict) else None
queue_stale_wave_signal["last_handoff_at"] = latest_handoff_at
queue_stale_wave_signal["last_handoff_age_sec"] = age_sec(latest_handoff_at)

queue_stale_wave_auto_remediation["queue_stale_wave_active_after"] = bool(ready_backlog_stale)
queue_stale_wave_auto_remediation["ready_count_after"] = int(queue_ready_count)
queue_stale_wave_auto_remediation["ready_oldest_age_sec_after"] = queue_ready_oldest_age_sec
queue_stale_wave_auto_remediation["in_flight_effective_after"] = bool(in_flight_effective)
queue_stale_wave_auto_remediation["last_transition_at_after"] = transition_last_at
queue_stale_wave_auto_remediation["last_handoff_at_after"] = latest_handoff_at

queue_wave_progress_reasons: List[str] = []
queue_wave_ready_count_before = int(queue_stale_wave_auto_remediation.get("ready_count_before") or 0)
queue_wave_ready_count_after = int(queue_ready_count)
if queue_wave_ready_count_after < queue_wave_ready_count_before:
    queue_wave_progress_reasons.append("ready_count_reduced")
if bool(in_flight_effective):
    queue_wave_progress_reasons.append("in_flight_effective_after")

queue_wave_transition_before_ts = iso_to_ts(queue_stale_wave_auto_remediation.get("last_transition_at_before"))
queue_wave_transition_after_ts = iso_to_ts(transition_last_at)
if queue_wave_transition_after_ts is not None and (
    queue_wave_transition_before_ts is None
    or int(queue_wave_transition_after_ts) > int(queue_wave_transition_before_ts)
):
    queue_wave_progress_reasons.append("transition_advanced")

queue_wave_handoff_before_ts = iso_to_ts(queue_stale_wave_auto_remediation.get("last_handoff_at_before"))
queue_wave_handoff_after_ts = iso_to_ts(latest_handoff_at)
if queue_wave_handoff_after_ts is not None and (
    queue_wave_handoff_before_ts is None
    or int(queue_wave_handoff_after_ts) > int(queue_wave_handoff_before_ts)
):
    queue_wave_progress_reasons.append("handoff_advanced")

queue_wave_progress_detected = bool(queue_wave_progress_reasons)
queue_stale_wave_auto_remediation["progress_detected"] = queue_wave_progress_detected
queue_stale_wave_auto_remediation["progress_reasons"] = queue_wave_progress_reasons

if queue_stale_wave_auto_remediation.get("attempted"):
    attempted_status = str(queue_stale_wave_auto_remediation.get("status") or "").strip()
    if attempted_status == "queue_sync_applied":
        if bool(ready_backlog_stale):
            queue_stale_wave_auto_remediation["status"] = "remediation_degraded"
            queue_stale_wave_auto_remediation["reason"] = "stale_wave_persisted_after_queue_sync"
            queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
            queue_stale_wave_auto_remediation["recovered"] = False
        elif not queue_wave_progress_detected:
            queue_stale_wave_auto_remediation["status"] = "remediation_degraded"
            queue_stale_wave_auto_remediation["reason"] = "stale_wave_cleared_without_queue_progress"
            queue_stale_wave_auto_remediation["last_failure_at"] = current_eval_iso
            queue_stale_wave_auto_remediation["recovered"] = False
            queue_stale_wave_signal["active"] = True
            queue_stale_wave_signal["reason"] = "ready_backlog_idle_no_progress_after_queue_sync"
            queue_stale_wave_auto_remediation["queue_stale_wave_active_after"] = True
        else:
            queue_stale_wave_auto_remediation["status"] = "remediated"
            queue_stale_wave_auto_remediation["reason"] = "stale_wave_cleared_after_queue_sync"
            queue_stale_wave_auto_remediation["last_success_at"] = current_eval_iso
            queue_stale_wave_auto_remediation["recovered"] = True
    elif attempted_status in {"command_failed", "payload_not_ok", "remediator_missing"}:
        queue_stale_wave_auto_remediation["recovered"] = False
    queue_stale_wave_auto_remediation["last_attempt_status"] = str(queue_stale_wave_auto_remediation.get("status") or "") or None

queue_wave_current_status = str(queue_stale_wave_auto_remediation.get("status") or "").strip()
queue_wave_current_reason = str(queue_stale_wave_auto_remediation.get("reason") or "").strip()
queue_wave_failure_now = queue_wave_current_status in QUEUE_STALE_WAVE_AUTO_FAILURE_STATUSES
queue_wave_last_attempt_status = str(queue_stale_wave_auto_remediation.get("last_attempt_status") or "").strip()
queue_wave_last_attempt_failed = queue_wave_last_attempt_status in QUEUE_STALE_WAVE_AUTO_FAILURE_STATUSES

if queue_stale_wave_auto_remediation.get("attempted"):
    queue_stale_wave_auto_remediation["attempt_sequence"] = int(prior_queue_wave_attempt_sequence) + 1
else:
    queue_stale_wave_auto_remediation["attempt_sequence"] = int(prior_queue_wave_attempt_sequence)

if queue_stale_wave_auto_remediation.get("attempted") and queue_wave_failure_now:
    queue_stale_wave_auto_remediation["consecutive_failures"] = int(prior_queue_wave_consecutive_failures) + 1
elif queue_stale_wave_auto_remediation.get("attempted"):
    queue_stale_wave_auto_remediation["consecutive_failures"] = 0
elif queue_wave_last_attempt_failed and queue_wave_current_status == "cooldown_active":
    queue_stale_wave_auto_remediation["consecutive_failures"] = max(1, int(prior_queue_wave_consecutive_failures))
elif not bool(ready_backlog_stale):
    queue_stale_wave_auto_remediation["consecutive_failures"] = 0
else:
    queue_stale_wave_auto_remediation["consecutive_failures"] = int(prior_queue_wave_consecutive_failures)

queue_wave_failure_evidence = queue_stale_wave_auto_remediation.get("failure_evidence")
if not isinstance(queue_wave_failure_evidence, dict):
    queue_wave_failure_evidence = {}

if queue_stale_wave_auto_remediation.get("attempted") and queue_wave_failure_now:
    queue_wave_failure_meta = derive_queue_stale_wave_failure_details(queue_wave_current_status, queue_wave_current_reason)
    queue_wave_failure_evidence = {
        "schema_version": "continuity.queue_stale_wave_auto_remediation_failure_evidence.v1",
        "present": True,
        "captured_at": current_eval_iso,
        "attempt_at": queue_stale_wave_auto_remediation.get("last_attempt_at") or current_eval_iso,
        "attempt_sequence": int(queue_stale_wave_auto_remediation.get("attempt_sequence") or 0),
        "status": queue_wave_current_status,
        "reason": queue_wave_current_reason,
        "category": queue_wave_failure_meta.get("category"),
        "code": queue_wave_failure_meta.get("code"),
        "retryable": bool(queue_wave_failure_meta.get("retryable") is not False),
        "command": queue_stale_wave_auto_remediation.get("command"),
        "command_rc": queue_stale_wave_auto_remediation.get("command_rc"),
        "command_ok": queue_stale_wave_auto_remediation.get("command_ok"),
        "payload_ok": queue_stale_wave_auto_remediation.get("payload_ok"),
        "payload_parse_error": queue_stale_wave_auto_remediation.get("payload_parse_error"),
        "stdout_tail": queue_stale_wave_auto_remediation.get("stdout_tail"),
        "stderr_tail": queue_stale_wave_auto_remediation.get("stderr_tail"),
        "queue_stale_wave_active_before": bool(queue_stale_wave_auto_remediation.get("queue_stale_wave_active_before")),
        "queue_stale_wave_active_after": bool(queue_stale_wave_auto_remediation.get("queue_stale_wave_active_after")),
        "ready_count_before": int(queue_stale_wave_auto_remediation.get("ready_count_before") or 0),
        "ready_count_after": int(queue_stale_wave_auto_remediation.get("ready_count_after") or 0),
        "ready_oldest_age_sec_before": queue_stale_wave_auto_remediation.get("ready_oldest_age_sec_before"),
        "ready_oldest_age_sec_after": queue_stale_wave_auto_remediation.get("ready_oldest_age_sec_after"),
        "in_flight_effective_before": bool(queue_stale_wave_auto_remediation.get("in_flight_effective_before")),
        "in_flight_effective_after": bool(queue_stale_wave_auto_remediation.get("in_flight_effective_after")),
        "progress_detected": bool(queue_wave_progress_detected),
        "progress_reasons": list(queue_wave_progress_reasons),
        "next_retry_after_ts": int(queue_wave_next_attempt_after_ts or 0),
        "next_retry_after_iso": queue_stale_wave_auto_remediation.get("next_attempt_after_iso"),
        "retry_cooldown_sec": int(queue_stale_wave_auto_remediate_cooldown_sec),
    }
elif queue_wave_last_attempt_failed and queue_wave_current_status == "cooldown_active" and bool(queue_stale_wave_signal.get("active")):
    queue_wave_failure_evidence = dict(queue_wave_failure_evidence)
    queue_wave_failure_evidence.setdefault("schema_version", "continuity.queue_stale_wave_auto_remediation_failure_evidence.v1")
    queue_wave_failure_evidence["present"] = True
    queue_wave_failure_evidence.setdefault("captured_at", queue_stale_wave_auto_remediation.get("last_failure_at") or current_eval_iso)
    queue_wave_failure_evidence.setdefault("attempt_at", queue_stale_wave_auto_remediation.get("last_attempt_at"))
    queue_wave_failure_evidence.setdefault("attempt_sequence", int(queue_stale_wave_auto_remediation.get("attempt_sequence") or 0))
    if not str(queue_wave_failure_evidence.get("status") or "").strip():
        queue_wave_failure_evidence["status"] = queue_wave_last_attempt_status
    if not str(queue_wave_failure_evidence.get("reason") or "").strip():
        queue_wave_failure_evidence["reason"] = prior_queue_wave_reason or queue_wave_current_reason
    queue_wave_failure_meta = derive_queue_stale_wave_failure_details(
        queue_wave_failure_evidence.get("status"),
        queue_wave_failure_evidence.get("reason"),
    )
    queue_wave_failure_evidence.setdefault("category", queue_wave_failure_meta.get("category"))
    queue_wave_failure_evidence.setdefault("code", queue_wave_failure_meta.get("code"))
    queue_wave_failure_evidence.setdefault("retryable", bool(queue_wave_failure_meta.get("retryable") is not False))
    queue_wave_failure_evidence["next_retry_after_ts"] = int(queue_wave_next_attempt_after_ts or 0)
    queue_wave_failure_evidence["next_retry_after_iso"] = queue_stale_wave_auto_remediation.get("next_attempt_after_iso")
    queue_wave_failure_evidence["retry_cooldown_sec"] = int(queue_stale_wave_auto_remediate_cooldown_sec)
else:
    queue_wave_failure_evidence = dict(queue_wave_failure_evidence)
    if queue_wave_failure_evidence:
        queue_wave_failure_evidence.setdefault("schema_version", "continuity.queue_stale_wave_auto_remediation_failure_evidence.v1")
        queue_wave_failure_evidence["present"] = False
        queue_wave_failure_evidence["next_retry_after_ts"] = 0
        queue_wave_failure_evidence["next_retry_after_iso"] = None

queue_stale_wave_auto_remediation["failure_evidence"] = queue_wave_failure_evidence
queue_stale_wave_auto_remediation["failure_taxonomy_version"] = "continuity.queue_stale_wave_auto_remediation_failure.v1"
if isinstance(queue_wave_failure_evidence, dict) and bool(queue_wave_failure_evidence.get("present")):
    queue_stale_wave_auto_remediation["failure_category"] = queue_wave_failure_evidence.get("category")
    queue_stale_wave_auto_remediation["failure_code"] = queue_wave_failure_evidence.get("code")
    queue_stale_wave_auto_remediation["failure_retryable"] = queue_wave_failure_evidence.get("retryable")
else:
    queue_stale_wave_auto_remediation["failure_category"] = None
    queue_stale_wave_auto_remediation["failure_code"] = None
    queue_stale_wave_auto_remediation["failure_retryable"] = None

queue_wave_retry_scheduled = bool(
    queue_wave_remediate_eligible
    and queue_wave_next_attempt_after_ts
    and int(queue_wave_next_attempt_after_ts) > int(current_eval_ts)
)
queue_wave_retry_due = bool(queue_wave_remediate_eligible and not queue_wave_retry_scheduled)
queue_wave_retry_blocked_reason = None
if not queue_stale_wave_auto_remediate_enabled:
    queue_wave_retry_blocked_reason = "disabled_by_env"
elif not queue_wave_remediate_eligible:
    queue_wave_retry_blocked_reason = str(ready_backlog_stale_reason_before)
elif queue_wave_retry_scheduled:
    queue_wave_retry_blocked_reason = "cooldown_active"

queue_stale_wave_auto_remediation["retry_contract"] = {
    "schema_version": "continuity.queue_stale_wave_auto_remediation_retry_contract.v1",
    "policy": "fixed_cooldown",
    "deterministic_key": "queue_stale_wave_auto_remediation",
    "eligible_now": bool(queue_wave_remediate_eligible),
    "retry_scheduled": bool(queue_wave_retry_scheduled),
    "retry_due": bool(queue_wave_retry_due),
    "retry_blocked_reason": queue_wave_retry_blocked_reason,
    "cooldown_sec": int(queue_stale_wave_auto_remediate_cooldown_sec),
    "cooldown_remaining_sec": int(queue_stale_wave_auto_remediation.get("cooldown_remaining_sec") or 0),
    "next_attempt_after_ts": int(queue_wave_next_attempt_after_ts or 0),
    "next_attempt_after_iso": queue_stale_wave_auto_remediation.get("next_attempt_after_iso"),
}

atomic_write(queue_stale_wave_auto_remediation_state_path, queue_stale_wave_auto_remediation)

queue_stale_wave_auto_remediation_contract_issues: List[str] = []
queue_stale_wave_auto_remediation_contract_state = load_json(queue_stale_wave_auto_remediation_state_path)
queue_stale_wave_contract_expected_schema = "continuity.queue_stale_wave_auto_remediation.v1"
queue_stale_wave_projection_present = bool(
    isinstance(queue_stale_wave_auto_remediation, dict) and queue_stale_wave_auto_remediation
)
queue_stale_wave_projection_schema_version = str(
    ((queue_stale_wave_auto_remediation or {}).get("schema_version") if isinstance(queue_stale_wave_auto_remediation, dict) else "") or ""
).strip()
queue_stale_wave_projection_state_path = str(
    ((queue_stale_wave_auto_remediation or {}).get("state_path") if isinstance(queue_stale_wave_auto_remediation, dict) else "") or ""
).strip()

queue_stale_wave_state_exists = queue_stale_wave_auto_remediation_state_path.exists()
queue_stale_wave_state_valid = bool(
    isinstance(queue_stale_wave_auto_remediation_contract_state, dict)
    and queue_stale_wave_auto_remediation_contract_state
)
queue_stale_wave_state_schema_version = str(
    ((queue_stale_wave_auto_remediation_contract_state or {}).get("schema_version") if isinstance(queue_stale_wave_auto_remediation_contract_state, dict) else "") or ""
).strip()
queue_stale_wave_state_state_path = str(
    ((queue_stale_wave_auto_remediation_contract_state or {}).get("state_path") if isinstance(queue_stale_wave_auto_remediation_contract_state, dict) else "") or ""
).strip()

if not queue_stale_wave_projection_present:
    queue_stale_wave_auto_remediation_contract_issues.append("projection_missing")
if queue_stale_wave_projection_schema_version != queue_stale_wave_contract_expected_schema:
    queue_stale_wave_auto_remediation_contract_issues.append("projection_schema_invalid")
if queue_stale_wave_projection_state_path != queue_stale_wave_auto_remediation_state_rel:
    queue_stale_wave_auto_remediation_contract_issues.append("projection_state_path_mismatch")

if not queue_stale_wave_state_exists:
    queue_stale_wave_auto_remediation_contract_issues.append("state_missing")
elif not queue_stale_wave_state_valid:
    queue_stale_wave_auto_remediation_contract_issues.append("state_invalid")

if queue_stale_wave_state_valid:
    if queue_stale_wave_state_schema_version != queue_stale_wave_contract_expected_schema:
        queue_stale_wave_auto_remediation_contract_issues.append("state_schema_invalid")
    if queue_stale_wave_state_state_path != queue_stale_wave_auto_remediation_state_rel:
        queue_stale_wave_auto_remediation_contract_issues.append("state_state_path_mismatch")

queue_stale_wave_contract_schema_path = maybe_rel(queue_stale_wave_auto_remediation_schema_path)
queue_stale_wave_projection_schema_valid: Optional[bool] = None
queue_stale_wave_state_schema_valid: Optional[bool] = None
queue_stale_wave_schema_validation_errors: List[str] = []

if not callable(validate_contract_payload_schema):
    queue_stale_wave_auto_remediation_contract_issues.append("schema_validator_unavailable")
    queue_stale_wave_schema_validation_errors.append("queue_stale_wave_auto_remediation_schema_validator_unavailable")
else:
    if queue_stale_wave_projection_present and isinstance(queue_stale_wave_auto_remediation, dict):
        try:
            validate_contract_payload_schema(
                queue_stale_wave_auto_remediation,
                schema_path=queue_stale_wave_auto_remediation_schema_path,
                contract_prefix="queue_stale_wave_auto_remediation_projection_contract",
            )
            queue_stale_wave_projection_schema_valid = True
        except Exception as exc:
            queue_stale_wave_projection_schema_valid = False
            queue_stale_wave_auto_remediation_contract_issues.append("projection_schema_contract_invalid")
            queue_stale_wave_schema_validation_errors.append(str(exc))

    if queue_stale_wave_state_valid and isinstance(queue_stale_wave_auto_remediation_contract_state, dict):
        try:
            validate_contract_payload_schema(
                queue_stale_wave_auto_remediation_contract_state,
                schema_path=queue_stale_wave_auto_remediation_schema_path,
                contract_prefix="queue_stale_wave_auto_remediation_state_contract",
            )
            queue_stale_wave_state_schema_valid = True
        except Exception as exc:
            queue_stale_wave_state_schema_valid = False
            queue_stale_wave_auto_remediation_contract_issues.append("state_schema_contract_invalid")
            queue_stale_wave_schema_validation_errors.append(str(exc))

queue_stale_wave_projection_state_match: Optional[bool] = None
if queue_stale_wave_projection_present and queue_stale_wave_state_valid:
    try:
        projection_payload_canonical = json.dumps(
            queue_stale_wave_auto_remediation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        state_payload_canonical = json.dumps(
            queue_stale_wave_auto_remediation_contract_state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        queue_stale_wave_projection_state_match = projection_payload_canonical == state_payload_canonical
    except Exception:
        queue_stale_wave_projection_state_match = None
        queue_stale_wave_auto_remediation_contract_issues.append("projection_state_compare_failed")

if queue_stale_wave_projection_state_match is False:
    queue_stale_wave_auto_remediation_contract_issues.append("projection_state_mismatch")

if queue_stale_wave_auto_remediation_contract_issues:
    queue_stale_wave_auto_remediation_contract_issues = sorted(dict.fromkeys(queue_stale_wave_auto_remediation_contract_issues))
if queue_stale_wave_schema_validation_errors:
    queue_stale_wave_schema_validation_errors = list(dict.fromkeys(queue_stale_wave_schema_validation_errors))

queue_stale_wave_auto_remediation_contract_healthy = len(queue_stale_wave_auto_remediation_contract_issues) == 0
queue_stale_wave_auto_remediation_contract = {
    "schema_version": "continuity.queue_stale_wave_auto_remediation_contract.v1",
    "status": "ok" if queue_stale_wave_auto_remediation_contract_healthy else "degraded",
    "healthy": queue_stale_wave_auto_remediation_contract_healthy,
    "checked_at": clock_now_iso(),
    "projection_path": "queue.stale_wave_auto_remediation",
    "state_path": queue_stale_wave_auto_remediation_state_rel,
    "contract_schema_path": queue_stale_wave_contract_schema_path,
    "projection_present": queue_stale_wave_projection_present,
    "projection_schema_version": queue_stale_wave_projection_schema_version or None,
    "projection_state_path": queue_stale_wave_projection_state_path or None,
    "state_exists": queue_stale_wave_state_exists,
    "state_valid": queue_stale_wave_state_valid,
    "state_schema_version": queue_stale_wave_state_schema_version or None,
    "state_state_path": queue_stale_wave_state_state_path or None,
    "projection_schema_valid": queue_stale_wave_projection_schema_valid,
    "state_schema_valid": queue_stale_wave_state_schema_valid,
    "schema_validation_errors": queue_stale_wave_schema_validation_errors,
    "projection_state_match": queue_stale_wave_projection_state_match,
    "issues": queue_stale_wave_auto_remediation_contract_issues,
}

def load_login_contract_actionability(contract_ref: Any) -> Dict[str, Any]:
    rel = str(contract_ref or "").strip()
    if not rel:
        return {}
    path = pathlib.Path(rel)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    obj = load_json(path)
    if not obj:
        return {}

    incident = obj.get("incident_actionability") if isinstance(obj.get("incident_actionability"), dict) else {}
    commands = [
        normalize_operator_command(cmd)
        for cmd in (incident.get("recommended_commands") if isinstance(incident.get("recommended_commands"), list) else [])
        if isinstance(cmd, str) and cmd.strip()
    ]
    recommended_steps = []
    for step in (incident.get("recommended_steps") if isinstance(incident.get("recommended_steps"), list) else []):
        if not isinstance(step, dict):
            continue
        cmd = normalize_operator_command(step.get("command") or "")
        if not cmd:
            continue
        step_id = str(step.get("step_id") or step.get("id") or "").strip() or f"step_{len(recommended_steps) + 1}"
        summary = str(step.get("summary") or step.get("intent") or "").strip() or None
        recommended_steps.append(
            {
                "step_id": step_id,
                "summary": summary,
                "command": cmd,
            }
        )
    if not commands and recommended_steps:
        commands = [str(step.get("command") or "") for step in recommended_steps if str(step.get("command") or "").strip()]
    resume_command = normalize_operator_command(obj.get("resume_command") or "")
    operator_contract_md = obj.get("operator_contract_md")

    return {
        "contract_status": obj.get("status"),
        "operator_contract_md": operator_contract_md,
        "operator_resume_command": resume_command or None,
        "incident_actionability": {
            "incident_id": incident.get("incident_id"),
            "reason": incident.get("reason"),
            "severity": incident.get("severity"),
            "status": incident.get("status"),
            "action_required": bool(incident.get("action_required")),
            "recommended_commands": commands,
            "recommended_steps": recommended_steps,
            "evidence": incident.get("evidence") if isinstance(incident.get("evidence"), list) else [],
        }
        if incident
        else {},
    }


for p in sorted((latest_dir).glob("web_capture_domain_*.json")):
    obj = load_json(p)
    if not obj:
        continue

    cooldown_until_raw = obj.get("cooldown_until")
    cooldown_dt = parse_iso(cooldown_until_raw)
    cooldown_remaining = None
    if cooldown_dt is not None:
        if cooldown_dt.tzinfo is None:
            cooldown_dt = cooldown_dt.replace(tzinfo=dt.timezone.utc)
        cooldown_remaining = max(0, int((cooldown_dt - clock_now_dt()).total_seconds()))

    operator_action_required = bool(obj.get("operator_action_required"))
    contract_actionability = load_login_contract_actionability(obj.get("operator_contract_json")) if operator_action_required else {}
    incident_actionability = contract_actionability.get("incident_actionability") if isinstance(contract_actionability.get("incident_actionability"), dict) else {}
    incident_actionable = bool(operator_action_required and incident_actionability.get("action_required") and (incident_actionability.get("recommended_commands") or []))

    status = str(obj.get("last_status") or "unknown").strip()
    if status in {"blocked", "failed"}:
        web_capture_summary["blocked_domains"] += 1
    if cooldown_remaining is not None and cooldown_remaining > 0:
        web_capture_summary["cooldown_active_domains"] += 1
    if operator_action_required:
        web_capture_summary["operator_action_required_domains"] += 1
    if incident_actionable:
        web_capture_summary["actionable_incident_domains"] += 1

    updated_at = str(obj.get("updated_at") or "").strip() or None
    if updated_at:
        if web_capture_summary["latest_updated_at"] is None or updated_at > str(web_capture_summary["latest_updated_at"]):
            web_capture_summary["latest_updated_at"] = updated_at

    entry = {
        "domain": str(obj.get("domain") or p.stem.replace("web_capture_domain_", "")).strip(),
        "macro_slug": obj.get("macro_slug"),
        "status": status,
        "gate_class": obj.get("last_gate_class"),
        "cooldown_until": cooldown_until_raw,
        "cooldown_remaining_sec": cooldown_remaining,
        "operator_action_required": operator_action_required,
        "operator_contract_json": obj.get("operator_contract_json"),
        "operator_contract_md": contract_actionability.get("operator_contract_md") or obj.get("operator_contract_md"),
        "operator_resume_command": contract_actionability.get("operator_resume_command"),
        "incident_actionability": incident_actionability,
        "state_path": maybe_rel(p),
        "updated_at": updated_at,
    }
    web_capture_summary["domains"].append(entry)

web_capture_summary["tracked_domains"] = len(web_capture_summary["domains"])
web_capture_summary["domains"] = sorted(
    web_capture_summary["domains"],
    key=lambda x: (
        0 if x.get("operator_action_required") else 1,
        0 if x.get("status") in {"blocked", "failed"} else 1,
        str(x.get("domain") or ""),
    ),
)

scheduler_freshness_limit_sec = _read_nonnegative_int_env(
    "OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC",
    default=_DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
)

scheduler_state_path = root / "state" / "continuity" / "latest" / "web_capture_scheduler_state.json"
scheduler_state_exists = scheduler_state_path.exists()
scheduler_state = load_json(scheduler_state_path)
scheduler_summary = web_capture_summary.get("scheduler") if isinstance(web_capture_summary.get("scheduler"), dict) else {}
scheduler_summary.update(
    {
        "state_exists": scheduler_state_exists,
        "freshness_limit_sec": scheduler_freshness_limit_sec,
    }
)
if scheduler_state:
    scheduler_payload = scheduler_state.get("summary") if isinstance(scheduler_state.get("summary"), dict) else {}
    scheduler_contract = scheduler_state.get("contract") if isinstance(scheduler_state.get("contract"), dict) else {}
    scheduler_errors = scheduler_contract.get("validation_errors") if isinstance(scheduler_contract.get("validation_errors"), list) else []
    scheduler_updated_at = scheduler_state.get("updated_at")
    scheduler_updated_age_sec = age_sec(scheduler_updated_at)
    scheduler_summary.update(
        {
            "schema_version": scheduler_state.get("schema_version"),
            "selection_status": scheduler_state.get("selection_status"),
            "updated_at": scheduler_updated_at,
            "state_age_sec": scheduler_updated_age_sec,
            "fresh": scheduler_updated_age_sec is not None and scheduler_updated_age_sec <= scheduler_freshness_limit_sec,
            "contract_schema_path": scheduler_contract.get("schema_path"),
            "contract_state_valid": scheduler_contract.get("state_valid"),
            "contract_validation_errors": scheduler_errors,
            "contract_previous_state_valid": scheduler_contract.get("previous_state_valid"),
            "eligible_macros": scheduler_payload.get("eligible_macros"),
            "total_macros": scheduler_payload.get("total_macros"),
            "last_selected_domain": scheduler_state.get("last_selected_domain"),
            "last_selected_macro_slug": scheduler_state.get("last_selected_macro_slug"),
        }
    )
elif scheduler_state_exists:
    scheduler_summary.update(
        {
            "contract_state_valid": False,
            "contract_validation_errors": ["state_unreadable_or_not_object"],
        }
    )
web_capture_summary["scheduler"] = scheduler_summary

gtc_gateboard = load_json(root / "state" / "gtc-v2" / "latest" / "gateboard.json")
gtc_incident_replay = load_json(root / "state" / "gtc-v2" / "latest" / "incident_replay.json")
if gtc_gateboard:
    gtc_summary = {
        "enabled": True,
        "mutate_allowed": gtc_gateboard.get("mutate_allowed"),
        "status": gtc_gateboard.get("status"),
        "blocking_reasons": list(gtc_gateboard.get("blocking_reasons") or []),
        "warning_reasons": list(gtc_gateboard.get("warning_reasons") or []),
        "latest_path": "state/gtc-v2/latest/gateboard.json",
        "generated_at": gtc_gateboard.get("generated_at"),
        "open_incident_count": gtc_gateboard.get("open_incident_count"),
        "verify_status": gtc_gateboard.get("verify_status"),
        "incident_replay_path": "state/gtc-v2/latest/incident_replay.json",
        "incident_replay_commands": list(gtc_incident_replay.get("recommended_commands") or []),
    }

coherence_stamp: Dict[str, Any] = {}
if build_coherence_tuple is not None:
    try:
        coherence_stamp = build_coherence_tuple(root, update_policy_epoch=True)
    except Exception as exc:
        coherence_stamp = {
            "schema_version": "continuity.coherence_stamp.v1",
            "generated_at": clock_now_iso(),
            "error": f"coherence_tuple_failed:{exc}",
        }

try:
    reconcile_min_interval_sec = max(0, int(os.environ.get("OPENCLAW_CONTINUITY_RECONCILE_MIN_INTERVAL_SEC", "1800")))
except Exception:
    reconcile_min_interval_sec = 1800

checkpoint_trigger = str(checkpoint_meta.get("trigger") or "")
ground_truth_drift_cooldown_active = bool(
    gt_capture_ok is False
    and checkpoint_trigger == "drift_reconcile"
    and checkpoint_age is not None
    and checkpoint_age < reconcile_min_interval_sec
)
checkpoint_created_ts = iso_to_ts(checkpoint_created)
gt_snapshot_ts = iso_to_ts(gt_ts)
ground_truth_capture_drift_cooldown_policy_lag = bool(
    ground_truth_drift_cooldown_active
    and checkpoint_created_ts is not None
    and gt_snapshot_ts is not None
    and gt_snapshot_ts > checkpoint_created_ts
)

generated_at_dt = clock_now_dt().replace(microsecond=0)
generated_at_iso = generated_at_dt.isoformat().replace("+00:00", "Z")
coherence_valid_until_iso = (generated_at_dt + dt.timedelta(seconds=coherence_hard_ttl_sec)).isoformat().replace("+00:00", "Z")
build_generation_id = f"cohgen_{uuid.uuid4().hex[:16]}"

summary = {
    "schema_version": "continuity.now.v2",
    "generated_at": generated_at_iso,
    "checkpoint": {
        "id": checkpoint_id,
        "status": checkpoint_status,
        "trigger": checkpoint_meta.get("trigger"),
        "created_at": checkpoint_created or None,
        "age_sec": checkpoint_age,
        "objective": checkpoint_obj.get("primary_goal"),
        "blocker_reason": checkpoint_obj.get("blocker_reason"),
        "path": checkpoint_path or None,
    },
    "verify": {
        "status": verify_status,
        "reason": verify_reason or None,
        "timestamp": verify_ts or None,
        "age_sec": verify_age,
        "path": "state/continuity/latest/verify_last.json",
        "strict_autonomy_regressions": {
            "enabled": verify_strict_enabled,
            "source": verify_strict_source,
            "effective_source": verify_strict_effective_source,
            "result_ok": verify_strict_result_ok,
            "wrapper_effective": verify_strict_wrapper_effective,
            "wrapper_hint_mismatch": verify_strict_wrapper_hint_mismatch,
        },
        "gate_preflight": verify_gate_preflight,
    },
    "ground_truth": {
        "snapshot_id": gt_snapshot_id,
        "snapshot_ts_utc": gt_ts or None,
        "age_sec": gt_age,
        "snapshot_path": gt_snapshot_path or None,
        "anomaly_count": len(anomalies),
        "critical_keys": critical_keys,
        "warn_keys": warn_keys,
    },
    "bridge": {
        "path": "state/continuity/latest/runtime_truth_bridge.json",
        "updated_at": bridge.get("updated_at"),
        "stale_vs_live": bridge_stale,
        "pointer_matches_checkpoint": pointer_ok,
        "pointer_sha_match": pointer_sha_ok,
        "env_matches_checkpoint_capture": env_capture_ok,
        "ground_truth_matches_checkpoint_capture": gt_capture_ok,
        "reported_pointer_matches_checkpoint": bridge_pointer_reported,
        "reported_pointer_sha_match": bridge_pointer_sha_reported,
        "reported_env_matches_checkpoint_capture": bridge_env_capture_reported,
        "reported_ground_truth_matches_checkpoint_capture": bridge_gt_capture_reported,
    },
    "autopilot": {
        "paused": ap_state.get("paused") if isinstance(ap_state, dict) else None,
        "cycle": ap_state.get("cycle") if isinstance(ap_state, dict) else None,
        "max_cycles": ap_state.get("max_cycles") if isinstance(ap_state, dict) else None,
        "active_step": active_step,
        "active_evidence_refs": active_evidence_refs,
        "recent_evidence_refs": autopilot_recent_evidence_refs,
        "step_status_counts": step_counts,
        "degraded_pending_stale_signal": autopilot_degraded_pending_signal,
        "stale_task_recovery_counters": autopilot_stale_task_recovery_counters,
        "idle_lane_autospawn": idle_lane_autospawn,
        "idle_lane_autospawn_contract": idle_lane_autospawn_contract,
        "execution_frontier_controller": execution_frontier_controller,
        "execution_frontier_controller_contract": execution_frontier_controller_contract,
    },
    "queue": {
        "status_counts": queue_counts,
        "role_required_counts": queue_role_required_counts,
        "role_required_unset_count": queue_role_missing_count,
        "review_role_mismatch_count": queue_review_role_mismatch_count,
        "ready_count": queue_ready_count,
        "dependency_blocked_count": queue_dependency_blocked_count,
        "dependency_blocked_examples": queue_dependency_blocked_examples,
        "active_file_lock_count": active_file_lock_count,
        "stale_active_file_lock_count": stale_active_file_lock_count,
        "effective_active_file_lock_count": effective_active_file_lock_count,
        "orphaned_running_without_locks_count": orphaned_running_without_locks_count,
        "orphaned_running_without_locks_examples": orphaned_running_without_locks_examples,
        "orphaned_running_min_sec": orphaned_running_min_sec,
        "effective_running_count": effective_running_count,
        "in_flight_effective": in_flight_effective,
        "orphaned_running_auto_remediation": orphaned_running_auto_remediation,
        "orphaned_running_auto_remediation_contract": orphaned_running_auto_remediation_contract,
        "stale_wave_auto_remediation": queue_stale_wave_auto_remediation,
        "stale_wave_auto_remediation_contract": queue_stale_wave_auto_remediation_contract,
        "stale_task_recovery_counters": autopilot_stale_task_recovery_counters,
        "stale_active_lock_examples": active_lock_examples,
        "last_transition_at": transition_last_at,
        "last_transition_event": transition_last_event,
        "transition_history_24h": transition_history_24h,
        "handoff_history_24h": handoff_history_24h,
        "latest_handoff_packet": latest_handoff_packet,
        "stale_wave_signal": queue_stale_wave_signal,
        "recent_evidence_refs": recent_evidence_refs,
    },
    "parity": parity_summary,
    "web_capture": web_capture_summary,
    "gtc": gtc_summary,
    "reconcile_history": reconcile_history,
    "refresh_hooks": refresh_hooks,
    "refresh_preflight": refresh_preflight,
    "refresh_storm_guard": refresh_storm_guard,
    "reset_ready_refresh": {
        "path": reset_ready_refresh_latest_rel,
        "present": reset_ready_refresh_present,
        "ok": reset_ready_refresh_ok,
        "phase": reset_ready_refresh_phase,
        "error_code": reset_ready_refresh_error_code,
        "generated_at": reset_ready_refresh_projection.get("generated_at"),
        "freshness_limit_sec": reset_ready_refresh_freshness_limit_sec,
        "age_sec": reset_ready_refresh_age_sec,
        "fresh": reset_ready_refresh_fresh,
        "stale": reset_ready_refresh_stale,
        "partial_refresh": {
            "current_refreshed": reset_ready_refresh_partial_current,
            "proof_refreshed": reset_ready_refresh_partial_proof,
            "handover_refreshed": reset_ready_refresh_partial_handover,
        },
        "degraded": reset_ready_refresh_degraded,
        "partial_failure": reset_ready_refresh_partial_failure,
    },
    "coherence": coherence_stamp,
}

summary.setdefault("coherence", {})
summary["coherence"]["build_generation_id"] = build_generation_id
summary["coherence"]["published_at"] = generated_at_iso
summary["coherence"]["valid_until"] = coherence_valid_until_iso
summary["coherence"]["hard_ttl_sec"] = coherence_hard_ttl_sec

if isinstance(coherence_stamp, dict) and coherence_stamp:
    coherence_stamp.setdefault("publish", {})
    coherence_stamp["publish"]["build_generation_id"] = build_generation_id
    coherence_stamp["publish"]["published_at"] = generated_at_iso
    coherence_stamp["publish"]["valid_until"] = coherence_valid_until_iso
    coherence_stamp["publish"]["hard_ttl_sec"] = coherence_hard_ttl_sec

not_ready_reasons = []
verify_then_resume_active = str(os.environ.get("OPENCLAW_VERIFY_THEN_RESUME_ACTIVE", "0")).strip() == "1"
layered_health_snapshot_active = str(
    os.environ.get("OPENCLAW_LAYERED_HEALTH_SNAPSHOT_ACTIVE", "0")
).strip() == "1"
gtc_blocking_reasons = [
    str(x).strip() for x in (gtc_summary.get("blocking_reasons") or []) if str(x).strip()
]
gtc_verify_status_blocking_only = bool(gtc_blocking_reasons) and all(
    reason.startswith("verify_status_not_ready:") for reason in gtc_blocking_reasons
)
gtc_gateboard_verify_status = str(gtc_summary.get("verify_status") or "").strip()
gtc_generated_ts = iso_to_ts(gtc_summary.get("generated_at"))
verify_report_ts = iso_to_ts(verify_ts)
gtc_gateboard_blocked_suppression_reason = None
if (
    gtc_summary.get("enabled")
    and gtc_summary.get("mutate_allowed") is False
    and gtc_verify_status_blocking_only
):
    if verify_status == "BLOCKER":
        gtc_gateboard_blocked_suppression_reason = "gtc_verify_status_blocking_only"
    elif (
        verify_status == "READY"
        and gtc_gateboard_verify_status == "BLOCKER"
        and verify_report_ts is not None
        and gtc_generated_ts is not None
        and verify_report_ts > gtc_generated_ts
    ):
        gtc_gateboard_blocked_suppression_reason = "gtc_verify_status_stale_after_verify_ready"

gtc_gateboard_blocked_suppressed = gtc_gateboard_blocked_suppression_reason is not None
if gtc_gateboard_blocked_suppression_reason == "gtc_verify_status_stale_after_verify_ready":
    gtc_warning_reasons_for_projection = [
        str(x).strip() for x in (gtc_summary.get("warning_reasons") or []) if str(x).strip()
    ]
    gtc_summary["verify_status_raw"] = gtc_gateboard_verify_status or None
    gtc_summary["blocking_reasons_raw"] = list(gtc_blocking_reasons)
    gtc_summary["mutate_allowed_raw"] = gtc_summary.get("mutate_allowed")
    gtc_summary["status_raw"] = gtc_summary.get("status")
    gtc_summary["verify_status"] = verify_status or "READY"
    gtc_summary["blocking_reasons"] = [
        reason for reason in gtc_blocking_reasons if not reason.startswith("verify_status_not_ready:")
    ]
    if not (gtc_summary.get("blocking_reasons") or []):
        gtc_summary["mutate_allowed"] = True
        gtc_summary["status"] = "yellow" if gtc_warning_reasons_for_projection else "green"
    gtc_summary["verify_status_projection"] = "stale_gateboard_reclassified_from_verify_ready"

if checkpoint_status == "BLOCKER":
    not_ready_reasons.append("checkpoint_blocker")
if verify_status == "BLOCKER" and not verify_then_resume_active:
    not_ready_reasons.append("verify_blocker")
if critical_keys:
    not_ready_reasons.append("critical_anomalies")
if pointer_ok is False or pointer_sha_ok is False:
    not_ready_reasons.append("pointer_drift")
if gt_capture_ok is False and not ground_truth_drift_cooldown_active:
    not_ready_reasons.append("ground_truth_capture_drift")
if (
    gtc_summary.get("enabled")
    and gtc_summary.get("mutate_allowed") is False
    and not verify_then_resume_active
    and not gtc_gateboard_blocked_suppressed
):
    not_ready_reasons.append("gtc_gateboard_blocked")
if refresh_preflight_failed:
    not_ready_reasons.append("refresh_preflight_failed")

reset_ready_refresh_blocker_reason = _project_reset_ready_refresh_escalation_reason(
    posture=reset_ready_refresh_projection,
    degraded=reset_ready_refresh_degraded,
    phase=reset_ready_refresh_phase,
    error_code=reset_ready_refresh_error_code,
)
if reset_ready_refresh_blocker_reason:
    not_ready_reasons.append(str(reset_ready_refresh_blocker_reason))

coherence_policy = (coherence_stamp.get("policy") or {}) if isinstance(coherence_stamp, dict) else {}
verify_policy = ((verify_last.get("freshness") or {}).get("policy") or {}) if isinstance(verify_last, dict) else {}
live_policy_sig = str(coherence_policy.get("signature") or "").strip()
verify_policy_sig = str(verify_policy.get("signature") or "").strip()
if verify_status == "READY":
    if not verify_policy_sig:
        not_ready_reasons.append("policy_freshness_unverified")
    elif live_policy_sig and verify_policy_sig != live_policy_sig:
        not_ready_reasons.append("policy_freshness_drift")

verify_gate_preflight_status_evidence = (
    verify_gate_preflight.get("status_evidence_gate")
    if isinstance(verify_gate_preflight.get("status_evidence_gate"), dict)
    else {}
)
verify_gate_preflight_layered_health = (
    verify_gate_preflight.get("layered_health_gate")
    if isinstance(verify_gate_preflight.get("layered_health_gate"), dict)
    else {}
)
verify_gate_preflight_failover_stress_runtime = (
    verify_gate_preflight.get("failover_stress_runtime_evidence_gate")
    if isinstance(verify_gate_preflight.get("failover_stress_runtime_evidence_gate"), dict)
    else {}
)
verify_gate_preflight_launch_readiness_severity = (
    verify_gate_preflight.get("launch_readiness_severity_gate")
    if isinstance(verify_gate_preflight.get("launch_readiness_severity_gate"), dict)
    else {}
)
verify_gate_preflight_launch_readiness_worker_health_canary = (
    verify_gate_preflight.get("launch_readiness_worker_health_canary_gate")
    if isinstance(verify_gate_preflight.get("launch_readiness_worker_health_canary_gate"), dict)
    else {}
)
verify_gate_preflight_launch_readiness_probe_execution = (
    verify_gate_preflight.get("launch_readiness_probe_execution_gate")
    if isinstance(verify_gate_preflight.get("launch_readiness_probe_execution_gate"), dict)
    else {}
)
verify_status_evidence_failure_reason = str(verify_gate_preflight_status_evidence.get("failure_reason") or "").strip()
verify_layered_health_failure_reason = str(verify_gate_preflight_layered_health.get("failure_reason") or "").strip()
verify_layered_health_failing_required_lanes = [
    str(x).strip()
    for x in (verify_gate_preflight_layered_health.get("failing_required_lanes") or [])
    if str(x).strip()
]
verify_layered_health_required_lanes = [
    str(x).strip()
    for x in (verify_gate_preflight_layered_health.get("required_lanes") or [])
    if str(x).strip()
]
verify_layered_health_layer_insufficient_lanes = [
    str(x).strip()
    for x in (verify_gate_preflight_layered_health.get("layer_insufficient_required_lanes") or [])
    if str(x).strip()
]
verify_layered_health_failure_derivative_candidates = {
    "layered_health_not_pass",
    "layered_health_layer_insufficient",
    "layered_health_required_lanes_not_pass",
    "layered_health_required_lanes_layer_insufficient",
}
verify_layered_health_runtime_truth_lanes = {"A2_RUNTIME_CONTINUITY", "C1_OPERATOR_SURFACE"}
verify_failover_stress_runtime_failure_reason = str(
    verify_gate_preflight_failover_stress_runtime.get("failure_reason") or ""
).strip()
verify_failover_stress_runtime_active_blocker = bool(
    verify_gate_preflight_failover_stress_runtime.get("active_blocker") is True
)
verify_launch_readiness_active_blocker = bool(
    verify_gate_preflight_launch_readiness_severity.get("active_blocker") is True
)
verify_worker_health_canary_failure_reason = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("failure_reason") or ""
).strip()
verify_worker_health_canary_active_blocker = bool(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("active_blocker") is True
)
verify_probe_execution_active_blocker = bool(
    verify_gate_preflight_launch_readiness_probe_execution.get("active_blocker") is True
)
verify_probe_execution_failure_reason = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("failure_reason") or ""
).strip()
verify_probe_execution_due_now_count = max(
    0,
    _to_int(verify_gate_preflight_launch_readiness_probe_execution.get("due_now_worker_count"), 0),
)
verify_probe_execution_overdue_count = max(
    0,
    _to_int(verify_gate_preflight_launch_readiness_probe_execution.get("overdue_worker_count"), 0),
)
verify_gate_preflight_predicted_gate = (
    verify_gate_preflight.get("predicted_gate")
    if isinstance(verify_gate_preflight.get("predicted_gate"), dict)
    else {}
)
verify_predicted_blocker_reason = str(
    verify_gate_preflight_predicted_gate.get("predicted_blocker_reason") or ""
).strip()
verify_layered_health_failing_truth_lanes = verify_layered_health_failing_required_lanes + verify_layered_health_layer_insufficient_lanes
verify_layered_health_derivative_truth_residue = bool(
    verify_layered_health_failure_reason in verify_layered_health_failure_derivative_candidates
    and verify_layered_health_failing_truth_lanes
    and set(verify_layered_health_failing_truth_lanes).issubset(verify_layered_health_runtime_truth_lanes)
    and not verify_failover_stress_runtime_active_blocker
    and not verify_worker_health_canary_active_blocker
    and not verify_launch_readiness_active_blocker
    and not verify_probe_execution_active_blocker
)
verify_layered_health_derivative_verify_blocker_primary = bool(
    verify_layered_health_failure_reason
    and verify_status == "BLOCKER"
    and verify_reason == "a6_observability_failed"
    and verify_a6_layered_health_only_failure
    and verify_predicted_blocker_reason.startswith("layered_health_gate:")
    and not verify_failover_stress_runtime_active_blocker
    and not verify_worker_health_canary_active_blocker
    and not verify_launch_readiness_active_blocker
    and not verify_probe_execution_active_blocker
)
verify_layered_health_all_required_lanes_failing = bool(
    verify_layered_health_required_lanes
    and set(verify_layered_health_required_lanes).issubset(
        set(verify_layered_health_failing_required_lanes)
    )
    and set(verify_layered_health_required_lanes).issubset(
        set(verify_layered_health_layer_insufficient_lanes)
    )
)
verify_layered_health_snapshot_self_reference = bool(
    verify_layered_health_failure_reason in verify_layered_health_failure_derivative_candidates
    and verify_status == "READY"
    and not verify_status_evidence_failure_reason
    and verify_predicted_blocker_reason.startswith("layered_health_gate:")
    and not verify_failover_stress_runtime_active_blocker
    and not verify_worker_health_canary_active_blocker
    and not verify_launch_readiness_active_blocker
    and not verify_probe_execution_active_blocker
    and (
        layered_health_snapshot_active
        or (
            verify_layered_health_failure_reason == "layered_health_not_pass"
            and verify_layered_health_all_required_lanes_failing
        )
    )
)
verify_layered_health_derivative_suppressed = bool(
    verify_layered_health_derivative_truth_residue
    or verify_layered_health_derivative_verify_blocker_primary
    or verify_layered_health_snapshot_self_reference
)
verify_blocker_latch_residue_suppressed = bool(
    verify_status == "BLOCKER"
    and verify_reason == "a6_observability_failed"
    and verify_a6_layered_health_only_failure
    and bool(verify_gate_preflight.get("available") is True)
    and verify_gate_preflight_predicted_gate.get("ready_to_run") is True
    and not verify_predicted_blocker_reason
    and not verify_status_evidence_failure_reason
    and not verify_failover_stress_runtime_active_blocker
    and not verify_worker_health_canary_active_blocker
    and not verify_launch_readiness_active_blocker
    and not verify_probe_execution_active_blocker
)
if verify_status == "READY" and verify_status_evidence_failure_reason:
    verify_status_evidence_reason_map = {
        "verify_report_missing": "verify_status_evidence_missing",
        "verify_report_unreadable": "verify_status_evidence_invalid",
        "verify_report_timestamp_missing": "verify_status_evidence_missing",
        "verify_report_timestamp_invalid": "verify_status_evidence_invalid",
        "verify_report_stale": "verify_status_evidence_stale",
    }
    verify_status_evidence_not_ready_reason = verify_status_evidence_reason_map.get(
        verify_status_evidence_failure_reason,
        "verify_status_evidence_untrusted",
    )
    if not verify_then_resume_active:
        not_ready_reasons.append(verify_status_evidence_not_ready_reason)
if (
    verify_layered_health_failure_reason
    and not verify_then_resume_active
    and not verify_layered_health_derivative_suppressed
):
    not_ready_reasons.append("layered_health_gate_unready")
if verify_failover_stress_runtime_active_blocker:
    not_ready_reasons.append("failover_stress_runtime_evidence_gate_active")
if verify_worker_health_canary_active_blocker:
    not_ready_reasons.append("execution_supervisor_worker_health_canary_gate_active")
if verify_launch_readiness_active_blocker:
    not_ready_reasons.append("execution_supervisor_launch_readiness_severity_gate_active")
if verify_probe_execution_active_blocker:
    not_ready_reasons.append("execution_supervisor_probe_execution_overdue_gate_active")

coherence_connectors = (coherence_stamp.get("connectors") or {}) if isinstance(coherence_stamp, dict) else {}
connector_blocking_reasons = [
    str(x) for x in (coherence_connectors.get("blocking_reasons") or []) if str(x).strip()
]
connector_warning_reasons = [
    str(x) for x in (coherence_connectors.get("warning_reasons") or []) if str(x).strip()
]
gtc_connector_blocking_only = bool(gtc_blocking_reasons) and all(
    reason.startswith("connector_stale:") for reason in gtc_blocking_reasons
)
connector_blocking_expiry_only = bool(connector_blocking_reasons) and all(
    reason.startswith("connector_expired:") for reason in connector_blocking_reasons
)
connector_drift_suppressed_by_gtc_gateboard = bool(
    connector_blocking_expiry_only
    and "gtc_gateboard_blocked" in not_ready_reasons
    and gtc_blocking_reasons
    and not gtc_connector_blocking_only
)
if connector_blocking_reasons and not connector_drift_suppressed_by_gtc_gateboard:
    not_ready_reasons.append("connector_freshness_drift")

seen_not_ready = set()
ordered_not_ready = []
for reason in not_ready_reasons:
    if reason in seen_not_ready:
        continue
    seen_not_ready.add(reason)
    ordered_not_ready.append(reason)

raw_not_ready_reasons = ordered_not_ready
drift_ready_reclass_reason_set = {
    "ground_truth_capture_drift",
    "connector_freshness_drift",
    "policy_freshness_drift",
}
policy_drift_present = "policy_freshness_drift" in raw_not_ready_reasons
ground_truth_drift_present = "ground_truth_capture_drift" in raw_not_ready_reasons
policy_drift_reclass_allowed = (not policy_drift_present) or ground_truth_drift_present
verify_ready_drift_reclassified_reasons: List[str] = []
if raw_not_ready_reasons and verify_status == "READY" and policy_drift_reclass_allowed:
    verify_ready_drift_reclassified_reasons = [
        reason for reason in raw_not_ready_reasons if reason in drift_ready_reclass_reason_set
    ]

verify_blocker_latch_residue_reclassified_reasons: List[str] = []
if raw_not_ready_reasons and verify_blocker_latch_residue_suppressed:
    verify_blocker_latch_residue_reclassified_reasons = [
        reason for reason in raw_not_ready_reasons if reason == "verify_blocker"
    ]

reclassified_not_ready_reason_set = set(verify_ready_drift_reclassified_reasons)
reclassified_not_ready_reason_set.update(verify_blocker_latch_residue_reclassified_reasons)

not_ready_reasons = [
    reason for reason in raw_not_ready_reasons if reason not in reclassified_not_ready_reason_set
]
drift_reasons = set(_DRIFT_REASON_SET)
blocker_reasons = [reason for reason in not_ready_reasons if reason not in drift_reasons]
reconcile_only_reasons = [reason for reason in not_ready_reasons if reason in drift_reasons]
for reason in verify_ready_drift_reclassified_reasons:
    if reason not in reconcile_only_reasons:
        reconcile_only_reasons.append(reason)
rollout_blocker_reason_set = {
    "execution_supervisor_launch_readiness_severity_gate_active",
    "execution_supervisor_worker_health_canary_gate_active",
    "execution_supervisor_probe_execution_overdue_gate_active",
    "failover_stress_runtime_evidence_gate_active",
}
rollout_blocker_reasons = [reason for reason in blocker_reasons if reason in rollout_blocker_reason_set]
summary["ready"] = len(not_ready_reasons) == 0
summary["not_ready_reasons"] = not_ready_reasons
summary["blocker_reasons"] = blocker_reasons
summary["reconcile_only_reasons"] = reconcile_only_reasons
summary["rollout_blocker_reasons"] = rollout_blocker_reasons
summary["rollout_blocked"] = bool(rollout_blocker_reasons)
summary.setdefault("coherence", {})
summary["coherence"]["policy_verify_signature"] = verify_policy_sig or None
summary["coherence"]["policy_verify_signature_match"] = (verify_policy_sig == live_policy_sig) if verify_policy_sig and live_policy_sig else None
summary["coherence"]["connector_blocking_reasons"] = connector_blocking_reasons
summary["coherence"]["connector_warning_reasons"] = connector_warning_reasons
summary["coherence"]["connector_drift_suppressed_by_gtc_gateboard"] = connector_drift_suppressed_by_gtc_gateboard
summary["coherence"]["connector_drift_suppression_reason"] = (
    "connector_expiry_derivative_of_gtc_non_connector_blocker"
    if connector_drift_suppressed_by_gtc_gateboard
    else None
)
summary["coherence"]["gtc_gateboard_derivative_suppressed"] = gtc_gateboard_blocked_suppressed
summary["coherence"]["gtc_gateboard_derivative_suppression_reason"] = (
    gtc_gateboard_blocked_suppression_reason
    if gtc_gateboard_blocked_suppressed
    else None
)
summary["coherence"]["layered_health_derivative_suppressed"] = verify_layered_health_derivative_suppressed
summary["coherence"]["layered_health_derivative_suppression_reason"] = (
    "layered_health_runtime_truth_residue"
    if verify_layered_health_derivative_truth_residue
    else (
        "layered_health_verify_blocker_primary"
        if verify_layered_health_derivative_verify_blocker_primary
        else (
            "layered_health_snapshot_self_reference"
            if verify_layered_health_snapshot_self_reference
            else None
        )
    )
)
summary["coherence"]["layered_health_derivative_failing_lanes"] = verify_layered_health_failing_truth_lanes
summary["coherence"]["layered_health_derivative_verify_a6_components"] = verify_a6_failed_components
summary["coherence"]["verify_ready_drift_reclassified"] = bool(verify_ready_drift_reclassified_reasons)
summary["coherence"]["verify_ready_drift_reclassified_reasons"] = verify_ready_drift_reclassified_reasons
summary["coherence"]["verify_blocker_latch_residue_suppressed"] = bool(
    verify_blocker_latch_residue_reclassified_reasons
)
summary["coherence"]["verify_blocker_latch_residue_reclassified_reasons"] = (
    verify_blocker_latch_residue_reclassified_reasons
)
summary["coherence"]["verify_blocker_latch_residue_suppression_reason"] = (
    "a6_observability_verify_blocker_latch_residue"
    if verify_blocker_latch_residue_reclassified_reasons
    else None
)

warning_reasons = []
verify_gate_preflight_summary = ((summary.get("verify") or {}).get("gate_preflight") or {}) if isinstance((summary.get("verify") or {}).get("gate_preflight"), dict) else {}
verify_gate_preflight_predicted = verify_gate_preflight_summary.get("predicted_gate") if isinstance(verify_gate_preflight_summary.get("predicted_gate"), dict) else {}
if refresh_preflight_failed:
    warning_reasons.append("refresh_preflight_failed")
refresh_storm_guard_summary = summary.get("refresh_storm_guard") if isinstance(summary.get("refresh_storm_guard"), dict) else {}
if bool(summary.get("refresh_hooks", {}).get("skipped_due_to_storm_guard") is True) or (
    str(refresh_storm_guard_summary.get("decision") or "").strip().lower() == "skip"
):
    warning_reasons.append("refresh_hook_storm_guard_active")
if int(summary.get("refresh_hooks", {}).get("failure_count") or 0) > 0:
    warning_reasons.append("refresh_hook_failures")
if str(verify_gate_preflight_predicted.get("predicted_blocker_reason") or "").strip():
    warning_reasons.append("verify_gate_preflight_blocker_predicted")
if verify_layered_health_failure_reason:
    if verify_layered_health_derivative_truth_residue:
        warning_reasons.append("layered_health_gate_unready_suppressed_derivative")
    elif verify_layered_health_derivative_verify_blocker_primary:
        warning_reasons.append("layered_health_gate_unready_suppressed_by_verify_blocker")
    elif verify_layered_health_snapshot_self_reference:
        warning_reasons.append("layered_health_gate_unready_suppressed_derivative")
    else:
        warning_reasons.append("layered_health_gate_unready")
if verify_blocker_latch_residue_reclassified_reasons:
    warning_reasons.append("verify_blocker_latch_residue_suppressed")
if verify_failover_stress_runtime_failure_reason:
    verify_failover_stress_runtime_warning_map = {
        "failover_stress_runtime_evidence_missing": "failover_stress_runtime_evidence_missing",
        "failover_stress_runtime_evidence_not_regular_file": "failover_stress_runtime_evidence_invalid",
        "failover_stress_runtime_evidence_unreadable": "failover_stress_runtime_evidence_invalid",
        "failover_stress_runtime_evidence_invalid": "failover_stress_runtime_evidence_invalid",
        "failover_stress_runtime_generated_at_missing": "failover_stress_runtime_evidence_timestamp_missing",
        "failover_stress_runtime_generated_at_invalid": "failover_stress_runtime_evidence_timestamp_invalid",
        "failover_stress_runtime_stale": "failover_stress_runtime_evidence_stale",
        "failover_stress_runtime_overall_verdict_nonpass": "failover_stress_runtime_evidence_verdict_nonpass",
        "failover_stress_runtime_publish_chain_nonpass": "failover_stress_runtime_publish_chain_nonpass",
        "failover_stress_runtime_publish_assertions_failed": "failover_stress_runtime_publish_assertions_failed",
        "failover_stress_runtime_repeatability_mismatch": "failover_stress_runtime_repeatability_mismatch",
    }
    warning_reasons.append(
        verify_failover_stress_runtime_warning_map.get(
            verify_failover_stress_runtime_failure_reason,
            "failover_stress_runtime_evidence_attention",
        )
    )
if verify_failover_stress_runtime_active_blocker:
    warning_reasons.append("failover_stress_runtime_evidence_gate_active")
if verify_worker_health_canary_failure_reason:
    verify_worker_health_canary_warning_map = {
        "dispatch_qualification_missing": "execution_supervisor_dispatch_qualification_missing",
        "dispatch_qualification_not_regular_file": "execution_supervisor_dispatch_qualification_invalid",
        "dispatch_qualification_unreadable": "execution_supervisor_dispatch_qualification_invalid",
        "dispatch_qualification_invalid": "execution_supervisor_dispatch_qualification_invalid",
        "dispatch_qualification_generated_at_invalid": "execution_supervisor_dispatch_qualification_timestamp_invalid",
        "dispatch_qualification_stale": "execution_supervisor_dispatch_qualification_stale",
        "worker_health_canary_missing": "execution_supervisor_worker_health_canary_missing",
        "worker_health_canary_path_missing": "execution_supervisor_worker_health_canary_missing",
        "worker_health_canary_not_regular_file": "execution_supervisor_worker_health_canary_invalid",
        "worker_health_canary_unreadable": "execution_supervisor_worker_health_canary_invalid",
        "worker_health_canary_invalid": "execution_supervisor_worker_health_canary_invalid",
        "worker_health_canary_generated_at_missing": "execution_supervisor_worker_health_canary_timestamp_missing",
        "worker_health_canary_generated_at_invalid": "execution_supervisor_worker_health_canary_timestamp_invalid",
        "worker_health_canary_stale": "execution_supervisor_worker_health_canary_stale",
        "worker_health_canary_future": "execution_supervisor_worker_health_canary_future_timestamp",
        "dispatch_resource_preflight_blocked": "execution_supervisor_dispatch_resource_preflight_blocked",
        "dispatch_resource_preflight_degraded": "execution_supervisor_dispatch_resource_preflight_degraded",
        "dispatch_uncertainty_operator_review_required": "execution_supervisor_dispatch_uncertainty_review_required",
    }
    warning_reasons.append(
        verify_worker_health_canary_warning_map.get(
            verify_worker_health_canary_failure_reason,
            "execution_supervisor_worker_health_canary_attention",
        )
    )
if verify_worker_health_canary_active_blocker:
    warning_reasons.append("execution_supervisor_worker_health_canary_gate_active")
if bool(verify_gate_preflight_launch_readiness_worker_health_canary.get("uncertainty_requires_operator_review") is True):
    warning_reasons.append("execution_supervisor_dispatch_uncertainty_review_required")
resource_preflight_status_warning = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("resource_preflight_status") or ""
).strip().lower()
if resource_preflight_status_warning == "degraded":
    warning_reasons.append("execution_supervisor_dispatch_resource_preflight_degraded")
elif resource_preflight_status_warning == "blocked":
    warning_reasons.append("execution_supervisor_dispatch_resource_preflight_blocked")
if verify_launch_readiness_active_blocker:
    warning_reasons.append("execution_supervisor_launch_readiness_severity_gate_active")
if verify_probe_execution_due_now_count > 0:
    if verify_probe_execution_failure_reason == "launch_readiness_probe_execution_due_now_idle_no_dispatch_candidate":
        warning_reasons.append("execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate")
    else:
        warning_reasons.append("execution_supervisor_probe_execution_due_now")
if verify_probe_execution_overdue_count > 0:
    warning_reasons.append("execution_supervisor_probe_execution_overdue")
if verify_probe_execution_active_blocker:
    warning_reasons.append("execution_supervisor_probe_execution_overdue_gate_active")
if reset_ready_refresh_degraded:
    warning_reasons.append("reset_ready_refresh_degraded")
if reset_ready_refresh_partial_failure:
    warning_reasons.append("reset_ready_refresh_partial_failure")
if reset_ready_refresh_fresh is False:
    warning_reasons.append("reset_ready_refresh_stale")
if verify_status == "READY" and verify_status_evidence_failure_reason:
    verify_status_evidence_warning_map = {
        "verify_report_missing": "verify_status_evidence_missing",
        "verify_report_unreadable": "verify_status_evidence_invalid",
        "verify_report_timestamp_missing": "verify_status_evidence_missing",
        "verify_report_timestamp_invalid": "verify_status_evidence_invalid",
        "verify_report_stale": "verify_status_evidence_stale",
    }
    warning_reasons.append(
        verify_status_evidence_warning_map.get(
            verify_status_evidence_failure_reason,
            "verify_status_evidence_untrusted",
        )
    )
if summary.get("parity", {}).get("due") is True:
    warning_reasons.append("parity_weekly_freshness_due")
if ground_truth_drift_cooldown_active:
    if ground_truth_capture_drift_cooldown_policy_lag:
        warning_reasons.append("ground_truth_capture_drift_cooldown_policy_lag")
    else:
        warning_reasons.append("ground_truth_capture_drift_cooldown")
if queue_role_missing_count > 0:
    warning_reasons.append("queue_role_required_unset")
if queue_review_role_mismatch_count > 0:
    warning_reasons.append("queue_review_role_mismatch")
if stale_active_file_lock_count > 0:
    warning_reasons.append("stale_active_file_locks")
if orphaned_running_without_locks_count > 0:
    warning_reasons.append("orphaned_running_without_locks")
auto_remediate_status = str((orphaned_running_auto_remediation or {}).get("status") or "").strip()
if auto_remediate_status in {"command_failed", "payload_not_ok", "remediator_missing"}:
    warning_reasons.append("orphaned_running_auto_remediation_failed")
if not bool(orphaned_running_auto_remediation_contract.get("healthy")):
    warning_reasons.append("orphaned_running_auto_remediation_contract_invalid")
if bool(autopilot_degraded_pending_signal.get("active")):
    warning_reasons.append("degraded_pending_backlog_stale_sustained")
queue_stale_wave_auto_status = str((queue_stale_wave_auto_remediation or {}).get("status") or "").strip()
queue_stale_wave_auto_last_attempt_status = str((queue_stale_wave_auto_remediation or {}).get("last_attempt_status") or "").strip()
queue_stale_wave_auto_failed_statuses = {"command_failed", "payload_not_ok", "remediator_missing", "remediation_degraded"}
queue_stale_wave_auto_failure_present = bool(
    queue_stale_wave_auto_status in queue_stale_wave_auto_failed_statuses
    or (
        queue_stale_wave_auto_status == "cooldown_active"
        and queue_stale_wave_auto_last_attempt_status in queue_stale_wave_auto_failed_statuses
        and bool(queue_stale_wave_signal.get("active"))
    )
)
if bool(queue_stale_wave_signal.get("active")):
    if queue_stale_wave_auto_failure_present:
        warning_reasons.append("queue_stale_wave_auto_remediation_failed")
    else:
        warning_reasons.append("queue_stale_wave_ready_backlog")
if not bool(queue_stale_wave_auto_remediation_contract.get("healthy")):
    warning_reasons.append("queue_stale_wave_auto_remediation_contract_invalid")
idle_lane_actionable = bool(idle_lane_autospawn.get("ready_work_exists")) and bool(idle_lane_autospawn.get("idle_threshold_exceeded"))
if (
    idle_lane_autospawn.get("status") in {"tick_failed", "attempted_no_launch", "error"}
    and idle_lane_actionable
):
    warning_reasons.append("idle_lane_autospawn_stalled")
idle_lane_contract_source_degraded = bool(idle_lane_autospawn.get("contract_source_degraded"))
idle_lane_contract_degraded_reason = str(
    idle_lane_autospawn.get("contract_source_degraded_reason") or ""
).strip()
idle_lane_contradiction_latched = bool(
    idle_lane_actionable
    and not idle_lane_contract_source_degraded
    and (
        str(idle_lane_autospawn.get("skip_reason") or "") == "contradiction_latched_auto_abort"
        or bool(idle_lane_autospawn.get("contradiction_abort_active"))
    )
)
if idle_lane_contradiction_latched:
    warning_reasons.append("idle_lane_autospawn_contradiction_latched")
idle_lane_trace_stale_blocked_watchdog = bool(
    idle_lane_contract_source_degraded
    and idle_lane_contract_degraded_reason == "trace_stale"
    and (blocker_reasons or not_ready_reasons)
    and str(idle_lane_autospawn.get("status") or "") == "skipped"
    and not idle_lane_actionable
    and not bool(idle_lane_autospawn.get("launched"))
)
if idle_lane_contract_source_degraded:
    if not idle_lane_trace_stale_blocked_watchdog:
        warning_reasons.append("idle_lane_autospawn_source_degraded")
    idle_lane_degraded_reason_warning_map = {
        "trace_invalid": "idle_lane_autospawn_trace_invalid",
        "updated_at_missing": "idle_lane_autospawn_updated_at_missing",
        "updated_at_invalid": "idle_lane_autospawn_updated_at_invalid",
        "trace_stale": "idle_lane_autospawn_trace_stale",
    }
    idle_lane_reason_warning = idle_lane_degraded_reason_warning_map.get(
        idle_lane_contract_degraded_reason
    )
    if idle_lane_reason_warning and not idle_lane_trace_stale_blocked_watchdog:
        warning_reasons.append(idle_lane_reason_warning)
execution_frontier_controller_contract_source_degraded = bool(
    execution_frontier_controller.get("contract_source_degraded") is True
)
execution_frontier_controller_contract_degraded_reason = str(
    execution_frontier_controller.get("contract_source_degraded_reason") or ""
).strip()
execution_frontier_controller_trace_stale_blocked_watchdog = bool(
    execution_frontier_controller_contract_source_degraded
    and execution_frontier_controller_contract_degraded_reason == "trace_stale"
    and (blocker_reasons or not_ready_reasons)
    and str(execution_frontier_controller.get("status") or "") == "skipped"
    and str(execution_frontier_controller.get("decision") or "") == "SKIP"
    and not bool(execution_frontier_controller.get("dispatch_executed"))
    and not bool(execution_frontier_controller.get("autonomous_dispatch_eligible"))
)
if execution_frontier_controller_contract_source_degraded:
    if not execution_frontier_controller_trace_stale_blocked_watchdog:
        warning_reasons.append("execution_frontier_controller_source_degraded")
    execution_frontier_controller_degraded_reason_warning_map = {
        "trace_invalid": "execution_frontier_controller_trace_invalid",
        "recorded_at_missing": "execution_frontier_controller_recorded_at_missing",
        "recorded_at_invalid": "execution_frontier_controller_recorded_at_invalid",
        "trace_stale": "execution_frontier_controller_trace_stale",
    }
    execution_frontier_controller_reason_warning = execution_frontier_controller_degraded_reason_warning_map.get(
        execution_frontier_controller_contract_degraded_reason
    )
    if execution_frontier_controller_reason_warning and not execution_frontier_controller_trace_stale_blocked_watchdog:
        warning_reasons.append(execution_frontier_controller_reason_warning)
execution_frontier_controller_post_completion_required = bool(
    execution_frontier_controller.get("post_completion_enforcement_required") is True
)
execution_frontier_controller_post_completion_status = str(
    execution_frontier_controller.get("status") or "missing"
)
if execution_frontier_controller_post_completion_required:
    if bool(execution_frontier_controller.get("post_completion_enforcement_latched") is True):
        warning_reasons.append("execution_frontier_post_completion_enforcement_latched")
    if execution_frontier_controller_post_completion_status == "blocked":
        warning_reasons.append("execution_frontier_post_completion_enforcement_blocked")
    elif execution_frontier_controller_post_completion_status in {"error", "missing", "skipped"}:
        warning_reasons.append("execution_frontier_post_completion_enforcement_stalled")

execution_frontier_loop_state_warning = str(
    execution_frontier_controller.get("post_completion_loop_state") or ""
).strip()
if execution_frontier_loop_state_warning == "STALLED_LOOP":
    warning_reasons.append("execution_frontier_post_completion_stalled_loop")
elif execution_frontier_loop_state_warning == "BLOCKED_LOOP":
    warning_reasons.append("execution_frontier_post_completion_blocked_loop")

execution_frontier_cooldown_policy_warning = (
    execution_frontier_controller.get("post_completion_cooldown_policy")
    if isinstance(execution_frontier_controller.get("post_completion_cooldown_policy"), dict)
    else {}
)
if bool(execution_frontier_cooldown_policy_warning.get("active") is True):
    warning_reasons.append("execution_frontier_post_completion_cooldown_active")

execution_frontier_retry_contract_warning = (
    execution_frontier_controller.get("post_completion_retry_contract")
    if isinstance(execution_frontier_controller.get("post_completion_retry_contract"), dict)
    else {}
)
if str(execution_frontier_retry_contract_warning.get("state") or "") == "retry_exhausted":
    warning_reasons.append("execution_frontier_post_completion_retry_exhausted")

execution_frontier_parity_warning = (
    execution_frontier_controller.get("queue_truth_vs_narrative_parity")
    if isinstance(execution_frontier_controller.get("queue_truth_vs_narrative_parity"), dict)
    else {}
)
if str(execution_frontier_parity_warning.get("status") or "") == "mismatch":
    warning_reasons.append("execution_frontier_queue_truth_vs_narrative_mismatch")

if int(transition_history_24h.get("total") or 0) > 0 and int(handoff_history_24h.get("total") or 0) == 0:
    warning_reasons.append("handoff_packets_missing_recent")
if verify_ready_drift_reclassified_reasons:
    drift_reclass_warning_map = {
        "ground_truth_capture_drift": "ground_truth_capture_drift_reconcile_only",
        "connector_freshness_drift": "connector_freshness_drift_reconcile_only",
    }
    for reason in verify_ready_drift_reclassified_reasons:
        warning_reason = drift_reclass_warning_map.get(reason, f"{reason}_reconcile_only")
        if warning_reason not in warning_reasons:
            warning_reasons.append(warning_reason)
connector_freshness_drift_present = (
    "connector_freshness_drift" in not_ready_reasons
    or "connector_freshness_drift" in verify_ready_drift_reclassified_reasons
)
if connector_warning_reasons and not connector_freshness_drift_present:
    warning_reasons.append("connector_freshness_warning")
if connector_drift_suppressed_by_gtc_gateboard:
    warning_reasons.append("connector_freshness_drift_suppressed_by_gtc_gateboard")
if gtc_gateboard_blocked_suppressed:
    if gtc_gateboard_blocked_suppression_reason == "gtc_verify_status_stale_after_verify_ready":
        warning_reasons.append("gtc_gateboard_verify_status_lag")
    else:
        warning_reasons.append("gtc_gateboard_blocked_suppressed_by_verify")

gtc_warning_reasons = [
    str(x).strip() for x in (gtc_summary.get("warning_reasons") or []) if str(x).strip()
]
gtc_gateboard_blocked_present = "gtc_gateboard_blocked" in not_ready_reasons
gtc_connector_warning_only = bool(gtc_warning_reasons) and all(
    reason.startswith("connector_stale:") for reason in gtc_warning_reasons
)
if gtc_summary.get("enabled") and gtc_warning_reasons and not gtc_gateboard_blocked_present:
    # Avoid duplicate top-level warning residue when gateboard warnings merely mirror
    # connector freshness warning detail already exposed via coherence.
    if not (gtc_connector_warning_only and connector_warning_reasons):
        warning_reasons.append("gtc_gateboard_warning")

if (coherence_policy.get("policy_missing_paths") or []) or (coherence_policy.get("evaluator_missing_paths") or []):
    warning_reasons.append("policy_inputs_missing")
if int(web_capture_summary.get("operator_action_required_domains") or 0) > 0:
    warning_reasons.append("web_capture_login_wall_pending")
if int(web_capture_summary.get("operator_action_required_domains") or 0) > int(web_capture_summary.get("actionable_incident_domains") or 0):
    warning_reasons.append("web_capture_login_wall_actionability_missing")
if int(web_capture_summary.get("cooldown_active_domains") or 0) > 0:
    warning_reasons.append("web_capture_domain_backoff_active")
scheduler_state = web_capture_summary.get("scheduler") if isinstance(web_capture_summary.get("scheduler"), dict) else {}
if scheduler_state.get("selection_status") == "idle_no_eligible_macro" and int(scheduler_state.get("total_macros") or 0) > 0:
    warning_reasons.append("web_capture_scheduler_idle")
if scheduler_state.get("state_exists") is False:
    warning_reasons.append("web_capture_scheduler_state_missing")
if scheduler_state.get("contract_state_valid") is False:
    warning_reasons.append("web_capture_scheduler_contract_invalid")
# Avoid duplicate top-level scheduler warning residue: when the scheduler contract
# is already invalid, stale freshness is usually derivative/noisy. Keep stale as
# the warning only for contract-valid scheduler state.
if scheduler_state.get("fresh") is False and scheduler_state.get("contract_state_valid") is not False:
    warning_reasons.append("web_capture_scheduler_stale")

rollout_warning_reason_prefixes = (
    "execution_supervisor_launch_readiness_",
    "execution_supervisor_worker_health_canary_",
    "execution_supervisor_probe_execution_",
    "execution_supervisor_dispatch_qualification_",
    "execution_supervisor_dispatch_resource_preflight_",
    "execution_supervisor_dispatch_uncertainty_",
    "failover_stress_runtime_evidence_",
)
rollout_warning_reasons: List[str] = []
seen_rollout_warning_reasons = set()
for reason in warning_reasons:
    if reason in rollout_blocker_reason_set:
        continue
    if not any(reason.startswith(prefix) for prefix in rollout_warning_reason_prefixes):
        continue
    if reason in seen_rollout_warning_reasons:
        continue
    seen_rollout_warning_reasons.add(reason)
    rollout_warning_reasons.append(reason)

summary["warning_reasons"] = warning_reasons
summary["rollout_warning_reasons"] = rollout_warning_reasons
summary["rollout_warning_active"] = bool(rollout_warning_reasons)

mutation_projection_reasons: List[str] = []
if summary.get("ready") is not True:
    mutation_projection_reasons.append("continuity_not_ready")
if in_flight_effective:
    mutation_projection_reasons.append("in_flight_work_present")
mutation_projection_concurrency = [reason for reason in mutation_projection_reasons if reason == "in_flight_work_present"]
mutation_projection_blocking = [reason for reason in mutation_projection_reasons if reason != "in_flight_work_present"]
mutation_projection_expected_in_flight_guard = bool(summary.get("ready") is True and mutation_projection_concurrency and not mutation_projection_blocking)
summary["mutation_gate_projection"] = {
    "status": "allowed" if not mutation_projection_reasons else "forbidden",
    "reason": ["all_resume_gates_green"] if not mutation_projection_reasons else mutation_projection_reasons,
    "blocking_reasons": mutation_projection_blocking,
    "concurrency_reasons": mutation_projection_concurrency,
    "posture": (
        "open"
        if not mutation_projection_reasons
        else ("concurrency_guard" if mutation_projection_expected_in_flight_guard else "blocker")
    ),
    "expected_in_flight_guard": mutation_projection_expected_in_flight_guard,
}

reconcile_recommended = bool(reconcile_only_reasons) and not blocker_reasons
summary["reconcile"] = {
    "recommended": reconcile_recommended,
    "command": cmd_reconcile_with_token,
    "cooldown_active": ground_truth_drift_cooldown_active,
    "min_interval_sec": reconcile_min_interval_sec,
    "cooldown_policy_lag_warning_active": ground_truth_capture_drift_cooldown_policy_lag,
    # Backward-compatible alias for older consumers; use cooldown_policy_lag_warning_active.
    "cooldown_warning_suppressed_policy_lag": ground_truth_capture_drift_cooldown_policy_lag,
}

replay_cmds = []
for cmd in list(gtc_summary.get("incident_replay_commands") or [])[:6]:
    if isinstance(cmd, str) and cmd.strip():
        replay_cmds.append(normalize_operator_command(cmd))
for hook in list(summary.get("refresh_hooks", {}).get("hooks") or []):
    if not isinstance(hook, dict):
        continue
    if hook.get("ok") is not False:
        continue
    replay_cmd = normalize_operator_command(hook.get("command") or "")
    if replay_cmd:
        replay_cmds.append(replay_cmd)
if refresh_preflight_failed:
    replay_cmds.append(cmd_snapshot_ground_truth)
if active_step:
    replay_cmds.append(
        f"{cmd_queue_trace_prefix} --task-id autopilot:{active_step} --json"
    )
if transition_last_event and transition_last_event.get("task_id"):
    replay_cmds.append(
        f"{cmd_queue_trace_prefix} --task-id {transition_last_event.get('task_id')} --json"
    )
if not_ready_reasons:
    replay_cmds.append(
        cmd_cont_history_json
    )
if str(verify_gate_preflight_predicted.get("predicted_blocker_reason") or "").strip():
    replay_cmds.append(cmd_cont_verify_gate_status_json)
if verify_failover_stress_runtime_failure_reason:
    replay_cmds.append(cmd_cont_verify_gate_status_json)
    inspect_failover_runtime_command = normalize_operator_command(
        verify_gate_preflight_failover_stress_runtime.get("inspect_failover_stress_runtime_evidence_command")
        or cmd_read_failover_stress_runtime_evidence_json
    )
    if inspect_failover_runtime_command:
        replay_cmds.append(inspect_failover_runtime_command)
if verify_failover_stress_runtime_active_blocker:
    refresh_failover_runtime_command = normalize_operator_command(
        verify_gate_preflight_failover_stress_runtime.get("refresh_failover_stress_runtime_evidence_command")
        or cmd_cont_failover_stress_runtime_evidence_refresh_json
    )
    if refresh_failover_runtime_command:
        replay_cmds.append(refresh_failover_runtime_command)
if verify_worker_health_canary_failure_reason:
    replay_cmds.append(cmd_cont_verify_gate_status_json)
    if verify_worker_health_canary_failure_reason in {
        "dispatch_resource_preflight_blocked",
        "dispatch_resource_preflight_degraded",
        "dispatch_uncertainty_operator_review_required",
    }:
        inspect_dispatch_command = normalize_operator_command(
            verify_gate_preflight_launch_readiness_worker_health_canary.get("inspect_dispatch_qualification_command")
        )
        if inspect_dispatch_command:
            replay_cmds.append(inspect_dispatch_command)
    else:
        inspect_worker_health_canary_command = normalize_operator_command(
            verify_gate_preflight_launch_readiness_worker_health_canary.get("inspect_worker_health_canary_command")
            or cmd_read_execution_supervisor_worker_health_canary_json
        )
        if inspect_worker_health_canary_command:
            replay_cmds.append(inspect_worker_health_canary_command)
if verify_worker_health_canary_active_blocker:
    refresh_worker_health_canary_command = normalize_operator_command(
        verify_gate_preflight_launch_readiness_worker_health_canary.get("refresh_worker_health_canary_command")
        or cmd_cont_worker_health_canary_refresh_json
    )
    if refresh_worker_health_canary_command:
        replay_cmds.append(refresh_worker_health_canary_command)
if verify_status == "READY" and verify_status_evidence_failure_reason:
    replay_cmds.append(cmd_cont_verify_gate_status_json)
    replay_cmds.append(cmd_cont_verify_json)
if reset_ready_refresh_degraded or reset_ready_refresh_fresh is False:
    replay_cmds.append(cmd_cont_reset_ready_refresh_json)
    replay_cmds.append(cmd_read_reset_ready_refresh_latest_json)
if warning_reasons and "parity_weekly_freshness_due" in warning_reasons:
    replay_cmds.append(
        cmd_parity_force
    )
if queue_role_missing_count > 0 or queue_review_role_mismatch_count > 0:
    replay_cmds.append(
        cmd_db_integrity_strict_json
    )
if stale_active_file_lock_count > 0:
    replay_cmds.append(
        cmd_queue_locks_active_json
    )
if stale_active_file_lock_count > 0 or orphaned_running_without_locks_count > 0:
    replay_cmds.append(
        cmd_queue_remediate_extended_json
    )
if not bool(orphaned_running_auto_remediation_contract.get("healthy")):
    replay_cmds.append(
        cmd_read_orphaned_auto_json
    )
if not bool(queue_stale_wave_auto_remediation_contract.get("healthy")):
    replay_cmds.append(
        cmd_read_stale_wave_auto_json
    )
if queue_stale_wave_auto_failure_present:
    replay_cmds.append(
        cmd_read_stale_wave_auto_json
    )
if bool(autopilot_degraded_pending_signal.get("active")):
    replay_cmds.append(
        cmd_cont_queue_sync_json
    )
    trace_task_id = str(autopilot_degraded_pending_signal.get("pending_oldest_task_id") or "").strip()
    if trace_task_id:
        replay_cmds.append(
            f"{cmd_queue_trace_prefix} --task-id {trace_task_id} --json"
        )
if bool(queue_stale_wave_signal.get("active")):
    replay_cmds.append(normalize_operator_command(queue_stale_wave_signal.get("inspect_command") or cmd_queue_ready_list_json))
    replay_cmds.append(normalize_operator_command(queue_stale_wave_signal.get("recovery_command") or cmd_cont_queue_sync_json))
if (
    "idle_lane_autospawn_stalled" in warning_reasons
    or "idle_lane_autospawn_contradiction_latched" in warning_reasons
    or "idle_lane_autospawn_source_degraded" in warning_reasons
    or bool(idle_lane_autospawn.get("contradiction_latch_repaired") is True)
):
    replay_cmds.append(
        cmd_watchdog
    )
    replay_cmds.append(
        cmd_read_idle_lane_trace_json
    )
    replay_cmds.append(
        cmd_idle_lane_watchdog_history_json
    )
if (
    execution_frontier_controller_status in {"applied", "blocked", "error"}
    or execution_frontier_controller_contract_source_degraded
    or execution_frontier_loop_state_warning in {"STALLED_LOOP", "BLOCKED_LOOP"}
    or bool(execution_frontier_cooldown_policy_warning.get("active") is True)
    or str(execution_frontier_retry_contract_warning.get("state") or "") == "retry_exhausted"
    or str(execution_frontier_parity_warning.get("status") or "") == "mismatch"
    or (
        bool(execution_frontier_controller.get("post_completion_enforcement_required") is True)
        and execution_frontier_controller_status in {"missing", "skipped"}
    )
):
    replay_cmds.append(
        cmd_watchdog
    )
    replay_cmds.append(
        cmd_read_execution_frontier_controller_trace_json
    )
    replay_cmds.append(
        cmd_read_execution_frontier_controller_history_json
    )
    replay_cmds.append(
        cmd_read_execution_frontier_enforcement_latch_json
    )
    replay_cmds.append(
        cmd_read_execution_frontier_enforcement_latch_history_json
    )
    replay_cmds.append(
        cmd_read_autonomous_execution_intent_json
    )
    replay_cmds.append(
        cmd_read_autonomous_execution_intent_history_json
    )
if int(web_capture_summary.get("operator_action_required_domains") or 0) > 0:
    for row in list(web_capture_summary.get("domains") or [])[:3]:
        if not row.get("operator_action_required"):
            continue
        if row.get("operator_contract_json"):
            replay_cmds.append(cat_cmd_for(row.get("operator_contract_json")))
        if row.get("operator_contract_md"):
            replay_cmds.append(cat_cmd_for(row.get("operator_contract_md")))
        incident = row.get("incident_actionability") if isinstance(row.get("incident_actionability"), dict) else {}
        for cmd in list(incident.get("recommended_commands") or [])[:4]:
            replay_cmds.append(normalize_operator_command(cmd))
        if row.get("operator_resume_command"):
            replay_cmds.append(normalize_operator_command(row.get("operator_resume_command")))
if int(web_capture_summary.get("cooldown_active_domains") or 0) > 0:
    replay_cmds.append(
        cmd_web_capture_auto_dry_json
    )
scheduler_state = web_capture_summary.get("scheduler") if isinstance(web_capture_summary.get("scheduler"), dict) else {}
if scheduler_state.get("selection_status") == "idle_no_eligible_macro":
    replay_cmds.append(
        cmd_web_capture_scheduler_dry_json
    )
if scheduler_state.get("state_exists") is False:
    replay_cmds.append(
        cmd_web_capture_scheduler_dry_json
    )
if scheduler_state.get("contract_state_valid") is False:
    replay_cmds.append(
        cmd_web_capture_scheduler_dry_json
    )
if scheduler_state.get("fresh") is False:
    replay_cmds.append(
        cmd_web_capture_scheduler_dry_json
    )

seen_cmds = set()
incident_replay["recommended_commands"] = []
for cmd in replay_cmds:
    if cmd in seen_cmds:
        continue
    seen_cmds.add(cmd)
    incident_replay["recommended_commands"].append(cmd)

summary["incident_replay"] = incident_replay

coherence_bundle = {
    "schema_version": "continuity.coherence_bundle.v1",
    "generated_at": generated_at_iso,
    "build_generation_id": build_generation_id,
    "valid_until": coherence_valid_until_iso,
    "continuity_now": summary,
    "coherence_stamp": coherence_stamp if isinstance(coherence_stamp, dict) else {},
}
# Commit bundle first as single-source atomic payload; legacy surfaces are then rewritten from the same generation.
atomic_write(coherence_bundle_path, coherence_bundle)
atomic_write(continuity_now_latest_path, summary)
if isinstance(coherence_stamp, dict) and coherence_stamp:
    atomic_write(coherence_stamp_path, coherence_stamp)

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("CONTINUITY NOW")
    print(f"- ready: {summary['ready']}")
    mutation_gate_projection = summary.get("mutation_gate_projection") if isinstance(summary.get("mutation_gate_projection"), dict) else {}
    print(
        "- mutation_gate_projection: "
        f"status={mutation_gate_projection.get('status') or 'unknown'}; "
        f"posture={mutation_gate_projection.get('posture') or 'unknown'}; "
        f"expected_in_flight_guard={mutation_gate_projection.get('expected_in_flight_guard')}; "
        f"reasons={mutation_gate_projection.get('reason') or []}"
    )
    print(
        f"- checkpoint: {checkpoint_id} status={checkpoint_status} age={age_compact(checkpoint_age)} trigger={summary['checkpoint'].get('trigger') or 'n/a'}"
    )
    verify_gate_preflight_row = ((summary.get("verify") or {}).get("gate_preflight") or {}) if isinstance((summary.get("verify") or {}).get("gate_preflight"), dict) else {}
    verify_gate_preflight_strict = verify_gate_preflight_row.get("strict_autonomy") if isinstance(verify_gate_preflight_row.get("strict_autonomy"), dict) else {}
    verify_gate_preflight_predicted = verify_gate_preflight_row.get("predicted_gate") if isinstance(verify_gate_preflight_row.get("predicted_gate"), dict) else {}
    verify_gate_preflight_evidence = verify_gate_preflight_row.get("status_evidence_gate") if isinstance(verify_gate_preflight_row.get("status_evidence_gate"), dict) else {}
    print(
        "- verify: "
        f"status={verify_status} age={age_compact(verify_age)} reason={verify_reason or 'n/a'}; "
        f"strict_autonomy={verify_strict_enabled}; "
        f"strict_source={verify_strict_effective_source}; "
        f"strict_result_ok={verify_strict_result_ok if verify_strict_result_ok is not None else 'n/a'}; "
        f"strict_required={verify_gate_preflight_strict.get('required') if verify_gate_preflight_strict.get('required') is not None else 'n/a'}; "
        f"strict_override={verify_gate_preflight_strict.get('override') or 'none'}; "
        f"predicted_blocker={verify_gate_preflight_predicted.get('predicted_blocker_reason') or 'none'}; "
        f"status_evidence={verify_gate_preflight_evidence.get('failure_reason') or 'ok'}"
    )
    print(
        f"- ground_truth: {gt_snapshot_id} age={age_compact(gt_age)} anomalies={len(anomalies)} critical={len(critical_keys)} warn={len(warn_keys)}"
    )
    print(
        "- bridge: "
        f"pointer_match={pointer_ok if pointer_ok is not None else 'n/a'}; "
        f"pointer_sha={pointer_sha_ok if pointer_sha_ok is not None else 'n/a'}; "
        f"env_capture_match={env_capture_ok if env_capture_ok is not None else 'n/a'}; "
        f"gt_capture_match={gt_capture_ok if gt_capture_ok is not None else 'n/a'}"
    )
    print(
        "- autopilot: "
        f"paused={summary['autopilot'].get('paused')}; "
        f"cycle={summary['autopilot'].get('cycle')}/{summary['autopilot'].get('max_cycles')}; "
        f"active_step={active_step or 'none'}"
    )
    if active_evidence_refs:
        print(f"- autopilot_active_evidence: {', '.join(active_evidence_refs)}")
    if autopilot_recent_evidence_refs:
        print(f"- autopilot_recent_evidence: {', '.join(autopilot_recent_evidence_refs)}")
    if autopilot_degraded_pending_signal.get("active"):
        print(
            "- autopilot_degraded_pending_backlog: "
            f"stale_pending={autopilot_degraded_pending_signal.get('pending_stale_count')}; "
            f"pending_total={autopilot_degraded_pending_signal.get('pending_total')}; "
            f"oldest_age_sec={autopilot_degraded_pending_signal.get('pending_oldest_age_sec')}; "
            f"oldest_run={autopilot_degraded_pending_signal.get('pending_oldest_run_id') or 'none'}; "
            f"streak={autopilot_degraded_pending_signal.get('stale_ticks_consecutive')}/{autopilot_degraded_pending_signal.get('activate_after_ticks') or 'n/a'}"
        )
    if idle_lane_autospawn.get("status") != "missing":
        print(
            "- autopilot_idle_lane_autospawn: "
            f"status={idle_lane_autospawn.get('status')}; "
            f"ready_work={idle_lane_autospawn.get('ready_work_exists')}; "
            f"idle_exceeded={idle_lane_autospawn.get('idle_threshold_exceeded')}; "
            f"idle_sec={idle_lane_autospawn.get('idle_sec')}; "
            f"target_step={idle_lane_autospawn.get('target_step_id') or 'none'}; "
            f"launched_step={idle_lane_autospawn.get('launched_step_id') or 'none'}; "
            f"skip_reason={idle_lane_autospawn.get('skip_reason') or 'none'}; "
            f"contradiction_abort_active={idle_lane_autospawn.get('contradiction_abort_active')}; "
            f"contradiction_abort_remaining_sec={idle_lane_autospawn.get('contradiction_abort_remaining_sec')}; "
            f"contradiction_latch_repaired={idle_lane_autospawn.get('contradiction_latch_repaired')}; "
            f"contradiction_latch_repair_reason={idle_lane_autospawn.get('contradiction_latch_repair_reason') or 'none'}"
        )
    if execution_frontier_controller.get("status") != "missing":
        print(
            "- autopilot_execution_frontier_controller: "
            f"status={execution_frontier_controller.get('status')}; "
            f"decision={execution_frontier_controller.get('decision') or 'n/a'}; "
            f"dispatch_decision={execution_frontier_controller.get('dispatch_decision') or 'n/a'}; "
            f"dispatch_advance_applied={execution_frontier_controller.get('dispatch_advance_applied')}; "
            f"post_completion_required={execution_frontier_controller.get('post_completion_enforcement_required')}; "
            f"skip_reason={execution_frontier_controller.get('skip_reason') or 'none'}; "
            f"block_reason={execution_frontier_controller.get('block_reason') or 'none'}; "
            f"error={execution_frontier_controller.get('error') or 'none'}"
        )
    if queue_counts or queue_ready_count or queue_dependency_blocked_count or active_file_lock_count:
        print(
            "- continuity_queue: "
            f"status_counts={queue_counts or {}}; "
            f"role_required={queue_role_required_counts or {}}; "
            f"role_unset={queue_role_missing_count}; "
            f"review_role_mismatch={queue_review_role_mismatch_count}; "
            f"ready={queue_ready_count}; "
            f"dependency_blocked={queue_dependency_blocked_count}; "
            f"active_file_locks={active_file_lock_count}; "
            f"stale_active_locks={stale_active_file_lock_count}; "
            f"effective_active_locks={effective_active_file_lock_count}; "
            f"orphaned_running={orphaned_running_without_locks_count}; "
            f"effective_running={effective_running_count}; "
            f"in_flight_effective={in_flight_effective}"
        )
    auto_remediation_row = (summary.get("queue") or {}).get("orphaned_running_auto_remediation") or {}
    if auto_remediation_row:
        print(
            "- continuity_queue_orphaned_auto_remediation: "
            f"status={auto_remediation_row.get('status')}; "
            f"reason={auto_remediation_row.get('reason') or 'n/a'}; "
            f"eligible={auto_remediation_row.get('eligible')}; "
            f"triggered={auto_remediation_row.get('triggered')}; "
            f"attempted={auto_remediation_row.get('attempted')}; "
            f"requeued={auto_remediation_row.get('applied_requeued_orphaned_running')}; "
            f"cooldown_remaining={auto_remediation_row.get('cooldown_remaining_sec')}s; "
            f"last_attempt_at={auto_remediation_row.get('last_attempt_at') or 'n/a'}"
        )
    auto_remediation_contract_row = (summary.get("queue") or {}).get("orphaned_running_auto_remediation_contract") or {}
    if auto_remediation_contract_row:
        print(
            "- continuity_queue_orphaned_auto_remediation_contract: "
            f"status={auto_remediation_contract_row.get('status')}; "
            f"healthy={auto_remediation_contract_row.get('healthy')}; "
            f"projection_present={auto_remediation_contract_row.get('projection_present')}; "
            f"state_exists={auto_remediation_contract_row.get('state_exists')}; "
            f"state_valid={auto_remediation_contract_row.get('state_valid')}; "
            f"projection_state_match={auto_remediation_contract_row.get('projection_state_match')}; "
            f"issues={auto_remediation_contract_row.get('issues') or []}"
        )
    if queue_dependency_blocked_examples:
        print(
            "- queue_dependency_blocked_examples: "
            + " ; ".join(
                f"{row.get('task_id')}<=({', '.join(row.get('blocked_by') or [])})"
                for row in queue_dependency_blocked_examples
            )
        )
    if active_lock_examples:
        print(
            "- stale_active_lock_examples: "
            + " ; ".join(
                f"{row.get('file_path')}@{row.get('locked_by_task_id')} expires={row.get('lock_expires_at')}"
                for row in active_lock_examples
            )
        )
    if orphaned_running_without_locks_examples:
        print(
            "- orphaned_running_examples: "
            + " ; ".join(
                f"{row.get('task_id')}@{row.get('updated_at')} cutoff={row.get('stale_cutoff')}"
                for row in orphaned_running_without_locks_examples
            )
        )
    print(
        "- queue_transitions(24h): "
        f"total={transition_history_24h.get('total')}; "
        f"to_status={transition_history_24h.get('to_status') or {}}; "
        f"actor_role={transition_history_24h.get('actor_role') or {}}"
    )
    print(
        "- queue_handoffs(24h): "
        f"total={handoff_history_24h.get('total')}; "
        f"role_edges={handoff_history_24h.get('role_edges') or {}}"
    )
    if transition_last_event:
        print(
            "- queue_last_transition: "
            f"task={transition_last_event.get('task_id')}; "
            f"from={transition_last_event.get('from_status') or 'n/a'}; "
            f"to={transition_last_event.get('to_status')}; "
            f"actor={transition_last_event.get('actor_role') or 'n/a'}; "
            f"reason={transition_last_event.get('reason') or 'n/a'}; "
            f"at={transition_last_event.get('created_at')}; "
            f"evidence={transition_last_event.get('evidence_ref') or 'n/a'}"
        )
    if latest_handoff_packet:
        print(
            "- queue_last_handoff: "
            f"task={latest_handoff_packet.get('task_id')}; "
            f"from_role={latest_handoff_packet.get('from_role')}; "
            f"to_role={latest_handoff_packet.get('to_role')}; "
            f"status={latest_handoff_packet.get('from_status') or 'n/a'}->{latest_handoff_packet.get('to_status')}; "
            f"at={latest_handoff_packet.get('created_at')}"
        )
    queue_stale_row = (summary.get("queue") or {}).get("stale_wave_signal") if isinstance(summary.get("queue"), dict) else {}
    if queue_stale_row:
        print(
            "- queue_stale_wave: "
            f"active={queue_stale_row.get('active')}; "
            f"reason={queue_stale_row.get('reason') or 'n/a'}; "
            f"ready={queue_stale_row.get('ready_count')}; "
            f"ready_oldest_age={age_compact(queue_stale_row.get('ready_oldest_age_sec'))}; "
            f"in_flight={queue_stale_row.get('in_flight_effective')}"
        )
    queue_stale_auto_row = (summary.get("queue") or {}).get("stale_wave_auto_remediation") if isinstance(summary.get("queue"), dict) else {}
    if queue_stale_auto_row:
        print(
            "- queue_stale_wave_auto_remediation: "
            f"status={queue_stale_auto_row.get('status')}; "
            f"reason={queue_stale_auto_row.get('reason') or 'n/a'}; "
            f"eligible={queue_stale_auto_row.get('eligible')}; "
            f"triggered={queue_stale_auto_row.get('triggered')}; "
            f"attempted={queue_stale_auto_row.get('attempted')}; "
            f"recovered={queue_stale_auto_row.get('recovered')}; "
            f"cooldown_remaining={queue_stale_auto_row.get('cooldown_remaining_sec')}s; "
            f"last_attempt_at={queue_stale_auto_row.get('last_attempt_at') or 'n/a'}"
        )
    queue_stale_auto_contract_row = (summary.get("queue") or {}).get("stale_wave_auto_remediation_contract") if isinstance(summary.get("queue"), dict) else {}
    if queue_stale_auto_contract_row:
        print(
            "- queue_stale_wave_auto_remediation_contract: "
            f"status={queue_stale_auto_contract_row.get('status')}; "
            f"healthy={queue_stale_auto_contract_row.get('healthy')}; "
            f"projection_present={queue_stale_auto_contract_row.get('projection_present')}; "
            f"state_exists={queue_stale_auto_contract_row.get('state_exists')}; "
            f"state_valid={queue_stale_auto_contract_row.get('state_valid')}; "
            f"projection_state_match={queue_stale_auto_contract_row.get('projection_state_match')}; "
            f"issues={queue_stale_auto_contract_row.get('issues') or []}"
        )
    if recent_evidence_refs:
        print(f"- queue_evidence_refs: {', '.join(recent_evidence_refs)}")
    parity_status = parity_summary.get("status") or "missing"
    parity_age_human = age_compact(parity_summary.get("last_done_age_sec"))
    parity_due = parity_summary.get("due")
    if parity_summary.get("last_done_at"):
        print(
            "- parity_weekly: "
            f"status={parity_status}; "
            f"last_done_at={parity_summary.get('last_done_at')}; "
            f"age={parity_age_human}; "
            f"due={parity_due}"
        )
    else:
        print(f"- parity_weekly: status={parity_status}; last_done_at=n/a; due={parity_due}")
    if web_capture_summary.get("tracked_domains"):
        print(
            "- web_capture_domain_guard: "
            f"tracked={web_capture_summary.get('tracked_domains')}; "
            f"blocked={web_capture_summary.get('blocked_domains')}; "
            f"cooldown_active={web_capture_summary.get('cooldown_active_domains')}; "
            f"operator_required={web_capture_summary.get('operator_action_required_domains')}"
        )
        scheduler_row = web_capture_summary.get("scheduler") if isinstance(web_capture_summary.get("scheduler"), dict) else {}
        if scheduler_row:
            print(
                "- web_capture_scheduler: "
                f"status={scheduler_row.get('selection_status') or 'unknown'}; "
                f"eligible={scheduler_row.get('eligible_macros')}/{scheduler_row.get('total_macros')}; "
                f"fresh={scheduler_row.get('fresh') if scheduler_row.get('fresh') is not None else 'n/a'}; "
                f"age={age_compact(scheduler_row.get('state_age_sec'))}; "
                f"contract_valid={scheduler_row.get('contract_state_valid') if scheduler_row.get('contract_state_valid') is not None else 'n/a'}; "
                f"last={scheduler_row.get('last_selected_domain') or 'n/a'}:{scheduler_row.get('last_selected_macro_slug') or 'n/a'}; "
                f"updated_at={scheduler_row.get('updated_at') or 'n/a'}"
            )
        operator_domains = [
            row for row in (web_capture_summary.get("domains") or []) if row.get("operator_action_required")
        ]
        if operator_domains:
            print(
                "- web_capture_operator_contracts: "
                + " ; ".join(
                    f"{row.get('domain')}=>{row.get('operator_contract_json') or row.get('state_path')}"
                    for row in operator_domains[:4]
                )
            )
    if gtc_summary.get("enabled"):
        print(
            "- gtc_gateboard: "
            f"mutate_allowed={gtc_summary.get('mutate_allowed')}; "
            f"status={gtc_summary.get('status')}; "
            f"open_incidents={gtc_summary.get('open_incident_count')}; "
            f"blocking={gtc_summary.get('blocking_reasons') or []}; "
            f"warnings={gtc_summary.get('warning_reasons') or []}"
        )
    else:
        print("- gtc_gateboard: disabled (run continuity.sh gtc-sync --json)")
    refresh_preflight_state = summary.get("refresh_preflight") if isinstance(summary.get("refresh_preflight"), dict) else {}
    if refresh_preflight_state.get("requested"):
        print(
            "- refresh_preflight: "
            f"ok={refresh_preflight_state.get('ok')}; "
            f"stage={refresh_preflight_state.get('stage') or 'snapshot_ground_truth'}; "
            f"returncode={refresh_preflight_state.get('returncode')}"
        )
    refresh_hook_state = summary.get("refresh_hooks") if isinstance(summary.get("refresh_hooks"), dict) else {}
    if refresh_hook_state.get("requested"):
        failed_hooks = [
            str(x) for x in (refresh_hook_state.get("failed_hooks") or []) if str(x).strip()
        ]
        print(
            "- refresh_hooks: "
            f"requested=true; "
            f"failures={int(refresh_hook_state.get('failure_count') or 0)}; "
            f"failed={failed_hooks or ['none']}"
        )
    print(
        "- reconcile_history(24h): "
        f"checkpoints={reconcile_history.get('checkpoint_count')}; "
        f"events_emitted={reconcile_history.get('event_emitted_count')}; "
        f"events_suppressed={reconcile_history.get('event_suppressed_count')}; "
        f"latest_checkpoint={reconcile_history.get('latest_checkpoint_id') or 'n/a'}; "
        f"latest_event={reconcile_history.get('latest_event_key') or 'n/a'}"
    )
    if not_ready_reasons:
        print(f"- not_ready_reasons: {', '.join(not_ready_reasons)}")
    if warning_reasons:
        print(f"- warning_reasons: {', '.join(warning_reasons)}")
    if reconcile_recommended:
        print(f"- reconcile_hint: {summary['reconcile']['command']}")
    replay_reco = summary.get("incident_replay", {}).get("recommended_commands") or []
    if replay_reco:
        print("- incident_replay_commands:")
        for cmd in replay_reco:
            print(f"  - {cmd}")

if strict and not summary["ready"]:
    raise SystemExit(1)
PY
