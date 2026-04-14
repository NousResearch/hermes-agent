#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
RUNNER_SCRIPT="$ROOT/ops/web_capture/run_macro.sh"
LOCK_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_lock_timeout.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
DOMAIN_GUARD_SCRIPT="$ROOT/ops/openclaw/continuity/web_capture_domain_guard.sh"

STATE_DIR="$ROOT/state/cron_watchdog"
TELEMETRY_DIR="$ROOT/ops/telemetry/textfile"
CONTINUITY_LATEST_DIR="$ROOT/state/continuity/latest"

TIMEOUT_SEC="${OPENCLAW_WEB_CAPTURE_TIMEOUT_SEC:-240}"
GRACE_SEC="${OPENCLAW_WEB_CAPTURE_TIMEOUT_GRACE_SEC:-20}"
EVENT_COOLDOWN_SEC="${OPENCLAW_WEB_CAPTURE_EVENT_COOLDOWN_SEC:-21600}"
MIN_INTERVAL_SEC="${OPENCLAW_WEB_CAPTURE_MIN_INTERVAL_SEC:-1800}"
RUN_TIMEOUT_MS="${OPENCLAW_WEB_CAPTURE_TIMEOUT_MS:-30000}"
SUSTAINED_BOT_THRESHOLD="${OPENCLAW_WEB_CAPTURE_BOT_WALL_SUSTAINED_THRESHOLD:-2}"
BACKOFF_UNIT_SEC="${OPENCLAW_WEB_CAPTURE_BACKOFF_UNIT_SEC:-60}"
REGION_COOLDOWN_SEC="${OPENCLAW_WEB_CAPTURE_REGION_COOLDOWN_SEC:-21600}"
LOGIN_COOLDOWN_SEC="${OPENCLAW_WEB_CAPTURE_LOGIN_COOLDOWN_SEC:-1800}"

DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
INIT_DB_SCRIPT="$ROOT/ops/openclaw/continuity/init_db.sh"
QUEUE_ARB_SCRIPT="$ROOT/ops/openclaw/continuity/queue_arbitrator.sh"
QUEUE_ENABLED="${OPENCLAW_WEB_CAPTURE_QUEUE_ENABLED:-1}"

MACRO_PATH="$ROOT/ops/web_capture/macros/bybit_derivatives_capture.yaml"
MODE="auto"

FORCE_RUN=0
DRY_RUN=0
JSON_OUT=0
KEEP_TAB=0

usage() {
  cat <<'EOF'
Usage: run_web_capture_macro.sh [options]

Low-noise periodic wrapper around ops/web_capture/run_macro.sh.
Default cadence: at most once per macro every 30 minutes.

Options:
  --macro <path>            Macro file (default: canonical bybit macro)
  --mode <auto|fetch|browser>
  --timeout-ms <n>          Runner timeout passed to run_macro.sh
  --force                   Bypass cadence/domain guard gates
  --dry-run                 Print run/skip decision and exit
  --json                    With --dry-run, print machine JSON
  --keep-tab                Pass through to runner (debug only)
  --min-interval-sec <n>    Override cadence minimum interval
  --db <path>               Continuity sqlite path for queue integration
  --no-queue                Disable queue claim/transition integration
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --macro)
      MACRO_PATH="${2:-}"; shift 2 ;;
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --timeout-ms)
      RUN_TIMEOUT_MS="${2:-}"; shift 2 ;;
    --force)
      FORCE_RUN=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --keep-tab)
      KEEP_TAB=1; shift ;;
    --min-interval-sec)
      MIN_INTERVAL_SEC="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --no-queue)
      QUEUE_ENABLED=0; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$MODE" in
  auto|fetch|browser) ;;
  *)
    echo "invalid --mode: $MODE (expected auto|fetch|browser)" >&2
    exit 2 ;;
esac

if ! [[ "$MIN_INTERVAL_SEC" =~ ^[0-9]+$ ]]; then
  echo "invalid --min-interval-sec: $MIN_INTERVAL_SEC (expected integer >= 0)" >&2
  exit 2
fi
if ! [[ "$RUN_TIMEOUT_MS" =~ ^[0-9]+$ ]]; then
  echo "invalid --timeout-ms: $RUN_TIMEOUT_MS (expected integer >= 1000)" >&2
  exit 2
fi
if ! [[ "$QUEUE_ENABLED" =~ ^[01]$ ]]; then
  echo "invalid queue mode (expected 0|1): $QUEUE_ENABLED" >&2
  exit 2
fi
for n in "$SUSTAINED_BOT_THRESHOLD" "$BACKOFF_UNIT_SEC" "$REGION_COOLDOWN_SEC" "$LOGIN_COOLDOWN_SEC"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "invalid numeric web-capture guard value: $n" >&2
    exit 2
  fi
done

macro_slug="$(basename "$MACRO_PATH")"
macro_slug="${macro_slug%.yaml}"
macro_slug="${macro_slug%.yml}"
macro_slug="$(printf '%s' "$macro_slug" | tr -cs 'A-Za-z0-9._-' '_')"
[[ -n "$macro_slug" ]] || macro_slug="web_capture"

macro_meta_line="$(python3 - "$MACRO_PATH" "$macro_slug" <<'PY'
import pathlib
import re
import sys

macro_path = pathlib.Path(sys.argv[1])
macro_slug = sys.argv[2]
domain = macro_slug
retry_backoff = "exp_jitter_2_5_10"
target_url = ""

if macro_path.exists():
    try:
        import yaml
        obj = yaml.safe_load(macro_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            domain = str(obj.get("domain") or domain).strip() or domain
            target_url = str(obj.get("target_url") or "").strip()
            anti = obj.get("anti_bot_policy") if isinstance(obj.get("anti_bot_policy"), dict) else {}
            retry_backoff = str(anti.get("retry_backoff") or retry_backoff).strip() or retry_backoff
    except Exception:
        pass

domain_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", domain).strip("._-") or "domain"
print(f"{domain}\t{domain_slug}\t{retry_backoff}\t{target_url}")
PY
)"
IFS=$'\t' read -r macro_domain macro_domain_slug macro_retry_backoff macro_target_url <<<"$macro_meta_line"
[[ -n "$macro_domain" ]] || macro_domain="$macro_slug"
[[ -n "$macro_domain_slug" ]] || macro_domain_slug="$macro_slug"
[[ -n "$macro_retry_backoff" ]] || macro_retry_backoff="exp_jitter_2_5_10"

DOMAIN_STATE_JSON="$CONTINUITY_LATEST_DIR/web_capture_domain_${macro_domain_slug}.json"
LOGIN_CONTRACT_JSON="$CONTINUITY_LATEST_DIR/web_capture_login_contract_${macro_domain_slug}.json"
LOGIN_CONTRACT_MD="$CONTINUITY_LATEST_DIR/web_capture_login_contract_${macro_domain_slug}.md"

domain_guard_record_json=""
domain_guard_emit_blocker="1"
domain_guard_gate_class=""
domain_guard_summary=""

queue_task_id="continuity:web_capture:${macro_slug}"
queue_agent="web_capture_runner"
queue_claimed=0

LAST_JSON="$STATE_DIR/web_capture_${macro_slug}_last.json"
SCHEDULE_STATE_JSON="$STATE_DIR/web_capture_${macro_slug}_schedule_state.json"

mkdir -p "$STATE_DIR" "$TELEMETRY_DIR" "$CONTINUITY_LATEST_DIR"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.web_capture"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$LAST_JSON"

load_last_attempt_epoch() {
  python3 - "$SCHEDULE_STATE_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)

val = obj.get("last_attempt_epoch")
try:
    n = int(val)
except Exception:
    n = 0
print(max(0, n))
PY
}

extract_run_id() {
  python3 - "$LAST_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("")
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
except Exception:
    print("")
    raise SystemExit(0)

run_id = obj.get("run_id") or ""
print(str(run_id).strip())
PY
}

extract_blocker_summary() {
  python3 - "$LAST_JSON" "$macro_slug" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
macro_slug = sys.argv[2]
if not p.exists():
    print(f"task=web_capture; macro={macro_slug}; reason=blocked")
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
except Exception:
    print(f"task=web_capture; macro={macro_slug}; reason=blocked_unparseable_output")
    raise SystemExit(0)

run_id = obj.get("run_id") or "unknown"
reason = obj.get("result_reason") or obj.get("status") or "blocked"
route = obj.get("route") or "unknown"
gating = obj.get("gating_flags") if isinstance(obj.get("gating_flags"), dict) else {}
signals = gating.get("signals") if isinstance(gating.get("signals"), list) else []
source_url = obj.get("source_url") or ""
signals_part = ",".join(str(s) for s in signals[:6])
print(f"task=web_capture; macro={macro_slug}; reason={reason}; route={route}; run_id={run_id}; signals={signals_part}; url={source_url}")
PY
}

domain_guard_json_field() {
  local raw_json="${1:-}"
  local field="${2:-}"
  local default_value="${3:-}"
  python3 - "$raw_json" "$field" "$default_value" <<'PY'
import json
import sys

raw = sys.argv[1]
field = sys.argv[2]
default = sys.argv[3]

try:
    obj = json.loads(raw)
except Exception:
    print(default)
    raise SystemExit(0)

parts = [p for p in field.split('.') if p]
cur = obj
for part in parts:
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        print(default)
        raise SystemExit(0)

if cur is None:
    print(default)
elif isinstance(cur, bool):
    print("1" if cur else "0")
else:
    print(str(cur))
PY
}

domain_guard_precheck() {
  if [[ ! -x "$DOMAIN_GUARD_SCRIPT" ]]; then
    return 0
  fi
  local -a cmd=(
    "$DOMAIN_GUARD_SCRIPT" precheck
    --state "$DOMAIN_STATE_JSON"
    --domain "$macro_domain"
  )
  if [[ "$FORCE_RUN" -eq 1 ]]; then
    cmd+=(--force)
  fi
  cmd+=(--json)
  "${cmd[@]}"
}

domain_guard_record() {
  if [[ ! -x "$DOMAIN_GUARD_SCRIPT" ]]; then
    return 127
  fi
  "$DOMAIN_GUARD_SCRIPT" record \
    --root "$ROOT" \
    --state "$DOMAIN_STATE_JSON" \
    --last-json "$LAST_JSON" \
    --domain "$macro_domain" \
    --domain-slug "$macro_domain_slug" \
    --macro-slug "$macro_slug" \
    --macro-path "$MACRO_PATH" \
    --target-url "$macro_target_url" \
    --retry-backoff "$macro_retry_backoff" \
    --sustained-bot-threshold "$SUSTAINED_BOT_THRESHOLD" \
    --backoff-unit-sec "$BACKOFF_UNIT_SEC" \
    --region-cooldown-sec "$REGION_COOLDOWN_SEC" \
    --login-cooldown-sec "$LOGIN_COOLDOWN_SEC" \
    --login-contract-json "$LOGIN_CONTRACT_JSON" \
    --login-contract-md "$LOGIN_CONTRACT_MD" \
    --json
}

write_schedule_state() {
  local rc="$1"
  local status="$2"
  local run_id="$3"
  python3 - "$SCHEDULE_STATE_JSON" "$rc" "$status" "$run_id" "$macro_slug" "$MODE" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
status = sys.argv[3]
run_id = sys.argv[4]
macro_slug = sys.argv[5]
mode = sys.argv[6]

now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
payload = {
    "schema_version": "openclaw.web_capture.schedule.v1",
    "macro_slug": macro_slug,
    "mode": mode,
    "last_attempt_epoch": int(now.timestamp()),
    "last_attempt_iso": now.isoformat().replace("+00:00", "Z"),
    "last_exit_code": rc,
    "last_status": status,
    "last_run_id": run_id or None,
}

path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

queue_upsert_task() {
  python3 - "$ROOT" "$DB_PATH" "$queue_task_id" "$macro_slug" "$MACRO_PATH" "$LAST_JSON" "$SCHEDULE_STATE_JSON" "$DOMAIN_STATE_JSON" "$LOGIN_CONTRACT_JSON" "$LOGIN_CONTRACT_MD" "$DOMAIN_GUARD_SCRIPT" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sqlite3
import sys


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def artifact_id(task_id: str, artifact_path: str, artifact_type: str) -> str:
    seed = f"{task_id}|{artifact_type}|{artifact_path}"
    return "tart_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


root = pathlib.Path(sys.argv[1]).resolve()
db_path = pathlib.Path(sys.argv[2])
task_id = sys.argv[3]
macro_slug = sys.argv[4]
macro_path = pathlib.Path(sys.argv[5])
last_json = pathlib.Path(sys.argv[6])
schedule_json = pathlib.Path(sys.argv[7])
domain_state_json = pathlib.Path(sys.argv[8])
login_contract_json = pathlib.Path(sys.argv[9])
login_contract_md = pathlib.Path(sys.argv[10])
domain_guard_script = pathlib.Path(sys.argv[11])

con = sqlite3.connect(db_path)
cur = con.cursor()
ts = now_iso()

cur.execute(
    """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, 'QUEUED', 'validator', NULL, 0, 5, NULL, NULL, ?, ?)
ON CONFLICT(task_id) DO UPDATE SET
  source = excluded.source,
  title = excluded.title,
  acceptance_criteria = excluded.acceptance_criteria,
  status = COALESCE(NULLIF(TRIM(work_queue.status), ''), 'QUEUED'),
  role_required = CASE
    WHEN TRIM(COALESCE(work_queue.role_required, '')) <> '' THEN work_queue.role_required
    ELSE 'validator'
  END,
  assigned_agent = CASE WHEN work_queue.status = 'RUNNING' THEN work_queue.assigned_agent ELSE NULL END,
  last_error_log = NULL,
  cooldown_until = NULL,
  updated_at = excluded.updated_at
""",
    (
        task_id,
        "continuity_ops",
        f"Deterministic web capture macro ({macro_slug})",
        "Produce deterministic bundle (index/screenshot/dom/css/trace + manifest.lock + gate/failure packets) with artifact quality gates, per-domain backoff/login guard state, and queue traceability.",
        ts,
        ts,
    ),
)

for file_path in [
    str(macro_path),
    str(last_json),
    str(schedule_json),
    str(domain_state_json),
    str(login_contract_json),
    str(login_contract_md),
]:
    rel = file_path
    try:
        rel = str(pathlib.Path(file_path).resolve().relative_to(root))
    except Exception:
        pass
    cur.execute(
        """
INSERT INTO task_file_targets (task_id, file_path, lock_mode, created_at)
VALUES (?, ?, 'exclusive', ?)
ON CONFLICT(task_id, file_path) DO UPDATE SET
  lock_mode = excluded.lock_mode,
  created_at = excluded.created_at
""",
        (task_id, rel, ts),
    )

cur.execute("DELETE FROM task_artifacts WHERE task_id = ?", (task_id,))
for rel, a_type in [
    ("ops/web_capture/run_macro.sh", "runtime_script"),
    ("ops/openclaw/run_web_capture_macro.sh", "runtime_wrapper"),
    (str(domain_guard_script), "runtime_guard"),
    (str(macro_path), "macro"),
    (str(last_json), "runtime_state"),
    (str(schedule_json), "runtime_state"),
    (str(domain_state_json), "runtime_state"),
    (str(login_contract_json), "runtime_state"),
    (str(login_contract_md), "runtime_state"),
]:
    try:
        rel_norm = str(pathlib.Path(rel).resolve().relative_to(root))
    except Exception:
        rel_norm = rel
    cur.execute(
        """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, NULL, ?, ?)
""",
        (
            artifact_id(task_id, rel_norm, a_type),
            task_id,
            a_type,
            rel_norm,
            json.dumps({"macro_slug": macro_slug}, ensure_ascii=False),
            ts,
        ),
    )

con.commit()
con.close()
PY
}

queue_claim_task() {
  local ttl=$(( TIMEOUT_SEC + GRACE_SEC + 120 ))
  OPENCLAW_INTERNAL_MUTATION=1 \
  OPENCLAW_INTERNAL_MUTATION_CALLSITE="run_web_capture_macro.sh:queue_claim_task" \
    "$QUEUE_ARB_SCRIPT" claim \
      --agent "$queue_agent" \
      --actor-role validator \
      --task-id "$queue_task_id" \
      --lock-ttl-sec "$ttl" \
      --json
}

extract_queue_evidence_ref() {
  python3 - "$ROOT" "$LAST_JSON" "$MACRO_PATH" "$DOMAIN_STATE_JSON" "$LOGIN_CONTRACT_JSON" "$LOGIN_CONTRACT_MD" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
last_json = pathlib.Path(sys.argv[2])
macro_path = pathlib.Path(sys.argv[3])
domain_state = pathlib.Path(sys.argv[4])
login_contract_json = pathlib.Path(sys.argv[5])
login_contract_md = pathlib.Path(sys.argv[6])

refs = []

def add_ref(path_like: str):
    p = pathlib.Path(str(path_like or '').strip())
    if not str(p):
        return
    if p.exists():
        try:
            refs.append(str(p.resolve().relative_to(root)))
        except Exception:
            refs.append(str(p.resolve()))

add_ref(str(last_json))
add_ref(str(macro_path))
add_ref(str(domain_state))
add_ref(str(login_contract_json))
add_ref(str(login_contract_md))

if last_json.exists():
    try:
        obj = json.loads(last_json.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        obj = {}
    run_dir = pathlib.Path(str(obj.get('run_dir') or '').strip())
    if run_dir.exists() and run_dir.is_dir():
        for name in (
            'index.json',
            'execution_trace.json',
            'dom_snapshot.html',
            'screenshot.png',
            'css_tokens.json',
            'manifest.lock.json',
            'gate.classification.json',
            'failure.packet.json',
            'trace/trace.jsonl',
            'trace/trace.summary.json',
        ):
            add_ref(str(run_dir / name))

out = []
seen = set()
for r in refs:
    s = str(r).strip()
    if not s or s in seen:
        continue
    seen.add(s)
    out.append(s)
print('|'.join(out[:12]))
PY
}

queue_transition_task() {
  local to_status="$1"
  local reason="$2"
  local actor_role="$3"
  local evidence_ref="${4:-}"
  local best_effort="${5:-1}"
  local allow_any_transition="${6:-0}"

  local cmd=(
    env
    "OPENCLAW_INTERNAL_MUTATION=1"
    "OPENCLAW_INTERNAL_MUTATION_CALLSITE=run_web_capture_macro.sh:queue_transition_task"
    "$QUEUE_ARB_SCRIPT" transition
    --task-id "$queue_task_id"
    --to-status "$to_status"
    --actor-role "$actor_role"
    --reason "$reason"
    --evidence-ref "$evidence_ref"
    --release-locks
    --json
  )
  if [[ "$allow_any_transition" -eq 1 ]]; then
    cmd+=(--allow-any-transition)
  fi

  if [[ "$best_effort" -eq 1 ]]; then
    "${cmd[@]}" >/dev/null 2>&1 || true
  else
    "${cmd[@]}" >/dev/null 2>&1
  fi
}

queue_current_status() {
  python3 - "$DB_PATH" "$queue_task_id" <<'PY'
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
row = cur.execute("SELECT status FROM work_queue WHERE task_id = ?", (sys.argv[2],)).fetchone()
con.close()
status = str(row[0] or '').strip().upper() if row else ''
print(status)
PY
}

queue_force_role_required() {
  local role_required="$1"
  python3 - "$DB_PATH" "$queue_task_id" "$role_required" <<'PY'
import datetime as dt
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
cur.execute(
    "UPDATE work_queue SET role_required = ?, updated_at = ? WHERE task_id = ? AND status = 'QUEUED'",
    (sys.argv[3], now, sys.argv[2]),
)
con.commit()
con.close()
PY
}

queue_reopen_task() {
  local current_status
  current_status="$(queue_current_status)"
  case "$current_status" in
    "")
      return 0 ;;
    "QUEUED")
      queue_force_role_required "validator"
      return 0 ;;
    "RUNNING")
      return 0 ;;
  esac

  queue_transition_task "QUEUED" "web_capture_requeue_for_new_run" "validator" "$(extract_queue_evidence_ref)" 0 1
  queue_force_role_required "validator"
}

now_epoch="$(date +%s)"
last_attempt_epoch="$(load_last_attempt_epoch)"

domain_guard_precheck_json="{}"
if [[ -x "$DOMAIN_GUARD_SCRIPT" ]]; then
  set +e
  domain_guard_precheck_json="$(domain_guard_precheck 2>/dev/null)"
  domain_guard_precheck_rc=$?
  set -e
  if [[ "$domain_guard_precheck_rc" -ne 0 || -z "$domain_guard_precheck_json" ]]; then
    domain_guard_precheck_json='{}'
  fi
fi

domain_guard_allowed="$(domain_guard_json_field "$domain_guard_precheck_json" "run_allowed" "1")"
domain_guard_reason="$(domain_guard_json_field "$domain_guard_precheck_json" "reason" "ready")"
domain_guard_cooldown_remaining_sec="$(domain_guard_json_field "$domain_guard_precheck_json" "cooldown_remaining_sec" "0")"
domain_guard_operator_required="$(domain_guard_json_field "$domain_guard_precheck_json" "operator_action_required" "0")"
domain_guard_contract_json="$(domain_guard_json_field "$domain_guard_precheck_json" "operator_contract_json" "")"

run_allowed=1
skip_reason=""
age_sec=0
if [[ "$FORCE_RUN" -ne 1 && "$MIN_INTERVAL_SEC" -gt 0 && "$last_attempt_epoch" -gt 0 ]]; then
  age_sec=$(( now_epoch - last_attempt_epoch ))
  if (( age_sec < MIN_INTERVAL_SEC )); then
    run_allowed=0
    skip_reason="min_interval_not_elapsed"
  fi
fi

if [[ "$domain_guard_allowed" != "1" ]]; then
  run_allowed=0
  skip_reason="$domain_guard_reason"
fi

if [[ "$DRY_RUN" -eq 1 || "$run_allowed" -eq 0 ]]; then
  if [[ "$run_allowed" -eq 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 || "$JSON_OUT" -eq 1 ]]; then
      if [[ "$JSON_OUT" -eq 1 ]]; then
        printf '{"ok":true,"run_allowed":false,"reason":"%s","age_sec":%s,"min_interval_sec":%s,"last_attempt_epoch":%s,"macro":"%s","domain":"%s","domain_guard":{"reason":"%s","cooldown_remaining_sec":%s,"operator_action_required":%s,"operator_contract_json":"%s"}}\n' \
          "$skip_reason" "$age_sec" "$MIN_INTERVAL_SEC" "$last_attempt_epoch" "$macro_slug" "$macro_domain" "$domain_guard_reason" "$domain_guard_cooldown_remaining_sec" "$domain_guard_operator_required" "$domain_guard_contract_json"
      else
        printf 'SKIP: web capture gate reason=%s macro=%s domain=%s cooldown_remaining=%ss\n' "$skip_reason" "$macro_slug" "$macro_domain" "$domain_guard_cooldown_remaining_sec"
      fi
    fi
    exit 0
  fi

  if [[ "$JSON_OUT" -eq 1 ]]; then
    printf '{"ok":true,"run_allowed":true,"reason":"dry_run","min_interval_sec":%s,"force":%s,"macro":"%s","mode":"%s","domain":"%s","domain_guard":{"reason":"%s","cooldown_remaining_sec":%s,"operator_action_required":%s,"operator_contract_json":"%s"}}\n' "$MIN_INTERVAL_SEC" "$FORCE_RUN" "$macro_slug" "$MODE" "$macro_domain" "$domain_guard_reason" "$domain_guard_cooldown_remaining_sec" "$domain_guard_operator_required" "$domain_guard_contract_json"
  else
    printf 'READY: web capture runner would execute now (dry-run) macro=%s mode=%s domain=%s\n' "$macro_slug" "$MODE" "$macro_domain"
  fi
  exit 0
fi

if [[ ! -x "$RUNNER_SCRIPT" ]]; then
  openclaw_watchdog_route_blocker "runner_missing" "task=web_capture; macro=${macro_slug}; reason=runner_missing"
  exit 0
fi

if [[ ! -x "$LOCK_WRAP" ]]; then
  openclaw_watchdog_route_blocker "wrapper_dependency_missing" "task=web_capture; macro=${macro_slug}; reason=wrapper_dependency_missing"
  exit 0
fi

if [[ ! -x "$DOMAIN_GUARD_SCRIPT" ]]; then
  openclaw_watchdog_route_blocker "domain_guard_missing" "task=web_capture; macro=${macro_slug}; reason=domain_guard_missing"
  exit 0
fi

if [[ "$QUEUE_ENABLED" -eq 1 ]]; then
  if [[ ! -x "$INIT_DB_SCRIPT" || ! -x "$QUEUE_ARB_SCRIPT" ]]; then
    openclaw_watchdog_route_blocker "queue_integration_missing" "task=web_capture; macro=${macro_slug}; reason=queue_integration_missing"
    exit 0
  fi

  if ! OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$INIT_DB_SCRIPT" >/dev/null 2>&1; then
    openclaw_watchdog_route_blocker "queue_init_failed" "task=web_capture; macro=${macro_slug}; reason=queue_init_failed"
    exit 0
  fi
  if ! queue_upsert_task >/dev/null 2>&1; then
    openclaw_watchdog_route_blocker "queue_upsert_${macro_slug}" "task=web_capture; macro=${macro_slug}; reason=queue_upsert_failed"
    exit 0
  fi
  if ! queue_reopen_task >/dev/null 2>&1; then
    openclaw_watchdog_route_blocker "queue_reopen_${macro_slug}" "task=web_capture; macro=${macro_slug}; reason=queue_reopen_failed"
    exit 0
  fi

  set +e
  queue_claim_output="$(queue_claim_task 2>&1)"
  queue_claim_rc=$?
  set -e

  if [[ "$queue_claim_rc" -ne 0 ]]; then
    claim_reason="$(python3 - "$queue_claim_output" <<'PY'
import json
import sys
raw = sys.argv[1]
reason = "queue_claim_failed"
try:
    obj = json.loads(raw)
    if isinstance(obj, dict):
        if isinstance(obj.get("error"), str) and obj.get("error"):
            reason = str(obj.get("error"))
        skipped = obj.get("skipped") if isinstance(obj.get("skipped"), list) else []
        if skipped and isinstance(skipped[0], dict) and skipped[0].get("reason"):
            reason = str(skipped[0].get("reason"))
except Exception:
    pass
print(reason)
PY
)"
    if [[ "$claim_reason" == "no_claimable_task" ]]; then
      exit 0
    fi
    openclaw_watchdog_route_blocker "queue_claim_${macro_slug}" "task=web_capture; macro=${macro_slug}; reason=${claim_reason}"
    exit 0
  fi
  queue_claimed=1
fi

runner_cmd=(
  bash "$RUNNER_SCRIPT"
  --macro "$MACRO_PATH"
  --mode "$MODE"
  --timeout-ms "$RUN_TIMEOUT_MS"
  --json
)
if [[ "$KEEP_TAB" -eq 1 ]]; then
  runner_cmd+=(--keep-tab)
fi

set +e
output="$($LOCK_WRAP \
  --lock-name "web_capture_${macro_slug}" \
  --lock-dir "$STATE_DIR/locks" \
  --timeout-sec "$TIMEOUT_SEC" \
  --grace-sec "$GRACE_SEC" \
  --busy-exit-code 75 \
  --emit-blocker \
  --soft-timeout \
  -- "${runner_cmd[@]}" 2>&1)"
rc=$?
set -e

if [[ -n "$output" ]]; then
  printf '%s\n' "$output" > "$LAST_JSON"
fi

if [[ "$rc" -eq 75 ]]; then
  if [[ "$queue_claimed" -eq 1 ]]; then
    queue_transition_task "QUEUED" "web_capture_lock_busy" "sre_watchdog" "$(extract_queue_evidence_ref)"
  fi
  exit 0
fi

run_id="$(extract_run_id)"
status="wrapper_exit"
if [[ "$rc" -eq 0 ]]; then
  status="done"
elif [[ "$rc" -eq 2 ]]; then
  status="blocked"
fi
write_schedule_state "$rc" "$status" "$run_id"

if [[ -x "$DOMAIN_GUARD_SCRIPT" && -s "$LAST_JSON" ]]; then
  set +e
  domain_guard_record_json="$(domain_guard_record 2>/dev/null)"
  domain_guard_record_rc=$?
  set -e
  if [[ "$domain_guard_record_rc" -eq 0 && -n "$domain_guard_record_json" ]]; then
    domain_guard_emit_blocker="$(domain_guard_json_field "$domain_guard_record_json" "emit_blocker" "1")"
    domain_guard_gate_class="$(domain_guard_json_field "$domain_guard_record_json" "gate_class" "")"
    domain_guard_summary="$(domain_guard_json_field "$domain_guard_record_json" "blocker_summary" "")"
  fi
fi

evidence_ref="$(extract_queue_evidence_ref)"
if [[ "$queue_claimed" -eq 1 ]]; then
  if [[ "$rc" -eq 0 ]]; then
    queue_transition_task "DONE" "web_capture_completed" "validator" "$evidence_ref"
  elif [[ "$rc" -eq 2 ]]; then
    queue_reason="web_capture_blocked"
    if [[ -n "$domain_guard_gate_class" ]]; then
      queue_reason="web_capture_blocked_${domain_guard_gate_class}"
    fi
    queue_transition_task "BLOCKED" "$queue_reason" "validator" "$evidence_ref"
  else
    queue_transition_task "FAILED" "web_capture_wrapper_exit_${rc}" "sre_watchdog" "$evidence_ref"
  fi
fi

if [[ "$rc" -eq 0 ]]; then
  now_epoch="$(date +%s)"
  printf 'openclaw_web_capture_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_web_capture_last_success_epoch.prom"
  exit 0
fi

if [[ "$rc" -eq 2 ]]; then
  if [[ "$domain_guard_emit_blocker" != "1" ]]; then
    exit 0
  fi
  summary="$domain_guard_summary"
  if [[ -z "$summary" ]]; then
    summary="$(extract_blocker_summary)"
  fi
  key="macro_blocked_${macro_slug}"
  if [[ -n "$domain_guard_gate_class" ]]; then
    key="macro_blocked_${macro_slug}_${domain_guard_gate_class}"
  fi
  openclaw_watchdog_route_blocker "$key" "$summary" "$LAST_JSON"
  exit 0
fi

openclaw_watchdog_route_blocker "wrapper_exit_${macro_slug}" "task=web_capture; macro=${macro_slug}; reason=wrapper_exit_${rc}" "$LAST_JSON"
exit 0
