#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
STATE_FILE="${OPENCLAW_GATEWAY_WATCHDOG_STATE_FILE:-$ROOT/state/cron_watchdog/gateway_health_watchdog.json}"
FAIL_WINDOW_MIN="${OPENCLAW_GATEWAY_WATCHDOG_FAIL_WINDOW_MIN:-10}"
NOTIFY_AFTER="${OPENCLAW_GATEWAY_WATCHDOG_NOTIFY_AFTER:-3}"
NOTIFY_REMINDER_SEC="${OPENCLAW_GATEWAY_WATCHDOG_NOTIFY_REMINDER_SEC:-1800}"
RESTART_COOLDOWN_SEC="${OPENCLAW_GATEWAY_WATCHDOG_RESTART_COOLDOWN_SEC:-900}"
EVENT_ROUTER="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
EVENT_COOLDOWN_SEC="${OPENCLAW_GATEWAY_WATCHDOG_EVENT_COOLDOWN_SEC:-1800}"

mkdir -p "$(dirname "$STATE_FILE")"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.gateway_health"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$STATE_FILE"

set +e
py_output="$(python3 - "$STATE_FILE" "$FAIL_WINDOW_MIN" "$NOTIFY_AFTER" "$NOTIFY_REMINDER_SEC" "$RESTART_COOLDOWN_SEC" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

state_file = pathlib.Path(sys.argv[1])
fail_window_min = max(1, int(sys.argv[2]))
notify_after = max(1, int(sys.argv[3]))
notify_reminder_sec = max(60, int(sys.argv[4]))
restart_cooldown_sec = max(60, int(sys.argv[5]))


def load_state() -> dict:
    if not state_file.exists():
        return {
            "consecutive_failures": 0,
            "first_failure_epoch": 0,
            "last_failure_epoch": 0,
            "last_failure_reason": "",
            "last_restart_epoch": 0,
            "last_notified_epoch": 0,
            "last_notified_signature": "",
            "updated_epoch": 0,
        }
    try:
        obj = json.loads(state_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {
        "consecutive_failures": 0,
        "first_failure_epoch": 0,
        "last_failure_epoch": 0,
        "last_failure_reason": "",
        "last_restart_epoch": 0,
        "last_notified_epoch": 0,
        "last_notified_signature": "",
        "updated_epoch": 0,
    }


def save_state(state: dict) -> None:
    state["updated_epoch"] = int(time.time())
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, state_file)


def run_cmd(args, timeout=20):
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    except Exception as exc:
        return 99, "", str(exc)


def check_endpoint(base_http: str):
    # Prefer explicit API endpoints; if absent, fall back to /health if non-404.
    for path in ("/api/health", "/healthz", "/health"):
        url = f"{base_http}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "openclaw-gateway-watchdog/1"})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                code = int(getattr(resp, "status", 200))
                if code == 200:
                    return True, url, "ok"
                return False, url, f"http_{code}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return False, url, f"http_{e.code}"
        except Exception as exc:
            return False, url, f"endpoint_error:{exc}"
    return True, "", "unavailable"


def check_gateway_health():
    rc, out, err = run_cmd(["openclaw", "gateway", "status", "--json"], timeout=25)
    if rc != 0:
        reason = (err or out or "gateway_status_failed").strip().splitlines()[-1][:240]
        return False, f"gateway_status_failed:{reason}"

    try:
        data = json.loads(out)
    except Exception as exc:
        return False, f"gateway_status_json_invalid:{exc}"

    runtime_status = str(((data.get("service") or {}).get("runtime") or {}).get("status") or "")
    rpc_ok = bool(((data.get("rpc") or {}).get("ok")))
    host = str(((data.get("gateway") or {}).get("bindHost") or "127.0.0.1"))
    port = int(((data.get("gateway") or {}).get("port") or 18789))

    if runtime_status.lower() != "running":
        return False, f"runtime_not_running:{runtime_status or 'unknown'}"
    if not rpc_ok:
        return False, "rpc_probe_failed"

    endpoint_ok, endpoint_url, endpoint_reason = check_endpoint(f"http://{host}:{port}")
    if not endpoint_ok:
        return False, f"endpoint_failed:{endpoint_reason}:{endpoint_url}"

    return True, "ok"


state = load_state()
now = int(time.time())

ok, reason = check_gateway_health()

if ok:
    if int(state.get("consecutive_failures", 0) or 0) != 0:
        state["consecutive_failures"] = 0
        state["first_failure_epoch"] = 0
        state["last_failure_epoch"] = 0
        state["last_failure_reason"] = ""
    save_state(state)
    raise SystemExit(0)

consecutive = int(state.get("consecutive_failures", 0) or 0) + 1
first_fail = int(state.get("first_failure_epoch", 0) or 0)
if first_fail <= 0:
    first_fail = now

state["consecutive_failures"] = consecutive
state["first_failure_epoch"] = first_fail
state["last_failure_epoch"] = now
state["last_failure_reason"] = reason

unresponsive_sec = max(0, now - first_fail)
restart_attempted = False

if unresponsive_sec >= fail_window_min * 60:
    last_restart = int(state.get("last_restart_epoch", 0) or 0)
    if now - last_restart >= restart_cooldown_sec:
        restart_attempted = True
        state["last_restart_epoch"] = now
        run_cmd(["openclaw", "gateway", "restart"], timeout=90)
        time.sleep(5)
        ok2, reason2 = check_gateway_health()
        if ok2:
            state["consecutive_failures"] = 0
            state["first_failure_epoch"] = 0
            state["last_failure_epoch"] = 0
            state["last_failure_reason"] = ""
            save_state(state)
            raise SystemExit(0)
        reason = f"post_restart_unhealthy:{reason2}"
        state["last_failure_reason"] = reason

should_notify = False
signature = f"{reason}|restart={int(restart_attempted)}"
if consecutive >= notify_after:
    last_sig = str(state.get("last_notified_signature") or "")
    last_notified = int(state.get("last_notified_epoch", 0) or 0)
    if signature != last_sig or (now - last_notified) >= notify_reminder_sec:
        should_notify = True
        state["last_notified_signature"] = signature
        state["last_notified_epoch"] = now

save_state(state)

if should_notify:
    print(
        "BLOCKER: gateway_watchdog "
        f"failures={consecutive}; "
        f"unresponsive_sec={unresponsive_sec}; "
        f"restart_attempted={int(restart_attempted)}; "
        f"reason={reason[:200]}"
    )

raise SystemExit(0)
PY
)"
py_rc=$?
set -e

if [[ "$py_rc" -ne 0 ]]; then
  openclaw_watchdog_route_blocker "gateway_watchdog_script_error" "task=gateway_watchdog; reason=script_error_${py_rc}" "$STATE_FILE"
  exit 0
fi

if [[ -z "$py_output" ]]; then
  exit 0
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  if [[ "$line" == BLOCKER:* ]]; then
    openclaw_watchdog_route_blocker "gateway_watchdog" "$(openclaw_blocker_summary_from_line "$line")" "$STATE_FILE"
  else
    printf '%s\n' "$line"
  fi
done <<< "$py_output"

exit 0
