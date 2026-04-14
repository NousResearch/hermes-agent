#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-}"
if [[ -n "$CMD" ]]; then
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
DEFAULT_ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"

usage() {
  cat <<'EOF'
Usage: web_capture_domain_guard.sh <precheck|record> [options]

Deterministic per-domain backoff/login-wall state guard for web-capture runtime.

Commands:
  precheck   Read domain state and decide whether execution is allowed now.
  record     Update domain state from latest run output and emit blocker-routing decision.

Common options:
  --state <path>        Domain state json path.
  --domain <name>       Domain key (example.com).
  --json                Print JSON output.
EOF
}

if [[ -z "$CMD" ]]; then
  usage >&2
  exit 2
fi

case "$CMD" in
  precheck)
    STATE_PATH=""
    DOMAIN=""
    FORCE=0
    JSON_OUT=0

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --state)
          STATE_PATH="${2:-}"; shift 2 ;;
        --domain)
          DOMAIN="${2:-}"; shift 2 ;;
        --force)
          FORCE=1; shift ;;
        --json)
          JSON_OUT=1; shift ;;
        -h|--help)
          usage
          exit 0 ;;
        *)
          echo "unknown argument: $1" >&2
          exit 2 ;;
      esac
    done

    if [[ -z "$STATE_PATH" || -z "$DOMAIN" ]]; then
      echo "missing --state/--domain" >&2
      exit 2
    fi

    python3 - "$STATE_PATH" "$DOMAIN" "$FORCE" "$JSON_OUT" <<'PY'
import datetime as dt
import json
import pathlib
import sys
from typing import Any, Dict

state_path = pathlib.Path(sys.argv[1])
domain = str(sys.argv[2] or "").strip().lower()
force = bool(int(sys.argv[3]))
json_out = bool(int(sys.argv[4]))


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


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
    return out


state: Dict[str, Any] = {}
if state_path.exists():
    try:
        obj = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            state = obj
    except Exception:
        state = {}

cooldown_until = parse_iso(state.get("cooldown_until"))
operator_required = bool(state.get("operator_action_required"))
operator_contract = str(state.get("operator_contract_json") or "").strip() or None

reason = "ready"
run_allowed = True
remaining_sec = 0

if not force and operator_required:
    run_allowed = False
    reason = "operator_assisted_login_required"
elif not force and cooldown_until is not None:
    rem = int((cooldown_until - now_utc()).total_seconds())
    if rem > 0:
        run_allowed = False
        reason = str(state.get("cooldown_reason") or "domain_backoff_active")
        remaining_sec = rem

payload = {
    "ok": True,
    "schema_version": "openclaw.web_capture.domain_guard.precheck.v1",
    "checked_at": now_iso(),
    "domain": domain,
    "run_allowed": run_allowed,
    "reason": reason,
    "force": force,
    "cooldown_until": cooldown_until.isoformat().replace("+00:00", "Z") if cooldown_until else None,
    "cooldown_remaining_sec": remaining_sec,
    "operator_action_required": operator_required,
    "operator_contract_json": operator_contract,
    "state_path": str(state_path),
}

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
else:
    if run_allowed:
        print(f"WEB_DOMAIN_GUARD: ready domain={domain}")
    else:
        print(f"WEB_DOMAIN_GUARD: blocked domain={domain} reason={reason} remaining={remaining_sec}s")
PY
    ;;

  record)
    ROOT="$DEFAULT_ROOT"
    STATE_PATH=""
    LAST_JSON=""
    DOMAIN=""
    DOMAIN_SLUG=""
    MACRO_SLUG=""
    MACRO_PATH=""
    TARGET_URL=""
    RETRY_BACKOFF=""
    SUSTAINED_BOT_THRESHOLD=2
    BACKOFF_UNIT_SEC=60
    REGION_COOLDOWN_SEC=21600
    LOGIN_COOLDOWN_SEC=1800
    LOGIN_CONTRACT_JSON=""
    LOGIN_CONTRACT_MD=""
    JSON_OUT=0

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --root)
          ROOT="${2:-}"; shift 2 ;;
        --state)
          STATE_PATH="${2:-}"; shift 2 ;;
        --last-json)
          LAST_JSON="${2:-}"; shift 2 ;;
        --domain)
          DOMAIN="${2:-}"; shift 2 ;;
        --domain-slug)
          DOMAIN_SLUG="${2:-}"; shift 2 ;;
        --macro-slug)
          MACRO_SLUG="${2:-}"; shift 2 ;;
        --macro-path)
          MACRO_PATH="${2:-}"; shift 2 ;;
        --target-url)
          TARGET_URL="${2:-}"; shift 2 ;;
        --retry-backoff)
          RETRY_BACKOFF="${2:-}"; shift 2 ;;
        --sustained-bot-threshold)
          SUSTAINED_BOT_THRESHOLD="${2:-}"; shift 2 ;;
        --backoff-unit-sec)
          BACKOFF_UNIT_SEC="${2:-}"; shift 2 ;;
        --region-cooldown-sec)
          REGION_COOLDOWN_SEC="${2:-}"; shift 2 ;;
        --login-cooldown-sec)
          LOGIN_COOLDOWN_SEC="${2:-}"; shift 2 ;;
        --login-contract-json)
          LOGIN_CONTRACT_JSON="${2:-}"; shift 2 ;;
        --login-contract-md)
          LOGIN_CONTRACT_MD="${2:-}"; shift 2 ;;
        --json)
          JSON_OUT=1; shift ;;
        -h|--help)
          usage
          exit 0 ;;
        *)
          echo "unknown argument: $1" >&2
          exit 2 ;;
      esac
    done

    if [[ -z "$STATE_PATH" || -z "$LAST_JSON" || -z "$DOMAIN" || -z "$MACRO_SLUG" ]]; then
      echo "missing required args for record" >&2
      exit 2
    fi

    python3 - "$ROOT" "$STATE_PATH" "$LAST_JSON" "$DOMAIN" "$DOMAIN_SLUG" "$MACRO_SLUG" "$MACRO_PATH" "$TARGET_URL" "$RETRY_BACKOFF" "$SUSTAINED_BOT_THRESHOLD" "$BACKOFF_UNIT_SEC" "$REGION_COOLDOWN_SEC" "$LOGIN_COOLDOWN_SEC" "$LOGIN_CONTRACT_JSON" "$LOGIN_CONTRACT_MD" "$JSON_OUT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import sys
from typing import Any, Dict, List

root = pathlib.Path(sys.argv[1]).resolve()
state_path = pathlib.Path(sys.argv[2])
last_json_path = pathlib.Path(sys.argv[3])
domain = str(sys.argv[4] or "").strip().lower()
domain_slug = str(sys.argv[5] or "").strip() or re.sub(r"[^A-Za-z0-9._-]+", "_", domain).strip("._-") or "domain"
macro_slug = str(sys.argv[6] or "").strip()
macro_path = pathlib.Path(str(sys.argv[7] or "").strip()) if str(sys.argv[7] or "").strip() else None
target_url = str(sys.argv[8] or "").strip()
retry_backoff = str(sys.argv[9] or "").strip()
sustained_bot_threshold = max(1, int(sys.argv[10] or 2))
backoff_unit_sec = max(1, int(sys.argv[11] or 60))
region_cooldown_sec = max(60, int(sys.argv[12] or 21600))
login_cooldown_sec = max(60, int(sys.argv[13] or 1800))
login_contract_json = pathlib.Path(str(sys.argv[14] or "").strip()) if str(sys.argv[14] or "").strip() else None
login_contract_md = pathlib.Path(str(sys.argv[15] or "").strip()) if str(sys.argv[15] or "").strip() else None
json_out = bool(int(sys.argv[16]))


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


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
    return out


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def rel_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def shlex_quote(value: str) -> str:
    return shlex.quote(str(value))


def parse_schedule(backoff_policy: str, unit_sec: int) -> List[int]:
    nums = [int(x) for x in re.findall(r"(\d+)", backoff_policy or "") if int(x) > 0]
    if not nums:
        nums = [2, 5, 10]
    out = [max(5, min(86400, int(n) * unit_sec)) for n in nums]
    seen = []
    for v in out:
        if v not in seen:
            seen.append(v)
    return seen[:12]


def derive_gate_class(payload: Dict[str, Any]) -> str:
    gc = payload.get("gate_classification") if isinstance(payload.get("gate_classification"), dict) else {}
    gate_class = str(gc.get("class") or "").strip()
    if gate_class:
        return gate_class
    gating = payload.get("gating_flags") if isinstance(payload.get("gating_flags"), dict) else {}
    if gating.get("bot_wall"):
        return "bot_wall"
    if gating.get("login_wall"):
        return "login_wall"
    if gating.get("region_wall"):
        return "region_block"
    status = str(payload.get("status") or "").strip().lower()
    if status == "ok":
        return "pass"
    if status == "blocked":
        return "runtime_failure"
    return "runtime_failure"


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def write_login_contract(opened: bool, run_payload: Dict[str, Any], state: Dict[str, Any], gate_class: str, signals: List[str]) -> Dict[str, Any]:
    if login_contract_json is None or login_contract_md is None:
        return {}

    now = now_iso()
    contract = load_json(login_contract_json)
    if not contract:
        contract = {
            "schema_version": "openclaw.web_capture.login_wall_contract.v1",
            "domain": domain,
            "domain_slug": domain_slug,
            "macro_slug": macro_slug,
            "target_url": target_url,
            "opened_at": now,
            "status": "open",
        }

    run_id = str(run_payload.get("run_id") or state.get("last_run_id") or "").strip() or None

    if opened:
        resume_command = f"bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_web_capture_macro.sh --macro {str(macro_path) if macro_path else '<macro_path>'} --mode browser --force"
        inspect_contract_cmd = f"cat {shlex_quote(str(login_contract_md.resolve()))}" if login_contract_md is not None else None
        inspect_packet_cmd = f"cat {shlex_quote(str(last_json_path.resolve()))}"
        recommended_commands = [cmd for cmd in [inspect_contract_cmd, resume_command, inspect_packet_cmd] if cmd]
        recommended_steps = []
        if inspect_contract_cmd:
            recommended_steps.append(
                {
                    "step_id": "inspect_login_operator_contract",
                    "summary": "Read the login-wall operator contract for exact bounded actions.",
                    "command": inspect_contract_cmd,
                }
            )
        recommended_steps.append(
            {
                "step_id": "resume_web_capture_after_manual_login",
                "summary": "Resume deterministic capture after completing login/captcha in browser.",
                "command": resume_command,
            }
        )
        if inspect_packet_cmd:
            recommended_steps.append(
                {
                    "step_id": "inspect_latest_web_capture_packet",
                    "summary": "Inspect latest web-capture packet/evidence for residual blockers.",
                    "command": inspect_packet_cmd,
                }
            )

        contract.update(
            {
                "status": "open",
                "opened_at": contract.get("opened_at") or now,
                "updated_at": now,
                "last_run_id": run_id,
                "gate_class": gate_class,
                "signals": signals,
                "state_path": rel_path(state_path),
                "evidence_ref": rel_path(last_json_path),
                "operator_actions": [
                    "Open the target URL in Chrome and complete required authentication/captcha manually.",
                    "Click the OpenClaw Browser Relay toolbar button on the authenticated tab (badge ON).",
                    "Re-run the macro with force in browser mode to continue deterministic capture.",
                ],
                "resume_command": resume_command,
                "incident_actionability": {
                    "schema_version": "openclaw.web_capture.login_wall_incident_actionability.v1",
                    "incident_id": f"web_login_wall::{domain_slug}::{macro_slug}",
                    "reason": "login_wall_operator_required",
                    "severity": "action_required",
                    "status": "open",
                    "action_required": True,
                    "recommended_commands": recommended_commands,
                    "recommended_steps": recommended_steps,
                    "evidence": [
                        rel_path(last_json_path),
                        rel_path(state_path),
                    ],
                },
            }
        )
    else:
        if contract.get("status") == "open":
            contract.update(
                {
                    "status": "resolved",
                    "resolved_at": now,
                    "updated_at": now,
                    "resolution": "successful_capture_after_login_wall",
                }
            )
        incident = contract.get("incident_actionability") if isinstance(contract.get("incident_actionability"), dict) else {}
        if incident:
            incident.update(
                {
                    "status": "resolved",
                    "action_required": False,
                    "resolved_at": now,
                }
            )
            contract["incident_actionability"] = incident

    atomic_write(login_contract_json, json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    md_lines = [
        "# Web Capture Login-Wall Operator Contract",
        "",
        f"- status: {contract.get('status')}",
        f"- domain: {domain}",
        f"- macro_slug: {macro_slug}",
        f"- target_url: {target_url}",
        f"- last_run_id: {contract.get('last_run_id') or 'n/a'}",
        f"- gate_class: {contract.get('gate_class') or 'n/a'}",
        f"- updated_at: {contract.get('updated_at') or contract.get('resolved_at') or now}",
        "",
    ]
    incident = contract.get("incident_actionability") if isinstance(contract.get("incident_actionability"), dict) else {}

    if contract.get("status") == "open":
        md_lines.extend(
            [
                "## Required operator actions",
                "1. Open target URL in Chrome and finish login/captcha manually.",
                "2. Attach Browser Relay on the authenticated tab (OpenClaw extension badge ON).",
                f"3. Resume: `{contract.get('resume_command')}`",
                "",
            ]
        )
        if incident:
            md_lines.append("## Incident actionability")
            md_lines.append(f"- incident_id: {incident.get('incident_id') or 'n/a'}")
            md_lines.append(f"- reason: {incident.get('reason') or 'n/a'}")
            md_lines.append(f"- severity: {incident.get('severity') or 'n/a'}")
            md_lines.append("- recommended_commands:")
            for cmd in list(incident.get("recommended_commands") or [])[:6]:
                md_lines.append(f"  - `{cmd}`")
            md_lines.append("")
    else:
        md_lines.extend(["Contract resolved after successful capture.", ""])

    atomic_write(login_contract_md, "\n".join(md_lines))

    return {
        "operator_contract_json": rel_path(login_contract_json),
        "operator_contract_md": rel_path(login_contract_md),
        "operator_contract_status": contract.get("status"),
    }


schedule = parse_schedule(retry_backoff, backoff_unit_sec)
state: Dict[str, Any] = load_json(state_path)

run_payload = load_json(last_json_path)
status = str(run_payload.get("status") or "failed").strip().lower()
if status not in {"ok", "blocked", "failed"}:
    status = "failed"

gate_class = derive_gate_class(run_payload)
result_reason = str(run_payload.get("result_reason") or run_payload.get("reason") or status).strip()
run_id = str(run_payload.get("run_id") or "").strip() or None
gating = run_payload.get("gating_flags") if isinstance(run_payload.get("gating_flags"), dict) else {}
signals = [str(s) for s in (gating.get("signals") or []) if str(s).strip()][:12]

bot_wall = bool(gating.get("bot_wall") or gate_class == "bot_wall")
login_wall = bool(gating.get("login_wall") or gate_class == "login_wall")
region_wall = bool(gating.get("region_wall") or gate_class == "region_block")

now = now_utc()
now_z = now_iso()

cursor_prev = int(state.get("backoff_cursor") or 0)
if cursor_prev < 0:
    cursor_prev = 0
if cursor_prev >= len(schedule):
    cursor_prev = max(0, len(schedule) - 1)

consecutive_failures_prev = int(state.get("consecutive_failures") or 0)
consecutive_blocked_prev = int(state.get("consecutive_blocked") or 0)
consecutive_bot_prev = int(state.get("consecutive_bot_wall") or 0)
consecutive_login_prev = int(state.get("consecutive_login_wall") or 0)
consecutive_region_prev = int(state.get("consecutive_region_wall") or 0)

if status == "ok":
    consecutive_failures = 0
    consecutive_blocked = 0
    consecutive_bot = 0
    consecutive_login = 0
    consecutive_region = 0
    backoff_cursor = 0
    cooldown_until = None
    cooldown_reason = None
else:
    consecutive_failures = consecutive_failures_prev + 1
    consecutive_blocked = (consecutive_blocked_prev + 1) if status == "blocked" else 0
    consecutive_bot = (consecutive_bot_prev + 1) if bot_wall else 0
    consecutive_login = (consecutive_login_prev + 1) if login_wall else 0
    consecutive_region = (consecutive_region_prev + 1) if region_wall else 0

    next_backoff = schedule[cursor_prev]
    if region_wall:
        next_backoff = max(next_backoff, region_cooldown_sec)
    if login_wall:
        next_backoff = max(next_backoff, login_cooldown_sec)

    cooldown_until_dt = now + dt.timedelta(seconds=int(next_backoff))
    cooldown_until = cooldown_until_dt.isoformat().replace("+00:00", "Z")
    cooldown_reason = gate_class if gate_class else status

    backoff_cursor = min(cursor_prev + 1, max(0, len(schedule) - 1))

operator_action_required = bool(state.get("operator_action_required"))
contract_meta = {}

if login_wall:
    operator_action_required = True
    contract_meta = write_login_contract(True, run_payload, state, gate_class, signals)
elif status == "ok" and operator_action_required:
    contract_meta = write_login_contract(False, run_payload, state, gate_class, signals)
    operator_action_required = False
elif status == "ok":
    operator_action_required = False

if status == "ok":
    contract_meta = write_login_contract(False, run_payload, state, gate_class, signals) if (login_contract_json and login_contract_json.exists()) else contract_meta

emit_blocker = status != "ok"
emit_reason = "runtime_failure"
if status == "blocked":
    if gate_class == "bot_wall":
        emit_blocker = consecutive_bot >= sustained_bot_threshold
        emit_reason = "bot_wall_sustained" if emit_blocker else "bot_wall_below_sustained_threshold"
    elif gate_class == "login_wall":
        emit_blocker = True
        emit_reason = "login_wall_operator_required"
    elif gate_class == "region_block":
        emit_blocker = True
        emit_reason = "region_block"
    else:
        emit_blocker = True
        emit_reason = gate_class or "blocked"
elif status == "failed":
    emit_blocker = True
    emit_reason = "failed"

new_state = {
    "schema_version": "openclaw.web_capture.domain_state.v1",
    "domain": domain,
    "domain_slug": domain_slug,
    "macro_slug": macro_slug,
    "target_url": target_url,
    "retry_backoff": retry_backoff,
    "backoff_schedule_sec": schedule,
    "backoff_cursor": backoff_cursor,
    "cooldown_until": cooldown_until,
    "cooldown_reason": cooldown_reason,
    "consecutive_failures": consecutive_failures,
    "consecutive_blocked": consecutive_blocked,
    "consecutive_bot_wall": consecutive_bot,
    "consecutive_login_wall": consecutive_login,
    "consecutive_region_wall": consecutive_region,
    "last_status": status,
    "last_gate_class": gate_class,
    "last_run_id": run_id,
    "last_reason": result_reason,
    "last_signals": signals,
    "last_attempt_at": now_z,
    "last_success_at": now_z if status == "ok" else state.get("last_success_at"),
    "operator_action_required": bool(operator_action_required),
    "operator_contract_json": contract_meta.get("operator_contract_json") or state.get("operator_contract_json"),
    "operator_contract_md": contract_meta.get("operator_contract_md") or state.get("operator_contract_md"),
    "updated_at": now_z,
    "created_at": state.get("created_at") or now_z,
    "last_record_from": rel_path(last_json_path),
}

atomic_write(state_path, json.dumps(new_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

cooldown_remaining_sec = 0
cooldown_until_parsed = parse_iso(new_state.get("cooldown_until"))
if cooldown_until_parsed is not None:
    cooldown_remaining_sec = max(0, int((cooldown_until_parsed - now).total_seconds()))

summary = (
    f"task=web_capture; macro={macro_slug}; domain={domain}; "
    f"reason={result_reason or status}; gate={gate_class or 'unknown'}; run_id={run_id or 'unknown'}; "
    f"signals={','.join(signals[:6])}; url={target_url}"
)
if new_state.get("operator_action_required"):
    summary += f"; operator_contract={new_state.get('operator_contract_json') or ''}"

payload = {
    "ok": True,
    "schema_version": "openclaw.web_capture.domain_guard.record.v1",
    "updated_at": now_z,
    "domain": domain,
    "domain_slug": domain_slug,
    "macro_slug": macro_slug,
    "run_id": run_id,
    "status": status,
    "gate_class": gate_class,
    "result_reason": result_reason,
    "signals": signals,
    "emit_blocker": bool(emit_blocker),
    "emit_reason": emit_reason,
    "cooldown_until": new_state.get("cooldown_until"),
    "cooldown_remaining_sec": cooldown_remaining_sec,
    "consecutive_bot_wall": consecutive_bot,
    "consecutive_failures": consecutive_failures,
    "operator_action_required": bool(new_state.get("operator_action_required")),
    "operator_contract_json": new_state.get("operator_contract_json"),
    "operator_contract_md": new_state.get("operator_contract_md"),
    "state_path": rel_path(state_path),
    "blocker_summary": summary,
}

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print(
        "WEB_DOMAIN_GUARD RECORD "
        f"domain={domain} status={status} gate={gate_class} "
        f"emit_blocker={emit_blocker} cooldown_remaining_sec={cooldown_remaining_sec}"
    )
PY
    ;;

  *)
    usage >&2
    exit 2 ;;
esac
