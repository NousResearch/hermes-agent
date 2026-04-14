#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
TARGET_SESSION_KEY="${OPENCLAW_TARGET_SESSION_KEY:-agent:codex-orchestrator-pro:telegram:direct:5936691533}"
TARGET_AGENT_ID="${OPENCLAW_TARGET_AGENT_ID:-$(printf '%s' "$TARGET_SESSION_KEY" | cut -d: -f2)}"
if [[ -z "$TARGET_AGENT_ID" || "$TARGET_AGENT_ID" == "$TARGET_SESSION_KEY" ]]; then
  TARGET_AGENT_ID="codex-orchestrator-pro"
fi
SESSION_STORE_PATH="${OPENCLAW_SESSION_STORE_PATH:-/home/yeqiuqiu/.openclaw/agents/${TARGET_AGENT_ID}/sessions/sessions.json}"
CONTEXT_WARN_THRESHOLD_PCT="${OPENCLAW_GROUND_TRUTH_CONTEXT_WARN_PCT:-0.85}"

python3 - "$ROOT" "$TARGET_SESSION_KEY" "$TARGET_AGENT_ID" "$SESSION_STORE_PATH" "$CONTEXT_WARN_THRESHOLD_PCT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
target_session_key = sys.argv[2]
target_agent_id = str(sys.argv[3] or "").strip() or "codex-orchestrator-pro"
session_store_path = pathlib.Path(sys.argv[4]).expanduser()
try:
    context_warn_threshold = float(sys.argv[5])
except Exception:
    context_warn_threshold = 0.85

gt_dir = root / "state" / "ground_truth"
snap_dir = gt_dir / "snapshots"
gt_dir.mkdir(parents=True, exist_ok=True)
snap_dir.mkdir(parents=True, exist_ok=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def atomic_write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def run_cmd(args: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "command": args,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 99,
            "stdout": "",
            "stderr": str(exc),
            "command": args,
        }


def run_json(args: List[str], timeout: int = 30) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    out = run_cmd(args, timeout=timeout)
    if not out["ok"]:
        return None, out
    raw = (out.get("stdout") or "").strip()
    if not raw:
        return {}, None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
        return {"value": obj}, None
    except Exception as exc:
        out["stderr"] = f"json_parse_error: {exc}; raw_prefix={raw[:200]}"
        return None, out


def next_append_only_snapshot(base_id: str) -> Tuple[str, pathlib.Path]:
    direct = snap_dir / f"{base_id}.json"
    if not direct.exists():
        return base_id, direct
    for i in range(1, 1000):
        sid = f"{base_id}_{i:02d}"
        p = snap_dir / f"{sid}.json"
        if not p.exists():
            return sid, p
    raise RuntimeError("unable to allocate append-only snapshot filename")


def parse_prometheus_metrics(path: pathlib.Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value = parts[0], parts[-1]
            try:
                metrics[name] = float(value)
            except ValueError:
                continue
    except Exception:
        pass
    return metrics


def collect_git_state(repo_path: pathlib.Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "repo_path": str(repo_path),
        "exists": repo_path.exists(),
    }
    if not repo_path.exists():
        return data

    head = run_cmd(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    branch = run_cmd(["git", "-C", str(repo_path), "branch", "--show-current"])
    status = run_cmd(["git", "-C", str(repo_path), "status", "--short", "--branch"])

    data["head"] = (head.get("stdout") or "").strip() if head["ok"] else None
    data["branch"] = (branch.get("stdout") or "").strip() if branch["ok"] else None

    status_lines = [ln for ln in (status.get("stdout") or "").splitlines() if ln.strip()]
    data["git_status_short"] = status_lines
    data["dirty"] = any(not ln.startswith("##") for ln in status_lines)

    if not head["ok"]:
        data.setdefault("errors", []).append((head.get("stderr") or "git rev-parse failed").strip())
    if not status["ok"]:
        data.setdefault("errors", []).append((status.get("stderr") or "git status failed").strip())

    return data


def parse_systemd_show(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


now = now_utc()
ts_iso = now.isoformat().replace("+00:00", "Z")
ts_file = now.strftime("%Y%m%dT%H%M%SZ")
base_snapshot_id = f"gt_{ts_file}"
snapshot_id, snapshot_path = next_append_only_snapshot(base_snapshot_id)

errors: List[Dict[str, Any]] = []

# Gateway / cron / sessions truth
gateway, gateway_err = run_json(["openclaw", "gateway", "status", "--json"], timeout=45)
cron, cron_err = run_json(["openclaw", "cron", "list", "--json"], timeout=45)
sessions, sessions_err = run_json(
    ["openclaw", "sessions", "--agent", target_agent_id, "--active", "1440", "--json"], timeout=45
)

if gateway_err:
    errors.append({"source": "openclaw gateway status", "error": (gateway_err.get("stderr") or "failed").strip()[:400]})
if cron_err:
    errors.append({"source": "openclaw cron list", "error": (cron_err.get("stderr") or "failed").strip()[:400]})
if sessions_err:
    errors.append({"source": "openclaw sessions", "error": (sessions_err.get("stderr") or "failed").strip()[:400]})

jobs = list((cron or {}).get("jobs") or [])
sessions_list = list((sessions or {}).get("sessions") or [])

target_session: Optional[Dict[str, Any]] = None
for row in sessions_list:
    if str(row.get("key") or "") == target_session_key:
        target_session = row
        break

session_pct = None
if target_session:
    total = float(target_session.get("totalTokens") or 0)
    ctx = float(target_session.get("contextTokens") or 0)
    if ctx > 0:
        session_pct = total / ctx

# enrich target session with deterministic session-file truth from local store
if target_session is None:
    target_session = {}
else:
    target_session = dict(target_session)

session_file_path = None
session_file_size_bytes = None
if session_store_path.exists():
    try:
        store_obj = json.loads(session_store_path.read_text(encoding="utf-8"))
        session_meta = store_obj.get(target_session_key) if isinstance(store_obj, dict) else None
        if isinstance(session_meta, dict):
            raw_file = str(session_meta.get("sessionFile") or "").strip()
            if raw_file:
                sf = pathlib.Path(raw_file)
                session_file_path = str(sf)
                if sf.exists() and sf.is_file():
                    session_file_size_bytes = sf.stat().st_size
    except Exception as exc:
        errors.append({"source": "sessions_store_parse", "error": str(exc)[:400]})

if session_file_path:
    target_session["sessionFile"] = session_file_path
if session_file_size_bytes is not None:
    target_session["sessionFileSizeBytes"] = int(session_file_size_bytes)

# systemd timers (focused set only)
timer_units = [
    "openclaw-watchdog-session-bloat.timer",
    "openclaw-watchdog-telegram-409.timer",
    "openclaw-verify-daily.timer",
    "openclaw-update-status.timer",
    "openclaw-security-audit.timer",
    "hl-terminal-ops-doctor.timer",
    "hl-terminal-ops-verify.timer",
]

timers: Dict[str, Any] = {}
for unit in timer_units:
    cmd = [
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=Id,ActiveState,SubState,UnitFileState,NextElapseUSecRealtime,LastTriggerUSecRealtime",
    ]
    r = run_cmd(cmd, timeout=20)
    if r["ok"]:
        timers[unit] = parse_systemd_show(r.get("stdout") or "")
    else:
        timers[unit] = {"error": (r.get("stderr") or "systemctl show failed").strip()[:240]}

# process/network truth
ss_out = run_cmd(["ss", "-ltnp"], timeout=20)
ps_out = run_cmd(["ps", "-eo", "pid,ppid,etimes,cmd", "--sort=-etimes"], timeout=20)

ss_lines = [ln for ln in (ss_out.get("stdout") or "").splitlines() if ln.strip()]
ps_lines = [ln for ln in (ps_out.get("stdout") or "").splitlines() if ln.strip()]

# watchdog + telemetry state files
watchdog_state_files: Dict[str, Any] = {}
for p in sorted((root / "state" / "cron_watchdog").glob("*.json")):
    key = to_rel(p)
    try:
        watchdog_state_files[key] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        watchdog_state_files[key] = {"_error": str(exc)}

telemetry_metrics: Dict[str, float] = {}
for p in sorted((root / "ops" / "telemetry" / "textfile").glob("*.prom")):
    telemetry_metrics.update(parse_prometheus_metrics(p))

# local state probes
autopilot_state_path = root / "ops" / "autopilot" / "state" / "hl_terminal_v1.json"
autopilot_state: Dict[str, Any] = {"path": str(autopilot_state_path), "exists": autopilot_state_path.exists()}
if autopilot_state_path.exists():
    try:
        autopilot_state["state"] = json.loads(autopilot_state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        autopilot_state["error"] = str(exc)

obsidian_state: Dict[str, Any] = {}
canary_state = pathlib.Path("/tmp/obsidian_hourly_canary_state_canary.json")
obsidian_state["canary_state_path"] = str(canary_state)
obsidian_state["canary_state_exists"] = canary_state.exists()
if canary_state.exists():
    try:
        obsidian_state["canary_state"] = json.loads(canary_state.read_text(encoding="utf-8"))
    except Exception as exc:
        obsidian_state["canary_state_error"] = str(exc)

obsidian_sig = root / "state" / "cron_watchdog" / "obsidian_hourly_canary_input_hash.json"
obsidian_state["input_hash_path"] = str(obsidian_sig)
obsidian_state["input_hash_exists"] = obsidian_sig.exists()
if obsidian_sig.exists():
    try:
        obsidian_state["input_hash_state"] = json.loads(obsidian_sig.read_text(encoding="utf-8"))
    except Exception as exc:
        obsidian_state["input_hash_error"] = str(exc)

# git truth
git_state = {
    "workspace": collect_git_state(root),
    "hl_terminal_repo": collect_git_state(pathlib.Path("/home/yeqiuqiu/projects/hl-terminal-gemini-canonical")),
}

# deterministic anomalies
anomalies: List[Dict[str, Any]] = []

runtime_status = str((((gateway or {}).get("service") or {}).get("runtime") or {}).get("status") or "")
rpc_ok = bool((((gateway or {}).get("rpc") or {}).get("ok")))
if gateway is None or runtime_status.lower() != "running" or not rpc_ok:
    anomalies.append(
        {
            "severity": "critical",
            "key": "gateway_unhealthy",
            "details": {
                "runtime_status": runtime_status or "unknown",
                "rpc_ok": rpc_ok,
            },
        }
    )

cron_errors = []
for job in jobs:
    st = (job.get("state") or {}) if isinstance(job, dict) else {}
    if str(st.get("lastStatus") or "").lower() == "error" or str(st.get("lastRunStatus") or "").lower() == "error":
        cron_errors.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "lastError": st.get("lastError"),
            }
        )
if cron_errors:
    anomalies.append(
        {
            "severity": "warn",
            "key": "cron_jobs_error_state",
            "details": {"count": len(cron_errors), "jobs": cron_errors},
        }
    )

if session_pct is not None and session_pct >= context_warn_threshold:
    anomalies.append(
        {
            "severity": "warn",
            "key": "target_session_context_high",
            "details": {
                "session_key": target_session_key,
                "pct": round(session_pct, 6),
                "warn_threshold_pct": context_warn_threshold,
                "total_tokens": target_session.get("totalTokens"),
                "context_tokens": target_session.get("contextTokens"),
            },
        }
    )

snapshot: Dict[str, Any] = {
    "schema_version": "ground_truth.snapshot.v1",
    "snapshot_id": snapshot_id,
    "snapshot_ts_utc": ts_iso,
    "host": socket.gethostname(),
    "workspace": str(root),
    "target_session_key": target_session_key,
    "target_agent_id": target_agent_id,
    "gateway": {
        "status": gateway,
        "error": None if gateway_err is None else (gateway_err.get("stderr") or "failed").strip()[:400],
    },
    "cron": {
        "jobs": jobs,
        "total": len(jobs),
        "error": None if cron_err is None else (cron_err.get("stderr") or "failed").strip()[:400],
    },
    "systemd_timers": {
        "units": timers,
    },
    "sessions": {
        "count": len(sessions_list),
        "target_session": target_session,
        "target_session_pct": session_pct,
        "top_by_total_tokens": sorted(
            [
                {
                    "key": s.get("key"),
                    "kind": s.get("kind"),
                    "totalTokens": s.get("totalTokens"),
                    "contextTokens": s.get("contextTokens"),
                    "updatedAt": s.get("updatedAt"),
                    "sessionFileSizeBytes": s.get("sessionFileSizeBytes"),
                }
                for s in sessions_list
                if isinstance(s, dict)
            ],
            key=lambda x: float(x.get("totalTokens") or 0),
            reverse=True,
        )[:10],
        "error": None if sessions_err is None else (sessions_err.get("stderr") or "failed").strip()[:400],
    },
    "process_ports": {
        "ss_ok": bool(ss_out.get("ok")),
        "ss_error": None if ss_out.get("ok") else (ss_out.get("stderr") or "ss failed").strip()[:400],
        "listening_ports": ss_lines[:200],
        "ps_ok": bool(ps_out.get("ok")),
        "ps_error": None if ps_out.get("ok") else (ps_out.get("stderr") or "ps failed").strip()[:400],
        "top_processes": ps_lines[:200],
    },
    "watchdog_state": {
        "state_files": watchdog_state_files,
        "telemetry_metrics": telemetry_metrics,
    },
    "autopilot_state": autopilot_state,
    "obsidian_state": obsidian_state,
    "git_state": git_state,
    "anomalies": anomalies,
    "collection_errors": errors,
}

snapshot_text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
snapshot_sha = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
atomic_write(snapshot_path, snapshot_text)

latest = {
    "schema_version": "ground_truth.latest.v1",
    "updated_at": ts_iso,
    "snapshot_id": snapshot_id,
    "snapshot_ts_utc": ts_iso,
    "snapshot_path": to_rel(snapshot_path),
    "snapshot_sha256": snapshot_sha,
    "host": socket.gethostname(),
    "workspace": str(root),
}
latest_text = json.dumps(latest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
atomic_write(gt_dir / "latest.json", latest_text)

print(
    json.dumps(
        {
            "ok": True,
            "snapshot_id": snapshot_id,
            "snapshot_path": to_rel(snapshot_path),
            "latest_path": "state/ground_truth/latest.json",
            "snapshot_sha256": snapshot_sha,
        },
        ensure_ascii=False,
    )
)
PY
