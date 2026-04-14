#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
TRIGGER="manual"
STATUS="PROGRESS"
OBJECTIVE="Continuity checkpoint"
BLOCKER_REASON=""
SESSION_KEY="${OPENCLAW_TARGET_SESSION_KEY:-agent:codex-executioner:telegram:direct:5936691533}"
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

declare -a NEXT_ACTIONS=()
declare -a VERIFY_CMDS=()
declare -a ROLLBACK_CMDS=()
PRESERVE_HANDOVER_LATEST=0

usage() {
  cat <<'EOF'
Usage: write_checkpoint.sh [options]

Options:
  --trigger <name>            Trigger name (default: manual)
  --status <READY|PROGRESS|BLOCKER>
  --objective <text>
  --blocker-reason <text>
  --session-key <key>
  --next-action <command>     Repeatable; defaults to a no-op echo command
  --verify-cmd <command>      Repeatable; defaults to safe read-only checks
  --rollback-cmd <command>    Repeatable; defaults to autopilot pause command
  --preserve-handover-latest  Keep continuity/latest/handover_latest.{json,md} unchanged while updating latest_pointer.
  --action-token <value>      Canonical mutation token for direct entrypoint use.
  --truth-anchor <value>      Legacy alias of --action-token.
  --allow-legacy-anchor       Allow legacy anchor-only token mode for direct token validation.
  --mutation-ticket <value>   Authority ticket JSON string, @path, or path (high-risk token path).
  --attestation <name>        Satisfied authority attestation (repeatable).
  --attestation-object <value> Structured attestation JSON string, @path, or path (repeatable).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trigger)
      TRIGGER="$2"; shift 2 ;;
    --status)
      STATUS="$2"; shift 2 ;;
    --objective)
      OBJECTIVE="$2"; shift 2 ;;
    --blocker-reason)
      BLOCKER_REASON="$2"; shift 2 ;;
    --session-key)
      SESSION_KEY="$2"; shift 2 ;;
    --next-action)
      NEXT_ACTIONS+=("$2"); shift 2 ;;
    --verify-cmd)
      VERIFY_CMDS+=("$2"); shift 2 ;;
    --rollback-cmd)
      ROLLBACK_CMDS+=("$2"); shift 2 ;;
    --preserve-handover-latest)
      PRESERVE_HANDOVER_LATEST=1; shift ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1 ;;
  esac
done

guard_args=(
  --script "write_checkpoint.sh"
  --risk-tier "high"
  --mutation-operation "write_checkpoint:emit"
)
if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args+=(--action-token "$ACTION_TOKEN")
fi
if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  guard_args+=(--allow-legacy-anchor)
fi
if [[ -n "$MUTATION_TICKET" ]]; then
  guard_args+=(--mutation-ticket "$MUTATION_TICKET")
fi
for att in "${MUTATION_ATTESTATIONS[@]}"; do
  if [[ -n "${att:-}" ]]; then
    guard_args+=(--attestation "$att")
  fi
done
for att_obj in "${MUTATION_ATTESTATION_OBJECTS[@]}"; do
  if [[ -n "${att_obj:-}" ]]; then
    guard_args+=(--attestation-object "$att_obj")
  fi
done
"$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"

case "$STATUS" in
  READY|PROGRESS|BLOCKER) ;;
  *)
    echo "Invalid --status: $STATUS (expected READY|PROGRESS|BLOCKER)" >&2
    exit 1 ;;
esac

if [[ ${#NEXT_ACTIONS[@]} -eq 0 ]]; then
  NEXT_ACTIONS=("echo 'Checkpoint captured; no next action specified.'")
fi
if [[ ${#VERIFY_CMDS[@]} -eq 0 ]]; then
  VERIFY_CMDS=(
    "git -C $ROOT rev-parse HEAD"
    "openclaw gateway status --json >/dev/null"
    "openclaw cron list --json >/dev/null"
    "ss -ltn >/dev/null"
  )
fi
if [[ ${#ROLLBACK_CMDS[@]} -eq 0 ]]; then
  ROLLBACK_CMDS=("$ROOT/ops/autopilot/bin/hl_autopilot_ctl.sh pause || true")
fi

json_array_from_args() {
  if [[ $# -eq 0 ]]; then
    printf '[]'
    return 0
  fi
  printf '%s\n' "$@" | python3 -c 'import json,sys; print(json.dumps([ln.rstrip("\n") for ln in sys.stdin if ln.strip()]))'
}

NEXT_ACTIONS_JSON="$(json_array_from_args "${NEXT_ACTIONS[@]}")"
VERIFY_CMDS_JSON="$(json_array_from_args "${VERIFY_CMDS[@]}")"
ROLLBACK_CMDS_JSON="$(json_array_from_args "${ROLLBACK_CMDS[@]}")"

"$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null
"$ROOT/ops/openclaw/continuity/capture_env_snapshot.sh" >/dev/null

checkpoint_result="$(python3 - "$ROOT" "$TRIGGER" "$STATUS" "$OBJECTIVE" "$BLOCKER_REASON" "$SESSION_KEY" "$NEXT_ACTIONS_JSON" "$VERIFY_CMDS_JSON" "$ROLLBACK_CMDS_JSON" "$PRESERVE_HANDOVER_LATEST" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any, Dict, List, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
trigger = sys.argv[2]
status = sys.argv[3]
objective = sys.argv[4]
blocker_reason = sys.argv[5].strip()
session_key = sys.argv[6]
next_actions = json.loads(sys.argv[7])
verification_commands = json.loads(sys.argv[8])
rollback_commands = json.loads(sys.argv[9])
preserve_handover_latest = str(sys.argv[10] or "0").strip() == "1"

continuity_dir = root / "state" / "continuity"
checkpoints_dir = continuity_dir / "checkpoints"
latest_dir = continuity_dir / "latest"
checkpoints_dir.mkdir(parents=True, exist_ok=True)
latest_dir.mkdir(parents=True, exist_ok=True)


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


def atomic_symlink(target_rel: str, dest: pathlib.Path) -> None:
    tmp = dest.with_name(f"{dest.name}.tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target_rel, tmp)
    os.replace(tmp, dest)


def next_checkpoint_path(base_id: str) -> Tuple[str, pathlib.Path, pathlib.Path]:
    j = checkpoints_dir / f"{base_id}.json"
    m = checkpoints_dir / f"{base_id}.md"
    if not j.exists() and not m.exists():
        return base_id, j, m
    for i in range(1, 1000):
        cid = f"{base_id}_{i:02d}"
        cj = checkpoints_dir / f"{cid}.json"
        cm = checkpoints_dir / f"{cid}.md"
        if not cj.exists() and not cm.exists():
            return cid, cj, cm
    raise RuntimeError("unable to allocate checkpoint filename")


env_latest_path = latest_dir / "env_snapshot_latest.json"
if not env_latest_path.exists():
    raise SystemExit(f"missing env snapshot latest pointer: {env_latest_path}")

env_latest = json.loads(env_latest_path.read_text(encoding="utf-8"))
env_snapshot_rel = str(env_latest.get("env_snapshot_path") or "")
if not env_snapshot_rel:
    raise SystemExit("env snapshot latest pointer missing env_snapshot_path")

env_snapshot_path = (root / env_snapshot_rel).resolve()
if not env_snapshot_path.exists():
    raise SystemExit(f"missing env snapshot file: {env_snapshot_path}")

env_snapshot = json.loads(env_snapshot_path.read_text(encoding="utf-8"))

gt_snapshot_rel = str(((env_snapshot.get("source_ground_truth") or {}).get("snapshot_path")) or "")

latest_pointer_path = latest_dir / "latest_pointer.json"
parent_checkpoint_id = None
if latest_pointer_path.exists():
    try:
        parent_checkpoint_id = (json.loads(latest_pointer_path.read_text(encoding="utf-8")) or {}).get("checkpoint_id")
    except Exception:
        parent_checkpoint_id = None

now = now_utc()
ts_iso = now.isoformat().replace("+00:00", "Z")
ts_file = now.strftime("%Y%m%dT%H%M%SZ")
seed = f"{ts_iso}|{trigger}|{objective}|{parent_checkpoint_id or 'root'}"
base_id = f"chk_{ts_file}_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:6]}"
checkpoint_id, checkpoint_json_path, checkpoint_md_path = next_checkpoint_path(base_id)

cron_jobs = ((env_snapshot.get("cron_summary") or {}).get("jobs") or [])
cron_capture = [
    {
        "id": row.get("id"),
        "name": row.get("name"),
        "enabled": row.get("enabled"),
        "last_status": ((row.get("state") or {}).get("lastStatus")),
    }
    for row in cron_jobs
    if isinstance(row, dict)
]

sessions_top = ((env_snapshot.get("sessions_summary") or {}).get("top_by_total_tokens") or [])
active_subagents = [
    row for row in sessions_top if ":subagent:" in str(row.get("key") or "")
]

ps_lines = ((env_snapshot.get("process_summary") or {}).get("top_processes") or [])
active_pids: List[int] = []
for line in ps_lines[1:200]:
    first = (line.strip().split() or [""])[0]
    if first.isdigit():
        active_pids.append(int(first))

ss_lines = ((env_snapshot.get("process_summary") or {}).get("listening_ports") or [])
open_ports = ss_lines[1:120] if len(ss_lines) > 1 else ss_lines[:120]

repo_state = (env_snapshot.get("repo_state") or {})
workspace_repo = (repo_state.get("workspace") or {}) if isinstance(repo_state, dict) else {}

checkpoint: Dict[str, Any] = {
    "schema_version": "continuity.checkpoint.v1",
    "metadata": {
        "checkpoint_id": checkpoint_id,
        "created_at": ts_iso,
        "trigger": trigger,
        "parent_checkpoint_id": parent_checkpoint_id,
        "session_key": session_key,
    },
    "objective": {
        "primary_goal": objective,
        "status": status,
        "blocker_reason": blocker_reason or None,
    },
    "repo_state": {
        "repo_path": workspace_repo.get("repo_path") or str(root),
        "branch": workspace_repo.get("branch"),
        "head": workspace_repo.get("head"),
        "git_status_short": workspace_repo.get("git_status_short") or [],
    },
    "state_capture": {
        "env_snapshot_path": env_snapshot_rel,
        "ground_truth_snapshot_path": gt_snapshot_rel,
        "active_cron_jobs": cron_capture,
        "active_pids": active_pids,
        "active_subagents": active_subagents,
        "open_ports": open_ports,
        "gateway_health": env_snapshot.get("gateway"),
        "watchdog_state_refs": (env_snapshot.get("watchdog_state") or {}).get("state_refs") or [],
        "anomalies": env_snapshot.get("anomalies") or [],
    },
    "execution_plan": {
        "next_actions": [{"intent": "next_action", "command": cmd} for cmd in next_actions],
        "verification_commands": verification_commands,
        "rollback_commands": rollback_commands,
    },
    "integrity": {
        "snapshot_sha256": env_latest.get("env_snapshot_sha256"),
        "checkpoint_payload_sha256": None,
        "checkpoint_file_sha256": None,
    },
}

canonical_payload = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
checkpoint["integrity"]["checkpoint_payload_sha256"] = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

checkpoint_text = json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
atomic_write(checkpoint_json_path, checkpoint_text)

checkpoint_file_sha = hashlib.sha256(checkpoint_json_path.read_bytes()).hexdigest()
checkpoint["integrity"]["checkpoint_file_sha256"] = checkpoint_file_sha
checkpoint_text = json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
atomic_write(checkpoint_json_path, checkpoint_text)
checkpoint_file_sha = hashlib.sha256(checkpoint_json_path.read_bytes()).hexdigest()

json_rel = to_rel(checkpoint_json_path)
md_rel = to_rel(checkpoint_md_path)

frontmatter = {
    "checkpoint_id": checkpoint_id,
    "created_at": ts_iso,
    "trigger": trigger,
    "status": status,
    "objective": objective,
    "json_path": json_rel,
    "env_snapshot_path": env_snapshot_rel,
    "ground_truth_snapshot_path": gt_snapshot_rel,
}

fm_lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in frontmatter.items()] + ["---", ""]

md_lines = fm_lines + [
    "## Objective",
    f"- **Primary goal:** {objective}",
    f"- **Status:** {status}",
    f"- **Blocker reason:** {blocker_reason or 'none'}",
    "",
    "## Environment State",
    f"- Env snapshot: `{env_snapshot_rel}`",
    f"- Ground-truth snapshot: `{gt_snapshot_rel or 'n/a'}`",
    f"- Active cron jobs captured: {len(cron_capture)}",
    f"- Active subagents captured: {len(active_subagents)}",
    f"- Open port rows captured: {len(open_ports)}",
    f"- Anomalies captured: {len((env_snapshot.get('anomalies') or []))}",
    "",
    "## Next Actions",
]
for cmd in next_actions:
    md_lines.append(f"- `{cmd}`")

md_lines += ["", "## Verification Commands"]
for cmd in verification_commands:
    md_lines.append(f"- `{cmd}`")

md_lines += ["", "## Rollback Commands"]
for cmd in rollback_commands:
    md_lines.append(f"- `{cmd}`")

md_lines += ["", "## Notes", "- Generated by `ops/openclaw/continuity/write_checkpoint.sh`.", ""]

atomic_write(checkpoint_md_path, "\n".join(md_lines))

if not preserve_handover_latest:
    atomic_symlink(f"../checkpoints/{checkpoint_json_path.name}", latest_dir / "handover_latest.json")
    atomic_symlink(f"../checkpoints/{checkpoint_md_path.name}", latest_dir / "handover_latest.md")

latest_pointer = {
    "schema_version": "continuity.latest_pointer.v1",
    "updated_at": ts_iso,
    "checkpoint_id": checkpoint_id,
    "json_path": json_rel,
    "md_path": md_rel,
    "json_sha256": checkpoint_file_sha,
}
atomic_write(latest_pointer_path, json.dumps(latest_pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

# Persist checkpoint row in continuity DB.
db_path = continuity_dir / "continuity_os.sqlite"
try:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO checkpoints (
          checkpoint_id, created_at, trigger, status, objective,
          parent_checkpoint_id, json_path, md_path, snapshot_path,
          repo_branch, repo_head, integrity_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            checkpoint_id,
            ts_iso,
            trigger,
            status,
            objective,
            parent_checkpoint_id,
            json_rel,
            md_rel,
            env_snapshot_rel,
            (workspace_repo.get("branch") if isinstance(workspace_repo, dict) else None),
            (workspace_repo.get("head") if isinstance(workspace_repo, dict) else None),
            checkpoint_file_sha,
        ),
    )
    con.commit()
    con.close()
except Exception:
    pass

print(
    json.dumps(
        {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "json_path": json_rel,
            "md_path": md_rel,
            "latest_pointer": "state/continuity/latest/latest_pointer.json",
            "handover_latest_promoted": not preserve_handover_latest,
        },
        ensure_ascii=False,
    )
)
PY
)"

checkpoint_id="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("checkpoint_id") or ""))' "$checkpoint_result")"

sync_result='{}'
if [[ -n "$checkpoint_id" ]]; then
  sync_cmd=("$ROOT/ops/openclaw/continuity/sync_latest_artifacts.sh")
  if [[ "$PRESERVE_HANDOVER_LATEST" != "1" ]]; then
    sync_cmd+=(--checkpoint "$checkpoint_id")
  fi
  set +e
  sync_result="$(OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="write_checkpoint.sh:sync_latest_artifacts" "${sync_cmd[@]}" 2>/tmp/write_checkpoint_sync.err)"
  sync_rc=$?
  set -e
  if [[ "$sync_rc" -ne 0 ]]; then
    err="$(cat /tmp/write_checkpoint_sync.err 2>/dev/null || true)"
    sync_result="$(python3 -c 'import json,sys; print(json.dumps({"ok": False, "error": (sys.argv[1] or "sync_latest_artifacts_failed")[:240]}, ensure_ascii=False))' "$err")"
  fi
fi

python3 - "$checkpoint_result" "$sync_result" <<'PY'
import json
import sys

base = json.loads(sys.argv[1])
try:
    sync = json.loads(sys.argv[2]) if sys.argv[2].strip() else {}
except Exception:
    sync = {"ok": False, "error": "sync_result_parse_failed"}

if isinstance(sync, dict) and sync:
    base["runtime_truth_sync"] = sync

print(json.dumps(base, ensure_ascii=False))
PY
