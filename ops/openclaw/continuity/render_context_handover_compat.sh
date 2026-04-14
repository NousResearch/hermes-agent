#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CHECKPOINT_REF=""
OUT_PATH="${OPENCLAW_CONTEXT_HANDOVER_COMPAT_PATH:-$ROOT/reports/handover_context_latest.md}"

usage() {
  cat <<'EOF'
Usage: render_context_handover_compat.sh [options]

Options:
  --checkpoint <path-or-id>   Optional checkpoint JSON path or checkpoint id.
                              If omitted, uses continuity latest pointer.
  --out <path>                Output markdown path (default: reports/handover_context_latest.md)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_REF="${2:-}"; shift 2 ;;
    --out)
      OUT_PATH="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$CHECKPOINT_REF" "$OUT_PATH" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys
from typing import Any, Dict, Optional

root = pathlib.Path(sys.argv[1]).resolve()
checkpoint_ref = (sys.argv[2] or "").strip()
out_path = pathlib.Path(sys.argv[3])
if not out_path.is_absolute():
    out_path = (root / out_path).resolve()


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def load_json_if_exists(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def read_nonnegative_int_env(name: str, *, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(int(default)))))
    except Exception:
        return int(default)


def age_sec_from_iso(raw: Any, *, now_dt: dt.datetime) -> Optional[int]:
    parsed = parse_iso(raw)
    if parsed is None:
        return None
    return max(0, int((now_dt - parsed).total_seconds()))


def resolve_proactive_lead(max_age_sec: int, lead_raw: int) -> int:
    if max_age_sec <= 1:
        return 0
    return min(max(0, int(lead_raw)), max_age_sec - 1)


def resolve_checkpoint_path() -> pathlib.Path:
    if checkpoint_ref:
        c = pathlib.Path(checkpoint_ref)
        if not c.is_absolute():
            c = (root / checkpoint_ref).resolve()
        if c.exists():
            return c
        by_id = root / "state" / "continuity" / "checkpoints" / f"{checkpoint_ref}.json"
        if by_id.exists():
            return by_id.resolve()
        raise SystemExit(f"checkpoint not found: {checkpoint_ref}")

    latest_surface = root / "state" / "continuity" / "latest" / "handover_latest.json"
    if latest_surface.exists():
        try:
            surface_obj = load_json(latest_surface)
            checkpoint = surface_obj.get("checkpoint") if isinstance(surface_obj.get("checkpoint"), dict) else {}
            rel = str(checkpoint.get("path") or "").strip()
            if rel:
                p = (root / rel).resolve()
                if p.exists():
                    return p
        except Exception:
            pass
        return latest_surface.resolve()

    pointer_path = root / "state" / "continuity" / "latest" / "latest_pointer.json"
    if pointer_path.exists():
        pointer = load_json(pointer_path)
        rel = str(pointer.get("json_path") or "")
        if rel:
            p = (root / rel).resolve()
            if p.exists():
                return p

    raise SystemExit("no continuity checkpoint found")


checkpoint_path = resolve_checkpoint_path()
checkpoint = load_json(checkpoint_path)

metadata = checkpoint.get("metadata") or {}
state_capture = checkpoint.get("state_capture") or {}
repo_state = checkpoint.get("repo_state") or {}
execution = checkpoint.get("execution_plan") or {}
objective = checkpoint.get("objective") or {}

env_snapshot_rel = str(state_capture.get("env_snapshot_path") or "")
gt_snapshot_rel = str(state_capture.get("ground_truth_snapshot_path") or "")

env_snapshot: Dict[str, Any] = {}
gt_snapshot: Dict[str, Any] = {}
if env_snapshot_rel:
    p = (root / env_snapshot_rel).resolve()
    if p.exists():
        env_snapshot = load_json(p)
if gt_snapshot_rel:
    p = (root / gt_snapshot_rel).resolve()
    if p.exists():
        gt_snapshot = load_json(p)

sessions = gt_snapshot.get("sessions") or {}
target_session = sessions.get("target_session") or {}
target_pct = sessions.get("target_session_pct")
total_tokens = target_session.get("totalTokens")
context_tokens = target_session.get("contextTokens")

session_file_size_bytes = target_session.get("sessionFileSizeBytes")
if session_file_size_bytes is None:
    session_file_size_bytes = (sessions.get("target_session") or {}).get("sessionFileSizeBytes")

pct_line = "n/a"
if target_pct is not None:
    pct_line = f"{target_pct:.6f}"

gateway = gt_snapshot.get("gateway") or {}
gateway_status = (((gateway.get("status") or {}).get("service") or {}).get("runtime") or {}).get("status")
gateway_rpc = (((gateway.get("status") or {}).get("rpc") or {}).get("ok"))

hl_repo = ((gt_snapshot.get("git_state") or {}).get("hl_terminal_repo") or {})

checkpoint_json_rel = to_rel(checkpoint_path)
checkpoint_md_rel = checkpoint_json_rel[:-5] + ".md" if checkpoint_json_rel.endswith(".json") else checkpoint_json_rel
if not (root / checkpoint_md_rel).exists():
    md_from_pointer = (root / "state" / "continuity" / "latest" / "handover_latest.md")
    if md_from_pointer.exists():
        checkpoint_md_rel = to_rel(md_from_pointer.resolve())

next_actions = execution.get("next_actions") or []
next_action_lines = []
for item in next_actions[:5]:
    if isinstance(item, dict):
        cmd = item.get("command") or ""
    else:
        cmd = str(item)
    if cmd:
        next_action_lines.append(f"- `{cmd}`")

if not next_action_lines:
    next_action_lines = ["- (none)"]

now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
now_iso = now_dt.isoformat().replace("+00:00", "Z")

handover_latest_path = root / "state" / "handover" / "latest.json"
handover_latest = load_json_if_exists(handover_latest_path)
handover_generated_at = str(handover_latest.get("generated_at") or "").strip() or None
handover_age_sec = age_sec_from_iso(handover_generated_at, now_dt=now_dt)
handover_freshness_max_age_sec = read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_HANDOVER_FRESHNESS_MAX_AGE_SEC",
    default=1800,
)
handover_proactive_refresh_lead_sec = resolve_proactive_lead(
    handover_freshness_max_age_sec,
    read_nonnegative_int_env(
        "OPENCLAW_CONTINUITY_HANDOVER_PROACTIVE_REFRESH_LEAD_SEC",
        default=300,
    ),
)
handover_freshness_remaining_sec = (
    max(0, handover_freshness_max_age_sec - handover_age_sec)
    if handover_age_sec is not None and handover_freshness_max_age_sec > 0
    else None
)
handover_proactive_refresh_due = bool(
    handover_freshness_remaining_sec is not None
    and handover_proactive_refresh_lead_sec > 0
    and handover_freshness_remaining_sec <= handover_proactive_refresh_lead_sec
)

reset_ready_refresh_path = root / "state" / "continuity" / "latest" / "reset_ready_refresh_latest.json"
reset_ready_refresh = load_json_if_exists(reset_ready_refresh_path)
reset_ready_refresh_generated_at = str(reset_ready_refresh.get("generated_at") or "").strip() or None
reset_ready_refresh_age_sec = age_sec_from_iso(reset_ready_refresh_generated_at, now_dt=now_dt)
reset_ready_refresh_freshness_max_age_sec = read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC",
    default=21600,
)
reset_ready_refresh_stale = bool(
    reset_ready_refresh_age_sec is not None
    and reset_ready_refresh_freshness_max_age_sec > 0
    and reset_ready_refresh_age_sec > reset_ready_refresh_freshness_max_age_sec
)

lines = [
    "## Context Watcher Handover",
    "",
    f"- generated_at_utc: `{now_iso}`",
    f"- checkpoint_id: `{metadata.get('checkpoint_id') or checkpoint_path.stem}`",
    f"- watcher_key: `{metadata.get('session_key') or 'unknown'}`",
    f"- pct: `{total_tokens} / {context_tokens} = {pct_line}`",
    f"- session_file_size_bytes: `{session_file_size_bytes if session_file_size_bytes is not None else 'n/a'}`",
    f"- checkpoint_json: `{checkpoint_json_rel}`",
    f"- checkpoint_md: `{checkpoint_md_rel}`",
    f"- env_snapshot: `{env_snapshot_rel or 'n/a'}`",
    f"- ground_truth_snapshot: `{gt_snapshot_rel or 'n/a'}`",
    "",
    "### Runtime truth",
    f"- gateway_runtime_status: `{gateway_status if gateway_status is not None else 'unknown'}`",
    f"- gateway_rpc_ok: `{gateway_rpc if gateway_rpc is not None else 'unknown'}`",
    f"- anomalies_count: `{len(gt_snapshot.get('anomalies') or [])}`",
    "",
    "### Successor packet freshness",
    f"- handover_latest_path: `{to_rel(handover_latest_path)}` present=`{handover_latest_path.exists()}` generated_at=`{handover_generated_at or 'n/a'}`",
    f"- handover_age_sec: `{handover_age_sec if handover_age_sec is not None else 'n/a'}` max_age_sec=`{handover_freshness_max_age_sec}` remaining_sec=`{handover_freshness_remaining_sec if handover_freshness_remaining_sec is not None else 'n/a'}`",
    f"- handover_proactive_refresh_lead_sec: `{handover_proactive_refresh_lead_sec}` proactive_refresh_due=`{handover_proactive_refresh_due}`",
    f"- reset_ready_refresh_path: `{to_rel(reset_ready_refresh_path)}` present=`{reset_ready_refresh_path.exists()}` generated_at=`{reset_ready_refresh_generated_at or 'n/a'}`",
    f"- reset_ready_refresh_ok: `{reset_ready_refresh.get('ok') if 'ok' in reset_ready_refresh else 'n/a'}` phase=`{reset_ready_refresh.get('phase') or 'n/a'}`",
    f"- reset_ready_refresh_age_sec: `{reset_ready_refresh_age_sec if reset_ready_refresh_age_sec is not None else 'n/a'}` max_age_sec=`{reset_ready_refresh_freshness_max_age_sec}` stale=`{reset_ready_refresh_stale}`",
    f"- freshness_action_hint: `{('bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/reset_ready_refresh.sh --json' if (handover_proactive_refresh_due or reset_ready_refresh_stale) else 'none')}`",
    "",
    "### Repo",
    f"- workspace repo: `{repo_state.get('repo_path') or root}`",
    f"- workspace branch: `{repo_state.get('branch') or 'unknown'}`",
    f"- workspace head: `{repo_state.get('head') or 'unknown'}`",
    f"- hl-terminal repo path: `{hl_repo.get('repo_path') or '/home/yeqiuqiu/projects/hl-terminal-gemini-canonical'}`",
    f"- hl-terminal branch: `{hl_repo.get('branch') or 'unknown'}`",
    f"- hl-terminal head: `{hl_repo.get('head') or 'unknown'}`",
    "",
    "### Objective",
    f"- status: `{objective.get('status') or 'unknown'}`",
    f"- primary_goal: `{objective.get('primary_goal') or ''}`",
    f"- blocker_reason: `{objective.get('blocker_reason') or 'none'}`",
    "",
    "### Next actions",
]
lines.extend(next_action_lines)

lines.extend(
    [
        "",
        "### Operating mode (important)",
        "- Workspace role: control-plane (`/home/yeqiuqiu/clawd-architect`), not main product repo.",
        "- Obsidian role: guarded hourly canary (silent-success, blocker-only).",
        "- Scope guard: do not touch NGMI unless explicitly requested.",
        "",
        "### Telegram thin-lane continuity",
        "- Telegram DM is cockpit-only: summaries, approvals, and continuity steering.",
        "- Keep heavy orchestration in worker/subagent lanes; do not run engine-room loops in the DM thread.",
        "- If context pressure rises, trust checkpoint + handover artifacts and reset early rather than stretching the transcript.",
        "",
        "### Safe resume commands",
        "- `openclaw gateway status`",
        "- `openclaw doctor --check`",
        "- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/snapshot_ground_truth.sh`",
        "- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/verify_then_resume.sh`",
        "",
        "### Key continuity references",
        "- `/home/yeqiuqiu/clawd-architect/reports/continuity_checkpoint_os_blueprint_2026-03-08.md`",
        "- `/home/yeqiuqiu/clawd-architect/reports/ground_truth_connectors_plan_2026-03-08.md`",
        "- `/home/yeqiuqiu/clawd-architect/reports/p0_upgrade_batch1_verification_2026-03-08.md`",
        "- `/home/yeqiuqiu/clawd-architect/reports/successor_control_plane_handover_2026-03-08.md`",
        "",
        "### Instruction",
        "Run /reset and continue using this handover.",
        "",
    ]
)

out_path.parent.mkdir(parents=True, exist_ok=True)
tmp = out_path.with_name(out_path.name + ".tmp")
tmp.write_text("\n".join(lines), encoding="utf-8")
os.replace(tmp, out_path)

print(
    json.dumps(
        {
            "ok": True,
            "checkpoint_json": checkpoint_json_rel,
            "checkpoint_md": checkpoint_md_rel,
            "handover_path": to_rel(out_path),
        },
        ensure_ascii=False,
    )
)
PY
