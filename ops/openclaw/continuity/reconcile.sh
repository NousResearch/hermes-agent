#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0
FORCE=0
DRY_RUN=0
REFRESH_SNAPSHOT=1
MIN_INTERVAL_SEC="${OPENCLAW_CONTINUITY_RECONCILE_MIN_INTERVAL_SEC:-1800}"
EVENT_ROUTER="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
TELEMETRY_ENABLED="${OPENCLAW_CONTINUITY_RECONCILE_TELEMETRY:-1}"
TELEMETRY_COOLDOWN_SEC="${OPENCLAW_CONTINUITY_RECONCILE_TELEMETRY_COOLDOWN_SEC:-1800}"
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: reconcile.sh [options]

Low-risk continuity drift reconcile path.
When not-ready is drift-only (pointer/ground-truth capture drift, connector freshness drift), this script refreshes
latest continuity artifacts by writing a fresh checkpoint and re-syncing latest bridge/handover.

Options:
  --json                         Print machine JSON output.
  --force                        Reconcile even when status is not drift-only.
  --dry-run                      Evaluate and report, but do not write checkpoint/sync artifacts.
  --no-snapshot                  Skip pre-reconcile ground-truth snapshot refresh.
  --min-interval-sec <n>         Minimum seconds between drift_reconcile checkpoint writes
                                 (default: OPENCLAW_CONTINUITY_RECONCILE_MIN_INTERVAL_SEC or 1800).
  --no-telemetry                 Disable reconcile telemetry routing for this invocation.
  --telemetry-cooldown-sec <n>   Telemetry dedupe cooldown seconds
                                 (default: OPENCLAW_CONTINUITY_RECONCILE_TELEMETRY_COOLDOWN_SEC or 1800).
  --action-token <value>         Canonical mutation token for direct entrypoint use.
  --truth-anchor <value>         Legacy alias of --action-token.
  --allow-legacy-anchor          Allow legacy anchor-only token mode for direct token validation.
  --mutation-ticket <value>      Authority ticket JSON string, @path, or path (high-risk token path).
  --attestation <name>           Satisfied authority attestation (repeatable).
  --attestation-object <value>   Structured attestation JSON string, @path, or path (repeatable).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    --force)
      FORCE=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --no-snapshot)
      REFRESH_SNAPSHOT=0; shift ;;
    --min-interval-sec)
      MIN_INTERVAL_SEC="${2:-}"; shift 2 ;;
    --no-telemetry)
      TELEMETRY_ENABLED=0; shift ;;
    --telemetry-cooldown-sec)
      TELEMETRY_COOLDOWN_SEC="${2:-}"; shift 2 ;;
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
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

guard_args=(
  --script "reconcile.sh"
  --risk-tier "high"
  --mutation-operation "reconcile:execute"
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

python3 - "$ROOT" "$JSON_OUT" "$FORCE" "$DRY_RUN" "$REFRESH_SNAPSHOT" "$MIN_INTERVAL_SEC" "$TELEMETRY_ENABLED" "$EVENT_ROUTER" "$TELEMETRY_COOLDOWN_SEC" <<'PY'
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
force = bool(int(sys.argv[3]))
dry_run = bool(int(sys.argv[4]))
refresh_snapshot = bool(int(sys.argv[5]))
try:
    min_interval_sec = max(0, int(sys.argv[6]))
except Exception:
    min_interval_sec = 1800
telemetry_enabled = bool(int(sys.argv[7]))
event_router = pathlib.Path(sys.argv[8]).resolve()
try:
    telemetry_cooldown_sec = max(0, int(sys.argv[9]))
except Exception:
    telemetry_cooldown_sec = 1800

now_script = root / "ops" / "openclaw" / "continuity" / "continuity_now.sh"
snapshot_script = root / "ops" / "openclaw" / "snapshot_ground_truth.sh"
write_checkpoint_script = root / "ops" / "openclaw" / "continuity" / "write_checkpoint.sh"
sync_latest_script = root / "ops" / "openclaw" / "continuity" / "sync_latest_artifacts.sh"
continuity_db_path = root / "state" / "continuity" / "continuity_os.sqlite"

hard_reasons = {"checkpoint_blocker", "verify_blocker", "critical_anomalies"}

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from continuity_policy import DRIFT_REASON_SET as _DRIFT_REASON_SET
except Exception:
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }

drift_reasons = set(_DRIFT_REASON_SET)

RECONCILE_EVENT_POLICY_VERSION = "continuity.reconcile.event_policy.v2"
RECONCILE_EVENT_POLICY = {
    "status_before_failed": {"event_key": "status.before_failed", "severity": "critical"},
    "refused_not_drift_only": {"event_key": "policy.refused_non_drift_only", "severity": "warn"},
    "cooldown_sync_failed": {"event_key": "cooldown.sync_failed", "severity": "critical"},
    "status_after_cooldown_failed": {"event_key": "cooldown.status_after_failed", "severity": "critical"},
    "cooldown_sync_remaining_drift": {"event_key": "cooldown.remaining_drift", "severity": "info"},
    "cooldown_sync_only": {"event_key": "cooldown.sync_only", "severity": "info"},
    "snapshot_failed": {"event_key": "execute.snapshot_failed", "severity": "critical"},
    "checkpoint_write_failed": {"event_key": "execute.checkpoint_write_failed", "severity": "critical"},
    "sync_failed": {"event_key": "execute.sync_failed", "severity": "critical"},
    "status_after_failed": {"event_key": "execute.status_after_failed", "severity": "critical"},
    "remaining_drift_after_checkpoint": {"event_key": "execute.remaining_drift", "severity": "warn"},
    "checkpoint_written": {"event_key": "execute.checkpoint_written", "severity": "info"},
}


def run(cmd: List[str], env_extra: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = os.environ.copy()
    if env_extra:
        for key, value in env_extra.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    cp = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return {
        "cmd": cmd,
        "rc": cp.returncode,
        "stdout": (cp.stdout or "").strip(),
        "stderr": (cp.stderr or "").strip(),
    }


def parse_now(payload: str) -> Dict[str, Any]:
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _text(raw: Any) -> str:
    return str(raw or "").strip()


def _string_list(raw: Any) -> List[str]:
    out: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        txt = _text(item)
        if txt:
            out.append(txt)
    return out


def load_json_if_exists(path: pathlib.Path) -> Dict[str, Any]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
    except Exception:
        return {}
    return {}


def load_latest_handover_checkpoint() -> Dict[str, Any]:
    surface_path = root / "state" / "continuity" / "latest" / "handover_latest.json"
    try:
        if surface_path.exists():
            obj = load_json_if_exists(surface_path) or {}
            preserved = obj.get("preserved_handover_checkpoint") if isinstance(obj.get("preserved_handover_checkpoint"), dict) else {}
            if preserved:
                return preserved

            checkpoint_ref = obj.get("checkpoint") if isinstance(obj.get("checkpoint"), dict) else {}
            checkpoint_rel = _text(checkpoint_ref.get("path"))
            if checkpoint_rel:
                checkpoint_path = (root / checkpoint_rel).resolve()
                checkpoint_obj = load_json_if_exists(checkpoint_path)
                if checkpoint_obj:
                    return checkpoint_obj

            resolved_obj = load_json_if_exists(surface_path.resolve())
            if resolved_obj:
                return resolved_obj
    except Exception:
        return {}
    return {}


def checkpoint_objective_text(checkpoint_obj: Dict[str, Any]) -> str:
    objective = checkpoint_obj.get("objective") if isinstance(checkpoint_obj.get("objective"), dict) else {}
    return _text(objective.get("primary_goal") or checkpoint_obj.get("objective"))


def is_drift_reconcile_objective(raw: Any) -> bool:
    txt = _text(raw).lower()
    return bool(txt and "reconcile continuity drift" in txt)


def handover_checkpoint_is_preservable(checkpoint_obj: Dict[str, Any]) -> bool:
    metadata = checkpoint_obj.get("metadata") if isinstance(checkpoint_obj.get("metadata"), dict) else {}
    objective = checkpoint_obj.get("objective") if isinstance(checkpoint_obj.get("objective"), dict) else {}
    trigger = _text(metadata.get("trigger"))
    status = _text(objective.get("status"))
    mission = checkpoint_objective_text(checkpoint_obj)
    if trigger != "post_completion_closeout":
        return False
    if status != "READY":
        return False
    if not mission or is_drift_reconcile_objective(mission):
        return False
    return True


def before_is_quiet_post_completion_steady_state(now_obj: Dict[str, Any]) -> bool:
    queue = now_obj.get("queue") if isinstance(now_obj.get("queue"), dict) else {}
    status_counts = queue.get("status_counts") if isinstance(queue.get("status_counts"), dict) else {}
    verify = now_obj.get("verify") if isinstance(now_obj.get("verify"), dict) else {}
    autopilot = now_obj.get("autopilot") if isinstance(now_obj.get("autopilot"), dict) else {}
    controller = autopilot.get("execution_frontier_controller") if isinstance(autopilot.get("execution_frontier_controller"), dict) else {}

    if _text(verify.get("status")) not in {"", "READY"}:
        return False
    try:
        if int(queue.get("ready_count") or 0) != 0:
            return False
    except Exception:
        return False
    try:
        if int(status_counts.get("RUNNING") or 0) != 0:
            return False
    except Exception:
        return False
    if _text(controller.get("selector_state")) != "idle_no_candidate":
        return False
    if _text(controller.get("decision")) not in {"", "SKIP"}:
        return False
    if _text(controller.get("skip_reason")) not in {"", "autonomous_dispatch_not_eligible"}:
        return False
    if bool(controller.get("post_completion_enforcement_required") is True):
        return False
    if bool(controller.get("post_completion_enforcement_latched") is True):
        return False
    if bool(controller.get("autonomous_execution_intent_active") is True):
        return False
    benign_reasons = {"selector_state_not_ready_for_dispatch", "next_candidate_missing"}
    block_reasons = _string_list(controller.get("block_reasons") or controller.get("autonomous_dispatch_block_reasons"))
    if any(reason not in benign_reasons for reason in block_reasons):
        return False
    return True


def persist_emitted_event(event_obj: Dict[str, Any]) -> None:
    if not isinstance(event_obj, dict):
        return
    event_id = str(event_obj.get("event_id") or "").strip()
    created_at = str(event_obj.get("created_at") or event_obj.get("timestamp") or "").strip()
    source = str(event_obj.get("source") or "").strip()
    key = str(event_obj.get("key") or event_obj.get("event_key") or "").strip()
    severity = str(event_obj.get("severity") or "info").strip() or "info"
    fingerprint = str(event_obj.get("fingerprint") or "").strip()
    route_key = str(event_obj.get("route_key") or f"{source}|{key}").strip()

    if not (event_id and created_at and source and key and fingerprint):
        return

    continuity_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(continuity_db_path)
    cur = con.cursor()
    cur.executescript(
        """
CREATE TABLE IF NOT EXISTS continuity_events (
  event_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source TEXT NOT NULL,
  event_key TEXT NOT NULL,
  severity TEXT NOT NULL CHECK(severity IN ('info','warn','critical')),
  fingerprint TEXT NOT NULL,
  emitted INTEGER NOT NULL,
  changed INTEGER NOT NULL,
  cooldown_elapsed INTEGER NOT NULL,
  suppress_reason TEXT,
  summary TEXT,
  evidence_ref TEXT,
  route_key TEXT NOT NULL,
  state_file TEXT,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_continuity_events_route_created ON continuity_events(route_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_continuity_events_key_created ON continuity_events(event_key, created_at DESC);
"""
    )
    cur.execute(
        """
INSERT OR REPLACE INTO continuity_events (
  event_id, created_at, source, event_key, severity, fingerprint,
  emitted, changed, cooldown_elapsed, suppress_reason, summary,
  evidence_ref, route_key, state_file, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            event_id,
            created_at,
            source,
            key,
            severity,
            fingerprint,
            int(bool(event_obj.get("emit", True))),
            int(bool(event_obj.get("changed", True))),
            int(bool(event_obj.get("cooldown_elapsed", True))),
            event_obj.get("suppress_reason"),
            event_obj.get("summary"),
            event_obj.get("evidence_ref"),
            route_key,
            event_obj.get("state_file"),
            json.dumps(event_obj, ensure_ascii=False, sort_keys=True),
        ),
    )
    con.commit()
    con.close()


def resolve_reconcile_event_policy(policy_code: str) -> Dict[str, str]:
    default = {
        "event_key": str(policy_code or "execute.unknown"),
        "severity": "info",
    }
    spec = RECONCILE_EVENT_POLICY.get(str(policy_code or "").strip())
    if not isinstance(spec, dict):
        return default
    event_key = str(spec.get("event_key") or default["event_key"]).strip() or default["event_key"]
    severity = str(spec.get("severity") or default["severity"]).strip() or default["severity"]
    if severity not in {"info", "warn", "critical"}:
        severity = "info"
    return {"event_key": event_key, "severity": severity}


def emit_reconcile_event(
    *,
    policy_code: str,
    summary: str,
    evidence_ref: str = "",
    fingerprint_input: str = "",
) -> Optional[Dict[str, Any]]:
    policy = resolve_reconcile_event_policy(policy_code)
    event_key = str(policy.get("event_key") or policy_code)
    severity = str(policy.get("severity") or "info")

    if not telemetry_enabled:
        return {
            "enabled": False,
            "source": "continuity.reconcile",
            "policy_version": RECONCILE_EVENT_POLICY_VERSION,
            "policy_code": policy_code,
            "event_key": event_key,
            "severity": severity,
            "reason": "telemetry_disabled",
        }

    if not event_router.exists() or not os.access(event_router, os.X_OK):
        return {
            "enabled": True,
            "ok": False,
            "source": "continuity.reconcile",
            "policy_version": RECONCILE_EVENT_POLICY_VERSION,
            "policy_code": policy_code,
            "event_key": event_key,
            "severity": severity,
            "error": "event_router_missing",
            "event_router": str(event_router),
        }

    cmd = [
        str(event_router),
        "--source",
        "continuity.reconcile",
        "--key",
        event_key,
        "--severity",
        severity,
        "--summary",
        summary,
        "--evidence-ref",
        evidence_ref,
        "--cooldown-sec",
        str(telemetry_cooldown_sec),
        "--no-persist",
    ]
    if fingerprint_input:
        cmd.extend(["--fingerprint-input", fingerprint_input])

    cp = subprocess.run(cmd, text=True, capture_output=True)
    router_payload = None
    out = (cp.stdout or "").strip()
    if out:
        try:
            maybe_obj = json.loads(out)
            if isinstance(maybe_obj, dict):
                router_payload = maybe_obj
        except Exception:
            router_payload = None

    if cp.returncode == 0 and isinstance(router_payload, dict):
        try:
            persist_emitted_event(router_payload)
        except Exception:
            pass

    return {
        "enabled": True,
        "ok": cp.returncode in (0, 20),
        "source": "continuity.reconcile",
        "policy_version": RECONCILE_EVENT_POLICY_VERSION,
        "policy_code": policy_code,
        "event_key": event_key,
        "severity": severity,
        "returncode": cp.returncode,
        "emitted": cp.returncode == 0,
        "suppressed": cp.returncode == 20,
        "stderr": (cp.stderr or "")[:300],
        "router": router_payload,
    }


before_run = run([str(now_script), "--json"])
if before_run["rc"] != 0:
    telemetry = emit_reconcile_event(
        policy_code="status_before_failed",
        summary="reconcile status_before_failed",
        evidence_ref="ops/openclaw/continuity/continuity_now.sh",
        fingerprint_input="stage=before|status=continuity_now_failed",
    )
    msg = {
        "ok": False,
        "error": "continuity_now_failed",
        "stage": "before",
        "stderr": before_run["stderr"][:400],
        "telemetry": telemetry,
    }
    print(json.dumps(msg, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)

before = parse_now(before_run["stdout"])
before_reasons = [str(r) for r in (before.get("not_ready_reasons") or [])]
before_ready = bool(before.get("ready"))
before_reason_sig = ",".join(sorted(before_reasons)) if before_reasons else "none"
freshness_drift_present = any(r in {"connector_freshness_drift", "policy_freshness_drift"} for r in before_reasons)

has_hard_reason = any(r in hard_reasons for r in before_reasons)
is_drift_only = bool(before_reasons) and (not has_hard_reason) and all(r in drift_reasons for r in before_reasons)

checkpoint_before = before.get("checkpoint") or {}
checkpoint_trigger = str(checkpoint_before.get("trigger") or "")
checkpoint_age_raw = checkpoint_before.get("age_sec")
preserved_handover_checkpoint = load_latest_handover_checkpoint()
preserve_handover_latest = bool(
    is_drift_only
    and before_is_quiet_post_completion_steady_state(before)
    and handover_checkpoint_is_preservable(preserved_handover_checkpoint)
)
preserved_handover_checkpoint_id = _text(
    ((preserved_handover_checkpoint.get("metadata") or {}).get("checkpoint_id"))
)
try:
    checkpoint_age_sec = int(checkpoint_age_raw)
except Exception:
    checkpoint_age_sec = None

cooldown_active = (
    (min_interval_sec > 0)
    and (checkpoint_trigger == "drift_reconcile")
    and (checkpoint_age_sec is not None)
    and (checkpoint_age_sec < min_interval_sec)
    and (not force)
)
cooldown_remaining_sec = max(0, min_interval_sec - int(checkpoint_age_sec or 0)) if cooldown_active else 0

if before_ready and not force:
    out = {
        "ok": True,
        "changed": False,
        "reconciled": True,
        "reason": "already_ready",
        "before": before,
        "after": before,
        "dry_run": dry_run,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("READY: continuity already ready; no reconcile needed")
    raise SystemExit(0)

if (not is_drift_only) and (not force):
    telemetry = emit_reconcile_event(
        policy_code="refused_not_drift_only",
        summary=f"reconcile refused not_drift_only reasons={before_reason_sig}",
        evidence_ref="state/continuity/latest/verify_last.json",
        fingerprint_input=f"reason=not_drift_only|reasons={before_reason_sig}",
    )
    out = {
        "ok": False,
        "changed": False,
        "reconciled": False,
        "reason": "not_drift_only",
        "before_reasons": before_reasons,
        "hint": "Use reconcile --force only after reviewing non-drift blockers.",
        "before": before,
        "dry_run": dry_run,
        "telemetry": telemetry,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        joined = ", ".join(before_reasons) if before_reasons else "none"
        print(f"BLOCKER: reconcile refused; not_ready_reasons={joined}")
    raise SystemExit(1)

if dry_run:
    planned_actions = ["snapshot_ground_truth" if refresh_snapshot else "snapshot_skipped"]
    if cooldown_active:
        planned_actions.append("sync_latest_artifacts (cooldown gate active; checkpoint write skipped)")
    else:
        if preserve_handover_latest:
            planned_actions.append(
                f"preserve handover_latest checkpoint={preserved_handover_checkpoint_id or 'unknown'} during drift_reconcile write"
            )
        planned_actions.extend([
            "write_checkpoint(trigger=drift_reconcile)",
            "sync_latest_artifacts",
        ])
    if freshness_drift_present:
        planned_actions.append("continuity_now --refresh (policy/connector freshness refresh)")

    out = {
        "ok": True,
        "changed": False,
        "reconciled": is_drift_only,
        "reason": "dry_run",
        "before_reasons": before_reasons,
        "cooldown_active": cooldown_active,
        "cooldown_remaining_sec": cooldown_remaining_sec,
        "min_interval_sec": min_interval_sec,
        "planned_actions": planned_actions,
        "handover_latest_preserved": preserve_handover_latest,
        "preserved_handover_checkpoint_id": preserved_handover_checkpoint_id or None,
        "before": before,
        "dry_run": True,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        joined = ", ".join(before_reasons) if before_reasons else "none"
        print(f"PROGRESS: reconcile dry-run; not_ready_reasons={joined}")
    raise SystemExit(0)

if cooldown_active:
    sync_run = run(
        [str(sync_latest_script)],
        env_extra={
            "OPENCLAW_INTERNAL_MUTATION": "1",
            "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "reconcile.sh:sync_latest_artifacts_cooldown",
        },
    )
    if sync_run["rc"] != 0:
        telemetry = emit_reconcile_event(
            policy_code="cooldown_sync_failed",
            summary="reconcile cooldown_sync_failed",
            evidence_ref="ops/openclaw/continuity/sync_latest_artifacts.sh",
            fingerprint_input=f"reason=cooldown_sync_failed|before={before_reason_sig}",
        )
        out = {
            "ok": False,
            "error": "sync_latest_artifacts_failed",
            "reason": "cooldown_active_recent_drift_reconcile",
            "cooldown_remaining_sec": cooldown_remaining_sec,
            "min_interval_sec": min_interval_sec,
            "stderr": sync_run["stderr"][:400],
            "stdout": sync_run["stdout"][:400],
            "before": before,
            "telemetry": telemetry,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
        raise SystemExit(1)

    after_cmd = [str(now_script), "--json"]
    if freshness_drift_present:
        after_cmd = [str(now_script), "--refresh", "--json"]
    after_run = run(after_cmd)
    if after_run["rc"] != 0:
        telemetry = emit_reconcile_event(
            policy_code="status_after_cooldown_failed",
            summary="reconcile status_after_cooldown_failed",
            evidence_ref="ops/openclaw/continuity/continuity_now.sh",
            fingerprint_input="stage=after_cooldown_sync|status=continuity_now_failed",
        )
        out = {
            "ok": False,
            "error": "continuity_now_failed",
            "stage": "after_cooldown_sync",
            "reason": "cooldown_active_recent_drift_reconcile",
            "cooldown_remaining_sec": cooldown_remaining_sec,
            "min_interval_sec": min_interval_sec,
            "stderr": after_run["stderr"][:400],
            "before": before,
            "telemetry": telemetry,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
        raise SystemExit(1)

    after = parse_now(after_run["stdout"])
    after_reasons = [str(r) for r in (after.get("not_ready_reasons") or [])]
    after_reason_sig = ",".join(sorted(after_reasons)) if after_reasons else "none"
    remaining_drift = [r for r in after_reasons if r in drift_reasons]

    try:
        sync_obj = json.loads(sync_run["stdout"])
    except Exception:
        sync_obj = {}

    if remaining_drift:
        telemetry = emit_reconcile_event(
            policy_code="cooldown_sync_remaining_drift",
            summary=f"reconcile cooldown_sync_remaining_drift reasons={after_reason_sig}",
            evidence_ref="state/continuity/latest/runtime_truth_bridge.json",
            fingerprint_input=f"reason=cooldown_remaining|after={after_reason_sig}|min={min_interval_sec}",
        )
    else:
        telemetry = emit_reconcile_event(
            policy_code="cooldown_sync_only",
            summary="reconcile cooldown_sync_only",
            evidence_ref="state/continuity/latest/runtime_truth_bridge.json",
            fingerprint_input=f"reason=cooldown_sync_only|min={min_interval_sec}",
        )

    out = {
        "ok": len(remaining_drift) == 0,
        "changed": True,
        "reconciled": len(remaining_drift) == 0,
        "reason": "cooldown_active_recent_drift_reconcile",
        "cooldown_active": True,
        "cooldown_remaining_sec": cooldown_remaining_sec,
        "min_interval_sec": min_interval_sec,
        "before_reasons": before_reasons,
        "after_reasons": after_reasons,
        "remaining_drift_reasons": remaining_drift,
        "before": before,
        "after": after,
        "sync_latest": sync_obj,
        "telemetry": telemetry,
    }

    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        joined = ", ".join(after_reasons) if after_reasons else "none"
        print(
            "PROGRESS: reconcile cooldown active; "
            f"remaining={cooldown_remaining_sec}s; after_not_ready={joined}"
        )

    if not out["reconciled"]:
        raise SystemExit(1)
    raise SystemExit(0)

snapshot_run = None
if refresh_snapshot:
    snapshot_run = run([str(snapshot_script)])
    if snapshot_run["rc"] != 0:
        telemetry = emit_reconcile_event(
            policy_code="snapshot_failed",
            summary="reconcile snapshot_ground_truth_failed",
            evidence_ref="ops/openclaw/snapshot_ground_truth.sh",
            fingerprint_input=f"reason=snapshot_failed|before={before_reason_sig}",
        )
        out = {
            "ok": False,
            "error": "snapshot_ground_truth_failed",
            "stderr": snapshot_run["stderr"][:400],
            "stdout": snapshot_run["stdout"][:400],
            "before": before,
            "telemetry": telemetry,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
        raise SystemExit(1)

objective = (
    "Reconcile continuity drift by refreshing checkpoint capture + bridge/handover "
    f"(before_reasons={','.join(before_reasons) or 'none'})"
)

write_cmd = [
    str(write_checkpoint_script),
    "--trigger", "drift_reconcile",
    "--status", "PROGRESS",
    "--objective", objective,
    "--next-action", "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/continuity_now.sh --strict",
    "--verify-cmd", "openclaw gateway status --json >/dev/null",
    "--verify-cmd", "openclaw cron list --json >/dev/null",
]
if preserve_handover_latest:
    write_cmd.append("--preserve-handover-latest")
write_run = run(
    write_cmd,
    env_extra={
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "reconcile.sh:write_checkpoint",
    },
)
if write_run["rc"] != 0:
    telemetry = emit_reconcile_event(
        policy_code="checkpoint_write_failed",
        summary="reconcile write_checkpoint_failed",
        evidence_ref="ops/openclaw/continuity/write_checkpoint.sh",
        fingerprint_input=f"reason=write_checkpoint_failed|before={before_reason_sig}",
    )
    out = {
        "ok": False,
        "error": "write_checkpoint_failed",
        "stderr": write_run["stderr"][:400],
        "stdout": write_run["stdout"][:400],
        "before": before,
        "telemetry": telemetry,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)

try:
    write_obj = json.loads(write_run["stdout"])
except Exception:
    write_obj = {}
checkpoint_id = str(write_obj.get("checkpoint_id") or "")

sync_cmd = [str(sync_latest_script)]
if checkpoint_id and not preserve_handover_latest:
    sync_cmd += ["--checkpoint", checkpoint_id]
sync_run = run(
    sync_cmd,
    env_extra={
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "reconcile.sh:sync_latest_artifacts_post_checkpoint",
    },
)
if sync_run["rc"] != 0:
    telemetry = emit_reconcile_event(
        policy_code="sync_failed",
        summary="reconcile sync_latest_artifacts_failed",
        evidence_ref="ops/openclaw/continuity/sync_latest_artifacts.sh",
        fingerprint_input=f"reason=sync_failed|checkpoint_id={checkpoint_id or 'none'}",
    )
    out = {
        "ok": False,
        "error": "sync_latest_artifacts_failed",
        "stderr": sync_run["stderr"][:400],
        "stdout": sync_run["stdout"][:400],
        "checkpoint_id": checkpoint_id or None,
        "handover_latest_preserved": preserve_handover_latest,
        "preserved_handover_checkpoint_id": preserved_handover_checkpoint_id or None,
        "before": before,
        "telemetry": telemetry,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)

try:
    sync_obj = json.loads(sync_run["stdout"])
except Exception:
    sync_obj = {}

after_cmd = [str(now_script), "--json"]
if freshness_drift_present:
    after_cmd = [str(now_script), "--refresh", "--json"]
after_run = run(after_cmd)
if after_run["rc"] != 0:
    telemetry = emit_reconcile_event(
        policy_code="status_after_failed",
        summary="reconcile status_after_failed",
        evidence_ref="ops/openclaw/continuity/continuity_now.sh",
        fingerprint_input=f"stage=after|checkpoint_id={checkpoint_id or 'none'}",
    )
    out = {
        "ok": False,
        "error": "continuity_now_failed",
        "stage": "after",
        "stderr": after_run["stderr"][:400],
        "checkpoint_id": checkpoint_id or None,
        "handover_latest_preserved": preserve_handover_latest,
        "preserved_handover_checkpoint_id": preserved_handover_checkpoint_id or None,
        "telemetry": telemetry,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)

after = parse_now(after_run["stdout"])
after_reasons = [str(r) for r in (after.get("not_ready_reasons") or [])]
after_reason_sig = ",".join(sorted(after_reasons)) if after_reasons else "none"
remaining_drift = [r for r in after_reasons if r in drift_reasons]

if remaining_drift:
    telemetry = emit_reconcile_event(
        policy_code="remaining_drift_after_checkpoint",
        summary=f"reconcile remaining_drift_after_checkpoint reasons={after_reason_sig}",
        evidence_ref="state/continuity/latest/runtime_truth_bridge.json",
        fingerprint_input=f"reason=remaining_drift|after={after_reason_sig}",
    )
else:
    telemetry = emit_reconcile_event(
        policy_code="checkpoint_written",
        summary=f"reconcile checkpoint_written checkpoint_id={checkpoint_id or 'n/a'}",
        evidence_ref=write_obj.get("json_path") or "state/continuity/latest/latest_pointer.json",
        fingerprint_input=f"reason=checkpoint_written|checkpoint_id={checkpoint_id or 'none'}",
    )

result = {
    "ok": True,
    "changed": True,
    "reconciled": len(remaining_drift) == 0,
    "before_reasons": before_reasons,
    "after_reasons": after_reasons,
    "remaining_drift_reasons": remaining_drift,
    "checkpoint_id": checkpoint_id or None,
    "handover_latest_preserved": preserve_handover_latest,
    "preserved_handover_checkpoint_id": preserved_handover_checkpoint_id or None,
    "snapshot_refreshed": bool(refresh_snapshot),
    "min_interval_sec": min_interval_sec,
    "telemetry_enabled": telemetry_enabled,
    "telemetry": telemetry,
    "write_checkpoint": write_obj,
    "sync_latest": sync_obj,
    "before": before,
    "after": after,
}

if json_out:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("CONTINUITY RECONCILE")
    print(f"- checkpoint: {checkpoint_id or 'n/a'}")
    print(f"- before_not_ready: {', '.join(before_reasons) if before_reasons else 'none'}")
    print(f"- after_not_ready: {', '.join(after_reasons) if after_reasons else 'none'}")
    print(f"- reconciled: {result['reconciled']}")

if not result["reconciled"]:
    raise SystemExit(1)
PY
