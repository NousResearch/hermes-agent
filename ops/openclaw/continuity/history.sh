#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0
WINDOW_HOURS="${OPENCLAW_CONTINUITY_HISTORY_WINDOW_HOURS:-72}"
LIMIT="${OPENCLAW_CONTINUITY_HISTORY_LIMIT:-20}"
INCLUDE_SUPPRESSED=0
SOURCE_PRESET="${OPENCLAW_CONTINUITY_HISTORY_SOURCE_PRESET:-reconcile}"
LEGACY_RECONCILE_SOURCE="${OPENCLAW_CONTINUITY_RECONCILE_SOURCE:-}"
LEGACY_RECONCILE_TRIGGER="${OPENCLAW_CONTINUITY_RECONCILE_TRIGGER:-drift_reconcile}"
TRIGGER_FILTER_SET=0
TRIGGER_FILTER_VALUE=""
SINCE_CHECKPOINT=""
UNTIL_VALUE=""
SOURCE_FILTERS=()
TASK_FILTERS=()
ACTOR_ROLE_FILTERS=()

usage() {
  cat <<'EOF'
Usage: history.sh [options]

Continuity audit rollup/history surface built from continuity DB + checkpoints + continuity events.

Options:
  --json                         Print machine JSON output.
  --hours <n>                    Window size in hours when --since-checkpoint is not set (default: 72).
  --limit <n>                    Max rows per recent list (default: 20).
  --include-suppressed           Include suppressed continuity events in recent event list.

  --since-checkpoint <ref>       Start range at checkpoint timestamp.
                                 Ref supports: <checkpoint_id>, latest, latest:<trigger>.
  --until <ref-or-iso>           End range at ISO timestamp or checkpoint ref.
                                 Ref supports: <checkpoint_id>, latest, latest:<trigger>.

  --source-preset <name>         Event source preset:
                                   reconcile (default), watchdogs, control-plane, autopilot, all
  --source <name-or-pattern>     Add explicit event source filter. Repeatable.
                                 Supports SQL-like '%' wildcard or shell '*' wildcard.
  --sources <csv>                Add comma-separated explicit event source filters.

  --task <id-or-pattern>         Filter work_queue/task_transitions by task id. Repeatable.
                                 Supports SQL-like '%' wildcard or shell '*' wildcard.
  --tasks <csv>                  Add comma-separated task filters.
  --actor-role <name-or-pattern> Filter task_transitions by actor role. Repeatable.
  --actor-roles <csv>            Add comma-separated actor-role filters.

  --trigger <name|any>           Checkpoint trigger filter for recent checkpoint lists.
                                 Default comes from source preset (reconcile => drift_reconcile).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    --hours)
      WINDOW_HOURS="${2:-}"; shift 2 ;;
    --limit)
      LIMIT="${2:-}"; shift 2 ;;
    --include-suppressed)
      INCLUDE_SUPPRESSED=1; shift ;;
    --since-checkpoint)
      SINCE_CHECKPOINT="${2:-}"; shift 2 ;;
    --until)
      UNTIL_VALUE="${2:-}"; shift 2 ;;
    --source-preset)
      SOURCE_PRESET="${2:-}"; shift 2 ;;
    --source)
      SOURCE_FILTERS+=("${2:-}"); shift 2 ;;
    --sources)
      IFS=',' read -r -a _csv_parts <<< "${2:-}"
      for _part in "${_csv_parts[@]}"; do
        _trimmed="${_part#${_part%%[![:space:]]*}}"
        _trimmed="${_trimmed%${_trimmed##*[![:space:]]}}"
        [[ -n "$_trimmed" ]] && SOURCE_FILTERS+=("$_trimmed")
      done
      shift 2 ;;
    --task)
      TASK_FILTERS+=("${2:-}"); shift 2 ;;
    --tasks)
      IFS=',' read -r -a _csv_parts <<< "${2:-}"
      for _part in "${_csv_parts[@]}"; do
        _trimmed="${_part#${_part%%[![:space:]]*}}"
        _trimmed="${_trimmed%${_trimmed##*[![:space:]]}}"
        [[ -n "$_trimmed" ]] && TASK_FILTERS+=("$_trimmed")
      done
      shift 2 ;;
    --actor-role)
      ACTOR_ROLE_FILTERS+=("${2:-}"); shift 2 ;;
    --actor-roles)
      IFS=',' read -r -a _csv_parts <<< "${2:-}"
      for _part in "${_csv_parts[@]}"; do
        _trimmed="${_part#${_part%%[![:space:]]*}}"
        _trimmed="${_trimmed%${_trimmed##*[![:space:]]}}"
        [[ -n "$_trimmed" ]] && ACTOR_ROLE_FILTERS+=("$_trimmed")
      done
      shift 2 ;;
    --trigger)
      TRIGGER_FILTER_SET=1
      TRIGGER_FILTER_VALUE="${2:-}"
      shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

SOURCE_FILTERS_CSV=""
if [[ ${#SOURCE_FILTERS[@]} -gt 0 ]]; then
  SOURCE_FILTERS_CSV="$(IFS=,; echo "${SOURCE_FILTERS[*]}")"
fi

TASK_FILTERS_CSV=""
if [[ ${#TASK_FILTERS[@]} -gt 0 ]]; then
  TASK_FILTERS_CSV="$(IFS=,; echo "${TASK_FILTERS[*]}")"
fi

ACTOR_ROLE_FILTERS_CSV=""
if [[ ${#ACTOR_ROLE_FILTERS[@]} -gt 0 ]]; then
  ACTOR_ROLE_FILTERS_CSV="$(IFS=,; echo "${ACTOR_ROLE_FILTERS[*]}")"
fi

python3 - "$ROOT" "$JSON_OUT" "$WINDOW_HOURS" "$LIMIT" "$INCLUDE_SUPPRESSED" "$SOURCE_PRESET" "$SOURCE_FILTERS_CSV" "$TRIGGER_FILTER_SET" "$TRIGGER_FILTER_VALUE" "$SINCE_CHECKPOINT" "$UNTIL_VALUE" "$LEGACY_RECONCILE_SOURCE" "$LEGACY_RECONCILE_TRIGGER" "$TASK_FILTERS_CSV" "$ACTOR_ROLE_FILTERS_CSV" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
try:
    window_hours = max(1, int(sys.argv[3]))
except Exception:
    window_hours = 72
try:
    limit = max(1, min(200, int(sys.argv[4])))
except Exception:
    limit = 20
include_suppressed = bool(int(sys.argv[5]))
source_preset_raw = str(sys.argv[6] or "reconcile").strip().lower()
source_filters_csv = str(sys.argv[7] or "").strip()
trigger_filter_set = bool(int(sys.argv[8]))
trigger_filter_value_raw = str(sys.argv[9] or "").strip()
since_checkpoint_ref = str(sys.argv[10] or "").strip()
until_ref_or_iso = str(sys.argv[11] or "").strip()
legacy_reconcile_source = str(sys.argv[12] or "").strip()
legacy_reconcile_trigger = str(sys.argv[13] or "").strip()
task_filters_csv = str(sys.argv[14] or "").strip()
actor_role_filters_csv = str(sys.argv[15] or "").strip()

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from continuity_policy import (
        is_severe_verify_gate_preflight_blocker as _policy_is_severe_verify_gate_preflight_blocker,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS = {"strict_autonomy_required_override_denied"}
    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES = (
        "layered_health_gate:",
        "execution_supervisor_launch_readiness_severity_gate:",
        "execution_supervisor_probe_execution_gate:",
        "execution_supervisor_worker_health_canary_gate:",
    )

    def _policy_is_severe_verify_gate_preflight_blocker(reason: Any) -> bool:
        blocker = str(reason or "").strip()
        if not blocker:
            return False
        if blocker in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS:
            return True
        return any(blocker.startswith(prefix) for prefix in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES)


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


db_path = root / "state" / "continuity" / "continuity_os.sqlite"
checkpoints_dir = root / "state" / "continuity" / "checkpoints"
latest_pointer_path = root / "state" / "continuity" / "latest" / "latest_pointer.json"
continuity_now_latest_path = root / "state" / "continuity" / "latest" / "continuity_now_latest.json"
continuity_current_path = root / "state" / "continuity" / "current.json"
blocker_registry_path = root / "state" / "continuity" / "latest" / "blocker_registry.json"


PRESETS = {
    "reconcile": {
        "sources": ["continuity.reconcile"],
        "default_trigger": "drift_reconcile",
        "default_task_filters": [],
    },
    "watchdog": {
        "sources": ["watchdog.%", "runtime.slo_staleness"],
        "default_trigger": None,
        "default_task_filters": [],
    },
    "watchdogs": {
        "sources": ["watchdog.%", "runtime.slo_staleness"],
        "default_trigger": None,
        "default_task_filters": [],
    },
    "control-plane": {
        "sources": ["continuity.%", "watchdog.%", "runtime.%", "local.context_runtime_watch"],
        "default_trigger": None,
        "default_task_filters": [],
    },
    "control_plane": {
        "sources": ["continuity.%", "watchdog.%", "runtime.%", "local.context_runtime_watch"],
        "default_trigger": None,
        "default_task_filters": [],
    },
    "autopilot": {
        "sources": ["watchdog.%", "runtime.%", "local.context_runtime_watch", "continuity.%"],
        "default_trigger": None,
        "default_task_filters": ["autopilot:%"],
    },
    "all": {
        "sources": [],
        "default_trigger": None,
        "default_task_filters": [],
    },
}

def is_severe_verify_gate_preflight_blocker(reason: Optional[str]) -> bool:
    return bool(_policy_is_severe_verify_gate_preflight_blocker(reason))


def parse_iso(value: str):
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def to_iso_z(value: dt.datetime) -> str:
    out = value.astimezone(dt.timezone.utc).replace(microsecond=0)
    return out.isoformat().replace("+00:00", "Z")


def age_sec(value: str):
    d = parse_iso(value)
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


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_source_filters(raw_csv: str, preset_name: str, legacy_source: str) -> Dict[str, Any]:
    explicit = [x.strip() for x in raw_csv.split(",") if x.strip()]
    effective_preset = preset_name if preset_name in PRESETS else "reconcile"
    preset = PRESETS.get(effective_preset) or PRESETS["reconcile"]
    if explicit:
        return {
            "source_preset": effective_preset,
            "source_filters": explicit,
            "source_filter_origin": "explicit",
            "default_trigger": preset.get("default_trigger"),
        }

    if effective_preset == "reconcile" and legacy_source:
        return {
            "source_preset": effective_preset,
            "source_filters": [legacy_source],
            "source_filter_origin": "legacy_env",
            "default_trigger": preset.get("default_trigger"),
        }

    return {
        "source_preset": effective_preset,
        "source_filters": list(preset.get("sources") or []),
        "source_filter_origin": "preset",
        "default_trigger": preset.get("default_trigger"),
    }


def normalize_task_filters(raw_csv: str, preset_name: str) -> Dict[str, Any]:
    explicit = [x.strip() for x in raw_csv.split(",") if x.strip()]
    effective_preset = preset_name if preset_name in PRESETS else "reconcile"
    preset = PRESETS.get(effective_preset) or PRESETS["reconcile"]

    if explicit:
        return {
            "task_filters": explicit,
            "task_filter_origin": "explicit",
        }

    preset_defaults = [x for x in (preset.get("default_task_filters") or []) if str(x).strip()]
    if preset_defaults:
        return {
            "task_filters": preset_defaults,
            "task_filter_origin": "preset",
        }

    return {
        "task_filters": [],
        "task_filter_origin": "none",
    }


def normalize_actor_role_filters(raw_csv: str) -> Dict[str, Any]:
    explicit = [x.strip() for x in raw_csv.split(",") if x.strip()]
    return {
        "actor_role_filters": explicit,
        "actor_role_filter_origin": "explicit" if explicit else "none",
    }


def normalize_trigger_filter(
    trigger_set: bool,
    trigger_value: str,
    default_trigger: Optional[str],
    preset_name: str,
    legacy_trigger: str,
) -> Optional[str]:
    if trigger_set:
        raw = trigger_value.strip().lower()
        if raw in {"", "any", "all", "*", "none"}:
            return None
        return trigger_value.strip()

    if preset_name == "reconcile" and legacy_trigger:
        return legacy_trigger

    return default_trigger


def db_checkpoint_created_at(con: sqlite3.Connection, checkpoint_id: str) -> Optional[str]:
    row = con.execute(
        "SELECT created_at FROM checkpoints WHERE checkpoint_id = ? ORDER BY created_at DESC LIMIT 1",
        (checkpoint_id,),
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    return None


def db_latest_checkpoint_created_at(con: sqlite3.Connection, trigger: Optional[str]) -> Optional[Dict[str, str]]:
    if trigger:
        row = con.execute(
            "SELECT checkpoint_id, created_at FROM checkpoints WHERE trigger = ? ORDER BY created_at DESC LIMIT 1",
            (trigger,),
        ).fetchone()
    else:
        row = con.execute(
            "SELECT checkpoint_id, created_at FROM checkpoints ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row and row[0] and row[1]:
        return {"checkpoint_id": str(row[0]), "created_at": str(row[1]), "resolved_from": "db"}
    return None


def file_checkpoint_created_at(checkpoint_ref: str) -> Optional[Dict[str, str]]:
    candidates: List[pathlib.Path] = []
    raw = checkpoint_ref.strip()
    if not raw:
        return None

    path_like = pathlib.Path(raw)
    if path_like.suffix == ".json":
        candidates.append(path_like if path_like.is_absolute() else (root / path_like))

    cid = raw
    if not cid.startswith("chk_") and cid:
        cid = f"chk_{cid}"
    candidates.append(checkpoints_dir / f"{cid}.json")

    for p in candidates:
        if not p.exists():
            continue
        obj = load_json(p)
        meta = obj.get("metadata") or {}
        created_at = str(meta.get("created_at") or "").strip()
        checkpoint_id = str(meta.get("checkpoint_id") or p.stem)
        if created_at:
            return {
                "checkpoint_id": checkpoint_id,
                "created_at": created_at,
                "resolved_from": "file",
            }
    return None


def resolve_checkpoint_ref(ref: str, con: Optional[sqlite3.Connection]) -> Optional[Dict[str, str]]:
    raw = (ref or "").strip()
    if not raw:
        return None

    if raw.startswith("latest"):
        trigger = None
        if raw.startswith("latest:"):
            trigger = raw.split(":", 1)[1].strip() or None
        if con is not None:
            latest = db_latest_checkpoint_created_at(con, trigger)
            if latest:
                return latest
        lp = load_json(latest_pointer_path)
        if raw == "latest" and lp:
            cid = str(lp.get("checkpoint_id") or "").strip()
            created_at = ""
            json_path = str(lp.get("json_path") or "").strip()
            if json_path:
                chk = load_json(root / json_path)
                created_at = str((chk.get("metadata") or {}).get("created_at") or "").strip()
            if cid and created_at:
                return {
                    "checkpoint_id": cid,
                    "created_at": created_at,
                    "resolved_from": "latest_pointer",
                }
        return None

    checkpoint_id = raw
    if con is not None:
        created_at = db_checkpoint_created_at(con, checkpoint_id)
        if created_at:
            return {
                "checkpoint_id": checkpoint_id,
                "created_at": created_at,
                "resolved_from": "db",
            }

    file_match = file_checkpoint_created_at(raw)
    if file_match:
        return file_match

    return None


def build_like_where(column: str, patterns: List[str]):
    if not patterns:
        return "", []
    clauses: List[str] = []
    params: List[Any] = []
    for raw in patterns:
        token = raw.strip()
        if not token:
            continue
        if "*" in token:
            token = token.replace("*", "%")
        if "%" in token:
            clauses.append(f"{column} LIKE ?")
            params.append(token)
        else:
            clauses.append(f"{column} = ?")
            params.append(token)
    if not clauses:
        return "", []
    return " AND (" + " OR ".join(clauses) + ")", params


def compute_verify_gate_preflight_posture(now_payload: Dict[str, Any]) -> Dict[str, Any]:
    verify = now_payload.get("verify") if isinstance(now_payload.get("verify"), dict) else {}
    gate_preflight = verify.get("gate_preflight") if isinstance(verify.get("gate_preflight"), dict) else {}
    if not gate_preflight:
        return {
            "mode": "unknown",
            "source": "unavailable",
            "ready_to_run": None,
            "predicted_blocker_reason": None,
            "severity": "warn",
        }

    strict_mode = gate_preflight.get("strict_autonomy") if isinstance(gate_preflight.get("strict_autonomy"), dict) else {}
    predicted = gate_preflight.get("predicted_gate") if isinstance(gate_preflight.get("predicted_gate"), dict) else {}

    available = bool(gate_preflight.get("available") is True)
    enabled = bool(strict_mode.get("enabled") is True)
    required = strict_mode.get("required") if isinstance(strict_mode.get("required"), bool) else None
    source = str(strict_mode.get("source") or "disabled").strip() or "disabled"
    ready_to_run = predicted.get("ready_to_run") if isinstance(predicted.get("ready_to_run"), bool) else None
    predicted_blocker = str(predicted.get("predicted_blocker_reason") or "").strip() or None

    if not available:
        mode = "unknown"
    elif enabled and required is True:
        mode = "required"
    elif enabled:
        mode = "enabled"
    else:
        mode = "disabled"

    severity = "info"
    if not available:
        severity = "warn"
    elif predicted_blocker:
        severity = "blocker" if is_severe_verify_gate_preflight_blocker(predicted_blocker) else "warn"
    elif ready_to_run is False:
        severity = "warn"

    return {
        "mode": mode,
        "source": source,
        "ready_to_run": ready_to_run,
        "predicted_blocker_reason": predicted_blocker,
        "severity": severity,
    }


def summarize_runtime_operator_failures(
    *,
    current_payload: Dict[str, Any],
    blocker_registry_payload: Dict[str, Any],
    open_limit: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    source = "unavailable"
    source_path = None
    generated_at = None
    status = "unknown"
    readiness = None
    mutation_gate_status = None

    registry_rows = blocker_registry_payload.get("blockers") if isinstance(blocker_registry_payload.get("blockers"), list) else None
    if registry_rows is not None:
        source = "blocker_registry"
        source_path = rel(blocker_registry_path)
        generated_at = blocker_registry_payload.get("generated_at")
        status = str(blocker_registry_payload.get("status") or "unknown").strip() or "unknown"
        readiness = str(blocker_registry_payload.get("readiness") or "").strip() or None
        mutation_gate = blocker_registry_payload.get("mutation_gate") if isinstance(blocker_registry_payload.get("mutation_gate"), dict) else {}
        mutation_gate_status = str(mutation_gate.get("status") or "").strip() or None
        for raw in registry_rows:
            if not isinstance(raw, dict):
                continue
            reason = str(raw.get("reason") or "").strip() or "unspecified"
            severity = str(raw.get("severity") or "warn").strip().lower() or "warn"
            if severity not in {"blocker", "warn", "info"}:
                severity = "warn"
            evidence = [str(x).strip() for x in (raw.get("evidence") or []) if str(x).strip()]
            rows.append(
                {
                    "reason": reason,
                    "severity": severity,
                    "status": str(raw.get("status") or "open").strip() or "open",
                    "owner": str(raw.get("owner") or "").strip() or None,
                    "evidence": evidence,
                    "recommended_action": str(raw.get("recommended_action") or "").strip() or None,
                    "action_required": raw.get("action_required") is True,
                    "context": raw.get("context") if isinstance(raw.get("context"), dict) else None,
                }
            )
    elif current_payload:
        source = "continuity_current"
        source_path = rel(continuity_current_path)
        generated_at = current_payload.get("generated_at")
        readiness = str(current_payload.get("readiness") or "").strip() or None
        mutation_gate = current_payload.get("mutation_gate") if isinstance(current_payload.get("mutation_gate"), dict) else {}
        mutation_gate_status = str(mutation_gate.get("status") or "").strip() or None
        status = "BLOCKER" if mutation_gate_status == "forbidden" else "READY"
        doctrine_surface = current_payload.get("doctrine_drift") if isinstance(current_payload.get("doctrine_drift"), dict) else {}
        doctrine_evidence_by_reason: Dict[str, List[str]] = {}
        for row in (doctrine_surface.get("open") or []):
            if not isinstance(row, dict):
                continue
            reason_key = str(row.get("reason") or "").strip()
            if not reason_key:
                continue
            evidence_rows = [str(x).strip() for x in (row.get("evidence") or []) if str(x).strip()]
            if evidence_rows:
                doctrine_evidence_by_reason[reason_key] = evidence_rows

        for raw in (current_payload.get("drifts") or []):
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("code") or "").strip() or "UNKNOWN"
            detail = str(raw.get("detail") or "").strip()
            severity = "warn" if code == "CONTINUITY_WARNING" else "blocker"

            evidence_rows = [f"{rel(continuity_current_path)}#drifts"]
            if code == "CONTINUITY_WARNING" and detail.startswith("doctrine_drift:"):
                doctrine_reason = detail.split(":", 1)[1].strip()
                if doctrine_reason:
                    evidence_rows.append(f"{rel(continuity_current_path)}#doctrine_drift")
                    evidence_rows.extend(doctrine_evidence_by_reason.get(doctrine_reason) or [])

            evidence: List[str] = []
            seen_evidence = set()
            for ref in evidence_rows:
                if not ref or ref in seen_evidence:
                    continue
                seen_evidence.add(ref)
                evidence.append(ref)

            rows.append(
                {
                    "reason": detail or code,
                    "severity": severity,
                    "status": "open",
                    "owner": "continuity_current",
                    "evidence": evidence,
                    "drift_code": code,
                }
            )

    blocker_total = sum(1 for row in rows if row.get("severity") == "blocker")
    warn_total = sum(1 for row in rows if row.get("severity") == "warn")
    info_total = sum(1 for row in rows if row.get("severity") == "info")

    top_reason_counts: Dict[str, int] = {}
    evidence_refs: List[str] = []
    evidence_seen = set()
    for row in rows:
        reason = str(row.get("reason") or "unspecified").strip() or "unspecified"
        top_reason_counts[reason] = top_reason_counts.get(reason, 0) + 1
        for ref in (row.get("evidence") or []):
            if ref in evidence_seen:
                continue
            evidence_seen.add(ref)
            evidence_refs.append(ref)

    top_reasons = {
        reason: count
        for reason, count in sorted(top_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    }

    return {
        "source": source,
        "source_path": source_path,
        "generated_at": generated_at,
        "status": status,
        "readiness": readiness,
        "mutation_gate_status": mutation_gate_status,
        "open_total": len(rows),
        "blocker_total": blocker_total,
        "warn_total": warn_total,
        "info_total": info_total,
        "top_reasons": top_reasons,
        "evidence_refs": evidence_refs[:10],
        "open": rows[: max(1, open_limit)],
    }


con: Optional[sqlite3.Connection] = None
if db_path.exists():
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
    except Exception:
        con = None

source_meta = normalize_source_filters(source_filters_csv, source_preset_raw, legacy_reconcile_source)
source_preset = source_meta["source_preset"]
source_filters = source_meta["source_filters"]
source_filter_origin = source_meta["source_filter_origin"]
default_trigger = source_meta.get("default_trigger")

task_filter_meta = normalize_task_filters(task_filters_csv, source_preset)
task_filters = task_filter_meta["task_filters"]
task_filter_origin = task_filter_meta["task_filter_origin"]

actor_role_meta = normalize_actor_role_filters(actor_role_filters_csv)
actor_role_filters = actor_role_meta["actor_role_filters"]
actor_role_filter_origin = actor_role_meta["actor_role_filter_origin"]

trigger_filter = normalize_trigger_filter(
    trigger_filter_set,
    trigger_filter_value_raw,
    default_trigger,
    source_preset,
    legacy_reconcile_trigger,
)

now_dt = clock_now_dt()
range_source = "hours"
since_checkpoint_resolved = None
if since_checkpoint_ref:
    since_checkpoint_resolved = resolve_checkpoint_ref(since_checkpoint_ref, con)
    if not since_checkpoint_resolved:
        msg = {"ok": False, "error": "since_checkpoint_not_found", "since_checkpoint": since_checkpoint_ref}
        print(json.dumps(msg, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    start_dt = parse_iso(str(since_checkpoint_resolved.get("created_at") or ""))
    if start_dt is None:
        msg = {
            "ok": False,
            "error": "since_checkpoint_timestamp_invalid",
            "since_checkpoint": since_checkpoint_ref,
            "resolved": since_checkpoint_resolved,
        }
        print(json.dumps(msg, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    range_source = "since_checkpoint"
else:
    start_dt = now_dt - dt.timedelta(hours=window_hours)

until_resolved = None
if until_ref_or_iso:
    parsed_until = parse_iso(until_ref_or_iso)
    if parsed_until is not None:
        end_dt = parsed_until
        until_resolved = {"raw": until_ref_or_iso, "resolved_from": "iso", "created_at": to_iso_z(parsed_until)}
    else:
        until_resolved = resolve_checkpoint_ref(until_ref_or_iso, con)
        if not until_resolved:
            msg = {"ok": False, "error": "until_not_found_or_invalid", "until": until_ref_or_iso}
            print(json.dumps(msg, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        end_dt = parse_iso(str(until_resolved.get("created_at") or ""))
        if end_dt is None:
            msg = {
                "ok": False,
                "error": "until_timestamp_invalid",
                "until": until_ref_or_iso,
                "resolved": until_resolved,
            }
            print(json.dumps(msg, ensure_ascii=False, indent=2))
            raise SystemExit(2)
else:
    end_dt = now_dt

if start_dt.tzinfo is None:
    start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
if end_dt.tzinfo is None:
    end_dt = end_dt.replace(tzinfo=dt.timezone.utc)

if end_dt < start_dt:
    msg = {
        "ok": False,
        "error": "invalid_time_range",
        "from_utc": to_iso_z(start_dt),
        "until_utc": to_iso_z(end_dt),
    }
    print(json.dumps(msg, ensure_ascii=False, indent=2))
    raise SystemExit(2)

start_iso = to_iso_z(start_dt)
end_iso = to_iso_z(end_dt)

recent_checkpoints_db: List[Dict[str, Any]] = []
recent_events_db: List[Dict[str, Any]] = []
recent_checkpoint_files: List[Dict[str, Any]] = []
recent_transitions_db: List[Dict[str, Any]] = []
recent_transition_evidence_refs: List[str] = []
checkpoint_trigger_counts: Dict[str, int] = {}
event_source_counts: Dict[str, int] = {}
queue_status_counts: Dict[str, int] = {}
queue_role_required_counts: Dict[str, int] = {}
queue_dependency_blocked_examples: List[Dict[str, Any]] = []
active_file_locks: Dict[str, Any] = {
    "active_count": 0,
    "stale_active_count": 0,
    "examples": [],
}
transition_rollup = {
    "total": 0,
    "to_status": {},
    "actor_role": {},
    "reason_top": {},
}
event_rollup = {
    "total": 0,
    "emitted": 0,
    "suppressed": 0,
    "severity": {"info": 0, "warn": 0, "critical": 0},
}

verify_gate_preflight_posture = {
    "mode": "unknown",
    "source": "unavailable",
    "ready_to_run": None,
    "predicted_blocker_reason": None,
    "severity": "warn",
}
continuity_now_latest = load_json(continuity_now_latest_path)
if continuity_now_latest:
    verify_gate_preflight_posture = compute_verify_gate_preflight_posture(continuity_now_latest)

continuity_current = load_json(continuity_current_path)
blocker_registry = load_json(blocker_registry_path)
runtime_operator_failures = summarize_runtime_operator_failures(
    current_payload=continuity_current,
    blocker_registry_payload=blocker_registry,
    open_limit=limit,
)
headline_runtime_operator_top_reason = next(iter(runtime_operator_failures.get("top_reasons") or {}), None)

source_where_sql, source_where_params = build_like_where("source", source_filters)
task_where_sql, task_where_params = build_like_where("task_id", task_filters)
actor_where_sql, actor_where_params = build_like_where("actor_role", actor_role_filters)

if con is not None:
    try:
        cur = con.cursor()

        cp_where = "created_at >= ? AND created_at <= ?"
        cp_params: List[Any] = [start_iso, end_iso]

        for row in cur.execute(
            f"SELECT trigger, COUNT(*) FROM checkpoints WHERE {cp_where} GROUP BY trigger"
            , cp_params
        ).fetchall():
            trig = str(row[0] or "")
            checkpoint_trigger_counts[trig] = int(row[1] or 0)

        cp_recent_where = cp_where
        cp_recent_params = list(cp_params)
        if trigger_filter:
            cp_recent_where += " AND trigger = ?"
            cp_recent_params.append(trigger_filter)

        for row in cur.execute(
            f"""
SELECT checkpoint_id, created_at, trigger, status, objective, json_path, md_path
FROM checkpoints
WHERE {cp_recent_where}
ORDER BY created_at DESC
LIMIT ?
""",
            (*cp_recent_params, limit),
        ).fetchall():
            recent_checkpoints_db.append(
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "created_at": row["created_at"],
                    "age_sec": age_sec(str(row["created_at"] or "")),
                    "trigger": row["trigger"],
                    "status": row["status"],
                    "objective": row["objective"],
                    "json_path": row["json_path"],
                    "md_path": row["md_path"],
                }
            )

        ev_where = "created_at >= ? AND created_at <= ?"
        ev_params: List[Any] = [start_iso, end_iso]
        if source_where_sql:
            ev_where += source_where_sql
            ev_params.extend(source_where_params)
        if not include_suppressed:
            ev_where += " AND emitted = 1"

        for row in cur.execute(
            f"""
SELECT created_at, source, event_key, severity, emitted, changed, cooldown_elapsed,
       suppress_reason, summary, evidence_ref
FROM continuity_events
WHERE {ev_where}
ORDER BY created_at DESC
LIMIT ?
""",
            (*ev_params, limit),
        ).fetchall():
            recent_events_db.append(
                {
                    "created_at": row["created_at"],
                    "age_sec": age_sec(str(row["created_at"] or "")),
                    "source": row["source"],
                    "event_key": row["event_key"],
                    "severity": row["severity"],
                    "emitted": bool(row["emitted"]),
                    "changed": bool(row["changed"]),
                    "cooldown_elapsed": bool(row["cooldown_elapsed"]),
                    "suppress_reason": row["suppress_reason"],
                    "summary": row["summary"],
                    "evidence_ref": row["evidence_ref"],
                }
            )

        ev_rollup_where = "created_at >= ? AND created_at <= ?"
        ev_rollup_params: List[Any] = [start_iso, end_iso]
        if source_where_sql:
            ev_rollup_where += source_where_sql
            ev_rollup_params.extend(source_where_params)

        for row in cur.execute(
            f"""
SELECT severity,
       COUNT(*) AS total_count,
       SUM(CASE WHEN emitted = 1 THEN 1 ELSE 0 END) AS emitted_count,
       SUM(CASE WHEN emitted = 0 THEN 1 ELSE 0 END) AS suppressed_count
FROM continuity_events
WHERE {ev_rollup_where}
GROUP BY severity
""",
            ev_rollup_params,
        ).fetchall():
            sev = str(row["severity"] or "info")
            total_count = int(row["total_count"] or 0)
            emitted_count = int(row["emitted_count"] or 0)
            suppressed_count = int(row["suppressed_count"] or 0)
            event_rollup["total"] += total_count
            event_rollup["emitted"] += emitted_count
            event_rollup["suppressed"] += suppressed_count
            event_rollup["severity"][sev] = int(event_rollup["severity"].get(sev) or 0) + total_count

        for row in cur.execute(
            f"""
SELECT source, COUNT(*) AS total_count
FROM continuity_events
WHERE {ev_rollup_where}
GROUP BY source
ORDER BY total_count DESC
LIMIT 20
""",
            ev_rollup_params,
        ).fetchall():
            event_source_counts[str(row["source"] or "")] = int(row["total_count"] or 0)

        queue_where = "1=1"
        queue_params: List[Any] = []
        if task_where_sql:
            queue_where += task_where_sql
            queue_params.extend(task_where_params)

        for row in cur.execute(
            f"SELECT status, COUNT(*) AS total_count FROM work_queue WHERE {queue_where} GROUP BY status",
            queue_params,
        ).fetchall():
            queue_status_counts[str(row["status"] or "")] = int(row["total_count"] or 0)

        for row in cur.execute(
            f"SELECT COALESCE(NULLIF(TRIM(role_required), ''), 'UNSET') AS role_key, COUNT(*) AS total_count "
            f"FROM work_queue WHERE {queue_where} GROUP BY role_key",
            queue_params,
        ).fetchall():
            queue_role_required_counts[str(row["role_key"] or "")] = int(row["total_count"] or 0)

        for row in cur.execute(
            f"""
SELECT
  w.task_id,
  GROUP_CONCAT(d.depends_on_task_id || ':' || COALESCE(dep.status, 'MISSING'), ' | ') AS blockers
FROM work_queue w
JOIN task_dependencies d ON d.task_id = w.task_id AND d.relation = 'blocks'
LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
WHERE {queue_where}
  AND w.status = 'QUEUED'
  AND COALESCE(dep.status, 'MISSING') <> 'DONE'
GROUP BY w.task_id
ORDER BY w.updated_at DESC, w.task_id ASC
LIMIT 10
""",
            queue_params,
        ).fetchall():
            queue_dependency_blocked_examples.append(
                {
                    "task_id": row["task_id"],
                    "blocked_by": [
                        p.strip()
                        for p in str(row["blockers"] or "").split("|")
                        if str(p).strip()
                    ],
                }
            )

        active_row = cur.execute(
            "SELECT COUNT(*) AS total_count FROM file_locks WHERE lock_state = 'ACTIVE'"
        ).fetchone()
        active_file_locks["active_count"] = int((active_row["total_count"] if active_row else 0) or 0)

        stale_rows = cur.execute(
            """
SELECT lock_id, file_path, locked_by_task_id, acquired_at, lock_expires_at
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
ORDER BY lock_expires_at ASC
LIMIT 10
""",
            (to_iso_z(now_dt),),
        ).fetchall()
        active_file_locks["stale_active_count"] = len(stale_rows)
        active_file_locks["examples"] = [
            {
                "lock_id": row["lock_id"],
                "file_path": row["file_path"],
                "locked_by_task_id": row["locked_by_task_id"],
                "acquired_at": row["acquired_at"],
                "lock_expires_at": row["lock_expires_at"],
            }
            for row in stale_rows
        ]

        tr_where = "created_at >= ? AND created_at <= ?"
        tr_params: List[Any] = [start_iso, end_iso]
        if task_where_sql:
            tr_where += task_where_sql
            tr_params.extend(task_where_params)
        if actor_where_sql:
            tr_where += actor_where_sql
            tr_params.extend(actor_where_params)

        tr_total_row = cur.execute(
            f"SELECT COUNT(*) AS total_count FROM task_transitions WHERE {tr_where}",
            tr_params,
        ).fetchone()
        transition_rollup["total"] = int((tr_total_row["total_count"] if tr_total_row else 0) or 0)

        for row in cur.execute(
            f"SELECT to_status, COUNT(*) AS total_count FROM task_transitions WHERE {tr_where} GROUP BY to_status",
            tr_params,
        ).fetchall():
            transition_rollup["to_status"][str(row["to_status"] or "")] = int(row["total_count"] or 0)

        for row in cur.execute(
            f"SELECT actor_role, COUNT(*) AS total_count FROM task_transitions WHERE {tr_where} GROUP BY actor_role",
            tr_params,
        ).fetchall():
            transition_rollup["actor_role"][str(row["actor_role"] or "")] = int(row["total_count"] or 0)

        for row in cur.execute(
            f"""
SELECT reason, COUNT(*) AS total_count
FROM task_transitions
WHERE {tr_where}
GROUP BY reason
ORDER BY total_count DESC
LIMIT 10
""",
            tr_params,
        ).fetchall():
            reason_key = str(row["reason"] or "").strip() or "unspecified"
            transition_rollup["reason_top"][reason_key] = int(row["total_count"] or 0)

        transition_evidence_seen = set()
        for row in cur.execute(
            f"""
SELECT event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
FROM task_transitions
WHERE {tr_where}
ORDER BY created_at DESC, rowid DESC
LIMIT ?
""",
            (*tr_params, limit),
        ).fetchall():
            evidence_ref = str(row["evidence_ref"] or "")
            recent_transitions_db.append(
                {
                    "event_id": row["event_id"],
                    "task_id": row["task_id"],
                    "from_status": row["from_status"],
                    "to_status": row["to_status"],
                    "actor_role": row["actor_role"],
                    "reason": row["reason"],
                    "evidence_ref": row["evidence_ref"],
                    "created_at": row["created_at"],
                    "age_sec": age_sec(str(row["created_at"] or "")),
                }
            )

            if evidence_ref:
                for part in [p.strip() for p in evidence_ref.split("|") if str(p).strip()]:
                    if part in transition_evidence_seen:
                        continue
                    transition_evidence_seen.add(part)
                    recent_transition_evidence_refs.append(part)
                    if len(recent_transition_evidence_refs) >= 10:
                        break
            if len(recent_transition_evidence_refs) >= 10:
                continue

    except Exception:
        pass
    finally:
        con.close()

if checkpoints_dir.exists():
    candidates = sorted(checkpoints_dir.glob("chk_*.json"), key=lambda p: p.name, reverse=True)
    for path in candidates:
        if len(recent_checkpoint_files) >= limit:
            break
        obj = load_json(path)
        if not obj:
            continue
        meta = obj.get("metadata") or {}
        trig = str(meta.get("trigger") or "")
        if trigger_filter and trig != trigger_filter:
            continue
        created_at = str(meta.get("created_at") or "")
        created_dt = parse_iso(created_at)
        if created_dt is not None:
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=dt.timezone.utc)
            if created_dt < start_dt or created_dt > end_dt:
                continue
        objective = obj.get("objective") or {}
        recent_checkpoint_files.append(
            {
                "checkpoint_id": str(meta.get("checkpoint_id") or path.stem),
                "created_at": created_at or None,
                "age_sec": age_sec(created_at),
                "trigger": trig,
                "status": objective.get("status"),
                "objective": objective.get("primary_goal"),
                "json_path": rel(path),
            }
        )

latest_pointer = load_json(latest_pointer_path)

summary = {
    "schema_version": "continuity.history.v3",
    "generated_at": to_iso_z(now_dt),
    "window_hours_default": window_hours,
    "range": {
        "from_utc": start_iso,
        "until_utc": end_iso,
        "derived_from": range_source,
        "since_checkpoint": since_checkpoint_resolved,
        "until": until_resolved,
    },
    "filters": {
        "source_preset": source_preset,
        "source_filter_origin": source_filter_origin,
        "source_filters": source_filters,
        "task_filter_origin": task_filter_origin,
        "task_filters": task_filters,
        "actor_role_filter_origin": actor_role_filter_origin,
        "actor_role_filters": actor_role_filters,
        "trigger_filter": trigger_filter,
        "include_suppressed": include_suppressed,
        "limit": limit,
    },
    "sources": {
        "continuity_db": str(db_path),
        "checkpoints_dir": str(checkpoints_dir),
        "continuity_now_latest": str(continuity_now_latest_path),
        "continuity_current": str(continuity_current_path),
        "blocker_registry": str(blocker_registry_path),
    },
    "headline": {
        "verify_gate_mode": verify_gate_preflight_posture.get("mode"),
        "verify_gate_source": verify_gate_preflight_posture.get("source"),
        "verify_gate_ready_to_run": verify_gate_preflight_posture.get("ready_to_run"),
        "verify_gate_predicted_blocker_reason": verify_gate_preflight_posture.get("predicted_blocker_reason"),
        "verify_gate_severity": verify_gate_preflight_posture.get("severity"),
        "runtime_operator_failures_source": runtime_operator_failures.get("source"),
        "runtime_operator_failures_status": runtime_operator_failures.get("status"),
        "runtime_operator_failures_open_total": runtime_operator_failures.get("open_total"),
        "runtime_operator_failures_blocker_total": runtime_operator_failures.get("blocker_total"),
        "runtime_operator_failures_warn_total": runtime_operator_failures.get("warn_total"),
        "runtime_operator_failures_top_reason": headline_runtime_operator_top_reason,
    },
    "latest_pointer": {
        "checkpoint_id": latest_pointer.get("checkpoint_id"),
        "updated_at": latest_pointer.get("updated_at"),
        "json_path": latest_pointer.get("json_path"),
    },
    "rollup": {
        "checkpoint_trigger_counts": checkpoint_trigger_counts,
        "event_counts": event_rollup,
        "event_source_counts": event_source_counts,
        "queue": {
            "work_queue_status_counts": queue_status_counts,
            "work_queue_role_required_counts": queue_role_required_counts,
            "dependency_blocked_examples": queue_dependency_blocked_examples,
            "active_file_locks": active_file_locks,
            "transition_counts": transition_rollup,
            "recent_transition_evidence_refs": recent_transition_evidence_refs,
        },
        "verify_gate_preflight": verify_gate_preflight_posture,
        "runtime_operator_failures": runtime_operator_failures,
        # backward-compatible alias
        "reconcile_event_counts": event_rollup,
    },
    "history": {
        "recent_checkpoints_db": recent_checkpoints_db,
        "recent_checkpoints_files": recent_checkpoint_files,
        "recent_events": recent_events_db,
        "recent_transitions": recent_transitions_db,
    },
    # backward-compatible alias from v1
    "reconcile": {
        "recent_checkpoints_db": recent_checkpoints_db,
        "recent_checkpoints_files": recent_checkpoint_files,
        "recent_events": recent_events_db,
    },
    "queue_history": {
        "recent_transitions": recent_transitions_db,
        "work_queue_status_counts": queue_status_counts,
        "work_queue_role_required_counts": queue_role_required_counts,
        "dependency_blocked_examples": queue_dependency_blocked_examples,
        "active_file_locks": active_file_locks,
        "transition_counts": transition_rollup,
        "recent_transition_evidence_refs": recent_transition_evidence_refs,
    },
}

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    latest_cp = recent_checkpoints_db[0] if recent_checkpoints_db else (recent_checkpoint_files[0] if recent_checkpoint_files else None)
    latest_ev = recent_events_db[0] if recent_events_db else None
    latest_transition = recent_transitions_db[0] if recent_transitions_db else None

    src_filters_human = ", ".join(source_filters) if source_filters else "all"
    task_filters_human = ", ".join(task_filters) if task_filters else "all"
    actor_filters_human = ", ".join(actor_role_filters) if actor_role_filters else "all"
    trig_human = trigger_filter if trigger_filter else "any"

    print("CONTINUITY HISTORY")
    print(f"- range: {start_iso} -> {end_iso} (derived_from={range_source})")
    if since_checkpoint_resolved:
        print(
            "- since_checkpoint: "
            f"{since_checkpoint_resolved.get('checkpoint_id')} "
            f"@ {since_checkpoint_resolved.get('created_at')} "
            f"(resolved_from={since_checkpoint_resolved.get('resolved_from')})"
        )
    if until_resolved and until_resolved.get("resolved_from") != "iso":
        print(
            "- until_checkpoint: "
            f"{until_resolved.get('checkpoint_id')} "
            f"@ {until_resolved.get('created_at')} "
            f"(resolved_from={until_resolved.get('resolved_from')})"
        )
    print(
        "- filters: "
        f"source_preset={source_preset}; "
        f"sources={src_filters_human}; "
        f"tasks={task_filters_human}; "
        f"actor_roles={actor_filters_human}; "
        f"trigger={trig_human}; "
        f"include_suppressed={int(include_suppressed)}"
    )
    print(f"- checkpoints(db): {len(recent_checkpoints_db)}")
    print(f"- checkpoints(files): {len(recent_checkpoint_files)}")
    print(
        "- events: "
        f"total={event_rollup['total']} emitted={event_rollup['emitted']} suppressed={event_rollup['suppressed']} "
        f"sev={event_rollup['severity']}"
    )
    print(
        "- verify_gate_preflight: "
        f"mode={verify_gate_preflight_posture.get('mode')}; "
        f"source={verify_gate_preflight_posture.get('source')}; "
        f"ready_to_run={verify_gate_preflight_posture.get('ready_to_run') if verify_gate_preflight_posture.get('ready_to_run') is not None else 'n/a'}; "
        f"predicted_blocker={verify_gate_preflight_posture.get('predicted_blocker_reason') or 'none'}; "
        f"severity={verify_gate_preflight_posture.get('severity')}"
    )
    print(
        "- runtime_operator_failures: "
        f"source={runtime_operator_failures.get('source')}; "
        f"status={runtime_operator_failures.get('status')}; "
        f"open={runtime_operator_failures.get('open_total')}; "
        f"blockers={runtime_operator_failures.get('blocker_total')}; "
        f"warns={runtime_operator_failures.get('warn_total')}; "
        f"top_reason={headline_runtime_operator_top_reason or 'none'}"
    )
    print(
        "- queue(work_queue): "
        f"status={queue_status_counts if queue_status_counts else {}}; "
        f"role_required={queue_role_required_counts if queue_role_required_counts else {}}"
    )
    print(
        "- queue(transitions): "
        f"total={transition_rollup.get('total')}; "
        f"to_status={transition_rollup.get('to_status') or {}}; "
        f"actor_role={transition_rollup.get('actor_role') or {}}"
    )
    if queue_dependency_blocked_examples:
        print(
            "- queue_dependency_blocked_examples: "
            + " ; ".join(
                f"{row.get('task_id')}<=({', '.join(row.get('blocked_by') or [])})"
                for row in queue_dependency_blocked_examples
            )
        )
    if int(active_file_locks.get("stale_active_count") or 0) > 0:
        print(
            "- stale_active_file_locks: "
            f"count={active_file_locks.get('stale_active_count')}; "
            + " ; ".join(
                f"{row.get('file_path')}@{row.get('locked_by_task_id')} expires={row.get('lock_expires_at')}"
                for row in (active_file_locks.get('examples') or [])
            )
        )

    if latest_cp:
        print(
            "- latest_checkpoint: "
            f"{latest_cp.get('checkpoint_id')} "
            f"trigger={latest_cp.get('trigger') or 'n/a'} "
            f"status={latest_cp.get('status') or 'n/a'} "
            f"age={age_compact(latest_cp.get('age_sec'))}"
        )
    if latest_ev:
        print(
            "- latest_event: "
            f"{latest_ev.get('source')}::{latest_ev.get('event_key')} "
            f"severity={latest_ev.get('severity')} "
            f"age={age_compact(latest_ev.get('age_sec'))}"
        )
    if latest_transition:
        print(
            "- latest_transition: "
            f"{latest_transition.get('task_id')} "
            f"{latest_transition.get('from_status') or 'n/a'}->{latest_transition.get('to_status')} "
            f"actor={latest_transition.get('actor_role') or 'n/a'} "
            f"age={age_compact(latest_transition.get('age_sec'))}"
        )
    if recent_transition_evidence_refs:
        print(f"- queue_evidence_refs: {', '.join(recent_transition_evidence_refs[:6])}")
PY
