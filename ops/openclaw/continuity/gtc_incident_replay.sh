#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
GTC_ROOT="${OPENCLAW_GTC_ROOT:-$ROOT/state/gtc-v2}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"

ROUTE_KEY=""
LATEST_EVENT_ID=""
INCIDENT_ID=""
INCIDENT_INDEX=""
OUTPUT_PATH=""
OUTPUT_MD_PATH=""
JSON_OUT=0
WRITE_OUTPUT=1
MAX_ROUTE_EVENTS=200
MAX_TASK_EVENTS=200
MAX_CHECKPOINT_EVENTS=160
CHECKPOINT_SCOPE="incident"

usage() {
  cat <<'EOF'
Usage: gtc_incident_replay.sh [options]

Build deterministic incident replay bundle (evidence chain + artifact pack)
from GTC latest surfaces + GTC evidence index.

Incident selector options (choose one; default = first open incident):
  --route-key <route_key>         Select by event route key
  --latest-event-id <event_id>    Select by latest_event_id
  --incident-id <incident_id>     Select by deterministic replay incident id
  --incident-index <n>            Select Nth open incident (1-based)

I/O and limits:
  --gtc-root <path>               GTC root (default: state/gtc-v2)
  --db <path>                     Continuity sqlite path (default: state/continuity/continuity_os.sqlite)
  --out <path>                    Output replay JSON path (default: state/gtc-v2/incident_replay/incident-<id>.json)
  --out-md <path>                 Output replay markdown path (default: .md next to JSON)
  --no-write                      Do not write replay bundle files
  --max-route-events <n>          Max operator/actions evidence rows (default: 200)
  --max-task-events <n>           Max queue/task evidence rows (default: 200)
  --max-checkpoint-events <n>     Max checkpoint-linked evidence rows (default: 160)
  --checkpoint-scope <mode>       Checkpoint expansion scope: incident|full (default: incident)
  --json                          Emit JSON to stdout
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route-key)
      ROUTE_KEY="${2:-}"; shift 2 ;;
    --latest-event-id)
      LATEST_EVENT_ID="${2:-}"; shift 2 ;;
    --incident-id)
      INCIDENT_ID="${2:-}"; shift 2 ;;
    --incident-index)
      INCIDENT_INDEX="${2:-}"; shift 2 ;;
    --gtc-root)
      GTC_ROOT="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --out)
      OUTPUT_PATH="${2:-}"; shift 2 ;;
    --out-md)
      OUTPUT_MD_PATH="${2:-}"; shift 2 ;;
    --no-write)
      WRITE_OUTPUT=0; shift ;;
    --max-route-events)
      MAX_ROUTE_EVENTS="${2:-}"; shift 2 ;;
    --max-task-events)
      MAX_TASK_EVENTS="${2:-}"; shift 2 ;;
    --max-checkpoint-events)
      MAX_CHECKPOINT_EVENTS="${2:-}"; shift 2 ;;
    --checkpoint-scope)
      CHECKPOINT_SCOPE="${2:-}"; shift 2 ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$GTC_ROOT" "$DB_PATH" "$ROUTE_KEY" "$LATEST_EVENT_ID" "$INCIDENT_ID" "$INCIDENT_INDEX" "$OUTPUT_PATH" "$OUTPUT_MD_PATH" "$JSON_OUT" "$WRITE_OUTPUT" "$MAX_ROUTE_EVENTS" "$MAX_TASK_EVENTS" "$MAX_CHECKPOINT_EVENTS" "$CHECKPOINT_SCOPE" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_int(raw: str, default: int) -> int:
    try:
        val = int(str(raw or "").strip())
        return val if val > 0 else default
    except Exception:
        return default


root = pathlib.Path(sys.argv[1]).resolve()
gtc_root = pathlib.Path(sys.argv[2]).resolve()
db_path = pathlib.Path(sys.argv[3]).resolve()
route_key_arg = str(sys.argv[4] or "").strip()
latest_event_id_arg = str(sys.argv[5] or "").strip()
incident_id_arg = str(sys.argv[6] or "").strip()
incident_index_arg = str(sys.argv[7] or "").strip()
out_path_arg = str(sys.argv[8] or "").strip()
out_md_arg = str(sys.argv[9] or "").strip()
json_out = bool(int(sys.argv[10]))
write_output = bool(int(sys.argv[11]))
max_route_events = parse_int(sys.argv[12], 200)
max_task_events = parse_int(sys.argv[13], 200)
max_checkpoint_events = parse_int(sys.argv[14], 160)
checkpoint_scope_raw = str(sys.argv[15] or "").strip().lower()
if checkpoint_scope_raw in {"full", "all", "checkpoint"}:
    checkpoint_scope = "full"
elif checkpoint_scope_raw in {"incident", "focused", "linked", "incident-linked"}:
    checkpoint_scope = "incident"
else:
    checkpoint_scope = "incident"

continuity_dir = (root / "ops" / "openclaw" / "continuity").resolve()
if str(continuity_dir) not in sys.path:
    sys.path.insert(0, str(continuity_dir))

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


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path_obj: pathlib.Path) -> str:
    try:
        return str(path_obj.resolve().relative_to(root))
    except Exception:
        return str(path_obj)


def load_json_obj(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def is_severe_verify_gate_preflight_blocker(reason: Optional[str]) -> bool:
    return bool(_policy_is_severe_verify_gate_preflight_blocker(reason))


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


def parse_obj(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    txt = str(raw or "").strip()
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_array(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    txt = str(raw or "").strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def split_pipe_tokens(raw: Any) -> List[str]:
    text = str(raw or "")
    return [part.strip() for part in text.split("|") if part and part.strip()]


def incident_id_for_route(row: Dict[str, Any]) -> str:
    seed = {
        "route_key": str(row.get("route_key") or ""),
        "source": str(row.get("source") or ""),
        "event_key": str(row.get("event_key") or ""),
        "latest_event_id": str(row.get("latest_event_id") or ""),
        "open_since": str(row.get("open_since") or ""),
    }
    return "inc_" + sha256_text(canonical_json(seed))[:16]


def choose_incident(route_rows: List[Dict[str, Any]], open_rows: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    route_by_key = {str(r.get("route_key") or ""): r for r in route_rows}

    def merge_route(base: Dict[str, Any]) -> Dict[str, Any]:
        rk = str(base.get("route_key") or "")
        merged = dict(route_by_key.get(rk) or {})
        merged.update(base)
        return merged

    if route_key_arg:
        row = route_by_key.get(route_key_arg)
        if row is None:
            return None, {"selector": "route_key", "value": route_key_arg, "error": "route_key_not_found"}
        return dict(row), {"selector": "route_key", "value": route_key_arg}

    if latest_event_id_arg:
        for row in route_rows:
            if str(row.get("latest_event_id") or "") == latest_event_id_arg:
                return dict(row), {"selector": "latest_event_id", "value": latest_event_id_arg}
        return None, {"selector": "latest_event_id", "value": latest_event_id_arg, "error": "latest_event_id_not_found"}

    if incident_id_arg:
        for row in route_rows:
            if incident_id_for_route(row) == incident_id_arg:
                return dict(row), {"selector": "incident_id", "value": incident_id_arg}
        return None, {"selector": "incident_id", "value": incident_id_arg, "error": "incident_id_not_found"}

    if incident_index_arg:
        idx = parse_int(incident_index_arg, -1)
        if idx <= 0 or idx > len(open_rows):
            return None, {
                "selector": "incident_index",
                "value": incident_index_arg,
                "error": "incident_index_out_of_range",
                "open_incident_count": len(open_rows),
            }
        selected_open = open_rows[idx - 1]
        return merge_route(selected_open), {
            "selector": "incident_index",
            "value": idx,
            "open_incident_count": len(open_rows),
        }

    if open_rows:
        return merge_route(open_rows[0]), {"selector": "default_first_open", "open_incident_count": len(open_rows)}
    if route_rows:
        return dict(route_rows[0]), {"selector": "default_first_route", "route_count": len(route_rows)}
    return None, {"selector": "default", "error": "no_routes_available"}


def sql_rows_for_ids(prefix_sql: str, ids: Sequence[str], tail_sql: str = "") -> Tuple[str, Tuple[Any, ...]]:
    placeholders = ",".join("?" for _ in ids)
    sql = f"{prefix_sql} ({placeholders}) {tail_sql}".strip()
    return sql, tuple(ids)


def row_to_evidence(row: sqlite3.Row, *, chain_scope: str, linked_task_id: Optional[str] = None, linked_checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    facts = parse_obj(row["facts_json"])
    refs = parse_obj(row["refs_json"])
    task_ids = []
    if linked_task_id:
        task_ids.append(str(linked_task_id))
    if str(refs.get("task_id") or "").strip():
        task_ids.append(str(refs.get("task_id") or "").strip())

    checkpoint_ids = []
    if linked_checkpoint_id:
        checkpoint_ids.append(str(linked_checkpoint_id))
    if str(refs.get("checkpoint_id") or "").strip():
        checkpoint_ids.append(str(refs.get("checkpoint_id") or "").strip())

    return {
        "evidence_id": str(row["evidence_id"] or ""),
        "connector_type": str(row["connector_type"] or ""),
        "connector_id": str(row["connector_id"] or ""),
        "observed_at": str(row["observed_at"] or ""),
        "monotonic_seq": int(row["monotonic_seq"] or 0),
        "subject": {
            "kind": str(row["subject_kind"] or ""),
            "id": str(row["subject_id"] or ""),
        },
        "severity_max": str(row["severity_max"] or ""),
        "facts": facts,
        "refs": refs,
        "jsonl": {
            "path": str(row["jsonl_path"] or ""),
            "line_no": int(row["jsonl_line_no"] or 0),
        },
        "lineage": {
            "continuity_event_id": str(refs.get("continuity_event_id") or "") or None,
            "linked_task_ids": sorted({x for x in task_ids if x}),
            "linked_checkpoint_ids": sorted({x for x in checkpoint_ids if x}),
        },
        "chain_scopes": [chain_scope],
    }


def merge_chain_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            continue
        if evidence_id not in merged:
            merged[evidence_id] = row
            continue
        existing = merged[evidence_id]
        scopes = set(existing.get("chain_scopes") or [])
        scopes.update(row.get("chain_scopes") or [])
        existing["chain_scopes"] = sorted(s for s in scopes if s)

        ex_lineage = existing.get("lineage") if isinstance(existing.get("lineage"), dict) else {}
        new_lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}

        task_ids = set(ex_lineage.get("linked_task_ids") or [])
        task_ids.update(new_lineage.get("linked_task_ids") or [])
        checkpoint_ids = set(ex_lineage.get("linked_checkpoint_ids") or [])
        checkpoint_ids.update(new_lineage.get("linked_checkpoint_ids") or [])

        ex_lineage["linked_task_ids"] = sorted(x for x in task_ids if x)
        ex_lineage["linked_checkpoint_ids"] = sorted(x for x in checkpoint_ids if x)
        if not ex_lineage.get("continuity_event_id"):
            ex_lineage["continuity_event_id"] = new_lineage.get("continuity_event_id")
        existing["lineage"] = ex_lineage

    out = list(merged.values())
    out.sort(
        key=lambda r: (
            str(r.get("observed_at") or ""),
            str(r.get("connector_type") or ""),
            int(r.get("monotonic_seq") or 0),
            str(r.get("evidence_id") or ""),
        )
    )
    return out


def artifact_key(role: str, sha: str, path: str) -> Tuple[str, str, str]:
    return (role.strip(), sha.strip().lower(), path.strip())


def resolve_exists(path_value: str) -> Optional[bool]:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    p = pathlib.Path(raw)
    if not p.is_absolute():
        p = (root / raw).resolve()
    return p.exists() and p.is_file()


def summarize_role(role: str) -> str:
    rv = str(role or "").strip()
    return rv or "artifact"


def normalize_binding_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    if status in {"verified", "degraded", "not_applicable"}:
        return status
    return "unknown" if status else "not_reported"


def summarize_handoff_decisions(chain_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    queue_rows = [row for row in chain_rows if str(row.get("connector_type") or "") == "queue.task"]
    handoff_rows: List[Dict[str, Any]] = []

    queue_reason_counts: Dict[str, int] = defaultdict(int)
    ingress_classification_counts: Dict[str, int] = defaultdict(int)
    binding_status_counts: Dict[str, int] = defaultdict(int)
    binding_issue_code_counts: Dict[str, int] = defaultdict(int)
    degraded_binding_task_ids: set[str] = set()

    for row in queue_rows:
        refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        handoff_packet = refs.get("handoff_packet") if isinstance(refs.get("handoff_packet"), dict) else {}
        if not handoff_packet:
            continue

        handoff_rows.append(row)
        gate_summary = handoff_packet.get("gate_summary") if isinstance(handoff_packet.get("gate_summary"), dict) else {}

        queue_reason = str(gate_summary.get("queue_reason") or "").strip()
        if queue_reason:
            queue_reason_counts[queue_reason] += 1

        ingress_classification = str(gate_summary.get("ingress_classification") or "").strip()
        if ingress_classification:
            ingress_classification_counts[ingress_classification] += 1

        binding_status = normalize_binding_status(facts.get("handoff_gate_binding_status"))
        binding_status_counts[binding_status] += 1

        if binding_status == "degraded":
            subject_obj = row.get("subject") if isinstance(row.get("subject"), dict) else {}
            task_id = str(refs.get("task_id") or subject_obj.get("id") or "").strip()
            if task_id:
                degraded_binding_task_ids.add(task_id)

        projection = refs.get("handoff_gate_binding_projection") if isinstance(refs.get("handoff_gate_binding_projection"), dict) else {}
        issues = projection.get("issues") if isinstance(projection.get("issues"), list) else []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or "").strip()
            if code:
                binding_issue_code_counts[code] += 1

    if not handoff_rows:
        binding_integrity_status = "not_observed"
    elif int(binding_status_counts.get("degraded") or 0) > 0:
        binding_integrity_status = "degraded"
    elif int(binding_status_counts.get("verified") or 0) == len(handoff_rows):
        binding_integrity_status = "verified"
    elif int(binding_status_counts.get("verified") or 0) > 0:
        binding_integrity_status = "partial"
    else:
        binding_integrity_status = "unknown"

    with_gate_summary_count = 0
    for row in handoff_rows:
        refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}
        handoff_packet = refs.get("handoff_packet") if isinstance(refs.get("handoff_packet"), dict) else {}
        gate_summary = handoff_packet.get("gate_summary") if isinstance(handoff_packet.get("gate_summary"), dict) else {}
        if gate_summary:
            with_gate_summary_count += 1

    return {
        "count": len(handoff_rows),
        "queue_task_rows_considered": len(queue_rows),
        "with_gate_summary_count": with_gate_summary_count,
        "queue_reason_counts": dict(sorted(queue_reason_counts.items())),
        "ingress_classification_counts": dict(sorted(ingress_classification_counts.items())),
        "binding_integrity_status": binding_integrity_status,
        "binding_status_counts": dict(sorted(binding_status_counts.items())),
        "binding_issue_code_counts": dict(sorted(binding_issue_code_counts.items())),
        "degraded_binding_task_ids": sorted(degraded_binding_task_ids),
        "warning_reasons": ["queue_task_handoff_gate_binding_degraded"]
        if int(binding_status_counts.get("degraded") or 0) > 0
        else [],
    }


def build_markdown(bundle: Dict[str, Any]) -> str:
    incident = bundle.get("incident") if isinstance(bundle.get("incident"), dict) else {}
    chain = bundle.get("evidence_chain") if isinstance(bundle.get("evidence_chain"), dict) else {}
    artifacts = bundle.get("artifact_pack") if isinstance(bundle.get("artifact_pack"), dict) else {}
    handoff = bundle.get("handoff_decisions") if isinstance(bundle.get("handoff_decisions"), dict) else {}
    verify_gate = bundle.get("verify_gate_preflight") if isinstance(bundle.get("verify_gate_preflight"), dict) else {}
    lines: List[str] = []
    lines.extend(
        [
            "# GTC Incident Replay",
            "",
            f"- incident_id: `{incident.get('incident_id')}`",
            f"- route_key: `{incident.get('route_key')}`",
            f"- source/event: `{incident.get('source')}` / `{incident.get('event_key')}`",
            f"- status/severity: `{incident.get('status')}` / `{incident.get('severity')}`",
            f"- generated_at: `{bundle.get('generated_at')}`",
            f"- build_generation_id: `{bundle.get('build_generation_id')}`",
            "",
            "## Verify-gate preflight",
            f"- mode: `{verify_gate.get('mode') or 'unknown'}`",
            f"- source: `{verify_gate.get('source') or 'unavailable'}`",
            f"- ready_to_run: `{verify_gate.get('ready_to_run') if verify_gate.get('ready_to_run') is not None else 'n/a'}`",
            f"- predicted_blocker: `{verify_gate.get('predicted_blocker_reason') or 'none'}`",
            f"- severity: `{verify_gate.get('severity') or 'warn'}`",
            "",
            "## Evidence chain",
            f"- total_evidence: {chain.get('count', 0)}",
            f"- connectors: {chain.get('connector_counts') or {}}",
            f"- related_task_ids: {chain.get('task_ids') or []}",
            f"- selected_event_ids: {chain.get('selected_event_ids') or []}",
            f"- checkpoint_ids: {chain.get('checkpoint_ids') or []}",
            f"- checkpoint_scope: {chain.get('checkpoint_scope')}",
            "",
            "## Artifact pack",
            f"- artifact_count: {artifacts.get('count', 0)}",
            f"- role_counts: {artifacts.get('role_counts') or {}}",
            f"- typed_task_roles: {artifacts.get('typed_task_roles') or []}",
            "",
            "## Handoff decisions",
            f"- handoff_count: {handoff.get('count', 0)}",
            f"- queue_task_rows_considered: {handoff.get('queue_task_rows_considered', 0)}",
            f"- with_gate_summary_count: {handoff.get('with_gate_summary_count', 0)}",
            f"- queue_reason_counts: {handoff.get('queue_reason_counts') or {}}",
            f"- ingress_classification_counts: {handoff.get('ingress_classification_counts') or {}}",
            f"- binding_integrity_status: {handoff.get('binding_integrity_status') or 'not_observed'}",
            f"- binding_status_counts: {handoff.get('binding_status_counts') or {}}",
            f"- binding_issue_code_counts: {handoff.get('binding_issue_code_counts') or {}}",
            f"- degraded_binding_task_ids: {handoff.get('degraded_binding_task_ids') or []}",
            "",
            "## Top evidence rows",
        ]
    )

    rows = chain.get("rows") if isinstance(chain.get("rows"), list) else []
    for row in rows[:16]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('observed_at')}` `{row.get('connector_type')}` `{row.get('evidence_id')}` scopes={row.get('chain_scopes') or []}"
        )

    if len(rows) > 16:
        lines.append(f"- ... ({len(rows) - 16} more)")

    lines.append("")
    lines.append("## Top artifact rows")
    arows = artifacts.get("entries") if isinstance(artifacts.get("entries"), list) else []
    for row in arows[:20]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('role')}` sha=`{row.get('sha256') or ''}` path=`{row.get('path') or ''}` evidence={row.get('evidence_ids') or []}"
        )
    if len(arows) > 20:
        lines.append(f"- ... ({len(arows) - 20} more)")

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


latest_dir = gtc_root / "latest"
continuity_now_latest_path = root / "state" / "continuity" / "latest" / "continuity_now_latest.json"
incident_surface = load_json_obj(latest_dir / "incident_replay.json")
event_surface = load_json_obj(latest_dir / "event_projection.json")
verify_gate_preflight_posture = compute_verify_gate_preflight_posture(load_json_obj(continuity_now_latest_path))

open_rows = incident_surface.get("open_incidents") if isinstance(incident_surface.get("open_incidents"), list) else []
route_rows = event_surface.get("routes") if isinstance(event_surface.get("routes"), list) else []
open_rows = [row for row in open_rows if isinstance(row, dict)]
route_rows = [row for row in route_rows if isinstance(row, dict)]

selected_row, selection_meta = choose_incident(route_rows, open_rows)
if selected_row is None:
    payload = {
        "ok": False,
        "schema_version": "gtc.incident_replay.bundle.v1",
        "error": "incident_not_found",
        "selection": selection_meta,
        "surfaces": {
            "incident_replay_path": rel(latest_dir / "incident_replay.json"),
            "event_projection_path": rel(latest_dir / "event_projection.json"),
            "open_incident_count": len(open_rows),
            "route_count": len(route_rows),
        },
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"ERROR incident_not_found: {selection_meta}", file=sys.stderr)
    raise SystemExit(1)

selected_row = dict(selected_row)
incident_id = incident_id_for_route(selected_row)
selected_row["incident_id"] = incident_id

if not db_path.exists():
    payload = {
        "ok": False,
        "schema_version": "gtc.incident_replay.bundle.v1",
        "error": "continuity_db_missing",
        "db_path": str(db_path),
        "incident": {
            "incident_id": incident_id,
            "route_key": selected_row.get("route_key"),
            "latest_event_id": selected_row.get("latest_event_id"),
        },
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"ERROR continuity db missing: {db_path}", file=sys.stderr)
    raise SystemExit(1)

con = sqlite3.connect(str(db_path))
con.row_factory = sqlite3.Row
cur = con.cursor()

base_select = """
SELECT
  evidence_id,
  connector_type,
  connector_id,
  observed_at,
  monotonic_seq,
  subject_kind,
  subject_id,
  severity_max,
  jsonl_path,
  jsonl_line_no,
  facts_json,
  refs_json
FROM gtc_evidence_index
""".strip()

route_chain_rows: List[Dict[str, Any]] = []
recent_event_ids: List[str] = []
for ev_id in selected_row.get("recent_event_ids") or []:
    ev_txt = str(ev_id or "").strip()
    if ev_txt:
        recent_event_ids.append(ev_txt)
latest_event_txt = str(selected_row.get("latest_event_id") or "").strip()
if latest_event_txt:
    recent_event_ids.append(latest_event_txt)
recent_event_ids = sorted({x for x in recent_event_ids if x})
selected_event_ids_sorted = list(recent_event_ids)

if recent_event_ids:
    sql, params = sql_rows_for_ids(
        f"{base_select} WHERE connector_type = 'operator.actions' AND json_extract(refs_json, '$.continuity_event_id') IN",
        recent_event_ids,
        "ORDER BY observed_at ASC, monotonic_seq ASC LIMIT ?",
    )
    rows = cur.execute(sql, tuple(params) + (max_route_events,)).fetchall()
    for row in rows:
        route_chain_rows.append(row_to_evidence(row, chain_scope="route"))

if not route_chain_rows:
    src = str(selected_row.get("source") or "").strip()
    event_key = str(selected_row.get("event_key") or "").strip()
    if src and event_key:
        rows = cur.execute(
            f"""
{base_select}
WHERE connector_type = 'operator.actions'
  AND json_extract(facts_json, '$.source') = ?
  AND json_extract(facts_json, '$.event_key') = ?
ORDER BY observed_at ASC, monotonic_seq ASC
LIMIT ?
""",
            (src, event_key, max_route_events),
        ).fetchall()
        for row in rows:
            route_chain_rows.append(row_to_evidence(row, chain_scope="route"))

related_task_ids = set(str(t).strip() for t in (selected_row.get("related_task_ids") or []) if str(t).strip())
for row in route_chain_rows:
    refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}
    task_id = str(refs.get("task_id") or "").strip()
    if task_id:
        related_task_ids.add(task_id)
    for key in ("evidence_ref",):
        for token in split_pipe_tokens(refs.get(key)):
            if token.startswith("autopilot:") or token.startswith("continuity:") or token.startswith("parity:"):
                related_task_ids.add(token)

related_task_ids_sorted = sorted(related_task_ids)

task_chain_rows: List[Dict[str, Any]] = []
if related_task_ids_sorted:
    sql, params = sql_rows_for_ids(
        f"""
SELECT
  t.task_id AS linked_task_id,
  e.evidence_id,
  e.connector_type,
  e.connector_id,
  e.observed_at,
  e.monotonic_seq,
  e.subject_kind,
  e.subject_id,
  e.severity_max,
  e.jsonl_path,
  e.jsonl_line_no,
  e.facts_json,
  e.refs_json
FROM gtc_task_evidence t
JOIN gtc_evidence_index e ON e.evidence_id = t.evidence_id
WHERE t.task_id IN
""".strip(),
        related_task_ids_sorted,
        "ORDER BY e.observed_at ASC, e.monotonic_seq ASC LIMIT ?",
    )
    rows = cur.execute(sql, tuple(params) + (max_task_events,)).fetchall()
    for row in rows:
        task_chain_rows.append(
            row_to_evidence(
                row,
                chain_scope="task",
                linked_task_id=str(row["linked_task_id"] or "").strip() or None,
            )
        )

checkpoint_ids = set()
for row in route_chain_rows + task_chain_rows:
    refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}
    checkpoint_id = str(refs.get("checkpoint_id") or "").strip()
    if checkpoint_id:
        checkpoint_ids.add(checkpoint_id)

selected_event_ids = set(selected_event_ids_sorted)
for row in route_chain_rows:
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    ev_id = str(lineage.get("continuity_event_id") or "").strip()
    if ev_id:
        selected_event_ids.add(ev_id)
selected_event_ids_sorted = sorted(selected_event_ids)

selected_task_ids = set(related_task_ids_sorted)
for row in task_chain_rows:
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    for task_id in lineage.get("linked_task_ids") or []:
        task_txt = str(task_id or "").strip()
        if task_txt:
            selected_task_ids.add(task_txt)
selected_task_ids_sorted = sorted(selected_task_ids)

checkpoint_rows: List[Dict[str, Any]] = []
checkpoint_ids_sorted = sorted(checkpoint_ids)
if checkpoint_ids_sorted:
    checkpoint_placeholders = ",".join("?" for _ in checkpoint_ids_sorted)
    sql_parts = [
        """
SELECT
  c.checkpoint_id AS linked_checkpoint_id,
  e.evidence_id,
  e.connector_type,
  e.connector_id,
  e.observed_at,
  e.monotonic_seq,
  e.subject_kind,
  e.subject_id,
  e.severity_max,
  e.jsonl_path,
  e.jsonl_line_no,
  e.facts_json,
  e.refs_json
FROM gtc_checkpoint_evidence c
JOIN gtc_evidence_index e ON e.evidence_id = c.evidence_id
""".strip(),
        f"WHERE c.checkpoint_id IN ({checkpoint_placeholders})",
    ]
    params_list: List[Any] = list(checkpoint_ids_sorted)

    if checkpoint_scope == "incident":
        scope_clauses: List[str] = []
        if selected_event_ids_sorted:
            ev_placeholders = ",".join("?" for _ in selected_event_ids_sorted)
            scope_clauses.append(f"json_extract(e.refs_json, '$.continuity_event_id') IN ({ev_placeholders})")
            params_list.extend(selected_event_ids_sorted)
        if selected_task_ids_sorted:
            task_placeholders = ",".join("?" for _ in selected_task_ids_sorted)
            scope_clauses.append(f"json_extract(e.refs_json, '$.task_id') IN ({task_placeholders})")
            params_list.extend(selected_task_ids_sorted)
        if scope_clauses:
            sql_parts.append("AND (" + " OR ".join(scope_clauses) + ")")

    sql_parts.append("ORDER BY e.observed_at ASC, e.monotonic_seq ASC LIMIT ?")
    params_list.append(max_checkpoint_events)
    rows = cur.execute("\n".join(sql_parts), tuple(params_list)).fetchall()
    for row in rows:
        checkpoint_rows.append(
            row_to_evidence(
                row,
                chain_scope="checkpoint",
                linked_checkpoint_id=str(row["linked_checkpoint_id"] or "").strip() or None,
            )
        )

chain_rows = merge_chain_rows(route_chain_rows + task_chain_rows + checkpoint_rows)
evidence_ids = [str(r.get("evidence_id") or "") for r in chain_rows if str(r.get("evidence_id") or "")]

artifact_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def upsert_artifact(
    *,
    role: str,
    sha: str,
    path: str,
    media_type: Optional[str] = None,
    bytes_size: Optional[int] = None,
    evidence_id: Optional[str] = None,
    artifact_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    role_norm = summarize_role(role)
    sha_norm = str(sha or "").strip().lower()
    path_norm = str(path or "").strip()
    if not sha_norm and not path_norm:
        return
    key = artifact_key(role_norm, sha_norm, path_norm)
    row = artifact_map.get(key)
    if row is None:
        row = {
            "role": role_norm,
            "sha256": sha_norm or None,
            "path": path_norm or None,
            "media_type": str(media_type or "").strip() or None,
            "bytes": int(bytes_size) if isinstance(bytes_size, int) and bytes_size >= 0 else None,
            "exists": resolve_exists(path_norm) if path_norm else None,
            "evidence_ids": set(),
            "artifact_types": set(),
            "metadata_samples": [],
        }
        artifact_map[key] = row

    if not row.get("media_type") and media_type:
        row["media_type"] = str(media_type)
    if row.get("bytes") is None and isinstance(bytes_size, int) and bytes_size >= 0:
        row["bytes"] = int(bytes_size)
    if row.get("exists") is None and path_norm:
        row["exists"] = resolve_exists(path_norm)

    if evidence_id:
        row["evidence_ids"].add(str(evidence_id))
    if artifact_type:
        row["artifact_types"].add(str(artifact_type))
    if isinstance(metadata, dict) and metadata:
        sample = metadata
        if sample not in row["metadata_samples"]:
            row["metadata_samples"].append(sample)


if evidence_ids:
    sql, params = sql_rows_for_ids(
        """
SELECT ea.evidence_id, ea.sha256, ea.role, a.media_type, a.bytes, a.path
FROM gtc_evidence_artifact ea
JOIN gtc_artifact a ON a.sha256 = ea.sha256
WHERE ea.evidence_id IN
""".strip(),
        evidence_ids,
        "ORDER BY ea.role ASC, ea.sha256 ASC, a.path ASC, ea.evidence_id ASC",
    )
    for row in cur.execute(sql, params).fetchall():
        upsert_artifact(
            role=str(row["role"] or "artifact"),
            sha=str(row["sha256"] or ""),
            path=str(row["path"] or ""),
            media_type=str(row["media_type"] or ""),
            bytes_size=int(row["bytes"] or 0),
            evidence_id=str(row["evidence_id"] or ""),
        )

for row in chain_rows:
    evidence_id = str(row.get("evidence_id") or "")
    refs = row.get("refs") if isinstance(row.get("refs"), dict) else {}

    refs_artifacts = refs.get("artifacts") if isinstance(refs.get("artifacts"), list) else []
    for art in refs_artifacts:
        if not isinstance(art, dict):
            continue
        upsert_artifact(
            role=str(art.get("role") or "artifact"),
            sha=str(art.get("sha256") or ""),
            path=str(art.get("path") or ""),
            media_type=str(art.get("media_type") or ""),
            evidence_id=evidence_id,
            metadata=art.get("metadata") if isinstance(art.get("metadata"), dict) else None,
        )

    manifest_rows = refs.get("task_artifact_manifest") if isinstance(refs.get("task_artifact_manifest"), list) else []
    for mrow in manifest_rows:
        if not isinstance(mrow, dict):
            continue
        artifact_type = str(mrow.get("artifact_type") or "").strip()
        role = str(mrow.get("role") or "").strip() or (f"task_artifact:{artifact_type}" if artifact_type else "artifact")
        upsert_artifact(
            role=role,
            sha=str(mrow.get("sha256") or ""),
            path=str(mrow.get("artifact_path") or mrow.get("path") or ""),
            media_type=str(mrow.get("media_type") or ""),
            evidence_id=evidence_id,
            artifact_type=artifact_type or None,
            metadata=mrow.get("metadata") if isinstance(mrow.get("metadata"), dict) else None,
        )

artifact_entries: List[Dict[str, Any]] = []
role_counts: Dict[str, int] = defaultdict(int)
media_counts: Dict[str, int] = defaultdict(int)
typed_task_roles: set[str] = set()

for key in sorted(artifact_map.keys(), key=lambda k: (k[0], k[1], k[2])):
    row = artifact_map[key]
    role = str(row.get("role") or "artifact")
    media_type = str(row.get("media_type") or "")
    role_counts[role] += 1
    if media_type:
        media_counts[media_type] += 1
    if role.startswith("task_artifact:"):
        typed_task_roles.add(role)

    artifact_entries.append(
        {
            "role": role,
            "sha256": row.get("sha256"),
            "path": row.get("path"),
            "media_type": row.get("media_type"),
            "bytes": row.get("bytes"),
            "exists": row.get("exists"),
            "evidence_ids": sorted(str(x) for x in (row.get("evidence_ids") or set()) if str(x)),
            "artifact_types": sorted(str(x) for x in (row.get("artifact_types") or set()) if str(x)),
            "metadata_samples": row.get("metadata_samples") or [],
        }
    )

connector_counts: Dict[str, int] = defaultdict(int)
for row in chain_rows:
    connector_counts[str(row.get("connector_type") or "unknown")] += 1

handoff_decisions = summarize_handoff_decisions(chain_rows)

bundle = {
    "ok": True,
    "schema_version": "gtc.incident_replay.bundle.v1",
    "generated_at": now_iso(),
    "build_generation_id": str(
        incident_surface.get("build_generation_id")
        or event_surface.get("build_generation_id")
        or ""
    ),
    "selection": {
        **selection_meta,
        "route_key": selected_row.get("route_key"),
        "incident_id": incident_id,
    },
    "source_surfaces": {
        "incident_replay_path": rel(latest_dir / "incident_replay.json"),
        "event_projection_path": rel(latest_dir / "event_projection.json"),
        "continuity_now_latest_path": rel(continuity_now_latest_path),
        "incident_replay_generated_at": incident_surface.get("generated_at"),
        "event_projection_generated_at": event_surface.get("generated_at"),
    },
    "incident": {
        "incident_id": incident_id,
        "route_key": selected_row.get("route_key"),
        "source": selected_row.get("source"),
        "event_key": selected_row.get("event_key"),
        "status": selected_row.get("status"),
        "severity": selected_row.get("severity"),
        "latest_event_id": selected_row.get("latest_event_id"),
        "latest_gtc_evidence_id": selected_row.get("latest_gtc_evidence_id"),
        "latest_summary": selected_row.get("latest_summary"),
        "open_since": selected_row.get("open_since"),
        "open_age_sec": selected_row.get("open_age_sec"),
        "recent_event_ids": selected_row.get("recent_event_ids") or [],
        "related_task_ids": selected_row.get("related_task_ids") or [],
    },
    "verify_gate_preflight": verify_gate_preflight_posture,
    "evidence_chain": {
        "count": len(chain_rows),
        "connector_counts": dict(sorted(connector_counts.items())),
        "task_ids": selected_task_ids_sorted,
        "checkpoint_ids": checkpoint_ids_sorted,
        "selected_event_ids": selected_event_ids_sorted,
        "checkpoint_scope": checkpoint_scope,
        "rows": chain_rows,
    },
    "artifact_pack": {
        "count": len(artifact_entries),
        "role_counts": dict(sorted(role_counts.items())),
        "media_type_counts": dict(sorted(media_counts.items())),
        "typed_task_roles": sorted(typed_task_roles),
        "entries": artifact_entries,
    },
    "handoff_decisions": handoff_decisions,
    "limits": {
        "max_route_events": max_route_events,
        "max_task_events": max_task_events,
        "max_checkpoint_events": max_checkpoint_events,
        "checkpoint_scope": checkpoint_scope,
    },
}

if write_output:
    if out_path_arg:
        out_json_path = pathlib.Path(out_path_arg)
        if not out_json_path.is_absolute():
            out_json_path = (root / out_json_path).resolve()
    else:
        out_json_path = (gtc_root / "incident_replay" / f"incident-{incident_id}.json").resolve()

    if out_md_arg:
        out_md_path = pathlib.Path(out_md_arg)
        if not out_md_path.is_absolute():
            out_md_path = (root / out_md_arg).resolve()
    else:
        out_md_path = out_json_path.with_suffix(".md")

    atomic_write_text(out_json_path, json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(out_md_path, build_markdown(bundle))

    bundle["bundle_paths"] = {
        "json": rel(out_json_path),
        "markdown": rel(out_md_path),
    }

con.close()

if json_out:
    print(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True))
else:
    lines = [
        "GTC INCIDENT REPLAY",
        f"- incident_id: {bundle['incident'].get('incident_id')}",
        f"- route_key: {bundle['incident'].get('route_key')}",
        f"- status/severity: {bundle['incident'].get('status')} / {bundle['incident'].get('severity')}",
        f"- evidence_chain.count: {bundle['evidence_chain'].get('count')}",
        f"- checkpoint_scope: {bundle['evidence_chain'].get('checkpoint_scope')}",
        f"- artifact_pack.count: {bundle['artifact_pack'].get('count')}",
        f"- typed_task_roles: {bundle['artifact_pack'].get('typed_task_roles')}",
        (
            "- handoff_binding: "
            f"status={((bundle.get('handoff_decisions') or {}).get('binding_integrity_status') or 'not_observed')} "
            f"degraded={(((bundle.get('handoff_decisions') or {}).get('binding_status_counts') or {}).get('degraded') or 0)}"
        ),
        (
            "- verify_gate_preflight: "
            f"mode={((bundle.get('verify_gate_preflight') or {}).get('mode') or 'unknown')} "
            f"blocker={((bundle.get('verify_gate_preflight') or {}).get('predicted_blocker_reason') or 'none')}"
        ),
    ]
    if isinstance(bundle.get("bundle_paths"), dict):
        lines.append(f"- replay_json: {bundle['bundle_paths'].get('json')}")
        lines.append(f"- replay_markdown: {bundle['bundle_paths'].get('markdown')}")
    print("\n".join(lines))
PY
