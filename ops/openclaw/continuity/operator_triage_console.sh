#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
REFRESH=0
JSON_OUT=0
MAX_ACTIONS=6
CRITIQUE_TASK=""
UI_EVIDENCE_BUNDLE=""
FEDERATED_EVIDENCE_TASK=""
FEDERATED_EVIDENCE_MAX_ITEMS="${OPENCLAW_OPERATOR_FEDERATED_EVIDENCE_MAX_ITEMS:-8}"
FEDERATED_EVIDENCE_QUERY=""
CRITIQUE_COOLDOWN_SEC="${OPENCLAW_OPERATOR_CRITIQUE_COOLDOWN_SEC:-300}"
CRITIQUE_MIN_AGE_SEC="${OPENCLAW_OPERATOR_CRITIQUE_MIN_AGE_SEC:-60}"
SHOW_STALE=0
SHOW_REDUNDANT=0
VERBOSE_STATE=0
SHOW_CHATTER=0
SHOW_RECOVERABLE_ERRORS=0
TASK_FRESHNESS_AGING_AFTER_SEC="${OPENCLAW_OPERATOR_TASK_FRESHNESS_AGING_AFTER_SEC:-600}"
TASK_FRESHNESS_STALE_AFTER_SEC="${OPENCLAW_OPERATOR_TASK_FRESHNESS_STALE_AFTER_SEC:-1800}"
VISIBILITY_HEALTHY_MIN="${OPENCLAW_OPERATOR_VISIBILITY_HEALTHY_MIN:-80}"
VISIBILITY_DEGRADED_MIN="${OPENCLAW_OPERATOR_VISIBILITY_DEGRADED_MIN:-55}"

usage() {
  cat <<'EOF'
Usage: operator_triage_console.sh [options]

High-signal operator triage console built from operator mission-control.

Options:
  --refresh             Refresh mission-control first
  --json                Print JSON payload
  --max-actions <n>     Limit recommended actions in output (default: 6)
  --critique-task <id>  Generate bounded critique packet for one active task
  --ui-evidence-bundle <path>
                        Attach UI evidence bundle context to critique output
  --federated-evidence-task <id>
                        Build federated evidence context for task id
                        (defaults to --critique-task when omitted)
  --federated-evidence-max-items <n>
                        Max federated evidence items to emit (default: 8)
  --federated-evidence-query <text>
                        Optional retrieval query text for federated evidence
                        ranking (defaults to task id when omitted)
  --critique-cooldown-sec <sec>
                        Per-task critique cooldown (default: 300)
  --critique-min-age-sec <sec>
                        Minimum observed task age required (default: 60)
  --show-stale          Show stale/aging low-signal cards and issues
  --show-redundant      Show redundant derivative cards/issues
  --verbose-state       Show low-impact state transitions in task cards
  --show-chatter        Show high-frequency low-information chatter cards/issues
  --show-recoverable-errors
                        Show recoverable errors before escalation threshold
  --task-aging-after-sec <sec>
                        Task freshness aging threshold (default: 600)
  --task-stale-after-sec <sec>
                        Task freshness stale threshold (default: 1800)
  --visibility-healthy-min <0-100>
                        Visibility score threshold for healthy (default: 80)
  --visibility-degraded-min <0-100>
                        Visibility score threshold for degraded (default: 55)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1
      shift
      ;;
    --json)
      JSON_OUT=1
      shift
      ;;
    --max-actions)
      MAX_ACTIONS="${2:-}"
      shift 2
      ;;
    --critique-task)
      CRITIQUE_TASK="${2:-}"
      shift 2
      ;;
    --ui-evidence-bundle)
      UI_EVIDENCE_BUNDLE="${2:-}"
      shift 2
      ;;
    --federated-evidence-task)
      FEDERATED_EVIDENCE_TASK="${2:-}"
      shift 2
      ;;
    --federated-evidence-max-items)
      FEDERATED_EVIDENCE_MAX_ITEMS="${2:-}"
      shift 2
      ;;
    --federated-evidence-query)
      FEDERATED_EVIDENCE_QUERY="${2:-}"
      shift 2
      ;;
    --critique-cooldown-sec)
      CRITIQUE_COOLDOWN_SEC="${2:-}"
      shift 2
      ;;
    --critique-min-age-sec)
      CRITIQUE_MIN_AGE_SEC="${2:-}"
      shift 2
      ;;
    --show-stale)
      SHOW_STALE=1
      shift
      ;;
    --show-redundant)
      SHOW_REDUNDANT=1
      shift
      ;;
    --verbose-state)
      VERBOSE_STATE=1
      shift
      ;;
    --show-chatter)
      SHOW_CHATTER=1
      shift
      ;;
    --show-recoverable-errors)
      SHOW_RECOVERABLE_ERRORS=1
      shift
      ;;
    --task-aging-after-sec)
      TASK_FRESHNESS_AGING_AFTER_SEC="${2:-}"
      shift 2
      ;;
    --task-stale-after-sec)
      TASK_FRESHNESS_STALE_AFTER_SEC="${2:-}"
      shift 2
      ;;
    --visibility-healthy-min)
      VISIBILITY_HEALTHY_MIN="${2:-}"
      shift 2
      ;;
    --visibility-degraded-min)
      VISIBILITY_DEGRADED_MIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

python3 - "$ROOT" "$REFRESH" "$JSON_OUT" "$MAX_ACTIONS" "$CRITIQUE_TASK" "$UI_EVIDENCE_BUNDLE" "$CRITIQUE_COOLDOWN_SEC" "$CRITIQUE_MIN_AGE_SEC" "$SHOW_STALE" "$SHOW_REDUNDANT" "$VERBOSE_STATE" "$SHOW_CHATTER" "$SHOW_RECOVERABLE_ERRORS" "$TASK_FRESHNESS_AGING_AFTER_SEC" "$TASK_FRESHNESS_STALE_AFTER_SEC" "$VISIBILITY_HEALTHY_MIN" "$VISIBILITY_DEGRADED_MIN" "$FEDERATED_EVIDENCE_TASK" "$FEDERATED_EVIDENCE_MAX_ITEMS" "$FEDERATED_EVIDENCE_QUERY" <<'PY'
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
refresh = bool(int(sys.argv[2]))
json_out = bool(int(sys.argv[3]))

try:
    max_actions = max(1, int(sys.argv[4]))
except Exception:
    raise SystemExit("operator_triage_console_invalid_max_actions")

critique_task = str(sys.argv[5] or "").strip()
ui_evidence_bundle_arg = str(sys.argv[6] or "").strip()

try:
    critique_cooldown_sec = max(0, int(sys.argv[7]))
except Exception:
    raise SystemExit("operator_triage_console_invalid_critique_cooldown_sec")

try:
    critique_min_age_sec = max(0, int(sys.argv[8]))
except Exception:
    raise SystemExit("operator_triage_console_invalid_critique_min_age_sec")

show_stale = bool(int(sys.argv[9]))
show_redundant = bool(int(sys.argv[10]))
verbose_state = bool(int(sys.argv[11]))
show_chatter = bool(int(sys.argv[12]))
show_recoverable_errors = bool(int(sys.argv[13]))

try:
    task_freshness_aging_after_sec = max(60, int(sys.argv[14]))
except Exception:
    raise SystemExit("operator_triage_console_invalid_task_freshness_aging_after_sec")

try:
    task_freshness_stale_after_sec = max(task_freshness_aging_after_sec + 60, int(sys.argv[15]))
except Exception:
    raise SystemExit("operator_triage_console_invalid_task_freshness_stale_after_sec")

try:
    visibility_healthy_min = max(0, min(100, int(sys.argv[16])))
except Exception:
    raise SystemExit("operator_triage_console_invalid_visibility_healthy_min")

try:
    visibility_degraded_min = max(0, min(100, int(sys.argv[17])))
except Exception:
    raise SystemExit("operator_triage_console_invalid_visibility_degraded_min")

federated_evidence_task_arg = str(sys.argv[18] or "").strip()

try:
    federated_evidence_max_items = max(1, min(32, int(sys.argv[19])))
except Exception:
    raise SystemExit("operator_triage_console_invalid_federated_evidence_max_items")

federated_evidence_query_arg = str(sys.argv[20] or "").strip()

if visibility_degraded_min > visibility_healthy_min:
    visibility_degraded_min = visibility_healthy_min


def env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    return max(minimum, min(maximum, value))


b7_candidate_max_total = env_int(
    "OPENCLAW_OPERATOR_B7_CANDIDATE_MAX_TOTAL",
    6,
    minimum=1,
    maximum=20,
)
b7_candidate_max_now = env_int(
    "OPENCLAW_OPERATOR_B7_CANDIDATE_MAX_NOW",
    3,
    minimum=1,
    maximum=b7_candidate_max_total,
)
b7_candidate_max_later = env_int(
    "OPENCLAW_OPERATOR_B7_CANDIDATE_MAX_LATER",
    3,
    minimum=1,
    maximum=b7_candidate_max_total,
)

mission_script = root / "ops" / "openclaw" / "continuity" / "operator_mission_control.sh"
mission_export_path = root / "state" / "continuity" / "latest" / "operator_mission_control.json"
triage_export_path = root / "state" / "continuity" / "latest" / "operator_triage_console.json"
component_consistency_overlay_path = root / "state" / "continuity" / "latest" / "operator_component_consistency_audit_overlay.json"
critique_packet_dir = root / "state" / "continuity" / "latest" / "operator_task_state_critiques"
critique_index_path = root / "state" / "continuity" / "latest" / "operator_task_state_critique_index.json"
memory_consolidation_ledger_path = root / "memory" / "consolidation_ledger.jsonl"
memory_consolidation_runtime_latest_path = root / "state" / "continuity" / "latest" / "memory_consolidation_latest.json"
research_case_registry_path = root / "state" / "continuity" / "latest" / "research_case_registry.json"
research_case_capacity_runtime_path = root / "state" / "continuity" / "latest" / "research_case_capacity_orchestration_runtime.json"
production_knowledge_ingestion_latest_path = root / "state" / "continuity" / "latest" / "production_knowledge_ingestion_latest.json"
source_of_truth_map_guard_latest_path = root / "state" / "continuity" / "latest" / "source_of_truth_map_drift_latest.json"
federated_doc_stale_after_sec = 7 * 24 * 3600


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    return parsed


def age_sec(raw: Any) -> Optional[int]:
    ts = parse_iso(raw)
    if ts is None:
        return None
    return max(0, int((dt.datetime.now(dt.timezone.utc) - ts).total_seconds()))


def unique_preserve(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        txt = str(value or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


def clamp_percent(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


def visibility_rating(score: float) -> str:
    val = clamp_percent(score)
    if val >= visibility_healthy_min:
        return "healthy"
    if val >= visibility_degraded_min:
        return "degraded"
    return "critical"


def score_freshness_dimension(status: str, worst_age_sec: Optional[int]) -> float:
    normalized = str(status or "unknown").strip().lower()
    if normalized == "fresh":
        if isinstance(worst_age_sec, int) and worst_age_sec >= 0:
            ratio = min(1.0, float(worst_age_sec) / float(max(1, task_freshness_aging_after_sec)))
            return clamp_percent(100.0 - (ratio * 18.0))
        return 96.0
    if normalized == "aging":
        if isinstance(worst_age_sec, int) and worst_age_sec >= 0:
            span = max(1, task_freshness_stale_after_sec - task_freshness_aging_after_sec)
            over = max(0, worst_age_sec - task_freshness_aging_after_sec)
            ratio = min(1.0, float(over) / float(span))
            return clamp_percent(72.0 - (ratio * 24.0))
        return 58.0
    if normalized == "stale":
        if isinstance(worst_age_sec, int) and worst_age_sec >= 0:
            over = max(0, worst_age_sec - task_freshness_stale_after_sec)
            decay = min(22.0, float(over) / 120.0)
            return clamp_percent(42.0 - decay)
        return 34.0
    return 48.0


def score_worker_health_dimension(worker_state_status: str, worker_state_obj: Dict[str, Any]) -> float:
    status = str(worker_state_status or "unknown").strip().lower()
    base = 85.0
    if status == "ready":
        base = 96.0
    elif status == "degraded":
        base = 64.0
    elif status == "blocked":
        base = 32.0
    base -= 10.0 * float(int(worker_state_obj.get("demoted_worker_count") or 0))
    base -= 14.0 * float(int(worker_state_obj.get("probe_overdue_worker_count") or 0))
    base -= 6.0 * float(int(worker_state_obj.get("probe_due_now_worker_count") or 0))
    if bool(worker_state_obj.get("fail_closed") is True):
        base -= 12.0
    return clamp_percent(base)


def score_execution_stability_dimension(worker_state_obj: Dict[str, Any]) -> float:
    score = 100.0
    score -= 12.0 * float(int(worker_state_obj.get("demoted_worker_count") or 0))
    score -= 8.0 * float(int(worker_state_obj.get("restore_pending_worker_count") or 0))
    score -= 10.0 * float(int(worker_state_obj.get("probe_due_now_worker_count") or 0))
    score -= 20.0 * float(int(worker_state_obj.get("probe_overdue_worker_count") or 0))
    if bool(worker_state_obj.get("fail_closed") is True):
        score -= 24.0
    return clamp_percent(score)


def task_slug(value: str) -> str:
    allowed = []
    for ch in str(value or ""):
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    return slug[:120] if slug else "task"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def optional_nonnegative_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return max(0, parsed)


def first_nonempty(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        txt = str(candidate or "").strip()
        if txt:
            return txt
    return None


def first_signal_reason(*candidates: Any) -> Optional[str]:
    ignore_tokens = {
        "",
        "none",
        "null",
        "n/a",
        "unknown",
        "idle",
        "clear",
        "ok",
    }
    for candidate in candidates:
        values = candidate if isinstance(candidate, list) else [candidate]
        for value in values:
            txt = str(value or "").strip()
            if not txt:
                continue
            if txt.lower() in ignore_tokens:
                continue
            return txt
    return None


def unique_signal_tokens(*candidates: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for candidate in candidates:
        values = candidate if isinstance(candidate, list) else [candidate]
        for value in values:
            txt = str(value or "").strip()
            if not txt or txt in seen:
                continue
            seen.add(txt)
            merged.append(txt)
    return merged


def classify_freshness(*ages: Optional[int], stale_after_sec: int = 1800, aging_after_sec: int = 600) -> Dict[str, Any]:
    valid = [age for age in ages if isinstance(age, int) and age >= 0]
    if not valid:
        return {
            "status": "unknown",
            "reason": "age_unknown",
            "worst_age_sec": None,
        }
    worst = max(valid)
    if worst >= stale_after_sec:
        return {
            "status": "stale",
            "reason": f"worst_age_exceeds_{stale_after_sec}s",
            "worst_age_sec": worst,
        }
    if worst >= aging_after_sec:
        return {
            "status": "aging",
            "reason": f"worst_age_exceeds_{aging_after_sec}s",
            "worst_age_sec": worst,
        }
    return {
        "status": "fresh",
        "reason": None,
        "worst_age_sec": worst,
    }


_MEMORY_CONSOLIDATION_FAILURE_CLASSIFICATION_BRIDGE: Dict[str, Dict[str, str]] = {
    "FAILED_GOVERNANCE_GATE": {
        "classification": "FAILED_OTHER",
        "outcome_class": "FAILED",
        "reason": "memory_governance_gate_blocked",
    },
    "FAILED_SOURCE_VALIDATION": {
        "classification": "FAILED_NO_ARTIFACT",
        "outcome_class": "INVALID_OUTPUT",
        "reason": "memory_source_validation_failed",
    },
    "FAILED_ARCHIVE_MOVE": {
        "classification": "FAILED_OTHER",
        "outcome_class": "FAILED",
        "reason": "memory_archive_move_failed",
    },
    "FAILED_POST_ARTIFACT_VALIDATION": {
        "classification": "FAILED_NO_ARTIFACT",
        "outcome_class": "INVALID_OUTPUT",
        "reason": "memory_post_artifact_validation_failed",
    },
    "FAILED_ROLLBACK_VALIDATION": {
        "classification": "FAILED_OTHER",
        "outcome_class": "FAILED",
        "reason": "memory_rollback_validation_failed",
    },
}


def memory_consolidation_failure_bridge(code: Any) -> Optional[Dict[str, str]]:
    token = str(code or "").strip().upper()
    if not token:
        return None
    row = _MEMORY_CONSOLIDATION_FAILURE_CLASSIFICATION_BRIDGE.get(token)
    if not isinstance(row, dict):
        return None
    return {
        "failure_code": token,
        "classification": str(row.get("classification") or "FAILED_OTHER").strip() or "FAILED_OTHER",
        "outcome_class": str(row.get("outcome_class") or "FAILED").strip() or "FAILED",
        "reason": str(row.get("reason") or "memory_consolidation_failure").strip() or "memory_consolidation_failure",
    }


def collect_id_list(*candidates: Any) -> List[str]:
    collected: List[str] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            for row in candidate:
                txt = str(row or "").strip()
                if txt:
                    collected.append(txt)
            continue
        txt = str(candidate or "").strip()
        if txt:
            collected.append(txt)
    return unique_preserve(collected)


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


def load_json_if_exists(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def summarize_source_of_truth_map_guard(path: pathlib.Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": rel_path_for(path),
        "present": bool(path.exists()),
        "payload_valid": False,
        "status": "neutral",
        "reason": "source_of_truth_map_guard_artifact_missing",
        "decision": None,
        "block_reason": None,
        "block_reasons": [],
        "generated_at": None,
        "checker_version": None,
    }

    if not path.exists():
        return summary

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        summary.update(
            {
                "status": "critical",
                "reason": "source_of_truth_map_guard_artifact_unreadable",
                "payload_valid": False,
            }
        )
        return summary

    if not isinstance(payload, dict):
        summary.update(
            {
                "status": "critical",
                "reason": "source_of_truth_map_guard_artifact_not_object",
                "payload_valid": False,
            }
        )
        return summary

    decision_raw = str(payload.get("decision") or "").strip().upper()
    block_reason = first_signal_reason(payload.get("block_reason"))
    block_reasons = unique_preserve([str(x) for x in (payload.get("block_reasons") or []) if str(x or "").strip()])

    summary.update(
        {
            "payload_valid": True,
            "decision": decision_raw or None,
            "block_reason": block_reason,
            "block_reasons": block_reasons,
            "generated_at": payload.get("generated_at"),
            "checker_version": payload.get("checker_version"),
        }
    )

    if decision_raw == "PASS":
        summary["status"] = "healthy"
        summary["reason"] = "source_of_truth_map_guard_pass"
    elif decision_raw == "BLOCK":
        summary["status"] = "critical"
        summary["reason"] = block_reason or (block_reasons[0] if block_reasons else "source_of_truth_map_guard_blocked")
    else:
        summary["status"] = "degraded"
        summary["reason"] = "source_of_truth_map_guard_decision_unknown"

    return summary


def rel_path_for(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def resolve_workspace_path(raw_path: str) -> Optional[pathlib.Path]:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return None
    p = pathlib.Path(candidate).expanduser()
    return p if p.is_absolute() else (root / p)


def bool_from_any(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_overlay_severity(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if txt in {"critical", "high", "medium", "low"}:
        return txt
    return "low"


def visibility_informed_overlay_consumption_decision(
    *,
    overlay_severity: str,
    visibility_state: str,
    visibility_score: float,
    task_noise_signal_tags: List[str],
    task_suppressed: bool,
    task_suppression_reasons: List[str],
) -> Tuple[str, str]:
    """
    Decide whether a B7 overlay finding should be promoted into B8 critique findings
    or ignored as low-signal for this critique context.

    Returns:
      ("promote"|"ignore", reason_code)
    """

    severity = normalize_overlay_severity(overlay_severity)
    visibility = str(visibility_state or "unknown").strip().lower()

    noise_tags = {
        str(tag or "").strip().lower()
        for tag in (task_noise_signal_tags or [])
        if str(tag or "").strip()
    }
    suppression_reasons = {
        str(reason or "").strip().lower()
        for reason in (task_suppression_reasons or [])
        if str(reason or "").strip()
    }
    low_signal_noise_tags = {
        "staleness",
        "redundancy",
        "low_impact_state_change",
        "chatter",
        "recoverable_error",
    }

    if severity in {"critical", "high"}:
        return ("promote", "high_severity")

    if visibility == "critical":
        return ("promote", "critical_visibility")

    if severity == "medium" and visibility in {"degraded", "critical"}:
        return ("promote", "degraded_visibility")

    if task_suppressed and severity in {"low", "medium"}:
        return ("ignore", "task_suppressed_by_noise_signal_policy")

    if visibility == "healthy":
        if severity == "low":
            if (noise_tags & low_signal_noise_tags) or suppression_reasons:
                return ("ignore", "healthy_visibility_low_severity_noise")
            return ("ignore", "healthy_visibility_low_severity")

        if severity == "medium" and (noise_tags & low_signal_noise_tags):
            return ("ignore", "healthy_visibility_medium_severity_noise")

        if severity == "medium" and float(visibility_score) >= float(visibility_healthy_min):
            return ("ignore", "healthy_visibility_medium_severity")

    return ("promote", "default")


def canonical_lane_for_overlay_rule(rule_id: Any) -> Optional[str]:
    rule = str(rule_id or "").strip().lower()
    mapping = {
        "count_mismatch": "CPL-04",
        "state_discrepancy": "EX-11",
        "status_rollup_error": "EX-11",
        "freshness_contradiction": "OBS-01",
        "suppression_logic_error": "PR-02",
        "blockage_reason_mismatch": "OBS-03",
    }
    return mapping.get(rule)


def build_candidate_opportunity_surface(
    *,
    overlay_findings: List[Dict[str, Any]],
    promoted_finding_ids: List[str],
    ignored_finding_ids: List[str],
    ignored_reason_by_id: Dict[str, str],
    critique_status: str,
    visibility_state: str,
    task_suppressed: bool,
    task_noise_signal_tags: List[str],
) -> Dict[str, Any]:
    promoted_set = set([str(x or "").strip() for x in promoted_finding_ids if str(x or "").strip()])
    ignored_set = set([str(x or "").strip() for x in ignored_finding_ids if str(x or "").strip()])

    preliminary: List[Dict[str, Any]] = []
    duplicate_input_count = 0
    seen_keys = set()

    for finding in overlay_findings:
        if not isinstance(finding, dict):
            continue

        finding_id = str(finding.get("finding_id") or "").strip()
        audit_rule_id = str(finding.get("audit_rule_id") or "unknown").strip().lower() or "unknown"
        severity = normalize_overlay_severity(finding.get("severity"))
        finding_text = str(finding.get("finding_text") or "").strip() or "Consistency mismatch detected."

        dedupe_key = "|".join(
            [
                audit_rule_id,
                " ".join(finding_text.lower().split()),
            ]
        )
        if dedupe_key in seen_keys:
            duplicate_input_count += 1
            continue
        seen_keys.add(dedupe_key)

        canonical_lane = canonical_lane_for_overlay_rule(audit_rule_id)

        verdict = "later"
        verdict_reason = "queued_for_review"

        if not finding_id:
            verdict = "reject"
            verdict_reason = "missing_finding_id"
        elif canonical_lane is None:
            verdict = "reject"
            verdict_reason = "unmapped_rule"
        elif finding_id in ignored_set:
            verdict = "reject"
            verdict_reason = ignored_reason_by_id.get(finding_id) or "low_signal_policy_reject"
        elif finding_id in promoted_set or severity in {"critical", "high"}:
            verdict = "now"
            verdict_reason = "promoted_or_high_severity"
        elif severity == "medium":
            verdict = "later"
            verdict_reason = "medium_severity_follow_on"
        elif severity == "low":
            verdict = "reject"
            verdict_reason = "low_severity_anti_spam"
        else:
            verdict = "reject"
            verdict_reason = "unknown_severity"

        confidence = 0.62
        if verdict == "now":
            confidence = 0.9 if severity in {"critical", "high"} else 0.78
        elif verdict == "later":
            confidence = 0.7
        elif verdict_reason in {
            "low_severity_anti_spam",
            "low_signal_policy_reject",
            "healthy_visibility_low_severity",
            "healthy_visibility_low_severity_noise",
        }:
            confidence = 0.84

        preliminary.append(
            {
                "source_finding_id": finding_id or None,
                "audit_rule_id": audit_rule_id,
                "title": short_text(finding_text, limit=180),
                "severity": severity,
                "canonical_lane": canonical_lane,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "confidence": round(confidence, 2),
            }
        )

    verdict_rank = {"now": 0, "later": 1, "reject": 2}
    preliminary = sorted(
        preliminary,
        key=lambda row: (
            verdict_rank.get(str(row.get("verdict") or "reject"), 9),
            -severity_rank(row.get("severity")),
            str(row.get("audit_rule_id") or ""),
            str(row.get("source_finding_id") or ""),
        ),
    )

    surfaced: List[Dict[str, Any]] = []
    bounded_drop_count = 0
    now_count = 0
    later_count = 0
    reject_count = 0

    for row in preliminary:
        verdict = str(row.get("verdict") or "reject")
        reason = str(row.get("verdict_reason") or "")

        if verdict == "now" and now_count >= b7_candidate_max_now:
            verdict = "later"
            reason = "demoted_now_cap_reached"

        if verdict == "later" and later_count >= b7_candidate_max_later:
            verdict = "reject"
            reason = "later_cap_reached"

        if len(surfaced) >= b7_candidate_max_total:
            bounded_drop_count += 1
            continue

        row_out = dict(row)
        row_out["verdict"] = verdict
        row_out["verdict_reason"] = reason
        row_out["candidate_id"] = f"sys01_{task_slug(str(row.get('source_finding_id') or row.get('audit_rule_id') or 'candidate'))}"
        row_out["review_state"] = "review_required" if verdict in {"now", "later"} else "rejected"
        surfaced.append(row_out)

        if verdict == "now":
            now_count += 1
        elif verdict == "later":
            later_count += 1
        else:
            reject_count += 1

    low_signal_reject_count = sum(
        1
        for row in surfaced
        if str(row.get("verdict") or "") == "reject"
        and str(row.get("verdict_reason") or "")
        in {
            "low_severity_anti_spam",
            "low_signal_policy_reject",
            "healthy_visibility_low_severity",
            "healthy_visibility_low_severity_noise",
            "task_suppressed_by_noise_signal_policy",
        }
    )

    status = "ready" if surfaced else "empty"
    return {
        "schema": "clawd.b7_candidate_opportunity_surface.v1",
        "generated_at": now_iso(),
        "status": status,
        "policy": {
            "max_total_candidates": b7_candidate_max_total,
            "max_now_candidates": b7_candidate_max_now,
            "max_later_candidates": b7_candidate_max_later,
            "reject_low_signal_candidates": True,
            "reject_unmapped_rules": True,
            "deduplicate_by_rule_and_text": True,
        },
        "summary": {
            "input_finding_count": len(overlay_findings),
            "considered_finding_count": len(preliminary),
            "candidate_count": len(surfaced),
            "now_count": now_count,
            "later_count": later_count,
            "reject_count": reject_count,
            "duplicate_input_count": duplicate_input_count,
            "bounded_drop_count": bounded_drop_count,
            "low_signal_reject_count": low_signal_reject_count,
        },
        "context": {
            "critique_status": str(critique_status or "not_requested"),
            "visibility_state": str(visibility_state or "unknown"),
            "task_suppressed": bool(task_suppressed is True),
            "task_noise_signal_tags": unique_preserve(
                [str(tag or "").strip().lower() for tag in task_noise_signal_tags if str(tag or "").strip()]
            ),
        },
        "candidates": surfaced,
    }


def overlay_finding_id(rule_id: str, index: int) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in str(rule_id or ""))
    token = token.strip("_-") or "rule"
    return f"cf_{token}_{max(1, index):03d}"


def extract_ui_state_from_bundle(bundle_payload: Dict[str, Any], bundle_path: pathlib.Path) -> Dict[str, Any]:
    evidence = bundle_payload.get("evidence") if isinstance(bundle_payload.get("evidence"), dict) else {}
    inline_ui_state = evidence.get("ui_state") if isinstance(evidence.get("ui_state"), dict) else None
    if isinstance(inline_ui_state, dict):
        return {
            "status": "ready",
            "ui_state": inline_ui_state,
            "source": "evidence.ui_state",
            "artifact_path": None,
        }

    state_snapshot = evidence.get("state_snapshot") if isinstance(evidence.get("state_snapshot"), dict) else {}
    inline_snapshot_state = state_snapshot.get("ui_state") if isinstance(state_snapshot.get("ui_state"), dict) else None
    if isinstance(inline_snapshot_state, dict):
        return {
            "status": "ready",
            "ui_state": inline_snapshot_state,
            "source": "evidence.state_snapshot.ui_state",
            "artifact_path": None,
        }

    artifact_ref = state_snapshot.get("artifact_ref") if isinstance(state_snapshot.get("artifact_ref"), dict) else {}
    ref_uri = str(artifact_ref.get("ref_uri") or "").strip()
    if not ref_uri:
        return {
            "status": "missing_ui_state",
            "reason": "ui_state_not_present",
            "ui_state": None,
            "source": None,
            "artifact_path": None,
        }

    ref_path = pathlib.Path(ref_uri).expanduser()
    candidates: List[pathlib.Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.append((root / ref_path))
        candidates.append((bundle_path.parent / ref_path))

    seen_paths = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        resolved_key = str(resolved)
        if resolved_key in seen_paths:
            continue
        seen_paths.add(resolved_key)

        payload = load_json_if_exists(resolved)
        if not isinstance(payload, dict):
            continue
        ui_state = payload.get("ui_state") if isinstance(payload.get("ui_state"), dict) else payload
        if isinstance(ui_state, dict):
            return {
                "status": "ready",
                "ui_state": ui_state,
                "source": "evidence.state_snapshot.artifact_ref",
                "artifact_path": rel_path_for(resolved),
            }

    return {
        "status": "missing_ui_state",
        "reason": "state_snapshot_artifact_unreadable",
        "ui_state": None,
        "source": "evidence.state_snapshot.artifact_ref",
        "artifact_path": None,
    }


def audit_component_consistency_overlay(
    *,
    ui_state: Dict[str, Any],
    bundle_id: Optional[str],
    bundle_path: Optional[str],
    ui_state_source: Optional[str],
    ui_state_artifact_path: Optional[str],
) -> Dict[str, Any]:
    execution_snapshot = ui_state.get("execution_snapshot") if isinstance(ui_state.get("execution_snapshot"), dict) else {}
    task_detail = execution_snapshot.get("task_detail") if isinstance(execution_snapshot.get("task_detail"), dict) else {}
    worker_state = execution_snapshot.get("worker_state") if isinstance(execution_snapshot.get("worker_state"), dict) else {}
    worker_state_detail = execution_snapshot.get("worker_state_detail") if isinstance(execution_snapshot.get("worker_state_detail"), dict) else {}
    blockage = execution_snapshot.get("blockage") if isinstance(execution_snapshot.get("blockage"), dict) else {}
    system_visibility = execution_snapshot.get("visibility_scorecard") if isinstance(execution_snapshot.get("visibility_scorecard"), dict) else {}
    noise_signal = ui_state.get("noise_signal") if isinstance(ui_state.get("noise_signal"), dict) else {}
    policy_heuristics = noise_signal.get("heuristics") if isinstance(noise_signal.get("heuristics"), dict) else {}
    task_policy = task_detail.get("noise_signal_policy") if isinstance(task_detail.get("noise_signal_policy"), dict) else {}
    thresholds = noise_signal.get("thresholds") if isinstance(noise_signal.get("thresholds"), dict) else {}

    overlay_policy = {
        "show_stale": bool_from_any(task_policy.get("show_stale") if "show_stale" in task_policy else policy_heuristics.get("show_stale"), False),
        "show_redundant": bool_from_any(task_policy.get("show_redundant") if "show_redundant" in task_policy else policy_heuristics.get("show_redundant"), False),
        "show_chatter": bool_from_any(task_policy.get("show_chatter") if "show_chatter" in task_policy else policy_heuristics.get("show_chatter"), False),
    }
    aging_threshold_sec = max(60, safe_int(thresholds.get("task_freshness_aging_after_sec"), task_freshness_aging_after_sec))

    active_task_cards = task_detail.get("active_task_cards") if isinstance(task_detail.get("active_task_cards"), list) else []
    suppressed_task_cards = task_detail.get("suppressed_task_cards") if isinstance(task_detail.get("suppressed_task_cards"), list) else []

    critical_visibility_present = False
    for row in [*active_task_cards, *suppressed_task_cards]:
        if not isinstance(row, dict):
            continue
        if str(row.get("visibility_rating") or "").strip().lower() == "critical":
            critical_visibility_present = True
            break

    findings: List[Dict[str, Any]] = []

    def add_finding(
        *,
        rule_id: str,
        finding_text: str,
        severity: str,
        inconsistent_paths: List[str],
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
    ) -> None:
        normalized_paths = [str(path or "").strip() for path in inconsistent_paths if str(path or "").strip()]
        row: Dict[str, Any] = {
            "finding_id": overlay_finding_id(rule_id, len(findings) + 1),
            "author_id": "component_consistency_audit_overlay.v1",
            "created_at": now_iso(),
            "finding_text": str(finding_text or "").strip(),
            "severity": normalize_overlay_severity(severity),
            "audit_rule_id": str(rule_id or "").strip() or "unknown_rule",
            "inconsistent_paths": normalized_paths,
        }
        if expected_value is not None:
            row["expected_value"] = str(expected_value)
        if actual_value is not None:
            row["actual_value"] = str(actual_value)
        findings.append(row)

    active_task_count = optional_nonnegative_int(execution_snapshot.get("active_task_count"))
    all_task_card_count = optional_nonnegative_int(task_detail.get("all_task_card_count"))
    if all_task_card_count is None:
        all_task_card_count = len([row for row in [*active_task_cards, *suppressed_task_cards] if isinstance(row, dict)])
    if active_task_count is not None and all_task_card_count is not None and active_task_count != all_task_card_count:
        severity = "high"
        if str(system_visibility.get("rating") or "").strip().lower() == "healthy":
            severity = "medium"
        add_finding(
            rule_id="count_mismatch",
            finding_text=(
                "Execution snapshot active task count does not match operator task-card rollup count."
            ),
            severity=severity,
            inconsistent_paths=[
                "/execution_snapshot/active_task_count",
                "/execution_snapshot/task_detail/all_task_card_count",
            ],
            expected_value=f"matching counts ({all_task_card_count})",
            actual_value=f"active_task_count={active_task_count}, all_task_card_count={all_task_card_count}",
        )

    foreground_ids = {
        str(task_id or "").strip()
        for task_id in (task_detail.get("foreground_task_ids") or [])
        if str(task_id or "").strip()
    }
    visible_index: Dict[str, int] = {}
    for idx, row in enumerate(active_task_cards):
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        if task_id and task_id not in visible_index:
            visible_index[task_id] = idx
    for task_id in sorted(foreground_ids):
        idx = visible_index.get(task_id)
        if idx is None:
            continue
        row = active_task_cards[idx]
        if bool_from_any((row or {}).get("foreground"), False):
            continue
        add_finding(
            rule_id="state_discrepancy",
            finding_text=(
                f"Foreground task '{task_id}' is reported as non-foreground in active task-card detail."
            ),
            severity="medium",
            inconsistent_paths=[
                "/execution_snapshot/task_detail/foreground_task_ids",
                f"/execution_snapshot/task_detail/active_task_cards/{idx}/foreground",
            ],
            expected_value="foreground=true for listed foreground task ids",
            actual_value=f"task_id={task_id}, foreground={row.get('foreground')}",
        )
        if len(findings) >= 24:
            break

    reported_worker_status = str(worker_state_detail.get("status") or "unknown").strip().lower()
    expected_worker_status = "ready"
    if int(worker_state.get("probe_overdue_worker_count") or 0) > 0 or bool(worker_state.get("fail_closed") is True):
        expected_worker_status = "blocked"
    elif (
        int(worker_state.get("demoted_worker_count") or 0) > 0
        or int(worker_state.get("restore_pending_worker_count") or 0) > 0
        or int(worker_state.get("probe_due_now_worker_count") or 0) > 0
    ):
        expected_worker_status = "degraded"

    if reported_worker_status not in {"", "unknown"} and reported_worker_status != expected_worker_status:
        add_finding(
            rule_id="status_rollup_error",
            finding_text="Worker status rollup conflicts with detailed worker-health counters.",
            severity="high",
            inconsistent_paths=[
                "/execution_snapshot/worker_state_detail/status",
                "/execution_snapshot/worker_state",
            ],
            expected_value=f"worker_state_detail.status={expected_worker_status}",
            actual_value=f"worker_state_detail.status={reported_worker_status}",
        )

    freshness_obj = task_detail.get("freshness") if isinstance(task_detail.get("freshness"), dict) else {}
    freshness_status = str(freshness_obj.get("status") or "unknown").strip().lower()
    freshness_age_candidates = [
        optional_nonnegative_int(freshness_obj.get("worst_age_sec")),
        optional_nonnegative_int(freshness_obj.get("last_progress_age_sec")),
    ]
    freshness_age = max([age for age in freshness_age_candidates if isinstance(age, int)], default=None)
    if freshness_status == "fresh" and isinstance(freshness_age, int) and freshness_age >= aging_threshold_sec:
        if overlay_policy.get("show_stale") or critical_visibility_present:
            add_finding(
                rule_id="freshness_contradiction",
                finding_text=(
                    "Task freshness is marked healthy/fresh while freshness ages exceed the configured aging threshold."
                ),
                severity="medium",
                inconsistent_paths=[
                    "/execution_snapshot/task_detail/freshness/status",
                    "/execution_snapshot/task_detail/freshness/last_progress_age_sec",
                ],
                expected_value=f"freshness age below {aging_threshold_sec}s when status=fresh",
                actual_value=f"status=fresh, age_sec={freshness_age}",
            )

    for idx, row in enumerate(suppressed_task_cards):
        if not isinstance(row, dict):
            continue
        if str(row.get("visibility_rating") or "").strip().lower() != "critical":
            continue
        add_finding(
            rule_id="suppression_logic_error",
            finding_text=(
                f"Suppressed task-card '{str(row.get('task_id') or 'unknown')}' has critical visibility rating."
            ),
            severity="low",
            inconsistent_paths=[
                f"/execution_snapshot/task_detail/suppressed_task_cards/{idx}/visibility_rating",
                "/execution_snapshot/task_detail/noise_signal_policy",
            ],
            expected_value="critical visibility cards should not be suppressed",
            actual_value=f"visibility_rating={row.get('visibility_rating')}",
        )

    blocked_rollup = (
        reported_worker_status == "blocked"
        or str(blockage.get("program_state") or "").strip().lower() == "blocked"
        or str(blockage.get("dispatch_status") or "").strip().lower() == "blocked"
    )
    if blocked_rollup:
        reason_count = optional_nonnegative_int(blockage.get("reason_count"))
        block_reason = str(blockage.get("autonomous_dispatch_block_reason") or "").strip()
        if (reason_count is not None and reason_count <= 0) or not block_reason:
            add_finding(
                rule_id="blockage_reason_mismatch",
                finding_text="Blocked rollup is set, but blockage reason details are empty or internally inconsistent.",
                severity="high",
                inconsistent_paths=[
                    "/execution_snapshot/worker_state_detail/status",
                    "/execution_snapshot/blockage/reason_count",
                    "/execution_snapshot/blockage/autonomous_dispatch_block_reason",
                ],
                expected_value="blocked rollup must include reason_count>0 and explicit autonomous_dispatch_block_reason",
                actual_value=f"reason_count={reason_count}, autonomous_dispatch_block_reason={block_reason or '<empty>'}",
            )

    critical_finding_count = len(
        [
            row
            for row in findings
            if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() in {"critical", "high"}
        ]
    )

    return {
        "schema": "clawd.component_consistency_audit_overlay.v1",
        "overlay_id": "component_consistency_audit_overlay.v1",
        "generated_at": now_iso(),
        "status": "applied" if findings else "clean",
        "bundle_id": bundle_id,
        "bundle_path": bundle_path,
        "ui_state_source": ui_state_source,
        "ui_state_artifact_path": ui_state_artifact_path,
        "policy": overlay_policy,
        "rule_ids_evaluated": [
            "count_mismatch",
            "state_discrepancy",
            "status_rollup_error",
            "freshness_contradiction",
            "suppression_logic_error",
            "blockage_reason_mismatch",
        ],
        "finding_count": len(findings),
        "critical_finding_count": critical_finding_count,
        "findings": findings,
    }


def summarize_ui_evidence_bundle(bundle_arg: str) -> Optional[Dict[str, Any]]:
    candidate = str(bundle_arg or "").strip()
    if not candidate:
        return None
    bundle_path_raw = pathlib.Path(candidate).expanduser()
    bundle_path = bundle_path_raw if bundle_path_raw.is_absolute() else (root / bundle_path_raw)
    bundle_path = bundle_path.resolve()

    summary: Dict[str, Any] = {
        "status": "invalid",
        "bundle_path": rel_path_for(bundle_path),
    }
    bundle_payload = load_json_if_exists(bundle_path)
    if bundle_payload is None:
        summary["error"] = "bundle_unreadable"
        return summary

    schema_version = str(bundle_payload.get("schema_version") or "").strip()
    bundle_id = str(bundle_payload.get("bundle_id") or "").strip()
    findings_list = bundle_payload.get("findings") if isinstance(bundle_payload.get("findings"), list) else []
    valid_schema = schema_version in {
        "clawd.b8_ui_evidence_bundle.v1",
        "ui_evidence_bundle.v1",
        "clawd.b8_ui_evidence_bundle.v2",
        "ui_evidence_bundle.v2",
    }
    if not valid_schema:
        summary["error"] = "unsupported_schema_version"
        summary["schema_version"] = schema_version or None
        return summary
    if not bundle_id:
        summary["error"] = "bundle_id_missing"
        return summary

    execution_evidence_link_count = 0
    execution_context = (
        bundle_payload.get("execution_context")
        if isinstance(bundle_payload.get("execution_context"), dict)
        else {}
    )
    for finding in findings_list:
        if not isinstance(finding, dict):
            continue
        if isinstance(finding.get("execution_evidence_link"), dict):
            execution_evidence_link_count += 1
        evidence_links = finding.get("evidence_links") if isinstance(finding.get("evidence_links"), list) else []
        for link in evidence_links:
            if not isinstance(link, dict):
                continue
            if isinstance(link.get("execution_evidence_link"), dict):
                execution_evidence_link_count += 1
                continue
            link_type = str(link.get("link_type") or "").strip().lower()
            execution_context_ref = str(link.get("execution_context_ref") or "").strip()
            if link_type in {
                "artifact_awareness",
                "quota_routing",
                "stale_wave",
                "blocking_classification",
                "drift_truthfulness",
            } and execution_context_ref:
                execution_evidence_link_count += 1

    validation_gate_log = (
        bundle_payload.get("validation_gate_log")
        if isinstance(bundle_payload.get("validation_gate_log"), list)
        else []
    )
    validation_gate_failure_count = len(
        [
            row
            for row in validation_gate_log
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "failed"
        ]
    )
    validation_gate_pass_count = len(
        [
            row
            for row in validation_gate_log
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "passed"
        ]
    )
    validation_gate_skipped_count = len(
        [
            row
            for row in validation_gate_log
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "skipped"
        ]
    )

    is_v2_bundle = schema_version in {"clawd.b8_ui_evidence_bundle.v2", "ui_evidence_bundle.v2"}
    execution_link_status = "not_reported"
    if is_v2_bundle:
        if execution_evidence_link_count > 0:
            execution_link_status = "linked"
        elif findings_list:
            execution_link_status = "missing"
        else:
            execution_link_status = "not_required"

    validation_gate_status = "not_reported"
    if validation_gate_log:
        if validation_gate_failure_count > 0:
            validation_gate_status = "failed"
        elif validation_gate_pass_count > 0:
            validation_gate_status = "passed"
        else:
            validation_gate_status = "neutral"

    return {
        "status": "linked",
        "bundle_path": rel_path_for(bundle_path),
        "bundle_id": bundle_id,
        "schema_version": schema_version,
        "finding_count": len(findings_list),
        "critical_finding_count": len(
            [
                row
                for row in findings_list
                if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() in {"critical", "high"}
            ]
        ),
        "captured_at": (
            (bundle_payload.get("capture") or {}).get("captured_at")
            if isinstance(bundle_payload.get("capture"), dict)
            else (bundle_payload.get("evidence") or {}).get("captured_at")
            if isinstance(bundle_payload.get("evidence"), dict)
            else None
        ),
        "execution_link_status": execution_link_status,
        "execution_evidence_link_count": execution_evidence_link_count,
        "execution_lesson_tag_count": len(
            execution_context.get("execution_lesson_tags")
            if isinstance(execution_context.get("execution_lesson_tags"), list)
            else []
        ),
        "validation_gate_status": validation_gate_status,
        "validation_gate_count": len(validation_gate_log),
        "validation_gate_failure_count": validation_gate_failure_count,
        "validation_gate_pass_count": validation_gate_pass_count,
        "validation_gate_skipped_count": validation_gate_skipped_count,
    }


def parse_jsonl_if_exists(path: pathlib.Path, *, max_rows: int = 4000) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                if idx >= max_rows:
                    break
                txt = line.strip()
                if not txt:
                    continue
                try:
                    obj = json.loads(txt)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def build_task_terms(task_id: str) -> List[str]:
    raw = str(task_id or "").strip().lower()
    if not raw:
        return []
    terms = [raw]
    pieces = []
    token = []
    for ch in raw:
        if ch.isalnum():
            token.append(ch)
            continue
        if token:
            pieces.append("".join(token))
            token = []
    if token:
        pieces.append("".join(token))
    for piece in pieces:
        if len(piece) >= 3:
            terms.append(piece)
    return unique_preserve(terms)


def build_retrieval_terms(*values: Any) -> List[str]:
    terms: List[str] = []
    for value in values:
        txt = str(value or "").strip()
        if not txt:
            continue
        terms.extend(build_task_terms(txt))
    return unique_preserve([term for term in terms if str(term or "").strip()])


def matched_retrieval_terms(terms: List[str], *fields: Any, limit: int = 8) -> List[str]:
    if not terms:
        return []
    haystack = " ".join([str(field or "") for field in fields]).lower()
    if not haystack.strip():
        return []
    out: List[str] = []
    for term in terms:
        txt = str(term or "").strip().lower()
        if not txt:
            continue
        if txt in haystack and txt not in out:
            out.append(txt)
        if len(out) >= max(1, int(limit)):
            break
    return out


def match_retrieval_relevance(terms: List[str], *fields: Any) -> float:
    matches = matched_retrieval_terms(terms, *fields, limit=32)
    if not matches:
        return 0.0

    score = 0.22
    score += min(0.58, 0.1 * float(len(matches)))
    longest = max((len(str(term or "")) for term in matches), default=0)
    if longest >= 10:
        score += 0.12
    elif longest >= 6:
        score += 0.06
    return clamp_score(score)


def match_task_relevance(task_id: str, *fields: Any) -> float:
    terms = build_task_terms(task_id)
    if not terms:
        return 0.0
    haystack = " ".join([str(field or "") for field in fields]).lower()
    if not haystack.strip():
        return 0.0

    score = 0.0
    full = terms[0]
    if full and full in haystack:
        score += 0.7

    bonus_terms = [term for term in terms[1:] if term and term in haystack]
    score += min(0.3, 0.08 * float(len(bonus_terms)))
    return clamp_score(score)


def short_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def severity_rank(value: Any) -> int:
    txt = str(value or "").strip().lower()
    if txt == "critical":
        return 4
    if txt == "high":
        return 3
    if txt == "medium":
        return 2
    if txt == "low":
        return 1
    return 0


def scalar_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    txt = str(value).strip()
    return txt if txt else "null"


def build_component_consistency_audit_overlay(
    *,
    execution_snapshot: Dict[str, Any],
    all_task_cards: List[Dict[str, Any]],
    visible_task_cards: List[Dict[str, Any]],
    suppressed_task_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    finding_counter = 0
    suppressed_by_policy_count = 0

    def add_finding(
        *,
        rule_id: str,
        severity: str,
        finding_text: str,
        inconsistent_paths: List[str],
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        context_task_ids: Optional[List[str]] = None,
    ) -> None:
        nonlocal finding_counter
        finding_counter += 1
        suffix = task_slug(f"{rule_id}_{finding_counter}").lower()
        row: Dict[str, Any] = {
            "finding_id": f"cf_{suffix}",
            "author_id": "component_consistency_audit_overlay.v1",
            "created_at": now_iso(),
            "finding_text": str(finding_text or "").strip() or "Consistency mismatch detected.",
            "severity": severity if severity in {"critical", "high", "medium", "low"} else "low",
            "audit_rule_id": str(rule_id or "unknown").strip() or "unknown",
            "inconsistent_paths": [
                str(path or "").strip()
                for path in inconsistent_paths
                if str(path or "").strip()
            ],
        }
        if expected_value is not None:
            row["expected_value"] = scalar_text(expected_value)
        if actual_value is not None:
            row["actual_value"] = scalar_text(actual_value)
        normalized_task_ids = unique_preserve(
            [
                str(task_id or "").strip()
                for task_id in (context_task_ids or [])
                if str(task_id or "").strip()
            ]
        )
        if normalized_task_ids:
            row["context_task_ids"] = normalized_task_ids
        findings.append(row)

    task_detail = execution_snapshot.get("task_detail") if isinstance(execution_snapshot.get("task_detail"), dict) else {}
    worker_state = execution_snapshot.get("worker_state") if isinstance(execution_snapshot.get("worker_state"), dict) else {}
    worker_state_detail = (
        execution_snapshot.get("worker_state_detail")
        if isinstance(execution_snapshot.get("worker_state_detail"), dict)
        else {}
    )
    blockage = execution_snapshot.get("blockage") if isinstance(execution_snapshot.get("blockage"), dict) else {}
    visibility_scorecard = (
        execution_snapshot.get("visibility_scorecard")
        if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
        else {}
    )

    active_task_ids = [str(task_id or "").strip() for task_id in (execution_snapshot.get("active_task_ids") or []) if str(task_id or "").strip()]
    foreground_ids = set(str(task_id or "").strip() for task_id in (task_detail.get("foreground_task_ids") or []) if str(task_id or "").strip())
    blocked_ids = set(str(task_id or "").strip() for task_id in (task_detail.get("blocked_candidate_task_ids") or []) if str(task_id or "").strip())
    critique_available_ids = set(
        str(task_id or "").strip()
        for task_id in (task_detail.get("critique_available_task_ids") or [])
        if str(task_id or "").strip()
    )

    count_checks = [
        (
            "active_task_count",
            "/execution_snapshot/active_task_count",
            int(execution_snapshot.get("active_task_count") or 0),
            len(active_task_ids),
            [
                "/execution_snapshot/active_task_count",
                "/execution_snapshot/active_task_ids",
            ],
        ),
        (
            "blocked_candidate_count",
            "/execution_snapshot/blockage/blocked_candidate_count",
            int(blockage.get("blocked_candidate_count") or 0),
            len(blocked_ids),
            [
                "/execution_snapshot/blockage/blocked_candidate_count",
                "/execution_snapshot/task_detail/blocked_candidate_task_ids",
            ],
        ),
        (
            "all_task_card_count",
            "/execution_snapshot/task_detail/all_task_card_count",
            int(task_detail.get("all_task_card_count") or 0),
            len(all_task_cards),
            [
                "/execution_snapshot/task_detail/all_task_card_count",
                "/execution_snapshot/task_detail/active_task_cards",
                "/execution_snapshot/task_detail/suppressed_task_cards",
            ],
        ),
        (
            "suppressed_task_card_count",
            "/execution_snapshot/task_detail/suppressed_task_card_count",
            int(task_detail.get("suppressed_task_card_count") or 0),
            len(suppressed_task_cards),
            [
                "/execution_snapshot/task_detail/suppressed_task_card_count",
                "/execution_snapshot/task_detail/suppressed_task_cards",
            ],
        ),
        (
            "visibility_task_count",
            "/execution_snapshot/visibility_scorecard/task_count",
            int(visibility_scorecard.get("task_count") or 0),
            len(all_task_cards),
            [
                "/execution_snapshot/visibility_scorecard/task_count",
                "/execution_snapshot/task_detail/all_task_card_count",
            ],
        ),
        (
            "visibility_visible_task_count",
            "/execution_snapshot/visibility_scorecard/visible_task_count",
            int(visibility_scorecard.get("visible_task_count") or 0),
            len(visible_task_cards),
            [
                "/execution_snapshot/visibility_scorecard/visible_task_count",
                "/execution_snapshot/task_detail/active_task_cards",
            ],
        ),
        (
            "visibility_suppressed_task_count",
            "/execution_snapshot/visibility_scorecard/suppressed_task_count",
            int(visibility_scorecard.get("suppressed_task_count") or 0),
            len(suppressed_task_cards),
            [
                "/execution_snapshot/visibility_scorecard/suppressed_task_count",
                "/execution_snapshot/task_detail/suppressed_task_cards",
            ],
        ),
    ]

    for count_name, _count_path, expected, actual, pointers in count_checks:
        if expected != actual:
            add_finding(
                rule_id="count_mismatch",
                severity="high",
                finding_text=(
                    f"Summary count `{count_name}` does not match derived card/list cardinality."
                ),
                inconsistent_paths=pointers,
                expected_value=expected,
                actual_value=actual,
            )

    for idx, card in enumerate(visible_task_cards):
        if not isinstance(card, dict):
            continue
        task_id = str(card.get("task_id") or "").strip()
        if not task_id:
            continue
        card_foreground = bool(card.get("foreground") is True)
        card_blocked = bool(card.get("blocked") is True)
        card_critique_available = bool(card.get("critique_available") is True)

        expected_foreground = task_id in foreground_ids
        if card_foreground != expected_foreground:
            add_finding(
                rule_id="state_discrepancy",
                severity="medium",
                finding_text=(
                    f"Task `{task_id}` foreground membership is inconsistent between rollup list and task card."
                ),
                inconsistent_paths=[
                    "/execution_snapshot/task_detail/foreground_task_ids",
                    f"/execution_snapshot/task_detail/active_task_cards/{idx}/foreground",
                ],
                expected_value=expected_foreground,
                actual_value=card_foreground,
                context_task_ids=[task_id],
            )

        expected_blocked = task_id in blocked_ids
        if card_blocked != expected_blocked:
            add_finding(
                rule_id="state_discrepancy",
                severity="medium",
                finding_text=(
                    f"Task `{task_id}` blocked membership is inconsistent between blocked candidate rollup and task card."
                ),
                inconsistent_paths=[
                    "/execution_snapshot/task_detail/blocked_candidate_task_ids",
                    f"/execution_snapshot/task_detail/active_task_cards/{idx}/blocked",
                ],
                expected_value=expected_blocked,
                actual_value=card_blocked,
                context_task_ids=[task_id],
            )

        expected_critique = task_id in critique_available_ids
        if card_critique_available != expected_critique:
            add_finding(
                rule_id="state_discrepancy",
                severity="medium",
                finding_text=(
                    f"Task `{task_id}` critique-availability projection disagrees with task-detail rollup list."
                ),
                inconsistent_paths=[
                    "/execution_snapshot/task_detail/critique_available_task_ids",
                    f"/execution_snapshot/task_detail/active_task_cards/{idx}/critique_available",
                ],
                expected_value=expected_critique,
                actual_value=card_critique_available,
                context_task_ids=[task_id],
            )

    rollup_status = str(worker_state_detail.get("status") or "unknown").strip().lower()
    degraded_signals = int(worker_state.get("demoted_worker_count") or 0) + int(worker_state.get("restore_pending_worker_count") or 0) + int(worker_state.get("probe_due_now_worker_count") or 0)
    blocked_signals = int(worker_state.get("probe_overdue_worker_count") or 0) + (1 if bool(worker_state.get("fail_closed") is True) else 0)

    if rollup_status == "ready" and (degraded_signals > 0 or blocked_signals > 0):
        add_finding(
            rule_id="status_rollup_error",
            severity="high",
            finding_text="Worker rollup status reports ready despite degraded/blocked worker-level signals.",
            inconsistent_paths=[
                "/execution_snapshot/worker_state_detail/status",
                "/execution_snapshot/worker_state",
            ],
            expected_value="degraded_or_blocked",
            actual_value="ready",
        )
    if rollup_status == "blocked" and blocked_signals == 0:
        add_finding(
            rule_id="status_rollup_error",
            severity="high",
            finding_text="Worker rollup status reports blocked with no blocked-class worker indicators present.",
            inconsistent_paths=[
                "/execution_snapshot/worker_state_detail/status",
                "/execution_snapshot/worker_state/probe_overdue_worker_count",
                "/execution_snapshot/worker_state/fail_closed",
            ],
            expected_value="blocked_signal_present",
            actual_value="none",
        )
    if rollup_status == "degraded" and degraded_signals == 0 and blocked_signals == 0:
        add_finding(
            rule_id="status_rollup_error",
            severity="high",
            finding_text="Worker rollup status reports degraded with no degraded/blocked worker indicators present.",
            inconsistent_paths=[
                "/execution_snapshot/worker_state_detail/status",
                "/execution_snapshot/worker_state",
            ],
            expected_value="degraded_signal_present",
            actual_value="none",
        )

    freshness_obj = task_detail.get("freshness") if isinstance(task_detail.get("freshness"), dict) else {}
    freshness_status = str(freshness_obj.get("status") or "unknown").strip().lower()
    worst_age_sec = optional_nonnegative_int(freshness_obj.get("worst_age_sec"))
    freshness_contradiction_text: Optional[str] = None
    if freshness_status == "fresh" and isinstance(worst_age_sec, int) and worst_age_sec >= task_freshness_aging_after_sec:
        freshness_contradiction_text = (
            f"Freshness status is fresh while worst_age_sec ({worst_age_sec}) exceeds aging threshold ({task_freshness_aging_after_sec})."
        )
    elif freshness_status == "aging" and isinstance(worst_age_sec, int) and (
        worst_age_sec < task_freshness_aging_after_sec or worst_age_sec >= task_freshness_stale_after_sec
    ):
        freshness_contradiction_text = (
            f"Freshness status is aging but worst_age_sec ({worst_age_sec}) falls outside aging range [{task_freshness_aging_after_sec}, {task_freshness_stale_after_sec})."
        )
    elif freshness_status == "stale" and isinstance(worst_age_sec, int) and worst_age_sec < task_freshness_stale_after_sec:
        freshness_contradiction_text = (
            f"Freshness status is stale while worst_age_sec ({worst_age_sec}) is below stale threshold ({task_freshness_stale_after_sec})."
        )

    if freshness_contradiction_text:
        if not show_stale and str(visibility_scorecard.get("rating") or "").strip().lower() != "critical":
            suppressed_by_policy_count += 1
        else:
            add_finding(
                rule_id="freshness_contradiction",
                severity="medium",
                finding_text=freshness_contradiction_text,
                inconsistent_paths=[
                    "/execution_snapshot/task_detail/freshness/status",
                    "/execution_snapshot/task_detail/freshness/worst_age_sec",
                ],
                expected_value="status_threshold_alignment",
                actual_value=f"status={freshness_status};worst_age_sec={scalar_text(worst_age_sec)}",
            )

    for idx, card in enumerate(suppressed_task_cards):
        if not isinstance(card, dict):
            continue
        task_id = str(card.get("task_id") or "").strip()
        visibility_rating_txt = str(card.get("visibility_rating") or "").strip().lower()
        if visibility_rating_txt == "critical":
            add_finding(
                rule_id="suppression_logic_error",
                severity="low",
                finding_text=(
                    f"Task `{task_id or f'suppressed_card_{idx}'}` is suppressed despite critical visibility rating."
                ),
                inconsistent_paths=[
                    f"/execution_snapshot/task_detail/suppressed_task_cards/{idx}/visibility_rating",
                    "/execution_snapshot/task_detail/noise_signal_policy",
                ],
                expected_value="not_suppressed_when_visibility_is_critical",
                actual_value="suppressed",
                context_task_ids=[task_id] if task_id else None,
            )

    blockage_reason_count = int(blockage.get("reason_count") or 0)
    autonomous_block_reason = str(blockage.get("autonomous_dispatch_block_reason") or "").strip()
    if rollup_status == "blocked" and (blockage_reason_count <= 0 or not autonomous_block_reason):
        add_finding(
            rule_id="blockage_reason_mismatch",
            severity="high",
            finding_text=(
                "Execution is reported blocked while blockage reason rollup is incomplete "
                "(reason_count missing/zero or autonomous dispatch block reason absent)."
            ),
            inconsistent_paths=[
                "/execution_snapshot/worker_state_detail/status",
                "/execution_snapshot/blockage/reason_count",
                "/execution_snapshot/blockage/autonomous_dispatch_block_reason",
            ],
            expected_value="reason_count>0_and_autonomous_dispatch_block_reason_present",
            actual_value=(
                f"reason_count={blockage_reason_count};"
                f"autonomous_dispatch_block_reason={'present' if autonomous_block_reason else 'missing'}"
            ),
        )

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in findings:
        severity = str(row.get("severity") or "").strip().lower()
        if severity in severity_counts:
            severity_counts[severity] += 1

    return {
        "schema": "clawd.component_consistency_audit_overlay.v1",
        "author_id": "component_consistency_audit_overlay.v1",
        "generated_at": now_iso(),
        "policy": {
            "show_stale": show_stale,
            "show_redundant": show_redundant,
            "show_chatter": show_chatter,
            "include_suppressed_items": False,
            "suppressed_by_policy_count": suppressed_by_policy_count,
        },
        "summary": {
            "rule_count": 6,
            "finding_count": len(findings),
            "critical_count": severity_counts["critical"],
            "high_count": severity_counts["high"],
            "medium_count": severity_counts["medium"],
            "low_count": severity_counts["low"],
        },
        "findings": findings,
    }


def build_federated_evidence(
    *,
    task_id: str,
    max_items: int,
    federated_query: str,
    ui_bundle_summary: Optional[Dict[str, Any]],
    ui_bundle_arg: str,
    component_consistency_overlay: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    started_at = dt.datetime.now(dt.timezone.utc)
    clean_task_id = str(task_id or "").strip()
    retrieval_query = str(federated_query or "").strip() or clean_task_id
    retrieval_terms = build_retrieval_terms(clean_task_id, retrieval_query)
    if not retrieval_terms:
        retrieval_terms = build_task_terms(clean_task_id)

    sources_consulted: List[Dict[str, Any]] = []
    evidence_items: List[Dict[str, Any]] = []
    degraded_reasons: List[str] = []

    memory_source: Dict[str, Any] = {
        "source_type": "memory_consolidation_ledger",
        "path": rel_path_for(memory_consolidation_ledger_path),
        "status": "missing",
        "rows_scanned": 0,
        "applied_batches_seen": 0,
        "matched_entries": 0,
    }
    if memory_consolidation_ledger_path.exists() and memory_consolidation_ledger_path.is_file():
        ledger_rows = parse_jsonl_if_exists(memory_consolidation_ledger_path, max_rows=5000)
        memory_source["rows_scanned"] = len(ledger_rows)
        applied_rows = [
            row
            for row in ledger_rows
            if isinstance(row, dict) and str(row.get("event_type") or "").strip() == "CONSOLIDATION_APPLIED"
        ]
        memory_source["applied_batches_seen"] = len(applied_rows)
        if not applied_rows:
            memory_source["status"] = "no_applied_batches"
        else:
            memory_source["status"] = "ok"
            for row in reversed(applied_rows[-20:]):
                outputs = row.get("outputs") if isinstance(row.get("outputs"), dict) else {}
                artifact_rel = str(outputs.get("artifact_path") or "").strip()
                if not artifact_rel:
                    continue
                artifact_abs = (root / artifact_rel).resolve()
                artifact_payload = load_json_if_exists(artifact_abs)
                if not isinstance(artifact_payload, dict):
                    memory_source["status"] = "invalid"
                    degraded_reasons.append("memory_consolidation_artifact_invalid")
                    continue
                consolidated_entries = (
                    artifact_payload.get("consolidated_entries")
                    if isinstance(artifact_payload.get("consolidated_entries"), list)
                    else []
                )
                for entry in consolidated_entries:
                    if not isinstance(entry, dict):
                        continue
                    source_path = str(entry.get("source_path") or "").strip()
                    title = str(entry.get("title") or "").strip()
                    summary = str(entry.get("summary") or "").strip()
                    relevance = max(
                        match_task_relevance(clean_task_id, source_path, title, summary),
                        match_retrieval_relevance(retrieval_terms, source_path, title, summary),
                    )
                    if relevance <= 0.0:
                        continue
                    matched_terms = matched_retrieval_terms(retrieval_terms, source_path, title, summary)
                    memory_source["matched_entries"] = int(memory_source.get("matched_entries") or 0) + 1
                    evidence_items.append(
                        {
                            "type": "memory_consolidation_entry",
                            "source": rel_path_for(artifact_abs),
                            "timestamp": row.get("recorded_at") or artifact_payload.get("generated_at"),
                            "batch_id": row.get("batch_id") or artifact_payload.get("batch_id"),
                            "source_path": source_path or None,
                            "title": title or None,
                            "summary": short_text(summary or title, limit=240),
                            "relevance": relevance,
                            "matched_terms": matched_terms,
                        }
                    )
            if int(memory_source.get("matched_entries") or 0) == 0:
                memory_source["status"] = "no_task_matches"

    sources_consulted.append(memory_source)

    runtime_source: Dict[str, Any] = {
        "source_type": "memory_consolidation_runtime_latest",
        "path": rel_path_for(memory_consolidation_runtime_latest_path),
        "status": "missing",
    }
    runtime_payload = load_json_if_exists(memory_consolidation_runtime_latest_path)
    if memory_consolidation_runtime_latest_path.exists() and memory_consolidation_runtime_latest_path.is_file():
        if isinstance(runtime_payload, dict):
            runtime_status_raw = str(runtime_payload.get("status") or "").strip() or "unknown"
            runtime_failure_code = str(runtime_payload.get("failure_code") or "").strip() or None
            runtime_failure_bridge = memory_consolidation_failure_bridge(runtime_failure_code)
            runtime_gate_failure = (
                runtime_payload.get("last_validation_gate_failure")
                if isinstance(runtime_payload.get("last_validation_gate_failure"), dict)
                else {}
            )
            runtime_failed_gate = str(runtime_gate_failure.get("gate") or "").strip() or None

            runtime_source.update(
                {
                    "raw_status": runtime_status_raw,
                    "failure_code": runtime_failure_code,
                    "last_validation_gate_failure_gate": runtime_failed_gate,
                    "failure_classification": (
                        str(runtime_failure_bridge.get("classification") or "").strip() or None
                        if isinstance(runtime_failure_bridge, dict)
                        else None
                    ),
                    "failure_outcome_class": (
                        str(runtime_failure_bridge.get("outcome_class") or "").strip() or None
                        if isinstance(runtime_failure_bridge, dict)
                        else None
                    ),
                }
            )
            if runtime_status_raw in {"applied", "dry_run"} and runtime_failure_code is None:
                runtime_source["status"] = "ok"
            elif runtime_status_raw in {"blocked_governance_gate"}:
                runtime_source["status"] = "blocked"
            elif runtime_status_raw in {"failed", "failed_rolled_back", "error"} or runtime_failure_code:
                runtime_source["status"] = "fail_closed"
            else:
                runtime_source["status"] = "unknown"

            if runtime_source.get("status") in {"blocked", "fail_closed"}:
                reason_bits = [
                    f"status={runtime_status_raw}",
                    f"failure_code={runtime_failure_code or 'none'}",
                    f"failed_gate={runtime_failed_gate or 'none'}",
                ]
                degraded_reasons.append(f"memory_runtime_{runtime_status_raw}")
                if runtime_failure_code:
                    degraded_reasons.append(f"memory_runtime_failure_code:{runtime_failure_code}")
                if isinstance(runtime_failure_bridge, dict):
                    degraded_reasons.append(
                        f"memory_runtime_failure_classification:{runtime_failure_bridge.get('classification')}"
                    )
                evidence_items.append(
                    {
                        "type": "memory_consolidation_runtime_status",
                        "source": rel_path_for(memory_consolidation_runtime_latest_path),
                        "timestamp": runtime_payload.get("generated_at") or runtime_payload.get("updated_at") or now_iso(),
                        "summary": short_text("; ".join(reason_bits), limit=220),
                        "failure_code": runtime_failure_code,
                        "failure_classification": (
                            str(runtime_failure_bridge.get("classification") or "").strip() or None
                            if isinstance(runtime_failure_bridge, dict)
                            else None
                        ),
                        "failure_outcome_class": (
                            str(runtime_failure_bridge.get("outcome_class") or "").strip() or None
                            if isinstance(runtime_failure_bridge, dict)
                            else None
                        ),
                        "relevance": 0.78 if runtime_source.get("status") == "fail_closed" else 0.6,
                        "matched_terms": matched_retrieval_terms(retrieval_terms, reason_bits),
                    }
                )
        else:
            runtime_source["status"] = "invalid"
            degraded_reasons.append("memory_runtime_payload_invalid")

    sources_consulted.append(runtime_source)

    research_registry_source: Dict[str, Any] = {
        "source_type": "research_case_registry",
        "path": rel_path_for(research_case_registry_path),
        "status": "missing",
        "entry_count": 0,
        "matched_entries": 0,
    }
    research_registry_payload = load_json_if_exists(research_case_registry_path)
    if research_case_registry_path.exists() and research_case_registry_path.is_file():
        if isinstance(research_registry_payload, dict):
            research_entries = (
                research_registry_payload.get("cases")
                if isinstance(research_registry_payload.get("cases"), list)
                else []
            )
            research_registry_source["entry_count"] = len(research_entries)
            if not research_entries:
                research_registry_source["status"] = "empty_registry"
            else:
                research_registry_source["status"] = "ok"
                for case in research_entries[:200]:
                    if not isinstance(case, dict):
                        continue
                    case_id = str(case.get("case_id") or "").strip()
                    candidate_id = str(case.get("candidate_id") or "").strip()
                    primary_state = str(case.get("primary_state") or "").strip()
                    summary = str(case.get("summary") or "").strip()
                    hypothesis = str(case.get("hypothesis") or "").strip()
                    tags = case.get("tags") if isinstance(case.get("tags"), list) else []
                    case_path = str(case.get("path") or "").strip()
                    relevance = max(
                        match_task_relevance(clean_task_id, case_id, candidate_id, primary_state, summary, hypothesis, case_path),
                        match_retrieval_relevance(
                            retrieval_terms,
                            case_id,
                            candidate_id,
                            primary_state,
                            summary,
                            hypothesis,
                            case_path,
                            " ".join([str(tag or "") for tag in tags if str(tag or "").strip()]),
                        ),
                    )
                    if relevance <= 0.0:
                        continue
                    matched_terms = matched_retrieval_terms(
                        retrieval_terms,
                        case_id,
                        candidate_id,
                        primary_state,
                        summary,
                        hypothesis,
                        case_path,
                        " ".join([str(tag or "") for tag in tags if str(tag or "").strip()]),
                    )
                    research_registry_source["matched_entries"] = int(research_registry_source.get("matched_entries") or 0) + 1
                    evidence_items.append(
                        {
                            "type": "research_case_entry",
                            "source": rel_path_for(research_case_registry_path),
                            "timestamp": case.get("updated_at") or research_registry_payload.get("generated_at"),
                            "case_id": case_id or None,
                            "candidate_id": candidate_id or None,
                            "primary_state": primary_state or None,
                            "summary": short_text(summary or hypothesis or case_id, limit=240),
                            "relevance": relevance,
                            "matched_terms": matched_terms,
                        }
                    )
                if int(research_registry_source.get("matched_entries") or 0) == 0:
                    research_registry_source["status"] = "no_task_matches"
        else:
            research_registry_source["status"] = "invalid"
            degraded_reasons.append("research_case_registry_payload_invalid")
    sources_consulted.append(research_registry_source)

    research_capacity_source: Dict[str, Any] = {
        "source_type": "research_case_capacity_runtime",
        "path": rel_path_for(research_case_capacity_runtime_path),
        "status": "missing",
        "alert_count": 0,
        "selected_case_count": 0,
    }
    research_capacity_payload = load_json_if_exists(research_case_capacity_runtime_path)
    if research_case_capacity_runtime_path.exists() and research_case_capacity_runtime_path.is_file():
        if isinstance(research_capacity_payload, dict):
            alerts_raw = (
                research_capacity_payload.get("alerts")
                if isinstance(research_capacity_payload.get("alerts"), list)
                else []
            )
            alerts = [str(alert or "").strip() for alert in alerts_raw if str(alert or "").strip()]
            selected_case_ids = (
                research_capacity_payload.get("selected_case_ids")
                if isinstance(research_capacity_payload.get("selected_case_ids"), list)
                else []
            )
            fairness = (
                research_capacity_payload.get("fairness")
                if isinstance(research_capacity_payload.get("fairness"), dict)
                else {}
            )
            oldest_wait = int(fairness.get("oldest_runnable_case_wait_seconds") or 0)
            research_capacity_source.update(
                {
                    "alert_count": len(alerts),
                    "selected_case_count": len(selected_case_ids),
                    "oldest_runnable_case_wait_seconds": oldest_wait,
                    "policy_id": str(research_capacity_payload.get("policy_id") or "").strip() or None,
                }
            )
            research_capacity_source["status"] = "blocked" if alerts else "ok"
            if alerts:
                for alert in alerts:
                    degraded_reasons.append(f"research_capacity_alert:{alert}")
                summary_bits = [
                    f"alerts={','.join(alerts)}",
                    f"selected_case_count={len(selected_case_ids)}",
                    f"oldest_wait_seconds={oldest_wait}",
                ]
                matched_terms = matched_retrieval_terms(
                    retrieval_terms,
                    " ".join(summary_bits),
                    " ".join(alerts),
                    research_capacity_payload.get("policy_id"),
                )
                query_relevance = match_retrieval_relevance(
                    retrieval_terms,
                    " ".join(summary_bits),
                    " ".join(alerts),
                    research_capacity_payload.get("policy_id"),
                )
                evidence_items.append(
                    {
                        "type": "research_capacity_runtime_status",
                        "source": rel_path_for(research_case_capacity_runtime_path),
                        "timestamp": research_capacity_payload.get("generated_at") or now_iso(),
                        "summary": short_text("; ".join(summary_bits), limit=240),
                        "policy_id": research_capacity_payload.get("policy_id"),
                        "alerts": alerts,
                        "relevance": clamp_score(max(0.52, query_relevance)),
                        "matched_terms": matched_terms,
                    }
                )
        else:
            research_capacity_source["status"] = "invalid"
            degraded_reasons.append("research_case_capacity_runtime_invalid")
    sources_consulted.append(research_capacity_source)

    ingestion_source: Dict[str, Any] = {
        "source_type": "production_knowledge_ingestion_latest",
        "path": rel_path_for(production_knowledge_ingestion_latest_path),
        "status": "missing",
        "record_present": False,
    }
    ingestion_latest_payload = load_json_if_exists(production_knowledge_ingestion_latest_path)
    if production_knowledge_ingestion_latest_path.exists() and production_knowledge_ingestion_latest_path.is_file():
        if isinstance(ingestion_latest_payload, dict):
            last_record = (
                ingestion_latest_payload.get("last_record")
                if isinstance(ingestion_latest_payload.get("last_record"), dict)
                else {}
            )
            ingestion_id = str(last_record.get("ingestion_id") or "").strip()
            destination_path = str(last_record.get("destination_path") or "").strip()
            classification_label = str(last_record.get("classification_label") or "").strip()
            ingestion_mode = str(last_record.get("ingestion_mode") or "").strip()
            updated_at = (
                str(ingestion_latest_payload.get("updated_at") or "").strip()
                or str(last_record.get("recorded_at") or "").strip()
            )
            updated_age_sec = age_sec(updated_at)
            ingestion_source.update(
                {
                    "record_present": bool(last_record),
                    "ingestion_id": ingestion_id or None,
                    "destination_path": destination_path or None,
                    "classification_label": classification_label or None,
                    "ingestion_mode": ingestion_mode or None,
                    "updated_age_sec": updated_age_sec,
                }
            )
            if not last_record:
                ingestion_source["status"] = "no_records"
            else:
                is_stale = updated_age_sec is not None and updated_age_sec > federated_doc_stale_after_sec
                ingestion_source["status"] = "stale" if is_stale else "ok"
                relevance = max(
                    match_task_relevance(
                        clean_task_id,
                        ingestion_id,
                        destination_path,
                        classification_label,
                        ingestion_mode,
                    ),
                    match_retrieval_relevance(
                        retrieval_terms,
                        ingestion_id,
                        destination_path,
                        classification_label,
                        ingestion_mode,
                    ),
                )
                matched_terms = matched_retrieval_terms(
                    retrieval_terms,
                    ingestion_id,
                    destination_path,
                    classification_label,
                    ingestion_mode,
                )
                if relevance > 0.0 or is_stale:
                    evidence_items.append(
                        {
                            "type": "document_ingestion_latest",
                            "source": rel_path_for(production_knowledge_ingestion_latest_path),
                            "timestamp": updated_at or now_iso(),
                            "summary": short_text(
                                " | ".join(
                                    [
                                        f"ingestion_id={ingestion_id or 'unknown'}",
                                        f"destination={destination_path or 'unknown'}",
                                        f"classification={classification_label or 'unknown'}",
                                        f"mode={ingestion_mode or 'unknown'}",
                                        f"age_sec={updated_age_sec if updated_age_sec is not None else 'unknown'}",
                                    ]
                                ),
                                limit=240,
                            ),
                            "ingestion_id": ingestion_id or None,
                            "destination_path": destination_path or None,
                            "classification_label": classification_label or None,
                            "ingestion_mode": ingestion_mode or None,
                            "relevance": clamp_score(max(0.45 if is_stale else 0.0, relevance)),
                            "matched_terms": matched_terms,
                        }
                    )
                if is_stale:
                    degraded_reasons.append("production_knowledge_ingestion_stale")
        else:
            ingestion_source["status"] = "invalid"
            degraded_reasons.append("production_knowledge_ingestion_latest_invalid")
    sources_consulted.append(ingestion_source)

    if ui_bundle_summary is not None:
        ui_source: Dict[str, Any] = {
            "source_type": "ui_evidence_bundle",
            "path": ui_bundle_summary.get("bundle_path"),
            "bundle_id": ui_bundle_summary.get("bundle_id"),
            "status": str(ui_bundle_summary.get("status") or "invalid"),
            "finding_count": int(ui_bundle_summary.get("finding_count") or 0),
            "critical_finding_count": int(ui_bundle_summary.get("critical_finding_count") or 0),
        }
        if ui_bundle_summary.get("error"):
            ui_source["error"] = ui_bundle_summary.get("error")
        if ui_source["status"] == "linked":
            bundle_path_raw = pathlib.Path(str(ui_bundle_arg or "").strip()).expanduser()
            bundle_path = bundle_path_raw if bundle_path_raw.is_absolute() else (root / bundle_path_raw)
            bundle_payload = load_json_if_exists(bundle_path.resolve())
            findings = bundle_payload.get("findings") if isinstance(bundle_payload, dict) and isinstance(bundle_payload.get("findings"), list) else []
            sorted_findings = sorted(
                [row for row in findings if isinstance(row, dict)],
                key=lambda row: (
                    severity_rank(row.get("severity")),
                    float(row.get("confidence") or 0.0),
                ),
                reverse=True,
            )
            for finding in sorted_findings[:max(1, max_items)]:
                severity = str(finding.get("severity") or "unknown").strip().lower() or "unknown"
                title = str(finding.get("title") or "").strip()
                rationale = str(finding.get("rationale") or "").strip()
                finding_id = str(finding.get("finding_id") or "").strip()
                matched_terms = matched_retrieval_terms(retrieval_terms, finding_id, title, rationale)
                query_relevance = match_retrieval_relevance(retrieval_terms, finding_id, title, rationale)
                evidence_items.append(
                    {
                        "type": "ui_finding",
                        "source": f"ui_evidence_bundle:{ui_source.get('bundle_id') or 'unknown'}",
                        "timestamp": (
                            (finding.get("provenance") or {}).get("evaluated_at")
                            if isinstance(finding.get("provenance"), dict)
                            else None
                        )
                        or ui_bundle_summary.get("captured_at"),
                        "finding_id": finding.get("finding_id"),
                        "severity": severity,
                        "title": title or None,
                        "summary": short_text(rationale or title, limit=220),
                        "relevance": clamp_score(max(0.35 + (0.15 * float(severity_rank(severity))), query_relevance)),
                        "matched_terms": matched_terms,
                    }
                )
        sources_consulted.append(ui_source)

    if isinstance(component_consistency_overlay, dict):
        overlay_source: Dict[str, Any] = {
            "source_type": "component_consistency_audit_overlay",
            "path": component_consistency_overlay.get("artifact_path"),
            "status": str(component_consistency_overlay.get("status") or "unknown"),
            "finding_count": int(component_consistency_overlay.get("finding_count") or 0),
            "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0),
        }
        if component_consistency_overlay.get("reason"):
            overlay_source["reason"] = component_consistency_overlay.get("reason")
        sources_consulted.append(overlay_source)

        overlay_findings = (
            component_consistency_overlay.get("findings")
            if isinstance(component_consistency_overlay.get("findings"), list)
            else []
        )
        for finding in overlay_findings[:max(1, max_items)]:
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity") or "low").strip().lower() or "low"
            finding_text = str(finding.get("finding_text") or "").strip()
            finding_id = str(finding.get("finding_id") or "").strip()
            matched_terms = matched_retrieval_terms(retrieval_terms, finding_id, finding_text, finding.get("audit_rule_id"))
            query_relevance = match_retrieval_relevance(retrieval_terms, finding_id, finding_text, finding.get("audit_rule_id"))
            evidence_items.append(
                {
                    "type": "ui_consistency_finding",
                    "source": "component_consistency_audit_overlay.v1",
                    "timestamp": finding.get("created_at") or component_consistency_overlay.get("generated_at"),
                    "finding_id": finding.get("finding_id"),
                    "severity": severity,
                    "audit_rule_id": finding.get("audit_rule_id"),
                    "summary": short_text(finding_text, limit=220),
                    "relevance": clamp_score(max(0.4 + (0.13 * float(severity_rank(severity))), query_relevance)),
                    "matched_terms": matched_terms,
                }
            )

    evidence_items = sorted(
        evidence_items,
        key=lambda row: (
            float(row.get("relevance") or 0.0),
            severity_rank(row.get("severity")),
            str(row.get("timestamp") or ""),
        ),
        reverse=True,
    )[:max_items]

    degraded_statuses = {"invalid", "missing", "fail_closed", "blocked", "stale"}
    healthy_statuses = {"ok", "linked"}
    degraded = any(str(src.get("status") or "").strip() in degraded_statuses for src in sources_consulted)

    source_health = {
        "healthy_source_count": 0,
        "degraded_source_count": 0,
        "neutral_source_count": 0,
    }
    for source in sources_consulted:
        status_value = str(source.get("status") or "").strip()
        if status_value in healthy_statuses:
            source_health["healthy_source_count"] += 1
        elif status_value in degraded_statuses:
            source_health["degraded_source_count"] += 1
            source_type = str(source.get("source_type") or "unknown_source")
            degraded_reasons.append(f"{source_type}:{status_value}")
        else:
            source_health["neutral_source_count"] += 1

    if evidence_items:
        status = "ready"
    elif degraded:
        status = "degraded"
    else:
        status = "empty"

    degraded_reasons = unique_preserve([str(reason or "").strip() for reason in degraded_reasons if str(reason or "").strip()])

    finished_at = dt.datetime.now(dt.timezone.utc)
    latency_ms = max(0, int((finished_at - started_at).total_seconds() * 1000))

    return {
        "schema": "clawd.federated_evidence.v1",
        "generated_at": now_iso(),
        "requested_task_id": clean_task_id,
        "status": status,
        "sources_consulted": sources_consulted,
        "evidence": evidence_items,
        "summary": {
            "source_count": len(sources_consulted),
            "evidence_count": len(evidence_items),
            "max_items": max_items,
            "latency_ms": latency_ms,
            "retrieval_query": retrieval_query,
            "retrieval_terms": retrieval_terms,
            "source_health": source_health,
            "degraded_reason_count": len(degraded_reasons),
            "degraded_reasons": degraded_reasons,
            "memory_consolidation_runtime_latest_path": rel_path_for(memory_consolidation_runtime_latest_path),
            "research_case_registry_path": rel_path_for(research_case_registry_path),
            "research_case_capacity_runtime_path": rel_path_for(research_case_capacity_runtime_path),
            "production_knowledge_ingestion_latest_path": rel_path_for(production_knowledge_ingestion_latest_path),
        },
    }


def build_b7_b8_packet_handshake(
    *,
    ui_bundle_summary: Optional[Dict[str, Any]],
    component_consistency_overlay: Optional[Dict[str, Any]],
    critique_request: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(component_consistency_overlay, dict):
        return None

    overlay_findings = (
        component_consistency_overlay.get("findings")
        if isinstance(component_consistency_overlay.get("findings"), list)
        else []
    )
    b7_finding_ids = unique_preserve(
        [
            str(row.get("finding_id") or "").strip()
            for row in overlay_findings
            if isinstance(row, dict) and str(row.get("finding_id") or "").strip()
        ]
    )

    critique_obj = critique_request if isinstance(critique_request, dict) else {}
    critique_status = str(critique_obj.get("status") or "not_requested").strip() or "not_requested"
    critique_packet = critique_obj.get("packet") if isinstance(critique_obj.get("packet"), dict) else {}
    bridge_obj = critique_obj.get("b7_consumption_bridge") if isinstance(critique_obj.get("b7_consumption_bridge"), dict) else {}
    if not bridge_obj and isinstance(critique_packet.get("b7_consumption_bridge"), dict):
        bridge_obj = critique_packet.get("b7_consumption_bridge") or {}

    b7_finding_id_set = set(b7_finding_ids)

    bridge_promoted_ids_raw = bridge_obj.get("promoted_finding_ids") if isinstance(bridge_obj.get("promoted_finding_ids"), list) else None
    bridge_ignored_ids_raw = bridge_obj.get("ignored_finding_ids") if isinstance(bridge_obj.get("ignored_finding_ids"), list) else None

    critique_findings = (
        critique_packet.get("findings") if isinstance(critique_packet.get("findings"), list) else []
    )
    if isinstance(bridge_promoted_ids_raw, list):
        promoted_finding_ids = unique_preserve(
            [
                str(finding_id or "").strip()
                for finding_id in bridge_promoted_ids_raw
                if str(finding_id or "").strip() in b7_finding_id_set
            ]
        )
    else:
        promoted_finding_ids = unique_preserve(
            [
                str(row.get("overlay_finding_id") or "").strip()
                for row in critique_findings
                if isinstance(row, dict) and str(row.get("overlay_finding_id") or "").strip()
            ]
        )
        promoted_finding_ids = [finding_id for finding_id in promoted_finding_ids if finding_id in b7_finding_id_set]

    promoted_finding_id_set = set(promoted_finding_ids)
    if isinstance(bridge_ignored_ids_raw, list):
        ignored_finding_ids = unique_preserve(
            [
                str(finding_id or "").strip()
                for finding_id in bridge_ignored_ids_raw
                if str(finding_id or "").strip() in b7_finding_id_set
                and str(finding_id or "").strip() not in promoted_finding_id_set
            ]
        )
    else:
        ignored_finding_ids = [finding_id for finding_id in b7_finding_ids if finding_id not in promoted_finding_id_set]

    ignored_reason_counts = (
        bridge_obj.get("ignored_reason_counts")
        if isinstance(bridge_obj.get("ignored_reason_counts"), dict)
        else {}
    )
    ignored_reason_by_id: Dict[str, str] = {}
    ignored_findings = bridge_obj.get("ignored_findings") if isinstance(bridge_obj.get("ignored_findings"), list) else []
    for row in ignored_findings:
        if not isinstance(row, dict):
            continue
        finding_id = str(row.get("overlay_finding_id") or "").strip()
        ignore_reason = str(row.get("ignore_reason") or "").strip()
        if finding_id and ignore_reason and finding_id not in ignored_reason_by_id:
            ignored_reason_by_id[finding_id] = ignore_reason

    candidate_opportunity_surface = build_candidate_opportunity_surface(
        overlay_findings=[row for row in overlay_findings if isinstance(row, dict)],
        promoted_finding_ids=promoted_finding_ids,
        ignored_finding_ids=ignored_finding_ids,
        ignored_reason_by_id=ignored_reason_by_id,
        critique_status=critique_status,
        visibility_state=bridge_obj.get("visibility_state") if isinstance(bridge_obj, dict) else "unknown",
        task_suppressed=bool(bridge_obj.get("task_suppressed") is True) if isinstance(bridge_obj, dict) else False,
        task_noise_signal_tags=bridge_obj.get("task_noise_signal_tags")
        if isinstance(bridge_obj.get("task_noise_signal_tags"), list)
        else [],
    )

    if not b7_finding_ids and critique_status == "not_requested":
        return None

    bundle_status = str((ui_bundle_summary or {}).get("status") or "unlinked").strip() or "unlinked"
    bundle_linked = bundle_status == "linked"
    bundle_finding_count = int((ui_bundle_summary or {}).get("finding_count") or 0) if bundle_linked else 0

    if not b7_finding_ids:
        handshake_status = "clean"
    elif promoted_finding_ids:
        handshake_status = "promoted"
    elif critique_status in {"generated", "healthy"} and ignored_finding_ids:
        handshake_status = "consumed_ignored"
    elif critique_status in {"generated", "healthy"}:
        handshake_status = "consumed_without_promotion"
    elif critique_status in {"cooldown", "task_not_active", "task_too_young", "age_unknown"}:
        handshake_status = "gated"
    else:
        handshake_status = "enriched_waiting_b8"

    return {
        "schema": "clawd.b7_b8_packet_handshake.v1",
        "generated_at": now_iso(),
        "status": handshake_status,
        "contract_version": "clawd.b7_b8_packet_handshake.v1",
        "packet": {
            "bundle_status": bundle_status,
            "bundle_id": (ui_bundle_summary or {}).get("bundle_id") if bundle_linked else None,
            "bundle_path": (ui_bundle_summary or {}).get("bundle_path") if bundle_linked else None,
            "bundle_schema_version": (ui_bundle_summary or {}).get("schema_version") if bundle_linked else None,
            "finding_count_before_enrichment": bundle_finding_count,
            "finding_count_after_enrichment": bundle_finding_count + len(b7_finding_ids),
            "append_only_projection": True,
            "source_scope": component_consistency_overlay.get("source_scope") or "ui_evidence_bundle",
        },
        "b7_enrichment": {
            "producer_id": "component_consistency_audit_overlay.v1",
            "status": component_consistency_overlay.get("status") or "unknown",
            "finding_count": len(b7_finding_ids),
            "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0),
            "finding_ids": b7_finding_ids,
            "quiet_on_success": len(b7_finding_ids) == 0,
        },
        "b8_consumption": {
            "consumer_id": "operator_task_state_critique_packet.v1",
            "requested_task_id": critique_obj.get("requested_task_id"),
            "critique_status": critique_status,
            "visibility_state": bridge_obj.get("visibility_state"),
            "visibility_score": bridge_obj.get("visibility_score"),
            "task_noise_signal_tags": bridge_obj.get("task_noise_signal_tags")
            if isinstance(bridge_obj.get("task_noise_signal_tags"), list)
            else [],
            "task_suppressed": bool(bridge_obj.get("task_suppressed") is True),
            "task_suppression_reasons": bridge_obj.get("task_suppression_reasons")
            if isinstance(bridge_obj.get("task_suppression_reasons"), list)
            else [],
            "consumption_policy": bridge_obj.get("policy") if isinstance(bridge_obj.get("policy"), dict) else None,
            "promotion_count": len(promoted_finding_ids),
            "ignored_count": len(ignored_finding_ids),
            "promoted_finding_ids": promoted_finding_ids,
            "ignored_finding_ids": ignored_finding_ids,
            "ignored_reason_counts": ignored_reason_counts,
        },
        "candidate_opportunity_surface": candidate_opportunity_surface,
        "provenance": {
            "overlay_artifact_path": component_consistency_overlay.get("artifact_path"),
            "critique_packet_path": critique_obj.get("packet_path"),
        },
    }


def load_mission_payload() -> Dict[str, Any]:
    if not mission_script.exists():
        raise SystemExit(f"operator_triage_console_missing_mission_script:{mission_script}")

    cmd = ["bash", str(mission_script)]
    if refresh:
        cmd.append("--refresh")
    cmd.append("--json")

    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=100)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("operator_triage_console_mission_control_timeout:100") from exc
    if cp.returncode != 0:
        err = (cp.stderr or cp.stdout or "mission_control_failed").strip()
        raise SystemExit(f"operator_triage_console_mission_control_failed:{err[:320]}")

    try:
        payload = json.loads(cp.stdout or "{}")
    except Exception as exc:
        raise SystemExit(f"operator_triage_console_mission_control_invalid_json:{exc.__class__.__name__}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("operator_triage_console_mission_control_not_object")
    return payload


mission_payload = load_mission_payload()
headline = mission_payload.get("headline") if isinstance(mission_payload.get("headline"), dict) else {}
actions_raw = mission_payload.get("actions") if isinstance(mission_payload.get("actions"), list) else []
freshness = mission_payload.get("freshness") if isinstance(mission_payload.get("freshness"), dict) else {}
execution_status = mission_payload.get("execution_status") if isinstance(mission_payload.get("execution_status"), dict) else {}
execution_frontier = mission_payload.get("execution_frontier") if isinstance(mission_payload.get("execution_frontier"), dict) else {}
execution_dispatch_intent = (
    mission_payload.get("execution_supervisor_dispatch_intent")
    if isinstance(mission_payload.get("execution_supervisor_dispatch_intent"), dict)
    else {}
)
execution_dispatch_qualification = (
    mission_payload.get("execution_supervisor_dispatch_qualification")
    if isinstance(mission_payload.get("execution_supervisor_dispatch_qualification"), dict)
    else {}
)
meaningful_event_reporting = (
    mission_payload.get("meaningful_event_reporting")
    if isinstance(mission_payload.get("meaningful_event_reporting"), dict)
    else {}
)
meaningful_event_contract = (
    mission_payload.get("meaningful_event_reporting_contract")
    if isinstance(mission_payload.get("meaningful_event_reporting_contract"), dict)
    else {}
)
routing_preflight = mission_payload.get("routing_preflight") if isinstance(mission_payload.get("routing_preflight"), dict) else {}
load_shedding = mission_payload.get("load_shedding") if isinstance(mission_payload.get("load_shedding"), dict) else {}
queue_stale_wave_signal = (
    mission_payload.get("queue_stale_wave_signal")
    if isinstance(mission_payload.get("queue_stale_wave_signal"), dict)
    else {}
)
web_domain_guard = mission_payload.get("web_domain_guard") if isinstance(mission_payload.get("web_domain_guard"), dict) else {}
model_rollout_remediation = (
    mission_payload.get("model_rollout_operator_mistake_remediation")
    if isinstance(mission_payload.get("model_rollout_operator_mistake_remediation"), dict)
    else {}
)
source_of_truth_map_guard_summary = summarize_source_of_truth_map_guard(source_of_truth_map_guard_latest_path)
ui_evidence_bundle_summary = summarize_ui_evidence_bundle(ui_evidence_bundle_arg)
component_consistency_overlay: Optional[Dict[str, Any]] = None
if ui_evidence_bundle_summary and ui_evidence_bundle_summary.get("status") == "linked":
    resolved_bundle_path = resolve_workspace_path(ui_evidence_bundle_arg)
    bundle_payload = load_json_if_exists(resolved_bundle_path.resolve()) if isinstance(resolved_bundle_path, pathlib.Path) else None
    if not isinstance(bundle_payload, dict):
        component_consistency_overlay = {
            "schema": "clawd.component_consistency_audit_overlay.v1",
            "overlay_id": "component_consistency_audit_overlay.v1",
            "generated_at": now_iso(),
            "status": "invalid",
            "bundle_id": ui_evidence_bundle_summary.get("bundle_id"),
            "bundle_path": ui_evidence_bundle_summary.get("bundle_path"),
            "reason": "bundle_unreadable",
            "finding_count": 0,
            "critical_finding_count": 0,
            "findings": [],
        }
    else:
        extracted_ui_state = extract_ui_state_from_bundle(bundle_payload, resolved_bundle_path.resolve())
        if extracted_ui_state.get("status") != "ready" or not isinstance(extracted_ui_state.get("ui_state"), dict):
            component_consistency_overlay = {
                "schema": "clawd.component_consistency_audit_overlay.v1",
                "overlay_id": "component_consistency_audit_overlay.v1",
                "generated_at": now_iso(),
                "status": "skipped",
                "bundle_id": ui_evidence_bundle_summary.get("bundle_id"),
                "bundle_path": ui_evidence_bundle_summary.get("bundle_path"),
                "reason": extracted_ui_state.get("reason") or extracted_ui_state.get("status") or "ui_state_unavailable",
                "ui_state_source": extracted_ui_state.get("source"),
                "ui_state_artifact_path": extracted_ui_state.get("artifact_path"),
                "finding_count": 0,
                "critical_finding_count": 0,
                "findings": [],
            }
        else:
            component_consistency_overlay = audit_component_consistency_overlay(
                ui_state=extracted_ui_state.get("ui_state") or {},
                bundle_id=str(ui_evidence_bundle_summary.get("bundle_id") or "").strip() or None,
                bundle_path=str(ui_evidence_bundle_summary.get("bundle_path") or "").strip() or None,
                ui_state_source=str(extracted_ui_state.get("source") or "").strip() or None,
                ui_state_artifact_path=str(extracted_ui_state.get("artifact_path") or "").strip() or None,
            )

if ui_evidence_bundle_summary is not None and component_consistency_overlay is not None:
    ui_evidence_bundle_summary["component_consistency_audit_overlay"] = {
        "status": component_consistency_overlay.get("status"),
        "finding_count": int(component_consistency_overlay.get("finding_count") or 0),
        "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0),
        "reason": component_consistency_overlay.get("reason"),
    }

priority_rank = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}

actions: List[Dict[str, Any]] = []
for row in actions_raw:
    if not isinstance(row, dict):
        continue
    action_name = str(row.get("action") or "").strip()
    command = str(row.get("command") or "").strip()
    if not action_name or not command:
        continue
    priority = str(row.get("priority") or "p2").strip().lower()
    actions.append(
        {
            "action": action_name,
            "command": command,
            "priority": priority if priority in priority_rank else "p2",
            "reason": str(row.get("reason") or "").strip() or None,
        }
    )

actions.sort(key=lambda item: (priority_rank.get(item["priority"], 9), item["action"]))


def find_action(*tokens: str) -> Optional[Dict[str, Any]]:
    lookup = [str(token or "").strip().lower() for token in tokens if str(token or "").strip()]
    if not lookup:
        return actions[0] if actions else None
    for action in actions:
        action_name = str(action.get("action") or "").lower()
        command = str(action.get("command") or "").lower()
        haystack = f"{action_name} {command}"
        if any(token in haystack for token in lookup):
            return action
    return actions[0] if actions else None


def action_brief(action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    command = str(action.get("command") or "").strip()
    action_name = str(action.get("action") or "").strip()
    if not command and not action_name:
        return None
    priority = str(action.get("priority") or "").strip().lower()
    return {
        "action": action_name or None,
        "priority": priority if priority in {"p0", "p1", "p2", "p3"} else None,
        "command": command or None,
    }


def action_brief_or_default(
    action: Optional[Dict[str, Any]],
    *,
    default_action: str,
    default_command: str,
    default_priority: str = "p2",
) -> Dict[str, Any]:
    brief = action_brief(action)
    if isinstance(brief, dict) and str(brief.get("command") or "").strip():
        return brief

    priority = str(default_priority or "p2").strip().lower()
    if priority not in {"p0", "p1", "p2", "p3"}:
        priority = "p2"
    return {
        "action": str(default_action or "safe_fallback").strip() or "safe_fallback",
        "priority": priority,
        "command": str(default_command or "").strip() or None,
    }


issues: List[Dict[str, Any]] = []
_seen_issue_ids = set()


def add_issue(
    *,
    issue_id: str,
    severity: str,
    lane: str,
    reason: str,
    evidence_ref: Optional[str] = None,
    action_hint: Optional[Dict[str, Any]] = None,
) -> None:
    normalized_id = str(issue_id or "").strip()
    if not normalized_id or normalized_id in _seen_issue_ids:
        return
    _seen_issue_ids.add(normalized_id)
    row: Dict[str, Any] = {
        "id": normalized_id,
        "severity": severity if severity in {"p0", "p1", "p2"} else "p2",
        "lane": lane,
        "reason": str(reason or "").strip() or "unspecified",
        "evidence_ref": evidence_ref,
        "recommended_action": None,
    }
    if isinstance(action_hint, dict):
        row["recommended_action"] = {
            "action": action_hint.get("action"),
            "priority": action_hint.get("priority"),
            "command": action_hint.get("command"),
        }
    issues.append(row)


readiness = str(headline.get("readiness") or "UNKNOWN")
mutation_gate = str(headline.get("mutation_gate") or "unknown")
hard_blockers = int(headline.get("hard_blockers") or 0)
warnings = int(headline.get("warnings") or 0)

if mutation_gate == "forbidden" or hard_blockers > 0:
    overall_severity = "p0"
    overall_label = "BLOCKER"
elif warnings > 0:
    overall_severity = "p1"
    overall_label = "DEGRADED"
else:
    overall_severity = "p2"
    overall_label = "STABLE"

execution_task_ids = collect_id_list(
    execution_status.get("current_focus"),
    execution_status.get("target_step_id"),
    execution_status.get("launched_step_id"),
    execution_frontier.get("next_candidate"),
    execution_dispatch_intent.get("ready_candidate_task_ids"),
    execution_dispatch_intent.get("blocked_candidate_task_ids"),
    execution_dispatch_qualification.get("qualified_candidate_task_ids"),
    execution_dispatch_qualification.get("blocked_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_intent_ready_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_intent_blocked_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_qualification_qualified_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_qualification_blocked_candidate_task_ids"),
)

foreground_task_ids = collect_id_list(
    execution_status.get("current_focus"),
    execution_status.get("target_step_id"),
    execution_status.get("launched_step_id"),
)

ready_candidate_task_ids = collect_id_list(
    execution_dispatch_intent.get("ready_candidate_task_ids"),
    execution_dispatch_qualification.get("qualified_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_intent_ready_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_qualification_qualified_candidate_task_ids"),
)

blocked_candidate_task_ids = collect_id_list(
    execution_dispatch_intent.get("blocked_candidate_task_ids"),
    execution_dispatch_qualification.get("blocked_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_intent_blocked_candidate_task_ids"),
    execution_status.get("supervisor_dispatch_qualification_blocked_candidate_task_ids"),
)

background_ready_task_ids = [task_id for task_id in ready_candidate_task_ids if task_id not in foreground_task_ids]
background_blocked_task_ids = [task_id for task_id in blocked_candidate_task_ids if task_id not in foreground_task_ids]

task_cards_by_id: Dict[str, Dict[str, Any]] = {}


def add_task_card(task_id: Any, state: str, *, foreground: bool = False, blocked: bool = False) -> None:
    task_key = str(task_id or "").strip()
    if not task_key:
        return
    row = task_cards_by_id.setdefault(
        task_key,
        {
            "task_id": task_key,
            "states": [],
            "foreground": False,
            "blocked": False,
            "critique_available": False,
        },
    )
    states = row.get("states") if isinstance(row.get("states"), list) else []
    if state not in states:
        states.append(state)
    row["states"] = states
    if foreground:
        row["foreground"] = True
    if blocked:
        row["blocked"] = True


for task_id in foreground_task_ids:
    add_task_card(task_id, "foreground", foreground=True, blocked=task_id in blocked_candidate_task_ids)

frontier_next_candidate = first_nonempty(
    execution_frontier.get("next_candidate"),
    execution_status.get("frontier_next_candidate"),
)
if frontier_next_candidate:
    add_task_card(
        frontier_next_candidate,
        "frontier_next_candidate",
        foreground=frontier_next_candidate in foreground_task_ids,
        blocked=frontier_next_candidate in blocked_candidate_task_ids,
    )

for task_id in ready_candidate_task_ids:
    add_task_card(task_id, "ready_candidate", foreground=task_id in foreground_task_ids)

for task_id in blocked_candidate_task_ids:
    add_task_card(task_id, "blocked_candidate", foreground=task_id in foreground_task_ids, blocked=True)

task_cards: List[Dict[str, Any]] = []
for card in task_cards_by_id.values():
    states = unique_preserve([str(state or "").strip() for state in (card.get("states") or []) if str(state or "").strip()])
    task_cards.append(
        {
            "task_id": str(card.get("task_id") or "").strip(),
            "state": "|".join(states) if states else "unknown",
            "foreground": bool(card.get("foreground") is True),
            "blocked": bool(card.get("blocked") is True),
            "critique_available": bool(card.get("critique_available") is True),
        }
    )

task_cards.sort(
    key=lambda row: (
        0 if bool(row.get("foreground") is True) else 1,
        0 if bool(row.get("blocked") is True) else 1,
        str(row.get("task_id") or ""),
    )
)

mission_control_age_sec = age_sec(mission_payload.get("generated_at"))
execution_last_signal_age_sec = optional_nonnegative_int(execution_status.get("last_signal_age_sec"))
if execution_last_signal_age_sec is None:
    execution_last_signal_age_sec = age_sec(execution_status.get("last_signal_at"))
execution_last_progress_age_sec = optional_nonnegative_int(execution_status.get("last_progress_age_sec"))
if execution_last_progress_age_sec is None:
    execution_last_progress_age_sec = age_sec(execution_status.get("last_progress_at"))
task_freshness = classify_freshness(
    execution_last_signal_age_sec,
    execution_last_progress_age_sec,
    stale_after_sec=task_freshness_stale_after_sec,
    aging_after_sec=task_freshness_aging_after_sec,
)

demoted_worker_count = max(
    safe_int(execution_dispatch_intent.get("launch_readiness_demoted_worker_count"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_demoted_worker_count"), -1),
    safe_int(execution_status.get("supervisor_launch_readiness_demoted_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_demoted_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_demoted_worker_count"), -1),
)
if demoted_worker_count < 0:
    demoted_worker_count = 0

restore_pending_worker_count = max(
    safe_int(execution_dispatch_intent.get("launch_readiness_restore_pending_worker_count"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_restore_pending_worker_count"), -1),
    safe_int(execution_status.get("supervisor_launch_readiness_restore_pending_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_restore_pending_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_restore_pending_worker_count"), -1),
)
if restore_pending_worker_count < 0:
    restore_pending_worker_count = 0

restored_worker_count = max(
    safe_int(execution_dispatch_intent.get("launch_readiness_restored_worker_count"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_restored_worker_count"), -1),
    safe_int(execution_status.get("supervisor_launch_readiness_restored_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_restored_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_restored_worker_count"), -1),
)
if restored_worker_count < 0:
    restored_worker_count = 0

probe_due_now_worker_count = max(
    safe_int(execution_dispatch_intent.get("launch_readiness_probe_execution_due_now_worker_count"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_probe_execution_due_now_worker_count"), -1),
    safe_int(execution_status.get("supervisor_launch_readiness_probe_execution_due_now_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_probe_execution_due_now_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_probe_execution_due_now_worker_count"), -1),
)
if probe_due_now_worker_count < 0:
    probe_due_now_worker_count = 0

probe_overdue_worker_count = max(
    safe_int(execution_dispatch_intent.get("launch_readiness_probe_execution_overdue_worker_count"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_probe_execution_overdue_worker_count"), -1),
    safe_int(execution_status.get("supervisor_launch_readiness_probe_execution_overdue_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_probe_execution_overdue_worker_count"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_probe_execution_overdue_worker_count"), -1),
)
if probe_overdue_worker_count < 0:
    probe_overdue_worker_count = 0

reason_candidates = collect_id_list(
    execution_frontier.get("blocked_reason"),
    execution_frontier.get("stalled_reason"),
    execution_frontier.get("transition_reason"),
    execution_status.get("skip_reason"),
    execution_status.get("autonomous_dispatch_block_reason"),
    execution_status.get("autonomous_dispatch_skip_reason"),
    execution_status.get("supervisor_dispatch_intent_launch_readiness_reason"),
    execution_status.get("supervisor_dispatch_qualification_launch_readiness_reason"),
    execution_dispatch_intent.get("launch_readiness_reason"),
    execution_dispatch_qualification.get("launch_readiness_reason"),
    execution_dispatch_intent.get("decision_reasons"),
    execution_dispatch_qualification.get("decision_reasons"),
)
ignore_reason_tokens = {
    "",
    "none",
    "null",
    "n/a",
    "unknown",
    "idle",
    "clear",
    "ok",
}
blockage_reasons = [
    reason
    for reason in reason_candidates
    if str(reason or "").strip().lower() not in ignore_reason_tokens
]

oldest_demoted_age_sec = max(
    safe_int(execution_status.get("supervisor_launch_readiness_oldest_demoted_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_oldest_demoted_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_oldest_demoted_age_sec"), -1),
    safe_int(execution_dispatch_intent.get("launch_readiness_oldest_demoted_age_sec"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_oldest_demoted_age_sec"), -1),
)
if oldest_demoted_age_sec < 0:
    oldest_demoted_age_sec = None

oldest_restore_pending_age_sec = max(
    safe_int(execution_status.get("supervisor_launch_readiness_oldest_restore_pending_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_oldest_restore_pending_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_oldest_restore_pending_age_sec"), -1),
    safe_int(execution_dispatch_intent.get("launch_readiness_oldest_restore_pending_age_sec"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_oldest_restore_pending_age_sec"), -1),
)
if oldest_restore_pending_age_sec < 0:
    oldest_restore_pending_age_sec = None

oldest_probe_due_now_age_sec = max(
    safe_int(execution_status.get("supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_due_now_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_due_now_age_sec"), -1),
    safe_int(execution_dispatch_intent.get("launch_readiness_probe_execution_oldest_due_now_age_sec"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_probe_execution_oldest_due_now_age_sec"), -1),
)
if oldest_probe_due_now_age_sec < 0:
    oldest_probe_due_now_age_sec = None

oldest_probe_overdue_age_sec = max(
    safe_int(execution_status.get("supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_overdue_age_sec"), -1),
    safe_int(execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_overdue_age_sec"), -1),
    safe_int(execution_dispatch_intent.get("launch_readiness_probe_execution_oldest_overdue_age_sec"), -1),
    safe_int(execution_dispatch_qualification.get("launch_readiness_probe_execution_oldest_overdue_age_sec"), -1),
)
if oldest_probe_overdue_age_sec < 0:
    oldest_probe_overdue_age_sec = None

dispatch_intent_age_sec = age_sec(execution_dispatch_intent.get("generated_at"))
dispatch_qualification_age_sec = age_sec(execution_dispatch_qualification.get("generated_at"))
worker_freshness = classify_freshness(
    dispatch_intent_age_sec,
    dispatch_qualification_age_sec,
    execution_last_signal_age_sec,
)

fail_closed_worker = bool(
    execution_dispatch_intent.get("fail_closed")
    if execution_dispatch_intent.get("fail_closed") is not None
    else execution_dispatch_qualification.get("fail_closed")
    if execution_dispatch_qualification.get("fail_closed") is not None
    else execution_status.get("supervisor_dispatch_intent_fail_closed")
    if execution_status.get("supervisor_dispatch_intent_fail_closed") is not None
    else execution_status.get("supervisor_dispatch_qualification_fail_closed")
    if execution_status.get("supervisor_dispatch_qualification_fail_closed") is not None
    else False
)

launch_mutation_allowed_worker = bool(
    execution_dispatch_intent.get("launch_mutation_allowed")
    if execution_dispatch_intent.get("launch_mutation_allowed") is not None
    else execution_dispatch_qualification.get("launch_mutation_allowed")
    if execution_dispatch_qualification.get("launch_mutation_allowed") is not None
    else execution_status.get("supervisor_dispatch_intent_launch_mutation_allowed")
    if execution_status.get("supervisor_dispatch_intent_launch_mutation_allowed") is not None
    else execution_status.get("supervisor_dispatch_qualification_launch_mutation_allowed")
    if execution_status.get("supervisor_dispatch_qualification_launch_mutation_allowed") is not None
    else False
)

worker_state_status = "ready"
if probe_overdue_worker_count > 0 or fail_closed_worker:
    worker_state_status = "blocked"
elif demoted_worker_count > 0 or restore_pending_worker_count > 0 or probe_due_now_worker_count > 0:
    worker_state_status = "degraded"

evidence_quality_score = 52.0
if ui_evidence_bundle_summary and ui_evidence_bundle_summary.get("status") == "linked":
    evidence_quality_score = 88.0
    if int(ui_evidence_bundle_summary.get("critical_finding_count") or 0) > 0:
        evidence_quality_score = 58.0

worker_state_score_input = {
    "demoted_worker_count": demoted_worker_count,
    "restore_pending_worker_count": restore_pending_worker_count,
    "probe_due_now_worker_count": probe_due_now_worker_count,
    "probe_overdue_worker_count": probe_overdue_worker_count,
    "fail_closed": fail_closed_worker,
}
task_freshness_dimension_score = score_freshness_dimension(
    str(task_freshness.get("status") or "unknown"),
    optional_nonnegative_int(task_freshness.get("worst_age_sec")),
)
worker_health_dimension_score = score_worker_health_dimension(worker_state_status, worker_state_score_input)
execution_stability_dimension_score = score_execution_stability_dimension(worker_state_score_input)

blocked_background_ids = [
    task_id
    for task_id in background_blocked_task_ids
    if task_id and task_id not in foreground_task_ids
]
primary_visible_blocked_task_id = blocked_background_ids[0] if blocked_background_ids else None

all_task_cards: List[Dict[str, Any]] = []
visible_task_cards: List[Dict[str, Any]] = []
suppressed_task_cards: List[Dict[str, Any]] = []

for card in task_cards:
    row = dict(card)
    task_id = str(row.get("task_id") or "").strip()
    state_txt = str(row.get("state") or "").strip()
    is_foreground = bool(row.get("foreground") is True)
    is_blocked = bool(row.get("blocked") is True)
    is_frontier_task = bool(frontier_next_candidate and task_id == str(frontier_next_candidate))

    blockage_severity_score = 100.0
    if is_blocked:
        blockage_severity_score = 28.0
    elif is_frontier_task and bool(execution_frontier.get("stalled") is True):
        blockage_severity_score = 42.0
    elif int(len(blocked_candidate_task_ids)) > 0:
        blockage_severity_score = 76.0

    visibility_score = clamp_percent(
        (task_freshness_dimension_score * 0.30)
        + (worker_health_dimension_score * 0.25)
        + (blockage_severity_score * 0.20)
        + (evidence_quality_score * 0.15)
        + (execution_stability_dimension_score * 0.10)
    )
    visibility_state = visibility_rating(visibility_score)

    noise_signal_tags: List[str] = []
    suppression_reasons: List[str] = []

    if str(task_freshness.get("status") or "") in {"aging", "stale"} and not is_foreground and not is_blocked:
        noise_signal_tags.append("staleness")
        if not show_stale:
            suppression_reasons.append("staleness")

    if (
        is_blocked
        and not is_foreground
        and primary_visible_blocked_task_id
        and task_id != primary_visible_blocked_task_id
        and len(blocked_background_ids) > 1
    ):
        noise_signal_tags.append("redundancy")
        if not show_redundant:
            suppression_reasons.append("redundancy")

    if (not is_foreground) and ("ready_candidate" in state_txt) and (not is_blocked):
        noise_signal_tags.append("low_impact_state_change")
        if not verbose_state:
            suppression_reasons.append("low_impact_state_change")

    if (not is_foreground) and ("ready_candidate" in state_txt) and (not is_blocked):
        noise_signal_tags.append("chatter")
        if not show_chatter:
            suppression_reasons.append("chatter")

    if (
        (not is_blocked)
        and worker_state_status == "degraded"
        and probe_due_now_worker_count > 0
        and probe_overdue_worker_count == 0
        and not fail_closed_worker
    ):
        noise_signal_tags.append("recoverable_error")
        if not show_recoverable_errors:
            suppression_reasons.append("recoverable_error")

    is_suppressed = len(suppression_reasons) > 0 and not is_foreground and visibility_state != "critical"

    row["visibility_score"] = visibility_score
    row["visibility_rating"] = visibility_state
    row["visibility_scorecard"] = {
        "dimensions": {
            "task_freshness": task_freshness_dimension_score,
            "worker_health": worker_health_dimension_score,
            "blockage_severity": clamp_percent(blockage_severity_score),
            "evidence_quality": clamp_percent(evidence_quality_score),
            "execution_path_stability": execution_stability_dimension_score,
        },
        "weights": {
            "task_freshness": 0.30,
            "worker_health": 0.25,
            "blockage_severity": 0.20,
            "evidence_quality": 0.15,
            "execution_path_stability": 0.10,
        },
        "weighted_score": visibility_score,
        "rating": visibility_state,
    }
    row["noise_signal_tags"] = unique_preserve(noise_signal_tags)
    row["suppressed"] = bool(is_suppressed)
    row["suppression_reasons"] = unique_preserve(suppression_reasons)

    all_task_cards.append(row)
    if is_suppressed:
        suppressed_task_cards.append(
            {
                "task_id": task_id,
                "state": state_txt,
                "suppression_reasons": unique_preserve(suppression_reasons),
                "visibility_score": visibility_score,
                "visibility_rating": visibility_state,
            }
        )
    else:
        visible_task_cards.append(row)

if not visible_task_cards and all_task_cards:
    fallback_card = all_task_cards[0]
    fallback_card["suppressed"] = False
    fallback_card["suppression_reasons"] = []
    visible_task_cards = [fallback_card]
    suppressed_task_cards = [
        row for row in suppressed_task_cards if str(row.get("task_id") or "") != str(fallback_card.get("task_id") or "")
    ]

scorecard_dimension_averages: Dict[str, float] = {
    "task_freshness": task_freshness_dimension_score,
    "worker_health": worker_health_dimension_score,
    "blockage_severity": 100.0 if int(len(blocked_candidate_task_ids)) == 0 else 54.0,
    "evidence_quality": clamp_percent(evidence_quality_score),
    "execution_path_stability": execution_stability_dimension_score,
}
if all_task_cards:
    dimension_accumulators = {
        "task_freshness": [],
        "worker_health": [],
        "blockage_severity": [],
        "evidence_quality": [],
        "execution_path_stability": [],
    }
    for row in all_task_cards:
        dims = (row.get("visibility_scorecard") or {}).get("dimensions") if isinstance(row.get("visibility_scorecard"), dict) else {}
        for key in list(dimension_accumulators.keys()):
            value = dims.get(key) if isinstance(dims, dict) else None
            if isinstance(value, (int, float)):
                dimension_accumulators[key].append(float(value))
    for key, values in dimension_accumulators.items():
        if values:
            scorecard_dimension_averages[key] = clamp_percent(sum(values) / float(len(values)))

system_visibility_score = clamp_percent(
    (scorecard_dimension_averages["task_freshness"] * 0.30)
    + (scorecard_dimension_averages["worker_health"] * 0.25)
    + (scorecard_dimension_averages["blockage_severity"] * 0.20)
    + (scorecard_dimension_averages["evidence_quality"] * 0.15)
    + (scorecard_dimension_averages["execution_path_stability"] * 0.10)
)
system_visibility_rating = visibility_rating(system_visibility_score)

visibility_rating_counts = {"healthy": 0, "degraded": 0, "critical": 0}
for row in all_task_cards:
    rating = str(row.get("visibility_rating") or "").strip().lower()
    if rating in visibility_rating_counts:
        visibility_rating_counts[rating] += 1

critique_eligible_task_ids = set(blocked_candidate_task_ids)
if task_freshness.get("status") in {"aging", "stale"}:
    critique_eligible_task_ids.update(foreground_task_ids)
if bool(execution_frontier.get("stalled") is True):
    stalled_task = str(frontier_next_candidate or "").strip()
    if stalled_task:
        critique_eligible_task_ids.add(stalled_task)
if probe_overdue_worker_count > 0 or bool(
    execution_dispatch_intent.get("fail_closed") is True
    or execution_dispatch_qualification.get("fail_closed") is True
):
    critique_eligible_task_ids.update(foreground_task_ids)

for card in all_task_cards:
    task_id = str(card.get("task_id") or "").strip()
    if task_id and task_id in critique_eligible_task_ids:
        card["critique_available"] = True

execution_snapshot = {
    "active_task_count": len(execution_task_ids),
    "active_task_ids": execution_task_ids,
    "reported_running_tasks": safe_int(execution_status.get("running_tasks"), 0),
    "current_focus": str(execution_status.get("current_focus") or "").strip() or None,
    "frontier_next_candidate": str(frontier_next_candidate or "").strip() or None,
    "target_step_id": str(execution_status.get("target_step_id") or "").strip() or None,
    "launched_step_id": str(execution_status.get("launched_step_id") or "").strip() or None,
    "task_detail": {
        "foreground_task_ids": foreground_task_ids,
        "background_ready_task_ids": background_ready_task_ids,
        "background_blocked_task_ids": background_blocked_task_ids,
        "ready_candidate_task_ids": ready_candidate_task_ids,
        "blocked_candidate_task_ids": blocked_candidate_task_ids,
        "critique_available_task_ids": sorted([task_id for task_id in critique_eligible_task_ids if task_id]),
        "active_task_cards": visible_task_cards,
        "suppressed_task_cards": suppressed_task_cards,
        "all_task_card_count": len(all_task_cards),
        "suppressed_task_card_count": len(suppressed_task_cards),
        "noise_signal_policy": {
            "show_stale": show_stale,
            "show_redundant": show_redundant,
            "verbose_state": verbose_state,
            "show_chatter": show_chatter,
            "show_recoverable_errors": show_recoverable_errors,
        },
        "freshness": {
            "status": task_freshness.get("status"),
            "reason": task_freshness.get("reason"),
            "worst_age_sec": task_freshness.get("worst_age_sec"),
            "last_signal_age_sec": execution_last_signal_age_sec,
            "last_progress_age_sec": execution_last_progress_age_sec,
            "mission_control_age_sec": mission_control_age_sec,
        },
    },
    "visibility_scorecard": {
        "score": system_visibility_score,
        "rating": system_visibility_rating,
        "dimensions": scorecard_dimension_averages,
        "weights": {
            "task_freshness": 0.30,
            "worker_health": 0.25,
            "blockage_severity": 0.20,
            "evidence_quality": 0.15,
            "execution_path_stability": 0.10,
        },
        "thresholds": {
            "healthy_min": visibility_healthy_min,
            "degraded_min": visibility_degraded_min,
        },
        "task_rating_counts": visibility_rating_counts,
        "task_count": len(all_task_cards),
        "visible_task_count": len(visible_task_cards),
        "suppressed_task_count": len(suppressed_task_cards),
    },
    "worker_state": {
        "demoted_worker_count": demoted_worker_count,
        "restore_pending_worker_count": restore_pending_worker_count,
        "restored_worker_count": restored_worker_count,
        "probe_due_now_worker_count": probe_due_now_worker_count,
        "probe_overdue_worker_count": probe_overdue_worker_count,
        "oldest_demoted_worker": str(
            execution_status.get("supervisor_launch_readiness_oldest_demoted_worker")
            or execution_status.get("supervisor_dispatch_intent_oldest_demoted_worker")
            or execution_status.get("supervisor_dispatch_qualification_oldest_demoted_worker")
            or execution_dispatch_intent.get("launch_readiness_oldest_demoted_worker")
            or execution_dispatch_qualification.get("launch_readiness_oldest_demoted_worker")
            or ""
        ).strip()
        or None,
        "oldest_demoted_age_sec": oldest_demoted_age_sec,
        "probe_execution_status": str(
            execution_status.get("supervisor_launch_readiness_probe_execution_status")
            or execution_status.get("supervisor_dispatch_intent_probe_execution_status")
            or execution_status.get("supervisor_dispatch_qualification_probe_execution_status")
            or execution_dispatch_intent.get("launch_readiness_probe_execution_status")
            or execution_dispatch_qualification.get("launch_readiness_probe_execution_status")
            or "unknown"
        ),
        "probe_execution_reason": str(
            execution_status.get("supervisor_launch_readiness_probe_execution_reason")
            or execution_status.get("supervisor_dispatch_intent_probe_execution_reason")
            or execution_status.get("supervisor_dispatch_qualification_probe_execution_reason")
            or execution_dispatch_intent.get("launch_readiness_probe_execution_reason")
            or execution_dispatch_qualification.get("launch_readiness_probe_execution_reason")
            or "none"
        ),
        "fail_closed": fail_closed_worker,
        "launch_mutation_allowed": launch_mutation_allowed_worker,
    },
    "worker_state_detail": {
        "status": worker_state_status,
        "launch_readiness_state": first_nonempty(
            execution_dispatch_qualification.get("launch_readiness_state"),
            execution_dispatch_intent.get("launch_readiness_state"),
            execution_status.get("supervisor_dispatch_qualification_launch_readiness_state"),
            execution_status.get("supervisor_dispatch_intent_launch_readiness_state"),
        )
        or "unknown",
        "launch_readiness_reason": first_nonempty(
            execution_dispatch_qualification.get("launch_readiness_reason"),
            execution_dispatch_intent.get("launch_readiness_reason"),
            execution_status.get("supervisor_dispatch_qualification_launch_readiness_reason"),
            execution_status.get("supervisor_dispatch_intent_launch_readiness_reason"),
        )
        or "none",
        "launch_readiness_severity_state": first_nonempty(
            execution_status.get("supervisor_launch_readiness_severity_state"),
            execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_state"),
            execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_state"),
            execution_dispatch_qualification.get("launch_readiness_severity_state"),
            execution_dispatch_intent.get("launch_readiness_severity_state"),
        )
        or "none",
        "launch_readiness_severity_reason": first_nonempty(
            execution_status.get("supervisor_launch_readiness_severity_reason"),
            execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_reason"),
            execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_reason"),
            execution_dispatch_qualification.get("launch_readiness_severity_reason"),
            execution_dispatch_intent.get("launch_readiness_severity_reason"),
        )
        or "none",
        "launch_readiness_severity_non_ready_ticks": max(
            safe_int(execution_status.get("supervisor_launch_readiness_severity_non_ready_ticks"), -1),
            safe_int(execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_non_ready_ticks"), -1),
            safe_int(execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_non_ready_ticks"), -1),
            safe_int(execution_dispatch_qualification.get("launch_readiness_severity_non_ready_ticks"), -1),
            safe_int(execution_dispatch_intent.get("launch_readiness_severity_non_ready_ticks"), -1),
            0,
        ),
        "launch_readiness_severity_threshold_ticks": max(
            safe_int(execution_status.get("supervisor_launch_readiness_severity_threshold_ticks"), -1),
            safe_int(execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_threshold_ticks"), -1),
            safe_int(execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_threshold_ticks"), -1),
            safe_int(execution_dispatch_qualification.get("launch_readiness_severity_threshold_ticks"), -1),
            safe_int(execution_dispatch_intent.get("launch_readiness_severity_threshold_ticks"), -1),
            0,
        ),
        "demotion_action_priority": first_nonempty(
            execution_status.get("supervisor_launch_readiness_demotion_action_priority"),
            execution_status.get("supervisor_dispatch_qualification_demotion_action_priority"),
            execution_status.get("supervisor_dispatch_intent_demotion_action_priority"),
            execution_dispatch_qualification.get("launch_readiness_demotion_action_priority"),
            execution_dispatch_intent.get("launch_readiness_demotion_action_priority"),
        )
        or "none",
        "probe_execution_action_priority": first_nonempty(
            execution_status.get("supervisor_launch_readiness_probe_execution_action_priority"),
            execution_status.get("supervisor_dispatch_qualification_probe_execution_action_priority"),
            execution_status.get("supervisor_dispatch_intent_probe_execution_action_priority"),
            execution_dispatch_qualification.get("launch_readiness_probe_execution_action_priority"),
            execution_dispatch_intent.get("launch_readiness_probe_execution_action_priority"),
        )
        or "none",
        "oldest_restore_pending_worker": first_nonempty(
            execution_status.get("supervisor_launch_readiness_oldest_restore_pending_worker"),
            execution_status.get("supervisor_dispatch_qualification_oldest_restore_pending_worker"),
            execution_status.get("supervisor_dispatch_intent_oldest_restore_pending_worker"),
            execution_dispatch_qualification.get("launch_readiness_oldest_restore_pending_worker"),
            execution_dispatch_intent.get("launch_readiness_oldest_restore_pending_worker"),
        ),
        "oldest_restore_pending_age_sec": oldest_restore_pending_age_sec,
        "oldest_probe_due_now_worker": first_nonempty(
            execution_status.get("supervisor_launch_readiness_probe_execution_oldest_due_now_worker"),
            execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_due_now_worker"),
            execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_due_now_worker"),
            execution_dispatch_qualification.get("launch_readiness_probe_execution_oldest_due_now_worker"),
            execution_dispatch_intent.get("launch_readiness_probe_execution_oldest_due_now_worker"),
        ),
        "oldest_probe_due_now_age_sec": oldest_probe_due_now_age_sec,
        "oldest_probe_overdue_worker": first_nonempty(
            execution_status.get("supervisor_launch_readiness_probe_execution_oldest_overdue_worker"),
            execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_overdue_worker"),
            execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_overdue_worker"),
            execution_dispatch_qualification.get("launch_readiness_probe_execution_oldest_overdue_worker"),
            execution_dispatch_intent.get("launch_readiness_probe_execution_oldest_overdue_worker"),
        ),
        "oldest_probe_overdue_age_sec": oldest_probe_overdue_age_sec,
        "freshness": {
            "status": worker_freshness.get("status"),
            "reason": worker_freshness.get("reason"),
            "worst_age_sec": worker_freshness.get("worst_age_sec"),
            "dispatch_intent_age_sec": dispatch_intent_age_sec,
            "dispatch_qualification_age_sec": dispatch_qualification_age_sec,
            "last_signal_age_sec": execution_last_signal_age_sec,
        },
    },
    "blockage": {
        "primary_reason": blockage_reasons[0] if blockage_reasons else None,
        "reasons": blockage_reasons,
        "blocked_candidate_count": max(
            safe_int(execution_dispatch_intent.get("blocked_candidate_count"), -1),
            safe_int(execution_dispatch_qualification.get("blocked_candidate_count"), -1),
            safe_int(execution_status.get("supervisor_dispatch_intent_blocked_candidate_count"), -1),
            safe_int(execution_status.get("supervisor_dispatch_qualification_blocked_candidate_count"), -1),
            0,
        ),
        "program_state": str(execution_frontier.get("program_state") or execution_status.get("program_state") or "unknown"),
        "dispatch_status": str(execution_status.get("dispatch_status") or execution_frontier.get("dispatch_status") or "unknown"),
        "frontier_selector_state": str(execution_frontier.get("selector_state") or execution_status.get("frontier_selector_state") or "unknown"),
        "frontier_supervisor_state": str(execution_frontier.get("supervisor_state") or execution_status.get("frontier_supervisor_state") or "unknown"),
        "frontier_stalled": bool(execution_frontier.get("stalled") is True),
        "autonomous_dispatch_status": str(execution_status.get("autonomous_dispatch_status") or "missing"),
        "autonomous_dispatch_block_reason": first_nonempty(execution_status.get("autonomous_dispatch_block_reason"), execution_status.get("skip_reason")),
        "autonomous_dispatch_error": first_nonempty(execution_status.get("autonomous_dispatch_error")),
        "reason_count": len(blockage_reasons),
    },
}

task_freshness_status_for_workflow = str(
    ((execution_snapshot.get("task_detail") or {}).get("freshness") or {}).get("status") or "unknown"
).strip().lower()
worker_freshness_status_for_workflow = str(
    ((execution_snapshot.get("worker_state_detail") or {}).get("freshness") or {}).get("status") or "unknown"
).strip().lower()
worker_rollup_status_for_workflow = str(
    ((execution_snapshot.get("worker_state_detail") or {}).get("status") or "unknown")
).strip().lower() or "unknown"
blockage_obj_for_workflow = execution_snapshot.get("blockage") if isinstance(execution_snapshot.get("blockage"), dict) else {}
worker_state_obj_for_workflow = execution_snapshot.get("worker_state") if isinstance(execution_snapshot.get("worker_state"), dict) else {}
is_blocked_for_workflow = (
    int(blockage_obj_for_workflow.get("blocked_candidate_count") or 0) > 0
    or str(blockage_obj_for_workflow.get("program_state") or "").strip().lower() == "blocked"
    or str(blockage_obj_for_workflow.get("dispatch_status") or "").strip().lower() == "blocked"
    or str((execution_snapshot.get("worker_state_detail") or {}).get("status") or "").strip().lower() == "blocked"
)
inspect_freshness_action = action_brief(find_action("current --refresh", "continuity_current.sh --json", "refresh"))
inspect_blockage_frontier_action = action_brief(find_action("inspect_execution_frontier", "execution-frontier", "watchdog", "queue"))
inspect_worker_state_action = action_brief(
    find_action(
        "inspect_execution_supervisor_demotion_restore_posture",
        "dispatch_qualification",
        "continuity_current.sh --json",
    )
)
freshness_requires_attention_for_workflow = (
    task_freshness_status_for_workflow in {"aging", "stale"}
    or worker_freshness_status_for_workflow in {"aging", "stale"}
)
active_slice_ids_for_surface = [
    str(task_id or "").strip()
    for task_id in execution_task_ids
    if str(task_id or "").strip()
]
blocked_task_ids_for_surface = [
    str(task_id or "").strip()
    for task_id in blocked_candidate_task_ids
    if str(task_id or "").strip()
]
task_state_surface_state = "ready"
if is_blocked_for_workflow:
    task_state_surface_state = "blocked"
elif freshness_requires_attention_for_workflow:
    task_state_surface_state = "stale"

execution_snapshot["inspection_workflow"] = {
    "freshness": {
        "task_status": task_freshness_status_for_workflow,
        "worker_status": worker_freshness_status_for_workflow,
        "requires_attention": freshness_requires_attention_for_workflow,
        "next_action": inspect_freshness_action,
    },
    "blockage": {
        "blocked": is_blocked_for_workflow,
        "primary_reason": blockage_obj_for_workflow.get("primary_reason"),
        "reason_count": int(blockage_obj_for_workflow.get("reason_count") or 0),
        "frontier_next_candidate": execution_snapshot.get("frontier_next_candidate"),
        "next_action": inspect_blockage_frontier_action,
        "worker_state_action": inspect_worker_state_action,
        "fail_closed": bool(worker_state_obj_for_workflow.get("fail_closed") is True),
    },
    "inspection_order": [
        {
            "step": "inspect_freshness",
            "next_action": inspect_freshness_action,
        },
        {
            "step": "inspect_blockage_frontier",
            "next_action": inspect_blockage_frontier_action,
        },
        {
            "step": "inspect_worker_state",
            "next_action": inspect_worker_state_action,
        },
    ],
}

execution_snapshot["task_state_surface"] = {
    "schema": "clawd.operator_task_state_surface.v1",
    "summary": {
        "state": task_state_surface_state,
        "blocked_work_count": len(blocked_task_ids_for_surface),
        "active_slice_count": len(active_slice_ids_for_surface),
        "freshness_status": task_freshness_status_for_workflow,
        "worker_status": worker_rollup_status_for_workflow,
        "requires_attention": bool(freshness_requires_attention_for_workflow or is_blocked_for_workflow),
    },
    "blocked_work": {
        "blocked": is_blocked_for_workflow,
        "task_ids": blocked_task_ids_for_surface,
        "primary_reason": blockage_obj_for_workflow.get("primary_reason"),
        "reason_count": int(blockage_obj_for_workflow.get("reason_count") or 0),
        "frontier_next_candidate": execution_snapshot.get("frontier_next_candidate"),
    },
    "active_slices": {
        "task_ids": active_slice_ids_for_surface,
        "foreground_task_ids": [
            str(task_id or "").strip()
            for task_id in foreground_task_ids
            if str(task_id or "").strip()
        ],
        "background_ready_task_ids": [
            str(task_id or "").strip()
            for task_id in background_ready_task_ids
            if str(task_id or "").strip()
        ],
        "current_focus": execution_snapshot.get("current_focus"),
        "frontier_next_candidate": execution_snapshot.get("frontier_next_candidate"),
    },
    "freshness": {
        "task_status": task_freshness_status_for_workflow,
        "task_worst_age_sec": (
            ((execution_snapshot.get("task_detail") or {}).get("freshness") or {}).get("worst_age_sec")
        ),
        "worker_status": worker_freshness_status_for_workflow,
        "worker_worst_age_sec": (
            ((execution_snapshot.get("worker_state_detail") or {}).get("freshness") or {}).get("worst_age_sec")
        ),
        "requires_attention": freshness_requires_attention_for_workflow,
    },
    "safe_next_actions": {
        "inspect_freshness": inspect_freshness_action,
        "inspect_blockage_frontier": inspect_blockage_frontier_action,
        "inspect_worker_state": inspect_worker_state_action,
    },
}

readiness_upper = str(readiness or "UNKNOWN").strip().upper() or "UNKNOWN"
mutation_gate_lower = str(mutation_gate or "unknown").strip().lower() or "unknown"
freshness_posture = str(
    freshness.get("posture")
    or headline.get("freshness_posture")
    or "unknown"
).strip().lower() or "unknown"
dispatch_status_for_state = str(blockage_obj_for_workflow.get("dispatch_status") or "").strip().lower()
program_state_for_state = str(blockage_obj_for_workflow.get("program_state") or "").strip().lower()

blocked_state_active = bool(
    is_blocked_for_workflow
    or mutation_gate_lower == "forbidden"
    or hard_blockers > 0
    or dispatch_status_for_state == "blocked"
    or program_state_for_state == "blocked"
)
stale_state_active = bool(
    freshness_posture in {"aging", "stale"}
    or task_freshness_status_for_workflow in {"aging", "stale"}
    or worker_freshness_status_for_workflow in {"aging", "stale"}
    or bool(execution_frontier.get("stalled") is True)
    or bool(queue_stale_wave_signal.get("active") is True)
)
ready_state_active = bool(
    readiness_upper in {"READY", "READY_WITH_DEBT"}
    and mutation_gate_lower == "allowed"
    and not blocked_state_active
    and not stale_state_active
)

blocked_state_triggers = unique_signal_tokens(
    "workflow_blocked" if is_blocked_for_workflow else None,
    "mutation_gate_forbidden" if mutation_gate_lower == "forbidden" else None,
    "hard_blockers_present" if hard_blockers > 0 else None,
    "dispatch_status_blocked" if dispatch_status_for_state == "blocked" else None,
    "program_state_blocked" if program_state_for_state == "blocked" else None,
)
stale_state_triggers = unique_signal_tokens(
    f"freshness_posture_{freshness_posture}" if freshness_posture in {"aging", "stale"} else None,
    (
        f"task_freshness_{task_freshness_status_for_workflow}"
        if task_freshness_status_for_workflow in {"aging", "stale"}
        else None
    ),
    (
        f"worker_freshness_{worker_freshness_status_for_workflow}"
        if worker_freshness_status_for_workflow in {"aging", "stale"}
        else None
    ),
    "frontier_stalled" if bool(execution_frontier.get("stalled") is True) else None,
    "queue_stale_wave_active" if bool(queue_stale_wave_signal.get("active") is True) else None,
)
ready_state_triggers = unique_signal_tokens(
    f"readiness_{readiness_upper.lower()}" if readiness_upper in {"READY", "READY_WITH_DEBT"} else None,
    "mutation_gate_allowed" if mutation_gate_lower == "allowed" else None,
    "no_blocked_signal" if not blocked_state_active else None,
    "no_stale_signal" if not stale_state_active else None,
)

blocked_reason = first_signal_reason(
    blockage_obj_for_workflow.get("primary_reason"),
    execution_frontier.get("blocked_reason"),
    execution_status.get("autonomous_dispatch_block_reason"),
    execution_status.get("skip_reason"),
    ((execution_snapshot.get("worker_state_detail") or {}).get("launch_readiness_reason")),
    freshness.get("failclose_reasons"),
    ((mission_payload.get("generation_pointer") or {}).get("failclose_reasons")),
)
if blocked_state_active and not blocked_reason:
    if mutation_gate_lower == "forbidden":
        blocked_reason = "mutation_gate_forbidden"
    elif hard_blockers > 0:
        blocked_reason = "hard_blockers_present"
    elif dispatch_status_for_state == "blocked":
        blocked_reason = "dispatch_status_blocked"
    elif program_state_for_state == "blocked":
        blocked_reason = "program_state_blocked"
    else:
        blocked_reason = "blocked_state_detected"

stale_reason = first_signal_reason(
    execution_frontier.get("stalled_reason"),
    queue_stale_wave_signal.get("reason"),
    ((execution_snapshot.get("task_detail") or {}).get("freshness") or {}).get("reason"),
    ((execution_snapshot.get("worker_state_detail") or {}).get("freshness") or {}).get("reason"),
    freshness.get("failclose_reasons"),
)
if stale_state_active and not stale_reason:
    stale_reason = "stale_or_aging_signal_detected"

ready_reason = first_signal_reason(
    (
        f"next_candidate:{execution_snapshot.get('frontier_next_candidate')}"
        if str(execution_snapshot.get("frontier_next_candidate") or "").strip()
        else None
    ),
    (
        f"current_focus:{execution_snapshot.get('current_focus')}"
        if str(execution_snapshot.get("current_focus") or "").strip()
        else None
    ),
)
if ready_state_active and not ready_reason:
    ready_reason = "verify_ready_mutation_allowed"

blocked_safe_next_action = action_brief_or_default(
    find_action("inspect_execution_frontier", "execution-frontier", "queue", "watchdog"),
    default_action="inspect_execution_frontier",
    default_command="cat state/continuity/latest/execution_frontier_ledger.json",
    default_priority="p1",
)
stale_safe_next_action = action_brief_or_default(
    find_action("current --refresh", "reconcile", "queue-sync", "queue-ready", "watchdog"),
    default_action="refresh_continuity_and_recheck_staleness",
    default_command=f"bash {root / 'ops' / 'openclaw' / 'continuity.sh'} current --refresh --json",
    default_priority="p1",
)
ready_safe_next_action = action_brief_or_default(
    find_action("inspect_execution_frontier", "ready-list", "queue-ready", "watchdog", "dispatch"),
    default_action="inspect_next_ready_candidate",
    default_command="cat state/continuity/latest/execution_frontier_ledger.json",
    default_priority="p2",
)

state_cards = {
    "blocked": {
        "active": blocked_state_active,
        "reason": blocked_reason,
        "triggers": blocked_state_triggers,
        "safe_next_action": blocked_safe_next_action,
    },
    "stale": {
        "active": stale_state_active,
        "reason": stale_reason,
        "triggers": stale_state_triggers,
        "safe_next_action": stale_safe_next_action,
    },
    "ready": {
        "active": ready_state_active,
        "reason": ready_reason,
        "triggers": ready_state_triggers,
        "safe_next_action": ready_safe_next_action,
    },
}

freshness_projection_expected_attention = bool(
    task_freshness_status_for_workflow in {"aging", "stale"}
    or worker_freshness_status_for_workflow in {"aging", "stale"}
)
inspection_workflow_obj = (
    execution_snapshot.get("inspection_workflow")
    if isinstance(execution_snapshot.get("inspection_workflow"), dict)
    else {}
)
freshness_workflow_obj = (
    inspection_workflow_obj.get("freshness")
    if isinstance(inspection_workflow_obj.get("freshness"), dict)
    else {}
)
task_state_surface_obj = (
    execution_snapshot.get("task_state_surface")
    if isinstance(execution_snapshot.get("task_state_surface"), dict)
    else {}
)
task_surface_freshness_obj = (
    task_state_surface_obj.get("freshness")
    if isinstance(task_state_surface_obj.get("freshness"), dict)
    else {}
)
freshness_projection_inspection_attention = bool(
    freshness_workflow_obj.get("requires_attention") is True
)
freshness_projection_task_surface_attention = bool(
    task_surface_freshness_obj.get("requires_attention") is True
)
freshness_projection_mission_attention = bool(freshness_posture in {"aging", "stale"})
freshness_projection_mission_failclose_reason_count = max(
    int(headline.get("freshness_failclose_reason_count") or 0),
    len([str(x) for x in (freshness.get("failclose_reasons") or []) if str(x).strip()]),
)
freshness_projection_parity_reasons: List[str] = []
if freshness_projection_inspection_attention != freshness_projection_expected_attention:
    freshness_projection_parity_reasons.append(
        "inspection_workflow_freshness_requires_attention_mismatch"
    )
if freshness_projection_task_surface_attention != freshness_projection_expected_attention:
    freshness_projection_parity_reasons.append(
        "task_state_surface_freshness_requires_attention_mismatch"
    )
if (
    freshness_projection_mission_failclose_reason_count <= 0
    and freshness_projection_mission_attention != freshness_projection_expected_attention
):
    freshness_projection_parity_reasons.append(
        "mission_freshness_posture_projection_mismatch"
    )

freshness_projection_fail_closed = bool(freshness_projection_parity_reasons)
freshness_projection_degraded = bool(
    freshness_projection_expected_attention
    or freshness_projection_mission_attention
    or freshness_projection_fail_closed
)
freshness_projection_reason = "freshness_healthy"
if freshness_projection_fail_closed:
    freshness_projection_reason = first_signal_reason(freshness_projection_parity_reasons) or "freshness_projection_parity_mismatch"
elif freshness_projection_expected_attention:
    freshness_projection_reason = "freshness_attention_required"
elif freshness_projection_mission_attention:
    freshness_projection_reason = "mission_freshness_posture_requires_attention"

freshness_projection_next_action = action_brief_or_default(
    inspect_freshness_action,
    default_action="refresh_continuity_and_recheck_staleness",
    default_command=f"bash {root / 'ops' / 'openclaw' / 'continuity.sh'} current --refresh --json",
    default_priority="p1",
)

execution_snapshot["freshness_degradation_projection"] = {
    "status": "degraded" if freshness_projection_degraded else "healthy",
    "degraded": freshness_projection_degraded,
    "fail_closed": freshness_projection_fail_closed,
    "reason": freshness_projection_reason,
    "reasons": freshness_projection_parity_reasons,
    "task_status": task_freshness_status_for_workflow,
    "worker_status": worker_freshness_status_for_workflow,
    "mission_freshness_posture": freshness_posture,
    "requires_attention": freshness_projection_expected_attention,
    "safe_next_action": freshness_projection_next_action,
    "parity": {
        "inspection_requires_attention": freshness_projection_inspection_attention,
        "task_surface_requires_attention": freshness_projection_task_surface_attention,
        "mission_posture_requires_attention": freshness_projection_mission_attention,
        "mission_failclose_reason_count": freshness_projection_mission_failclose_reason_count,
        "inspection_matches_projection": (
            freshness_projection_inspection_attention == freshness_projection_expected_attention
        ),
        "task_surface_matches_projection": (
            freshness_projection_task_surface_attention == freshness_projection_expected_attention
        ),
        "mission_posture_matches_projection": (
            (
                freshness_projection_mission_failclose_reason_count > 0
                or freshness_projection_mission_attention == freshness_projection_expected_attention
            )
        ),
    },
}

dominant_state = "unknown"
if blocked_state_active:
    dominant_state = "blocked"
elif stale_state_active:
    dominant_state = "stale"
elif ready_state_active:
    dominant_state = "ready"

dominant_reason = (
    first_signal_reason((state_cards.get(dominant_state) or {}).get("reason"))
    if dominant_state in state_cards
    else None
)
if dominant_state != "unknown" and not dominant_reason:
    dominant_reason = f"{dominant_state}_state_detected"

dominant_triggers = (
    unique_signal_tokens((state_cards.get(dominant_state) or {}).get("triggers"))
    if dominant_state in state_cards
    else []
)
if dominant_state != "unknown" and not dominant_triggers:
    dominant_triggers = [f"{dominant_state}_state_detected"]

dominant_safe_next_action = (
    (state_cards.get(dominant_state) or {}).get("safe_next_action")
    if dominant_state in state_cards
    else None
)
if not isinstance(dominant_safe_next_action, dict):
    dominant_safe_next_action = action_brief_or_default(
        find_action(),
        default_action="refresh_operator_mission_control",
        default_command=f"bash {root / 'ops' / 'openclaw' / 'continuity.sh'} mission-control --refresh --json",
        default_priority="p2",
    )

execution_snapshot["state_explainability"] = {
    "schema": "clawd.execution_state_explainability.v1",
    "decision_order": ["blocked", "stale", "ready"],
    "dominant_state": dominant_state,
    "dominant_reason": dominant_reason,
    "dominant_triggers": dominant_triggers,
    "safe_next_action": dominant_safe_next_action,
    "state_cards": state_cards,
    "signals": {
        "readiness": readiness_upper,
        "mutation_gate": mutation_gate_lower,
        "freshness_posture": freshness_posture,
        "hard_blockers": max(0, hard_blockers),
        "warnings": max(0, warnings),
        "frontier_stalled": bool(execution_frontier.get("stalled") is True),
        "queue_stale_wave_active": bool(queue_stale_wave_signal.get("active") is True),
        "dispatch_status": dispatch_status_for_state or None,
        "program_state": program_state_for_state or None,
    },
    "visibility_scorecard": {
        "score": (execution_snapshot.get("visibility_scorecard") or {}).get("score")
        if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
        else None,
        "rating": (execution_snapshot.get("visibility_scorecard") or {}).get("rating")
        if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
        else None,
        "task_count": (execution_snapshot.get("visibility_scorecard") or {}).get("task_count")
        if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
        else None,
        "suppressed_task_count": (execution_snapshot.get("visibility_scorecard") or {}).get("suppressed_task_count")
        if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
        else None,
    },
    "transcript_mining_required": False,
}

runtime_component_consistency_overlay_raw = build_component_consistency_audit_overlay(
    execution_snapshot=execution_snapshot,
    all_task_cards=all_task_cards,
    visible_task_cards=visible_task_cards,
    suppressed_task_cards=suppressed_task_cards,
)
runtime_overlay_findings = (
    runtime_component_consistency_overlay_raw.get("findings")
    if isinstance(runtime_component_consistency_overlay_raw.get("findings"), list)
    else []
)
runtime_overlay_critical_count = len(
    [
        row
        for row in runtime_overlay_findings
        if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() in {"critical", "high"}
    ]
)
runtime_component_consistency_overlay = {
    "schema": "clawd.component_consistency_audit_overlay.v1",
    "overlay_id": "component_consistency_audit_overlay.v1",
    "generated_at": runtime_component_consistency_overlay_raw.get("generated_at") or now_iso(),
    "status": "applied" if runtime_overlay_findings else "clean",
    "reason": "runtime_execution_snapshot",
    "policy": runtime_component_consistency_overlay_raw.get("policy"),
    "rule_ids_evaluated": [
        "count_mismatch",
        "state_discrepancy",
        "status_rollup_error",
        "freshness_contradiction",
        "suppression_logic_error",
        "blockage_reason_mismatch",
    ],
    "finding_count": len(runtime_overlay_findings),
    "critical_finding_count": runtime_overlay_critical_count,
    "findings": runtime_overlay_findings,
    "source_scope": "runtime_execution_snapshot",
}
if component_consistency_overlay is None:
    component_consistency_overlay = runtime_component_consistency_overlay
elif isinstance(component_consistency_overlay, dict):
    component_consistency_overlay["runtime_execution_snapshot_overlay"] = {
        "status": runtime_component_consistency_overlay.get("status"),
        "finding_count": int(runtime_component_consistency_overlay.get("finding_count") or 0),
        "critical_finding_count": int(runtime_component_consistency_overlay.get("critical_finding_count") or 0),
    }

if isinstance(component_consistency_overlay, dict):
    component_consistency_overlay["artifact_path"] = rel_path_for(component_consistency_overlay_path)
    atomic_write(component_consistency_overlay_path, component_consistency_overlay)

task_freshness_status = str((execution_snapshot.get("task_detail") or {}).get("freshness", {}).get("status") or "unknown")
if task_freshness_status in {"aging", "stale"} and (
    int(execution_snapshot.get("active_task_count") or 0) > 0
    or int((execution_snapshot.get("blockage") or {}).get("blocked_candidate_count") or 0) > 0
):
    task_freshness = (execution_snapshot.get("task_detail") or {}).get("freshness") or {}
    add_issue(
        issue_id=f"execution_task_freshness_{task_freshness_status}",
        severity="p1" if task_freshness_status == "aging" else "p0",
        lane="execution",
        reason=(
            f"task_freshness={task_freshness_status} "
            f"worst_age_sec={task_freshness.get('worst_age_sec') if task_freshness.get('worst_age_sec') is not None else 'unknown'}"
        ),
        evidence_ref="state/continuity/latest/operator_mission_control.json",
        action_hint=find_action("execution-frontier", "queue", "watchdog"),
    )

worker_freshness_status = str((execution_snapshot.get("worker_state_detail") or {}).get("freshness", {}).get("status") or "unknown")
if worker_freshness_status in {"aging", "stale"} and (
    int((execution_snapshot.get("worker_state") or {}).get("demoted_worker_count") or 0) > 0
    or int((execution_snapshot.get("worker_state") or {}).get("restore_pending_worker_count") or 0) > 0
    or int((execution_snapshot.get("worker_state") or {}).get("probe_due_now_worker_count") or 0) > 0
    or int((execution_snapshot.get("worker_state") or {}).get("probe_overdue_worker_count") or 0) > 0
    or bool((execution_snapshot.get("worker_state") or {}).get("fail_closed") is True)
):
    worker_freshness = (execution_snapshot.get("worker_state_detail") or {}).get("freshness") or {}
    add_issue(
        issue_id=f"execution_worker_surface_freshness_{worker_freshness_status}",
        severity="p1" if worker_freshness_status == "aging" else "p0",
        lane="execution",
        reason=(
            f"worker_surface_freshness={worker_freshness_status} "
            f"worst_age_sec={worker_freshness.get('worst_age_sec') if worker_freshness.get('worst_age_sec') is not None else 'unknown'}"
        ),
        evidence_ref="state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
        action_hint=find_action("dispatch_qualification", "verify_gate", "continuity_current.sh --json"),
    )

for reason in unique_preserve([str(x) for x in (freshness.get("failclose_reasons") or []) if str(x).strip()]):
    add_issue(
        issue_id=f"freshness_{reason}",
        severity="p0",
        lane="freshness",
        reason=reason,
        evidence_ref="state/continuity/current.json",
        action_hint=find_action("reconcile", "refresh", "continuity.sh mission-control --refresh"),
    )

freshness_degradation_projection_obj = (
    execution_snapshot.get("freshness_degradation_projection")
    if isinstance(execution_snapshot.get("freshness_degradation_projection"), dict)
    else {}
)
if bool(freshness_degradation_projection_obj.get("fail_closed") is True):
    freshness_projection_reasons = unique_preserve(
        [str(x) for x in (freshness_degradation_projection_obj.get("reasons") or []) if str(x).strip()]
    )
    add_issue(
        issue_id="execution_freshness_degradation_projection_fail_closed",
        severity="p0",
        lane="execution",
        reason=(
            f"freshness_degradation_projection_fail_closed "
            f"reason={first_signal_reason(freshness_projection_reasons) or 'projection_parity_mismatch'}"
        ),
        evidence_ref="state/continuity/latest/operator_triage_console.json",
        action_hint=action_brief_or_default(
            (freshness_degradation_projection_obj.get("safe_next_action") if isinstance(freshness_degradation_projection_obj.get("safe_next_action"), dict) else None),
            default_action="refresh_continuity_and_recheck_staleness",
            default_command=f"bash {root / 'ops' / 'openclaw' / 'continuity.sh'} current --refresh --json",
            default_priority="p1",
        ),
    )

for reason in unique_preserve([str(x) for x in (mission_payload.get("generation_pointer", {}).get("failclose_reasons") or []) if str(x).strip()]):
    add_issue(
        issue_id=f"generation_pointer_{reason}",
        severity="p0",
        lane="coherence",
        reason=reason,
        evidence_ref="state/continuity/latest/continuity_read_pointer.json",
        action_hint=find_action("reconcile", "current --refresh", "inspect_generation_pointer"),
    )

for reason in unique_preserve([str(x) for x in (meaningful_event_contract.get("failclose_reasons") or []) if str(x).strip()]):
    add_issue(
        issue_id=f"meaningful_event_contract_{reason}",
        severity="p0",
        lane="meaningful_event_reporting",
        reason=reason,
        evidence_ref="state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
        action_hint=find_action("meaningful_event", "current --refresh", "inspect_execution_meaningful_event_reporting_status"),
    )

if bool(execution_frontier.get("stalled") is True):
    add_issue(
        issue_id="execution_frontier_stalled",
        severity="p1",
        lane="execution",
        reason=str(execution_frontier.get("stalled_reason") or "execution_frontier_stalled"),
        evidence_ref=str(execution_frontier.get("source_path") or "state/continuity/latest/execution_frontier_ledger.json"),
        action_hint=find_action("execution-frontier", "queue-replay", "watchdog"),
    )

frontier_blocked_reason = str(execution_frontier.get("blocked_reason") or "").strip()
if frontier_blocked_reason:
    add_issue(
        issue_id="execution_frontier_blocked_reason",
        severity="p1",
        lane="execution",
        reason=frontier_blocked_reason,
        evidence_ref=str(execution_frontier.get("source_path") or "state/continuity/latest/execution_frontier_ledger.json"),
        action_hint=find_action("execution-frontier", "queue", "watchdog"),
    )

controller_status = str(execution_status.get("autonomous_dispatch_status") or "")
if controller_status in {"blocked", "error", "missing", "skipped"}:
    controller_reason = str(
        execution_status.get("autonomous_dispatch_block_reason")
        or execution_status.get("autonomous_dispatch_error")
        or execution_status.get("autonomous_dispatch_skip_reason")
        or controller_status
    )
    add_issue(
        issue_id=f"execution_frontier_controller_{controller_status}",
        severity="p1",
        lane="execution",
        reason=controller_reason,
        evidence_ref=str(execution_status.get("autonomous_dispatch_trace_path") or "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"),
        action_hint=find_action("watchdog", "autonomous-dispatch", "execution-frontier"),
    )

if int(execution_snapshot.get("worker_state", {}).get("demoted_worker_count") or 0) > 0:
    oldest_worker = str(execution_snapshot.get("worker_state", {}).get("oldest_demoted_worker") or "").strip() or "unknown"
    oldest_age = execution_snapshot.get("worker_state", {}).get("oldest_demoted_age_sec")
    oldest_age_txt = f" age_sec={oldest_age}" if oldest_age is not None else ""
    add_issue(
        issue_id="execution_workers_demoted",
        severity="p1",
        lane="execution",
        reason=(
            f"demoted_workers={int(execution_snapshot.get('worker_state', {}).get('demoted_worker_count') or 0)} "
            f"oldest={oldest_worker}{oldest_age_txt}"
        ),
        evidence_ref="state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
        action_hint=find_action("inspect_execution_supervisor_demotion_restore_posture", "dispatch_qualification", "continuity_current.sh --json"),
    )

if int(execution_snapshot.get("worker_state", {}).get("probe_overdue_worker_count") or 0) > 0:
    add_issue(
        issue_id="execution_probe_overdue_workers",
        severity="p0",
        lane="execution",
        reason=(
            f"probe_overdue_workers={int(execution_snapshot.get('worker_state', {}).get('probe_overdue_worker_count') or 0)} "
            f"reason={execution_snapshot.get('worker_state', {}).get('probe_execution_reason') or 'unknown'}"
        ),
        evidence_ref="state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
        action_hint=find_action("inspect_verify_gate_launch_readiness_probe_execution", "dispatch_qualification", "continuity_current.sh --json"),
    )

routing_failure_reason = str(routing_preflight.get("failure_reason") or "").strip()
if routing_failure_reason and routing_failure_reason != "none":
    add_issue(
        issue_id=f"routing_{routing_failure_reason}",
        severity="p1",
        lane="routing",
        reason=routing_failure_reason,
        evidence_ref="state/continuity/session_topology_router/decisions.jsonl",
        action_hint=find_action("model-route-policy-lint", "inspect_effective_routing", "routing"),
    )

if bool(model_rollout_remediation.get("active") is True):
    remediation_reason = str(
        model_rollout_remediation.get("reason_code")
        or model_rollout_remediation.get("reason_gate")
        or "operator_mistake_remediation_active"
    )
    add_issue(
        issue_id="model_rollout_remediation_active",
        severity="p1",
        lane="model_rollout",
        reason=remediation_reason,
        evidence_ref=str(model_rollout_remediation.get("correction_cycle_log_ref") or "state/continuity/model_rollout_ledger/events.jsonl"),
        action_hint=find_action("model-rollout-controller", "mistake_remediation", "model_rollout"),
    )

if bool(load_shedding.get("critical_tier") is True):
    add_issue(
        issue_id="load_shedding_critical",
        severity="p0",
        lane="load_shedding",
        reason=str(load_shedding.get("trigger_emitted") or "critical_tier"),
        evidence_ref="state/continuity/latest/load_shedding_decision.json",
        action_hint=find_action("inspect_load_shedding", "load_shedding", "queue"),
    )
elif bool(load_shedding.get("warning_tier") is True):
    add_issue(
        issue_id="load_shedding_warning",
        severity="p1",
        lane="load_shedding",
        reason=str(load_shedding.get("trigger_emitted") or "warning_tier"),
        evidence_ref="state/continuity/latest/load_shedding_decision.json",
        action_hint=find_action("inspect_load_shedding", "load_shedding", "queue"),
    )

if bool(queue_stale_wave_signal.get("active") is True):
    add_issue(
        issue_id="queue_stale_wave_active",
        severity="p1",
        lane="queue",
        reason=str(queue_stale_wave_signal.get("reason") or "queue_stale_wave_active"),
        evidence_ref="state/continuity/latest/operator_mission_control.json",
        action_hint=find_action("queue", "queue-ready", "queue-replay"),
    )

if int(web_domain_guard.get("actionable_incident_domains") or 0) > 0:
    add_issue(
        issue_id="web_capture_actionable_incidents",
        severity="p1",
        lane="web_capture",
        reason=f"actionable_incident_domains={int(web_domain_guard.get('actionable_incident_domains') or 0)}",
        evidence_ref="state/continuity/latest",
        action_hint=find_action("web", "capture", "operator_contract"),
    )

source_of_truth_map_guard_status = str(source_of_truth_map_guard_summary.get("status") or "neutral").strip().lower()
if source_of_truth_map_guard_status in {"critical", "degraded"}:
    source_of_truth_map_guard_issue_id = "source_of_truth_map_guard_degraded"
    if str(source_of_truth_map_guard_summary.get("decision") or "").strip().upper() == "BLOCK":
        source_of_truth_map_guard_issue_id = "source_of_truth_map_guard_blocked"
    elif str(source_of_truth_map_guard_summary.get("reason") or "").strip().startswith("source_of_truth_map_guard_artifact_"):
        source_of_truth_map_guard_issue_id = "source_of_truth_map_guard_artifact_invalid"

    source_of_truth_map_guard_reason = first_signal_reason(
        source_of_truth_map_guard_summary.get("reason"),
        source_of_truth_map_guard_summary.get("block_reason"),
    ) or "source_of_truth_map_guard_unhealthy"

    add_issue(
        issue_id=source_of_truth_map_guard_issue_id,
        severity="p0" if source_of_truth_map_guard_status == "critical" else "p1",
        lane="truth_gate",
        reason=(
            f"source_of_truth_map_guard_status={source_of_truth_map_guard_status} "
            f"decision={source_of_truth_map_guard_summary.get('decision') or 'unknown'} "
            f"reason={source_of_truth_map_guard_reason}"
        ),
        evidence_ref=str(source_of_truth_map_guard_summary.get("path") or rel_path_for(source_of_truth_map_guard_latest_path)),
        action_hint=action_brief_or_default(
            find_action("source_of_truth_map_guard", "anti_drift", "check_source_of_truth_map_regressions.py"),
            default_action="run_source_of_truth_map_guard",
            default_command=(
                "python3 ops/openclaw/continuity/check_source_of_truth_map_regressions.py "
                "--repo-root . "
                "--map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md "
                "--json"
            ),
            default_priority="p1",
        ),
    )

issues.sort(key=lambda item: (priority_rank.get(str(item.get("severity") or "p2"), 9), str(item.get("id") or "")))

suppressed_issues: List[Dict[str, Any]] = []
visible_issues: List[Dict[str, Any]] = []
for issue in issues:
    issue_id = str(issue.get("id") or "").strip()
    issue_severity = str(issue.get("severity") or "p2").strip().lower()
    heuristic_tags: List[str] = []
    suppression_reasons: List[str] = []

    if issue_id == "execution_frontier_blocked_reason" and bool(execution_frontier.get("stalled") is True):
        heuristic_tags.append("redundancy")
        if not show_redundant:
            suppression_reasons.append("redundancy")

    if issue_id == "queue_stale_wave_active":
        heuristic_tags.append("chatter")
        if not show_chatter:
            suppression_reasons.append("chatter")

    if issue_id == "execution_workers_demoted" and probe_overdue_worker_count <= 0 and not fail_closed_worker:
        heuristic_tags.append("recoverable_error")
        if not show_recoverable_errors:
            suppression_reasons.append("recoverable_error")

    if issue_id.endswith("_aging"):
        heuristic_tags.append("staleness")
        if not show_stale:
            suppression_reasons.append("staleness")

    suppressed = (issue_severity != "p0") and len(suppression_reasons) > 0
    if suppressed:
        suppressed_row = dict(issue)
        suppressed_row["noise_signal_tags"] = unique_preserve(heuristic_tags)
        suppressed_row["suppression_reasons"] = unique_preserve(suppression_reasons)
        suppressed_issues.append(suppressed_row)
    else:
        visible_row = dict(issue)
        visible_row["noise_signal_tags"] = unique_preserve(heuristic_tags)
        visible_issues.append(visible_row)

issues = visible_issues

lane_cards: List[Dict[str, Any]] = []


def add_lane(
    lane_id: str,
    severity: str,
    status: str,
    summary: str,
    *,
    evidence_ref: Optional[str],
    action_hint: Optional[Dict[str, Any]],
) -> None:
    lane_cards.append(
        {
            "lane": lane_id,
            "severity": severity,
            "status": status,
            "summary": summary,
            "evidence_ref": evidence_ref,
            "next_action": (
                {
                    "action": action_hint.get("action"),
                    "priority": action_hint.get("priority"),
                    "command": action_hint.get("command"),
                }
                if isinstance(action_hint, dict)
                else None
            ),
        }
    )


truth_lane_severity = "p0" if overall_severity == "p0" else ("p1" if overall_severity == "p1" else "p2")
source_of_truth_map_guard_status = str(source_of_truth_map_guard_summary.get("status") or "neutral").strip().lower()
if source_of_truth_map_guard_status == "critical":
    truth_lane_severity = "p0"
elif source_of_truth_map_guard_status == "degraded" and truth_lane_severity == "p2":
    truth_lane_severity = "p1"

source_of_truth_map_guard_reason = first_signal_reason(
    source_of_truth_map_guard_summary.get("reason"),
    source_of_truth_map_guard_summary.get("block_reason"),
) or "source_of_truth_map_guard_unknown"
add_lane(
    "truth_gate",
    truth_lane_severity,
    mutation_gate,
    (
        f"readiness={readiness} hard_blockers={hard_blockers} warnings={warnings} "
        f"source_of_truth_map_guard={source_of_truth_map_guard_status} "
        f"guard_reason={source_of_truth_map_guard_reason}"
    ),
    evidence_ref="state/continuity/latest/operator_mission_control.json",
    action_hint=find_action("reconcile", "stay_read_only", "refresh"),
)

execution_lane_severity = "p2"
if bool(execution_frontier.get("stalled") is True) or str(execution_status.get("dispatch_status") or "") in {"blocked", "stalled"}:
    execution_lane_severity = "p1"
if str(execution_status.get("autonomous_dispatch_status") or "") in {"blocked", "error"}:
    execution_lane_severity = "p0"
add_lane(
    "execution",
    execution_lane_severity,
    str(execution_status.get("dispatch_status") or "unknown"),
    (
        f"frontier_selector={execution_frontier.get('selector_state') or 'unknown'} "
        f"supervisor={execution_frontier.get('supervisor_state') or 'unknown'} "
        f"controller={execution_status.get('autonomous_dispatch_status') or 'missing'} "
        f"tasks={int(execution_snapshot.get('active_task_count') or 0)} "
        f"task_freshness={((execution_snapshot.get('task_detail') or {}).get('freshness') or {}).get('status') or 'unknown'} "
        f"demoted_workers={int(execution_snapshot.get('worker_state', {}).get('demoted_worker_count') or 0)} "
        f"worker_status={((execution_snapshot.get('worker_state_detail') or {}).get('status') or 'unknown')} "
        f"blocker={execution_snapshot.get('blockage', {}).get('primary_reason') or 'none'}"
    ),
    evidence_ref=str(execution_frontier.get("source_path") or "state/continuity/latest/execution_frontier_ledger.json"),
    action_hint=find_action("execution-frontier", "watchdog", "queue"),
)

meaningful_contract_status = str(meaningful_event_contract.get("status") or "ok")
meaningful_lane_severity = "p0" if meaningful_contract_status == "failclose" else (
    "p1" if bool(meaningful_event_reporting.get("attention_required") is True) else "p2"
)
add_lane(
    "meaningful_event_reporting",
    meaningful_lane_severity,
    meaningful_contract_status,
    (
        f"status={meaningful_event_reporting.get('status') or 'clear'} "
        f"pending={int(meaningful_event_reporting.get('pending_required_event_count') or 0)} "
        f"critical_pending={int(meaningful_event_reporting.get('critical_pending_event_count') or 0)}"
    ),
    evidence_ref="state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
    action_hint=find_action("meaningful_event", "current --refresh", "inspect_execution_meaningful_event_reporting_status"),
)

routing_failure = str(routing_preflight.get("failure_reason") or "none")
routing_lane_severity = "p2" if routing_failure in {"", "none"} else "p1"
if routing_failure == "routing_blocked":
    routing_lane_severity = "p0"
add_lane(
    "routing_and_rollout",
    routing_lane_severity,
    str((routing_preflight.get("latest") or {}).get("decision") or "unknown"),
    (
        f"routing_failure={routing_failure} "
        f"route_class={str((routing_preflight.get('effective') or {}).get('route_class') or (routing_preflight.get('latest') or {}).get('route_class') or 'none')} "
        f"rollout_prompt={str(headline.get('model_rollout_prompt_status') or 'none')}"
    ),
    evidence_ref="state/continuity/session_topology_router/decisions.jsonl",
    action_hint=find_action("model-route-policy-lint", "model-rollout", "inspect_effective_routing"),
)

freshness_posture = str(freshness.get("posture") or headline.get("freshness_posture") or "unknown")
freshness_lane_severity = "p2"
if int(headline.get("freshness_failclose_reason_count") or 0) > 0:
    freshness_lane_severity = "p0"
elif int(headline.get("warnings") or 0) > 0 or bool(headline.get("continuity_current_publish_lock_surface_active") is True):
    freshness_lane_severity = "p1"
add_lane(
    "surface_freshness",
    freshness_lane_severity,
    freshness_posture,
    (
        f"failclose={int(headline.get('freshness_failclose_reason_count') or 0)} "
        f"publish_lock_active={bool(headline.get('continuity_current_publish_lock_surface_active') is True)}"
    ),
    evidence_ref="state/continuity/current.json",
    action_hint=find_action("refresh", "inspect_current_publish_lock_owner", "reconcile"),
)

federated_evidence_task = (
    federated_evidence_task_arg
    or critique_task
    or str(execution_snapshot.get("current_focus") or "").strip()
)
federated_evidence: Optional[Dict[str, Any]] = None
if federated_evidence_task:
    federated_evidence = build_federated_evidence(
        task_id=federated_evidence_task,
        max_items=federated_evidence_max_items,
        federated_query=federated_evidence_query_arg,
        ui_bundle_summary=ui_evidence_bundle_summary,
        ui_bundle_arg=ui_evidence_bundle_arg,
        component_consistency_overlay=component_consistency_overlay,
    )

recommended_actions = actions[:max_actions]

critique_request: Optional[Dict[str, Any]] = None
if critique_task:
    critique_request = {
        "requested_task_id": critique_task,
        "cooldown_sec": critique_cooldown_sec,
        "minimum_age_sec": critique_min_age_sec,
    }

    active_task_ids = set(str(task_id or "").strip() for task_id in execution_task_ids if str(task_id or "").strip())
    if critique_task not in active_task_ids:
        critique_request.update(
            {
                "status": "task_not_active",
                "message": "Task is not active in current execution snapshot.",
            }
        )
    else:
        observed_task_age = max(
            [
                age
                for age in [execution_last_signal_age_sec, execution_last_progress_age_sec]
                if isinstance(age, int) and age >= 0
            ],
            default=None,
        )
        critique_request["observed_age_sec"] = observed_task_age

        if observed_task_age is None:
            critique_request.update(
                {
                    "status": "age_unknown",
                    "message": "Cannot establish minimum task age from current mission-control evidence.",
                }
            )
        elif observed_task_age < critique_min_age_sec:
            critique_request.update(
                {
                    "status": "task_too_young",
                    "message": (
                        f"Task observed age ({observed_task_age}s) is below minimum critique age "
                        f"({critique_min_age_sec}s)."
                    ),
                    "age_sec_remaining": max(0, critique_min_age_sec - observed_task_age),
                }
            )
        else:
            critique_index = load_json_if_exists(critique_index_path) or {}
            critique_tasks = critique_index.get("tasks") if isinstance(critique_index.get("tasks"), dict) else {}
            previous = critique_tasks.get(critique_task) if isinstance(critique_tasks.get(critique_task), dict) else {}
            previous_requested_at = previous.get("last_requested_at")
            previous_age_sec = age_sec(previous_requested_at)

            if (
                critique_cooldown_sec > 0
                and previous_requested_at
                and isinstance(previous_age_sec, int)
                and previous_age_sec < critique_cooldown_sec
            ):
                critique_request.update(
                    {
                        "status": "cooldown",
                        "message": "Critique cooldown active for this task.",
                        "cooldown_sec_remaining": max(0, critique_cooldown_sec - previous_age_sec),
                        "last_requested_at": previous_requested_at,
                    }
                )
            else:
                task_detail_obj = execution_snapshot.get("task_detail") if isinstance(execution_snapshot.get("task_detail"), dict) else {}
                worker_state_obj = execution_snapshot.get("worker_state") if isinstance(execution_snapshot.get("worker_state"), dict) else {}
                blockage_obj = execution_snapshot.get("blockage") if isinstance(execution_snapshot.get("blockage"), dict) else {}

                is_blocked = critique_task in set(task_detail_obj.get("blocked_candidate_task_ids") or [])
                is_foreground = critique_task in set(task_detail_obj.get("foreground_task_ids") or [])
                frontier_task = str(execution_snapshot.get("frontier_next_candidate") or "").strip()
                frontier_stalled = bool(blockage_obj.get("frontier_stalled") is True)
                task_freshness_status = str((task_detail_obj.get("freshness") or {}).get("status") or "unknown")

                task_card_snapshot = None
                for row in all_task_cards:
                    if str(row.get("task_id") or "").strip() == critique_task:
                        task_card_snapshot = row
                        break
                if not isinstance(task_card_snapshot, dict):
                    task_card_snapshot = {
                        "visibility_score": system_visibility_score,
                        "visibility_rating": system_visibility_rating,
                        "visibility_scorecard": {
                            "dimensions": scorecard_dimension_averages,
                            "weights": {
                                "task_freshness": 0.30,
                                "worker_health": 0.25,
                                "blockage_severity": 0.20,
                                "evidence_quality": 0.15,
                                "execution_path_stability": 0.10,
                            },
                            "weighted_score": system_visibility_score,
                            "rating": system_visibility_rating,
                        },
                    }
                visibility_state = str(task_card_snapshot.get("visibility_rating") or "unknown")
                visibility_score = clamp_percent(float(task_card_snapshot.get("visibility_score") or 0.0))
                task_noise_signal_tags = unique_preserve(
                    [
                        str(tag or "").strip()
                        for tag in (
                            task_card_snapshot.get("noise_signal_tags")
                            if isinstance(task_card_snapshot.get("noise_signal_tags"), list)
                            else []
                        )
                        if str(tag or "").strip()
                    ]
                )
                task_suppression_reasons = unique_preserve(
                    [
                        str(reason or "").strip()
                        for reason in (
                            task_card_snapshot.get("suppression_reasons")
                            if isinstance(task_card_snapshot.get("suppression_reasons"), list)
                            else []
                        )
                        if str(reason or "").strip()
                    ]
                )
                task_suppressed = bool(task_card_snapshot.get("suppressed") is True or len(task_suppression_reasons) > 0)

                stall_risk = 0.08
                if task_freshness_status == "aging":
                    stall_risk += 0.32
                elif task_freshness_status == "stale":
                    stall_risk += 0.56
                if is_blocked:
                    stall_risk += 0.26
                if frontier_stalled and frontier_task and frontier_task == critique_task:
                    stall_risk += 0.2

                thrashing_risk = 0.05
                thrashing_risk += 0.12 * int(worker_state_obj.get("demoted_worker_count") or 0)
                thrashing_risk += 0.2 * int(worker_state_obj.get("probe_overdue_worker_count") or 0)
                if bool(worker_state_obj.get("fail_closed") is True):
                    thrashing_risk += 0.2
                thrashing_risk += min(0.2, 0.04 * int(blockage_obj.get("reason_count") or 0))

                operator_burden = 0.05 + min(0.7, 0.06 * len(issues))
                operator_burden += 0.12 if is_blocked else 0.0
                operator_burden += 0.1 if bool(worker_state_obj.get("fail_closed") is True) else 0.0

                stall_risk = clamp_score(stall_risk)
                thrashing_risk = clamp_score(thrashing_risk)
                operator_burden = clamp_score(operator_burden)
                efficiency = clamp_score(1.0 - max(stall_risk, thrashing_risk * 0.85, operator_burden * 0.7))

                findings: List[Dict[str, Any]] = []
                if is_blocked:
                    findings.append(
                        {
                            "finding_code": "BLOCKED_CANDIDATE",
                            "description": "Task currently appears in blocked candidate set.",
                            "severity": "warning",
                            "evidence_pointers": [
                                "execution_snapshot.task_detail.blocked_candidate_task_ids",
                                "execution_snapshot.blockage.primary_reason",
                            ],
                        }
                    )
                if frontier_stalled and frontier_task and frontier_task == critique_task:
                    findings.append(
                        {
                            "finding_code": "STALL_NO_PROGRESS_SIGNAL",
                            "description": "Execution frontier is stalled and this task is the frontier next candidate.",
                            "severity": "critical",
                            "evidence_pointers": [
                                "execution_snapshot.blockage.frontier_stalled",
                                "execution_snapshot.frontier_next_candidate",
                            ],
                        }
                    )
                if task_freshness_status in {"aging", "stale"}:
                    findings.append(
                        {
                            "finding_code": "TASK_FRESHNESS_DEGRADED",
                            "description": (
                                f"Task freshness status is {task_freshness_status}; snapshot signals are aging/stale."
                            ),
                            "severity": "warning" if task_freshness_status == "aging" else "critical",
                            "evidence_pointers": [
                                "execution_snapshot.task_detail.freshness",
                            ],
                        }
                    )
                if int(worker_state_obj.get("probe_overdue_worker_count") or 0) > 0:
                    findings.append(
                        {
                            "finding_code": "WORKER_PROBE_OVERDUE",
                            "description": "One or more workers are overdue on launch-readiness probe execution.",
                            "severity": "critical",
                            "evidence_pointers": [
                                "execution_snapshot.worker_state.probe_overdue_worker_count",
                                "execution_snapshot.worker_state.probe_execution_reason",
                            ],
                        }
                    )
                if visibility_state in {"degraded", "critical"}:
                    findings.append(
                        {
                            "finding_code": "VISIBILITY_SCORECARD_DEGRADED",
                            "description": (
                                f"Task visibility scorecard is {visibility_state} (score={visibility_score})."
                            ),
                            "severity": "warning" if visibility_state == "degraded" else "critical",
                            "evidence_pointers": [
                                "execution_snapshot.task_detail.active_task_cards[].visibility_scorecard",
                                "execution_snapshot.visibility_scorecard",
                            ],
                        }
                    )
                if ui_evidence_bundle_summary and ui_evidence_bundle_summary.get("status") == "linked":
                    critical_findings = int(ui_evidence_bundle_summary.get("critical_finding_count") or 0)
                    if critical_findings > 0:
                        findings.append(
                            {
                                "finding_code": "UI_EVIDENCE_CRITICAL_FINDINGS",
                                "description": (
                                    f"Linked UI evidence bundle reports {critical_findings} critical/high finding(s)."
                                ),
                                "severity": "warning",
                                "evidence_pointers": [
                                    f"ui_evidence_bundle:{ui_evidence_bundle_summary.get('bundle_path')}",
                                ],
                            }
                        )

                overlay_findings = (
                    component_consistency_overlay.get("findings")
                    if isinstance(component_consistency_overlay, dict)
                    and isinstance(component_consistency_overlay.get("findings"), list)
                    else []
                )
                promoted_overlay_finding_ids: List[str] = []
                ignored_overlay_findings: List[Dict[str, Any]] = []
                ignored_overlay_reason_counts: Dict[str, int] = {}
                for overlay_finding in overlay_findings[:12]:
                    if not isinstance(overlay_finding, dict):
                        continue
                    overlay_severity = str(overlay_finding.get("severity") or "low").strip().lower()
                    if overlay_severity in {"critical", "high"}:
                        mapped_severity = "critical"
                    elif overlay_severity == "medium":
                        mapped_severity = "warning"
                    else:
                        mapped_severity = "info"

                    consumption_action, consumption_reason = visibility_informed_overlay_consumption_decision(
                        overlay_severity=overlay_severity,
                        visibility_state=visibility_state,
                        visibility_score=visibility_score,
                        task_noise_signal_tags=task_noise_signal_tags,
                        task_suppressed=task_suppressed,
                        task_suppression_reasons=task_suppression_reasons,
                    )

                    rule_id = str(overlay_finding.get("audit_rule_id") or "consistency_rule").strip().upper()
                    overlay_finding_id_value = str(overlay_finding.get("finding_id") or "").strip()
                    inconsistency_paths = (
                        overlay_finding.get("inconsistent_paths")
                        if isinstance(overlay_finding.get("inconsistent_paths"), list)
                        else []
                    )

                    if consumption_action == "ignore":
                        if overlay_finding_id_value:
                            ignored_overlay_reason_counts[consumption_reason] = int(
                                ignored_overlay_reason_counts.get(consumption_reason) or 0
                            ) + 1
                            ignored_overlay_findings.append(
                                {
                                    "overlay_finding_id": overlay_finding_id_value,
                                    "overlay_rule_id": overlay_finding.get("audit_rule_id"),
                                    "severity": normalize_overlay_severity(overlay_severity),
                                    "ignore_reason": consumption_reason,
                                }
                            )
                        continue

                    if overlay_finding_id_value:
                        promoted_overlay_finding_ids.append(overlay_finding_id_value)

                    findings.append(
                        {
                            "finding_code": f"COMPONENT_CONSISTENCY_{rule_id}",
                            "description": str(overlay_finding.get("finding_text") or "").strip()
                            or "Component consistency audit overlay reported a cross-component inconsistency.",
                            "severity": mapped_severity,
                            "evidence_pointers": [
                                "component_consistency_audit_overlay.findings",
                                *[str(path or "").strip() for path in inconsistency_paths if str(path or "").strip()],
                            ],
                            "overlay_finding_id": overlay_finding_id_value or None,
                            "overlay_rule_id": overlay_finding.get("audit_rule_id"),
                        }
                    )

                promoted_overlay_finding_ids = unique_preserve(promoted_overlay_finding_ids)
                ignored_overlay_finding_ids = unique_preserve(
                    [
                        str(row.get("overlay_finding_id") or "").strip()
                        for row in ignored_overlay_findings
                        if isinstance(row, dict) and str(row.get("overlay_finding_id") or "").strip()
                    ]
                )
                b7_consumption_bridge = {
                    "visibility_state": visibility_state,
                    "visibility_score": visibility_score,
                    "task_noise_signal_tags": task_noise_signal_tags,
                    "task_suppressed": task_suppressed,
                    "task_suppression_reasons": task_suppression_reasons,
                    "policy": {
                        "always_promote_severity": ["critical", "high"],
                        "promote_medium_when_visibility_at_or_below": "degraded",
                        "ignore_low_when_visibility": "healthy",
                        "ignore_low_or_medium_when_task_suppressed": True,
                        "ignore_medium_when_noise_tags_present_and_visibility_healthy": True,
                    },
                    "promotion_count": len(promoted_overlay_finding_ids),
                    "ignored_count": len(ignored_overlay_finding_ids),
                    "promoted_finding_ids": promoted_overlay_finding_ids,
                    "ignored_finding_ids": ignored_overlay_finding_ids,
                    "ignored_reason_counts": ignored_overlay_reason_counts,
                    "ignored_findings": ignored_overlay_findings,
                }

                recommendations: List[Dict[str, Any]] = []
                if is_blocked or frontier_stalled:
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_INSPECT_EXECUTION_FRONTIER",
                            "description": "Inspect execution frontier blockage and unblock the candidate path.",
                            "confidence": 0.86,
                            "actionable": True,
                        }
                    )
                if int(worker_state_obj.get("probe_overdue_worker_count") or 0) > 0:
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_RECONCILE_LAUNCH_READINESS",
                            "description": "Run launch-readiness probe reconciliation before dispatching this task.",
                            "confidence": 0.9,
                            "actionable": True,
                        }
                    )
                if ui_evidence_bundle_summary and ui_evidence_bundle_summary.get("status") == "linked":
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_REVIEW_UI_EVIDENCE_BUNDLE",
                            "description": "Review linked UI evidence findings before changing operator surface behavior.",
                            "confidence": 0.74,
                            "actionable": True,
                        }
                    )
                if isinstance(component_consistency_overlay, dict) and int(component_consistency_overlay.get("finding_count") or 0) > 0:
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_REVIEW_COMPONENT_CONSISTENCY_AUDIT",
                            "description": "Resolve component-consistency overlay findings before operator-surface mutations.",
                            "confidence": 0.79,
                            "actionable": True,
                        }
                    )
                if (
                    federated_evidence
                    and str(federated_evidence.get("requested_task_id") or "").strip() == critique_task
                    and int((federated_evidence.get("summary") or {}).get("evidence_count") or 0) > 0
                ):
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_REVIEW_FEDERATED_EVIDENCE",
                            "description": "Review federated evidence context before mutating task handling.",
                            "confidence": 0.78,
                            "actionable": True,
                        }
                    )
                if not recommendations:
                    recommendations.append(
                        {
                            "recommendation_code": "ACTION_MONITOR",
                            "description": "Continue monitoring task signals; no immediate intervention indicated.",
                            "confidence": 0.62,
                            "actionable": False,
                        }
                    )

                significant_findings = [
                    row
                    for row in findings
                    if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() in {"warning", "critical"}
                ]
                quiet_success = (
                    len(significant_findings) == 0
                    and stall_risk < 0.45
                    and thrashing_risk < 0.4
                    and operator_burden < 0.45
                    and visibility_state == "healthy"
                )

                request_stamp = now_iso()
                critique_tasks[critique_task] = {
                    "last_requested_at": request_stamp,
                    "last_status": "healthy" if quiet_success else "generated",
                }

                if quiet_success:
                    critique_request.update(
                        {
                            "status": "healthy",
                            "message": "Task is healthy; no significant critique findings.",
                            "scoring": {
                                "stall_risk": stall_risk,
                                "thrashing_risk": thrashing_risk,
                                "operator_burden": operator_burden,
                                "efficiency": efficiency,
                            },
                            "visibility_scorecard": {
                                "task": {
                                    "task_id": critique_task,
                                    "score": visibility_score,
                                    "rating": visibility_state,
                                    "scorecard": task_card_snapshot.get("visibility_scorecard"),
                                },
                                "system": execution_snapshot.get("visibility_scorecard"),
                            },
                            "b7_consumption_bridge": b7_consumption_bridge,
                            "component_consistency_audit_overlay": (
                                {
                                    "status": component_consistency_overlay.get("status"),
                                    "finding_count": int(component_consistency_overlay.get("finding_count") or 0),
                                    "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0),
                                }
                                if isinstance(component_consistency_overlay, dict)
                                else None
                            ),
                        }
                    )
                else:
                    critique_packet = {
                        "packet_id": f"otc_{task_slug(critique_task)}_{request_stamp.replace(':', '').replace('-', '').replace('T', 't').replace('Z', 'z')}",
                        "task_id": critique_task,
                        "generated_at": request_stamp,
                        "version": "1.0",
                        "scoring": {
                            "stall_risk": stall_risk,
                            "thrashing_risk": thrashing_risk,
                            "operator_burden": operator_burden,
                            "efficiency": efficiency,
                        },
                        "visibility_scorecard": {
                            "task": {
                                "task_id": critique_task,
                                "score": visibility_score,
                                "rating": visibility_state,
                                "scorecard": task_card_snapshot.get("visibility_scorecard"),
                            },
                            "system": execution_snapshot.get("visibility_scorecard"),
                        },
                        "evidence_summary": {
                            "execution_snapshot_age_sec": mission_control_age_sec,
                            "log_entries_analyzed": 0,
                            "state_transitions_observed": int(blockage_obj.get("reason_count") or 0),
                            "failed_worker_dispatches": int(worker_state_obj.get("probe_overdue_worker_count") or 0),
                        },
                        "findings": findings,
                        "recommendations": recommendations,
                        "guardrails": {
                            "non_mutating": True,
                            "cooldown_sec": critique_cooldown_sec,
                            "minimum_age_sec": critique_min_age_sec,
                            "quiet_on_success": True,
                        },
                        "b7_consumption_bridge": b7_consumption_bridge,
                    }
                    if isinstance(component_consistency_overlay, dict):
                        critique_packet["component_consistency_audit_overlay"] = {
                            "schema": component_consistency_overlay.get("schema"),
                            "overlay_id": component_consistency_overlay.get("overlay_id"),
                            "status": component_consistency_overlay.get("status"),
                            "finding_count": int(component_consistency_overlay.get("finding_count") or 0),
                            "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0),
                            "findings": component_consistency_overlay.get("findings") if isinstance(component_consistency_overlay.get("findings"), list) else [],
                        }
                    if ui_evidence_bundle_summary and ui_evidence_bundle_summary.get("status") == "linked":
                        critique_packet["ui_evidence_bundle"] = {
                            "bundle_path": ui_evidence_bundle_summary.get("bundle_path"),
                            "bundle_id": ui_evidence_bundle_summary.get("bundle_id"),
                            "schema_version": ui_evidence_bundle_summary.get("schema_version"),
                            "finding_count": ui_evidence_bundle_summary.get("finding_count"),
                            "critical_finding_count": ui_evidence_bundle_summary.get("critical_finding_count"),
                        }
                    if (
                        federated_evidence
                        and str(federated_evidence.get("requested_task_id") or "").strip() == critique_task
                    ):
                        critique_packet["federated_evidence"] = federated_evidence

                    critique_packet_dir.mkdir(parents=True, exist_ok=True)
                    critique_packet_path = critique_packet_dir / f"{task_slug(critique_task)}.json"
                    atomic_write(critique_packet_path, critique_packet)

                    critique_tasks[critique_task]["last_packet_path"] = rel_path_for(critique_packet_path)
                    critique_tasks[critique_task]["last_generated_at"] = request_stamp
                    critique_request.update(
                        {
                            "status": "generated",
                            "message": "Critique packet generated.",
                            "packet_path": rel_path_for(critique_packet_path),
                            "b7_consumption_bridge": b7_consumption_bridge,
                            "packet": critique_packet,
                        }
                    )

                critique_index_payload = {
                    "schema": "clawd.operator_task_state_critique_index.v1",
                    "generated_at": request_stamp,
                    "tasks": critique_tasks,
                }
                atomic_write(critique_index_path, critique_index_payload)

b7_b8_packet_handshake = build_b7_b8_packet_handshake(
    ui_bundle_summary=ui_evidence_bundle_summary,
    component_consistency_overlay=component_consistency_overlay,
    critique_request=critique_request,
)

if isinstance(b7_b8_packet_handshake, dict):
    if isinstance(critique_request, dict) and isinstance(critique_request.get("packet"), dict):
        critique_packet = dict(critique_request.get("packet") or {})
        critique_packet["b7_b8_packet_handshake"] = b7_b8_packet_handshake
        critique_request["packet"] = critique_packet

        critique_packet_path_raw = str(critique_request.get("packet_path") or "").strip()
        if critique_packet_path_raw:
            critique_packet_path_obj = (root / critique_packet_path_raw).resolve()
            atomic_write(critique_packet_path_obj, critique_packet)

state_explainability_obj = (
    execution_snapshot.get("state_explainability")
    if isinstance(execution_snapshot.get("state_explainability"), dict)
    else {}
)
visibility_scorecard_obj = (
    execution_snapshot.get("visibility_scorecard")
    if isinstance(execution_snapshot.get("visibility_scorecard"), dict)
    else {}
)

verify_status_evidence_fresh_raw = headline.get("verify_status_evidence_fresh")
verify_status_evidence_fresh = (
    verify_status_evidence_fresh_raw if isinstance(verify_status_evidence_fresh_raw, bool) else None
)
verify_status_evidence_failure_reason = first_signal_reason(headline.get("verify_status_evidence_failure_reason"))
verify_layered_health_status = str(headline.get("verify_layered_health_status") or "unknown").strip().lower() or "unknown"
verify_layered_health_failure_reason = first_signal_reason(headline.get("verify_layered_health_failure_reason"))
verify_gate_predicted_blocker_reason = first_signal_reason(headline.get("verify_gate_predicted_blocker_reason"))
verify_probe_execution_gate_active_blocker = bool(headline.get("verify_probe_execution_gate_active_blocker") is True)
verify_worker_health_canary_gate_active_blocker = bool(headline.get("verify_worker_health_canary_gate_active_blocker") is True)
verify_stale_issue_active = any(
    isinstance(row, dict)
    and (
        "verify_status_evidence_stale" in str(row.get("id") or "")
        or "verify_status_evidence_stale" in str(row.get("reason") or "")
    )
    for row in issues
)

verify_posture_status = "healthy"
verify_posture_reason = "verify_posture_consistent"
if (
    verify_stale_issue_active
    or str(verify_status_evidence_failure_reason or "").strip().lower().find("stale") >= 0
    or verify_status_evidence_fresh is False
):
    verify_posture_status = "degraded"
    verify_posture_reason = (
        first_signal_reason(
            verify_status_evidence_failure_reason,
            "verify_status_evidence_stale",
        )
        or "verify_status_evidence_stale"
    )
elif (
    verify_layered_health_status in {"fail", "failed", "error", "blocked"}
    or verify_probe_execution_gate_active_blocker
    or verify_worker_health_canary_gate_active_blocker
):
    verify_posture_status = "degraded"
    verify_posture_reason = (
        first_signal_reason(
            verify_layered_health_failure_reason,
            verify_gate_predicted_blocker_reason,
            "verify_gate_blocked",
        )
        or "verify_gate_blocked"
    )

component_overlay_status = "neutral"
component_overlay_reason = "component_consistency_overlay_not_linked"
if isinstance(component_consistency_overlay, dict):
    critical_count = int(component_consistency_overlay.get("critical_finding_count") or 0)
    finding_count = int(component_consistency_overlay.get("finding_count") or 0)
    if critical_count > 0:
        component_overlay_status = "critical"
        component_overlay_reason = f"critical_findings={critical_count}"
    elif finding_count > 0:
        component_overlay_status = "degraded"
        component_overlay_reason = f"findings={finding_count}"
    else:
        component_overlay_status = "healthy"
        component_overlay_reason = "overlay_clean"

ui_bundle_status = "neutral"
ui_bundle_reason = "ui_bundle_not_linked"
if isinstance(ui_evidence_bundle_summary, dict):
    ui_bundle_status_value = str(ui_evidence_bundle_summary.get("status") or "invalid").strip().lower()
    ui_execution_link_status = str(ui_evidence_bundle_summary.get("execution_link_status") or "not_reported").strip().lower()
    ui_gate_status = str(ui_evidence_bundle_summary.get("validation_gate_status") or "not_reported").strip().lower()
    if ui_bundle_status_value != "linked":
        ui_bundle_status = "degraded"
        ui_bundle_reason = first_signal_reason(ui_evidence_bundle_summary.get("error"), "ui_bundle_invalid") or "ui_bundle_invalid"
    elif ui_execution_link_status == "missing":
        ui_bundle_status = "degraded"
        ui_bundle_reason = "ui_execution_evidence_link_missing"
    elif ui_gate_status == "failed":
        ui_bundle_status = "degraded"
        ui_bundle_reason = "ui_validation_gate_failed"
    else:
        ui_bundle_status = "healthy"
        ui_bundle_reason = "ui_bundle_linked"

state_explainability_status = "degraded"
state_explainability_reason = "state_explainability_missing"
dominant_state_value = str(state_explainability_obj.get("dominant_state") or "unknown").strip().lower() or "unknown"
dominant_reason_value = str(state_explainability_obj.get("dominant_reason") or "").strip()
dominant_triggers_value = unique_signal_tokens(state_explainability_obj.get("dominant_triggers"))
dominant_command_value = str(
    ((state_explainability_obj.get("safe_next_action") or {}).get("command") if isinstance(state_explainability_obj.get("safe_next_action"), dict) else "") or ""
).strip()
if (
    dominant_state_value in {"blocked", "stale", "ready"}
    and dominant_command_value
    and dominant_reason_value
    and dominant_triggers_value
):
    state_explainability_status = "healthy"
    state_explainability_reason = (
        f"dominant_state={dominant_state_value};trigger_count={len(dominant_triggers_value)}"
    )
elif dominant_state_value in {"blocked", "stale", "ready"} and dominant_command_value:
    state_explainability_reason = "dominant_reason_or_triggers_missing"

visibility_status = "degraded"
visibility_reason = "visibility_scorecard_missing"
if visibility_scorecard_obj:
    visibility_rating_value = str(visibility_scorecard_obj.get("rating") or "unknown").strip().lower() or "unknown"
    visibility_score_value = visibility_scorecard_obj.get("score")
    if visibility_rating_value in {"healthy", "degraded", "critical"} and isinstance(visibility_score_value, (int, float)):
        visibility_status = "healthy"
        visibility_reason = f"rating={visibility_rating_value}"

freshness_degradation_obj = (
    execution_snapshot.get("freshness_degradation_projection")
    if isinstance(execution_snapshot.get("freshness_degradation_projection"), dict)
    else {}
)
freshness_degradation_status = "degraded"
freshness_degradation_reason = "freshness_degradation_projection_missing"
if freshness_degradation_obj:
    if bool(freshness_degradation_obj.get("fail_closed") is True):
        freshness_degradation_status = "critical"
        freshness_degradation_reason = (
            first_signal_reason(
                freshness_degradation_obj.get("reason"),
                freshness_degradation_obj.get("reasons"),
            )
            or "freshness_degradation_projection_fail_closed"
        )
    elif bool(freshness_degradation_obj.get("degraded") is True):
        freshness_degradation_status = "degraded"
        freshness_degradation_reason = (
            first_signal_reason(freshness_degradation_obj.get("reason"))
            or "freshness_degradation_active"
        )
    else:
        freshness_degradation_status = "healthy"
        freshness_degradation_reason = "freshness_degradation_healthy"

source_of_truth_map_guard_status = str(source_of_truth_map_guard_summary.get("status") or "neutral").strip().lower()
if source_of_truth_map_guard_status not in {"healthy", "neutral", "degraded", "critical"}:
    source_of_truth_map_guard_status = "degraded"
source_of_truth_map_guard_reason = first_signal_reason(
    source_of_truth_map_guard_summary.get("reason"),
    source_of_truth_map_guard_summary.get("block_reason"),
) or "source_of_truth_map_guard_unknown"

convergence_checks = [
    {
        "check_id": "obs04_state_explainability",
        "status": state_explainability_status,
        "reason": state_explainability_reason,
    },
    {
        "check_id": "lt02_visibility_scorecard",
        "status": visibility_status,
        "reason": visibility_reason,
    },
    {
        "check_id": "exb05_freshness_degradation_projection_parity",
        "status": freshness_degradation_status,
        "reason": freshness_degradation_reason,
    },
    {
        "check_id": "pr03_ui_evidence_bundle",
        "status": ui_bundle_status,
        "reason": ui_bundle_reason,
    },
    {
        "check_id": "dsg06_component_consistency_overlay",
        "status": component_overlay_status,
        "reason": component_overlay_reason,
    },
    {
        "check_id": "ct06_verify_posture",
        "status": verify_posture_status,
        "reason": verify_posture_reason,
    },
    {
        "check_id": "cpl01_source_of_truth_map_guard",
        "status": source_of_truth_map_guard_status,
        "reason": source_of_truth_map_guard_reason,
    },
]

convergence_priority = {"healthy": 0, "neutral": 0, "degraded": 1, "critical": 2}
convergence_status = "converged"
convergence_reason = "operator_surface_signals_aligned"
max_convergence_priority = max([convergence_priority.get(str(row.get("status") or "healthy"), 0) for row in convergence_checks] or [0])
if max_convergence_priority >= 2:
    convergence_status = "diverged"
elif max_convergence_priority == 1:
    convergence_status = "degraded"

if convergence_status != "converged":
    first_problem = next(
        (
            row
            for row in convergence_checks
            if convergence_priority.get(str(row.get("status") or "healthy"), 0) == max_convergence_priority
        ),
        None,
        )
    if isinstance(first_problem, dict):
        convergence_reason = (
            first_signal_reason(
                first_problem.get("reason"),
                first_problem.get("check_id"),
            )
            or convergence_reason
        )

b8_consumption_obj = (
    b7_b8_packet_handshake.get("b8_consumption")
    if isinstance(b7_b8_packet_handshake, dict) and isinstance(b7_b8_packet_handshake.get("b8_consumption"), dict)
    else {}
)

operator_surface_convergence = {
    "schema": "clawd.operator_surface_convergence.v1",
    "status": convergence_status,
    "reason": convergence_reason,
    "checks": convergence_checks,
    "safe_next_action": action_brief_or_default(
        state_explainability_obj.get("safe_next_action") if isinstance(state_explainability_obj, dict) else None,
        default_action="refresh_operator_triage",
        default_command=f"bash {root / 'ops' / 'openclaw' / 'continuity.sh'} current --refresh --json",
        default_priority="p1",
    ),
    "projection": {
        "state_explainability": {
            "dominant_state": state_explainability_obj.get("dominant_state"),
            "dominant_reason": state_explainability_obj.get("dominant_reason"),
        },
        "freshness_degradation": {
            "status": freshness_degradation_obj.get("status") if freshness_degradation_obj else None,
            "degraded": bool(freshness_degradation_obj.get("degraded") is True),
            "fail_closed": bool(freshness_degradation_obj.get("fail_closed") is True),
            "reason": freshness_degradation_obj.get("reason") if freshness_degradation_obj else None,
            "parity": (
                freshness_degradation_obj.get("parity")
                if isinstance(freshness_degradation_obj.get("parity"), dict)
                else None
            ),
        },
        "visibility_scorecard": {
            "rating": visibility_scorecard_obj.get("rating") if visibility_scorecard_obj else None,
            "score": visibility_scorecard_obj.get("score") if visibility_scorecard_obj else None,
            "task_count": visibility_scorecard_obj.get("task_count") if visibility_scorecard_obj else None,
            "suppressed_task_count": visibility_scorecard_obj.get("suppressed_task_count") if visibility_scorecard_obj else None,
        },
        "noise_signal_summary": {
            "visible_issue_count": len(issues),
            "suppressed_issue_count": len(suppressed_issues),
            "visible_task_card_count": len(visible_task_cards),
            "suppressed_task_card_count": len(suppressed_task_cards),
        },
        "ui_evidence_bundle": {
            "status": ui_evidence_bundle_summary.get("status") if isinstance(ui_evidence_bundle_summary, dict) else "not_linked",
            "bundle_id": ui_evidence_bundle_summary.get("bundle_id") if isinstance(ui_evidence_bundle_summary, dict) else None,
            "execution_link_status": ui_evidence_bundle_summary.get("execution_link_status") if isinstance(ui_evidence_bundle_summary, dict) else None,
            "execution_evidence_link_count": ui_evidence_bundle_summary.get("execution_evidence_link_count") if isinstance(ui_evidence_bundle_summary, dict) else None,
            "validation_gate_status": ui_evidence_bundle_summary.get("validation_gate_status") if isinstance(ui_evidence_bundle_summary, dict) else None,
            "validation_gate_failure_count": ui_evidence_bundle_summary.get("validation_gate_failure_count") if isinstance(ui_evidence_bundle_summary, dict) else None,
        },
        "component_consistency_overlay": {
            "status": component_consistency_overlay.get("status") if isinstance(component_consistency_overlay, dict) else "not_linked",
            "finding_count": int(component_consistency_overlay.get("finding_count") or 0)
            if isinstance(component_consistency_overlay, dict)
            else 0,
            "critical_finding_count": int(component_consistency_overlay.get("critical_finding_count") or 0)
            if isinstance(component_consistency_overlay, dict)
            else 0,
        },
        "verify_posture": {
            "status": verify_posture_status,
            "reason": verify_posture_reason,
            "status_evidence_fresh": verify_status_evidence_fresh,
            "status_evidence_failure_reason": verify_status_evidence_failure_reason,
            "layered_health_status": verify_layered_health_status,
            "layered_health_failure_reason": verify_layered_health_failure_reason,
            "predicted_blocker_reason": verify_gate_predicted_blocker_reason,
            "probe_execution_gate_active_blocker": verify_probe_execution_gate_active_blocker,
            "worker_health_canary_gate_active_blocker": verify_worker_health_canary_gate_active_blocker,
            "stale_issue_active": verify_stale_issue_active,
        },
        "source_of_truth_map_guard": {
            "status": source_of_truth_map_guard_status,
            "reason": source_of_truth_map_guard_reason,
            "decision": source_of_truth_map_guard_summary.get("decision"),
            "block_reason": source_of_truth_map_guard_summary.get("block_reason"),
            "block_reasons": source_of_truth_map_guard_summary.get("block_reasons")
            if isinstance(source_of_truth_map_guard_summary.get("block_reasons"), list)
            else [],
            "generated_at": source_of_truth_map_guard_summary.get("generated_at"),
            "checker_version": source_of_truth_map_guard_summary.get("checker_version"),
            "path": source_of_truth_map_guard_summary.get("path"),
            "present": bool(source_of_truth_map_guard_summary.get("present") is True),
            "payload_valid": bool(source_of_truth_map_guard_summary.get("payload_valid") is True),
        },
        "b7_b8_packet_handshake": {
            "status": b7_b8_packet_handshake.get("status") if isinstance(b7_b8_packet_handshake, dict) else "not_linked",
            "promotion_count": int(b8_consumption_obj.get("promotion_count") or 0),
            "ignored_count": int(b8_consumption_obj.get("ignored_count") or 0),
        },
    },
}

triage_payload: Dict[str, Any] = {
    "schema": "clawd.operator_triage_console.v1",
    "generated_at": now_iso(),
    "source": {
        "mission_control_path": str(mission_export_path.relative_to(root)),
        "mission_control_generated_at": mission_payload.get("generated_at"),
        "mission_control_age_sec": age_sec(mission_payload.get("generated_at")),
    },
    "status": {
        "severity": overall_severity,
        "label": overall_label,
        "readiness": readiness,
        "mutation_gate": mutation_gate,
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "issue_count": len(issues),
        "suppressed_issue_count": len(suppressed_issues),
        "lane_count": len(lane_cards),
    },
    "execution_snapshot": execution_snapshot,
    "lanes": lane_cards,
    "issues": issues,
    "noise_signal": {
        "heuristics": {
            "show_stale": show_stale,
            "show_redundant": show_redundant,
            "verbose_state": verbose_state,
            "show_chatter": show_chatter,
            "show_recoverable_errors": show_recoverable_errors,
        },
        "thresholds": {
            "task_freshness_aging_after_sec": task_freshness_aging_after_sec,
            "task_freshness_stale_after_sec": task_freshness_stale_after_sec,
        },
        "summary": {
            "visible_issue_count": len(issues),
            "suppressed_issue_count": len(suppressed_issues),
            "visible_task_card_count": len(visible_task_cards),
            "suppressed_task_card_count": len(suppressed_task_cards),
        },
        "suppressed_issues": suppressed_issues,
    },
    "recommended_actions": recommended_actions,
    "source_of_truth_map_guard": source_of_truth_map_guard_summary,
}

triage_payload["operator_surface_convergence"] = operator_surface_convergence

if component_consistency_overlay is not None:
    triage_payload["component_consistency_audit_overlay"] = component_consistency_overlay

evidence_refs = [
    str(mission_export_path.relative_to(root)),
    "state/continuity/current.json",
    "state/continuity/latest/execution_frontier_ledger.json",
    "state/continuity/latest/execution_supervisor_dispatch_intent_latest.json",
    "state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
    "state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
    "state/continuity/session_topology_router/decisions.jsonl",
    "state/continuity/latest/load_shedding_decision.json",
    "state/continuity/latest/operator_triage_console.json",
]
evidence_refs.extend([str(x) for x in (mission_payload.get("evidence_refs") or []) if str(x or "").strip()][:30])
if ui_evidence_bundle_summary and str(ui_evidence_bundle_summary.get("bundle_path") or "").strip():
    evidence_refs.append(str(ui_evidence_bundle_summary.get("bundle_path")))
if isinstance(component_consistency_overlay, dict):
    overlay_artifact_path = str(component_consistency_overlay.get("artifact_path") or "").strip()
    if not overlay_artifact_path:
        overlay_artifact_path = str(component_consistency_overlay.get("ui_state_artifact_path") or "").strip()
    if overlay_artifact_path:
        evidence_refs.append(overlay_artifact_path)
if critique_request and str(critique_request.get("packet_path") or "").strip():
    evidence_refs.append(str(critique_request.get("packet_path")))
if bool(source_of_truth_map_guard_summary.get("present") is True):
    evidence_refs.append(str(source_of_truth_map_guard_summary.get("path") or rel_path_for(source_of_truth_map_guard_latest_path)))
if federated_evidence:
    for source in federated_evidence.get("sources_consulted") or []:
        if not isinstance(source, dict):
            continue
        source_path = str(source.get("path") or "").strip()
        if source_path:
            evidence_refs.append(source_path)
    for item in federated_evidence.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        source_ref = str(item.get("source") or "").strip()
        if source_ref.startswith("state/") or source_ref.startswith("memory/") or source_ref.startswith("reports/"):
            evidence_refs.append(source_ref)
triage_payload["evidence_refs"] = unique_preserve(evidence_refs)

if ui_evidence_bundle_summary is not None:
    triage_payload["ui_evidence_bundle"] = ui_evidence_bundle_summary
if federated_evidence is not None:
    triage_payload["federated_evidence"] = federated_evidence
if critique_request is not None:
    triage_payload["critique_request"] = critique_request
if b7_b8_packet_handshake is not None:
    triage_payload["b7_b8_packet_handshake"] = b7_b8_packet_handshake
    candidate_surface_obj = (
        b7_b8_packet_handshake.get("candidate_opportunity_surface")
        if isinstance(b7_b8_packet_handshake.get("candidate_opportunity_surface"), dict)
        else None
    )
    if candidate_surface_obj is not None:
        triage_payload["candidate_opportunity_surface"] = candidate_surface_obj

atomic_write(triage_export_path, triage_payload)

if json_out:
    print(json.dumps(triage_payload, ensure_ascii=False, indent=2))
    raise SystemExit(0)

icon = "🟢"
if overall_severity == "p0":
    icon = "🔴"
elif overall_severity == "p1":
    icon = "🟡"

print(f"{icon} OPERATOR TRIAGE {overall_label}")
print(
    f"readiness={readiness} mutation_gate={mutation_gate} "
    f"hard_blockers={hard_blockers} warnings={warnings}"
)
print()
print("Lanes:")
for lane in lane_cards:
    print(
        f"- [{lane.get('severity')}] {lane.get('lane')}: "
        f"{lane.get('summary')}"
    )

print()
print("Execution Snapshot:")
task_ids = execution_snapshot.get("active_task_ids") or []
task_ids_preview = ", ".join([str(x) for x in task_ids[:4]]) if task_ids else "none"
if task_ids and len(task_ids) > 4:
    task_ids_preview = f"{task_ids_preview} (+{len(task_ids) - 4} more)"
print(
    "- tasks: "
    f"count={int(execution_snapshot.get('active_task_count') or 0)} "
    f"running={int(execution_snapshot.get('reported_running_tasks') or 0)} "
    f"ids={task_ids_preview}"
)
worker_state = execution_snapshot.get("worker_state") or {}
print(
    "- workers: "
    f"demoted={int(worker_state.get('demoted_worker_count') or 0)} "
    f"restore_pending={int(worker_state.get('restore_pending_worker_count') or 0)} "
    f"probe_due_now={int(worker_state.get('probe_due_now_worker_count') or 0)} "
    f"probe_overdue={int(worker_state.get('probe_overdue_worker_count') or 0)} "
    f"fail_closed={bool(worker_state.get('fail_closed') is True)}"
)
blockage = execution_snapshot.get("blockage") or {}
blockage_reasons_preview = ", ".join([str(x) for x in (blockage.get("reasons") or [])[:3]]) or "none"
print(
    "- blockage: "
    f"primary={blockage.get('primary_reason') or 'none'} "
    f"blocked_candidates={int(blockage.get('blocked_candidate_count') or 0)} "
    f"reasons={blockage_reasons_preview}"
)
task_detail = execution_snapshot.get("task_detail") or {}
task_freshness = task_detail.get("freshness") if isinstance(task_detail.get("freshness"), dict) else {}
foreground_preview = ", ".join([str(x) for x in (task_detail.get("foreground_task_ids") or [])[:3]]) or "none"
background_blocked_preview = ", ".join([str(x) for x in (task_detail.get("background_blocked_task_ids") or [])[:3]]) or "none"
print(
    "- task_detail: "
    f"foreground={len(task_detail.get('foreground_task_ids') or [])}({foreground_preview}) "
    f"background_ready={len(task_detail.get('background_ready_task_ids') or [])} "
    f"background_blocked={len(task_detail.get('background_blocked_task_ids') or [])}({background_blocked_preview}) "
    f"freshness={task_freshness.get('status') or 'unknown'} "
    f"critique_available={len(task_detail.get('critique_available_task_ids') or [])} "
    f"visible_cards={len(task_detail.get('active_task_cards') or [])} "
    f"suppressed_cards={int(task_detail.get('suppressed_task_card_count') or 0)}"
)
worker_detail = execution_snapshot.get("worker_state_detail") or {}
worker_freshness = worker_detail.get("freshness") if isinstance(worker_detail.get("freshness"), dict) else {}
print(
    "- worker_detail: "
    f"status={worker_detail.get('status') or 'unknown'} "
    f"launch_readiness={worker_detail.get('launch_readiness_state') or 'unknown'} "
    f"severity={worker_detail.get('launch_readiness_severity_state') or 'none'} "
    f"demotion_priority={worker_detail.get('demotion_action_priority') or 'none'} "
    f"probe_priority={worker_detail.get('probe_execution_action_priority') or 'none'} "
    f"freshness={worker_freshness.get('status') or 'unknown'}"
)
inspection_workflow = execution_snapshot.get("inspection_workflow") if isinstance(execution_snapshot.get("inspection_workflow"), dict) else {}
freshness_workflow = inspection_workflow.get("freshness") if isinstance(inspection_workflow.get("freshness"), dict) else {}
blockage_workflow = inspection_workflow.get("blockage") if isinstance(inspection_workflow.get("blockage"), dict) else {}
freshness_next_command = (
    ((freshness_workflow.get("next_action") or {}).get("command") if isinstance(freshness_workflow.get("next_action"), dict) else None)
    or "none"
)
blockage_next_command = (
    ((blockage_workflow.get("next_action") or {}).get("command") if isinstance(blockage_workflow.get("next_action"), dict) else None)
    or "none"
)
print(
    "- inspect: "
    f"freshness_attention={bool(freshness_workflow.get('requires_attention') is True)} "
    f"blockage_blocked={bool(blockage_workflow.get('blocked') is True)} "
    f"freshness_next={freshness_next_command} "
    f"blockage_next={blockage_next_command}"
)
freshness_degradation_projection = (
    execution_snapshot.get("freshness_degradation_projection")
    if isinstance(execution_snapshot.get("freshness_degradation_projection"), dict)
    else {}
)
freshness_degradation_next_command = (
    ((freshness_degradation_projection.get("safe_next_action") or {}).get("command") if isinstance(freshness_degradation_projection.get("safe_next_action"), dict) else None)
    or "none"
)
freshness_degradation_parity = (
    freshness_degradation_projection.get("parity")
    if isinstance(freshness_degradation_projection.get("parity"), dict)
    else {}
)
print(
    "- freshness_degradation: "
    f"status={freshness_degradation_projection.get('status') or 'unknown'} "
    f"degraded={bool(freshness_degradation_projection.get('degraded') is True)} "
    f"fail_closed={bool(freshness_degradation_projection.get('fail_closed') is True)} "
    f"mission_parity={bool(freshness_degradation_parity.get('mission_posture_matches_projection') is True)} "
    f"next={freshness_degradation_next_command}"
)
task_state_surface = execution_snapshot.get("task_state_surface") if isinstance(execution_snapshot.get("task_state_surface"), dict) else {}
task_state_summary = task_state_surface.get("summary") if isinstance(task_state_surface.get("summary"), dict) else {}
task_state_next = (
    ((task_state_surface.get("safe_next_actions") or {}).get("inspect_blockage_frontier") or {}).get("command")
    if isinstance(task_state_surface.get("safe_next_actions"), dict)
    else None
)
print(
    "- task_state_surface: "
    f"state={task_state_summary.get('state') or 'unknown'} "
    f"blocked_work={int(task_state_summary.get('blocked_work_count') or 0)} "
    f"active_slices={int(task_state_summary.get('active_slice_count') or 0)} "
    f"freshness={task_state_summary.get('freshness_status') or 'unknown'} "
    f"worker={task_state_summary.get('worker_status') or 'unknown'} "
    f"requires_attention={bool(task_state_summary.get('requires_attention') is True)} "
    f"next={(str(task_state_next).strip() if str(task_state_next or '').strip() else 'none')}"
)
state_explainability = execution_snapshot.get("state_explainability") if isinstance(execution_snapshot.get("state_explainability"), dict) else {}
dominant_state = str(state_explainability.get("dominant_state") or "unknown").strip() or "unknown"
dominant_reason = str(state_explainability.get("dominant_reason") or "").strip() or "none"
dominant_next_command = (
    ((state_explainability.get("safe_next_action") or {}).get("command") if isinstance(state_explainability.get("safe_next_action"), dict) else None)
    or "none"
)
dominant_triggers = unique_signal_tokens(state_explainability.get("dominant_triggers"))
dominant_triggers_preview = ",".join(dominant_triggers[:3]) if dominant_triggers else "none"
print(
    "- explainability: "
    f"dominant={dominant_state} "
    f"reason={dominant_reason} "
    f"triggers={dominant_triggers_preview} "
    f"next={dominant_next_command}"
)
visibility_scorecard = execution_snapshot.get("visibility_scorecard") if isinstance(execution_snapshot.get("visibility_scorecard"), dict) else {}
print(
    "- visibility_scorecard: "
    f"score={visibility_scorecard.get('score') if visibility_scorecard else 'n/a'} "
    f"rating={visibility_scorecard.get('rating') if visibility_scorecard else 'unknown'} "
    f"tasks={int(visibility_scorecard.get('task_count') or 0)} "
    f"suppressed={int(visibility_scorecard.get('suppressed_task_count') or 0)}"
)
operator_surface_convergence_obj = (
    triage_payload.get("operator_surface_convergence")
    if isinstance(triage_payload.get("operator_surface_convergence"), dict)
    else {}
)
if operator_surface_convergence_obj:
    verify_projection = (
        (operator_surface_convergence_obj.get("projection") or {}).get("verify_posture")
        if isinstance((operator_surface_convergence_obj.get("projection") or {}).get("verify_posture"), dict)
        else {}
    )
    source_of_truth_map_guard_projection = (
        (operator_surface_convergence_obj.get("projection") or {}).get("source_of_truth_map_guard")
        if isinstance((operator_surface_convergence_obj.get("projection") or {}).get("source_of_truth_map_guard"), dict)
        else {}
    )
    freshness_degradation_projection = (
        (operator_surface_convergence_obj.get("projection") or {}).get("freshness_degradation")
        if isinstance((operator_surface_convergence_obj.get("projection") or {}).get("freshness_degradation"), dict)
        else {}
    )
    print(
        "- convergence: "
        f"status={operator_surface_convergence_obj.get('status') or 'unknown'} "
        f"reason={operator_surface_convergence_obj.get('reason') or 'unknown'} "
        f"verify={verify_projection.get('status') or 'unknown'} "
        f"freshness_degradation={freshness_degradation_projection.get('status') or 'unknown'} "
        f"map_guard={source_of_truth_map_guard_projection.get('status') or 'unknown'}"
    )

if issues:
    print()
    print("Top Issues:")
    for issue in issues[:8]:
        action = issue.get("recommended_action") or {}
        command = str(action.get("command") or "").strip()
        command_tail = f" | next={command}" if command else ""
        print(
            f"- [{issue.get('severity')}] {issue.get('id')}: "
            f"{issue.get('reason')}{command_tail}"
        )

noise_signal = triage_payload.get("noise_signal") if isinstance(triage_payload.get("noise_signal"), dict) else {}
noise_summary = noise_signal.get("summary") if isinstance(noise_signal.get("summary"), dict) else {}
if noise_summary:
    print()
    print(
        "Noise-vs-Signal: "
        f"visible_issues={int(noise_summary.get('visible_issue_count') or 0)} "
        f"suppressed_issues={int(noise_summary.get('suppressed_issue_count') or 0)} "
        f"visible_cards={int(noise_summary.get('visible_task_card_count') or 0)} "
        f"suppressed_cards={int(noise_summary.get('suppressed_task_card_count') or 0)}"
    )

if recommended_actions:
    print()
    print("Recommended Actions:")
    for action in recommended_actions:
        print(f"- [{action.get('priority')}] {action.get('action')}: {action.get('command')}")

if ui_evidence_bundle_summary is not None:
    print()
    print("UI Evidence Bundle:")
    print(
        f"- status={ui_evidence_bundle_summary.get('status')} "
        f"path={ui_evidence_bundle_summary.get('bundle_path')} "
        f"bundle_id={ui_evidence_bundle_summary.get('bundle_id') or 'n/a'}"
    )
    if ui_evidence_bundle_summary.get("error"):
        print(f"- error={ui_evidence_bundle_summary.get('error')}")

if component_consistency_overlay is not None:
    print()
    print("Component Consistency Audit Overlay:")
    print(
        f"- status={component_consistency_overlay.get('status')} "
        f"findings={int(component_consistency_overlay.get('finding_count') or 0)} "
        f"critical={int(component_consistency_overlay.get('critical_finding_count') or 0)}"
    )
    if str(component_consistency_overlay.get("reason") or "").strip():
        print(f"- reason={component_consistency_overlay.get('reason')}")

if federated_evidence is not None:
    fed_summary = federated_evidence.get("summary") if isinstance(federated_evidence.get("summary"), dict) else {}
    source_health = fed_summary.get("source_health") if isinstance(fed_summary.get("source_health"), dict) else {}
    print()
    print("Federated Evidence:")
    print(
        f"- task_id={federated_evidence.get('requested_task_id')} "
        f"query={fed_summary.get('retrieval_query') or federated_evidence.get('requested_task_id')} "
        f"status={federated_evidence.get('status')} "
        f"sources={int(fed_summary.get('source_count') or 0)} "
        f"evidence={int(fed_summary.get('evidence_count') or 0)} "
        f"latency_ms={int(fed_summary.get('latency_ms') or 0)}"
    )
    if source_health:
        print(
            "- source_health: "
            f"healthy={int(source_health.get('healthy_source_count') or 0)} "
            f"degraded={int(source_health.get('degraded_source_count') or 0)} "
            f"neutral={int(source_health.get('neutral_source_count') or 0)}"
        )
    degraded_reasons = fed_summary.get("degraded_reasons") if isinstance(fed_summary.get("degraded_reasons"), list) else []
    if degraded_reasons:
        preview = ", ".join([str(x) for x in degraded_reasons[:3]])
        if len(degraded_reasons) > 3:
            preview = f"{preview} (+{len(degraded_reasons) - 3} more)"
        print(f"- degraded_reasons={preview}")

if critique_request is not None:
    print()
    print("Critique Request:")
    print(
        f"- task_id={critique_request.get('requested_task_id')} "
        f"status={critique_request.get('status')}"
    )
    if str(critique_request.get("message") or "").strip():
        print(f"- message={critique_request.get('message')}")
    if critique_request.get("cooldown_sec_remaining") is not None:
        print(f"- cooldown_sec_remaining={critique_request.get('cooldown_sec_remaining')}")
    if critique_request.get("age_sec_remaining") is not None:
        print(f"- age_sec_remaining={critique_request.get('age_sec_remaining')}")
    if str(critique_request.get("packet_path") or "").strip():
        print(f"- packet_path={critique_request.get('packet_path')}")
    if isinstance(critique_request.get("packet"), dict):
        print()
        print("Critique Packet JSON:")
        print(json.dumps(critique_request.get("packet"), ensure_ascii=False, indent=2))

if b7_b8_packet_handshake is not None:
    b7_section = b7_b8_packet_handshake.get("b7_enrichment") if isinstance(b7_b8_packet_handshake.get("b7_enrichment"), dict) else {}
    b8_section = b7_b8_packet_handshake.get("b8_consumption") if isinstance(b7_b8_packet_handshake.get("b8_consumption"), dict) else {}
    candidate_surface = (
        b7_b8_packet_handshake.get("candidate_opportunity_surface")
        if isinstance(b7_b8_packet_handshake.get("candidate_opportunity_surface"), dict)
        else {}
    )
    candidate_summary = candidate_surface.get("summary") if isinstance(candidate_surface.get("summary"), dict) else {}
    print()
    print("B7↔B8 Handshake:")
    print(
        f"- status={b7_b8_packet_handshake.get('status')} "
        f"b7_findings={int(b7_section.get('finding_count') or 0)} "
        f"promoted={int(b8_section.get('promotion_count') or 0)} "
        f"ignored={int(b8_section.get('ignored_count') or 0)}"
    )
    if candidate_surface:
        print(
            "- candidate_surface: "
            f"status={candidate_surface.get('status')} "
            f"candidates={int(candidate_summary.get('candidate_count') or 0)} "
            f"now={int(candidate_summary.get('now_count') or 0)} "
            f"later={int(candidate_summary.get('later_count') or 0)} "
            f"reject={int(candidate_summary.get('reject_count') or 0)} "
            f"dropped={int(candidate_summary.get('bounded_drop_count') or 0)}"
        )

print()
print(f"triage_json=state/continuity/latest/{triage_export_path.name}")
PY
