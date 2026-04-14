#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CRON_JSON_PATH=""
SESSIONS_JSON_PATH=""
JSON_OUT=0
STRICT_EXIT=0
EXPECTED_NAMES_CSV="${OPENCLAW_NO_LLM_AUTHORITY_EXPECTED_NAMES:-continuity:backup-checkpoint-90m,continuity:stale-progress-45m,web-capture-scheduler-governance-watchdog,obsidian:hourly-canary,obsidian:vault-tick-hourly}"
SESSION_ACTIVE_MINUTES="${OPENCLAW_NO_LLM_AUTHORITY_SESSION_ACTIVE_MINUTES:-1440}"
SCHEMA_VERSION="openclaw.no_llm_watchdog_cron_authority_guard.summary.v1"
CLASSIFICATION_TAXONOMY_VERSION="openclaw.no_llm_watchdog_cron_authority_guard.classification.v1"
FAILURE_TAXONOMY_VERSION="openclaw.no_llm_watchdog_cron_authority_guard.failure_taxonomy.v1"

usage() {
  cat <<'EOF'
Usage: no_llm_watchdog_cron_authority_guard.sh [options]

Audit recurring watchdog/canary/checkpoint/scheduler-governance cron jobs so
LLM wrappers are not the authority path.

Contract for each expected enabled authority rail:
- sessionTarget = isolated
- payload.kind = agentTurn
- delivery.mode = none
- payload.message executes deterministic contract script
- payload.message says to always return NO_REPLY
- payload.message MUST NOT ask model to forward/decide BLOCKER routing

Options:
  --cron-json <path>       Use saved `openclaw cron list --json` payload
  --sessions-json <path>   Use saved `openclaw sessions --json` payload
  --expected-names <csv>   Override expected enabled authority job names
  --json                   Emit machine-readable summary payload
  --strict                 Exit non-zero when classification != ok
  -h, --help
EOF
}

_sanitize_inline_text() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

_is_gateway_connectivity_failure() {
  local raw="${1:-}"
  local lower
  lower="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"

  local -a tokens=(
    "gateway unavailable"
    "connection refused"
    "connection reset"
    "connect econn"
    "econnrefused"
    "ehostunreach"
    "enotfound"
    "network is unreachable"
    "temporary failure in name resolution"
    "timed out"
    "timeout"
    "socket hang up"
    "fetch failed"
    "service unavailable"
    "status 502"
    "status 503"
    "http 502"
    "http 503"
  )

  local token
  for token in "${tokens[@]}"; do
    if [[ "$lower" == *"$token"* ]]; then
      return 0
    fi
  done
  return 1
}

_emit_guard_summary_json() {
  local raw_payload="${1-}"
  if [[ -z "$raw_payload" ]]; then
    raw_payload='{}'
  fi

  python3 - "$raw_payload" "$EXPECTED_NAMES_CSV" "$SCHEMA_VERSION" "$CLASSIFICATION_TAXONOMY_VERSION" "$FAILURE_TAXONOMY_VERSION" <<'PY'
import json
import os
import sys
import time
from typing import Dict, List

raw = str(sys.argv[1] or "{}")
expected_names_csv = str(sys.argv[2] or "")
schema_version = str(sys.argv[3] or "openclaw.no_llm_watchdog_cron_authority_guard.summary.v1")
classification_taxonomy_version = str(sys.argv[4] or "openclaw.no_llm_watchdog_cron_authority_guard.classification.v1")
failure_taxonomy_version = str(sys.argv[5] or "openclaw.no_llm_watchdog_cron_authority_guard.failure_taxonomy.v1")


def dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        txt = str(item or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def expected_names() -> List[str]:
    rows: List[str] = []
    for token in expected_names_csv.split(","):
        name = token.strip()
        if not name or name in rows:
            continue
        rows.append(name)
    return rows


def command_for_name(name: str) -> str:
    if name in {"continuity:backup-checkpoint-90m", "continuity:stale-progress-45m"}:
        return "ops/openclaw/contract_no_nudge_continuity_watchdog.sh"
    if name == "core-roadmap-executor-idle-watchdog":
        return "ops/openclaw/contract_core_roadmap_floor_refill_watchdog.sh"
    if name in {"web-capture-scheduler-governance-watchdog", "web-capture-scheduler-governance"}:
        return "ops/openclaw/contract_web_capture_scheduler_governance_watchdog.sh"
    if name in {"obsidian:hourly-canary", "obsidian-vault-tick-hourly-canary"}:
        return "ops/openclaw/contract_obsidian_hourly_canary_watchdog.sh"
    if name == "obsidian:vault-tick-hourly":
        return "ops/openclaw/contract_obsidian_vault_tick_watchdog.sh"
    return ""


def timeout_floor_for_name(name: str):
    normalized = str(name or "").strip()
    if normalized == "core-roadmap-executor-idle-watchdog":
        raw = str(
            os.environ.get("OPENCLAW_CORE_ROADMAP_REFILL_TIMEOUT_FLOOR_SECONDS", "900")
        ).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        return 900
    return None


def runtime_family_ownership_projection_for_name(name: str) -> Dict[str, str]:
    normalized = str(name or "").strip().lower()
    if normalized in {"continuity:backup-checkpoint-90m", "continuity:stale-progress-45m", "core-roadmap-executor-idle-watchdog"}:
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "source": "name_exact:continuity_expected_rails",
        }
    if normalized in {"web-capture-scheduler-governance-watchdog", "web-capture-scheduler-governance"}:
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "source": "name_exact:web_capture_expected_rails",
        }
    if normalized in {
        "obsidian:hourly-canary",
        "obsidian-vault-tick-hourly-canary",
        "obsidian:vault-tick-hourly",
    }:
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "source": "name_exact:obsidian_expected_rails",
        }
    if normalized.startswith("continuity:"):
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "source": "name_prefix:continuity",
        }
    if normalized.startswith("web-capture"):
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "source": "name_prefix:web_capture",
        }
    if normalized.startswith("obsidian"):
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "source": "name_prefix:obsidian",
        }
    return {
        "family": "unknown",
        "action_lane": "unknown",
        "recovery_owner": "operator_general_triage",
        "routing_hint": "route:operator_general_triage",
        "source": "fallback:unknown_family",
    }


def reason_codes(reason: str) -> List[str]:
    lower = str(reason or "").lower()
    if lower.startswith("sessiontarget="):
        return ["policy:session_target_not_isolated"]
    if lower.startswith("payload.kind="):
        return ["policy:payload_kind_not_agent_turn"]
    if lower.startswith("delivery.mode="):
        return ["policy:delivery_mode_not_none"]
    if "missing expected deterministic command" in lower:
        return ["policy:missing_expected_deterministic_command"]
    if "missing no_reply contract" in lower:
        return ["policy:missing_no_reply_contract"]
    if "model authority forwarding instructions present" in lower:
        return ["policy:model_authority_forwarding_present", "policy:model_backed_authority_path"]
    if lower.startswith("payload.timeoutseconds="):
        return ["policy:payload_timeout_seconds_below_floor"]
    if "missing enabled expected authority job" in lower:
        return ["policy:missing_enabled_expected_authority_job"]
    if "duplicate enabled expected authority jobs" in lower:
        return ["policy:duplicate_enabled_expected_authority_jobs"]
    return ["policy:unknown_violation"]


def parse_json_payload(text: str):
    raw_text = str(text or "")
    parse_error = ""

    try:
        return json.loads(raw_text), parse_error
    except Exception as exc:
        parse_error = str(exc)

    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw_text.lstrip())
        return obj, ""
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1]), ""
        except Exception:
            pass

    for line in reversed(raw_text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate), ""
        except Exception:
            continue

    return None, parse_error


def normalize_status(raw: object) -> str:
    return str(raw or "").strip().lower().replace("-", "_")


def parse_nonnegative_int(raw: object):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        val = int(txt)
    except Exception:
        return None
    return val if val >= 0 else None


def runtime_family_ownership_for_name(name: str):
    normalized = str(name or "").strip().lower()
    if normalized in {"continuity:backup-checkpoint-90m", "continuity:stale-progress-45m", "core-roadmap-executor-idle-watchdog"}:
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "ownership_source": "name_exact:continuity_expected_rails",
        }
    if normalized in {"web-capture-scheduler-governance-watchdog", "web-capture-scheduler-governance"}:
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "ownership_source": "name_exact:web_capture_expected_rails",
        }
    if normalized in {"obsidian:hourly-canary", "obsidian-vault-tick-hourly-canary", "obsidian:vault-tick-hourly"}:
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "ownership_source": "name_exact:obsidian_expected_rails",
        }
    if normalized.startswith("continuity:"):
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "ownership_source": "name_prefix:continuity",
        }
    if normalized.startswith("web-capture"):
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "ownership_source": "name_prefix:web_capture",
        }
    if normalized.startswith("obsidian"):
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "ownership_source": "name_prefix:obsidian",
        }
    return {
        "family": "unknown",
        "action_lane": "unknown",
        "recovery_owner": "operator_general_triage",
        "routing_hint": "route:operator_general_triage",
        "ownership_source": "fallback:unknown_family",
    }


def historical_session_residue_playbook_hints(family: str) -> List[str]:
    normalized_family = str(family or "").strip().lower()
    family_mapping = {
        "continuity": [
            "runtime:playbook_continuity_run_no_nudge_authority_watchdog_contract",
            "runtime:playbook_core_roadmap_floor_refill_loop_contract",
            "runtime:playbook_continuity_validate_checkpoint_and_stale_progress_sessions",
        ],
        "web_capture": [
            "runtime:playbook_web_capture_run_scheduler_governance_watchdog_contract",
            "runtime:playbook_web_capture_reconcile_scheduler_manifest_and_capture_backlog",
        ],
        "obsidian": [
            "runtime:playbook_obsidian_run_hourly_canary_and_vault_tick_contracts",
            "runtime:playbook_obsidian_verify_vault_materialization_and_memory_sync",
        ],
    }

    hints = list(family_mapping.get(normalized_family, []))
    if not hints:
        hints = ["runtime:playbook_operator_general_triage_collect_runtime_artifacts"]
    hints.append("runtime:playbook_historical_session_residue_verify_current_rail_health_and_keep_low_urgency")
    return dedupe(hints)


def parse_epoch_ms(raw: object):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        val = int(float(txt))
    except Exception:
        return None
    return val if val >= 0 else None


def runtime_root_cause_codes(last_error_reason: object, last_error: object) -> List[str]:
    text = f"{str(last_error_reason or '').strip()} {str(last_error or '').strip()}".strip().lower()
    if not text:
        return []

    token_groups = [
        (
            "runtime:root_cause_provider_quota",
            [
                "failed_provider_quota",
                "insufficient_quota",
                "quota",
                "billing",
                "credit",
                "resource exhausted",
            ],
        ),
        (
            "runtime:root_cause_provider_rate_limited",
            ["rate limit", "too many requests", "http 429", "status 429", "429"],
        ),
        (
            "runtime:root_cause_timeout",
            ["timeout", "timed out", "deadline exceeded", "etimedout", "request timed out"],
        ),
        (
            "runtime:root_cause_connectivity",
            [
                "gateway unavailable",
                "connection refused",
                "connection reset",
                "econnrefused",
                "ehostunreach",
                "enotfound",
                "network is unreachable",
                "service unavailable",
                "socket hang up",
                "fetch failed",
                "http 502",
                "status 502",
                "http 503",
                "status 503",
            ],
        ),
        (
            "runtime:root_cause_provider_auth",
            ["unauthorized", "forbidden", "invalid api key", "authentication", "permission denied", "http 401", "http 403"],
        ),
        (
            "runtime:root_cause_missing_dependency",
            ["no such file", "command not found", "module not found", "enoent", "not found"],
        ),
    ]

    for code, tokens in token_groups:
        if any(token in text for token in tokens):
            return [code]

    return ["runtime:root_cause_unclassified"]


def runtime_root_cause_remediation_hints(root_cause_codes: List[str]) -> List[str]:
    mapping = {
        "runtime:root_cause_provider_quota": [
            "runtime:remediation_review_provider_quota_or_billing",
        ],
        "runtime:root_cause_provider_rate_limited": [
            "runtime:remediation_apply_backoff_and_reduce_runtime_concurrency",
        ],
        "runtime:root_cause_timeout": [
            "runtime:remediation_reduce_payload_scope_or_raise_timeout_budget",
        ],
        "runtime:root_cause_connectivity": [
            "runtime:remediation_check_gateway_network_and_upstream_status",
        ],
        "runtime:root_cause_provider_auth": [
            "runtime:remediation_rotate_or_refresh_provider_auth_credentials",
        ],
        "runtime:root_cause_missing_dependency": [
            "runtime:remediation_restore_missing_dependency_or_executable",
        ],
        "runtime:root_cause_unclassified": [
            "runtime:remediation_collect_failure_evidence_and_route_operator_triage",
        ],
    }

    hints: List[str] = []
    for code in root_cause_codes:
        hints.extend(mapping.get(str(code or "").strip(), []))

    return dedupe(hints)


def runtime_family_playbook_hints(
    family: str,
    row_recovery_state: str,
    row_retryable=None,
) -> List[str]:
    normalized_family = str(family or "").strip().lower()
    normalized_recovery_state = str(row_recovery_state or "").strip().lower()

    family_mapping = {
        "continuity": [
            "runtime:playbook_continuity_run_no_nudge_authority_watchdog_contract",
            "runtime:playbook_core_roadmap_floor_refill_loop_contract",
            "runtime:playbook_continuity_validate_checkpoint_and_stale_progress_sessions",
        ],
        "web_capture": [
            "runtime:playbook_web_capture_run_scheduler_governance_watchdog_contract",
            "runtime:playbook_web_capture_reconcile_scheduler_manifest_and_capture_backlog",
        ],
        "obsidian": [
            "runtime:playbook_obsidian_run_hourly_canary_and_vault_tick_contracts",
            "runtime:playbook_obsidian_verify_vault_materialization_and_memory_sync",
        ],
    }

    hints = list(family_mapping.get(normalized_family, []))
    if not hints:
        hints = [
            "runtime:playbook_operator_general_triage_collect_runtime_artifacts",
        ]

    if row_retryable is True:
        hints.append("runtime:playbook_recovery_wait_for_next_scheduled_retry_and_verify_clearance")
    elif row_retryable is False:
        hints.append("runtime:playbook_recovery_intervention_required_route_owner_now")
    elif normalized_recovery_state == "retrying":
        hints.append("runtime:playbook_recovery_wait_for_next_scheduled_retry_and_verify_clearance")
    else:
        hints.append("runtime:playbook_recovery_intervention_required_route_owner_now")

    return dedupe(hints)


def runtime_escalation_policy_projection(
    has_runtime_failures: bool,
    runtime_root_causes: List[str],
    runtime_recovery_state: str,
) -> Dict[str, object]:
    if not has_runtime_failures:
        return {
            "state": "none",
            "policy": "none",
            "urgency": "none",
            "source": "none",
            "retryable": None,
            "failure_category": None,
        }

    root_set = set(dedupe(runtime_root_causes))
    if "runtime:root_cause_provider_quota" in root_set:
        return {
            "state": "projected",
            "policy": "provider_quota_billing_intervention_required",
            "urgency": "billing_intervention_required",
            "source": "runtime:root_cause_provider_quota",
            "retryable": False,
            "failure_category": "runtime_provider_quota_intervention_required",
        }

    if "runtime:root_cause_provider_auth" in root_set:
        return {
            "state": "projected",
            "policy": "provider_auth_intervention_required",
            "urgency": "operator_intervention_required",
            "source": "runtime:root_cause_provider_auth",
            "retryable": False,
            "failure_category": "runtime_provider_auth_intervention_required",
        }

    if "runtime:root_cause_provider_rate_limited" in root_set:
        return {
            "state": "projected",
            "policy": "provider_rate_limit_backoff_retry_later",
            "urgency": "backoff_retry_later",
            "source": "runtime:root_cause_provider_rate_limited",
            "retryable": True,
            "failure_category": "runtime_provider_rate_limit_backoff_retry_later",
        }

    if runtime_recovery_state == "retrying":
        return {
            "state": "projected",
            "policy": "runtime_retry_scheduled_backoff_retry_later",
            "urgency": "backoff_retry_later",
            "source": "runtime_recovery_state:retrying",
            "retryable": True,
            "failure_category": "runtime_recovery_retrying",
        }

    return {
        "state": "projected",
        "policy": "runtime_operator_intervention_required",
        "urgency": "operator_intervention_required",
        "source": f"runtime_recovery_state:{runtime_recovery_state or 'unknown'}",
        "retryable": False,
        "failure_category": "runtime_operator_recovery_required",
    }


def runtime_row_recovery_state(row_codes: List[str]) -> str:
    row_set = set(dedupe(row_codes))
    if "runtime:enabled_authority_retry_scheduled" in row_set:
        return "retrying"
    if "runtime:enabled_authority_intervention_required" in row_set:
        return "intervention_required"
    return "intervention_required"


def runtime_row_escalation_policy_projection(
    row_root_causes: List[str],
    row_recovery_state: str,
) -> Dict[str, object]:
    root_set = set(dedupe(row_root_causes))
    if "runtime:root_cause_provider_quota" in root_set:
        return {
            "policy": "provider_quota_billing_intervention_required",
            "urgency": "billing_intervention_required",
            "source": "runtime:root_cause_provider_quota",
            "retryable": False,
            "failure_category": "runtime_provider_quota_intervention_required",
        }

    if "runtime:root_cause_provider_auth" in root_set:
        return {
            "policy": "provider_auth_intervention_required",
            "urgency": "operator_intervention_required",
            "source": "runtime:root_cause_provider_auth",
            "retryable": False,
            "failure_category": "runtime_provider_auth_intervention_required",
        }

    if "runtime:root_cause_provider_rate_limited" in root_set:
        return {
            "policy": "provider_rate_limit_backoff_retry_later",
            "urgency": "backoff_retry_later",
            "source": "runtime:root_cause_provider_rate_limited",
            "retryable": True,
            "failure_category": "runtime_provider_rate_limit_backoff_retry_later",
        }

    if row_recovery_state == "retrying":
        return {
            "policy": "runtime_retry_scheduled_backoff_retry_later",
            "urgency": "backoff_retry_later",
            "source": "runtime_recovery_state:retrying",
            "retryable": True,
            "failure_category": "runtime_recovery_retrying",
        }

    return {
        "policy": "runtime_operator_intervention_required",
        "urgency": "operator_intervention_required",
        "source": f"runtime_recovery_state:{row_recovery_state or 'unknown'}",
        "retryable": False,
        "failure_category": "runtime_operator_recovery_required",
    }


def sanitize_reason_excerpt(raw: object) -> str:
    txt = str(raw or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    txt = " ".join(txt.split())
    txt = txt.replace(";", ",")
    return txt[:240]


summary: Dict[str, object]
obj, parse_error = parse_json_payload(raw)
if obj is None:
    summary = {
        "ok": False,
        "classification": "cron_contract_drift",
        "classification_bucket": "contract",
        "error": "invalid_cron_json",
        "detail": parse_error,
        "checked": 0,
        "checked_jobs": [],
        "coverage": {
            "expected_enabled_authority_names": expected_names(),
            "missing_enabled_authority_names": expected_names(),
            "duplicate_enabled_authority_names": [],
            "enabled_counts": {name: 0 for name in expected_names()},
        },
        "violations": [
            {
                "id": "",
                "name": "payload",
                "reasons": ["invalid cron json payload"],
                "codes": ["contract:invalid_cron_json"],
                "bucket": "contract",
            }
        ],
    }
elif not isinstance(obj, dict):
    summary = {
        "ok": False,
        "classification": "cron_contract_drift",
        "classification_bucket": "contract",
        "error": "invalid_cron_json",
        "detail": "cron payload root must be object",
        "checked": 0,
        "checked_jobs": [],
        "coverage": {
            "expected_enabled_authority_names": expected_names(),
            "missing_enabled_authority_names": expected_names(),
            "duplicate_enabled_authority_names": [],
            "enabled_counts": {name: 0 for name in expected_names()},
        },
        "violations": [
            {
                "id": "",
                "name": "payload",
                "reasons": ["invalid cron json payload"],
                "codes": ["contract:invalid_cron_json"],
                "bucket": "contract",
            }
        ],
    }
else:
    jobs = obj.get("jobs") if isinstance(obj, dict) else []
    if not isinstance(jobs, list):
        jobs = []

    expected = expected_names()
    expected_set = set(expected)
    enabled_counts = {name: 0 for name in expected}
    enabled_ids = {name: [] for name in expected}

    checked_jobs = []
    violations = []
    policy_codes: List[str] = []
    runtime_codes: List[str] = []
    runtime_failure_provenance_rows: List[Dict[str, object]] = []
    runtime_escalation_rows: List[Dict[str, object]] = []
    runtime_recovered_codes: List[str] = []
    recovery_annotations: List[Dict[str, object]] = []
    runtime_retrying_names: List[str] = []
    runtime_intervention_required_names: List[str] = []
    runtime_historical_recovered_names: List[str] = []
    now_ms = int(time.time() * 1000)
    unhealthy_runtime_statuses = {
        "error",
        "failed",
        "fail",
        "aborted",
        "timeout",
        "timed_out",
        "crashed",
        "cancelled",
        "canceled",
    }

    authority_bad_tokens = [
        "if output starts with \"blocker:\"",
        "send that exact line",
        "forward the exact line",
        "message tool",
        "if output is \"no_reply\"",
    ]

    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or "").strip()
        if name not in expected_set:
            continue
        if not bool(job.get("enabled", False)):
            continue

        job_id = str(job.get("id") or "")
        payload = job.get("payload") or {}
        delivery = job.get("delivery") or {}
        state = job.get("state") if isinstance(job.get("state"), dict) else {}

        session_target = str(job.get("sessionTarget") or "").strip()
        payload_kind = str(payload.get("kind") or "").strip()
        delivery_mode = str(delivery.get("mode") or "").strip()
        message = str(payload.get("message") or "")
        model = str(payload.get("model") or "").strip()
        timeout_seconds = parse_nonnegative_int(payload.get("timeoutSeconds"))
        last_status = normalize_status(state.get("lastStatus") or state.get("status"))
        last_run_status = normalize_status(state.get("lastRunStatus"))
        consecutive_errors = parse_nonnegative_int(state.get("consecutiveErrors"))
        next_run_at_ms = parse_epoch_ms(state.get("nextRunAtMs"))
        last_run_at_ms = parse_epoch_ms(state.get("lastRunAtMs"))
        last_error = str(state.get("lastError") or "").strip() or None
        last_error_reason = str(state.get("lastErrorReason") or "").strip() or None

        enabled_counts[name] = int(enabled_counts.get(name) or 0) + 1
        enabled_ids.setdefault(name, []).append(job_id)

        checked_jobs.append(
            {
                "id": job_id,
                "name": name,
                "sessionTarget": session_target,
                "payloadKind": payload_kind,
                "deliveryMode": delivery_mode,
                "model": model or None,
                "payloadTimeoutSeconds": timeout_seconds,
                "authorityMode": "llm_executor_non_authoritative" if model else "no_llm_native",
                "lastStatus": last_status or None,
                "lastRunStatus": last_run_status or None,
                "consecutiveErrors": consecutive_errors,
                "lastRunAtMs": last_run_at_ms,
                "nextRunAtMs": next_run_at_ms,
                "lastErrorReason": last_error_reason,
                "lastError": last_error,
            }
        )

        expected_command = command_for_name(name)
        message_lower = message.lower()
        reasons = []

        if session_target != "isolated":
            reasons.append(f"sessionTarget={session_target or '<empty>'}")
        if payload_kind != "agentTurn":
            reasons.append(f"payload.kind={payload_kind or '<empty>'}")
        if delivery_mode != "none":
            reasons.append(f"delivery.mode={delivery_mode or '<empty>'}")
        if expected_command and expected_command not in message:
            reasons.append(f"missing expected deterministic command: {expected_command}")
        if "reply exactly: no_reply" not in message_lower:
            reasons.append("payload.message missing NO_REPLY contract")

        timeout_floor = timeout_floor_for_name(name)
        if isinstance(timeout_floor, int):
            if not isinstance(timeout_seconds, int):
                reasons.append(
                    f"payload.timeoutSeconds=<missing>; required_min={timeout_floor}"
                )
            elif timeout_seconds < timeout_floor:
                reasons.append(
                    f"payload.timeoutSeconds={timeout_seconds}; required_min={timeout_floor}"
                )

        if any(token in message_lower for token in authority_bad_tokens):
            reasons.append("model authority forwarding instructions present")

        if reasons:
            codes = dedupe([code for reason in reasons for code in reason_codes(reason)])
            policy_codes.extend(codes)
            violations.append(
                {
                    "id": job_id,
                    "name": name,
                    "reasons": reasons,
                    "codes": codes,
                    "bucket": "policy",
                }
            )

        runtime_reasons: List[str] = []
        runtime_row_codes: List[str] = []
        has_unhealthy_status = (last_status and last_status in unhealthy_runtime_statuses) or (
            last_run_status and last_run_status in unhealthy_runtime_statuses
        )
        has_positive_consecutive_errors = isinstance(consecutive_errors, int) and consecutive_errors > 0

        if has_unhealthy_status:
            runtime_reasons.append(
                "enabled authority runtime unhealthy: "
                f"lastStatus={last_status or '<empty>'}; "
                f"lastRunStatus={last_run_status or '<empty>'}; "
                f"lastErrorReason={last_error_reason or '<none>'}"
            )
            runtime_row_codes.append("runtime:enabled_authority_last_status_unhealthy")

            if isinstance(next_run_at_ms, int) and next_run_at_ms > now_ms:
                runtime_reasons.append(
                    f"enabled authority recovery projection=retrying; nextRunAtMs={next_run_at_ms}; nowMs={now_ms}"
                )
                runtime_row_codes.append("runtime:enabled_authority_retry_scheduled")
                runtime_retrying_names.append(name)
            else:
                runtime_reasons.append(
                    f"enabled authority recovery projection=intervention_required; nextRunAtMs={next_run_at_ms if isinstance(next_run_at_ms, int) else '<missing_or_due>'}; nowMs={now_ms}"
                )
                runtime_row_codes.append("runtime:enabled_authority_intervention_required")
                runtime_intervention_required_names.append(name)

            if has_positive_consecutive_errors:
                runtime_reasons.append(f"enabled authority consecutiveErrors={consecutive_errors}")
                runtime_row_codes.append("runtime:enabled_authority_consecutive_errors_positive")

            root_cause_codes = runtime_root_cause_codes(last_error_reason, last_error)
            if root_cause_codes:
                runtime_row_codes.extend(root_cause_codes)
                runtime_reasons.append(
                    "enabled authority runtime root-cause classification=" + ",".join(root_cause_codes)
                )

            reason_source = "unavailable"
            reason_excerpt = ""
            if last_error_reason:
                reason_source = "state.lastErrorReason"
                reason_excerpt = sanitize_reason_excerpt(last_error_reason)
            elif last_error:
                reason_source = "state.lastError"
                reason_excerpt = sanitize_reason_excerpt(last_error)

            runtime_failure_provenance_rows.append(
                {
                    "id": job_id,
                    "name": name,
                    "lastRunAtMs": last_run_at_ms,
                    "reasonSource": reason_source,
                    "reasonExcerpt": reason_excerpt,
                }
            )

        elif has_positive_consecutive_errors:
            runtime_historical_recovered_names.append(name)
            recovered_code = "runtime:enabled_authority_historical_failure_recovered"
            runtime_recovered_codes.append(recovered_code)
            recovery_annotations.append(
                {
                    "id": job_id,
                    "name": name,
                    "bucket": "runtime_recovered",
                    "state": "historical_recovered",
                    "reasons": [
                        f"enabled authority historical failure recovered: lastStatus={last_status or '<empty>'}; lastRunStatus={last_run_status or '<empty>'}; consecutiveErrors={consecutive_errors}"
                    ],
                    "codes": [recovered_code],
                }
            )

        if runtime_reasons:
            row_codes = dedupe(runtime_row_codes or ["runtime:enabled_authority_runtime_unhealthy"])
            row_root_causes = dedupe([code for code in row_codes if code.startswith("runtime:root_cause_")])
            row_recovery_state = runtime_row_recovery_state(row_codes)
            row_escalation_policy = runtime_row_escalation_policy_projection(
                row_root_causes=row_root_causes,
                row_recovery_state=row_recovery_state,
            )
            row_ownership = runtime_family_ownership_projection_for_name(name)
            row_playbook_hints = runtime_family_playbook_hints(
                family=str(row_ownership.get("family") or "unknown"),
                row_recovery_state=row_recovery_state,
                row_retryable=row_escalation_policy.get("retryable"),
            )
            runtime_escalation_rows.append(
                {
                    "id": job_id,
                    "name": name,
                    "recovery_state": row_recovery_state,
                    "root_causes": row_root_causes,
                    "policy": str(row_escalation_policy.get("policy") or "runtime_operator_intervention_required"),
                    "urgency": str(row_escalation_policy.get("urgency") or "operator_intervention_required"),
                    "source": str(row_escalation_policy.get("source") or "runtime_recovery_state:unknown"),
                    "retryable": bool(row_escalation_policy.get("retryable") is True),
                    "failure_category": str(
                        row_escalation_policy.get("failure_category") or "runtime_operator_recovery_required"
                    ),
                    "family": str(row_ownership.get("family") or "unknown"),
                    "action_lane": str(row_ownership.get("action_lane") or "unknown"),
                    "recovery_owner": str(row_ownership.get("recovery_owner") or "operator_general_triage"),
                    "routing_hint": str(row_ownership.get("routing_hint") or "route:operator_general_triage"),
                    "ownership_source": str(row_ownership.get("source") or "fallback:unknown_family"),
                    "playbook_hints": row_playbook_hints,
                }
            )
            runtime_codes.extend(row_codes)
            violations.append(
                {
                    "id": job_id,
                    "name": name,
                    "reasons": runtime_reasons,
                    "codes": row_codes,
                    "bucket": "runtime",
                }
            )

    missing_names = []
    duplicate_names = []
    for name in expected:
        count = int(enabled_counts.get(name) or 0)
        if count <= 0:
            missing_names.append(name)
            code = "policy:missing_enabled_expected_authority_job"
            policy_codes.append(code)
            violations.append(
                {
                    "id": "",
                    "name": name,
                    "reasons": ["missing enabled expected authority job"],
                    "codes": [code],
                    "bucket": "policy",
                }
            )
        elif count > 1:
            duplicate_names.append(name)
            ids = [row for row in (enabled_ids.get(name) or []) if row]
            code = "policy:duplicate_enabled_expected_authority_jobs"
            policy_codes.append(code)
            violations.append(
                {
                    "id": ",".join(ids[:5]),
                    "name": name,
                    "reasons": [
                        f"duplicate enabled expected authority jobs; count={count}; ids={','.join(ids[:5]) or 'none'}"
                    ],
                    "codes": [code],
                    "bucket": "policy",
                }
            )

    policy_codes = dedupe(policy_codes)
    runtime_codes = dedupe(runtime_codes)
    has_runtime_failures = len(runtime_codes) > 0
    runtime_root_causes = dedupe([code for code in runtime_codes if code.startswith("runtime:root_cause_")])
    runtime_remediation_hints = runtime_root_cause_remediation_hints(runtime_root_causes)
    if has_runtime_failures:
        if not runtime_remediation_hints:
            runtime_remediation_hints = [
                "runtime:remediation_collect_failure_evidence_and_route_operator_triage"
            ]
    else:
        runtime_remediation_hints = []

    runtime_recovered_codes = dedupe(runtime_recovered_codes)
    runtime_retrying_names = dedupe(runtime_retrying_names)
    runtime_intervention_required_names = dedupe(runtime_intervention_required_names)
    runtime_historical_recovered_names = dedupe(runtime_historical_recovered_names)
    runtime_escalation_rows = sorted(
        [row for row in runtime_escalation_rows if isinstance(row, dict)],
        key=lambda row: (
            str(row.get("name") or ""),
            str(row.get("policy") or ""),
        ),
    )
    runtime_escalation_policies = dedupe([str(row.get("policy") or "").strip() for row in runtime_escalation_rows])
    runtime_escalation_urgencies = dedupe([str(row.get("urgency") or "").strip() for row in runtime_escalation_rows])
    runtime_ownership_families = dedupe([str(row.get("family") or "").strip() for row in runtime_escalation_rows])
    runtime_ownership_action_lanes = dedupe([str(row.get("action_lane") or "").strip() for row in runtime_escalation_rows])
    runtime_ownership_recovery_owners = dedupe([str(row.get("recovery_owner") or "").strip() for row in runtime_escalation_rows])
    runtime_family_playbook_hints_union = dedupe(
        [
            str(hint or "").strip()
            for row in runtime_escalation_rows
            for hint in ((row.get("playbook_hints") or []) if isinstance(row.get("playbook_hints"), list) else [])
            if str(hint or "").strip()
        ]
    )
    runtime_escalation_retryable_count = len(
        [row for row in runtime_escalation_rows if bool(row.get("retryable") is True)]
    )
    runtime_family_ownership_state = "none"
    if runtime_escalation_rows:
        runtime_family_ownership_state = "projected"
    runtime_family_playbook_state = "none"
    if runtime_escalation_rows:
        runtime_family_playbook_state = "projected"
    runtime_escalation_policy_breakdown_state = "none"
    if runtime_escalation_rows:
        if len(runtime_escalation_policies) <= 1 and len(runtime_escalation_urgencies) <= 1:
            runtime_escalation_policy_breakdown_state = "single"
        else:
            runtime_escalation_policy_breakdown_state = "mixed"
    ok = len(violations) == 0
    has_policy_failures = len(policy_codes) > 0
    all_failure_codes = dedupe(runtime_codes + policy_codes)
    has_retrying_runtime_failures = len(runtime_retrying_names) > 0
    has_intervention_runtime_failures = len(runtime_intervention_required_names) > 0

    latest_runtime_failure = None
    if runtime_failure_provenance_rows:
        latest_runtime_failure = sorted(
            runtime_failure_provenance_rows,
            key=lambda row: (
                1 if isinstance(row.get("lastRunAtMs"), int) else 0,
                int(row.get("lastRunAtMs") or -1),
            ),
            reverse=True,
        )[0]

    latest_failure_at_ms = latest_runtime_failure.get("lastRunAtMs") if isinstance(latest_runtime_failure, dict) else None
    latest_failure_name = str((latest_runtime_failure or {}).get("name") or "").strip() if isinstance(latest_runtime_failure, dict) else ""
    latest_failure_id = str((latest_runtime_failure or {}).get("id") or "").strip() if isinstance(latest_runtime_failure, dict) else ""
    latest_failure_reason_source = str((latest_runtime_failure or {}).get("reasonSource") or "unavailable").strip() if isinstance(latest_runtime_failure, dict) else "unavailable"
    latest_failure_reason_excerpt = str((latest_runtime_failure or {}).get("reasonExcerpt") or "").strip() if isinstance(latest_runtime_failure, dict) else ""

    runtime_failure_provenance_state = "none"
    if runtime_failure_provenance_rows:
        runtime_failure_provenance_state = "latest_timestamp_available" if isinstance(latest_failure_at_ms, int) else "latest_timestamp_unavailable"

    runtime_recovery_state = "none"
    if has_runtime_failures:
        if has_retrying_runtime_failures and has_intervention_runtime_failures:
            runtime_recovery_state = "mixed"
        elif has_intervention_runtime_failures:
            runtime_recovery_state = "intervention_required"
        elif has_retrying_runtime_failures:
            runtime_recovery_state = "retrying"
    elif len(runtime_historical_recovered_names) > 0:
        runtime_recovery_state = "historical_recovered"

    runtime_escalation_policy = runtime_escalation_policy_projection(
        has_runtime_failures=has_runtime_failures,
        runtime_root_causes=runtime_root_causes,
        runtime_recovery_state=runtime_recovery_state,
    )

    classification = "ok"
    classification_bucket = "none"
    error = ""
    failure_category = None
    failure_code = None
    failure_retryable = None
    if not ok:
        if has_runtime_failures:
            classification = "cron_runtime_failed"
            classification_bucket = "runtime"
            error = "runtime_failures"
            failure_category = str(runtime_escalation_policy.get("failure_category") or "runtime_operator_recovery_required")
            failure_retryable = bool(runtime_escalation_policy.get("retryable") is True)
            failure_code = runtime_codes[0] if runtime_codes else "runtime:enabled_authority_runtime_unhealthy"
        elif has_policy_failures:
            classification = "cron_policy_failed"
            classification_bucket = "policy"
            error = "policy_violations"
            failure_category = "policy_contract_fail_close"
            failure_code = policy_codes[0] if policy_codes else "policy:violations_present"
            failure_retryable = False
        else:
            classification = "cron_contract_drift"
            classification_bucket = "contract"
            error = "unknown_violations"
            failure_category = "cron_contract_fail_close"
            failure_code = "contract:unknown_failure"
            failure_retryable = False

    summary = {
        "ok": ok,
        "classification": classification,
        "classification_bucket": classification_bucket,
        "error": error,
        "detail": "",
        "checked": len(checked_jobs),
        "checked_jobs": checked_jobs,
        "coverage": {
            "expected_enabled_authority_names": expected,
            "missing_enabled_authority_names": missing_names,
            "duplicate_enabled_authority_names": duplicate_names,
            "enabled_counts": enabled_counts,
        },
        "violations": violations,
        "fail_close_triggered": bool(not ok),
        "failure_category": failure_category,
        "failure_code": failure_code,
        "failure_retryable": failure_retryable,
        "runtime_recovery_state": runtime_recovery_state,
        "runtime_recovery_projection": {
            "retrying_names": runtime_retrying_names,
            "retrying_count": len(runtime_retrying_names),
            "intervention_required_names": runtime_intervention_required_names,
            "intervention_required_count": len(runtime_intervention_required_names),
            "historical_recovered_names": runtime_historical_recovered_names,
            "historical_recovered_count": len(runtime_historical_recovered_names),
        },
        "runtime_failure_provenance": {
            "state": runtime_failure_provenance_state,
            "failed_runtime_names": dedupe([str(row.get("name") or "").strip() for row in runtime_failure_provenance_rows]),
            "failed_runtime_count": len(runtime_failure_provenance_rows),
            "latest_failure_name": latest_failure_name or None,
            "latest_failure_id": latest_failure_id or None,
            "latest_failure_at_ms": latest_failure_at_ms if isinstance(latest_failure_at_ms, int) else None,
            "latest_failure_reason_source": latest_failure_reason_source,
            "latest_failure_reason_excerpt": latest_failure_reason_excerpt or None,
        },
        "runtime_remediation_projection": {
            "state": "projected" if has_runtime_failures else "none",
            "root_causes": runtime_root_causes if has_runtime_failures else [],
            "hints": runtime_remediation_hints if has_runtime_failures else [],
            "hint_count": len(runtime_remediation_hints if has_runtime_failures else []),
        },
        "runtime_escalation_policy_projection": {
            "state": str(runtime_escalation_policy.get("state") or "none"),
            "policy": str(runtime_escalation_policy.get("policy") or "none"),
            "urgency": str(runtime_escalation_policy.get("urgency") or "none"),
            "source": str(runtime_escalation_policy.get("source") or "none"),
            "retryable": runtime_escalation_policy.get("retryable"),
            "failure_category": runtime_escalation_policy.get("failure_category"),
        },
        "runtime_escalation_policy_breakdown": {
            "state": runtime_escalation_policy_breakdown_state,
            "policies": runtime_escalation_policies,
            "urgencies": runtime_escalation_urgencies,
            "retryable_count": runtime_escalation_retryable_count,
            "non_retryable_count": max(0, len(runtime_escalation_rows) - runtime_escalation_retryable_count),
            "rows": runtime_escalation_rows,
        },
        "runtime_family_ownership_projection": {
            "state": runtime_family_ownership_state,
            "families": runtime_ownership_families,
            "action_lanes": runtime_ownership_action_lanes,
            "recovery_owners": runtime_ownership_recovery_owners,
            "rows": runtime_escalation_rows,
        },
        "runtime_family_playbook_projection": {
            "state": runtime_family_playbook_state,
            "families": runtime_ownership_families,
            "hints": runtime_family_playbook_hints_union,
            "rows": runtime_escalation_rows,
        },
        "recovery_annotations": recovery_annotations,
        "failure_codes": {
            "all": [] if ok else all_failure_codes,
            "critical": [] if ok else (runtime_codes if has_runtime_failures else all_failure_codes),
            "policy": [] if ok else policy_codes,
            "contract": [],
            "connectivity": [],
            "runtime": [] if ok else runtime_codes,
            "runtime_recovered": runtime_recovered_codes,
        },
    }

summary.setdefault("ok", False)
summary.setdefault("classification", "cron_contract_drift")
summary.setdefault("classification_bucket", "contract")
summary.setdefault("error", "unknown_error")
summary.setdefault("detail", "")
summary.setdefault("checked", 0)
summary.setdefault("checked_jobs", [])
summary.setdefault(
    "coverage",
    {
        "expected_enabled_authority_names": expected_names(),
        "missing_enabled_authority_names": expected_names(),
        "duplicate_enabled_authority_names": [],
        "enabled_counts": {name: 0 for name in expected_names()},
    },
)
summary.setdefault("violations", [])
summary.setdefault("fail_close_triggered", True)
summary.setdefault("failure_category", "cron_contract_fail_close")
summary.setdefault("failure_code", "contract:unknown_failure")
summary.setdefault("failure_retryable", False)
summary.setdefault("runtime_recovery_state", "none")
summary.setdefault(
    "runtime_recovery_projection",
    {
        "retrying_names": [],
        "retrying_count": 0,
        "intervention_required_names": [],
        "intervention_required_count": 0,
        "historical_recovered_names": [],
        "historical_recovered_count": 0,
    },
)
summary.setdefault(
    "runtime_failure_provenance",
    {
        "state": "none",
        "failed_runtime_names": [],
        "failed_runtime_count": 0,
        "latest_failure_name": None,
        "latest_failure_id": None,
        "latest_failure_at_ms": None,
        "latest_failure_reason_source": "unavailable",
        "latest_failure_reason_excerpt": None,
    },
)
summary.setdefault(
    "runtime_remediation_projection",
    {
        "state": "none",
        "root_causes": [],
        "hints": [],
        "hint_count": 0,
    },
)
summary.setdefault(
    "runtime_escalation_policy_projection",
    {
        "state": "none",
        "policy": "none",
        "urgency": "none",
        "source": "none",
        "retryable": None,
        "failure_category": None,
    },
)
summary.setdefault(
    "runtime_escalation_policy_breakdown",
    {
        "state": "none",
        "policies": [],
        "urgencies": [],
        "retryable_count": 0,
        "non_retryable_count": 0,
        "rows": [],
    },
)
summary.setdefault(
    "runtime_family_ownership_projection",
    {
        "state": "none",
        "families": [],
        "action_lanes": [],
        "recovery_owners": [],
        "rows": [],
    },
)
summary.setdefault(
    "runtime_family_playbook_projection",
    {
        "state": "none",
        "families": [],
        "hints": [],
        "rows": [],
    },
)
summary.setdefault("recovery_annotations", [])
summary.setdefault(
    "failure_codes",
    {
        "all": ["contract:unknown_failure"],
        "critical": ["contract:unknown_failure"],
        "policy": [],
        "contract": ["contract:unknown_failure"],
        "connectivity": [],
        "runtime": [],
        "runtime_recovered": [],
    },
)

failure_codes_obj = summary.get("failure_codes")
if not isinstance(failure_codes_obj, dict):
    failure_codes_obj = {}
for key in ("all", "critical", "policy", "contract", "connectivity", "runtime", "runtime_recovered"):
    value = failure_codes_obj.get(key)
    if isinstance(value, list):
        failure_codes_obj[key] = dedupe([str(item or "").strip() for item in value if str(item or "").strip()])
    else:
        failure_codes_obj[key] = []
summary["failure_codes"] = failure_codes_obj

summary["schema_version"] = schema_version
summary["classification_taxonomy_version"] = classification_taxonomy_version
summary["failure_taxonomy_version"] = failure_taxonomy_version
summary["evidence_refs"] = dedupe(
    [
        "ops/openclaw/no_llm_watchdog_cron_authority_guard.sh",
        "docs/ops/model_routing_no_llm_matrix_v1.md",
        "docs/ops/openclaw_cron_jobs.md",
    ]
)

print(json.dumps(summary, ensure_ascii=False))
PY
}

_extract_failed_cron_session_names_json() {
  local sessions_payload_path="${1:-}"

  if [[ -z "$sessions_payload_path" || ! -f "$sessions_payload_path" ]]; then
    echo '{"failed_names":[],"failed_rows":[],"cron_session_rows_seen":0,"named_status_rows_seen":0,"status_metadata_available":false}'
    return 0
  fi

  jq -c '
    def cron_row:
      (
        ((.key // "") | tostring | contains(":cron:"))
        or ((.label // .displayName // "") | tostring | ascii_downcase | startswith("cron:"))
      );

    def cron_id_from_key:
      ((.key // "") | tostring | split(":cron:")
      | if length < 2 then "" else (.[1] | split(":") | .[0]) end);

    def norm_status:
      (.status // "" | tostring | ascii_downcase | gsub("-"; "_"));

    def parse_epoch_ms($v):
      if ($v | type) == "number" then ($v | floor)
      elif ($v | type) == "string" then ($v | tonumber? // null)
      else null
      end;

    def row_updated_at_ms:
      [
        (.updatedAt // null),
        (.lastMessageAt // null),
        (.lastOutputAt // null),
        (.lastToolAt // null),
        (.createdAt // null)
      ]
      | map(parse_epoch_ms(.))
      | map(select(. != null))
      | if length > 0 then max else null end;

    def cron_name:
      ((.label // .displayName // "") | tostring)
      | sub("^Cron:\\s*"; "")
      | gsub("^\\s+|\\s+$"; "");

    def status_from_row:
      (norm_status) as $s
      | if ($s | length) > 0 then $s
        elif (.abortedLastRun | type) == "boolean" then (if .abortedLastRun then "failed" else "ok" end)
        else ""
        end;

    def status_source:
      (norm_status) as $s
      | if ($s | length) > 0 then "status"
        elif (.abortedLastRun | type) == "boolean" then "aborted_last_run"
        else "unknown"
        end;

    def is_failed_status:
      (status_from_row) as $s
      | ($s == "failed"
         or $s == "error"
         or $s == "aborted"
         or $s == "timeout"
         or $s == "timed_out"
         or $s == "crashed"
         or $s == "cancelled"
         or $s == "canceled");

    (
      [
        .sessions[]?
        | select(cron_row)
        | (cron_name) as $name
        | (cron_id_from_key) as $cron_id
        | select(($name | length > 0) or ($cron_id | length > 0))
        | select(is_failed_status)
        | {
            name: $name,
            cron_id: $cron_id,
            session_key: ((.key // "") | tostring),
            status: status_from_row,
            status_source: status_source,
            updated_at_ms: row_updated_at_ms
          }
      ]
    ) as $failed_rows
    | {
      failed_names: ($failed_rows | map(.name) | map(select((. // "") | length > 0)) | unique),
      failed_rows: $failed_rows,
      cron_session_rows_seen: ([.sessions[]? | select(cron_row)] | length),
      named_status_rows_seen: ([
        .sessions[]?
        | select(cron_row)
        | (cron_name) as $name
        | (cron_id_from_key) as $cron_id
        | select(($name | length > 0) or ($cron_id | length > 0))
        | select((status_from_row | length) > 0)
      ] | length),
      status_metadata_available: (([
        .sessions[]?
        | select(cron_row)
        | (cron_name) as $name
        | (cron_id_from_key) as $cron_id
        | select(($name | length > 0) or ($cron_id | length > 0))
        | select((status_from_row | length) > 0)
      ] | length) > 0)
    }
  ' "$sessions_payload_path" 2>/dev/null || echo '{"failed_names":[],"failed_rows":[],"cron_session_rows_seen":0,"named_status_rows_seen":0,"status_metadata_available":false}'
}

_augment_summary_with_session_surface_reconciliation() {
  local summary_json_raw="${1-}"
  if [[ -z "$summary_json_raw" ]]; then
    summary_json_raw='{}'
  fi
  local failed_session_names_json="${2:-[]}"
  local active_minutes_raw="${3:-}"

  python3 - "$summary_json_raw" "$failed_session_names_json" "$active_minutes_raw" <<'PY'
import json
import sys
import time
from typing import List

summary_raw = str(sys.argv[1] or "{}")
session_probe_raw = str(sys.argv[2] or "{}")
active_minutes_raw = str(sys.argv[3] or "")


def dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        txt = str(raw or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def normalize_status(raw: object) -> str:
    return str(raw or "").strip().lower().replace("-", "_")


def parse_nonnegative_int(raw: object):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        val = int(txt)
    except Exception:
        return None
    return val if val >= 0 else None


def parse_epoch_ms(raw: object):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        val = int(float(txt))
    except Exception:
        return None
    return val if val >= 0 else None


def residue_staleness_bucket_from_age_sec(age_sec: object) -> str:
    if not isinstance(age_sec, int) or age_sec < 0:
        return "age_unknown"
    if age_sec < 3600:
        return "recent_lt_1h"
    if age_sec < 21600:
        return "recent_1h_to_6h"
    if age_sec < 86400:
        return "stale_6h_to_24h"
    if age_sec < 259200:
        return "stale_1d_to_3d"
    return "historical_gt_3d"


def runtime_family_ownership_for_name(name: str):
    normalized = str(name or "").strip().lower()
    if normalized in {"continuity:backup-checkpoint-90m", "continuity:stale-progress-45m", "core-roadmap-executor-idle-watchdog"}:
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "ownership_source": "name_exact:continuity_expected_rails",
        }
    if normalized in {"web-capture-scheduler-governance-watchdog", "web-capture-scheduler-governance"}:
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "ownership_source": "name_exact:web_capture_expected_rails",
        }
    if normalized in {"obsidian:hourly-canary", "obsidian-vault-tick-hourly-canary", "obsidian:vault-tick-hourly"}:
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "ownership_source": "name_exact:obsidian_expected_rails",
        }
    if normalized.startswith("continuity:"):
        return {
            "family": "continuity",
            "action_lane": "continuity_no_nudge_authority",
            "recovery_owner": "continuity_control_plane_operator",
            "routing_hint": "route:continuity_control_plane_lane",
            "ownership_source": "name_prefix:continuity",
        }
    if normalized.startswith("web-capture"):
        return {
            "family": "web_capture",
            "action_lane": "web_capture_scheduler_governance_watchdog",
            "recovery_owner": "web_capture_governance_operator",
            "routing_hint": "route:web_capture_governance_lane",
            "ownership_source": "name_prefix:web_capture",
        }
    if normalized.startswith("obsidian"):
        return {
            "family": "obsidian",
            "action_lane": "obsidian_hourly_integrity_watchdogs",
            "recovery_owner": "obsidian_memory_integrity_operator",
            "routing_hint": "route:obsidian_integrity_lane",
            "ownership_source": "name_prefix:obsidian",
        }
    return {
        "family": "unknown",
        "action_lane": "unknown",
        "recovery_owner": "operator_general_triage",
        "routing_hint": "route:operator_general_triage",
        "ownership_source": "fallback:unknown_family",
    }


def historical_session_residue_playbook_hints(family: str) -> List[str]:
    normalized_family = str(family or "").strip().lower()
    family_mapping = {
        "continuity": [
            "runtime:playbook_continuity_run_no_nudge_authority_watchdog_contract",
            "runtime:playbook_core_roadmap_floor_refill_loop_contract",
            "runtime:playbook_continuity_validate_checkpoint_and_stale_progress_sessions",
        ],
        "web_capture": [
            "runtime:playbook_web_capture_run_scheduler_governance_watchdog_contract",
            "runtime:playbook_web_capture_reconcile_scheduler_manifest_and_capture_backlog",
        ],
        "obsidian": [
            "runtime:playbook_obsidian_run_hourly_canary_and_vault_tick_contracts",
            "runtime:playbook_obsidian_verify_vault_materialization_and_memory_sync",
        ],
    }

    hints = list(family_mapping.get(normalized_family, []))
    if not hints:
        hints = ["runtime:playbook_operator_general_triage_collect_runtime_artifacts"]
    hints.append("runtime:playbook_historical_session_residue_verify_current_rail_health_and_keep_low_urgency")
    return dedupe(hints)


def parse_payload(text: str, fallback):
    raw = str(text or "").strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


summary = parse_payload(summary_raw, {})
if not isinstance(summary, dict):
    summary = {}

session_probe_obj = parse_payload(session_probe_raw, {})
failed_session_names: List[str] = []
failed_session_rows: List[dict] = []
cron_session_rows_seen = 0
named_status_rows_seen = 0
status_metadata_available = False

if isinstance(session_probe_obj, list):
    failed_session_names = dedupe([str(item or "") for item in session_probe_obj])
    failed_session_rows = [
        {
            "name": name,
            "cron_id": None,
            "session_key": None,
            "status": "failed",
            "status_source": "status",
            "updated_at_ms": None,
        }
        for name in failed_session_names
    ]
    status_metadata_available = True
elif isinstance(session_probe_obj, dict):
    raw_failed_names = session_probe_obj.get("failed_names")
    if isinstance(raw_failed_names, list):
        failed_session_names = dedupe([str(item or "") for item in raw_failed_names])

    raw_failed_rows = session_probe_obj.get("failed_rows")
    if isinstance(raw_failed_rows, list):
        for raw_row in raw_failed_rows:
            if not isinstance(raw_row, dict):
                continue
            row_name = str(raw_row.get("name") or "").strip()
            row_cron_id = str(raw_row.get("cron_id") or "").strip()
            row_session_key = str(raw_row.get("session_key") or "").strip()
            if not row_name and not row_cron_id and not row_session_key:
                continue
            status_source_raw = str(raw_row.get("status_source") or "").strip().lower().replace("-", "_")
            if status_source_raw not in {"status", "aborted_last_run", "unknown"}:
                status_source_raw = ""
            if not status_source_raw:
                if str(raw_row.get("status") or "").strip():
                    status_source_raw = "status"
                elif isinstance(raw_row.get("abortedLastRun"), bool):
                    status_source_raw = "aborted_last_run"
                else:
                    status_source_raw = "unknown"
            failed_session_rows.append(
                {
                    "name": row_name,
                    "cron_id": row_cron_id or None,
                    "session_key": row_session_key or None,
                    "status": normalize_status(raw_row.get("status") or "failed") or "failed",
                    "status_source": status_source_raw,
                    "updated_at_ms": parse_epoch_ms(raw_row.get("updated_at_ms")),
                }
            )

    if failed_session_names and not failed_session_rows:
        failed_session_rows = [
            {
                "name": name,
                "cron_id": None,
                "session_key": None,
                "status": "failed",
                "status_source": "status",
                "updated_at_ms": None,
            }
            for name in failed_session_names
        ]
    if failed_session_rows and not failed_session_names:
        failed_session_names = dedupe([str(row.get("name") or "") for row in failed_session_rows if str(row.get("name") or "").strip()])

    parsed_cron_rows_seen = parse_nonnegative_int(session_probe_obj.get("cron_session_rows_seen"))
    parsed_named_status_rows_seen = parse_nonnegative_int(session_probe_obj.get("named_status_rows_seen"))
    cron_session_rows_seen = parsed_cron_rows_seen or 0
    named_status_rows_seen = parsed_named_status_rows_seen or 0
    status_metadata_available = bool(session_probe_obj.get("status_metadata_available")) or named_status_rows_seen > 0

active_window_minutes = None
try:
    parsed_minutes = int(float(active_minutes_raw))
    if parsed_minutes > 0:
        active_window_minutes = parsed_minutes
except Exception:
    active_window_minutes = None

checked_jobs = summary.get("checked_jobs") if isinstance(summary.get("checked_jobs"), list) else []

unhealthy_statuses = {
    "error",
    "failed",
    "fail",
    "aborted",
    "timeout",
    "timed_out",
    "crashed",
    "cancelled",
    "canceled",
}
healthy_statuses = {
    "ok",
    "success",
    "succeeded",
    "healthy",
    "ready",
    "pass",
    "passed",
    "complete",
    "completed",
}

checked_jobs_by_name = {}
checked_jobs_by_id = {}
healthy_current_names: List[str] = []
for row in checked_jobs:
    if not isinstance(row, dict):
        continue
    name = str(row.get("name") or "").strip()
    if not name:
        continue
    job_id = str(row.get("id") or "").strip()
    last_status = normalize_status(row.get("lastStatus"))
    last_run_status = normalize_status(row.get("lastRunStatus"))
    consecutive_errors = parse_nonnegative_int(row.get("consecutiveErrors"))

    has_unhealthy_status = (last_status and last_status in unhealthy_statuses) or (
        last_run_status and last_run_status in unhealthy_statuses
    )
    has_positive_consecutive_errors = isinstance(consecutive_errors, int) and consecutive_errors > 0
    has_healthy_signal = (last_status in healthy_statuses) or (last_run_status in healthy_statuses)

    if name not in checked_jobs_by_name:
        checked_jobs_by_name[name] = {
            "name": name,
            "id": job_id or None,
            "lastStatus": last_status,
            "lastRunStatus": last_run_status,
            "consecutiveErrors": consecutive_errors,
        }

    if job_id and job_id not in checked_jobs_by_id:
        checked_jobs_by_id[job_id] = {
            "id": job_id,
            "name": name,
            "lastStatus": last_status,
            "lastRunStatus": last_run_status,
            "consecutiveErrors": consecutive_errors,
        }

    if (not has_unhealthy_status) and (not has_positive_consecutive_errors) and has_healthy_signal:
        healthy_current_names.append(name)

resolved_failed_session_rows: List[dict] = []
failed_session_rows_by_name = {}
for row in failed_session_rows:
    if not isinstance(row, dict):
        continue
    row_name = str(row.get("name") or "").strip()
    row_cron_id = str(row.get("cron_id") or "").strip()
    row_session_key = str(row.get("session_key") or "").strip()
    if not row_name and row_cron_id:
        mapped = checked_jobs_by_id.get(row_cron_id)
        if isinstance(mapped, dict):
            row_name = str(mapped.get("name") or "").strip()
    if not row_name:
        continue

    row_status = normalize_status(row.get("status") or "failed") or "failed"
    row_status_source = str(row.get("status_source") or "").strip().lower().replace("-", "_")
    if row_status_source not in {"status", "aborted_last_run", "unknown"}:
        row_status_source = "unknown"
    row_updated_at_ms = parse_epoch_ms(row.get("updated_at_ms"))
    resolved_row = {
        "name": row_name,
        "cron_id": row_cron_id or None,
        "session_key": row_session_key or None,
        "status": row_status,
        "status_source": row_status_source,
        "updated_at_ms": row_updated_at_ms,
    }
    resolved_failed_session_rows.append(resolved_row)

    existing = failed_session_rows_by_name.get(row_name)
    if existing is None:
        failed_session_rows_by_name[row_name] = dict(resolved_row)
        continue

    existing_updated_at_ms = parse_epoch_ms(existing.get("updated_at_ms"))
    if row_updated_at_ms is None:
        continue
    if existing_updated_at_ms is None or row_updated_at_ms > existing_updated_at_ms:
        failed_session_rows_by_name[row_name] = dict(resolved_row)

if failed_session_rows_by_name:
    failed_session_names = dedupe(failed_session_names + list(failed_session_rows_by_name.keys()))

healthy_current_names = dedupe(healthy_current_names)
healthy_set = set(healthy_current_names)
historical_failed_session_residue_names = [name for name in failed_session_names if name in healthy_set]

historical_failed_session_residue_rows = []
now_ms = int(time.time() * 1000)
for residue_name in historical_failed_session_residue_names:
    ownership = runtime_family_ownership_for_name(residue_name)
    current_row = checked_jobs_by_name.get(residue_name) if isinstance(checked_jobs_by_name, dict) else None
    current_last_status = normalize_status((current_row or {}).get("lastStatus")) if isinstance(current_row, dict) else None
    current_last_run_status = (
        normalize_status((current_row or {}).get("lastRunStatus")) if isinstance(current_row, dict) else None
    )
    current_consecutive_errors = (
        parse_nonnegative_int((current_row or {}).get("consecutiveErrors")) if isinstance(current_row, dict) else None
    )
    current_health_state = "healthy_now" if residue_name in healthy_set else "unhealthy_or_unknown_now"
    failed_row = failed_session_rows_by_name.get(residue_name) if isinstance(failed_session_rows_by_name, dict) else None
    last_failed_at_ms = parse_epoch_ms((failed_row or {}).get("updated_at_ms")) if isinstance(failed_row, dict) else None
    failed_session_key = str((failed_row or {}).get("session_key") or "").strip() if isinstance(failed_row, dict) else ""
    failed_cron_id = str((failed_row or {}).get("cron_id") or "").strip() if isinstance(failed_row, dict) else ""
    failed_status_source = str((failed_row or {}).get("status_source") or "").strip() if isinstance(failed_row, dict) else ""
    residue_age_sec = int(max(0, (now_ms - last_failed_at_ms) / 1000)) if isinstance(last_failed_at_ms, int) else None
    residue_staleness_bucket = residue_staleness_bucket_from_age_sec(residue_age_sec)
    historical_failed_session_residue_rows.append(
        {
            "name": residue_name,
            "family": ownership.get("family") or "unknown",
            "action_lane": ownership.get("action_lane") or "unknown",
            "recovery_owner": ownership.get("recovery_owner") or "operator_general_triage",
            "routing_hint": ownership.get("routing_hint") or "route:operator_general_triage",
            "ownership_source": ownership.get("ownership_source") or "fallback:unknown_family",
            "playbook_hints": historical_session_residue_playbook_hints(ownership.get("family") or "unknown"),
            "last_failed_status": str((failed_row or {}).get("status") or "failed") if isinstance(failed_row, dict) else "failed",
            "last_failed_status_source": failed_status_source or "unknown",
            "last_failed_session_key": failed_session_key or None,
            "last_failed_cron_id": failed_cron_id or None,
            "last_failed_at_ms": last_failed_at_ms,
            "residue_age_sec": residue_age_sec,
            "staleness_bucket": residue_staleness_bucket,
            "current_health_state": current_health_state,
            "current_reconciled_now": residue_name in healthy_set,
            "current_last_status": current_last_status,
            "current_last_run_status": current_last_run_status,
            "current_consecutive_errors": current_consecutive_errors,
        }
    )

historical_failed_session_residue_families = dedupe(
    [str(row.get("family") or "") for row in historical_failed_session_residue_rows]
)
historical_failed_session_residue_action_lanes = dedupe(
    [str(row.get("action_lane") or "") for row in historical_failed_session_residue_rows]
)
historical_failed_session_residue_recovery_owners = dedupe(
    [str(row.get("recovery_owner") or "") for row in historical_failed_session_residue_rows]
)
historical_failed_session_residue_playbook_hints_union = dedupe(
    [
        str(hint or "")
        for row in historical_failed_session_residue_rows
        for hint in ((row.get("playbook_hints") or []) if isinstance(row.get("playbook_hints"), list) else [])
    ]
)
historical_failed_session_residue_staleness_buckets = dedupe(
    [str(row.get("staleness_bucket") or "") for row in historical_failed_session_residue_rows]
)

historical_failed_session_residue_age_secs = [
    int(row.get("residue_age_sec"))
    for row in historical_failed_session_residue_rows
    if isinstance(row.get("residue_age_sec"), int) and int(row.get("residue_age_sec")) >= 0
]
historical_failed_session_residue_age_known_count = len(historical_failed_session_residue_age_secs)
historical_failed_session_residue_age_unknown_count = max(
    0,
    len(historical_failed_session_residue_rows) - historical_failed_session_residue_age_known_count,
)
historical_failed_session_residue_oldest_age_sec = (
    max(historical_failed_session_residue_age_secs) if historical_failed_session_residue_age_secs else None
)
historical_failed_session_residue_newest_age_sec = (
    min(historical_failed_session_residue_age_secs) if historical_failed_session_residue_age_secs else None
)

historical_failed_session_card_counts_by_name = {}
for row in resolved_failed_session_rows:
    if not isinstance(row, dict):
        continue
    row_name = str(row.get("name") or "").strip()
    if not row_name:
        continue
    historical_failed_session_card_counts_by_name[row_name] = historical_failed_session_card_counts_by_name.get(row_name, 0) + 1

historical_failed_session_residue_ownership_state = "projected" if historical_failed_session_residue_rows else "none"
historical_failed_session_residue_playbook_state = "projected" if historical_failed_session_residue_rows else "none"
historical_failed_session_residue_current_health_state = "projected" if historical_failed_session_residue_rows else "none"

residue_state = "none_observed"
if status_metadata_available:
    if historical_failed_session_residue_names:
        residue_state = "projected"
    elif failed_session_names:
        residue_state = "none_in_current_healthy_intersection"
else:
    if cron_session_rows_seen > 0:
        residue_state = "unknown_status_metadata_unavailable"
    else:
        residue_state = "no_cron_session_rows_observed"

if residue_state == "unknown_status_metadata_unavailable" and not historical_failed_session_residue_rows:
    historical_failed_session_residue_current_health_state = "unknown_status_metadata_unavailable"

historical_failed_session_residue_decay_state = "none"
historical_failed_session_residue_decay_urgency = "none"
if residue_state == "projected":
    if historical_failed_session_residue_rows and historical_failed_session_residue_age_unknown_count == 0:
        newest_age_sec = historical_failed_session_residue_newest_age_sec
        if isinstance(newest_age_sec, int) and newest_age_sec >= 86400:
            historical_failed_session_residue_decay_state = "projected_historical_decay"
            historical_failed_session_residue_decay_urgency = "historical_record_only"
        elif isinstance(newest_age_sec, int) and newest_age_sec >= 21600:
            historical_failed_session_residue_decay_state = "projected_stale_decay"
            historical_failed_session_residue_decay_urgency = "low_urgency_stale_reconciled"
        else:
            historical_failed_session_residue_decay_state = "projected_recent_reconciled"
            historical_failed_session_residue_decay_urgency = "low_urgency_recent_reconciled"
    elif historical_failed_session_residue_rows and historical_failed_session_residue_age_known_count > 0:
        historical_failed_session_residue_decay_state = "projected_partial_age_metadata"
        historical_failed_session_residue_decay_urgency = "low_urgency_reconciled_partial_age"
    elif historical_failed_session_residue_rows:
        historical_failed_session_residue_decay_state = "projected_age_metadata_unavailable"
        historical_failed_session_residue_decay_urgency = "low_urgency_reconciled_age_unknown"
elif residue_state == "unknown_status_metadata_unavailable":
    historical_failed_session_residue_decay_state = "unknown_status_metadata_unavailable"
    historical_failed_session_residue_decay_urgency = "unknown_status_metadata_unavailable"

historical_failed_session_card_projection_state = "none"
if historical_failed_session_residue_rows:
    historical_failed_session_card_projection_state = "projected"
elif residue_state == "unknown_status_metadata_unavailable":
    historical_failed_session_card_projection_state = "unknown_status_metadata_unavailable"

def historical_failed_card_status_projection_for_row(row: dict) -> str:
    current_health_state = str(row.get("current_health_state") or "").strip().lower()
    current_reconciled_now = bool(row.get("current_reconciled_now") is True)
    if current_reconciled_now and current_health_state == "healthy_now":
        return "historical_failed_reconciled_now"
    if current_reconciled_now:
        return "historical_failed_reconciled"
    if current_health_state.startswith("unhealthy"):
        return "active_failed_or_unreconciled"
    return "historical_failed_status_unknown"


def historical_failed_card_severity_projection_for_status(status_projection: str) -> str:
    normalized = str(status_projection or "").strip().lower()
    if normalized in {"historical_failed_reconciled_now", "historical_failed_reconciled"}:
        return "info"
    if normalized == "active_failed_or_unreconciled":
        return "critical"
    return "warn"


def historical_failed_card_retirement_projection_for_row(row: dict, status_projection: str) -> str:
    normalized_status = str(status_projection or "").strip().lower()
    if normalized_status not in {"historical_failed_reconciled_now", "historical_failed_reconciled"}:
        return "not_retired_unreconciled_or_unhealthy"
    staleness_bucket = str(row.get("staleness_bucket") or "").strip().lower()
    if staleness_bucket in {"stale_6h_to_24h", "stale_1d_to_3d", "historical_gt_3d"}:
        return "resolved_historical_retired"
    if staleness_bucket in {"recent_lt_1h", "recent_1h_to_6h"}:
        return "resolved_historical_recent"
    if staleness_bucket == "age_unknown":
        return "resolved_historical_age_unknown"
    return "resolved_historical_retirement_unknown"


def historical_failed_card_retirement_reason_for_state(retirement_projection: str) -> str:
    normalized = str(retirement_projection or "").strip().lower()
    if normalized == "resolved_historical_retired":
        return "current_authority_healthy_now_and_residue_stale"
    if normalized == "resolved_historical_recent":
        return "current_authority_healthy_now_recent_residue"
    if normalized == "resolved_historical_age_unknown":
        return "current_authority_healthy_now_age_metadata_unavailable"
    if normalized == "not_retired_unreconciled_or_unhealthy":
        return "current_authority_not_reconciled_or_unhealthy"
    return "retirement_projection_metadata_incomplete"


historical_failed_session_card_projection_rows = []
for row in historical_failed_session_residue_rows:
    if not isinstance(row, dict):
        continue
    card_status_projection = historical_failed_card_status_projection_for_row(row)
    card_severity_projection = historical_failed_card_severity_projection_for_status(card_status_projection)
    card_retirement_projection = historical_failed_card_retirement_projection_for_row(row, card_status_projection)
    card_retirement_reason = historical_failed_card_retirement_reason_for_state(card_retirement_projection)
    normalization_reason = "insufficient_current_authority_health_signal"
    if card_status_projection == "historical_failed_reconciled_now":
        normalization_reason = "current_authority_healthy_now"
    elif card_status_projection == "historical_failed_reconciled":
        normalization_reason = "current_authority_reconciled"
    elif card_status_projection == "active_failed_or_unreconciled":
        normalization_reason = "current_authority_unhealthy_or_unknown"

    historical_failed_session_card_projection_rows.append(
        {
            "name": row.get("name"),
            "failed_card_count": int(historical_failed_session_card_counts_by_name.get(str(row.get("name") or ""), 0) or 0),
            "last_failed_session_key": row.get("last_failed_session_key"),
            "last_failed_cron_id": row.get("last_failed_cron_id"),
            "last_failed_status": row.get("last_failed_status"),
            "last_failed_status_source": row.get("last_failed_status_source"),
            "last_failed_at_ms": row.get("last_failed_at_ms"),
            "current_health_state": row.get("current_health_state"),
            "current_reconciled_now": bool(row.get("current_reconciled_now") is True),
            "historical_card_status": card_status_projection,
            "historical_card_severity": card_severity_projection,
            "historical_status_normalization_reason": normalization_reason,
            "historical_card_retirement_state": card_retirement_projection,
            "historical_card_retirement_reason": card_retirement_reason,
        }
    )

historical_failed_session_card_status_projection_values = dedupe(
    [str(row.get("historical_card_status") or "") for row in historical_failed_session_card_projection_rows]
)
historical_failed_session_card_severity_projection_values = dedupe(
    [str(row.get("historical_card_severity") or "") for row in historical_failed_session_card_projection_rows]
)
historical_failed_session_card_retirement_projection_values = dedupe(
    [str(row.get("historical_card_retirement_state") or "") for row in historical_failed_session_card_projection_rows]
)
historical_failed_session_card_status_projection_state = "none"
historical_failed_session_card_severity_projection_state = "none"
historical_failed_session_card_retirement_projection_state = "none"
if historical_failed_session_card_projection_rows:
    historical_failed_session_card_status_projection_state = "projected"
    historical_failed_session_card_severity_projection_state = "projected"
    historical_failed_session_card_retirement_projection_state = "projected"
elif residue_state == "unknown_status_metadata_unavailable":
    historical_failed_session_card_status_projection_state = "unknown_status_metadata_unavailable"
    historical_failed_session_card_severity_projection_state = "unknown_status_metadata_unavailable"
    historical_failed_session_card_retirement_projection_state = "unknown_status_metadata_unavailable"

repeated_false_green_guard_rows = []
for row in historical_failed_session_card_projection_rows:
    if not isinstance(row, dict):
        continue
    row_name = str(row.get("name") or "").strip()
    if not row_name:
        continue
    failed_card_count = int(row.get("failed_card_count") or 0)
    status_projection = str(row.get("historical_card_status") or "").strip().lower()
    if failed_card_count < 2:
        continue
    if status_projection not in {"historical_failed_reconciled_now", "historical_failed_reconciled"}:
        continue
    repeated_false_green_guard_rows.append(
        {
            "name": row_name,
            "failed_card_count": failed_card_count,
            "historical_card_status": str(row.get("historical_card_status") or ""),
            "historical_card_severity": str(row.get("historical_card_severity") or ""),
            "current_health_state": str(row.get("current_health_state") or ""),
            "last_failed_status_source": str(row.get("last_failed_status_source") or ""),
        }
    )

repeated_false_green_guard_state = "none"
repeated_false_green_guard_reason = "none"
if repeated_false_green_guard_rows:
    repeated_false_green_guard_state = "active"
    repeated_false_green_guard_reason = "repeated_historical_failed_session_cards_reconciled_now"

historical_failed_session_card_rows_by_name = {
    str(row.get("name") or "").strip(): row
    for row in historical_failed_session_card_projection_rows
    if isinstance(row, dict) and str(row.get("name") or "").strip()
}

historical_failed_session_rows_projected = []
for row_name, row in failed_session_rows_by_name.items():
    if not isinstance(row, dict):
        continue

    observed_status = normalize_status(row.get("status") or "failed") or "failed"
    projected_status = observed_status
    projection_state = "active_or_unreconciled"
    projection_reason = "observed_failed_status"
    projected_severity = "critical" if observed_status in unhealthy_statuses else "warn"
    projected_retirement_state = "not_retired_unreconciled_or_unhealthy"

    current_reconciled_now = False
    current_health_state = "unhealthy_or_unknown_now"

    current_row = checked_jobs_by_name.get(row_name) if isinstance(checked_jobs_by_name, dict) else None
    current_last_status = normalize_status((current_row or {}).get("lastStatus")) if isinstance(current_row, dict) else None
    current_last_run_status = (
        normalize_status((current_row or {}).get("lastRunStatus")) if isinstance(current_row, dict) else None
    )
    current_consecutive_errors = (
        parse_nonnegative_int((current_row or {}).get("consecutiveErrors")) if isinstance(current_row, dict) else None
    )

    has_unhealthy_current = (current_last_status and current_last_status in unhealthy_statuses) or (
        current_last_run_status and current_last_run_status in unhealthy_statuses
    )
    has_positive_current_consecutive_errors = (
        isinstance(current_consecutive_errors, int) and current_consecutive_errors > 0
    )
    has_healthy_current_signal = (current_last_status in healthy_statuses) or (
        current_last_run_status in healthy_statuses
    )

    if (not has_unhealthy_current) and (not has_positive_current_consecutive_errors) and has_healthy_current_signal:
        current_reconciled_now = True
        current_health_state = "healthy_now"
    elif has_unhealthy_current or has_positive_current_consecutive_errors:
        current_health_state = "unhealthy_now"

    card_row = historical_failed_session_card_rows_by_name.get(row_name)
    if isinstance(card_row, dict):
        projected_status = str(card_row.get("historical_card_status") or "").strip() or projected_status
        projected_severity = str(card_row.get("historical_card_severity") or "").strip() or projected_severity
        projected_retirement_state = (
            str(card_row.get("historical_card_retirement_state") or "").strip() or projected_retirement_state
        )
        projection_state = "resolved_historical_projected"
        projection_reason = (
            str(card_row.get("historical_status_normalization_reason") or "").strip() or "historical_card_projection"
        )
        current_reconciled_now = bool(card_row.get("current_reconciled_now") is True) or current_reconciled_now
        current_health_state = str(card_row.get("current_health_state") or "").strip() or current_health_state
    elif current_reconciled_now:
        projected_status = "historical_failed_reconciled_now"
        projected_severity = "info"
        projected_retirement_state = "resolved_historical_age_unknown"
        projection_state = "resolved_historical_inferred"
        projection_reason = "current_authority_healthy_now_no_card_projection_row"

    projected_row = dict(row)
    projected_row["status"] = projected_status
    projected_row["status_observed"] = observed_status
    projected_row["status_projection_state"] = projection_state
    projected_row["status_projection_reason"] = projection_reason
    projected_row["status_severity"] = projected_severity
    projected_row["status_retirement_state"] = projected_retirement_state
    projected_row["current_reconciled_now"] = current_reconciled_now
    projected_row["current_health_state"] = current_health_state
    historical_failed_session_rows_projected.append(projected_row)

historical_failed_session_status_projection_state = "none"
if any(str(row.get("status_projection_state") or "").startswith("resolved_historical") for row in historical_failed_session_rows_projected):
    historical_failed_session_status_projection_state = "projected"
elif residue_state == "unknown_status_metadata_unavailable":
    historical_failed_session_status_projection_state = "unknown_status_metadata_unavailable"

historical_failed_session_status_projection_values = dedupe(
    [str(row.get("status") or "") for row in historical_failed_session_rows_projected]
)

runtime_projection = summary.get("runtime_recovery_projection")
if not isinstance(runtime_projection, dict):
    runtime_projection = {}
runtime_projection["historical_failed_session_residue_names"] = historical_failed_session_residue_names
runtime_projection["historical_failed_session_residue_count"] = len(historical_failed_session_residue_names)
runtime_projection["historical_failed_session_residue_state"] = residue_state
runtime_projection["historical_failed_session_status_metadata_available"] = status_metadata_available
runtime_projection["historical_failed_session_cron_rows_seen"] = cron_session_rows_seen
runtime_projection["historical_failed_session_named_status_rows_seen"] = named_status_rows_seen
runtime_projection["historical_failed_session_residue_ownership_projection"] = {
    "state": historical_failed_session_residue_ownership_state,
    "families": historical_failed_session_residue_families,
    "action_lanes": historical_failed_session_residue_action_lanes,
    "recovery_owners": historical_failed_session_residue_recovery_owners,
    "rows": [
        {
            "name": row.get("name"),
            "family": row.get("family"),
            "action_lane": row.get("action_lane"),
            "recovery_owner": row.get("recovery_owner"),
            "routing_hint": row.get("routing_hint"),
            "ownership_source": row.get("ownership_source"),
        }
        for row in historical_failed_session_residue_rows
    ],
}
runtime_projection["historical_failed_session_residue_playbook_projection"] = {
    "state": historical_failed_session_residue_playbook_state,
    "families": historical_failed_session_residue_families,
    "hints": historical_failed_session_residue_playbook_hints_union,
    "rows": [
        {
            "name": row.get("name"),
            "family": row.get("family"),
            "playbook_hints": row.get("playbook_hints") if isinstance(row.get("playbook_hints"), list) else [],
        }
        for row in historical_failed_session_residue_rows
    ],
}
runtime_projection["historical_failed_session_residue_current_health_projection"] = {
    "state": historical_failed_session_residue_current_health_state,
    "rows": [
        {
            "name": row.get("name"),
            "current_health_state": row.get("current_health_state"),
            "current_reconciled_now": bool(row.get("current_reconciled_now") is True),
            "current_last_status": row.get("current_last_status"),
            "current_last_run_status": row.get("current_last_run_status"),
            "current_consecutive_errors": row.get("current_consecutive_errors"),
        }
        for row in historical_failed_session_residue_rows
    ],
}
runtime_projection["historical_failed_session_residue_decay_projection"] = {
    "state": historical_failed_session_residue_decay_state,
    "urgency": historical_failed_session_residue_decay_urgency,
    "staleness_buckets": historical_failed_session_residue_staleness_buckets,
    "age_known_count": historical_failed_session_residue_age_known_count,
    "age_unknown_count": historical_failed_session_residue_age_unknown_count,
    "oldest_age_sec": historical_failed_session_residue_oldest_age_sec,
    "newest_age_sec": historical_failed_session_residue_newest_age_sec,
    "rows": [
        {
            "name": row.get("name"),
            "last_failed_status": row.get("last_failed_status"),
            "last_failed_at_ms": row.get("last_failed_at_ms"),
            "residue_age_sec": row.get("residue_age_sec"),
            "staleness_bucket": row.get("staleness_bucket"),
        }
        for row in historical_failed_session_residue_rows
    ],
}
runtime_projection["historical_failed_session_card_projection"] = {
    "state": historical_failed_session_card_projection_state,
    "status_projection_state": historical_failed_session_card_status_projection_state,
    "status_projection_values": historical_failed_session_card_status_projection_values,
    "severity_projection_state": historical_failed_session_card_severity_projection_state,
    "severity_projection_values": historical_failed_session_card_severity_projection_values,
    "retirement_projection_state": historical_failed_session_card_retirement_projection_state,
    "retirement_projection_values": historical_failed_session_card_retirement_projection_values,
    "rows": historical_failed_session_card_projection_rows,
}
runtime_projection["repeated_false_green_operator_interpretation_guard"] = {
    "state": repeated_false_green_guard_state,
    "reason": repeated_false_green_guard_reason,
    "rows": repeated_false_green_guard_rows,
    "row_count": len(repeated_false_green_guard_rows),
}
runtime_projection["historical_failed_session_status_projection"] = {
    "state": historical_failed_session_status_projection_state,
    "status_projection_values": historical_failed_session_status_projection_values,
    "rows": historical_failed_session_rows_projected,
}
summary["runtime_recovery_projection"] = runtime_projection

summary["session_surface_reconciliation"] = {
    "active_window_minutes": active_window_minutes,
    "status_metadata_available": status_metadata_available,
    "cron_session_rows_seen": cron_session_rows_seen,
    "named_status_rows_seen": named_status_rows_seen,
    "historical_failed_session_residue_state": residue_state,
    "healthy_current_authority_names": healthy_current_names,
    "healthy_current_authority_count": len(healthy_current_names),
    "historical_failed_session_names": failed_session_names,
    "historical_failed_session_rows": historical_failed_session_rows_projected,
    "historical_failed_session_count": len(failed_session_names),
    "historical_failed_session_residue_names": historical_failed_session_residue_names,
    "historical_failed_session_residue_count": len(historical_failed_session_residue_names),
    "historical_failed_session_status_projection": {
        "state": historical_failed_session_status_projection_state,
        "status_projection_values": historical_failed_session_status_projection_values,
        "rows": historical_failed_session_rows_projected,
    },
    "historical_failed_session_residue_decay_state": historical_failed_session_residue_decay_state,
    "historical_failed_session_residue_decay_urgency": historical_failed_session_residue_decay_urgency,
    "historical_failed_session_residue_ownership_projection": {
        "state": historical_failed_session_residue_ownership_state,
        "families": historical_failed_session_residue_families,
        "action_lanes": historical_failed_session_residue_action_lanes,
        "recovery_owners": historical_failed_session_residue_recovery_owners,
        "rows": [
            {
                "name": row.get("name"),
                "family": row.get("family"),
                "action_lane": row.get("action_lane"),
                "recovery_owner": row.get("recovery_owner"),
                "routing_hint": row.get("routing_hint"),
                "ownership_source": row.get("ownership_source"),
            }
            for row in historical_failed_session_residue_rows
        ],
    },
    "historical_failed_session_residue_playbook_projection": {
        "state": historical_failed_session_residue_playbook_state,
        "families": historical_failed_session_residue_families,
        "hints": historical_failed_session_residue_playbook_hints_union,
        "rows": [
            {
                "name": row.get("name"),
                "family": row.get("family"),
                "playbook_hints": row.get("playbook_hints") if isinstance(row.get("playbook_hints"), list) else [],
            }
            for row in historical_failed_session_residue_rows
        ],
    },
    "historical_failed_session_residue_current_health_projection": {
        "state": historical_failed_session_residue_current_health_state,
        "rows": [
            {
                "name": row.get("name"),
                "current_health_state": row.get("current_health_state"),
                "current_reconciled_now": bool(row.get("current_reconciled_now") is True),
                "current_last_status": row.get("current_last_status"),
                "current_last_run_status": row.get("current_last_run_status"),
                "current_consecutive_errors": row.get("current_consecutive_errors"),
            }
            for row in historical_failed_session_residue_rows
        ],
    },
    "historical_failed_session_residue_decay_projection": {
        "state": historical_failed_session_residue_decay_state,
        "urgency": historical_failed_session_residue_decay_urgency,
        "staleness_buckets": historical_failed_session_residue_staleness_buckets,
        "age_known_count": historical_failed_session_residue_age_known_count,
        "age_unknown_count": historical_failed_session_residue_age_unknown_count,
        "oldest_age_sec": historical_failed_session_residue_oldest_age_sec,
        "newest_age_sec": historical_failed_session_residue_newest_age_sec,
        "rows": [
            {
                "name": row.get("name"),
                "last_failed_status": row.get("last_failed_status"),
                "last_failed_at_ms": row.get("last_failed_at_ms"),
                "residue_age_sec": row.get("residue_age_sec"),
                "staleness_bucket": row.get("staleness_bucket"),
            }
            for row in historical_failed_session_residue_rows
        ],
    },
    "historical_failed_session_card_projection": {
        "state": historical_failed_session_card_projection_state,
        "status_projection_state": historical_failed_session_card_status_projection_state,
        "status_projection_values": historical_failed_session_card_status_projection_values,
        "severity_projection_state": historical_failed_session_card_severity_projection_state,
        "severity_projection_values": historical_failed_session_card_severity_projection_values,
        "retirement_projection_state": historical_failed_session_card_retirement_projection_state,
        "retirement_projection_values": historical_failed_session_card_retirement_projection_values,
        "rows": historical_failed_session_card_projection_rows,
    },
    "repeated_false_green_operator_interpretation_guard": {
        "state": repeated_false_green_guard_state,
        "reason": repeated_false_green_guard_reason,
        "rows": repeated_false_green_guard_rows,
        "row_count": len(repeated_false_green_guard_rows),
    },
}

print(json.dumps(summary, ensure_ascii=False))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cron-json)
      CRON_JSON_PATH="${2:-}"
      shift 2
      ;;
    --sessions-json)
      SESSIONS_JSON_PATH="${2:-}"
      shift 2
      ;;
    --expected-names)
      EXPECTED_NAMES_CSV="${2:-}"
      shift 2
      ;;
    --json)
      JSON_OUT=1
      shift
      ;;
    --strict)
      STRICT_EXIT=1
      shift
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

payload_json=""
if [[ -n "$CRON_JSON_PATH" ]]; then
  if [[ ! -f "$CRON_JSON_PATH" ]]; then
    detail="path=$CRON_JSON_PATH"
    summary_json="$(_emit_guard_summary_json "{\"ok\":false,\"classification\":\"cron_contract_drift\",\"classification_bucket\":\"contract\",\"error\":\"cron_json_input_missing\",\"detail\":\"$detail\",\"failure_codes\":{\"all\":[\"contract:cron_json_input_missing\"],\"critical\":[\"contract:cron_json_input_missing\"],\"policy\":[],\"contract\":[\"contract:cron_json_input_missing\"],\"connectivity\":[]}}")"
    echo "BLOCKER: no_llm_watchdog_cron_authority_contract_drift; reason=cron_json_input_missing; ${detail}"
    [[ "$JSON_OUT" -eq 1 ]] && echo "$summary_json"
    [[ "$STRICT_EXIT" -eq 1 ]] && exit 1 || exit 0
  fi
  payload_json="$(cat "$CRON_JSON_PATH")"
else
  set +e
  payload_json="$(openclaw cron list --json 2>/tmp/no_llm_watchdog_cron_authority_guard.err)"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    err_raw="$(cat /tmp/no_llm_watchdog_cron_authority_guard.err 2>/dev/null || true)"
    err="$(_sanitize_inline_text "$err_raw")"

    if _is_gateway_connectivity_failure "$err"; then
      summary_json="$(_emit_guard_summary_json "{\"ok\":false,\"classification\":\"gateway_connectivity_failure\",\"classification_bucket\":\"connectivity\",\"error\":\"cron_list_unreachable\",\"detail\":\"rc=${rc}; err=${err:0:180}\",\"failure_codes\":{\"all\":[\"connectivity:cron_list_unreachable\"],\"critical\":[\"connectivity:cron_list_unreachable\"],\"policy\":[],\"contract\":[],\"connectivity\":[\"connectivity:cron_list_unreachable\"]}}")"
      echo "BLOCKER: no_llm_watchdog_cron_authority_gateway_connectivity_failure; reason=cron_list_unreachable; rc=${rc}; err=${err:0:180}"
    else
      summary_json="$(_emit_guard_summary_json "{\"ok\":false,\"classification\":\"cron_contract_drift\",\"classification_bucket\":\"contract\",\"error\":\"cron_list_failed\",\"detail\":\"rc=${rc}; err=${err:0:180}\",\"failure_codes\":{\"all\":[\"contract:cron_list_failed\"],\"critical\":[\"contract:cron_list_failed\"],\"policy\":[],\"contract\":[\"contract:cron_list_failed\"],\"connectivity\":[]}}")"
      echo "BLOCKER: no_llm_watchdog_cron_authority_contract_drift; reason=cron_list_failed; rc=${rc}; err=${err:0:180}"
    fi

    [[ "$JSON_OUT" -eq 1 ]] && echo "$summary_json"
    [[ "$STRICT_EXIT" -eq 1 ]] && exit 1 || exit 0
  fi
fi

EXPECTED_NAMES_CSV="$(python3 - "$payload_json" "$EXPECTED_NAMES_CSV" <<'PY'
import json
import sys

payload_raw = str(sys.argv[1] or "")
expected_csv = str(sys.argv[2] or "")
expected = []
for token in expected_csv.split(","):
    name = token.strip()
    if not name or name in expected:
        continue
    expected.append(name)

try:
    obj = json.loads(payload_raw)
except Exception:
    obj = {}

jobs = obj.get("jobs") if isinstance(obj, dict) else []
if not isinstance(jobs, list):
    jobs = []

auto_name = "core-roadmap-executor-idle-watchdog"
auto_enabled = any(
    isinstance(job, dict)
    and bool(job.get("enabled", False))
    and str(job.get("name") or "").strip() == auto_name
    for job in jobs
)
if auto_enabled and auto_name not in expected:
    expected.append(auto_name)

print(",".join(expected))
PY
)"

session_failed_names_json='{"failed_names":[],"failed_rows":[],"cron_session_rows_seen":0,"named_status_rows_seen":0,"status_metadata_available":false}'
if [[ -n "$SESSIONS_JSON_PATH" ]]; then
  session_failed_names_json="$(_extract_failed_cron_session_names_json "$SESSIONS_JSON_PATH")"
elif [[ -z "$CRON_JSON_PATH" ]]; then
  sessions_tmp="$(mktemp)"
  set +e
  openclaw sessions --active "$SESSION_ACTIVE_MINUTES" --json >"$sessions_tmp" 2>/tmp/no_llm_watchdog_cron_authority_guard_sessions.err
  sessions_rc=$?
  set -e
  if [[ "$sessions_rc" -eq 0 ]]; then
    session_failed_names_json="$(_extract_failed_cron_session_names_json "$sessions_tmp")"
  fi
  rm -f "$sessions_tmp" 2>/dev/null || true
fi

summary_json="$(_emit_guard_summary_json "$payload_json")"
summary_json="$(_augment_summary_with_session_surface_reconciliation "$summary_json" "$session_failed_names_json" "$SESSION_ACTIVE_MINUTES")"

readarray -t summary_fields < <(python3 - 3<<<"$summary_json" <<'PY'
import json
import os
import hashlib


def dedupe(items):
    out = []
    seen = set()
    for item in items:
        txt = str(item or '').strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


with os.fdopen(3, 'r', encoding='utf-8') as fp:
    obj = json.loads(fp.read())
rows = obj.get('violations') or []
ids = [str(r.get('id') or '') for r in rows if isinstance(r, dict)]
ids = [x for x in ids if x]
runtime_rows = [r for r in rows if isinstance(r, dict) and str(r.get('bucket') or '').strip().lower() == 'runtime']
runtime_names = dedupe([str(r.get('name') or '') for r in runtime_rows])
runtime_codes = dedupe(
    [str(code or '') for r in runtime_rows for code in ((r.get('codes') or []) if isinstance(r.get('codes'), list) else [])]
)


def row_recovery_state(row_codes):
    row_code_set = set(row_codes or [])
    if 'runtime:enabled_authority_retry_scheduled' in row_code_set:
        return 'retrying'
    if 'runtime:enabled_authority_intervention_required' in row_code_set:
        return 'intervention_required'
    return 'intervention_required'


def row_escalation_policy(row_root_causes, recovery_state):
    root_set = set(row_root_causes or [])
    if 'runtime:root_cause_provider_quota' in root_set:
        return 'provider_quota_billing_intervention_required'
    if 'runtime:root_cause_provider_auth' in root_set:
        return 'provider_auth_intervention_required'
    if 'runtime:root_cause_provider_rate_limited' in root_set:
        return 'provider_rate_limit_backoff_retry_later'
    if recovery_state == 'retrying':
        return 'runtime_retry_backoff_retry_later'
    return 'runtime_operator_intervention_required'


def runtime_family_ownership_for_name(name):
    normalized = str(name or '').strip().lower()
    if normalized in {'continuity:backup-checkpoint-90m', 'continuity:stale-progress-45m', 'core-roadmap-executor-idle-watchdog'}:
        return {
            'family': 'continuity',
            'action_lane': 'continuity_no_nudge_authority',
            'recovery_owner': 'continuity_control_plane_operator',
        }
    if normalized in {'web-capture-scheduler-governance-watchdog', 'web-capture-scheduler-governance'}:
        return {
            'family': 'web_capture',
            'action_lane': 'web_capture_scheduler_governance_watchdog',
            'recovery_owner': 'web_capture_governance_operator',
        }
    if normalized in {'obsidian:hourly-canary', 'obsidian-vault-tick-hourly-canary', 'obsidian:vault-tick-hourly'}:
        return {
            'family': 'obsidian',
            'action_lane': 'obsidian_hourly_integrity_watchdogs',
            'recovery_owner': 'obsidian_memory_integrity_operator',
        }
    if normalized.startswith('continuity:'):
        return {
            'family': 'continuity',
            'action_lane': 'continuity_no_nudge_authority',
            'recovery_owner': 'continuity_control_plane_operator',
        }
    if normalized.startswith('web-capture'):
        return {
            'family': 'web_capture',
            'action_lane': 'web_capture_scheduler_governance_watchdog',
            'recovery_owner': 'web_capture_governance_operator',
        }
    if normalized.startswith('obsidian'):
        return {
            'family': 'obsidian',
            'action_lane': 'obsidian_hourly_integrity_watchdogs',
            'recovery_owner': 'obsidian_memory_integrity_operator',
        }
    return {
        'family': 'unknown',
        'action_lane': 'unknown',
        'recovery_owner': 'operator_general_triage',
    }


def runtime_family_playbook_hints_for_family(family, recovery_state, row_retryable=None):
    normalized_family = str(family or '').strip().lower()
    normalized_recovery_state = str(recovery_state or '').strip().lower()
    family_mapping = {
        'continuity': [
            'runtime:playbook_continuity_run_no_nudge_authority_watchdog_contract',
            'runtime:playbook_core_roadmap_floor_refill_loop_contract',
            'runtime:playbook_continuity_validate_checkpoint_and_stale_progress_sessions',
        ],
        'web_capture': [
            'runtime:playbook_web_capture_run_scheduler_governance_watchdog_contract',
            'runtime:playbook_web_capture_reconcile_scheduler_manifest_and_capture_backlog',
        ],
        'obsidian': [
            'runtime:playbook_obsidian_run_hourly_canary_and_vault_tick_contracts',
            'runtime:playbook_obsidian_verify_vault_materialization_and_memory_sync',
        ],
    }
    hints = list(family_mapping.get(normalized_family, []))
    if not hints:
        hints = ['runtime:playbook_operator_general_triage_collect_runtime_artifacts']
    if row_retryable is True:
        hints.append('runtime:playbook_recovery_wait_for_next_scheduled_retry_and_verify_clearance')
    elif row_retryable is False:
        hints.append('runtime:playbook_recovery_intervention_required_route_owner_now')
    elif normalized_recovery_state == 'retrying':
        hints.append('runtime:playbook_recovery_wait_for_next_scheduled_retry_and_verify_clearance')
    else:
        hints.append('runtime:playbook_recovery_intervention_required_route_owner_now')
    return dedupe(hints)


def historical_session_residue_playbook_hints_for_family(family):
    normalized_family = str(family or '').strip().lower()
    family_mapping = {
        'continuity': [
            'runtime:playbook_continuity_run_no_nudge_authority_watchdog_contract',
            'runtime:playbook_core_roadmap_floor_refill_loop_contract',
            'runtime:playbook_continuity_validate_checkpoint_and_stale_progress_sessions',
        ],
        'web_capture': [
            'runtime:playbook_web_capture_run_scheduler_governance_watchdog_contract',
            'runtime:playbook_web_capture_reconcile_scheduler_manifest_and_capture_backlog',
        ],
        'obsidian': [
            'runtime:playbook_obsidian_run_hourly_canary_and_vault_tick_contracts',
            'runtime:playbook_obsidian_verify_vault_materialization_and_memory_sync',
        ],
    }
    hints = list(family_mapping.get(normalized_family, []))
    if not hints:
        hints = ['runtime:playbook_operator_general_triage_collect_runtime_artifacts']
    hints.append('runtime:playbook_historical_session_residue_verify_current_rail_health_and_keep_low_urgency')
    return dedupe(hints)


runtime_transition_rows = []
for row in runtime_rows:
    row_name = str(row.get('name') or '').strip() or 'unknown'
    row_codes = dedupe([
        str(code or '').strip()
        for code in ((row.get('codes') or []) if isinstance(row.get('codes'), list) else [])
    ])
    row_root_causes = dedupe([code for code in row_codes if code.startswith('runtime:root_cause_')])
    row_recovery = row_recovery_state(row_codes)
    row_policy = row_escalation_policy(row_root_causes, row_recovery)
    runtime_transition_rows.append(
        f"{row_name}|recovery_state={row_recovery}|escalation_policy={row_policy}|root_causes={','.join(row_root_causes) or 'none'}"
    )

runtime_transition_rows = dedupe(sorted(runtime_transition_rows))
runtime_transition_signature = 'none'
if runtime_transition_rows:
    runtime_transition_signature = hashlib.sha256(
        '\n'.join(runtime_transition_rows).encode('utf-8')
    ).hexdigest()[:16]

runtime_root_causes = dedupe([code for code in runtime_codes if code.startswith('runtime:root_cause_')])
runtime_remediation_projection = obj.get('runtime_remediation_projection') if isinstance(obj.get('runtime_remediation_projection'), dict) else {}
runtime_remediation_hints = dedupe(
    [
        str(hint or '')
        for hint in ((runtime_remediation_projection.get('hints') or []) if isinstance(runtime_remediation_projection.get('hints'), list) else [])
    ]
)
runtime_escalation_projection = obj.get('runtime_escalation_policy_projection') if isinstance(obj.get('runtime_escalation_policy_projection'), dict) else {}
runtime_escalation_policy = str(runtime_escalation_projection.get('policy') or 'none').strip() or 'none'
runtime_escalation_urgency = str(runtime_escalation_projection.get('urgency') or 'none').strip() or 'none'
runtime_escalation_source = str(runtime_escalation_projection.get('source') or 'none').strip() or 'none'
runtime_escalation_breakdown = obj.get('runtime_escalation_policy_breakdown') if isinstance(obj.get('runtime_escalation_policy_breakdown'), dict) else {}
runtime_escalation_scope = str(runtime_escalation_breakdown.get('state') or 'none').strip() or 'none'
runtime_escalation_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('policy') or '').strip()}"
        for row in ((runtime_escalation_breakdown.get('rows') or []) if isinstance(runtime_escalation_breakdown.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and str(row.get('policy') or '').strip()
    ]
)
if not runtime_escalation_per_rail:
    fallback_escalation_rows = []
    for row in runtime_rows:
        if not isinstance(row, dict):
            continue
        row_name = str(row.get('name') or '').strip()
        if not row_name:
            continue
        row_codes = dedupe([
            str(code or '').strip()
            for code in ((row.get('codes') or []) if isinstance(row.get('codes'), list) else [])
        ])
        row_root_causes = dedupe([code for code in row_codes if code.startswith('runtime:root_cause_')])
        row_recovery = row_recovery_state(row_codes)
        row_policy = row_escalation_policy(row_root_causes, row_recovery)
        fallback_escalation_rows.append(f"{row_name}={row_policy}")
    runtime_escalation_per_rail = dedupe(sorted(fallback_escalation_rows))
if runtime_escalation_scope == 'none' and runtime_escalation_per_rail:
    policy_set = {
        token.split('=', 1)[1]
        for token in runtime_escalation_per_rail
        if '=' in token and token.split('=', 1)[1]
    }
    runtime_escalation_scope = 'single' if len(policy_set) <= 1 else 'mixed'

runtime_family_ownership_projection = obj.get('runtime_family_ownership_projection') if isinstance(obj.get('runtime_family_ownership_projection'), dict) else {}
runtime_ownership_scope = str(runtime_family_ownership_projection.get('state') or 'none').strip() or 'none'
runtime_ownership_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('family') or '').strip()}|{str(row.get('action_lane') or '').strip()}|{str(row.get('recovery_owner') or '').strip()}"
        for row in ((runtime_family_ownership_projection.get('rows') or []) if isinstance(runtime_family_ownership_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and str(row.get('family') or '').strip()
        and str(row.get('action_lane') or '').strip()
        and str(row.get('recovery_owner') or '').strip()
    ]
)
if not runtime_ownership_per_rail:
    runtime_ownership_per_rail = dedupe(
        sorted(
            [
                f"{name}={runtime_family_ownership_for_name(name).get('family','unknown')}|{runtime_family_ownership_for_name(name).get('action_lane','unknown')}|{runtime_family_ownership_for_name(name).get('recovery_owner','operator_general_triage')}"
                for name in runtime_names
                if str(name or '').strip()
            ]
        )
)
if runtime_ownership_scope == 'none' and runtime_ownership_per_rail:
    runtime_ownership_scope = 'projected'

runtime_family_playbook_projection = obj.get('runtime_family_playbook_projection') if isinstance(obj.get('runtime_family_playbook_projection'), dict) else {}
runtime_playbook_scope = str(runtime_family_playbook_projection.get('state') or 'none').strip() or 'none'
runtime_playbook_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={'+'.join(dedupe([str(h or '').strip() for h in ((row.get('playbook_hints') or []) if isinstance(row.get('playbook_hints'), list) else []) if str(h or '').strip()]))}"
        for row in ((runtime_family_playbook_projection.get('rows') or []) if isinstance(runtime_family_playbook_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and isinstance(row.get('playbook_hints'), list)
        and len([str(h or '').strip() for h in (row.get('playbook_hints') or []) if str(h or '').strip()]) > 0
    ]
)
if not runtime_playbook_per_rail:
    fallback_playbook_rows = []
    for row in runtime_rows:
        if not isinstance(row, dict):
            continue
        row_name = str(row.get('name') or '').strip()
        if not row_name:
            continue
        row_codes = dedupe([
            str(code or '').strip()
            for code in ((row.get('codes') or []) if isinstance(row.get('codes'), list) else [])
        ])
        row_root_causes = dedupe([code for code in row_codes if code.startswith('runtime:root_cause_')])
        row_recovery = row_recovery_state(row_codes)
        row_policy = row_escalation_policy(row_root_causes, row_recovery)
        row_family = str(runtime_family_ownership_for_name(row_name).get('family') or 'unknown')
        row_retryable = row_policy in {'provider_rate_limit_backoff_retry_later', 'runtime_retry_backoff_retry_later'}
        row_hints = runtime_family_playbook_hints_for_family(row_family, row_recovery, row_retryable)
        if row_hints:
            fallback_playbook_rows.append(f"{row_name}={'+'.join(row_hints)}")
    runtime_playbook_per_rail = dedupe(sorted(fallback_playbook_rows))
if runtime_playbook_scope == 'none' and runtime_playbook_per_rail:
    runtime_playbook_scope = 'projected'

runtime_recovery_state = str(obj.get('runtime_recovery_state') or 'none').strip() or 'none'
runtime_retryable = bool(obj.get('failure_retryable') is True)
runtime_projection = obj.get('runtime_recovery_projection') if isinstance(obj.get('runtime_recovery_projection'), dict) else {}
runtime_failure_provenance = obj.get('runtime_failure_provenance') if isinstance(obj.get('runtime_failure_provenance'), dict) else {}
session_surface_reconciliation = obj.get('session_surface_reconciliation') if isinstance(obj.get('session_surface_reconciliation'), dict) else {}
historical_recovered_names = dedupe(
    [str(name or '') for name in ((runtime_projection.get('historical_recovered_names') or []) if isinstance(runtime_projection.get('historical_recovered_names'), list) else [])]
)
historical_failed_session_residue_names = dedupe(
    [str(name or '') for name in ((runtime_projection.get('historical_failed_session_residue_names') or []) if isinstance(runtime_projection.get('historical_failed_session_residue_names'), list) else [])]
)
historical_failed_session_residue_state = str(runtime_projection.get('historical_failed_session_residue_state') or '').strip().lower() or 'none_observed'

historical_residue_ownership_projection = session_surface_reconciliation.get('historical_failed_session_residue_ownership_projection') if isinstance(session_surface_reconciliation.get('historical_failed_session_residue_ownership_projection'), dict) else {}
historical_residue_ownership_scope = str(historical_residue_ownership_projection.get('state') or 'none').strip() or 'none'
historical_residue_ownership_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('family') or '').strip()}|{str(row.get('action_lane') or '').strip()}|{str(row.get('recovery_owner') or '').strip()}"
        for row in ((historical_residue_ownership_projection.get('rows') or []) if isinstance(historical_residue_ownership_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and str(row.get('family') or '').strip()
        and str(row.get('action_lane') or '').strip()
        and str(row.get('recovery_owner') or '').strip()
    ]
)
if not historical_residue_ownership_per_rail:
    historical_residue_ownership_per_rail = dedupe(
        sorted(
            [
                f"{name}={runtime_family_ownership_for_name(name).get('family','unknown')}|{runtime_family_ownership_for_name(name).get('action_lane','unknown')}|{runtime_family_ownership_for_name(name).get('recovery_owner','operator_general_triage')}"
                for name in historical_failed_session_residue_names
                if str(name or '').strip()
            ]
        )
    )
if historical_residue_ownership_scope == 'none' and historical_residue_ownership_per_rail:
    historical_residue_ownership_scope = 'projected'

historical_residue_playbook_projection = session_surface_reconciliation.get('historical_failed_session_residue_playbook_projection') if isinstance(session_surface_reconciliation.get('historical_failed_session_residue_playbook_projection'), dict) else {}
historical_residue_playbook_scope = str(historical_residue_playbook_projection.get('state') or 'none').strip() or 'none'
historical_residue_playbook_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={'+'.join(dedupe([str(h or '').strip() for h in ((row.get('playbook_hints') or []) if isinstance(row.get('playbook_hints'), list) else []) if str(h or '').strip()]))}"
        for row in ((historical_residue_playbook_projection.get('rows') or []) if isinstance(historical_residue_playbook_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and isinstance(row.get('playbook_hints'), list)
        and len([str(h or '').strip() for h in (row.get('playbook_hints') or []) if str(h or '').strip()]) > 0
    ]
)
if not historical_residue_playbook_per_rail:
    fallback_historical_playbook_rows = []
    for row_name in historical_failed_session_residue_names:
        if not str(row_name or '').strip():
            continue
        row_family = str(runtime_family_ownership_for_name(row_name).get('family') or 'unknown')
        row_hints = historical_session_residue_playbook_hints_for_family(row_family)
        if row_hints:
            fallback_historical_playbook_rows.append(f"{row_name}={'+'.join(row_hints)}")
    historical_residue_playbook_per_rail = dedupe(sorted(fallback_historical_playbook_rows))
if historical_residue_playbook_scope == 'none' and historical_residue_playbook_per_rail:
    historical_residue_playbook_scope = 'projected'

historical_residue_current_health_projection = session_surface_reconciliation.get('historical_failed_session_residue_current_health_projection') if isinstance(session_surface_reconciliation.get('historical_failed_session_residue_current_health_projection'), dict) else {}
if not historical_residue_current_health_projection:
    historical_residue_current_health_projection = runtime_projection.get('historical_failed_session_residue_current_health_projection') if isinstance(runtime_projection.get('historical_failed_session_residue_current_health_projection'), dict) else {}
historical_residue_current_health_scope = str(historical_residue_current_health_projection.get('state') or 'none').strip() or 'none'
historical_residue_current_health_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('current_health_state') or '').strip() or 'healthy_now'}|reconciled_now={'1' if row.get('current_reconciled_now') is True else '0'}|last_status={str(row.get('current_last_status') or '').strip() or 'unknown'}|last_run_status={str(row.get('current_last_run_status') or '').strip() or 'unknown'}|consecutive_errors={str(row.get('current_consecutive_errors')) if isinstance(row.get('current_consecutive_errors'), int) else 'unknown'}"
        for row in ((historical_residue_current_health_projection.get('rows') or []) if isinstance(historical_residue_current_health_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
if not historical_residue_current_health_per_rail and historical_failed_session_residue_names:
    historical_residue_current_health_per_rail = [
        f"{name}=healthy_now|reconciled_now=1|last_status=unknown|last_run_status=unknown|consecutive_errors=unknown"
        for name in historical_failed_session_residue_names
        if str(name or '').strip()
    ]
if historical_residue_current_health_scope == 'none' and historical_residue_current_health_per_rail:
    historical_residue_current_health_scope = 'projected'
if historical_failed_session_residue_state == 'unknown_status_metadata_unavailable' and not historical_residue_current_health_per_rail:
    historical_residue_current_health_scope = 'unknown_status_metadata_unavailable'

historical_residue_decay_projection = session_surface_reconciliation.get('historical_failed_session_residue_decay_projection') if isinstance(session_surface_reconciliation.get('historical_failed_session_residue_decay_projection'), dict) else {}
historical_residue_decay_state = str(historical_residue_decay_projection.get('state') or 'none').strip() or 'none'
historical_residue_decay_urgency = str(historical_residue_decay_projection.get('urgency') or 'none').strip() or 'none'
historical_residue_staleness_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('staleness_bucket') or '').strip()}"
        for row in ((historical_residue_decay_projection.get('rows') or []) if isinstance(historical_residue_decay_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
        and str(row.get('staleness_bucket') or '').strip()
    ]
)
if not historical_residue_staleness_per_rail and historical_failed_session_residue_names:
    historical_residue_staleness_per_rail = [f"{name}=age_unknown" for name in historical_failed_session_residue_names if str(name or '').strip()]

historical_residue_card_projection = session_surface_reconciliation.get('historical_failed_session_card_projection') if isinstance(session_surface_reconciliation.get('historical_failed_session_card_projection'), dict) else {}
if not historical_residue_card_projection:
    historical_residue_card_projection = runtime_projection.get('historical_failed_session_card_projection') if isinstance(runtime_projection.get('historical_failed_session_card_projection'), dict) else {}
historical_residue_card_scope = str(historical_residue_card_projection.get('state') or 'none').strip() or 'none'
historical_residue_card_status_scope = str(historical_residue_card_projection.get('status_projection_state') or 'none').strip() or 'none'
historical_residue_card_severity_scope = str(historical_residue_card_projection.get('severity_projection_state') or 'none').strip() or 'none'
historical_residue_card_retirement_scope = str(historical_residue_card_projection.get('retirement_projection_state') or 'none').strip() or 'none'
historical_residue_card_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}=cards={int(row.get('failed_card_count') or 0)}|last_key={str(row.get('last_failed_session_key') or '').strip() or 'none'}|status_source={str(row.get('last_failed_status_source') or '').strip() or 'unknown'}"
        for row in ((historical_residue_card_projection.get('rows') or []) if isinstance(historical_residue_card_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
historical_residue_card_status_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('historical_card_status') or '').strip() or 'historical_failed_status_unknown'}"
        for row in ((historical_residue_card_projection.get('rows') or []) if isinstance(historical_residue_card_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
historical_residue_card_severity_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('historical_card_severity') or '').strip() or 'warn'}"
        for row in ((historical_residue_card_projection.get('rows') or []) if isinstance(historical_residue_card_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
historical_residue_card_retirement_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}={str(row.get('historical_card_retirement_state') or '').strip() or 'resolved_historical_retirement_unknown'}"
        for row in ((historical_residue_card_projection.get('rows') or []) if isinstance(historical_residue_card_projection.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
if historical_residue_card_scope == 'none' and historical_residue_card_per_rail:
    historical_residue_card_scope = 'projected'
if historical_residue_card_status_scope == 'none' and historical_residue_card_status_per_rail:
    historical_residue_card_status_scope = 'projected'
if historical_residue_card_severity_scope == 'none' and historical_residue_card_severity_per_rail:
    historical_residue_card_severity_scope = 'projected'
if historical_residue_card_retirement_scope == 'none' and historical_residue_card_retirement_per_rail:
    historical_residue_card_retirement_scope = 'projected'
if historical_failed_session_residue_state == 'unknown_status_metadata_unavailable' and not historical_residue_card_per_rail:
    historical_residue_card_scope = 'unknown_status_metadata_unavailable'
if historical_failed_session_residue_state == 'unknown_status_metadata_unavailable' and not historical_residue_card_status_per_rail:
    historical_residue_card_status_scope = 'unknown_status_metadata_unavailable'
if historical_failed_session_residue_state == 'unknown_status_metadata_unavailable' and not historical_residue_card_severity_per_rail:
    historical_residue_card_severity_scope = 'unknown_status_metadata_unavailable'
if historical_failed_session_residue_state == 'unknown_status_metadata_unavailable' and not historical_residue_card_retirement_per_rail:
    historical_residue_card_retirement_scope = 'unknown_status_metadata_unavailable'

repeated_false_green_guard = (
    session_surface_reconciliation.get('repeated_false_green_operator_interpretation_guard')
    if isinstance(session_surface_reconciliation.get('repeated_false_green_operator_interpretation_guard'), dict)
    else {}
)
if not repeated_false_green_guard:
    repeated_false_green_guard = (
        runtime_projection.get('repeated_false_green_operator_interpretation_guard')
        if isinstance(runtime_projection.get('repeated_false_green_operator_interpretation_guard'), dict)
        else {}
    )

repeated_false_green_guard_state = str(repeated_false_green_guard.get('state') or 'none').strip() or 'none'
repeated_false_green_guard_reason = str(repeated_false_green_guard.get('reason') or 'none').strip() or 'none'
repeated_false_green_guard_per_rail = dedupe(
    [
        f"{str(row.get('name') or '').strip()}=cards={int(row.get('failed_card_count') or 0)}|status={str(row.get('historical_card_status') or '').strip() or 'unknown'}"
        for row in ((repeated_false_green_guard.get('rows') or []) if isinstance(repeated_false_green_guard.get('rows'), list) else [])
        if isinstance(row, dict)
        and str(row.get('name') or '').strip()
    ]
)
if not repeated_false_green_guard_per_rail:
    fallback_rows = []
    for row in ((historical_residue_card_projection.get('rows') or []) if isinstance(historical_residue_card_projection.get('rows'), list) else []):
        if not isinstance(row, dict):
            continue
        row_name = str(row.get('name') or '').strip()
        if not row_name:
            continue
        row_count = int(row.get('failed_card_count') or 0)
        row_status = str(row.get('historical_card_status') or '').strip().lower()
        if row_count < 2:
            continue
        if row_status not in {'historical_failed_reconciled_now', 'historical_failed_reconciled'}:
            continue
        fallback_rows.append(
            f"{row_name}=cards={row_count}|status={str(row.get('historical_card_status') or '').strip() or 'unknown'}"
        )
    repeated_false_green_guard_per_rail = dedupe(sorted(fallback_rows))
if repeated_false_green_guard_state == 'none' and repeated_false_green_guard_per_rail:
    repeated_false_green_guard_state = 'active'
if repeated_false_green_guard_state == 'active' and repeated_false_green_guard_reason == 'none':
    repeated_false_green_guard_reason = 'repeated_historical_failed_session_cards_reconciled_now'

latest_failure_name = str(runtime_failure_provenance.get('latest_failure_name') or '').strip() or 'none'
latest_failure_at_ms = runtime_failure_provenance.get('latest_failure_at_ms')
latest_failure_at_ms_txt = str(latest_failure_at_ms) if isinstance(latest_failure_at_ms, int) else 'unknown'
latest_failure_reason_source = str(runtime_failure_provenance.get('latest_failure_reason_source') or 'unavailable').strip() or 'unavailable'
latest_failure_reason_excerpt = str(runtime_failure_provenance.get('latest_failure_reason_excerpt') or '').strip() or 'none'

print('1' if obj.get('ok') else '0')
print(int(obj.get('checked') or 0))
print(str(obj.get('classification') or 'cron_policy_failed'))
print(str(obj.get('error') or ''))
print(len(rows))
print(','.join(ids[:5]))
print(','.join(runtime_names[:5]))
print(','.join(runtime_codes[:5]))
print(','.join(runtime_root_causes[:5]))
print(','.join(runtime_remediation_hints[:5]))
print(runtime_transition_signature)
print(runtime_escalation_policy)
print(runtime_escalation_urgency)
print(runtime_escalation_source)
print(runtime_escalation_scope)
print(','.join(runtime_escalation_per_rail[:5]))
print(runtime_ownership_scope)
print(','.join(runtime_ownership_per_rail[:5]))
print(runtime_playbook_scope)
print(','.join(runtime_playbook_per_rail[:5]))
print(runtime_recovery_state)
print('1' if runtime_retryable else '0')
print(','.join(historical_recovered_names[:5]))
print(','.join(historical_failed_session_residue_names[:5]))
print(historical_residue_ownership_scope)
print(','.join(historical_residue_ownership_per_rail[:5]))
print(historical_residue_playbook_scope)
print(','.join(historical_residue_playbook_per_rail[:5]))
print(historical_residue_current_health_scope)
print(','.join(historical_residue_current_health_per_rail[:5]))
print(historical_residue_decay_state)
print(historical_residue_decay_urgency)
print(','.join(historical_residue_staleness_per_rail[:5]))
print(historical_residue_card_scope)
print(','.join(historical_residue_card_per_rail[:5]))
print(historical_residue_card_status_scope)
print(','.join(historical_residue_card_status_per_rail[:5]))
print(historical_residue_card_severity_scope)
print(','.join(historical_residue_card_severity_per_rail[:5]))
print(historical_residue_card_retirement_scope)
print(','.join(historical_residue_card_retirement_per_rail[:5]))
print(historical_failed_session_residue_state)
print(latest_failure_name)
print(latest_failure_at_ms_txt)
print(latest_failure_reason_source)
print(latest_failure_reason_excerpt)
print(repeated_false_green_guard_state)
print(repeated_false_green_guard_reason)
print(','.join(repeated_false_green_guard_per_rail[:5]))
PY
)

ok_flag="${summary_fields[0]:-0}"
checked_count="${summary_fields[1]:-0}"
classification="${summary_fields[2]:-cron_policy_failed}"
summary_error="${summary_fields[3]:-}"
viol_count="${summary_fields[4]:-0}"
viol_ids="${summary_fields[5]:-}"
runtime_names="${summary_fields[6]:-}"
runtime_codes="${summary_fields[7]:-}"
runtime_root_causes="${summary_fields[8]:-}"
runtime_remediation_hints="${summary_fields[9]:-}"
runtime_transition_signature="${summary_fields[10]:-none}"
runtime_escalation_policy="${summary_fields[11]:-none}"
runtime_escalation_urgency="${summary_fields[12]:-none}"
runtime_escalation_source="${summary_fields[13]:-none}"
runtime_escalation_scope="${summary_fields[14]:-none}"
runtime_escalation_per_rail="${summary_fields[15]:-}"
runtime_ownership_scope="${summary_fields[16]:-none}"
runtime_ownership_per_rail="${summary_fields[17]:-}"
runtime_playbook_scope="${summary_fields[18]:-none}"
runtime_playbook_per_rail="${summary_fields[19]:-}"
runtime_recovery_state="${summary_fields[20]:-none}"
runtime_retryable="${summary_fields[21]:-0}"
historical_recovered_names="${summary_fields[22]:-}"
historical_failed_session_residue_names="${summary_fields[23]:-}"
historical_failed_session_residue_ownership_scope="${summary_fields[24]:-none}"
historical_failed_session_residue_ownership_per_rail="${summary_fields[25]:-}"
historical_failed_session_residue_playbook_scope="${summary_fields[26]:-none}"
historical_failed_session_residue_playbook_per_rail="${summary_fields[27]:-}"
historical_failed_session_residue_current_health_scope="${summary_fields[28]:-none}"
historical_failed_session_residue_current_health_per_rail="${summary_fields[29]:-}"
historical_failed_session_residue_decay_state="${summary_fields[30]:-none}"
historical_failed_session_residue_decay_urgency="${summary_fields[31]:-none}"
historical_failed_session_residue_staleness_per_rail="${summary_fields[32]:-}"
historical_failed_session_residue_card_scope="${summary_fields[33]:-none}"
historical_failed_session_residue_card_per_rail="${summary_fields[34]:-}"
historical_failed_session_residue_card_status_scope="${summary_fields[35]:-none}"
historical_failed_session_residue_card_status_per_rail="${summary_fields[36]:-}"
historical_failed_session_residue_card_severity_scope="${summary_fields[37]:-none}"
historical_failed_session_residue_card_severity_per_rail="${summary_fields[38]:-}"
historical_failed_session_residue_card_retirement_scope="${summary_fields[39]:-none}"
historical_failed_session_residue_card_retirement_per_rail="${summary_fields[40]:-}"
historical_failed_session_residue_state="${summary_fields[41]:-none_observed}"
latest_failure_name="${summary_fields[42]:-none}"
latest_failure_at_ms="${summary_fields[43]:-unknown}"
latest_failure_reason_source="${summary_fields[44]:-unavailable}"
latest_failure_reason_excerpt="${summary_fields[45]:-none}"
historical_failed_session_repeated_false_green_guard_state="${summary_fields[46]:-none}"
historical_failed_session_repeated_false_green_guard_reason="${summary_fields[47]:-none}"
historical_failed_session_repeated_false_green_guard_per_rail="${summary_fields[48]:-}"

if [[ "$ok_flag" == "1" ]]; then
  ready_line="READY: no-llm watchdog cron authority contract is enforced; checked=${checked_count}"
  if [[ -n "$historical_recovered_names" ]]; then
    ready_line+="; recovered_historical_names=${historical_recovered_names}"
  fi
  if [[ -n "$historical_failed_session_residue_names" ]]; then
    ready_line+="; historical_failed_session_residue_names=${historical_failed_session_residue_names}"
    ready_line+="; historical_failed_session_residue_ownership_scope=${historical_failed_session_residue_ownership_scope:-none}"
    ready_line+="; historical_failed_session_residue_ownership_per_rail=${historical_failed_session_residue_ownership_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_playbook_scope=${historical_failed_session_residue_playbook_scope:-none}"
    ready_line+="; historical_failed_session_residue_playbook_per_rail=${historical_failed_session_residue_playbook_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_current_health_scope=${historical_failed_session_residue_current_health_scope:-none}"
    ready_line+="; historical_failed_session_residue_current_health_per_rail=${historical_failed_session_residue_current_health_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_decay_state=${historical_failed_session_residue_decay_state:-none}"
    ready_line+="; historical_failed_session_residue_decay_urgency=${historical_failed_session_residue_decay_urgency:-none}"
    ready_line+="; historical_failed_session_residue_staleness_per_rail=${historical_failed_session_residue_staleness_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_card_scope=${historical_failed_session_residue_card_scope:-none}"
    ready_line+="; historical_failed_session_residue_card_per_rail=${historical_failed_session_residue_card_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_card_status_scope=${historical_failed_session_residue_card_status_scope:-none}"
    ready_line+="; historical_failed_session_residue_card_status_per_rail=${historical_failed_session_residue_card_status_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_card_severity_scope=${historical_failed_session_residue_card_severity_scope:-none}"
    ready_line+="; historical_failed_session_residue_card_severity_per_rail=${historical_failed_session_residue_card_severity_per_rail:-none}"
    ready_line+="; historical_failed_session_residue_card_retirement_scope=${historical_failed_session_residue_card_retirement_scope:-none}"
    ready_line+="; historical_failed_session_residue_card_retirement_per_rail=${historical_failed_session_residue_card_retirement_per_rail:-none}"
    if [[ "${historical_failed_session_repeated_false_green_guard_state:-none}" == "active" ]]; then
      ready_line+="; historical_failed_session_repeated_false_green_guard=active"
      ready_line+="; historical_failed_session_repeated_false_green_guard_reason=${historical_failed_session_repeated_false_green_guard_reason:-repeated_historical_failed_session_cards_reconciled_now}"
      ready_line+="; historical_failed_session_repeated_false_green_guard_per_rail=${historical_failed_session_repeated_false_green_guard_per_rail:-none}"
    fi
  elif [[ "$historical_failed_session_residue_state" == "unknown_status_metadata_unavailable" ]]; then
    ready_line+="; historical_failed_session_residue_state=unknown_status_metadata_unavailable"
  elif [[ "$historical_failed_session_residue_state" == "none_observed" ]]; then
    ready_line+="; historical_failed_session_residue_state=none_observed"
    ready_line+="; historical_failed_session_residue_clear_state=reconciled_now"
  fi
  echo "$ready_line"
elif [[ "$classification" == "cron_runtime_failed" ]]; then
  echo "BLOCKER: no_llm_watchdog_cron_authority_runtime_failed; reason=${summary_error:-runtime_failures}; recovery_state=${runtime_recovery_state:-none}; retryable=${runtime_retryable}; escalation_policy=${runtime_escalation_policy:-none}; escalation_urgency=${runtime_escalation_urgency:-none}; escalation_source=${runtime_escalation_source:-none}; escalation_scope=${runtime_escalation_scope:-none}; escalation_per_rail=${runtime_escalation_per_rail:-none}; ownership_scope=${runtime_ownership_scope:-none}; ownership_per_rail=${runtime_ownership_per_rail:-none}; family_playbook_scope=${runtime_playbook_scope:-none}; family_playbook_per_rail=${runtime_playbook_per_rail:-none}; transition_signature=${runtime_transition_signature:-none}; ids=${viol_ids:-none}; names=${runtime_names:-none}; codes=${runtime_codes:-runtime:enabled_authority_runtime_unhealthy}; root_causes=${runtime_root_causes:-none}; remediation_hints=${runtime_remediation_hints:-runtime:remediation_collect_failure_evidence_and_route_operator_triage}; latest_failure_name=${latest_failure_name:-none}; latest_failure_at_ms=${latest_failure_at_ms:-unknown}; latest_failure_reason_source=${latest_failure_reason_source:-unavailable}; latest_failure_reason=${latest_failure_reason_excerpt:-none}"
elif [[ "$classification" == "gateway_connectivity_failure" ]]; then
  echo "BLOCKER: no_llm_watchdog_cron_authority_gateway_connectivity_failure; reason=${summary_error:-cron_list_unreachable}"
elif [[ "$classification" == "cron_contract_drift" ]]; then
  echo "BLOCKER: no_llm_watchdog_cron_authority_contract_drift; reason=${summary_error:-contract_drift}"
else
  echo "BLOCKER: no_llm_watchdog_cron_authority_policy_failed; violations=${viol_count}; ids=${viol_ids:-none}"
fi

if [[ "$JSON_OUT" -eq 1 ]]; then
  echo "$summary_json"
fi

if [[ "$STRICT_EXIT" -eq 1 && "$ok_flag" != "1" ]]; then
  exit 1
fi

exit 0
