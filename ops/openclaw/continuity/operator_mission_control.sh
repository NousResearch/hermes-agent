#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
REFRESH=0
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: operator_mission_control.sh [options]

Operator Mission Control single-pane summary (truth strip + actionable commands).

Options:
  --refresh     Refresh continuity/current + handover surfaces first
  --json        Print JSON payload
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1; shift ;;
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

python3 - "$ROOT" "$REFRESH" "$JSON_OUT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
refresh = bool(int(sys.argv[2]))
json_out = bool(int(sys.argv[3]))

current_script = root / "ops" / "openclaw" / "continuity" / "continuity_current.sh"
handover_script = root / "ops" / "openclaw" / "continuity" / "handover_latest.sh"
continuity_now_script = root / "ops" / "openclaw" / "continuity" / "continuity_now.sh"
gate_os_script = root / "ops" / "openclaw" / "continuity" / "gate_os_snapshot.sh"
queue_replay_script = root / "ops" / "openclaw" / "continuity" / "queue_replay_verify.sh"

current_path = root / "state" / "continuity" / "current.json"
blocker_registry_path = root / "state" / "continuity" / "latest" / "blocker_registry.json"
export_path = root / "state" / "continuity" / "latest" / "operator_mission_control.json"
execution_program_status_path = root / "state" / "continuity" / "latest" / "execution_program_status.json"
execution_frontier_ledger_path = root / "state" / "continuity" / "latest" / "execution_frontier_ledger.json"
execution_supervisor_dispatch_intent_path = (
    root / "state" / "continuity" / "latest" / "execution_supervisor_dispatch_intent_latest.json"
)
execution_supervisor_dispatch_qualification_path = (
    root / "state" / "continuity" / "latest" / "execution_supervisor_dispatch_qualification_latest.json"
)
execution_supervisor_probe_execution_plan_path = (
    root / "state" / "continuity" / "latest" / "execution_supervisor_probe_execution_plan_latest.json"
)
execution_meaningful_event_reporting_path = root / "state" / "continuity" / "latest" / "execution_meaningful_event_reporting_latest.json"
execution_meaningful_event_reporting_status_path = (
    root / "state" / "continuity" / "latest" / "execution_meaningful_event_reporting_status_latest.json"
)
model_rollout_dashboard_path = root / "state" / "continuity" / "model_rollout_dashboard" / "latest.json"
dependency_policy_pack_path = root / "state" / "continuity" / "latest" / "core_roadmap_dependency_unblock_policy_pack_v1.json"
efficiency_kpi_baseline_path = root / "state" / "continuity" / "latest" / "efficiency_kpi_baseline_latest.json"
efficiency_kpi_validation_path = root / "state" / "continuity" / "latest" / "efficiency_kpi_baseline_validation_latest.json"
mission_control_schema_path = root / "ops" / "openclaw" / "architecture" / "schemas" / "operator_mission_control.schema.json"
load_shedding_decision_path = root / "state" / "continuity" / "latest" / "load_shedding_decision.json"
load_shedding_signal_snapshot_path = root / "state" / "continuity" / "latest" / "load_shedding_signal_snapshot.json"
wave2_replay_evidence_index_path = root / "state" / "continuity" / "latest" / "wave2_replay_evidence_index.json"
failover_stress_soak_evidence_path = root / "state" / "continuity" / "latest" / "failover_stress_soak_evidence.json"
failover_stress_runtime_evidence_path = root / "state" / "continuity" / "latest" / "failover_stress_runtime_evidence.json"
core_roadmap_execution_queue_path = root / "state" / "continuity" / "latest" / "core_roadmap_execution_queue.json"
core_roadmap_slice_queue_path = root / "state" / "continuity" / "latest" / "core_roadmap_slice_queue_2026-03-28.json"
librarian_promotions_path = root / "state" / "continuity" / "librarian" / "promotions.jsonl"
evidence_trace_viewer_export_path = root / "state" / "continuity" / "latest" / "evidence_trace_viewer_latest.json"
current_publish_lock_owner_rel = "state/continuity/latest/current_publish.lock.owner.json"

continuity_dir = (root / "ops" / "openclaw" / "continuity").resolve()
schema_helper_path = continuity_dir / "schema_contract_validation.py"
sys.path.insert(0, str(continuity_dir))
try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from continuity_policy import (
        CURRENT_PUBLISH_LOCK_OWNER_REL as _CURRENT_PUBLISH_LOCK_OWNER_REL,
        DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC as _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
        DRIFT_REASON_SET as _DRIFT_REASON_SET,
        GENERATION_POINTER_READ_PHASE_PIN_SUPPRESSIBLE_REASON_SET as _GENERATION_POINTER_READ_PHASE_PIN_SUPPRESSIBLE_REASON_SET,
        PUBLISH_LOCK_ACTIONABLE_STATUS_SET as _PUBLISH_LOCK_ACTIONABLE_STATUS_SET,
        PUBLISH_LOCK_WARNING_REASON_SET as _PUBLISH_LOCK_WARNING_REASON_SET,
        continuity_now_contract_declared as _continuity_now_contract_declared,
        continuity_now_contract_expected_fields as _continuity_now_contract_expected_fields,
        continuity_now_contract_failclose_reasons as _continuity_now_contract_failclose_reasons,
        generation_pointer_core_failclose_reasons as _generation_pointer_core_failclose_reasons,
        project_blocker_registry_publish_lock_signal as _project_blocker_registry_publish_lock_signal,
        project_reset_ready_refresh_posture as _project_reset_ready_refresh_posture,
        read_nonnegative_int_env as _read_nonnegative_int_env,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _CURRENT_PUBLISH_LOCK_OWNER_REL = current_publish_lock_owner_rel
    _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC = 21600
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }
    _GENERATION_POINTER_READ_PHASE_PIN_SUPPRESSIBLE_REASON_SET = {
        "generation_pointer_current_sha_mismatch",
        "generation_pointer_current_generated_at_mismatch",
        "generation_pointer_generation_mismatch",
    }
    _PUBLISH_LOCK_WARNING_REASON_SET = {
        "continuity_current_publish_lock_wait_budget_exceeded",
        "continuity_current_publish_lock_hold_budget_exceeded",
    }
    _PUBLISH_LOCK_ACTIONABLE_STATUS_SET = {
        "invalid",
        "unreadable",
        "wait_budget_exceeded",
        "hold_budget_exceeded",
    }

    def _read_nonnegative_int_env(name: str, *, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(int(default)))))
        except Exception:
            return int(default)

    def _project_reset_ready_refresh_posture(
        *,
        surface: Any = None,
        latest_payload: Any = None,
        path: Any = None,
        sha256: Any = None,
        present: Any = None,
    ) -> Dict[str, Any]:
        surface_map = surface if isinstance(surface, dict) else {}
        latest_map = latest_payload if isinstance(latest_payload, dict) else {}

        path_text = str(path or surface_map.get("path") or "").strip()
        sha_text = str(sha256 or surface_map.get("sha256") or "").strip() or None

        if isinstance(present, bool):
            present_value = present
        else:
            present_value = bool(surface_map.get("present") is True or bool(latest_map))

        ok = surface_map.get("ok") if isinstance(surface_map.get("ok"), bool) else None
        if ok is None and isinstance(latest_map.get("ok"), bool):
            ok = latest_map.get("ok")

        phase = str(surface_map.get("phase") or latest_map.get("phase") or "").strip() or None
        if phase is None and ok is True:
            phase = "complete"

        partial_refresh = surface_map.get("partial_refresh") if isinstance(surface_map.get("partial_refresh"), dict) else {}
        if not partial_refresh and isinstance(latest_map.get("partial_refresh"), dict):
            partial_refresh = latest_map.get("partial_refresh")

        def _partial_flag(name: str) -> Optional[bool]:
            raw_value = partial_refresh.get(name)
            return raw_value if isinstance(raw_value, bool) else None

        partial_current = _partial_flag("current_refreshed")
        partial_proof = _partial_flag("proof_refreshed")
        partial_handover = _partial_flag("handover_refreshed")

        explicit_partial_failure = surface_map.get("partial_failure")
        if isinstance(explicit_partial_failure, bool):
            partial_failure = explicit_partial_failure
        else:
            partial_failure = bool(
                present_value
                and any(value is False for value in [partial_current, partial_proof, partial_handover])
            )

        error_code = str(
            surface_map.get("error_code")
            or (((latest_map.get("error") or {}).get("code")) if isinstance(latest_map.get("error"), dict) else "")
            or ""
        ).strip() or None

        explicit_degraded = surface_map.get("degraded")
        if isinstance(explicit_degraded, bool):
            degraded = explicit_degraded
        else:
            degraded = bool(present_value and (ok is False or partial_failure))

        generated_at = str(surface_map.get("generated_at") or latest_map.get("generated_at") or "").strip() or None

        status = "missing"
        if present_value:
            if degraded:
                status = "degraded"
            elif ok is True:
                status = "ok"
            else:
                status = "present"

        recommended_action = None
        if degraded:
            recommended_action = "rerun_reset_ready_refresh"
        elif present_value:
            recommended_action = "inspect_reset_ready_refresh_result"

        return {
            "path": path_text,
            "sha256": sha_text,
            "generated_at": generated_at,
            "present": present_value,
            "status": status,
            "ok": ok,
            "phase": phase,
            "error_code": error_code,
            "partial_refresh": {
                "current_refreshed": partial_current,
                "proof_refreshed": partial_proof,
                "handover_refreshed": partial_handover,
            },
            "degraded": degraded,
            "partial_failure": partial_failure,
            "action_required": degraded,
            "recommended_action": recommended_action,
        }

    def _continuity_now_contract_expected_fields(
        *,
        contract_obj: Any,
        source_refs: Any,
    ) -> tuple[str, str, str]:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}
        expected_sha = str(contract_map.get("sha256") or source_map.get("continuity_now_sha256") or "").strip()
        expected_generated_at = str(contract_map.get("generated_at") or "").strip()
        expected_generation = str(contract_map.get("coherence_build_generation_id") or "").strip()
        return expected_sha, expected_generated_at, expected_generation

    def _continuity_now_contract_declared(
        *,
        contract_obj: Any,
        source_refs: Any,
        require_sha_pin: bool,
    ) -> bool:
        if not require_sha_pin:
            return isinstance(contract_obj, dict)
        expected_sha, _, _ = _continuity_now_contract_expected_fields(
            contract_obj=contract_obj,
            source_refs=source_refs,
        )
        return bool(expected_sha)

    def _continuity_now_contract_failclose_reasons(
        *,
        contract_declared: Any,
        contract_path: pathlib.Path,
        expected_sha256: Any,
        expected_generated_at: Any,
        expected_coherence_build_generation_id: Any,
    ) -> tuple[List[str], Optional[str], Optional[Dict[str, Any]]]:
        declared = bool(contract_declared)
        if not declared:
            return [], None, None

        expected_sha = str(expected_sha256 or "").strip()
        expected_generated = str(expected_generated_at or "").strip()
        expected_generation = str(expected_coherence_build_generation_id or "").strip()

        if not contract_path.exists():
            return ["continuity_now_contract_missing"], None, None

        try:
            raw = contract_path.read_text(encoding="utf-8")
            actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("continuity_now_contract_not_object")

            reasons: List[str] = []
            if expected_sha and actual_sha != expected_sha:
                reasons.append("continuity_now_contract_sha_mismatch")

            actual_generated = str(payload.get("generated_at") or "").strip()
            if expected_generated and actual_generated != expected_generated:
                reasons.append("continuity_now_contract_generated_at_mismatch")

            actual_generation = str((((payload.get("coherence") or {}).get("build_generation_id") or "")).strip())
            if expected_generation and actual_generation != expected_generation:
                reasons.append("continuity_now_contract_generation_mismatch")

            return unique_preserve(reasons), actual_sha, payload
        except Exception:
            return ["continuity_now_contract_unreadable"], None, None

    def _generation_pointer_core_failclose_reasons(
        *,
        pointer_current_sha256: Any,
        current_sha256: Any,
        pointer_current_generated_at: Any,
        current_generated_at: Any,
        pointer_generation_id: Any,
        current_generation_id: Any,
    ) -> List[str]:
        reasons: List[str] = []

        pointer_current_sha = str(pointer_current_sha256 or "").strip()
        current_sha = str(current_sha256 or "").strip()
        pointer_current_ts = str(pointer_current_generated_at or "").strip()
        current_ts = str(current_generated_at or "").strip()
        pointer_generation = str(pointer_generation_id or "").strip()
        current_generation = str(current_generation_id or "").strip()

        if not pointer_current_sha:
            reasons.append("generation_pointer_missing_current_sha256")
        elif current_sha and pointer_current_sha != current_sha:
            reasons.append("generation_pointer_current_sha_mismatch")

        if not pointer_current_ts:
            reasons.append("generation_pointer_missing_current_generated_at")
        elif current_ts and pointer_current_ts != current_ts:
            reasons.append("generation_pointer_current_generated_at_mismatch")

        current_dt = parse_iso(current_ts)
        pointer_dt = parse_iso(pointer_current_ts)
        if current_dt is not None and pointer_dt is not None and pointer_dt < current_dt:
            reasons.append("generation_pointer_stale")

        if current_generation and not pointer_generation:
            reasons.append("generation_pointer_missing_generation_id")
        elif current_generation and pointer_generation and current_generation != pointer_generation:
            reasons.append("generation_pointer_generation_mismatch")

        return unique_preserve(reasons)

try:
    from continuity_now_paths import (
        CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON as _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON,
        DEFAULT_CONTINUITY_NOW_LATEST_REL as _CONTINUITY_NOW_LATEST_REL,
        continuity_now_contract_path_conflict_reason as _continuity_now_contract_path_conflict_reason,
        resolve_continuity_now_contract_path as _resolve_continuity_now_contract_path,
        resolve_continuity_now_evidence_path as _resolve_continuity_now_evidence_path,
        to_rel_or_abs as _to_rel_or_abs_path,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _CONTINUITY_NOW_LATEST_REL = "state/continuity/latest/continuity_now_latest.json"
    _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON = "continuity_now_contract_path_conflict"

    def _resolve_continuity_now_contract_path(root_path: pathlib.Path, *, contract_obj: Any = None, source_refs: Any = None) -> pathlib.Path:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}
        path_txt = str(
            contract_map.get("path")
            or source_map.get("continuity_now")
            or _CONTINUITY_NOW_LATEST_REL
        ).strip() or _CONTINUITY_NOW_LATEST_REL
        path = pathlib.Path(path_txt)
        if not path.is_absolute():
            path = (pathlib.Path(root_path).resolve() / path).resolve()
        else:
            path = path.resolve()
        return path

    def _continuity_now_contract_path_conflict_reason(root_path: pathlib.Path, *, contract_obj: Any = None, source_refs: Any = None) -> str | None:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}

        contract_raw = str(contract_map.get("path") or "").strip()
        source_raw = str(source_map.get("continuity_now") or "").strip()
        if not contract_raw or not source_raw:
            return None

        contract_path = pathlib.Path(contract_raw)
        if not contract_path.is_absolute():
            contract_path = (pathlib.Path(root_path).resolve() / contract_path).resolve()
        else:
            contract_path = contract_path.resolve()

        source_path = pathlib.Path(source_raw)
        if not source_path.is_absolute():
            source_path = (pathlib.Path(root_path).resolve() / source_path).resolve()
        else:
            source_path = source_path.resolve()

        if contract_path == source_path:
            return None
        return _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON

    def _to_rel_or_abs_path(root_path: pathlib.Path, path: pathlib.Path) -> str:
        path_obj = pathlib.Path(path).resolve()
        try:
            return str(path_obj.relative_to(pathlib.Path(root_path).resolve()))
        except Exception:
            return str(path_obj)

    def _resolve_continuity_now_evidence_path(
        root_path: pathlib.Path,
        *,
        contract_obj: Any = None,
        source_refs: Any = None,
        raw_path: Any = None,
        fallback_rel: str = _CONTINUITY_NOW_LATEST_REL,
    ) -> str:
        raw_txt = str(raw_path or "").strip()
        if raw_txt:
            raw_path_obj = pathlib.Path(raw_txt)
            if raw_path_obj.is_absolute():
                return _to_rel_or_abs_path(pathlib.Path(root_path), raw_path_obj)
            return str(raw_path_obj)
        contract_path = _resolve_continuity_now_contract_path(
            pathlib.Path(root_path),
            contract_obj=contract_obj,
            source_refs=source_refs,
        )
        return _to_rel_or_abs_path(pathlib.Path(root_path), contract_path)

try:
    from schema_helper_guard import (
        format_schema_helper_unavailable_error as _format_schema_helper_unavailable_error,
        load_contract_schema_validator as _load_contract_schema_validator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by sidecar-fixture runtime regressions
    if getattr(exc, "name", "") != "schema_helper_guard":
        raise
    import importlib.util

    def _load_contract_schema_validator(helper_path: pathlib.Path):
        if not helper_path.exists():
            return None, FileNotFoundError(str(helper_path))

        try:
            spec = importlib.util.spec_from_file_location(
                "openclaw_schema_contract_validation_sidecar",
                helper_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError("schema_helper_spec_unavailable")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validator = getattr(module, "validate_contract_payload_schema", None)
            if not callable(validator):
                raise AttributeError("validate_contract_payload_schema_missing")

            return validator, None
        except Exception as exc:
            return None, exc

    def _format_schema_helper_unavailable_error(
        *,
        contract_prefix: str,
        helper_path: pathlib.Path,
        import_error: Exception | None,
    ) -> str:
        import_error_name = type(import_error).__name__ if import_error is not None else "unknown"
        import_failure_reason = "missing_sidecar" if isinstance(import_error, FileNotFoundError) else "import_failed"
        return (
            f"{contract_prefix}_schema_helper_unavailable:"
            f"helper_path={helper_path}:"
            f"reason={import_failure_reason}:"
            f"error={import_error_name}"
        )

_validate_contract_payload_schema, _schema_helper_import_error = _load_contract_schema_validator(schema_helper_path)


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


def now_iso() -> str:
    return clock_now_iso()


# continuity_current --refresh and the downstream GateOS refresh path routinely
# traverse verify/current publish-lock windows. Keep fail-closed semantics, but
# align the default operator mission-control budgets with the safer handover
# current budget so refreshes do not degrade healthy-but-slower live surfaces.
refresh_current_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_REFRESH_CURRENT_TIMEOUT_SEC",
        default=120,
    ),
)
refresh_handover_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_REFRESH_HANDOVER_TIMEOUT_SEC",
        default=60,
    ),
)
refresh_gate_os_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_REFRESH_GATE_OS_TIMEOUT_SEC",
        default=180,
    ),
)
live_current_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_LIVE_CURRENT_TIMEOUT_SEC",
        default=120,
    ),
)
live_handover_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_LIVE_HANDOVER_TIMEOUT_SEC",
        default=60,
    ),
)
live_gate_os_timeout_sec = max(
    1,
    _read_nonnegative_int_env(
        "OPENCLAW_OPERATOR_MISSION_CONTROL_LIVE_GATE_OS_TIMEOUT_SEC",
        default=60,
    ),
)


if "_project_blocker_registry_publish_lock_signal" not in globals():
    def _project_blocker_registry_publish_lock_signal(
        *,
        blocker_registry: Any = None,
        publish_lock: Any = None,
        blockers: Any = None,
        current_generated_at: Any = None,
    ) -> Dict[str, Any]:
        registry_map = blocker_registry if isinstance(blocker_registry, dict) else {}
        publish_lock_map = publish_lock if isinstance(publish_lock, dict) else {}
        if not publish_lock_map and isinstance(registry_map.get("publish_lock"), dict):
            publish_lock_map = registry_map.get("publish_lock")

        blocker_rows = blockers if isinstance(blockers, list) else registry_map.get("blockers")
        warning_reasons: List[str] = []
        if isinstance(blocker_rows, list):
            for row in blocker_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("severity") or "").strip() != "warn":
                    continue
                reason = str(row.get("reason") or "").strip()
                if reason in _PUBLISH_LOCK_WARNING_REASON_SET:
                    warning_reasons.append(reason)

        registry_freshness = registry_map.get("freshness") if isinstance(registry_map.get("freshness"), dict) else {}
        source_current = registry_map.get("source_current") if isinstance(registry_map.get("source_current"), dict) else {}
        source_current_path = str(source_current.get("path") or registry_freshness.get("source") or "").strip() or None
        source_current_generated_at = str(source_current.get("generated_at") or registry_freshness.get("source_generated_at") or "").strip() or None
        expected_current_generated = str(current_generated_at or "").strip() or None
        source_current_matches_current_generated_at = None
        if source_current_generated_at and expected_current_generated:
            source_current_matches_current_generated_at = source_current_generated_at == expected_current_generated

        source_current_fresh = registry_freshness.get("fresh")
        if not isinstance(source_current_fresh, bool):
            source_current_fresh = None

        source_degraded_reasons: List[str] = []
        if source_current_fresh is False:
            source_degraded_reasons.append("blocker_registry_source_current_stale")
        if source_current_matches_current_generated_at is False:
            source_degraded_reasons.append("blocker_registry_source_current_generated_at_mismatch")
        source_degraded_reasons = unique_preserve(source_degraded_reasons)

        warning_reasons = unique_preserve(warning_reasons)
        status = str(publish_lock_map.get("status") or "").strip() or None
        action_required = bool(publish_lock_map.get("action_required") is True or warning_reasons)
        surface_active = bool(
            action_required
            or (status in _PUBLISH_LOCK_ACTIONABLE_STATUS_SET if status else False)
            or warning_reasons
        )

        return {
            "generated_at": str(registry_map.get("generated_at") or "").strip() or None,
            "present": publish_lock_map.get("present") is True,
            "path": str(publish_lock_map.get("path") or "").strip() or None,
            "status": status,
            "owner_pid": publish_lock_map.get("owner_pid"),
            "owner_alive": publish_lock_map.get("owner_alive"),
            "owner_age_sec": publish_lock_map.get("owner_age_sec"),
            "lock_wait_sec": publish_lock_map.get("lock_wait_sec"),
            "lock_hold_warn_sec": publish_lock_map.get("lock_hold_warn_sec"),
            "owner_exceeds_wait_budget": publish_lock_map.get("owner_exceeds_wait_budget"),
            "owner_exceeds_lock_hold_warn": publish_lock_map.get("owner_exceeds_lock_hold_warn"),
            "owner_host": publish_lock_map.get("owner_host"),
            "owner_command": publish_lock_map.get("owner_command"),
            "recommended_action": str(publish_lock_map.get("recommended_action") or "").strip() or None,
            "inspect_command": str(publish_lock_map.get("inspect_command") or "").strip() or None,
            "warning_reasons": warning_reasons,
            "action_required": action_required,
            "surface_active": surface_active,
            "source_current_path": source_current_path,
            "source_current_generated_at": source_current_generated_at,
            "source_current_age_sec": registry_freshness.get("source_age_sec") if isinstance(registry_freshness.get("source_age_sec"), (int, float)) else None,
            "source_current_max_age_sec": registry_freshness.get("max_age_sec") if isinstance(registry_freshness.get("max_age_sec"), (int, float)) else None,
            "source_current_fresh": source_current_fresh,
            "source_current_matches_current_generated_at": source_current_matches_current_generated_at,
            "source_degraded": bool(source_degraded_reasons),
            "source_degraded_reasons": source_degraded_reasons,
        }


current_publish_lock_owner_rel = str(_CURRENT_PUBLISH_LOCK_OWNER_REL or current_publish_lock_owner_rel).strip() or current_publish_lock_owner_rel


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


def age_sec(raw: Any):
    ts = parse_iso(raw)
    if ts is None:
        return None
    return max(0, int((clock_now_dt() - ts).total_seconds()))


def _nonnegative_int(raw: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(raw or 0))
    except Exception:
        return default


def _optional_nonnegative_int(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return max(0, int(raw))
    except Exception:
        return None


def normalize_dispatch_context(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "status": str(raw.get("status") or "missing"),
        "source": str(raw.get("source") or "continuity_current"),
        "autopilot_status": str(raw.get("autopilot_status") or "missing"),
        "ready_work_exists": bool(raw.get("ready_work_exists") is True),
        "idle_threshold_exceeded": bool(raw.get("idle_threshold_exceeded") is True),
        "idle_sec": _nonnegative_int(raw.get("idle_sec")),
        "target_step_id": str(raw.get("target_step_id") or "").strip() or None,
        "launched_step_id": str(raw.get("launched_step_id") or "").strip() or None,
        "skip_reason": str(raw.get("skip_reason") or "").strip() or None,
        "trace_path": str(raw.get("trace_path") or "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json"),
        "updated_at": str(raw.get("updated_at") or "").strip() or None,
        "autonomous_dispatch_status": str(raw.get("autonomous_dispatch_status") or "missing"),
        "autonomous_dispatch_decision": str(raw.get("autonomous_dispatch_decision") or "").strip() or None,
        "autonomous_dispatch_skip_reason": str(raw.get("autonomous_dispatch_skip_reason") or "").strip() or None,
        "autonomous_dispatch_block_reason": str(raw.get("autonomous_dispatch_block_reason") or "").strip() or None,
        "autonomous_dispatch_block_reasons": _reason_list(raw.get("autonomous_dispatch_block_reasons")),
        "autonomous_dispatch_error": str(raw.get("autonomous_dispatch_error") or "").strip() or None,
        "autonomous_dispatch_updated_at": str(raw.get("autonomous_dispatch_updated_at") or "").strip() or None,
        "autonomous_dispatch_selector_state": str(raw.get("autonomous_dispatch_selector_state") or "").strip() or None,
        "autonomous_dispatch_close_condition_met": (
            raw.get("autonomous_dispatch_close_condition_met")
            if isinstance(raw.get("autonomous_dispatch_close_condition_met"), bool)
            else None
        ),
        "autonomous_dispatch_post_completion_enforcement_required": bool(
            raw.get("autonomous_dispatch_post_completion_enforcement_required") is True
        ),
        "autonomous_dispatch_post_completion_enforcement_latched": bool(
            raw.get("autonomous_dispatch_post_completion_enforcement_latched") is True
        ),
        "autonomous_dispatch_post_completion_loop_state": str(
            raw.get("autonomous_dispatch_post_completion_loop_state") or ""
        ).strip()
        or None,
        "autonomous_dispatch_retry_contract": (
            raw.get("autonomous_dispatch_retry_contract")
            if isinstance(raw.get("autonomous_dispatch_retry_contract"), dict)
            else {}
        ),
        "autonomous_dispatch_cooldown_policy": (
            raw.get("autonomous_dispatch_cooldown_policy")
            if isinstance(raw.get("autonomous_dispatch_cooldown_policy"), dict)
            else {}
        ),
        "autonomous_dispatch_queue_truth_vs_narrative_parity": (
            raw.get("autonomous_dispatch_queue_truth_vs_narrative_parity")
            if isinstance(raw.get("autonomous_dispatch_queue_truth_vs_narrative_parity"), dict)
            else {}
        ),
        "autonomous_dispatch_intent_active": bool(raw.get("autonomous_dispatch_intent_active") is True),
        "autonomous_dispatch_latch_path": str(
            raw.get("autonomous_dispatch_latch_path")
            or "state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json"
        ),
        "autonomous_dispatch_latch_history_path": str(
            raw.get("autonomous_dispatch_latch_history_path")
            or "state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl"
        ),
        "autonomous_execution_intent_path": str(
            raw.get("autonomous_execution_intent_path")
            or "state/continuity/latest/autonomous_execution_intent_latest.json"
        ),
        "autonomous_execution_intent_history_path": str(
            raw.get("autonomous_execution_intent_history_path")
            or "state/continuity/history/autonomous_execution_intent_history.jsonl"
        ),
        "autonomous_dispatch_trace_path": str(
            raw.get("autonomous_dispatch_trace_path")
            or "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"
        ),
        "autonomous_dispatch_history_path": str(
            raw.get("autonomous_dispatch_history_path")
            or "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl"
        ),
        "autonomous_dispatch_source_degraded": bool(raw.get("autonomous_dispatch_source_degraded") is True),
    }


def build_dispatch_context_from_idle_lane(idle_lane: Dict[str, Any]) -> Dict[str, Any]:
    raw_status = str(idle_lane.get("status") or "missing")
    ready_work_exists = bool(idle_lane.get("ready_work_exists") is True)
    idle_threshold_exceeded = bool(idle_lane.get("idle_threshold_exceeded") is True)
    target_step_id = str(idle_lane.get("target_step_id") or "").strip() or None
    launched_step_id = str(idle_lane.get("launched_step_id") or "").strip() or None
    skip_reason = str(idle_lane.get("skip_reason") or "").strip() or None
    trace_path = str(idle_lane.get("trace_path") or "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json")
    idle_sec = _nonnegative_int(idle_lane.get("idle_sec"))
    failure_like = raw_status in {"tick_failed", "attempted_no_launch", "error"}

    if launched_step_id or raw_status == "launched":
        dispatch_status = "launched"
    elif failure_like and ready_work_exists and idle_threshold_exceeded:
        dispatch_status = "stalled"
    elif raw_status == "skipped" and ready_work_exists and idle_threshold_exceeded:
        dispatch_status = "blocked"
    elif target_step_id or ready_work_exists:
        dispatch_status = "pending"
    elif raw_status == "missing":
        dispatch_status = "missing"
    else:
        dispatch_status = "idle"

    return {
        "status": dispatch_status,
        "source": "continuity_now.autopilot.idle_lane_autospawn",
        "autopilot_status": raw_status,
        "ready_work_exists": ready_work_exists,
        "idle_threshold_exceeded": idle_threshold_exceeded,
        "idle_sec": idle_sec,
        "target_step_id": target_step_id,
        "launched_step_id": launched_step_id,
        "skip_reason": skip_reason,
        "trace_path": trace_path,
        "updated_at": str(idle_lane.get("updated_at") or "").strip() or None,
        "autonomous_dispatch_status": "missing",
        "autonomous_dispatch_decision": None,
        "autonomous_dispatch_skip_reason": None,
        "autonomous_dispatch_block_reason": None,
        "autonomous_dispatch_block_reasons": [],
        "autonomous_dispatch_error": None,
        "autonomous_dispatch_updated_at": None,
        "autonomous_dispatch_selector_state": None,
        "autonomous_dispatch_close_condition_met": None,
        "autonomous_dispatch_post_completion_enforcement_required": False,
        "autonomous_dispatch_post_completion_enforcement_latched": False,
        "autonomous_dispatch_post_completion_loop_state": None,
        "autonomous_dispatch_retry_contract": {},
        "autonomous_dispatch_cooldown_policy": {},
        "autonomous_dispatch_queue_truth_vs_narrative_parity": {},
        "autonomous_dispatch_intent_active": False,
        "autonomous_dispatch_latch_path": "state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json",
        "autonomous_dispatch_latch_history_path": "state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl",
        "autonomous_execution_intent_path": "state/continuity/latest/autonomous_execution_intent_latest.json",
        "autonomous_execution_intent_history_path": "state/continuity/history/autonomous_execution_intent_history.jsonl",
        "autonomous_dispatch_trace_path": "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json",
        "autonomous_dispatch_history_path": "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl",
        "autonomous_dispatch_source_degraded": False,
    }


def normalize_execution_context(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    expected_in_flight_guard = raw.get("expected_in_flight_guard")
    if not isinstance(expected_in_flight_guard, bool):
        expected_in_flight_guard = None
    return {
        "posture": str(raw.get("posture") or "idle"),
        "source": str(raw.get("source") or "continuity_current"),
        "readiness": str(raw.get("readiness") or "UNKNOWN"),
        "in_flight": bool(raw.get("in_flight") is True),
        "running_tasks": _nonnegative_int(raw.get("running_tasks")),
        "active_locks": _nonnegative_int(raw.get("active_locks")),
        "mutation_gate_status": str(raw.get("mutation_gate_status") or "unknown"),
        "mutation_gate_posture": str(raw.get("mutation_gate_posture") or "unknown"),
        "expected_in_flight_guard": expected_in_flight_guard,
        "dispatch_status": str(raw.get("dispatch_status") or "missing"),
    }


def build_execution_context(
    *,
    readiness: str,
    in_flight: Dict[str, Any],
    mutation_gate: Dict[str, Any],
    dispatch_context: Dict[str, Any],
) -> Dict[str, Any]:
    dispatch_status = str(dispatch_context.get("status") or "missing")
    if bool(in_flight.get("value") is True):
        posture = "in_flight"
    elif dispatch_status == "launched":
        posture = "dispatch_launched"
    elif dispatch_status == "stalled":
        posture = "dispatch_stalled"
    elif dispatch_status == "blocked":
        posture = "dispatch_blocked"
    elif dispatch_status == "pending":
        posture = "dispatch_pending"
    else:
        posture = "idle"

    expected_in_flight_guard = mutation_gate.get("expected_in_flight_guard")
    if not isinstance(expected_in_flight_guard, bool):
        expected_in_flight_guard = None

    return {
        "posture": posture,
        "source": "derived_from_current_queue_mutation_gate_and_autopilot",
        "readiness": readiness,
        "in_flight": bool(in_flight.get("value") is True),
        "running_tasks": _nonnegative_int(in_flight.get("running_tasks")),
        "active_locks": _nonnegative_int(in_flight.get("active_locks")),
        "mutation_gate_status": str(mutation_gate.get("status") or "unknown"),
        "mutation_gate_posture": str(mutation_gate.get("posture") or "unknown"),
        "expected_in_flight_guard": expected_in_flight_guard,
        "dispatch_status": dispatch_status,
    }


def load_json_if_exists(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def summarize_live_routing_preflight(
    *,
    decisions_path: pathlib.Path,
    max_age_sec: int,
    inspect_decisions_command: str,
    recheck_policy_command: str,
    reference_now: Any = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "decision_log_path": str(decisions_path),
        "decision_log_present": decisions_path.exists(),
        "max_age_sec": max_age_sec,
        "rows_scanned": 0,
        "decision_rows_seen": 0,
        "parse_error_count": 0,
        "latest": {
            "decision": None,
            "evaluated_at": None,
            "age_sec": None,
            "fresh": None,
            "route_class": None,
            "selected_model": None,
            "required_rollout_stage": None,
            "selected_rule_id": None,
            "block_gate": None,
            "block_reason": None,
            "actionable_failure": {
                "gate": None,
                "reason": None,
                "hint": None,
                "commands": [],
            },
        },
        "effective": {
            "blocked": None,
            "blocked_fresh": None,
            "route_class": None,
            "selected_model": None,
            "required_rollout_stage": None,
            "block_gate": None,
            "block_reason": None,
            "actionable_hint": None,
            "first_actionable_command": None,
            "inspect_decisions_command": inspect_decisions_command,
            "recheck_policy_command": recheck_policy_command,
        },
        "failure_reason": None,
    }

    if not decisions_path.exists():
        summary["failure_reason"] = "routing_decisions_missing"
        return summary
    if not decisions_path.is_file():
        summary["failure_reason"] = "routing_decisions_not_regular_file"
        return summary

    latest_row: Optional[Dict[str, Any]] = None
    latest_ts: Optional[dt.datetime] = None
    reference_now_dt = parse_iso(reference_now) if reference_now is not None else None
    if reference_now_dt is None:
        reference_now_dt = clock_now_dt()

    try:
        with decisions_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                summary["rows_scanned"] = int(summary.get("rows_scanned") or 0) + 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    summary["parse_error_count"] = int(summary.get("parse_error_count") or 0) + 1
                    continue
                if not isinstance(row, dict):
                    summary["parse_error_count"] = int(summary.get("parse_error_count") or 0) + 1
                    continue
                if str(row.get("schema") or "").strip() != "clawd.session_topology_routing.decision.v1":
                    continue

                summary["decision_rows_seen"] = int(summary.get("decision_rows_seen") or 0) + 1
                row_ts = parse_iso(row.get("evaluated_at"))
                if latest_row is None:
                    latest_row = row
                    latest_ts = row_ts
                    continue

                if row_ts is not None:
                    if latest_ts is None or row_ts >= latest_ts:
                        latest_row = row
                        latest_ts = row_ts
                elif latest_ts is None:
                    latest_row = row
    except Exception:
        summary["failure_reason"] = "routing_decisions_unreadable"
        return summary

    if latest_row is None:
        summary["failure_reason"] = "routing_decisions_no_valid_rows"
        return summary

    route = latest_row.get("route") if isinstance(latest_row.get("route"), dict) else {}
    actionable = latest_row.get("actionable_failure") if isinstance(latest_row.get("actionable_failure"), dict) else {}
    actionable_commands = [
        str(cmd).strip()
        for cmd in (actionable.get("commands") if isinstance(actionable.get("commands"), list) else [])
        if str(cmd).strip()
    ]

    age = None
    fresh = None
    if latest_ts is not None:
        age = max(0, int((reference_now_dt - latest_ts).total_seconds()))
        fresh = True if max_age_sec <= 0 else age <= max_age_sec

    decision = str(latest_row.get("decision") or "").strip().upper() or None
    route_class = str(route.get("route_class") or "").strip() or None
    selected_model = str(route.get("selected_model") or "").strip() or None
    required_stage = str(route.get("required_rollout_stage") or "").strip() or None
    selected_rule_id = str(route.get("selected_rule_id") or "").strip() or None
    block_gate = str(latest_row.get("block_gate") or "").strip() or None
    block_reason = str(latest_row.get("block_reason") or "").strip() or None

    blocked = decision == "BLOCK"
    blocked_fresh = bool(blocked and (fresh is not False))
    effective_route_class = route_class if fresh is not False else None
    effective_selected_model = selected_model if fresh is not False else None
    effective_required_stage = required_stage if fresh is not False else None
    effective_block_gate = block_gate if blocked_fresh else None
    effective_block_reason = block_reason if blocked_fresh else None
    effective_actionable_hint = actionable.get("hint") if blocked_fresh else None
    effective_first_actionable_command = actionable_commands[0] if (blocked_fresh and actionable_commands) else None

    summary["latest"] = {
        "decision": decision,
        "evaluated_at": latest_row.get("evaluated_at"),
        "age_sec": age,
        "fresh": fresh,
        "route_class": route_class,
        "selected_model": selected_model,
        "required_rollout_stage": required_stage,
        "selected_rule_id": selected_rule_id,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "actionable_failure": {
            "gate": actionable.get("gate"),
            "reason": actionable.get("reason"),
            "hint": actionable.get("hint"),
            "commands": actionable_commands,
        },
    }
    summary["effective"] = {
        "blocked": blocked_fresh,
        "blocked_fresh": blocked_fresh,
        "route_class": effective_route_class,
        "selected_model": effective_selected_model,
        "required_rollout_stage": effective_required_stage,
        "block_gate": effective_block_gate,
        "block_reason": effective_block_reason,
        "actionable_hint": effective_actionable_hint,
        "first_actionable_command": effective_first_actionable_command,
        "inspect_decisions_command": inspect_decisions_command,
        "recheck_policy_command": recheck_policy_command,
    }

    if blocked_fresh:
        summary["failure_reason"] = "routing_blocked"
    elif fresh is False:
        summary["failure_reason"] = "routing_decision_stale"
    else:
        summary["failure_reason"] = None

    return summary


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_preserve(rows: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        txt = str(row or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


def parse_core_roadmap_slice_id(raw_focus: Any) -> Optional[int]:
    text = str(raw_focus or "").strip()
    if not text:
        return None
    match = re.search(r"(?:core_roadmap:wave)?(\d+)", text)
    if match:
        try:
            value = int(match.group(1))
        except Exception:
            value = None
        if value and value > 0:
            return value
    return None


def build_evidence_trace_viewer_projection(
    *,
    queue_payload: Dict[str, Any],
    execution_status_payload: Dict[str, Any],
    current_payload: Dict[str, Any],
    promotions_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    projection: Dict[str, Any] = {
        "schema": "clawd.evidence_trace_viewer.v1",
        "status": "degraded",
        "focus_task_id": None,
        "focus_slice_id": None,
        "focus_slice_title": None,
        "focus_slice_state": None,
        "trace": [],
        "blocked_reasons": [],
        "artifact_provenance": [],
        "source_degraded_reasons": [],
        "generated_at": now_iso(),
    }

    if not isinstance(queue_payload, dict) or not isinstance(queue_payload.get("slices"), list):
        projection["source_degraded_reasons"] = ["core_roadmap_execution_queue_missing_or_invalid"]
        return projection

    slices = [row for row in queue_payload.get("slices") if isinstance(row, dict)]
    if not slices:
        projection["source_degraded_reasons"] = ["core_roadmap_execution_queue_slices_empty"]
        return projection

    slice_index: Dict[int, Dict[str, Any]] = {}
    for row in slices:
        sid = row.get("id")
        if isinstance(sid, int) and sid > 0:
            slice_index[sid] = row

    focus_task_id = str(execution_status_payload.get("current_focus") or "").strip()
    focus_slice_id = parse_core_roadmap_slice_id(focus_task_id)
    if not focus_slice_id:
        next_candidate = str((current_payload.get("queue") or {}).get("next_candidate") or "").strip()
        focus_slice_id = parse_core_roadmap_slice_id(next_candidate)
        if next_candidate:
            focus_task_id = next_candidate
    if not focus_slice_id:
        for row in slices:
            if str(row.get("state") or "").strip() == "READY_NOW" and isinstance(row.get("id"), int):
                focus_slice_id = int(row.get("id"))
                if not focus_task_id:
                    focus_task_id = f"core_roadmap:wave{focus_slice_id}:{str(row.get('title') or '').strip()}"
                break

    if not focus_slice_id:
        for row in slices:
            title = str(row.get("title") or "").strip()
            if title == "b5_operator_explainability_ux" and isinstance(row.get("id"), int):
                focus_slice_id = int(row.get("id"))
                if not focus_task_id:
                    focus_task_id = f"core_roadmap:wave{focus_slice_id}:{title}"
                break

    if not focus_slice_id:
        b5_done = [
            row for row in slices
            if str(row.get("state") or "").strip() == "DONE"
            and isinstance(row.get("id"), int)
            and "B5" in [str(x).strip().upper() for x in (row.get("lane") or [])]
        ]
        if b5_done:
            row = sorted(b5_done, key=lambda x: int(x.get("id") or 0), reverse=True)[0]
            focus_slice_id = int(row.get("id"))
            if not focus_task_id:
                focus_task_id = f"core_roadmap:wave{focus_slice_id}:{str(row.get('title') or '').strip()}"

    focus_slice = slice_index.get(focus_slice_id or -1) if focus_slice_id else None
    if not isinstance(focus_slice, dict):
        projection["status"] = "degraded"
        projection["focus_task_id"] = focus_task_id or None
        projection["focus_slice_id"] = focus_slice_id
        projection["source_degraded_reasons"] = ["focus_slice_unresolved"]
        return projection

    visited: set[int] = set()

    def _walk(slice_obj: Dict[str, Any], depth: int = 0) -> None:
        sid = slice_obj.get("id")
        if not isinstance(sid, int) or sid in visited:
            return
        visited.add(sid)
        trace_row = {
            "slice_id": sid,
            "title": str(slice_obj.get("title") or "").strip() or f"slice_{sid}",
            "state": str(slice_obj.get("state") or "unknown").strip() or "unknown",
            "lane": [str(x).strip() for x in (slice_obj.get("lane") or []) if str(x).strip()],
            "depth": depth,
            "completed_at": str(slice_obj.get("completed_at") or "").strip() or None,
            "completed_via": str(slice_obj.get("completed_via") or "").strip() or None,
            "status_reason": str(slice_obj.get("status_reason") or "").strip() or None,
            "unblock_condition": str(slice_obj.get("unblock_condition") or "").strip() or None,
            "evidence_ref": str(slice_obj.get("evidence_ref") or "").strip() or None,
            "report_ref": str(slice_obj.get("report_ref") or "").strip() or None,
            "dependencies": [dep for dep in (slice_obj.get("dependencies") or []) if isinstance(dep, int)],
        }
        projection["trace"].append(trace_row)
        for dep_id in trace_row["dependencies"]:
            dep_obj = slice_index.get(dep_id)
            if isinstance(dep_obj, dict):
                _walk(dep_obj, depth + 1)

    _walk(focus_slice, 0)

    focus_status_reason = str(focus_slice.get("status_reason") or "").strip()
    blocked_reasons: List[str] = []
    if str(focus_slice.get("state") or "").strip() in {"DEPENDENCY_BLOCKED", "BLOCKED"} and focus_status_reason:
        blocked_reasons.append(focus_status_reason)

    artifact_provenance: List[Dict[str, Any]] = []
    for row in projection["trace"]:
        if not isinstance(row, dict):
            continue
        for key in ("evidence_ref", "report_ref"):
            ref = str(row.get(key) or "").strip()
            if not ref:
                continue
            provenance_row = {
                "slice_id": row.get("slice_id"),
                "source": key,
                "ref": ref,
            }
            if ref.startswith("state/"):
                abs_ref = root / ref
                provenance_row["exists"] = abs_ref.exists()
            artifact_provenance.append(provenance_row)

    promotion_refs = set(
        str(row.get("ref") or "").strip()
        for row in artifact_provenance
        if isinstance(row, dict) and str(row.get("ref") or "").strip()
    )
    for promo in promotions_rows[:200]:
        source_path = str(promo.get("source_path") or "").strip()
        if not source_path or source_path not in promotion_refs:
            continue
        artifact_provenance.append(
            {
                "slice_id": None,
                "source": "librarian_promotion",
                "ref": source_path,
                "promotion_id": str(promo.get("promotion_id") or "").strip() or None,
                "promoted_at": str(promo.get("promoted_at") or "").strip() or None,
                "operator": str(promo.get("operator") or "").strip() or None,
                "reason": str(promo.get("reason") or "").strip() or None,
            }
        )

    projection["status"] = "ready"
    projection["focus_task_id"] = focus_task_id or None
    projection["focus_slice_id"] = focus_slice.get("id")
    projection["focus_slice_title"] = str(focus_slice.get("title") or "").strip() or None
    projection["focus_slice_state"] = str(focus_slice.get("state") or "unknown").strip() or "unknown"
    projection["trace_count"] = len(projection["trace"])
    projection["blocked_reason_count"] = len(blocked_reasons)
    projection["blocked_reasons"] = blocked_reasons
    projection["artifact_provenance"] = artifact_provenance
    projection["artifact_ref_count"] = len(
        [row for row in artifact_provenance if isinstance(row, dict) and str(row.get("source") or "") != "librarian_promotion"]
    )
    projection["source_degraded_reasons"] = []
    return projection


def summarize_codes_for_state(codes: List[str], *, max_items: int = 3) -> str:
    values = unique_preserve(codes)
    if not values:
        return "none"
    try:
        limit = max(1, int(max_items))
    except Exception:
        limit = 3
    shown = values[:limit]
    remainder = max(0, len(values) - len(shown))
    summary = ",".join(shown)
    if remainder > 0:
        summary = f"{summary},+{remainder}"
    return summary


def _coerce_int(raw: Any) -> Optional[int]:
    try:
        return int(raw)
    except Exception:
        return None


def _coerce_float(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    if not (value == value and abs(value) != float("inf")):
        return None
    return value


def parse_continuity_current_publish_lock_timeout(raw: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "detected": False,
        "path": None,
        "wait_sec": None,
        "owner_hint_present": False,
        "owner_hint": None,
        "owner_pid": None,
        "owner_alive": None,
        "owner_age_sec": None,
        "lock_hold_warn_sec": None,
        "owner_exceeds_lock_hold_warn": None,
        "owner_host": None,
        "owner_command": None,
        "line": None,
    }

    text = str(raw or "").strip()
    if not text:
        return out

    prefix = "continuity_current publish lock timeout:"
    timeout_line = None
    for line in text.splitlines():
        candidate = str(line or "").strip()
        if candidate.startswith(prefix):
            timeout_line = candidate
            break
    if timeout_line is None and text.startswith(prefix):
        timeout_line = text
    if not timeout_line:
        return out

    out["detected"] = True
    out["line"] = timeout_line

    path_match = re.search(r"\bpath=([^\s]+)", timeout_line)
    if path_match:
        out["path"] = path_match.group(1).strip() or None

    wait_match = re.search(r"\bwait_sec=([0-9]+(?:\.[0-9]+)?)", timeout_line)
    if wait_match:
        out["wait_sec"] = _coerce_float(wait_match.group(1))

    owner_hint_raw = None
    owner_hint_idx = timeout_line.find("owner_hint=")
    if owner_hint_idx >= 0:
        owner_hint_raw = timeout_line[owner_hint_idx + len("owner_hint="):].strip()

    owner_hint: Optional[Dict[str, Any]] = None
    if owner_hint_raw:
        try:
            owner_hint_obj = json.loads(owner_hint_raw)
            if isinstance(owner_hint_obj, dict):
                owner_hint = owner_hint_obj
        except Exception:
            owner_hint = None

    if isinstance(owner_hint, dict):
        out["owner_hint_present"] = True
        out["owner_hint"] = owner_hint

        owner_pid = _coerce_int(owner_hint.get("owner_pid"))
        if isinstance(owner_pid, int):
            out["owner_pid"] = owner_pid

        owner_alive = owner_hint.get("owner_alive")
        if isinstance(owner_alive, bool):
            out["owner_alive"] = owner_alive

        owner_age_sec = _coerce_float(owner_hint.get("owner_age_sec"))
        if owner_age_sec is not None:
            out["owner_age_sec"] = max(0, int(owner_age_sec))

        hold_warn_sec = _coerce_float(owner_hint.get("lock_hold_warn_sec"))
        if hold_warn_sec is not None:
            out["lock_hold_warn_sec"] = hold_warn_sec

        exceeds_hold_warn = owner_hint.get("owner_exceeds_lock_hold_warn")
        if isinstance(exceeds_hold_warn, bool):
            out["owner_exceeds_lock_hold_warn"] = exceeds_hold_warn
        elif out.get("owner_age_sec") is not None and hold_warn_sec is not None:
            out["owner_exceeds_lock_hold_warn"] = bool(float(out["owner_age_sec"]) >= float(hold_warn_sec))

        out["owner_host"] = str(owner_hint.get("owner_host") or "").strip() or None
        out["owner_command"] = str(owner_hint.get("owner_command") or "").strip() or None

        if out.get("wait_sec") is None:
            out["wait_sec"] = _coerce_float(owner_hint.get("lock_wait_sec"))

    return out


def sha256_json_canonical(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_live_current_artifact() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(current_path.relative_to(root)) if current_path.is_absolute() else str(current_path),
        "present": current_path.exists(),
        "sha256": None,
        "generated_at": None,
        "generation_id": None,
        "payload": None,
        "read_error": None,
    }

    if not current_path.exists():
        return out

    try:
        raw = current_path.read_text(encoding="utf-8")
        out["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        payload = json.loads(raw)
        if isinstance(payload, dict):
            out["payload"] = payload
            out["generated_at"] = str(payload.get("generated_at") or "").strip() or None
            out["generation_id"] = str((((payload.get("coherence") or {}).get("build_generation_id") or "")).strip()) or None
        else:
            out["read_error"] = "current_not_object"
    except Exception as exc:
        out["read_error"] = str(exc)

    return out


def resolve_current_snapshot_sha(current_obj: Dict[str, Any], live_current_artifact: Dict[str, Any]) -> tuple[str, str]:
    live_payload = live_current_artifact.get("payload") if isinstance(live_current_artifact, dict) else None
    live_sha = str((live_current_artifact or {}).get("sha256") or "").strip()

    transient_contract_keys = {
        "contract_source",
        "contract_source_canonical",
        "contract_source_degraded",
        "contract_source_degraded_reason",
        "contract_source_degraded_path",
    }

    def _strip_transient_contract_keys(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        cleaned = dict(payload)
        for key in transient_contract_keys:
            cleaned.pop(key, None)
        return cleaned

    normalized_live_payload = _strip_transient_contract_keys(live_payload)
    normalized_current_payload = _strip_transient_contract_keys(current_obj)

    if isinstance(normalized_live_payload, dict) and normalized_live_payload == normalized_current_payload and live_sha:
        return live_sha, "current_file_match"

    return sha256_json_canonical(current_obj), "current_payload_pinned"


def maybe_apply_generation_pointer_read_phase_pin(
    *,
    generation_pointer: Dict[str, Any],
    current_obj: Dict[str, Any],
    now_obj: Dict[str, Any],
    live_current_artifact: Dict[str, Any],
    continuity_now_contract_failclose_reasons: List[str],
) -> Dict[str, Any]:
    reasons = unique_preserve([str(row or "").strip() for row in (generation_pointer.get("failclose_reasons") or []) if str(row or "").strip()])
    mismatch_reasons = set(_GENERATION_POINTER_READ_PHASE_PIN_SUPPRESSIBLE_REASON_SET)

    pin_info: Dict[str, Any] = {
        "status": "not_applied",
        "suppressed_failclose_reasons": [],
        "current_generation_id": str((((current_obj.get("coherence") or {}).get("build_generation_id") or "")).strip()) or None,
        "continuity_now_generation_id": str((((now_obj.get("coherence") or {}).get("build_generation_id") or "")).strip()) or None,
        "current_generated_at": str(current_obj.get("generated_at") or "").strip() or None,
        "pointer_current_generated_at": generation_pointer.get("pointer_current_generated_at"),
        "pointer_generation_id": generation_pointer.get("pointer_generation_id"),
        "live_current_generated_at": (live_current_artifact or {}).get("generated_at"),
        "live_current_generation_id": (live_current_artifact or {}).get("generation_id"),
        "live_current_matches_pointer": False,
        "continuity_now_contract_failclose_count": len(continuity_now_contract_failclose_reasons),
    }

    if not reasons:
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    if any(reason not in mismatch_reasons for reason in reasons):
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    current_generation = str((((current_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())
    now_generation = str((((now_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())
    if not current_generation or current_generation != now_generation:
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    if continuity_now_contract_failclose_reasons:
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    live_current_sha = str((live_current_artifact or {}).get("sha256") or "").strip()
    live_current_generation = str((live_current_artifact or {}).get("generation_id") or "").strip()
    live_current_generated_at = str((live_current_artifact or {}).get("generated_at") or "").strip()

    pointer_current_sha = str(generation_pointer.get("pointer_current_sha256") or "").strip()
    pointer_generation = str(generation_pointer.get("pointer_generation_id") or "").strip()
    pointer_current_generated_at = str(generation_pointer.get("pointer_current_generated_at") or "").strip()

    pointer_matches_live_current = bool(
        live_current_sha
        and pointer_current_sha
        and live_current_sha == pointer_current_sha
        and live_current_generation
        and pointer_generation
        and live_current_generation == pointer_generation
        and live_current_generated_at
        and pointer_current_generated_at
        and live_current_generated_at == pointer_current_generated_at
    )
    pin_info["live_current_matches_pointer"] = pointer_matches_live_current
    if not pointer_matches_live_current:
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    pinned_current_generated_at = str(current_obj.get("generated_at") or "").strip()
    pointer_current_generated_dt = parse_iso(pointer_current_generated_at)
    pinned_current_generated_dt = parse_iso(pinned_current_generated_at)
    live_pointer_newer_than_pinned = bool(
        pointer_current_generated_dt is not None
        and pinned_current_generated_dt is not None
        and pointer_current_generated_dt > pinned_current_generated_dt
    )

    if not live_pointer_newer_than_pinned:
        generation_pointer["read_phase_pin"] = pin_info
        return generation_pointer

    suppressed = [reason for reason in reasons if reason in mismatch_reasons]
    generation_pointer["failclose_reasons"] = [reason for reason in reasons if reason not in mismatch_reasons]
    pin_info["status"] = "applied"
    pin_info["suppressed_failclose_reasons"] = suppressed
    generation_pointer["read_phase_pin"] = pin_info
    return generation_pointer


def evaluate_generation_pointer_contract(
    *,
    current_obj: Dict[str, Any],
    now_obj: Dict[str, Any],
    current_sha: str,
) -> Dict[str, Any]:
    pointer_path = (root / "state" / "continuity" / "latest" / "continuity_read_pointer.json").resolve()
    out: Dict[str, Any] = {
        "path": str(pointer_path.relative_to(root)) if pointer_path.is_absolute() else str(pointer_path),
        "present": pointer_path.exists(),
        "schema": None,
        "pointer_generation_id": None,
        "pointer_generated_at": None,
        "pointer_current_path": None,
        "current_generation_id": None,
        "continuity_now_generation_id": None,
        "pointer_current_sha256": None,
        "current_sha256": current_sha,
        "current_sha_source": None,
        "pointer_current_generated_at": None,
        "current_generated_at": str(current_obj.get("generated_at") or "").strip() or None,
        "continuity_now_generated_at": str(now_obj.get("generated_at") or "").strip() or None,
        "failclose_reasons": [],
    }

    if not pointer_path.exists():
        return out

    try:
        pointer_obj = load_cached_json(pointer_path)
    except Exception as exc:
        out["failclose_reasons"] = ["generation_pointer_unreadable"]
        out["error"] = str(exc)
        return out

    if not isinstance(pointer_obj, dict):
        out["failclose_reasons"] = ["generation_pointer_not_object"]
        return out

    out["schema"] = pointer_obj.get("schema")
    if str(pointer_obj.get("schema") or "").strip() != "clawd.continuity.pointer.v1":
        out["failclose_reasons"].append("generation_pointer_schema_invalid")

    contract = pointer_obj.get("continuity_read_contract") if isinstance(pointer_obj.get("continuity_read_contract"), dict) else {}
    source_current = pointer_obj.get("source_current") if isinstance(pointer_obj.get("source_current"), dict) else {}

    out["pointer_generated_at"] = str(pointer_obj.get("generated_at") or "").strip() or None
    out["pointer_current_path"] = str(
        contract.get("continuity_current_path")
        or source_current.get("path")
        or ""
    ).strip() or None

    pointer_generation = str(
        contract.get("coherence_build_generation_id")
        or pointer_obj.get("coherence_build_generation_id")
        or ""
    ).strip()
    current_generation = str((((current_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())
    now_generation = str((((now_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())

    out["pointer_generation_id"] = pointer_generation or None
    out["current_generation_id"] = current_generation or None
    out["continuity_now_generation_id"] = now_generation or None

    pointer_current_sha = str(
        contract.get("continuity_current_sha256")
        or source_current.get("sha256")
        or ""
    ).strip()
    out["pointer_current_sha256"] = pointer_current_sha or None

    pointer_current_generated_at = str(
        contract.get("continuity_current_generated_at")
        or source_current.get("generated_at")
        or ""
    ).strip()
    out["pointer_current_generated_at"] = pointer_current_generated_at or None

    current_generated_at = str(current_obj.get("generated_at") or "").strip()

    out["failclose_reasons"] = unique_preserve(
        (out.get("failclose_reasons") or [])
        + _generation_pointer_core_failclose_reasons(
            pointer_current_sha256=pointer_current_sha,
            current_sha256=current_sha,
            pointer_current_generated_at=pointer_current_generated_at,
            current_generated_at=current_generated_at,
            pointer_generation_id=pointer_generation,
            current_generation_id=current_generation,
        )
    )
    return out


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


def validate_mission_control_contract(payload: Dict[str, Any]) -> None:
    if _validate_contract_payload_schema is None:
        raise RuntimeError(
            _format_schema_helper_unavailable_error(
                contract_prefix="operator_mission_control_contract",
                helper_path=schema_helper_path,
                import_error=_schema_helper_import_error,
            )
        )

    _validate_contract_payload_schema(
        payload,
        schema_path=mission_control_schema_path,
        contract_prefix="operator_mission_control_contract",
    )


def validate_reason_partition(*, not_ready_reasons: List[str], blocker_reasons: List[str], reconcile_only_reasons: List[str]) -> None:
    not_ready_set = set(not_ready_reasons)
    blocker_set = set(blocker_reasons)
    reconcile_set = set(reconcile_only_reasons)

    overlap = sorted(blocker_set & reconcile_set)
    combined = blocker_set | reconcile_set
    missing_from_partition = sorted(not_ready_set - combined)
    extras_outside_not_ready = sorted(combined - not_ready_set)

    if overlap or missing_from_partition or extras_outside_not_ready:
        raise RuntimeError(
            "operator_mission_control_contract_reason_partition_invalid:"
            f"overlap={json.dumps(overlap, ensure_ascii=False)}:"
            f"missing_from_partition={json.dumps(missing_from_partition, ensure_ascii=False)}:"
            f"extras_outside_not_ready={json.dumps(extras_outside_not_ready, ensure_ascii=False)}"
        )


def parse_json_object_output(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "")
    if not text.strip():
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None



def run_json(
    cmd: List[str],
    timeout_sec: Optional[int] = None,
    *,
    accept_nonzero_json_object: bool = False,
) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timeout:{timeout_sec}:{' '.join(cmd)}") from exc
    if cp.returncode != 0:
        nonzero_payload = parse_json_object_output(cp.stdout)
        if accept_nonzero_json_object and isinstance(nonzero_payload, dict):
            return nonzero_payload
        raise RuntimeError((cp.stderr or cp.stdout or "command_failed").strip())
    return json.loads(cp.stdout or "{}")


def load_cached_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_refresh_command(
    cmd: List[str],
    *,
    phase: str,
    timeout_sec: int,
    accept_nonzero_json_object: bool = False,
) -> None:
    try:
        cp = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"operator_mission_control_refresh_timeout:{phase}:timeout_sec={timeout_sec}:cmd={_format_cmd(cmd)}"
        ) from exc
    if cp.returncode != 0:
        nonzero_payload = parse_json_object_output(cp.stdout)
        if accept_nonzero_json_object and isinstance(nonzero_payload, dict):
            return
        err = (cp.stderr or cp.stdout or "command_failed").strip()[:240]
        raise SystemExit(
            f"operator_mission_control_refresh_failed:{phase}:{err}"
        )


def timed_or_cached_json(
    cmd: List[str],
    *,
    timeout_sec: int,
    cache_path: pathlib.Path,
    degraded_key: str,
    accept_nonzero_json_object: bool = False,
) -> Dict[str, Any]:
    try:
        payload = run_json(
            cmd,
            timeout_sec=timeout_sec,
            accept_nonzero_json_object=accept_nonzero_json_object,
        )
        if isinstance(payload, dict):
            payload.setdefault("contract_source", "live")
            payload.setdefault("contract_source_canonical", True)
            payload.setdefault("contract_source_degraded", False)
            payload.setdefault("contract_source_degraded_reason", None)
            payload.setdefault("contract_source_degraded_path", None)
        return payload
    except Exception as exc:
        try:
            payload = load_cached_json(cache_path)
            if isinstance(payload, dict):
                payload["contract_source"] = degraded_key
                payload["contract_source_canonical"] = False
                payload["contract_source_degraded"] = True
                payload["contract_source_degraded_reason"] = str(exc)
                payload["contract_source_degraded_path"] = str(cache_path.relative_to(root)) if cache_path.is_absolute() else str(cache_path)
            return payload
        except Exception:
            return {
                "generated_at": now_iso(),
                "summary": {
                    "status": "warn",
                },
                "contract_source": degraded_key,
                "contract_source_canonical": False,
                "contract_source_degraded": True,
                "contract_source_degraded_reason": str(exc),
                "contract_source_degraded_path": str(cache_path.relative_to(root)) if cache_path.is_absolute() else str(cache_path),
            }


def load_continuity_now_for_current(current_obj: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    raw_contract_obj = current_obj.get("continuity_now_contract")
    contract_obj = raw_contract_obj if isinstance(raw_contract_obj, dict) else {}
    source_refs = current_obj.get("source_refs") if isinstance(current_obj.get("source_refs"), dict) else {}

    contract_declared = _continuity_now_contract_declared(
        contract_obj=raw_contract_obj,
        source_refs=source_refs,
        require_sha_pin=False,
    )
    path_conflict_reason = _continuity_now_contract_path_conflict_reason(
        root,
        contract_obj=contract_obj,
        source_refs=source_refs,
    )
    contract_path = _resolve_continuity_now_contract_path(
        root,
        contract_obj=contract_obj,
        source_refs=source_refs,
    )
    expected_sha, expected_generated_at, expected_generation = _continuity_now_contract_expected_fields(
        contract_obj=contract_obj,
        source_refs=source_refs,
    )

    contract_info: Dict[str, Any] = {
        "declared": contract_declared,
        "path": _to_rel_or_abs_path(root, contract_path),
        "expected_sha256": expected_sha or None,
        "expected_generated_at": expected_generated_at or None,
        "expected_coherence_build_generation_id": expected_generation or None,
        "actual_sha256": None,
        "source": "live",
    }

    failclose_reasons, actual_sha, payload = _continuity_now_contract_failclose_reasons(
        contract_declared=contract_declared,
        contract_path=contract_path,
        expected_sha256=expected_sha,
        expected_generated_at=expected_generated_at,
        expected_coherence_build_generation_id=expected_generation,
    )
    failclose_reasons = unique_preserve(
        ([path_conflict_reason] if path_conflict_reason else [])
        + failclose_reasons
    )
    contract_info["actual_sha256"] = actual_sha

    if contract_declared and isinstance(payload, dict):
        payload["contract_source"] = "continuity_now_pinned_current_contract"
        payload["contract_source_canonical"] = True
        payload["contract_source_degraded"] = bool(failclose_reasons)
        payload["contract_source_degraded_reason"] = ";".join(failclose_reasons) if failclose_reasons else None
        payload["contract_source_degraded_path"] = contract_info["path"]
        contract_info["source"] = "current_contract"
        return payload, contract_info, unique_preserve(failclose_reasons)

    payload = timed_or_cached_json(
        [str(continuity_now_script), "--json"],
        timeout_sec=20,
        cache_path=root / "state" / "continuity" / "latest" / "continuity_now_latest.json",
        degraded_key="continuity_now_cached_fallback",
        accept_nonzero_json_object=True,
    )
    if not isinstance(payload, dict):
        payload = {}
    if contract_declared:
        payload["contract_source"] = "continuity_now_live_fallback"
        payload["contract_source_canonical"] = False
        payload["contract_source_degraded"] = True
        payload["contract_source_degraded_reason"] = ";".join(failclose_reasons) if failclose_reasons else "continuity_now_contract_degraded"
        payload["contract_source_degraded_path"] = contract_info["path"]
    return payload, contract_info, unique_preserve(failclose_reasons)


if refresh:
    run_refresh_command(
        [str(current_script), "--refresh", "--json"],
        phase="current_refresh",
        timeout_sec=refresh_current_timeout_sec,
        accept_nonzero_json_object=True,
    )
    # continuity_current has already republished state/continuity/current.json in
    # this refresh lane. Rebuild handover from that freshly published current
    # surface without forcing a second continuity_current --refresh call inside
    # handover_latest.sh, which would otherwise open another publish-lock window.
    run_refresh_command(
        [str(handover_script), "--json"],
        phase="handover_refresh",
        timeout_sec=refresh_handover_timeout_sec,
    )
    run_refresh_command(
        [str(gate_os_script), "--refresh", "--json"],
        phase="gate_os_refresh",
        timeout_sec=refresh_gate_os_timeout_sec,
    )

if refresh:
    current = load_json_if_exists(current_path) or {}
else:
    current = timed_or_cached_json(
        [str(current_script), "--json"],
        timeout_sec=live_current_timeout_sec,
        cache_path=root / "state" / "continuity" / "current.json",
        degraded_key="continuity_current_cached_fallback",
        accept_nonzero_json_object=True,
    )

if not isinstance(current, dict) or not current:
    current = timed_or_cached_json(
        [str(current_script), "--json"],
        timeout_sec=live_current_timeout_sec,
        cache_path=root / "state" / "continuity" / "current.json",
        degraded_key="continuity_current_cached_fallback",
        accept_nonzero_json_object=True,
    )
current_publish_lock_timeout = parse_continuity_current_publish_lock_timeout(
    current.get("contract_source_degraded_reason") if isinstance(current, dict) else None
)
blocker_registry_latest = load_json_if_exists(blocker_registry_path) or {}
execution_program_status_obj = load_json_if_exists(execution_program_status_path) or {}
execution_frontier_ledger_obj = load_json_if_exists(execution_frontier_ledger_path) or {}
execution_supervisor_dispatch_intent_obj = load_json_if_exists(execution_supervisor_dispatch_intent_path) or {}
execution_supervisor_dispatch_qualification_obj = load_json_if_exists(execution_supervisor_dispatch_qualification_path) or {}
execution_supervisor_probe_execution_plan_obj = load_json_if_exists(execution_supervisor_probe_execution_plan_path) or {}
execution_meaningful_event_reporting_obj = load_json_if_exists(execution_meaningful_event_reporting_path) or {}
execution_meaningful_event_reporting_status_obj = load_json_if_exists(execution_meaningful_event_reporting_status_path) or {}
model_rollout_dashboard_obj = load_json_if_exists(model_rollout_dashboard_path) or {}
dependency_policy_pack_obj = load_json_if_exists(dependency_policy_pack_path) or {}
efficiency_kpi_baseline_obj = load_json_if_exists(efficiency_kpi_baseline_path) or {}
efficiency_kpi_validation_obj = load_json_if_exists(efficiency_kpi_validation_path) or {}
publish_lock_registry_signal = _project_blocker_registry_publish_lock_signal(
    blocker_registry=blocker_registry_latest,
    current_generated_at=current.get("generated_at") if isinstance(current, dict) else None,
)
action_token = str(current.get("action_token") or "").strip()
action_token_arg = shlex.quote(action_token) if action_token else "<action_token>"

LEGACY_ROOT_LITERAL = "/home/yeqiuqiu/clawd-architect"


def shell_cmd_for(rel_path: str, *args: str) -> str:
    script_path = (root / rel_path).resolve()
    base = f"bash {shlex.quote(str(script_path))}"
    return f"{base} {' '.join(args)}".strip()


def cat_cmd_for(path_value: Any) -> str:
    path = pathlib.Path(str(path_value or "").strip())
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return f"cat {shlex.quote(str(path))}"


def normalize_operator_command(raw: Any) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return txt
    return txt.replace(LEGACY_ROOT_LITERAL, str(root))


cmd_cont_current_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "current", "--refresh", "--json")
cmd_cont_worker_health_canary_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "worker-health-canary", "--json")
cmd_cont_verify_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify", "--json")
cmd_cont_queue_sync_json = shell_cmd_for("ops/openclaw/continuity.sh", "queue-sync", "--json")
cmd_cont_queue_replay_json = shell_cmd_for("ops/openclaw/continuity.sh", "queue-replay", "--json")
cmd_cont_queue_replay_strict_json = shell_cmd_for("ops/openclaw/continuity.sh", "queue-replay", "--strict", "--json")
cmd_cont_execution_frontier_show_json = shell_cmd_for("ops/openclaw/continuity.sh", "execution-frontier", "--json")
cmd_cont_execution_frontier_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "execution-frontier", "--refresh", "--json")
cmd_cont_failover_replay_evidence_json = shell_cmd_for("ops/openclaw/continuity.sh", "failover-replay-evidence", "--json")
cmd_cont_failover_stress_soak_json = shell_cmd_for("ops/openclaw/continuity.sh", "failover-stress-soak", "--json")
cmd_cont_failover_stress_runtime_evidence_json = shell_cmd_for("ops/openclaw/continuity.sh", "failover-stress-runtime-evidence", "--json")
cmd_cont_gtc_replay_json = shell_cmd_for("ops/openclaw/continuity.sh", "gtc-replay", "--json")
cmd_cont_verify_gate_status_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify-gate-status", "--json")
cmd_cont_model_route_policy_lint_json = shell_cmd_for("ops/openclaw/continuity.sh", "model-route-policy-lint", "--json")
cmd_cont_model_rollout_dashboard_json = shell_cmd_for("ops/openclaw/continuity.sh", "model-rollout-dashboard", "--json")
cmd_cont_model_rollout_controller_json = shell_cmd_for("ops/openclaw/continuity.sh", "model-rollout-controller", "--json")
cmd_refresh_efficiency_kpi_baseline_json = shell_cmd_for("ops/openclaw/continuity/efficiency_kpi_baseline_snapshot.py", "--json")
cmd_cont_mission_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "mission-control", "--refresh", "--json")
cmd_cont_execution_frontier_advance_wave_close_json = (
    f"{shell_cmd_for('ops/openclaw/continuity.sh')} --action-token {action_token_arg} "
    f"execution-frontier supervisor-advance-wave-close --reason {shlex.quote('mission_control_frontier_wave_close')} --json"
)
cmd_cont_execution_frontier_autonomous_dispatch_json = (
    f"{shell_cmd_for('ops/openclaw/continuity.sh')} --action-token {action_token_arg} "
    f"execution-frontier supervisor-autonomous-dispatch --reason {shlex.quote('mission_control_frontier_autonomous_dispatch')} --json"
)
cmd_cont_reset_ready_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "reset-ready-refresh", "--json")
cmd_cont_librarian_hygiene_json = shell_cmd_for("ops/openclaw/continuity.sh", "librarian", "hygiene", "--skip-retrieval-eval", "--json")
cmd_idle_lane_watchdog_history_json = shell_cmd_for(
    "ops/openclaw/continuity/history.sh",
    "--source-preset",
    "watchdogs",
    "--source",
    "watchdog.no_nudge_continuity",
    "--hours",
    "24",
    "--include-suppressed",
    "--json",
)
cmd_queue_ready_list_json = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "ready-list", "--json")
cmd_queue_handoffs_json = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "handoffs", "--limit", "20", "--json")
cmd_queue_remediate_json = shell_cmd_for("ops/openclaw/continuity/queue_arbitrator.sh", "remediate", "--json")
cmd_queue_remediate_apply_json = shell_cmd_for(
    "ops/openclaw/continuity/queue_arbitrator.sh",
    "remediate",
    "--expire-overdue-locks",
    "--release-terminal-locks",
    "--requeue-orphaned-running",
    "--apply",
    "--json",
)
cmd_watchdog_json = shell_cmd_for("ops/openclaw/run_no_nudge_continuity_watchdog.sh", "--json")
cmd_parity_dry_run_json = shell_cmd_for("ops/openclaw/run_competitive_parity_harness.sh", "--dry-run", "--json")
cmd_parity_force = shell_cmd_for("ops/openclaw/run_competitive_parity_harness.sh", "--force")
cmd_web_capture_auto_json = shell_cmd_for("ops/openclaw/run_web_capture_macro.sh", "--mode", "auto", "--json")
cmd_web_capture_fetch_json = shell_cmd_for("ops/openclaw/run_web_capture_macro.sh", "--mode", "fetch", "--json")
cmd_web_capture_auto_dry_json = shell_cmd_for("ops/openclaw/run_web_capture_macro.sh", "--mode", "auto", "--dry-run", "--json")
cmd_web_capture_scheduler_dry_json = shell_cmd_for("ops/openclaw/run_web_capture_scheduler.sh", "--dry-run", "--json")
cmd_read_pointer_json = cat_cmd_for("state/continuity/latest/continuity_read_pointer.json")
cmd_read_proof_json = cat_cmd_for("state/continuity/latest/successor_safe_handover_proof.json")
cmd_read_proof_status_json = cat_cmd_for("state/continuity/latest/successor_safe_handover_proof_status.json")
cmd_read_reset_ready_refresh_latest_json = cat_cmd_for("state/continuity/latest/reset_ready_refresh_latest.json")
cmd_read_idle_lane_latch_json = cat_cmd_for("state/continuity/latest/no_nudge_idle_lane_autospawn_contradiction_latch.json")
cmd_read_execution_frontier_controller_trace_json = cat_cmd_for(
    "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"
)
cmd_read_execution_frontier_controller_history_json = cat_cmd_for(
    "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl"
)
cmd_read_execution_frontier_enforcement_latch_json = cat_cmd_for(
    "state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json"
)
cmd_read_execution_frontier_enforcement_latch_history_json = cat_cmd_for(
    "state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl"
)
cmd_read_autonomous_execution_intent_json = cat_cmd_for(
    "state/continuity/latest/autonomous_execution_intent_latest.json"
)
cmd_read_execution_supervisor_dispatch_intent_json = cat_cmd_for(
    "state/continuity/latest/execution_supervisor_dispatch_intent_latest.json"
)
cmd_read_execution_supervisor_dispatch_qualification_json = cat_cmd_for(
    "state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json"
)
cmd_read_execution_supervisor_probe_execution_plan_json = cat_cmd_for(
    "state/continuity/latest/execution_supervisor_probe_execution_plan_latest.json"
)
cmd_read_execution_supervisor_worker_health_canary_json = cat_cmd_for(
    "state/continuity/latest/execution_supervisor_worker_health_canary_latest.json"
)
cmd_read_autonomous_execution_intent_history_json = cat_cmd_for(
    "state/continuity/history/autonomous_execution_intent_history.jsonl"
)
cmd_read_load_shedding_decision_json = cat_cmd_for("state/continuity/latest/load_shedding_decision.json")
cmd_read_load_shedding_signal_snapshot_json = cat_cmd_for("state/continuity/latest/load_shedding_signal_snapshot.json")
cmd_read_wave2_replay_evidence_index_json = cat_cmd_for("state/continuity/latest/wave2_replay_evidence_index.json")
cmd_read_failover_stress_soak_evidence_json = cat_cmd_for("state/continuity/latest/failover_stress_soak_evidence.json")
cmd_read_failover_stress_runtime_evidence_json = cat_cmd_for("state/continuity/latest/failover_stress_runtime_evidence.json")
cmd_read_current_publish_lock_owner_json = cat_cmd_for(current_publish_lock_owner_rel)
cmd_read_execution_meaningful_event_reporting_json = cat_cmd_for(
    "state/continuity/latest/execution_meaningful_event_reporting_latest.json"
)
cmd_read_execution_meaningful_event_reporting_status_json = cat_cmd_for(
    "state/continuity/latest/execution_meaningful_event_reporting_status_latest.json"
)
cmd_read_model_rollout_dashboard_json = cat_cmd_for("state/continuity/model_rollout_dashboard/latest.json")
cmd_read_efficiency_kpi_baseline_json = cat_cmd_for("state/continuity/latest/efficiency_kpi_baseline_latest.json")
cmd_read_efficiency_kpi_validation_json = cat_cmd_for("state/continuity/latest/efficiency_kpi_baseline_validation_latest.json")
cmd_read_evidence_trace_viewer_json = cat_cmd_for("state/continuity/latest/evidence_trace_viewer_latest.json")

handover = timed_or_cached_json(
    [str(handover_script), "--json"],
    timeout_sec=live_handover_timeout_sec,
    cache_path=root / "state" / "handover" / "latest.json",
    degraded_key="handover_cached_fallback",
)
handover_safe_signals = handover.get("safe_signals") if isinstance(handover.get("safe_signals"), dict) else {}
handover_proof_status = handover.get("proof_status") if isinstance(handover.get("proof_status"), dict) else {}
proof_gate_enforced = bool(
    handover_safe_signals.get("proof_gate_enforced") is True
    or handover_proof_status.get("proof_gate_enforced") is True
)
proof_state = str(
    handover_safe_signals.get("proof_state")
    or handover_proof_status.get("proof_state")
    or "PROOF_MISSING"
).strip() or "PROOF_MISSING"
proof_top_blocker = str(
    handover_safe_signals.get("proof_top_blocker")
    or handover_proof_status.get("top_blocker")
    or ("BLK_PROOF_MISSING" if proof_state == "PROOF_MISSING" else "")
).strip() or None
proof_resume_allowed = bool(
    handover_safe_signals.get("proof_resume_allowed") is True
    or handover_proof_status.get("resume_allowed") is True
)
proof_reset_allowed = bool(
    handover_safe_signals.get("proof_reset_allowed") is True
    or handover_proof_status.get("reset_allowed") is True
)
proof_fail_closed = bool(
    proof_gate_enforced
    and (
        proof_state != "PROOF_VALID_PASS"
        or not proof_resume_allowed
        or not proof_reset_allowed
    )
)
proof_failclose_reasons = [f"successor_proof_gate:{proof_state}"] if proof_fail_closed else []
now_obj, continuity_now_contract, continuity_now_contract_failclose_reasons = load_continuity_now_for_current(current)
continuity_now_evidence_path = _resolve_continuity_now_evidence_path(
    root,
    raw_path=continuity_now_contract.get("path") if isinstance(continuity_now_contract, dict) else None,
    fallback_rel=_CONTINUITY_NOW_LATEST_REL,
)
gate_os = timed_or_cached_json(
    [str(gate_os_script), "--json"],
    timeout_sec=live_gate_os_timeout_sec,
    cache_path=root / "state" / "continuity" / "latest" / "gate_os_latest.json",
    degraded_key="gate_os_cached_fallback",
)
queue_replay = timed_or_cached_json(
    [str(queue_replay_script), "--json"],
    timeout_sec=10,
    cache_path=root / "state" / "continuity" / "latest" / "queue_replay_verify.json",
    degraded_key="queue_replay_cached_fallback",
)
queue = now_obj.get("queue") or {}
queue_stale_wave_signal = queue.get("stale_wave_signal") if isinstance(queue.get("stale_wave_signal"), dict) else {}
verify = now_obj.get("verify") or {}
verify_gate_preflight = verify.get("gate_preflight") if isinstance(verify.get("gate_preflight"), dict) else {}
verify_gate_preflight_strict = verify_gate_preflight.get("strict_autonomy") if isinstance(verify_gate_preflight.get("strict_autonomy"), dict) else {}
verify_gate_preflight_predicted = verify_gate_preflight.get("predicted_gate") if isinstance(verify_gate_preflight.get("predicted_gate"), dict) else {}
verify_gate_preflight_status_evidence = verify_gate_preflight.get("status_evidence_gate") if isinstance(verify_gate_preflight.get("status_evidence_gate"), dict) else {}
verify_gate_preflight_internal_bypass_stage_b = verify_gate_preflight.get("internal_bypass_stage_b") if isinstance(verify_gate_preflight.get("internal_bypass_stage_b"), dict) else {}
verify_gate_preflight_layered_health = verify_gate_preflight.get("layered_health_gate") if isinstance(verify_gate_preflight.get("layered_health_gate"), dict) else {}
verify_gate_preflight_launch_readiness_worker_health_canary = verify_gate_preflight.get("launch_readiness_worker_health_canary_gate") if isinstance(verify_gate_preflight.get("launch_readiness_worker_health_canary_gate"), dict) else {}
verify_gate_preflight_launch_readiness_probe_execution = verify_gate_preflight.get("launch_readiness_probe_execution_gate") if isinstance(verify_gate_preflight.get("launch_readiness_probe_execution_gate"), dict) else {}
verify_status_evidence_failure_reason = str(verify_gate_preflight_status_evidence.get("failure_reason") or "").strip()
verify_internal_bypass_closeout_failure_reason = str(verify_gate_preflight_internal_bypass_stage_b.get("closeout_failure_reason") or "").strip()
verify_internal_bypass_closeout_ready = verify_gate_preflight_internal_bypass_stage_b.get("closeout_ready") if isinstance(verify_gate_preflight_internal_bypass_stage_b.get("closeout_ready"), bool) else None
verify_internal_bypass_unknown_total = int(verify_gate_preflight_internal_bypass_stage_b.get("unknown_callsite_total") or 0)
verify_internal_bypass_break_glass_allow = int(verify_gate_preflight_internal_bypass_stage_b.get("break_glass_allow_count") or 0)
verify_internal_bypass_break_glass_denied = int(verify_gate_preflight_internal_bypass_stage_b.get("break_glass_denied_count") or 0)
verify_layered_health_failure_reason = str(verify_gate_preflight_layered_health.get("failure_reason") or "").strip()
verify_layered_health_closeout_ready = verify_gate_preflight_layered_health.get("closeout_ready") if isinstance(verify_gate_preflight_layered_health.get("closeout_ready"), bool) else None
verify_layered_health_status = str(verify_gate_preflight_layered_health.get("health_status") or "unknown").strip() or "unknown"
verify_layered_health_layer = str(verify_gate_preflight_layered_health.get("health_layer") or "unknown").strip() or "unknown"
verify_layered_health_restore_slo_status = str(verify_gate_preflight_layered_health.get("restore_slo_status") or "unknown").strip() or "unknown"
verify_worker_health_canary_preflight_failure_reason = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("failure_reason") or ""
).strip()
verify_worker_health_canary_preflight_active_blocker = bool(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("active_blocker") is True
)
verify_worker_health_canary_preflight_dispatch_failure_reason = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("dispatch_qualification_failure_reason") or ""
).strip()
verify_worker_health_canary_preflight_source = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("worker_health_canary_source") or ""
).strip() or None
verify_worker_health_canary_preflight_action_priority = str(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("action_priority") or ""
).strip().lower()
if verify_worker_health_canary_preflight_action_priority not in {"p1", "p2"}:
    verify_worker_health_canary_preflight_action_priority = None
verify_worker_health_canary_preflight_first_actionable_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("first_actionable_command")
) if verify_gate_preflight_launch_readiness_worker_health_canary.get("first_actionable_command") else None
verify_worker_health_canary_preflight_inspect_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("inspect_worker_health_canary_command")
) if verify_gate_preflight_launch_readiness_worker_health_canary.get("inspect_worker_health_canary_command") else cmd_read_execution_supervisor_worker_health_canary_json
verify_worker_health_canary_preflight_refresh_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_worker_health_canary.get("refresh_worker_health_canary_command")
) if verify_gate_preflight_launch_readiness_worker_health_canary.get("refresh_worker_health_canary_command") else cmd_cont_worker_health_canary_refresh_json
verify_probe_execution_gate_failure_reason = str(verify_gate_preflight_launch_readiness_probe_execution.get("failure_reason") or "").strip()
verify_probe_execution_gate_active_blocker = bool(verify_gate_preflight_launch_readiness_probe_execution.get("active_blocker") is True)
verify_probe_execution_gate_due_now_worker_count = _nonnegative_int(verify_gate_preflight_launch_readiness_probe_execution.get("due_now_worker_count"))
verify_probe_execution_gate_overdue_worker_count = _nonnegative_int(verify_gate_preflight_launch_readiness_probe_execution.get("overdue_worker_count"))
verify_probe_execution_gate_pending_worker_count = _nonnegative_int(verify_gate_preflight_launch_readiness_probe_execution.get("pending_worker_count"))
verify_probe_execution_gate_oldest_due_now_worker = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_due_now_worker") or ""
).strip() or None
verify_probe_execution_gate_oldest_due_now_started_at = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_due_now_started_at") or ""
).strip() or None
verify_probe_execution_gate_oldest_due_now_age_sec = _optional_nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_due_now_age_sec")
)
verify_probe_execution_gate_oldest_overdue_worker = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_overdue_worker") or ""
).strip() or None
verify_probe_execution_gate_oldest_overdue_started_at = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_overdue_started_at") or ""
).strip() or None
verify_probe_execution_gate_oldest_overdue_age_sec = _optional_nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("oldest_overdue_age_sec")
)
verify_probe_execution_gate_demotion_restore_pending_worker_count = _nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_restore_pending_worker_count")
)
verify_probe_execution_gate_demotion_demoted_worker_count = _nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_demoted_worker_count")
)
verify_probe_execution_gate_demotion_restored_worker_count = _nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_restored_worker_count")
)
verify_probe_execution_gate_demotion_oldest_restore_pending_since = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_restore_pending_since") or ""
).strip() or None
verify_probe_execution_gate_demotion_oldest_restore_pending_worker = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_restore_pending_worker") or ""
).strip() or None
verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec = _optional_nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_restore_pending_age_sec")
)
verify_probe_execution_gate_demotion_oldest_demoted_at = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_demoted_at") or ""
).strip() or None
verify_probe_execution_gate_demotion_oldest_demoted_worker = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_demoted_worker") or ""
).strip() or None
verify_probe_execution_gate_demotion_oldest_demoted_age_sec = _optional_nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_oldest_demoted_age_sec")
)
verify_probe_execution_gate_demotion_latest_restored_at = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_latest_restored_at") or ""
).strip() or None
verify_probe_execution_gate_demotion_latest_restored_worker = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_latest_restored_worker") or ""
).strip() or None
verify_probe_execution_gate_demotion_latest_restored_age_sec = _optional_nonnegative_int(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_latest_restored_age_sec")
)
verify_probe_execution_gate_demotion_action_priority = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("demotion_action_priority") or ""
).strip().lower() or None
if verify_probe_execution_gate_demotion_action_priority not in {"p1", "p2"}:
    verify_probe_execution_gate_demotion_action_priority = None
verify_probe_execution_gate_probe_plan_path = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("probe_execution_plan_path") or ""
).strip() or None
verify_probe_execution_gate_probe_plan_present = (
    verify_gate_preflight_launch_readiness_probe_execution.get("probe_execution_plan_present")
    if isinstance(verify_gate_preflight_launch_readiness_probe_execution.get("probe_execution_plan_present"), bool)
    else None
)
verify_probe_execution_gate_action_priority = str(
    verify_gate_preflight_launch_readiness_probe_execution.get("action_priority") or ""
).strip() or None
verify_probe_execution_gate_inspect_dispatch_qualification_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_probe_execution.get("inspect_dispatch_qualification_command")
) if verify_gate_preflight_launch_readiness_probe_execution.get("inspect_dispatch_qualification_command") else None
verify_probe_execution_gate_inspect_probe_execution_plan_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_probe_execution.get("inspect_probe_execution_plan_command")
) if verify_gate_preflight_launch_readiness_probe_execution.get("inspect_probe_execution_plan_command") else None
verify_probe_execution_gate_refresh_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_probe_execution.get("refresh_dispatch_qualification_command")
) if verify_gate_preflight_launch_readiness_probe_execution.get("refresh_dispatch_qualification_command") else None
verify_probe_execution_gate_first_actionable_command = normalize_operator_command(
    verify_gate_preflight_launch_readiness_probe_execution.get("first_actionable_command")
) if verify_gate_preflight_launch_readiness_probe_execution.get("first_actionable_command") else None
verify_layered_health_missing_required_lanes = [
    str(x).strip() for x in (verify_gate_preflight_layered_health.get("missing_required_lanes") or []) if str(x).strip()
]
verify_layered_health_failing_required_lanes = [
    str(x).strip() for x in (verify_gate_preflight_layered_health.get("failing_required_lanes") or []) if str(x).strip()
]
verify_layered_health_layer_insufficient_required_lanes = [
    str(x).strip() for x in (verify_gate_preflight_layered_health.get("layer_insufficient_required_lanes") or []) if str(x).strip()
]
routing_decisions_path = root / "state" / "continuity" / "session_topology_router" / "decisions.jsonl"
routing_max_age_sec = _read_nonnegative_int_env("OPENCLAW_OPERATOR_ROUTING_MAX_AGE_SEC", default=21600)
verify_gate_preflight_routing = verify_gate_preflight.get("routing_preflight") if isinstance(verify_gate_preflight.get("routing_preflight"), dict) else {}
if verify_gate_preflight_routing:
    routing_preflight = dict(verify_gate_preflight_routing)
    routing_preflight_source = "continuity_now_verify_gate_preflight"
else:
    routing_preflight = summarize_live_routing_preflight(
        decisions_path=routing_decisions_path,
        max_age_sec=routing_max_age_sec,
        inspect_decisions_command=f"tail -n 60 {routing_decisions_path}",
        recheck_policy_command=cmd_cont_model_route_policy_lint_json,
        reference_now=now_obj.get("generated_at") or current.get("generated_at"),
    )
    routing_preflight_source = "operator_local_scan"
routing_preflight_latest = routing_preflight.get("latest") if isinstance(routing_preflight.get("latest"), dict) else {}
routing_preflight_effective = routing_preflight.get("effective") if isinstance(routing_preflight.get("effective"), dict) else {}
routing_failure_reason = str(routing_preflight.get("failure_reason") or "").strip()
routing_blocked_fresh = bool(routing_preflight_effective.get("blocked_fresh") is True)
if routing_failure_reason == "routing_decision_stale":
    routing_decision = "STALE"
    routing_route_class = None
    routing_selected_model = None
    routing_block_gate = None
    routing_block_reason = None
else:
    routing_decision = str(routing_preflight_latest.get("decision") or "").strip().upper() or None
    routing_route_class = str(
        routing_preflight_effective.get("route_class")
        or routing_preflight_latest.get("route_class")
        or ""
    ).strip() or None
    routing_selected_model = str(
        routing_preflight_effective.get("selected_model")
        or routing_preflight_latest.get("selected_model")
        or ""
    ).strip() or None
    routing_block_gate = str(
        routing_preflight_effective.get("block_gate")
        or routing_preflight_latest.get("block_gate")
        or ""
    ).strip() or None
    routing_block_reason = str(
        routing_preflight_effective.get("block_reason")
        or routing_preflight_latest.get("block_reason")
        or ""
    ).strip() or None
routing_next_safe_action = normalize_operator_command(
    routing_preflight_effective.get("first_actionable_command")
    or routing_preflight_effective.get("inspect_decisions_command")
    or cmd_cont_verify_gate_status_json
)
model_rollout_dashboard = model_rollout_dashboard_obj if isinstance(model_rollout_dashboard_obj, dict) else {}
model_rollout_dashboard_status = str(model_rollout_dashboard.get("status") or "unknown").strip() or "unknown"
model_rollout_bakeoff = model_rollout_dashboard.get("bakeoff") if isinstance(model_rollout_dashboard.get("bakeoff"), dict) else {}
model_rollout_cockpit_action_prompt = (
    model_rollout_bakeoff.get("cockpit_action_prompt")
    if isinstance(model_rollout_bakeoff.get("cockpit_action_prompt"), dict)
    else (
        model_rollout_dashboard.get("cockpit_action_prompt")
        if isinstance(model_rollout_dashboard.get("cockpit_action_prompt"), dict)
        else {}
    )
)
model_rollout_prompt_status = str(model_rollout_cockpit_action_prompt.get("status") or "none").strip() or "none"
model_rollout_prompt_reason = str(model_rollout_cockpit_action_prompt.get("reason") or "none").strip() or "none"
model_rollout_prompt_hint = str(model_rollout_cockpit_action_prompt.get("hint") or "").strip() or None
model_rollout_prompt_requires_approval = bool(model_rollout_cockpit_action_prompt.get("requires_operator_approval") is True)
model_rollout_prompt_scorecard_ref = str(model_rollout_cockpit_action_prompt.get("scorecard_ref") or "").strip() or None
model_rollout_prompt_commands = [
    normalize_operator_command(cmd)
    for cmd in (
        model_rollout_cockpit_action_prompt.get("commands")
        if isinstance(model_rollout_cockpit_action_prompt.get("commands"), list)
        else []
    )
    if normalize_operator_command(cmd)
]
model_rollout_prompt_first_command = model_rollout_prompt_commands[0] if model_rollout_prompt_commands else None
model_rollout_operator_mistake_remediation = (
    model_rollout_dashboard.get("operator_mistake_remediation")
    if isinstance(model_rollout_dashboard.get("operator_mistake_remediation"), dict)
    else {}
)
model_rollout_remediation_status = str(model_rollout_operator_mistake_remediation.get("status") or "none").strip() or "none"
model_rollout_remediation_active = bool(model_rollout_operator_mistake_remediation.get("active") is True)
model_rollout_remediation_reason_gate = str(model_rollout_operator_mistake_remediation.get("reason_gate") or "none").strip() or "none"
model_rollout_remediation_reason_code = str(model_rollout_operator_mistake_remediation.get("reason_code") or "none").strip() or "none"
model_rollout_remediation_correction_cycle_log_ref = str(
    model_rollout_operator_mistake_remediation.get("correction_cycle_log_ref") or ""
).strip() or None
model_rollout_remediation_hint = str(model_rollout_operator_mistake_remediation.get("hint") or "").strip() or None
model_rollout_remediation_operator_message = str(
    model_rollout_operator_mistake_remediation.get("operator_message") or ""
).strip() or None
model_rollout_remediation_commands = [
    normalize_operator_command(cmd)
    for cmd in (
        model_rollout_operator_mistake_remediation.get("safe_remediation_commands")
        if isinstance(model_rollout_operator_mistake_remediation.get("safe_remediation_commands"), list)
        else []
    )
    if normalize_operator_command(cmd)
]
if not model_rollout_remediation_commands and isinstance(
    model_rollout_operator_mistake_remediation.get("safe_remediation_options"),
    list,
):
    model_rollout_remediation_commands = [
        normalize_operator_command(((row or {}).get("command") if isinstance(row, dict) else None))
        for row in model_rollout_operator_mistake_remediation.get("safe_remediation_options")
        if normalize_operator_command(((row or {}).get("command") if isinstance(row, dict) else None))
    ]
model_rollout_remediation_first_command = model_rollout_remediation_commands[0] if model_rollout_remediation_commands else None
model_rollout_next_safe_action = (
    model_rollout_remediation_first_command
    or model_rollout_prompt_first_command
    or cmd_cont_model_rollout_dashboard_json
)

dependency_policy_pack = dependency_policy_pack_obj if isinstance(dependency_policy_pack_obj, dict) else {}
dependency_policy_pack_slices = dependency_policy_pack.get("slices") if isinstance(dependency_policy_pack.get("slices"), dict) else {}
dependency_policy_pack_slice_ids = sorted(
    [
        str(slice_id).strip()
        for slice_id in dependency_policy_pack_slices.keys()
        if str(slice_id).strip()
    ]
)
dependency_policy_pack_required_slice_ids = ["12", "22", "30"]
dependency_policy_pack_required_slice_coverage = all(
    slice_id in dependency_policy_pack_slice_ids
    for slice_id in dependency_policy_pack_required_slice_ids
)
dependency_policy_pack_status = str(dependency_policy_pack.get("status") or "missing").strip() or "missing"
dependency_policy_pack_schema = str(dependency_policy_pack.get("schema") or "").strip()
dependency_policy_pack_schema_ok = dependency_policy_pack_schema == "clawd.core_roadmap_dependency_unblock_policy_pack.v1"
dependency_policy_pack_policy_ids = {
    slice_id: str(((dependency_policy_pack_slices.get(slice_id) or {}).get("policy_id") or "").strip()) or None
    for slice_id in dependency_policy_pack_required_slice_ids
}
dependency_policy_pack_projection = {
    "path": str(dependency_policy_pack_path.relative_to(root)),
    "present": bool(dependency_policy_pack),
    "schema": dependency_policy_pack_schema or None,
    "schema_ok": dependency_policy_pack_schema_ok,
    "status": dependency_policy_pack_status,
    "generated_at": str(dependency_policy_pack.get("generated_at") or "").strip() or None,
    "policy_pack_id": str(dependency_policy_pack.get("policy_pack_id") or "").strip() or None,
    "slice_count": len(dependency_policy_pack_slice_ids),
    "slice_ids": dependency_policy_pack_slice_ids,
    "required_slice_ids": dependency_policy_pack_required_slice_ids,
    "required_slice_coverage": dependency_policy_pack_required_slice_coverage,
    "policy_ids": dependency_policy_pack_policy_ids,
}

efficiency_kpi_baseline = efficiency_kpi_baseline_obj if isinstance(efficiency_kpi_baseline_obj, dict) else {}
efficiency_kpi_validation = efficiency_kpi_validation_obj if isinstance(efficiency_kpi_validation_obj, dict) else {}
efficiency_kpi_status = str(efficiency_kpi_baseline.get("status") or "missing").strip() or "missing"
efficiency_kpi_validation_status = str(efficiency_kpi_validation.get("status") or "missing").strip() or "missing"
efficiency_kpi_summary = efficiency_kpi_baseline.get("summary") if isinstance(efficiency_kpi_baseline.get("summary"), dict) else {}
efficiency_kpi_kpi_count = int(efficiency_kpi_summary.get("kpi_count") or 0)
efficiency_kpi_no_signal_count = int(efficiency_kpi_summary.get("no_signal_count") or 0)
efficiency_kpi_measured_count = int(efficiency_kpi_summary.get("measured_or_derived_count") or 0)
efficiency_kpi_next_safe_action = (
    cmd_read_efficiency_kpi_baseline_json
    if efficiency_kpi_baseline
    else cmd_refresh_efficiency_kpi_baseline_json
)

parity = now_obj.get("parity") or {}
autopilot_now = now_obj.get("autopilot") if isinstance(now_obj.get("autopilot"), dict) else {}
autopilot_degraded_pending_signal = autopilot_now.get("degraded_pending_stale_signal") if isinstance(autopilot_now.get("degraded_pending_stale_signal"), dict) else {}
autopilot_idle_lane_autospawn = autopilot_now.get("idle_lane_autospawn") if isinstance(autopilot_now.get("idle_lane_autospawn"), dict) else {}
not_ready_reasons = [str(x) for x in (now_obj.get("not_ready_reasons") or []) if str(x).strip()]
warning_reasons = [str(x) for x in (now_obj.get("warning_reasons") or []) if str(x).strip()]

reset_ready_refresh_rel = "state/continuity/latest/reset_ready_refresh_latest.json"
reset_ready_refresh_path = root / reset_ready_refresh_rel
reset_ready_refresh_latest = load_json_if_exists(reset_ready_refresh_path)
if not isinstance(reset_ready_refresh_latest, dict):
    reset_ready_refresh_latest = {}
reset_ready_refresh_projection = _project_reset_ready_refresh_posture(
    latest_payload=reset_ready_refresh_latest,
    path=reset_ready_refresh_rel,
    present=reset_ready_refresh_path.exists() and bool(reset_ready_refresh_latest),
)
reset_ready_refresh_present = bool(reset_ready_refresh_projection.get("present") is True)
reset_ready_refresh_ok = reset_ready_refresh_projection.get("ok") if isinstance(reset_ready_refresh_projection.get("ok"), bool) else None
reset_ready_refresh_phase = str(reset_ready_refresh_projection.get("phase") or "").strip() or None
reset_ready_refresh_partial = (
    reset_ready_refresh_projection.get("partial_refresh")
    if isinstance(reset_ready_refresh_projection.get("partial_refresh"), dict)
    else {}
)
reset_ready_refresh_partial_current = (
    reset_ready_refresh_partial.get("current_refreshed")
    if isinstance(reset_ready_refresh_partial.get("current_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_proof = (
    reset_ready_refresh_partial.get("proof_refreshed")
    if isinstance(reset_ready_refresh_partial.get("proof_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_handover = (
    reset_ready_refresh_partial.get("handover_refreshed")
    if isinstance(reset_ready_refresh_partial.get("handover_refreshed"), bool)
    else None
)
reset_ready_refresh_partial_failure = bool(reset_ready_refresh_projection.get("partial_failure") is True)
reset_ready_refresh_error_code = str(reset_ready_refresh_projection.get("error_code") or "").strip() or None
reset_ready_refresh_degraded = bool(reset_ready_refresh_projection.get("degraded") is True)

for reason in proof_failclose_reasons:
    if reason not in not_ready_reasons:
        not_ready_reasons.append(reason)
gtc_now = now_obj.get("gtc") if isinstance(now_obj.get("gtc"), dict) else {}
gtc_warning_reasons = [str(x) for x in (gtc_now.get("warning_reasons") or []) if str(x).strip()]
gtc_handoff_binding_degraded = "queue_task_handoff_gate_binding_degraded" in gtc_warning_reasons
gtc_incident_replay_commands = [
    normalize_operator_command(cmd)
    for cmd in (gtc_now.get("incident_replay_commands") or [])
    if isinstance(cmd, str) and cmd.strip()
]
gtc_handoff_binding_next_safe_action = cmd_cont_gtc_replay_json
for cmd in gtc_incident_replay_commands:
    if "gtc-replay" in cmd or "gtc_incident_replay" in cmd:
        gtc_handoff_binding_next_safe_action = cmd
        break
if gtc_handoff_binding_next_safe_action == cmd_cont_gtc_replay_json and gtc_incident_replay_commands:
    gtc_handoff_binding_next_safe_action = gtc_incident_replay_commands[0]
gtc_evidence_refs = [
    str(path).strip()
    for path in [gtc_now.get("latest_path"), gtc_now.get("incident_replay_path")]
    if str(path or "").strip()
]
autopilot_idle_lane_status = str(autopilot_idle_lane_autospawn.get("status") or "missing")
autopilot_idle_lane_ready_work_exists = bool(autopilot_idle_lane_autospawn.get("ready_work_exists") is True)
autopilot_idle_lane_idle_threshold_exceeded = bool(autopilot_idle_lane_autospawn.get("idle_threshold_exceeded") is True)
autopilot_idle_lane_trace_path = str(autopilot_idle_lane_autospawn.get("trace_path") or "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json")
autopilot_idle_lane_skip_reason = str(autopilot_idle_lane_autospawn.get("skip_reason") or "")
autopilot_idle_lane_contract_source_degraded = bool(autopilot_idle_lane_autospawn.get("contract_source_degraded") is True)
autopilot_idle_lane_contradiction_abort_active = bool(autopilot_idle_lane_autospawn.get("contradiction_abort_active") is True)
autopilot_idle_lane_contradiction_abort_remaining_sec = int(autopilot_idle_lane_autospawn.get("contradiction_abort_remaining_sec") or 0)
autopilot_idle_lane_contradiction_latch_repaired = bool(autopilot_idle_lane_autospawn.get("contradiction_latch_repaired") is True)
autopilot_idle_lane_contradiction_latch_repair_reason = str(autopilot_idle_lane_autospawn.get("contradiction_latch_repair_reason") or "")
autopilot_idle_lane_contradiction_latched = bool(
    autopilot_idle_lane_ready_work_exists
    and autopilot_idle_lane_idle_threshold_exceeded
    and not autopilot_idle_lane_contract_source_degraded
    and (
        autopilot_idle_lane_skip_reason == "contradiction_latched_auto_abort"
        or autopilot_idle_lane_contradiction_abort_active
    )
)
autopilot_idle_lane_failure_like = autopilot_idle_lane_status in {"tick_failed", "attempted_no_launch", "error"}
autopilot_idle_lane_stalled = bool(
    autopilot_idle_lane_failure_like
    and autopilot_idle_lane_ready_work_exists
    and autopilot_idle_lane_idle_threshold_exceeded
)

autopilot_execution_frontier_controller = (
    autopilot_now.get("execution_frontier_controller")
    if isinstance(autopilot_now.get("execution_frontier_controller"), dict)
    else {}
)
autopilot_execution_frontier_controller_status = str(
    autopilot_execution_frontier_controller.get("status") or "missing"
)
autopilot_execution_frontier_controller_decision = str(
    autopilot_execution_frontier_controller.get("decision") or ""
).strip() or None
autopilot_execution_frontier_controller_skip_reason = str(
    autopilot_execution_frontier_controller.get("skip_reason") or ""
).strip() or None
autopilot_execution_frontier_controller_block_reason = str(
    autopilot_execution_frontier_controller.get("block_reason") or ""
).strip() or None
autopilot_execution_frontier_controller_block_reasons = []
_autopilot_execution_frontier_controller_block_reason_seen = set()
for _raw_reason in (autopilot_execution_frontier_controller.get("block_reasons") or []):
    _reason_txt = str(_raw_reason or "").strip()
    if not _reason_txt or _reason_txt in _autopilot_execution_frontier_controller_block_reason_seen:
        continue
    _autopilot_execution_frontier_controller_block_reason_seen.add(_reason_txt)
    autopilot_execution_frontier_controller_block_reasons.append(_reason_txt)
autopilot_execution_frontier_controller_error = str(
    autopilot_execution_frontier_controller.get("error") or ""
).strip() or None
autopilot_execution_frontier_controller_recorded_at = str(
    autopilot_execution_frontier_controller.get("recorded_at") or ""
).strip() or None
autopilot_execution_frontier_controller_trace_path = str(
    autopilot_execution_frontier_controller.get("trace_path")
    or "state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json"
)
autopilot_execution_frontier_controller_history_path = str(
    autopilot_execution_frontier_controller.get("history_path")
    or "state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl"
)
autopilot_execution_frontier_controller_source_degraded = bool(
    autopilot_execution_frontier_controller.get("contract_source_degraded") is True
)
autopilot_execution_frontier_controller_dispatch_decision = str(
    autopilot_execution_frontier_controller.get("dispatch_decision") or ""
).strip() or None
autopilot_execution_frontier_controller_dispatch_advance_applied = bool(
    autopilot_execution_frontier_controller.get("dispatch_advance_applied") is True
)
autopilot_execution_frontier_controller_selector_state = str(
    autopilot_execution_frontier_controller.get("selector_state") or ""
).strip() or None
autopilot_execution_frontier_controller_close_condition_met = autopilot_execution_frontier_controller.get("close_condition_met")
if not isinstance(autopilot_execution_frontier_controller_close_condition_met, bool):
    autopilot_execution_frontier_controller_close_condition_met = None
autopilot_execution_frontier_controller_post_completion_enforcement_required = (
    autopilot_execution_frontier_controller.get("post_completion_enforcement_required")
)
if not isinstance(autopilot_execution_frontier_controller_post_completion_enforcement_required, bool):
    autopilot_execution_frontier_controller_post_completion_enforcement_required = bool(
        autopilot_execution_frontier_controller_close_condition_met is True
        and (autopilot_execution_frontier_controller_selector_state or "")
        in {"ready_for_dispatch", "closed_blocked", "idle_no_candidate"}
    )
autopilot_execution_frontier_controller_loop_state = str(
    autopilot_execution_frontier_controller.get("post_completion_loop_state") or ""
).strip() or None
autopilot_execution_frontier_controller_retry_contract = (
    autopilot_execution_frontier_controller.get("post_completion_retry_contract")
    if isinstance(autopilot_execution_frontier_controller.get("post_completion_retry_contract"), dict)
    else {}
)
autopilot_execution_frontier_controller_cooldown_policy = (
    autopilot_execution_frontier_controller.get("post_completion_cooldown_policy")
    if isinstance(autopilot_execution_frontier_controller.get("post_completion_cooldown_policy"), dict)
    else {}
)
autopilot_execution_frontier_controller_parity = (
    autopilot_execution_frontier_controller.get("queue_truth_vs_narrative_parity")
    if isinstance(autopilot_execution_frontier_controller.get("queue_truth_vs_narrative_parity"), dict)
    else {}
)
autopilot_execution_frontier_controller_blocked = bool(
    autopilot_execution_frontier_controller_status == "blocked"
)
autopilot_execution_frontier_controller_error_state = bool(
    autopilot_execution_frontier_controller_status == "error"
)
autopilot_execution_frontier_controller_applied = bool(
    autopilot_execution_frontier_controller_status == "applied"
)
autopilot_execution_frontier_controller_post_completion_blocked = bool(
    autopilot_execution_frontier_controller_post_completion_enforcement_required
    and autopilot_execution_frontier_controller_status == "blocked"
)
autopilot_execution_frontier_controller_post_completion_stalled = bool(
    autopilot_execution_frontier_controller_post_completion_enforcement_required
    and autopilot_execution_frontier_controller_status in {"error", "missing", "skipped"}
)
if (
    autopilot_execution_frontier_controller_post_completion_blocked
    or autopilot_execution_frontier_controller_post_completion_stalled
):
    if "execution_frontier_post_completion_enforcement_latched" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_enforcement_latched")
if autopilot_execution_frontier_controller_post_completion_blocked:
    if "execution_frontier_post_completion_enforcement_blocked" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_enforcement_blocked")
elif autopilot_execution_frontier_controller_post_completion_stalled:
    if "execution_frontier_post_completion_enforcement_stalled" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_enforcement_stalled")

if autopilot_execution_frontier_controller_loop_state == "STALLED_LOOP":
    if "execution_frontier_post_completion_stalled_loop" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_stalled_loop")
elif autopilot_execution_frontier_controller_loop_state == "BLOCKED_LOOP":
    if "execution_frontier_post_completion_blocked_loop" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_blocked_loop")

if bool(autopilot_execution_frontier_controller_cooldown_policy.get("active") is True):
    if "execution_frontier_post_completion_cooldown_active" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_cooldown_active")

if str(autopilot_execution_frontier_controller_retry_contract.get("state") or "") == "retry_exhausted":
    if "execution_frontier_post_completion_retry_exhausted" not in warning_reasons:
        warning_reasons.append("execution_frontier_post_completion_retry_exhausted")

if str(autopilot_execution_frontier_controller_parity.get("status") or "") == "mismatch":
    if "execution_frontier_queue_truth_vs_narrative_mismatch" not in warning_reasons:
        warning_reasons.append("execution_frontier_queue_truth_vs_narrative_mismatch")

load_shedding_projection = current.get("load_shedding") if isinstance(current.get("load_shedding"), dict) else {}
if not load_shedding_projection:
    load_shedding_decision_obj = load_json_if_exists(load_shedding_decision_path)
    load_shedding_signal_obj = load_json_if_exists(load_shedding_signal_snapshot_path)
    decision_artifact_present = load_shedding_decision_path.exists()
    signal_artifact_present = load_shedding_signal_snapshot_path.exists()
    artifact_family_present = decision_artifact_present or signal_artifact_present
    lane_tier = str((((load_shedding_decision_obj or {}).get("lane_tier") or (load_shedding_signal_obj or {}).get("derived_tier") or "UNKNOWN")).strip() or "UNKNOWN").upper()
    trigger_emitted = str(((load_shedding_decision_obj or {}).get("trigger_emitted") or "").strip()) or None
    load_shedding_projection = {
        "lane_health_state": lane_tier,
        "warning_tier": lane_tier in {"WARNING", "CRITICAL"},
        "critical_tier": lane_tier == "CRITICAL",
        "escape_triggered": trigger_emitted in {"TR_CRITICAL_THRESHOLD_REACHED", "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL"},
        "trigger_emitted": trigger_emitted,
        "thin_mode": bool(((load_shedding_decision_obj or {}).get("thin_mode") is True)),
        "evaluated_at": (load_shedding_decision_obj or {}).get("evaluated_at") or (load_shedding_signal_obj or {}).get("evaluated_at"),
        "contract_source_degraded": bool(
            artifact_family_present
            and (not isinstance(load_shedding_decision_obj, dict) or (signal_artifact_present and not isinstance(load_shedding_signal_obj, dict)))
        ),
        "contract_source_degraded_reason": (
            "load_shedding_decision_missing"
            if artifact_family_present and not isinstance(load_shedding_decision_obj, dict)
            else ("load_shedding_signal_snapshot_missing" if artifact_family_present and signal_artifact_present and not isinstance(load_shedding_signal_obj, dict) else None)
        ),
        "decision_path": str(load_shedding_decision_path.relative_to(root)),
        "signal_snapshot_path": str(load_shedding_signal_snapshot_path.relative_to(root)),
    }

load_shedding_lane_state = str(load_shedding_projection.get("lane_health_state") or "UNKNOWN").upper()
load_shedding_warning_tier = bool(load_shedding_projection.get("warning_tier") is True)
load_shedding_critical_tier = bool(load_shedding_projection.get("critical_tier") is True)
load_shedding_escape_triggered = bool(load_shedding_projection.get("escape_triggered") is True)
load_shedding_trigger_emitted = str(load_shedding_projection.get("trigger_emitted") or "").strip() or None
load_shedding_source_degraded = bool(load_shedding_projection.get("contract_source_degraded") is True)

WAVE2_REPLAY_EVIDENCE_MAX_AGE_SEC = 6 * 60 * 60
wave2_replay_evidence_obj = load_json_if_exists(wave2_replay_evidence_index_path)
wave2_replay_evidence_present = wave2_replay_evidence_index_path.exists()
wave2_replay_evidence_source_degraded = bool(
    wave2_replay_evidence_present and not isinstance(wave2_replay_evidence_obj, dict)
)
wave2_replay_summary = (
    wave2_replay_evidence_obj.get("summary")
    if isinstance(wave2_replay_evidence_obj, dict) and isinstance(wave2_replay_evidence_obj.get("summary"), dict)
    else {}
)
wave2_replay_verdict = str(wave2_replay_summary.get("overall_verdict") or "MISSING").upper()
wave2_replay_generated_at = (
    wave2_replay_evidence_obj.get("generated_at")
    if isinstance(wave2_replay_evidence_obj, dict)
    else None
)
wave2_replay_age_sec = age_sec(wave2_replay_generated_at)
wave2_replay_fresh = bool(
    wave2_replay_age_sec is not None and wave2_replay_age_sec <= WAVE2_REPLAY_EVIDENCE_MAX_AGE_SEC
)
wave2_replay_scenario_count = int(wave2_replay_summary.get("scenario_count") or 0)
wave2_replay_soak_runs_total = int(wave2_replay_summary.get("soak_runs_total") or 0)
wave2_replay_soak_drift_detected = bool(wave2_replay_summary.get("soak_drift_detected") is True)
wave2_replay_ready = bool(
    isinstance(wave2_replay_evidence_obj, dict)
    and wave2_replay_verdict == "PASS"
    and wave2_replay_fresh
    and not wave2_replay_soak_drift_detected
)

FAILOVER_STRESS_SOAK_MAX_AGE_SEC = 6 * 60 * 60
failover_stress_soak_obj = load_json_if_exists(failover_stress_soak_evidence_path)
failover_stress_soak_present = failover_stress_soak_evidence_path.exists()
failover_stress_soak_source_degraded = bool(
    failover_stress_soak_present and not isinstance(failover_stress_soak_obj, dict)
)
failover_stress_soak_summary = (
    failover_stress_soak_obj.get("summary")
    if isinstance(failover_stress_soak_obj, dict) and isinstance(failover_stress_soak_obj.get("summary"), dict)
    else {}
)
failover_stress_soak_determinism = (
    failover_stress_soak_obj.get("determinism")
    if isinstance(failover_stress_soak_obj, dict) and isinstance(failover_stress_soak_obj.get("determinism"), dict)
    else {}
)
failover_stress_soak_verdict = str(failover_stress_soak_summary.get("overall_verdict") or "MISSING").upper()
failover_stress_soak_generated_at = (
    failover_stress_soak_obj.get("generated_at")
    if isinstance(failover_stress_soak_obj, dict)
    else None
)
failover_stress_soak_age_sec = age_sec(failover_stress_soak_generated_at)
failover_stress_soak_fresh = bool(
    failover_stress_soak_age_sec is not None and failover_stress_soak_age_sec <= FAILOVER_STRESS_SOAK_MAX_AGE_SEC
)
failover_stress_soak_profile_count = int(failover_stress_soak_summary.get("profile_count") or 0)
failover_stress_soak_total_cycles = int(failover_stress_soak_summary.get("total_cycles") or 0)
failover_stress_soak_convergence_fail_count = int(failover_stress_soak_summary.get("convergence_fail_count") or 0)
failover_stress_soak_drift_detected = bool(
    failover_stress_soak_summary.get("stress_drift_detected") is True
    or failover_stress_soak_determinism.get("drift_detected") is True
)
failover_stress_soak_ready = bool(
    isinstance(failover_stress_soak_obj, dict)
    and failover_stress_soak_verdict == "PASS"
    and failover_stress_soak_fresh
    and failover_stress_soak_convergence_fail_count == 0
    and not failover_stress_soak_drift_detected
)

FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC = 6 * 60 * 60
failover_stress_runtime_obj = load_json_if_exists(failover_stress_runtime_evidence_path)
failover_stress_runtime_present = failover_stress_runtime_evidence_path.exists()
failover_stress_runtime_source_degraded = bool(
    failover_stress_runtime_present and not isinstance(failover_stress_runtime_obj, dict)
)
failover_stress_runtime_summary = (
    failover_stress_runtime_obj.get("summary")
    if isinstance(failover_stress_runtime_obj, dict) and isinstance(failover_stress_runtime_obj.get("summary"), dict)
    else {}
)
failover_stress_runtime_repeatability = (
    failover_stress_runtime_summary.get("repeatability")
    if isinstance(failover_stress_runtime_summary.get("repeatability"), dict)
    else {}
)
failover_stress_runtime_verdict = str(failover_stress_runtime_summary.get("overall_verdict") or "MISSING").upper()
failover_stress_runtime_publish_verdict = str(
    failover_stress_runtime_summary.get("publish_chain_verdict") or "MISSING"
).upper()
failover_stress_runtime_publish_assertions_failed = int(
    failover_stress_runtime_summary.get("publish_assertions_failed") or 0
)
failover_stress_runtime_active_top_blocker = (
    str(failover_stress_runtime_summary.get("active_top_blocker") or "").strip() or None
)
failover_stress_runtime_effective_top_blocker = (
    str(failover_stress_runtime_summary.get("effective_top_blocker") or "").strip() or None
)
failover_stress_runtime_generated_at = (
    failover_stress_runtime_obj.get("generated_at")
    if isinstance(failover_stress_runtime_obj, dict)
    else None
)
failover_stress_runtime_age_sec = age_sec(failover_stress_runtime_generated_at)
failover_stress_runtime_fresh = bool(
    failover_stress_runtime_age_sec is not None and failover_stress_runtime_age_sec <= FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC
)
failover_stress_runtime_repeatability_status = str(
    failover_stress_runtime_repeatability.get("status") or "missing"
).strip().lower()
failover_stress_runtime_repeatability_match = failover_stress_runtime_repeatability.get("match")
failover_stress_runtime_repeatability_mismatch_fields = [
    str(row).strip()
    for row in (failover_stress_runtime_repeatability.get("mismatch_fields") or [])
    if str(row).strip()
]
failover_stress_runtime_decision_log_ref = ""
failover_stress_runtime_decision_log_path = None
if isinstance(failover_stress_runtime_obj, dict):
    runtime_artifacts = (
        failover_stress_runtime_obj.get("artifacts")
        if isinstance(failover_stress_runtime_obj.get("artifacts"), dict)
        else {}
    )
    failover_stress_runtime_decision_log_ref = str(runtime_artifacts.get("decision_log_ref") or "").strip()
if failover_stress_runtime_decision_log_ref:
    candidate_path = (root / failover_stress_runtime_decision_log_ref).resolve()
    try:
        candidate_path.relative_to(root)
        failover_stress_runtime_decision_log_path = candidate_path
    except Exception:
        failover_stress_runtime_decision_log_path = None
failover_stress_runtime_decision_log_exists = bool(
    isinstance(failover_stress_runtime_decision_log_path, pathlib.Path)
    and failover_stress_runtime_decision_log_path.exists()
)
cmd_read_failover_stress_runtime_decision_log = cmd_read_failover_stress_runtime_evidence_json
if failover_stress_runtime_decision_log_exists and isinstance(failover_stress_runtime_decision_log_path, pathlib.Path):
    failover_stress_runtime_decision_log_rel = str(failover_stress_runtime_decision_log_path.relative_to(root))
    cmd_read_failover_stress_runtime_decision_log = shell_cmd_for("tail", "-n", "80", failover_stress_runtime_decision_log_rel)
failover_stress_runtime_ready = bool(
    isinstance(failover_stress_runtime_obj, dict)
    and failover_stress_runtime_verdict == "PASS"
    and failover_stress_runtime_publish_verdict == "PASS"
    and failover_stress_runtime_fresh
    and failover_stress_runtime_publish_assertions_failed == 0
    and failover_stress_runtime_repeatability_status != "mismatch"
)

# Browser artifact latest signal.
browser_latest = {
    "status": "unknown",
    "run_id": None,
    "gate_class": None,
    "captured_at": None,
}
latest_run_path = root / "ops" / "web_capture" / "artifacts" / "latest_run.json"
if latest_run_path.exists():
    try:
        latest_meta = json.loads(latest_run_path.read_text(encoding="utf-8"))
        run_id = str(latest_meta.get("run_id") or "")
        browser_latest["run_id"] = run_id or None
        if run_id:
            idx = root / "ops" / "web_capture" / "artifacts" / run_id / "index.json"
            if idx.exists():
                idx_obj = json.loads(idx.read_text(encoding="utf-8"))
                browser_latest["status"] = idx_obj.get("status") or "unknown"
                browser_latest["gate_class"] = ((idx_obj.get("gate_classification") or {}).get("class") if isinstance(idx_obj.get("gate_classification"), dict) else None)
                browser_latest["captured_at"] = idx_obj.get("captured_at")
    except Exception:
        pass

web_domain_guard = {
    "tracked_domains": 0,
    "blocked_domains": 0,
    "cooldown_active_domains": 0,
    "operator_action_required_domains": 0,
    "actionable_incident_domains": 0,
    "domains": [],
    "latest_updated_at": None,
}


def load_web_login_actionability(contract_ref: Any) -> Dict[str, Any]:
    rel = str(contract_ref or "").strip()
    if not rel:
        return {}
    path = pathlib.Path(rel)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    incident = payload.get("incident_actionability") if isinstance(payload.get("incident_actionability"), dict) else {}
    commands = [
        normalize_operator_command(cmd)
        for cmd in (incident.get("recommended_commands") if isinstance(incident.get("recommended_commands"), list) else [])
        if isinstance(cmd, str) and cmd.strip()
    ]
    recommended_steps = []
    for step in (incident.get("recommended_steps") if isinstance(incident.get("recommended_steps"), list) else []):
        if not isinstance(step, dict):
            continue
        cmd = normalize_operator_command(step.get("command") or "")
        if not cmd:
            continue
        step_id = str(step.get("step_id") or step.get("id") or "").strip() or f"step_{len(recommended_steps) + 1}"
        summary = str(step.get("summary") or step.get("intent") or "").strip() or None
        recommended_steps.append(
            {
                "step_id": step_id,
                "summary": summary,
                "command": cmd,
            }
        )
    if not commands and recommended_steps:
        commands = [str(step.get("command") or "") for step in recommended_steps if str(step.get("command") or "").strip()]

    return {
        "contract_status": payload.get("status"),
        "operator_resume_command": normalize_operator_command(payload.get("resume_command") or "") or None,
        "incident_actionability": {
            "incident_id": incident.get("incident_id"),
            "reason": incident.get("reason"),
            "severity": incident.get("severity"),
            "status": incident.get("status"),
            "action_required": bool(incident.get("action_required")),
            "recommended_commands": commands,
            "recommended_steps": recommended_steps,
            "evidence": incident.get("evidence") if isinstance(incident.get("evidence"), list) else [],
        }
        if incident
        else {},
    }


web_guard_paths: List[str] = []
for p in sorted((root / "state" / "continuity" / "latest").glob("web_capture_domain_*.json")):
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue

    dom = str(obj.get("domain") or p.stem.replace("web_capture_domain_", "")).strip()
    status = str(obj.get("last_status") or "unknown").strip()
    gate_class = str(obj.get("last_gate_class") or "").strip() or None
    cooldown_until = obj.get("cooldown_until")
    cooldown_dt = parse_iso(cooldown_until)
    cooldown_remaining = None
    if cooldown_dt is not None:
        cooldown_remaining = max(0, int((cooldown_dt - clock_now_dt()).total_seconds()))

    operator_required = bool(obj.get("operator_action_required"))
    contract_json = str(obj.get("operator_contract_json") or "").strip() or None
    contract_md = str(obj.get("operator_contract_md") or "").strip() or None
    contract_actionability = load_web_login_actionability(contract_json) if operator_required and contract_json else {}
    incident_actionability = contract_actionability.get("incident_actionability") if isinstance(contract_actionability.get("incident_actionability"), dict) else {}
    incident_actionable = bool(operator_required and incident_actionability.get("action_required") and (incident_actionability.get("recommended_commands") or []))

    if status in {"blocked", "failed"}:
        web_domain_guard["blocked_domains"] += 1
    if cooldown_remaining is not None and cooldown_remaining > 0:
        web_domain_guard["cooldown_active_domains"] += 1
    if operator_required:
        web_domain_guard["operator_action_required_domains"] += 1
    if incident_actionable:
        web_domain_guard["actionable_incident_domains"] += 1

    updated_at = str(obj.get("updated_at") or "").strip() or None
    if updated_at:
        if web_domain_guard["latest_updated_at"] is None or str(updated_at) > str(web_domain_guard["latest_updated_at"]):
            web_domain_guard["latest_updated_at"] = updated_at

    entry = {
        "domain": dom,
        "macro_slug": obj.get("macro_slug"),
        "status": status,
        "gate_class": gate_class,
        "cooldown_until": cooldown_until,
        "cooldown_remaining_sec": cooldown_remaining,
        "operator_action_required": operator_required,
        "operator_contract_json": contract_json,
        "operator_contract_md": contract_md,
        "operator_resume_command": contract_actionability.get("operator_resume_command"),
        "incident_actionability": incident_actionability,
        "state_path": str(p.relative_to(root)) if p.is_absolute() else str(p),
        "updated_at": updated_at,
    }
    web_domain_guard["domains"].append(entry)
    web_guard_paths.append(entry["state_path"])
    if contract_json:
        web_guard_paths.append(contract_json)
    if contract_md:
        web_guard_paths.append(contract_md)

web_domain_guard["tracked_domains"] = len(web_domain_guard["domains"])
web_domain_guard["domains"] = sorted(
    web_domain_guard["domains"],
    key=lambda x: (
        0 if x.get("operator_action_required") else 1,
        0 if x.get("status") in {"blocked", "failed"} else 1,
        str(x.get("domain") or ""),
    ),
)

web_scheduler = {
    "state_path": "state/continuity/latest/web_capture_scheduler_state.json",
    "selection_status": None,
    "updated_at": None,
    "state_age_sec": None,
    "freshness_limit_sec": None,
    "fresh": None,
    "state_exists": False,
    "schema_version": None,
    "contract_state_valid": None,
    "contract_validation_errors": [],
    "eligible_macros": None,
    "total_macros": None,
    "last_selected_domain": None,
    "last_selected_macro_slug": None,
}
scheduler_freshness_limit_sec = _read_nonnegative_int_env(
    "OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC",
    default=_DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
)

web_scheduler_path = root / "state" / "continuity" / "latest" / "web_capture_scheduler_state.json"
web_scheduler["state_exists"] = web_scheduler_path.exists()
web_scheduler["freshness_limit_sec"] = scheduler_freshness_limit_sec
if web_scheduler_path.exists():
    try:
        ws = json.loads(web_scheduler_path.read_text(encoding="utf-8"))
        if isinstance(ws, dict):
            summary_ws = ws.get("summary") if isinstance(ws.get("summary"), dict) else {}
            contract_ws = ws.get("contract") if isinstance(ws.get("contract"), dict) else {}
            contract_errors = contract_ws.get("validation_errors") if isinstance(contract_ws.get("validation_errors"), list) else []
            updated_at = ws.get("updated_at")
            updated_age_sec = age_sec(updated_at)
            web_scheduler.update(
                {
                    "schema_version": ws.get("schema_version"),
                    "selection_status": ws.get("selection_status"),
                    "updated_at": updated_at,
                    "state_age_sec": updated_age_sec,
                    "fresh": updated_age_sec is not None and updated_age_sec <= scheduler_freshness_limit_sec,
                    "contract_state_valid": contract_ws.get("state_valid"),
                    "contract_validation_errors": contract_errors,
                    "eligible_macros": summary_ws.get("eligible_macros"),
                    "total_macros": summary_ws.get("total_macros"),
                    "last_selected_domain": ws.get("last_selected_domain"),
                    "last_selected_macro_slug": ws.get("last_selected_macro_slug"),
                }
            )
        else:
            web_scheduler.update(
                {
                    "contract_state_valid": False,
                    "contract_validation_errors": ["state_not_object"],
                }
            )
    except Exception:
        web_scheduler.update(
            {
                "contract_state_valid": False,
                "contract_validation_errors": ["state_unreadable_or_not_object"],
            }
        )

lock_break_latest = {
    "audit_id": None,
    "generated_at": None,
    "released_lock_count": 0,
    "requeued_task_count": 0,
}
lock_break_last_path = root / "state" / "continuity" / "latest" / "lock_break_last.json"
if lock_break_last_path.exists():
    try:
        lb = json.loads(lock_break_last_path.read_text(encoding="utf-8"))
        lock_break_latest = {
            "audit_id": lb.get("audit_id"),
            "generated_at": lb.get("generated_at"),
            "released_lock_count": int(lb.get("released_lock_count") or 0),
            "requeued_task_count": int(lb.get("requeued_task_count") or 0),
        }
    except Exception:
        pass

# Queue contention summary from DB for mission-control headline.
queue_db_path = root / "state" / "continuity" / "continuity_os.sqlite"
contention = {
    "active_locks": int(queue.get("active_file_lock_count") or 0),
    "stale_active_locks": int(queue.get("stale_active_file_lock_count") or 0),
    "running_tasks": int((queue.get("status_counts") or {}).get("RUNNING") or 0),
    "ready_tasks": int(queue.get("ready_count") or 0),
    "dependency_blocked_tasks": int(queue.get("dependency_blocked_count") or 0),
    "in_flight_effective": bool(queue.get("in_flight_effective") is True),
}
if queue_db_path.exists():
    try:
        con = sqlite3.connect(queue_db_path)
        cur = con.cursor()
        contention["overdue_locks"] = int(
            cur.execute(
                "SELECT COUNT(1) FROM file_locks WHERE lock_state='ACTIVE' AND lock_expires_at IS NOT NULL AND lock_expires_at <= ?",
                (clock_now_iso(),),
            ).fetchone()[0]
            or 0
        )
        con.close()
    except Exception:
        contention["overdue_locks"] = None


def _reason_list(raw: Any) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def _contract_source_degraded(raw: Any) -> bool:
    return isinstance(raw, dict) and bool(raw.get("contract_source_degraded") is True)


def _truth_row_with_freshness_posture(row: Dict[str, Any], *, stale_reason: Optional[str] = None) -> Dict[str, Any]:
    out = dict(row)
    freshness_value = out.get("freshness")
    parsed = parse_iso(freshness_value)
    posture = "fresh" if parsed is not None else "unknown"
    if stale_reason:
        posture = "stale"
    out["freshness_posture"] = posture
    out["freshness_reason"] = stale_reason if posture == "stale" else None
    out["freshness_age_sec"] = age_sec(freshness_value)
    return out


live_current_artifact = load_live_current_artifact()
current_source_sha, current_source_sha_origin = resolve_current_snapshot_sha(current, live_current_artifact)

generation_pointer = evaluate_generation_pointer_contract(
    current_obj=current,
    now_obj=now_obj,
    current_sha=current_source_sha,
)
generation_pointer["current_sha_source"] = current_source_sha_origin
generation_pointer["live_current_observed"] = {
    "path": live_current_artifact.get("path"),
    "present": bool(live_current_artifact.get("present")),
    "sha256": live_current_artifact.get("sha256"),
    "generated_at": live_current_artifact.get("generated_at"),
    "generation_id": live_current_artifact.get("generation_id"),
    "read_error": live_current_artifact.get("read_error"),
}
generation_pointer = maybe_apply_generation_pointer_read_phase_pin(
    generation_pointer=generation_pointer,
    current_obj=current,
    now_obj=now_obj,
    live_current_artifact=live_current_artifact,
    continuity_now_contract_failclose_reasons=continuity_now_contract_failclose_reasons,
)
generation_pointer["continuity_now_contract"] = continuity_now_contract
generation_failclose_reasons = unique_preserve(
    _reason_list(generation_pointer.get("failclose_reasons"))
    + _reason_list(continuity_now_contract_failclose_reasons)
)
generation_pointer["failclose_reasons"] = generation_failclose_reasons
for reason in generation_failclose_reasons:
    if reason not in not_ready_reasons:
        not_ready_reasons.append(reason)

freshness_failclose_reasons = unique_preserve(
    [
        "continuity_current_source_degraded" if _contract_source_degraded(current) else "",
        "continuity_now_source_degraded" if _contract_source_degraded(now_obj) else "",
        "handover_source_degraded" if _contract_source_degraded(handover) else "",
        "gate_os_source_degraded" if _contract_source_degraded(gate_os) else "",
        "queue_replay_source_degraded" if _contract_source_degraded(queue_replay) else "",
    ]
)
for reason in freshness_failclose_reasons:
    if reason not in not_ready_reasons:
        not_ready_reasons.append(reason)

freshness_warning_reasons = unique_preserve(
    [
        "handover_stale" if bool(handover.get("stale") is True) else "",
        "parity_weekly_freshness_due" if bool(parity.get("due") is True) else "",
        "web_scheduler_stale" if web_scheduler.get("fresh") is False else "",
        "wave2_replay_evidence_nonpass" if isinstance(wave2_replay_evidence_obj, dict) and wave2_replay_verdict != "PASS" else "",
        "wave2_replay_evidence_stale" if isinstance(wave2_replay_evidence_obj, dict) and not wave2_replay_fresh else "",
        "wave2_replay_evidence_soak_drift" if wave2_replay_soak_drift_detected else "",
        "failover_stress_soak_nonpass" if isinstance(failover_stress_soak_obj, dict) and failover_stress_soak_verdict != "PASS" else "",
        "failover_stress_soak_stale" if isinstance(failover_stress_soak_obj, dict) and not failover_stress_soak_fresh else "",
        "failover_stress_soak_drift" if failover_stress_soak_drift_detected else "",
        "failover_stress_runtime_nonpass" if isinstance(failover_stress_runtime_obj, dict) and failover_stress_runtime_verdict != "PASS" else "",
        "failover_stress_runtime_publish_nonpass" if isinstance(failover_stress_runtime_obj, dict) and failover_stress_runtime_publish_verdict != "PASS" else "",
        "failover_stress_runtime_stale" if isinstance(failover_stress_runtime_obj, dict) and not failover_stress_runtime_fresh else "",
        "failover_stress_runtime_repeatability_mismatch" if failover_stress_runtime_repeatability_status == "mismatch" else "",
    ]
)

for reason in [
    "failover_stress_runtime_nonpass" if isinstance(failover_stress_runtime_obj, dict) and failover_stress_runtime_verdict != "PASS" else "",
    "failover_stress_runtime_publish_nonpass" if isinstance(failover_stress_runtime_obj, dict) and failover_stress_runtime_publish_verdict != "PASS" else "",
    "failover_stress_runtime_repeatability_mismatch" if failover_stress_runtime_repeatability_status == "mismatch" else "",
]:
    reason = str(reason).strip()
    if reason and reason not in not_ready_reasons:
        not_ready_reasons.append(reason)

current_mutation_gate = current.get("mutation_gate") if isinstance(current.get("mutation_gate"), dict) else {}
projection_mutation_gate = now_obj.get("mutation_gate_projection") if isinstance(now_obj.get("mutation_gate_projection"), dict) else {}
mutation_gate_status = str(current_mutation_gate.get("status") or projection_mutation_gate.get("status") or "unknown")
mutation_gate_posture = str(current_mutation_gate.get("posture") or projection_mutation_gate.get("posture") or "unknown")
mutation_gate_expected_in_flight_guard = current_mutation_gate.get("expected_in_flight_guard")
if not isinstance(mutation_gate_expected_in_flight_guard, bool):
    mutation_gate_expected_in_flight_guard = projection_mutation_gate.get("expected_in_flight_guard")
mutation_gate_blocking_reasons = _reason_list(current_mutation_gate.get("blocking_reasons") or projection_mutation_gate.get("blocking_reasons"))
mutation_gate_concurrency_reasons = _reason_list(current_mutation_gate.get("concurrency_reasons") or projection_mutation_gate.get("concurrency_reasons"))

if generation_failclose_reasons or freshness_failclose_reasons or proof_failclose_reasons:
    mutation_gate_status = "forbidden"
    mutation_gate_posture = "blocker"
    for reason in unique_preserve(generation_failclose_reasons + freshness_failclose_reasons + proof_failclose_reasons):
        if reason not in mutation_gate_blocking_reasons:
            mutation_gate_blocking_reasons.append(reason)

effective_readiness = str(current.get("readiness") or "UNKNOWN")
if generation_failclose_reasons or freshness_failclose_reasons or proof_failclose_reasons:
    effective_readiness = "RECONCILE_REQUIRED"

current_publish_lock_timeout_detected = bool(current_publish_lock_timeout.get("detected") is True)
current_publish_lock_timeout_owner_age_sec = current_publish_lock_timeout.get("owner_age_sec")
if not isinstance(current_publish_lock_timeout_owner_age_sec, int):
    current_publish_lock_timeout_owner_age_sec = None
current_publish_lock_timeout_wait_sec = current_publish_lock_timeout.get("wait_sec")
if not isinstance(current_publish_lock_timeout_wait_sec, (int, float)):
    current_publish_lock_timeout_wait_sec = None
current_publish_lock_timeout_hold_exceeded = current_publish_lock_timeout.get("owner_exceeds_lock_hold_warn")
if not isinstance(current_publish_lock_timeout_hold_exceeded, bool):
    current_publish_lock_timeout_hold_exceeded = None

continuity_current_publish_lock: Dict[str, Any] = {
    "source": None,
    "surface_active": False,
    "status": None,
    "path": None,
    "generated_at": None,
    "timeout_detected": current_publish_lock_timeout_detected,
    "wait_sec": current_publish_lock_timeout_wait_sec,
    "owner_pid": (
        current_publish_lock_timeout.get("owner_pid")
        if isinstance(current_publish_lock_timeout.get("owner_pid"), int)
        else None
    ),
    "owner_alive": (
        current_publish_lock_timeout.get("owner_alive")
        if isinstance(current_publish_lock_timeout.get("owner_alive"), bool)
        else None
    ),
    "owner_age_sec": current_publish_lock_timeout_owner_age_sec,
    "lock_hold_warn_sec": (
        current_publish_lock_timeout.get("lock_hold_warn_sec")
        if isinstance(current_publish_lock_timeout.get("lock_hold_warn_sec"), (int, float))
        else None
    ),
    "owner_exceeds_wait_budget": None,
    "owner_exceeds_lock_hold_warn": current_publish_lock_timeout_hold_exceeded,
    "owner_host": current_publish_lock_timeout.get("owner_host"),
    "owner_command": current_publish_lock_timeout.get("owner_command"),
    "recommended_action": None,
    "inspect_command": cmd_read_current_publish_lock_owner_json,
    "action_required": False,
    "warning_reasons": [],
    "source_current_path": None,
    "source_current_generated_at": None,
    "source_current_age_sec": None,
    "source_current_max_age_sec": None,
    "source_current_fresh": None,
    "source_current_matches_current_generated_at": None,
    "source_degraded": False,
    "source_degraded_reasons": [],
}

if publish_lock_registry_signal.get("surface_active") is True:
    continuity_current_publish_lock = {
        "source": "blocker_registry",
        "surface_active": True,
        "status": publish_lock_registry_signal.get("status") or "active",
        "path": publish_lock_registry_signal.get("path") or current_publish_lock_timeout.get("path"),
        "generated_at": publish_lock_registry_signal.get("generated_at"),
        "timeout_detected": current_publish_lock_timeout_detected,
        "wait_sec": (
            publish_lock_registry_signal.get("lock_wait_sec")
            if isinstance(publish_lock_registry_signal.get("lock_wait_sec"), (int, float))
            else current_publish_lock_timeout_wait_sec
        ),
        "owner_pid": (
            publish_lock_registry_signal.get("owner_pid")
            if isinstance(publish_lock_registry_signal.get("owner_pid"), int)
            else continuity_current_publish_lock.get("owner_pid")
        ),
        "owner_alive": (
            publish_lock_registry_signal.get("owner_alive")
            if isinstance(publish_lock_registry_signal.get("owner_alive"), bool)
            else continuity_current_publish_lock.get("owner_alive")
        ),
        "owner_age_sec": (
            publish_lock_registry_signal.get("owner_age_sec")
            if isinstance(publish_lock_registry_signal.get("owner_age_sec"), int)
            else continuity_current_publish_lock.get("owner_age_sec")
        ),
        "lock_hold_warn_sec": (
            publish_lock_registry_signal.get("lock_hold_warn_sec")
            if isinstance(publish_lock_registry_signal.get("lock_hold_warn_sec"), (int, float))
            else continuity_current_publish_lock.get("lock_hold_warn_sec")
        ),
        "owner_exceeds_wait_budget": (
            publish_lock_registry_signal.get("owner_exceeds_wait_budget")
            if isinstance(publish_lock_registry_signal.get("owner_exceeds_wait_budget"), bool)
            else None
        ),
        "owner_exceeds_lock_hold_warn": (
            publish_lock_registry_signal.get("owner_exceeds_lock_hold_warn")
            if isinstance(publish_lock_registry_signal.get("owner_exceeds_lock_hold_warn"), bool)
            else continuity_current_publish_lock.get("owner_exceeds_lock_hold_warn")
        ),
        "owner_host": publish_lock_registry_signal.get("owner_host") or continuity_current_publish_lock.get("owner_host"),
        "owner_command": publish_lock_registry_signal.get("owner_command") or continuity_current_publish_lock.get("owner_command"),
        "recommended_action": publish_lock_registry_signal.get("recommended_action") or "inspect_current_publish_lock_owner",
        "inspect_command": publish_lock_registry_signal.get("inspect_command") or cmd_read_current_publish_lock_owner_json,
        "action_required": publish_lock_registry_signal.get("action_required") is True,
        "warning_reasons": _reason_list(publish_lock_registry_signal.get("warning_reasons")),
        "source_current_path": publish_lock_registry_signal.get("source_current_path"),
        "source_current_generated_at": publish_lock_registry_signal.get("source_current_generated_at"),
        "source_current_age_sec": (
            publish_lock_registry_signal.get("source_current_age_sec")
            if isinstance(publish_lock_registry_signal.get("source_current_age_sec"), (int, float))
            else None
        ),
        "source_current_max_age_sec": (
            publish_lock_registry_signal.get("source_current_max_age_sec")
            if isinstance(publish_lock_registry_signal.get("source_current_max_age_sec"), (int, float))
            else None
        ),
        "source_current_fresh": (
            publish_lock_registry_signal.get("source_current_fresh")
            if isinstance(publish_lock_registry_signal.get("source_current_fresh"), bool)
            else None
        ),
        "source_current_matches_current_generated_at": (
            publish_lock_registry_signal.get("source_current_matches_current_generated_at")
            if isinstance(publish_lock_registry_signal.get("source_current_matches_current_generated_at"), bool)
            else None
        ),
        "source_degraded": publish_lock_registry_signal.get("source_degraded") is True,
        "source_degraded_reasons": _reason_list(publish_lock_registry_signal.get("source_degraded_reasons")),
    }
elif current_publish_lock_timeout_detected:
    continuity_current_publish_lock.update(
        {
            "source": "continuity_current_timeout_fallback",
            "surface_active": True,
            "status": "timeout_detected",
            "path": current_publish_lock_timeout.get("path") or current_publish_lock_owner_rel,
            "generated_at": current.get("generated_at"),
            "recommended_action": "inspect_current_publish_lock_owner",
            "action_required": True,
        }
    )

publish_lock_warning_reasons = _reason_list(continuity_current_publish_lock.get("warning_reasons"))

execution_dispatch_context = normalize_dispatch_context(current.get("dispatch_context"))
if not execution_dispatch_context:
    execution_dispatch_context = build_dispatch_context_from_idle_lane(autopilot_idle_lane_autospawn)

if execution_dispatch_context:
    if str(execution_dispatch_context.get("autonomous_dispatch_status") or "missing") == "missing":
        execution_dispatch_context["autonomous_dispatch_status"] = autopilot_execution_frontier_controller_status
    if execution_dispatch_context.get("autonomous_dispatch_decision") is None:
        execution_dispatch_context["autonomous_dispatch_decision"] = autopilot_execution_frontier_controller_decision
    if execution_dispatch_context.get("autonomous_dispatch_skip_reason") is None:
        execution_dispatch_context["autonomous_dispatch_skip_reason"] = autopilot_execution_frontier_controller_skip_reason
    if execution_dispatch_context.get("autonomous_dispatch_block_reason") is None:
        execution_dispatch_context["autonomous_dispatch_block_reason"] = autopilot_execution_frontier_controller_block_reason
    if not execution_dispatch_context.get("autonomous_dispatch_block_reasons"):
        execution_dispatch_context["autonomous_dispatch_block_reasons"] = autopilot_execution_frontier_controller_block_reasons
    if execution_dispatch_context.get("autonomous_dispatch_error") is None:
        execution_dispatch_context["autonomous_dispatch_error"] = autopilot_execution_frontier_controller_error
    if execution_dispatch_context.get("autonomous_dispatch_updated_at") is None:
        execution_dispatch_context["autonomous_dispatch_updated_at"] = autopilot_execution_frontier_controller_recorded_at
    if execution_dispatch_context.get("autonomous_dispatch_selector_state") is None:
        execution_dispatch_context["autonomous_dispatch_selector_state"] = autopilot_execution_frontier_controller_selector_state
    if execution_dispatch_context.get("autonomous_dispatch_close_condition_met") is None:
        execution_dispatch_context["autonomous_dispatch_close_condition_met"] = autopilot_execution_frontier_controller_close_condition_met
    if execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is not True:
        execution_dispatch_context["autonomous_dispatch_post_completion_enforcement_required"] = (
            autopilot_execution_frontier_controller_post_completion_enforcement_required
        )
    if execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_latched") is not True:
        execution_dispatch_context["autonomous_dispatch_post_completion_enforcement_latched"] = bool(
            execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is True
            and str(execution_dispatch_context.get("autonomous_dispatch_status") or "missing")
            in {"blocked", "error", "missing", "skipped"}
        )
    if execution_dispatch_context.get("autonomous_dispatch_post_completion_loop_state") is None:
        execution_dispatch_context["autonomous_dispatch_post_completion_loop_state"] = (
            autopilot_execution_frontier_controller_loop_state
        )
    if not isinstance(execution_dispatch_context.get("autonomous_dispatch_retry_contract"), dict):
        execution_dispatch_context["autonomous_dispatch_retry_contract"] = (
            autopilot_execution_frontier_controller_retry_contract
        )
    if not isinstance(execution_dispatch_context.get("autonomous_dispatch_cooldown_policy"), dict):
        execution_dispatch_context["autonomous_dispatch_cooldown_policy"] = (
            autopilot_execution_frontier_controller_cooldown_policy
        )
    if not isinstance(execution_dispatch_context.get("autonomous_dispatch_queue_truth_vs_narrative_parity"), dict):
        execution_dispatch_context["autonomous_dispatch_queue_truth_vs_narrative_parity"] = (
            autopilot_execution_frontier_controller_parity
        )
    if execution_dispatch_context.get("autonomous_dispatch_intent_active") is not True:
        execution_dispatch_context["autonomous_dispatch_intent_active"] = bool(
            autopilot_execution_frontier_controller.get("autonomous_execution_intent_active") is True
        )
    if not str(execution_dispatch_context.get("autonomous_dispatch_trace_path") or "").strip():
        execution_dispatch_context["autonomous_dispatch_trace_path"] = autopilot_execution_frontier_controller_trace_path
    if not str(execution_dispatch_context.get("autonomous_dispatch_history_path") or "").strip():
        execution_dispatch_context["autonomous_dispatch_history_path"] = autopilot_execution_frontier_controller_history_path
    if execution_dispatch_context.get("autonomous_dispatch_source_degraded") is not True:
        execution_dispatch_context["autonomous_dispatch_source_degraded"] = autopilot_execution_frontier_controller_source_degraded

execution_context = normalize_execution_context(current.get("execution_context"))
if not execution_context:
    execution_context = build_execution_context(
        readiness=effective_readiness,
        in_flight=current.get("in_flight") if isinstance(current.get("in_flight"), dict) else {},
        mutation_gate=current_mutation_gate or projection_mutation_gate,
        dispatch_context=execution_dispatch_context,
    )

execution_last_signal_at = (
    execution_dispatch_context.get("updated_at")
    or current.get("generated_at")
    or now_obj.get("generated_at")
)
execution_last_signal_source = (
    "dispatch_context.updated_at"
    if execution_dispatch_context.get("updated_at")
    else ("continuity_current.generated_at" if current.get("generated_at") else "continuity_now.generated_at")
)
execution_last_signal_age_sec = age_sec(execution_last_signal_at)
execution_program_state = (
    str(execution_program_status_obj.get("program_state") or "")
    if isinstance(execution_program_status_obj.get("program_state"), str)
    else None
)
execution_current_wave = execution_program_status_obj.get("current_wave")
if not isinstance(execution_current_wave, int) or execution_current_wave < 0:
    execution_current_wave = None
execution_frontier_lane = (
    str(execution_program_status_obj.get("frontier_lane") or "").strip() or None
    if isinstance(execution_program_status_obj.get("frontier_lane"), str)
    else None
)
execution_current_focus = (
    str(execution_program_status_obj.get("current_focus") or "").strip() or None
    if isinstance(execution_program_status_obj.get("current_focus"), str)
    else None
)
execution_last_progress_at = (
    str(execution_program_status_obj.get("last_progress_at") or "")
    if isinstance(execution_program_status_obj.get("last_progress_at"), str)
    else None
)
execution_last_progress_age_sec = age_sec(execution_last_progress_at)
execution_frontier_transition_obj = (
    execution_frontier_ledger_obj.get("transition")
    if isinstance(execution_frontier_ledger_obj.get("transition"), dict)
    else {}
)
execution_frontier_stalled_obj = (
    execution_frontier_ledger_obj.get("stalled_detection")
    if isinstance(execution_frontier_ledger_obj.get("stalled_detection"), dict)
    else {}
)
execution_frontier_selector_state = str(execution_frontier_transition_obj.get("selector_state") or "").strip() or "unknown"
if execution_frontier_selector_state not in {
    "wave_open",
    "ready_for_dispatch",
    "closed_blocked",
    "idle_no_candidate",
    "advanced_wave_closed",
}:
    execution_frontier_selector_state = "unknown"
execution_frontier_close_condition_met = (
    execution_frontier_transition_obj.get("close_condition_met")
    if isinstance(execution_frontier_transition_obj.get("close_condition_met"), bool)
    else None
)
execution_frontier_next_candidate = (
    str(execution_frontier_ledger_obj.get("next_candidate") or "").strip() or None
    if isinstance(execution_frontier_ledger_obj.get("next_candidate"), str)
    else None
)
execution_frontier_next_candidate_wave = execution_frontier_ledger_obj.get("next_candidate_wave")
if not isinstance(execution_frontier_next_candidate_wave, int) or execution_frontier_next_candidate_wave < 0:
    execution_frontier_next_candidate_wave = None
execution_frontier_transition_reason = (
    str(execution_frontier_transition_obj.get("reason") or "").strip() or None
    if isinstance(execution_frontier_transition_obj.get("reason"), str)
    else None
)
execution_frontier_stalled = (
    execution_frontier_stalled_obj.get("stalled")
    if isinstance(execution_frontier_stalled_obj.get("stalled"), bool)
    else None
)
execution_frontier_stalled_reason = (
    str(execution_frontier_stalled_obj.get("reason") or "").strip() or None
    if isinstance(execution_frontier_stalled_obj.get("reason"), str)
    else None
)
execution_frontier_supervisor_obj = (
    execution_frontier_ledger_obj.get("supervisor_state")
    if isinstance(execution_frontier_ledger_obj.get("supervisor_state"), dict)
    else {}
)
execution_frontier_supervisor_state = (
    str(execution_frontier_supervisor_obj.get("state") or "").strip() or None
    if isinstance(execution_frontier_supervisor_obj.get("state"), str)
    else None
)
execution_frontier_autonomous_dispatch_eligible = (
    execution_frontier_supervisor_obj.get("autonomous_dispatch_eligible")
    if isinstance(execution_frontier_supervisor_obj.get("autonomous_dispatch_eligible"), bool)
    else None
)
execution_frontier_autonomous_dispatch_block_reasons = _reason_list(
    execution_frontier_supervisor_obj.get("autonomous_dispatch_block_reasons")
)
execution_frontier = {
    "source_path": "state/continuity/latest/execution_frontier_ledger.json",
    "source_present": bool(execution_frontier_ledger_obj),
    "generated_at": str(execution_frontier_ledger_obj.get("generated_at") or "").strip() or None,
    "program_state": str(execution_frontier_ledger_obj.get("program_state") or "").strip() or None,
    "current_wave": execution_frontier_ledger_obj.get("current_wave") if isinstance(execution_frontier_ledger_obj.get("current_wave"), int) else None,
    "last_completed_wave": execution_frontier_ledger_obj.get("last_completed_wave") if isinstance(execution_frontier_ledger_obj.get("last_completed_wave"), int) else None,
    "selector_state": execution_frontier_selector_state,
    "close_condition_met": execution_frontier_close_condition_met,
    "transition_reason": execution_frontier_transition_reason,
    "next_candidate": execution_frontier_next_candidate,
    "next_candidate_wave": execution_frontier_next_candidate_wave,
    "active_worker_count": _nonnegative_int(execution_frontier_ledger_obj.get("active_worker_count")),
    "blocked_reason": str(execution_frontier_ledger_obj.get("blocked_reason") or "").strip() or None,
    "stalled": execution_frontier_stalled,
    "stalled_reason": execution_frontier_stalled_reason,
    "idle_for_sec": _nonnegative_int(execution_frontier_stalled_obj.get("idle_for_sec")),
    "stalled_after_sec": _nonnegative_int(execution_frontier_stalled_obj.get("stalled_after_sec")),
    "dispatch_status": str(execution_frontier_transition_obj.get("dispatch_status") or "").strip() or None,
    "supervisor_state": execution_frontier_supervisor_state,
    "autonomous_dispatch_eligible": execution_frontier_autonomous_dispatch_eligible,
    "autonomous_dispatch_block_reasons": execution_frontier_autonomous_dispatch_block_reasons,
}

dispatch_intent_launch_readiness_obj = (
    execution_supervisor_dispatch_intent_obj.get("launch_readiness")
    if isinstance(execution_supervisor_dispatch_intent_obj.get("launch_readiness"), dict)
    else {}
)
if not dispatch_intent_launch_readiness_obj:
    consumption_obj = (
        execution_supervisor_dispatch_intent_obj.get("qualification_consumption")
        if isinstance(execution_supervisor_dispatch_intent_obj.get("qualification_consumption"), dict)
        else {}
    )
    dispatch_intent_launch_readiness_obj = (
        consumption_obj.get("launch_readiness")
        if isinstance(consumption_obj.get("launch_readiness"), dict)
        else {}
    )

dispatch_qualification_launch_readiness_obj = (
    execution_supervisor_dispatch_qualification_obj.get("launch_readiness")
    if isinstance(execution_supervisor_dispatch_qualification_obj.get("launch_readiness"), dict)
    else {}
)
dispatch_qualification_source_obj = (
    execution_supervisor_dispatch_qualification_obj.get("source")
    if isinstance(execution_supervisor_dispatch_qualification_obj.get("source"), dict)
    else {}
)
dispatch_intent_launch_readiness_demotion_posture_obj = (
    dispatch_intent_launch_readiness_obj.get("demotion_restore_posture")
    if isinstance(dispatch_intent_launch_readiness_obj.get("demotion_restore_posture"), dict)
    else {}
)
dispatch_qualification_launch_readiness_demotion_posture_obj = (
    dispatch_qualification_launch_readiness_obj.get("demotion_restore_posture")
    if isinstance(dispatch_qualification_launch_readiness_obj.get("demotion_restore_posture"), dict)
    else {}
)
dispatch_intent_launch_readiness_severity_gate_obj = (
    dispatch_intent_launch_readiness_obj.get("severity_gate")
    if isinstance(dispatch_intent_launch_readiness_obj.get("severity_gate"), dict)
    else {}
)
dispatch_intent_launch_readiness_probe_execution_plan_obj = (
    dispatch_intent_launch_readiness_obj.get("probe_execution_plan")
    if isinstance(dispatch_intent_launch_readiness_obj.get("probe_execution_plan"), dict)
    else {}
)
dispatch_intent_probe_execution_plan_obj = (
    execution_supervisor_dispatch_intent_obj.get("probe_execution_plan")
    if isinstance(execution_supervisor_dispatch_intent_obj.get("probe_execution_plan"), dict)
    else {}
)
dispatch_intent_probe_execution_plan_source_obj = (
    dispatch_intent_launch_readiness_probe_execution_plan_obj
    if dispatch_intent_launch_readiness_probe_execution_plan_obj
    else dispatch_intent_probe_execution_plan_obj
)
dispatch_intent_probe_execution_plan_present = bool(dispatch_intent_probe_execution_plan_source_obj)
dispatch_qualification_launch_readiness_severity_gate_obj = (
    dispatch_qualification_launch_readiness_obj.get("severity_gate")
    if isinstance(dispatch_qualification_launch_readiness_obj.get("severity_gate"), dict)
    else {}
)
dispatch_qualification_launch_readiness_probe_execution_plan_obj = (
    dispatch_qualification_launch_readiness_obj.get("probe_execution_plan")
    if isinstance(dispatch_qualification_launch_readiness_obj.get("probe_execution_plan"), dict)
    else {}
)
dispatch_qualification_probe_execution_plan_obj = (
    execution_supervisor_dispatch_qualification_obj.get("probe_execution_plan")
    if isinstance(execution_supervisor_dispatch_qualification_obj.get("probe_execution_plan"), dict)
    else {}
)
dispatch_qualification_probe_execution_plan_source_obj = (
    dispatch_qualification_launch_readiness_probe_execution_plan_obj
    if dispatch_qualification_launch_readiness_probe_execution_plan_obj
    else dispatch_qualification_probe_execution_plan_obj
)
dispatch_qualification_probe_execution_plan_present = bool(dispatch_qualification_probe_execution_plan_source_obj)

execution_supervisor_dispatch_intent = {
    "source_path": "state/continuity/latest/execution_supervisor_dispatch_intent_latest.json",
    "source_present": bool(execution_supervisor_dispatch_intent_obj),
    "generated_at": str(execution_supervisor_dispatch_intent_obj.get("generated_at") or "").strip() or None,
    "status": str(execution_supervisor_dispatch_intent_obj.get("status") or "").strip() or None,
    "decision": str(execution_supervisor_dispatch_intent_obj.get("decision") or "").strip() or None,
    "active": (
        execution_supervisor_dispatch_intent_obj.get("active")
        if isinstance(execution_supervisor_dispatch_intent_obj.get("active"), bool)
        else None
    ),
    "fail_closed": (
        execution_supervisor_dispatch_intent_obj.get("fail_closed")
        if isinstance(execution_supervisor_dispatch_intent_obj.get("fail_closed"), bool)
        else None
    ),
    "launch_mutation_allowed": (
        execution_supervisor_dispatch_intent_obj.get("launch_mutation_allowed")
        if isinstance(execution_supervisor_dispatch_intent_obj.get("launch_mutation_allowed"), bool)
        else None
    ),
    "ready_candidate_count": _nonnegative_int(execution_supervisor_dispatch_intent_obj.get("ready_candidate_count")),
    "blocked_candidate_count": _nonnegative_int(execution_supervisor_dispatch_intent_obj.get("blocked_candidate_count")),
    "ready_candidate_task_ids": unique_preserve(
        [
            str(task_id or "").strip()
            for task_id in (execution_supervisor_dispatch_intent_obj.get("ready_candidate_task_ids") or [])
            if str(task_id or "").strip()
        ]
    ),
    "blocked_candidate_task_ids": unique_preserve(
        [
            str(task_id or "").strip()
            for task_id in (execution_supervisor_dispatch_intent_obj.get("blocked_candidate_task_ids") or [])
            if str(task_id or "").strip()
        ]
    ),
    "decision_reasons": _reason_list(execution_supervisor_dispatch_intent_obj.get("decision_reasons")),
    "launch_readiness_state": str(dispatch_intent_launch_readiness_obj.get("state") or "").strip() or None,
    "launch_readiness_reason": str(dispatch_intent_launch_readiness_obj.get("reason") or "").strip() or None,
    "launch_readiness_demotion_posture_present": bool(dispatch_intent_launch_readiness_demotion_posture_obj),
    "launch_readiness_restore_pending_worker_count": _nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("restore_pending_worker_count")
    ),
    "launch_readiness_demoted_worker_count": _nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("demoted_worker_count")
    ),
    "launch_readiness_restored_worker_count": _nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("restored_worker_count")
    ),
    "launch_readiness_oldest_restore_pending_since": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_since") or "").strip() or None
    ),
    "launch_readiness_oldest_restore_pending_worker": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_worker") or "").strip() or None
    ),
    "launch_readiness_oldest_restore_pending_age_sec": _optional_nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_age_sec")
    ),
    "launch_readiness_oldest_demoted_at": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_demoted_at") or "").strip() or None
    ),
    "launch_readiness_oldest_demoted_worker": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_demoted_worker") or "").strip() or None
    ),
    "launch_readiness_oldest_demoted_age_sec": _optional_nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("oldest_demoted_age_sec")
    ),
    "launch_readiness_latest_restored_at": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("latest_restored_at") or "").strip() or None
    ),
    "launch_readiness_latest_restored_worker": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("latest_restored_worker") or "").strip() or None
    ),
    "launch_readiness_latest_restored_age_sec": _optional_nonnegative_int(
        dispatch_intent_launch_readiness_demotion_posture_obj.get("latest_restored_age_sec")
    ),
    "launch_readiness_demotion_action_priority": (
        str(dispatch_intent_launch_readiness_demotion_posture_obj.get("action_priority") or "").strip().lower()
        if str(dispatch_intent_launch_readiness_demotion_posture_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_blocked_probe_candidate_count": _nonnegative_int(
        dispatch_intent_launch_readiness_obj.get("blocked_probe_candidate_count")
    ),
    "launch_readiness_probe_execution_status": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("status") or "").strip() or None
    ),
    "launch_readiness_probe_execution_reason": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("reason") or "").strip() or None
    ),
    "launch_readiness_probe_execution_action_priority": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower()
        if str(dispatch_intent_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_action_priority_source": (
        "probe_execution_plan"
        if str(dispatch_intent_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_pending_worker_count": _nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("pending_worker_count")
    ),
    "launch_readiness_probe_execution_due_now_worker_count": _nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("due_now_worker_count")
    ),
    "launch_readiness_probe_execution_scheduled_worker_count": _nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("scheduled_worker_count")
    ),
    "launch_readiness_probe_execution_overdue_worker_count": _nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("overdue_worker_count")
    ),
    "launch_readiness_probe_execution_oldest_due_now_worker": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("oldest_due_now_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_started_at": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("oldest_due_now_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_age_sec": _optional_nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("oldest_due_now_age_sec")
    ),
    "launch_readiness_probe_execution_oldest_overdue_worker": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("oldest_overdue_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_started_at": (
        str(dispatch_intent_probe_execution_plan_source_obj.get("oldest_overdue_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_age_sec": _optional_nonnegative_int(
        dispatch_intent_probe_execution_plan_source_obj.get("oldest_overdue_age_sec")
    ),
    "launch_readiness_severity_state": (
        str(dispatch_intent_launch_readiness_severity_gate_obj.get("state") or "").strip() or None
    ),
    "launch_readiness_severity_reason": (
        str(dispatch_intent_launch_readiness_severity_gate_obj.get("reason") or "").strip() or None
    ),
    "launch_readiness_severity_active": bool(
        dispatch_intent_launch_readiness_severity_gate_obj.get("active") is True
    ),
    "launch_readiness_severity_threshold_ticks": _nonnegative_int(
        dispatch_intent_launch_readiness_severity_gate_obj.get("threshold_ticks")
    ),
    "launch_readiness_severity_non_ready_ticks": _nonnegative_int(
        dispatch_intent_launch_readiness_severity_gate_obj.get("non_ready_ticks_consecutive")
    ),
    "launch_readiness_severity_cohort_worker_count": _nonnegative_int(
        dispatch_intent_launch_readiness_severity_gate_obj.get("cohort_worker_count")
    ),
}

execution_supervisor_dispatch_qualification = {
    "source_path": "state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
    "source_present": bool(execution_supervisor_dispatch_qualification_obj),
    "generated_at": str(execution_supervisor_dispatch_qualification_obj.get("generated_at") or "").strip() or None,
    "status": str(execution_supervisor_dispatch_qualification_obj.get("status") or "").strip() or None,
    "decision": str(execution_supervisor_dispatch_qualification_obj.get("decision") or "").strip() or None,
    "active": (
        execution_supervisor_dispatch_qualification_obj.get("active")
        if isinstance(execution_supervisor_dispatch_qualification_obj.get("active"), bool)
        else None
    ),
    "fail_closed": (
        execution_supervisor_dispatch_qualification_obj.get("fail_closed")
        if isinstance(execution_supervisor_dispatch_qualification_obj.get("fail_closed"), bool)
        else None
    ),
    "launch_mutation_allowed": (
        execution_supervisor_dispatch_qualification_obj.get("launch_mutation_allowed")
        if isinstance(execution_supervisor_dispatch_qualification_obj.get("launch_mutation_allowed"), bool)
        else None
    ),
    "ready_candidate_count": _nonnegative_int(execution_supervisor_dispatch_qualification_obj.get("ready_candidate_count")),
    "qualified_candidate_count": _nonnegative_int(
        execution_supervisor_dispatch_qualification_obj.get("qualified_candidate_count")
    ),
    "blocked_candidate_count": _nonnegative_int(execution_supervisor_dispatch_qualification_obj.get("blocked_candidate_count")),
    "qualified_candidate_task_ids": unique_preserve(
        [
            str(task_id or "").strip()
            for task_id in (execution_supervisor_dispatch_qualification_obj.get("qualified_candidate_task_ids") or [])
            if str(task_id or "").strip()
        ]
    ),
    "blocked_candidate_task_ids": unique_preserve(
        [
            str(task_id or "").strip()
            for task_id in (execution_supervisor_dispatch_qualification_obj.get("blocked_candidate_task_ids") or [])
            if str(task_id or "").strip()
        ]
    ),
    "decision_reasons": _reason_list(execution_supervisor_dispatch_qualification_obj.get("decision_reasons")),
    "worker_health_gate_required": bool(dispatch_qualification_source_obj.get("worker_health_gate_required") is True),
    "worker_health_canary_present": bool(dispatch_qualification_source_obj.get("worker_health_canary_present") is True),
    "worker_health_canary_path": str(dispatch_qualification_source_obj.get("worker_health_canary_path") or "").strip() or None,
    "worker_health_canary_schema": str(dispatch_qualification_source_obj.get("worker_health_canary_schema") or "").strip() or None,
    "launch_readiness_state": str(dispatch_qualification_launch_readiness_obj.get("state") or "").strip() or None,
    "launch_readiness_reason": str(dispatch_qualification_launch_readiness_obj.get("reason") or "").strip() or None,
    "launch_readiness_demotion_posture_present": bool(dispatch_qualification_launch_readiness_demotion_posture_obj),
    "launch_readiness_restore_pending_worker_count": _nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("restore_pending_worker_count")
    ),
    "launch_readiness_demoted_worker_count": _nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("demoted_worker_count")
    ),
    "launch_readiness_restored_worker_count": _nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("restored_worker_count")
    ),
    "launch_readiness_oldest_restore_pending_since": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_since") or "").strip() or None
    ),
    "launch_readiness_oldest_restore_pending_worker": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_worker") or "").strip() or None
    ),
    "launch_readiness_oldest_restore_pending_age_sec": _optional_nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_restore_pending_age_sec")
    ),
    "launch_readiness_oldest_demoted_at": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_demoted_at") or "").strip() or None
    ),
    "launch_readiness_oldest_demoted_worker": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_demoted_worker") or "").strip() or None
    ),
    "launch_readiness_oldest_demoted_age_sec": _optional_nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("oldest_demoted_age_sec")
    ),
    "launch_readiness_latest_restored_at": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("latest_restored_at") or "").strip() or None
    ),
    "launch_readiness_latest_restored_worker": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("latest_restored_worker") or "").strip() or None
    ),
    "launch_readiness_latest_restored_age_sec": _optional_nonnegative_int(
        dispatch_qualification_launch_readiness_demotion_posture_obj.get("latest_restored_age_sec")
    ),
    "launch_readiness_demotion_action_priority": (
        str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("action_priority") or "").strip().lower()
        if str(dispatch_qualification_launch_readiness_demotion_posture_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_blocked_probe_candidate_count": _nonnegative_int(
        dispatch_qualification_launch_readiness_obj.get("blocked_probe_candidate_count")
    ),
    "launch_readiness_probe_execution_status": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("status") or "").strip() or None
    ),
    "launch_readiness_probe_execution_reason": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("reason") or "").strip() or None
    ),
    "launch_readiness_probe_execution_action_priority": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower()
        if str(dispatch_qualification_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_action_priority_source": (
        "probe_execution_plan"
        if str(dispatch_qualification_probe_execution_plan_source_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_pending_worker_count": _nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("pending_worker_count")
    ),
    "launch_readiness_probe_execution_due_now_worker_count": _nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("due_now_worker_count")
    ),
    "launch_readiness_probe_execution_scheduled_worker_count": _nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("scheduled_worker_count")
    ),
    "launch_readiness_probe_execution_overdue_worker_count": _nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("overdue_worker_count")
    ),
    "launch_readiness_probe_execution_oldest_due_now_worker": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("oldest_due_now_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_started_at": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("oldest_due_now_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_age_sec": _optional_nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("oldest_due_now_age_sec")
    ),
    "launch_readiness_probe_execution_oldest_overdue_worker": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("oldest_overdue_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_started_at": (
        str(dispatch_qualification_probe_execution_plan_source_obj.get("oldest_overdue_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_age_sec": _optional_nonnegative_int(
        dispatch_qualification_probe_execution_plan_source_obj.get("oldest_overdue_age_sec")
    ),
    "launch_readiness_severity_state": (
        str(dispatch_qualification_launch_readiness_severity_gate_obj.get("state") or "").strip() or None
    ),
    "launch_readiness_severity_reason": (
        str(dispatch_qualification_launch_readiness_severity_gate_obj.get("reason") or "").strip() or None
    ),
    "launch_readiness_severity_active": bool(
        dispatch_qualification_launch_readiness_severity_gate_obj.get("active") is True
    ),
    "launch_readiness_severity_threshold_ticks": _nonnegative_int(
        dispatch_qualification_launch_readiness_severity_gate_obj.get("threshold_ticks")
    ),
    "launch_readiness_severity_non_ready_ticks": _nonnegative_int(
        dispatch_qualification_launch_readiness_severity_gate_obj.get("non_ready_ticks_consecutive")
    ),
    "launch_readiness_severity_cohort_worker_count": _nonnegative_int(
        dispatch_qualification_launch_readiness_severity_gate_obj.get("cohort_worker_count")
    ),
}

execution_supervisor_launch_readiness_severity_source = (
    execution_supervisor_dispatch_qualification
    if execution_supervisor_dispatch_qualification.get("launch_readiness_severity_state")
    else execution_supervisor_dispatch_intent
)
execution_supervisor_launch_readiness_severity_state = str(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_state") or ""
).strip() or None
execution_supervisor_launch_readiness_severity_reason = str(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_reason") or ""
).strip() or None
execution_supervisor_launch_readiness_severity_active = bool(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_active") is True
)
execution_supervisor_launch_readiness_severity_non_ready_ticks = _nonnegative_int(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_non_ready_ticks")
)
execution_supervisor_launch_readiness_severity_threshold_ticks = _nonnegative_int(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_threshold_ticks")
)
execution_supervisor_launch_readiness_severity_cohort_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_severity_source.get("launch_readiness_severity_cohort_worker_count")
)

execution_supervisor_probe_execution_plan_projection = {
    "source_path": "state/continuity/latest/execution_supervisor_probe_execution_plan_latest.json",
    "source_present": bool(execution_supervisor_probe_execution_plan_obj),
    "launch_readiness_probe_execution_status": (
        str(execution_supervisor_probe_execution_plan_obj.get("status") or "").strip() or None
    ),
    "launch_readiness_probe_execution_reason": (
        str(execution_supervisor_probe_execution_plan_obj.get("reason") or "").strip() or None
    ),
    "launch_readiness_probe_execution_action_priority": (
        str(execution_supervisor_probe_execution_plan_obj.get("action_priority") or "").strip().lower()
        if str(execution_supervisor_probe_execution_plan_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_action_priority_source": (
        "probe_execution_plan"
        if str(execution_supervisor_probe_execution_plan_obj.get("action_priority") or "").strip().lower() in {"p1", "p2"}
        else None
    ),
    "launch_readiness_probe_execution_pending_worker_count": _nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("pending_worker_count")
    ),
    "launch_readiness_probe_execution_due_now_worker_count": _nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("due_now_worker_count")
    ),
    "launch_readiness_probe_execution_scheduled_worker_count": _nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("scheduled_worker_count")
    ),
    "launch_readiness_probe_execution_overdue_worker_count": _nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("overdue_worker_count")
    ),
    "launch_readiness_probe_execution_oldest_due_now_worker": (
        str(execution_supervisor_probe_execution_plan_obj.get("oldest_due_now_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_started_at": (
        str(execution_supervisor_probe_execution_plan_obj.get("oldest_due_now_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_due_now_age_sec": _optional_nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("oldest_due_now_age_sec")
    ),
    "launch_readiness_probe_execution_oldest_overdue_worker": (
        str(execution_supervisor_probe_execution_plan_obj.get("oldest_overdue_worker") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_started_at": (
        str(execution_supervisor_probe_execution_plan_obj.get("oldest_overdue_started_at") or "").strip() or None
    ),
    "launch_readiness_probe_execution_oldest_overdue_age_sec": _optional_nonnegative_int(
        execution_supervisor_probe_execution_plan_obj.get("oldest_overdue_age_sec")
    ),
}

execution_supervisor_launch_readiness_probe_execution_source = (
    execution_supervisor_dispatch_qualification
    if dispatch_qualification_probe_execution_plan_present
    else (
        execution_supervisor_dispatch_intent
        if dispatch_intent_probe_execution_plan_present
        else execution_supervisor_probe_execution_plan_projection
    )
)
execution_supervisor_launch_readiness_probe_execution_status = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_status") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_reason = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_reason") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_action_priority = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_action_priority") or ""
).strip().lower() or None
execution_supervisor_launch_readiness_probe_execution_action_priority_source = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_action_priority_source") or ""
).strip().lower() or None
if execution_supervisor_launch_readiness_probe_execution_action_priority not in {"p1", "p2"}:
    execution_supervisor_launch_readiness_probe_execution_action_priority = None
if execution_supervisor_launch_readiness_probe_execution_action_priority_source not in {
    "probe_execution_plan",
    "probe_execution_gate",
    "demotion_restore_posture",
}:
    execution_supervisor_launch_readiness_probe_execution_action_priority_source = None
execution_supervisor_launch_readiness_probe_execution_pending_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_pending_worker_count")
)
execution_supervisor_launch_readiness_probe_execution_due_now_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_due_now_worker_count")
)
execution_supervisor_launch_readiness_probe_execution_overdue_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_overdue_worker_count")
)
execution_supervisor_launch_readiness_probe_execution_oldest_due_now_worker = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_due_now_worker") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_oldest_due_now_started_at = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_due_now_started_at") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec = _optional_nonnegative_int(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_due_now_age_sec")
)
execution_supervisor_launch_readiness_probe_execution_oldest_overdue_worker = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_overdue_worker") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_oldest_overdue_started_at = str(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_overdue_started_at") or ""
).strip() or None
execution_supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec = _optional_nonnegative_int(
    execution_supervisor_launch_readiness_probe_execution_source.get("launch_readiness_probe_execution_oldest_overdue_age_sec")
)

execution_supervisor_launch_readiness_demotion_projection = {
    "source": "verify_gate_preflight",
    "launch_readiness_demotion_posture_present": bool(
        verify_probe_execution_gate_demotion_restore_pending_worker_count > 0
        or verify_probe_execution_gate_demotion_demoted_worker_count > 0
        or verify_probe_execution_gate_demotion_restored_worker_count > 0
        or bool(verify_probe_execution_gate_demotion_oldest_restore_pending_since)
        or bool(verify_probe_execution_gate_demotion_oldest_demoted_at)
        or bool(verify_probe_execution_gate_demotion_latest_restored_at)
        or verify_probe_execution_gate_demotion_action_priority in {"p1", "p2"}
    ),
    "launch_readiness_restore_pending_worker_count": verify_probe_execution_gate_demotion_restore_pending_worker_count,
    "launch_readiness_demoted_worker_count": verify_probe_execution_gate_demotion_demoted_worker_count,
    "launch_readiness_restored_worker_count": verify_probe_execution_gate_demotion_restored_worker_count,
    "launch_readiness_oldest_restore_pending_since": verify_probe_execution_gate_demotion_oldest_restore_pending_since,
    "launch_readiness_oldest_restore_pending_worker": verify_probe_execution_gate_demotion_oldest_restore_pending_worker,
    "launch_readiness_oldest_restore_pending_age_sec": verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec,
    "launch_readiness_oldest_demoted_at": verify_probe_execution_gate_demotion_oldest_demoted_at,
    "launch_readiness_oldest_demoted_worker": verify_probe_execution_gate_demotion_oldest_demoted_worker,
    "launch_readiness_oldest_demoted_age_sec": verify_probe_execution_gate_demotion_oldest_demoted_age_sec,
    "launch_readiness_latest_restored_at": verify_probe_execution_gate_demotion_latest_restored_at,
    "launch_readiness_latest_restored_worker": verify_probe_execution_gate_demotion_latest_restored_worker,
    "launch_readiness_latest_restored_age_sec": verify_probe_execution_gate_demotion_latest_restored_age_sec,
    "launch_readiness_demotion_action_priority": verify_probe_execution_gate_demotion_action_priority,
}
execution_supervisor_launch_readiness_demotion_source = (
    execution_supervisor_dispatch_qualification
    if execution_supervisor_dispatch_qualification.get("launch_readiness_demotion_posture_present") is True
    else (
        execution_supervisor_dispatch_intent
        if execution_supervisor_dispatch_intent.get("launch_readiness_demotion_posture_present") is True
        else (
            execution_supervisor_launch_readiness_demotion_projection
            if execution_supervisor_launch_readiness_demotion_projection.get("launch_readiness_demotion_posture_present") is True
            else {}
        )
    )
)
execution_supervisor_launch_readiness_restore_pending_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_restore_pending_worker_count")
)
execution_supervisor_launch_readiness_demoted_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_demoted_worker_count")
)
execution_supervisor_launch_readiness_restored_worker_count = _nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_restored_worker_count")
)
execution_supervisor_launch_readiness_oldest_restore_pending_since = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_restore_pending_since") or ""
).strip() or None
execution_supervisor_launch_readiness_oldest_restore_pending_worker = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_restore_pending_worker") or ""
).strip() or None
execution_supervisor_launch_readiness_oldest_restore_pending_age_sec = _optional_nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_restore_pending_age_sec")
)
execution_supervisor_launch_readiness_oldest_demoted_at = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_demoted_at") or ""
).strip() or None
execution_supervisor_launch_readiness_oldest_demoted_worker = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_demoted_worker") or ""
).strip() or None
execution_supervisor_launch_readiness_oldest_demoted_age_sec = _optional_nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_oldest_demoted_age_sec")
)
execution_supervisor_launch_readiness_latest_restored_at = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_latest_restored_at") or ""
).strip() or None
execution_supervisor_launch_readiness_latest_restored_worker = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_latest_restored_worker") or ""
).strip() or None
execution_supervisor_launch_readiness_latest_restored_age_sec = _optional_nonnegative_int(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_latest_restored_age_sec")
)
execution_supervisor_launch_readiness_demotion_action_priority = str(
    execution_supervisor_launch_readiness_demotion_source.get("launch_readiness_demotion_action_priority") or ""
).strip().lower() or None
if execution_supervisor_launch_readiness_demotion_action_priority not in {"p1", "p2"}:
    execution_supervisor_launch_readiness_demotion_action_priority = None
if (
    execution_supervisor_launch_readiness_probe_execution_action_priority is None
    and execution_supervisor_launch_readiness_demotion_action_priority in {"p1", "p2"}
):
    execution_supervisor_launch_readiness_probe_execution_action_priority = (
        execution_supervisor_launch_readiness_demotion_action_priority
    )
    execution_supervisor_launch_readiness_probe_execution_action_priority_source = "demotion_restore_posture"

if execution_supervisor_launch_readiness_severity_active:
    severity_warning_reason = (
        execution_supervisor_launch_readiness_severity_reason
        or "launch_readiness_persistent_non_ready_demoted_cohort"
    )
    severity_warning_token = f"execution_supervisor_{severity_warning_reason}"
    if severity_warning_token not in warning_reasons:
        warning_reasons.append(severity_warning_token)

if execution_supervisor_launch_readiness_probe_execution_overdue_worker_count > 0:
    if "execution_supervisor_probe_execution_overdue" not in warning_reasons:
        warning_reasons.append("execution_supervisor_probe_execution_overdue")
elif execution_supervisor_launch_readiness_probe_execution_due_now_worker_count > 0:
    if "execution_supervisor_probe_execution_due_now" not in warning_reasons:
        warning_reasons.append("execution_supervisor_probe_execution_due_now")

if verify_probe_execution_gate_due_now_worker_count > 0:
    if "execution_supervisor_probe_execution_due_now" not in warning_reasons:
        warning_reasons.append("execution_supervisor_probe_execution_due_now")
if verify_probe_execution_gate_overdue_worker_count > 0:
    if "execution_supervisor_probe_execution_overdue" not in warning_reasons:
        warning_reasons.append("execution_supervisor_probe_execution_overdue")
if verify_probe_execution_gate_active_blocker:
    if "execution_supervisor_probe_execution_overdue_gate_active" not in warning_reasons:
        warning_reasons.append("execution_supervisor_probe_execution_overdue_gate_active")
if execution_supervisor_launch_readiness_restore_pending_worker_count > 0:
    if "execution_supervisor_demotion_restore_pending" not in warning_reasons:
        warning_reasons.append("execution_supervisor_demotion_restore_pending")

execution_status = {
    "posture": str(execution_context.get("posture") or "idle"),
    "readiness": str(execution_context.get("readiness") or effective_readiness),
    "in_flight": bool(execution_context.get("in_flight") is True),
    "running_tasks": _nonnegative_int(execution_context.get("running_tasks")),
    "active_locks": _nonnegative_int(execution_context.get("active_locks")),
    "program_state": execution_program_state,
    "current_wave": execution_current_wave,
    "frontier_lane": execution_frontier_lane,
    "current_focus": execution_current_focus,
    "last_progress_at": execution_last_progress_at,
    "last_progress_age_sec": execution_last_progress_age_sec,
    "frontier_selector_state": execution_frontier.get("selector_state"),
    "frontier_close_condition_met": execution_frontier.get("close_condition_met"),
    "frontier_next_candidate": execution_frontier.get("next_candidate"),
    "frontier_next_candidate_wave": execution_frontier.get("next_candidate_wave"),
    "frontier_supervisor_state": execution_frontier.get("supervisor_state"),
    "frontier_autonomous_dispatch_eligible": execution_frontier.get("autonomous_dispatch_eligible"),
    "mutation_gate_status": str(execution_context.get("mutation_gate_status") or mutation_gate_status or "unknown"),
    "mutation_gate_posture": str(execution_context.get("mutation_gate_posture") or mutation_gate_posture or "unknown"),
    "expected_in_flight_guard": execution_context.get("expected_in_flight_guard") if isinstance(execution_context.get("expected_in_flight_guard"), bool) else None,
    "dispatch_status": str(execution_dispatch_context.get("status") or execution_context.get("dispatch_status") or "missing"),
    "autopilot_status": str(execution_dispatch_context.get("autopilot_status") or "missing"),
    "target_step_id": execution_dispatch_context.get("target_step_id"),
    "launched_step_id": execution_dispatch_context.get("launched_step_id"),
    "ready_work_exists": bool(execution_dispatch_context.get("ready_work_exists") is True),
    "idle_threshold_exceeded": bool(execution_dispatch_context.get("idle_threshold_exceeded") is True),
    "idle_sec": _nonnegative_int(execution_dispatch_context.get("idle_sec")),
    "skip_reason": execution_dispatch_context.get("skip_reason"),
    "trace_path": execution_dispatch_context.get("trace_path"),
    "autonomous_dispatch_status": str(execution_dispatch_context.get("autonomous_dispatch_status") or "missing"),
    "autonomous_dispatch_decision": execution_dispatch_context.get("autonomous_dispatch_decision"),
    "autonomous_dispatch_skip_reason": execution_dispatch_context.get("autonomous_dispatch_skip_reason"),
    "autonomous_dispatch_block_reason": execution_dispatch_context.get("autonomous_dispatch_block_reason"),
    "autonomous_dispatch_block_reasons": _reason_list(execution_dispatch_context.get("autonomous_dispatch_block_reasons")),
    "autonomous_dispatch_error": execution_dispatch_context.get("autonomous_dispatch_error"),
    "autonomous_dispatch_updated_at": execution_dispatch_context.get("autonomous_dispatch_updated_at"),
    "autonomous_dispatch_selector_state": execution_dispatch_context.get("autonomous_dispatch_selector_state"),
    "autonomous_dispatch_close_condition_met": execution_dispatch_context.get("autonomous_dispatch_close_condition_met")
    if isinstance(execution_dispatch_context.get("autonomous_dispatch_close_condition_met"), bool)
    else None,
    "autonomous_dispatch_post_completion_enforcement_required": bool(
        execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is True
    ),
    "autonomous_dispatch_post_completion_enforcement_latched": bool(
        execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_latched") is True
    ),
    "autonomous_dispatch_post_completion_loop_state": execution_dispatch_context.get(
        "autonomous_dispatch_post_completion_loop_state"
    ),
    "autonomous_dispatch_retry_contract": (
        execution_dispatch_context.get("autonomous_dispatch_retry_contract")
        if isinstance(execution_dispatch_context.get("autonomous_dispatch_retry_contract"), dict)
        else {}
    ),
    "autonomous_dispatch_cooldown_policy": (
        execution_dispatch_context.get("autonomous_dispatch_cooldown_policy")
        if isinstance(execution_dispatch_context.get("autonomous_dispatch_cooldown_policy"), dict)
        else {}
    ),
    "autonomous_dispatch_queue_truth_vs_narrative_parity": (
        execution_dispatch_context.get("autonomous_dispatch_queue_truth_vs_narrative_parity")
        if isinstance(execution_dispatch_context.get("autonomous_dispatch_queue_truth_vs_narrative_parity"), dict)
        else {}
    ),
    "autonomous_dispatch_intent_active": bool(
        execution_dispatch_context.get("autonomous_dispatch_intent_active") is True
    ),
    "autonomous_dispatch_post_completion_blocked": bool(
        execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is True
        and (
            str(execution_dispatch_context.get("autonomous_dispatch_status") or "missing") == "blocked"
            or str(execution_dispatch_context.get("autonomous_dispatch_post_completion_loop_state") or "")
            == "BLOCKED_LOOP"
        )
    ),
    "autonomous_dispatch_post_completion_stalled": bool(
        execution_dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is True
        and (
            str(execution_dispatch_context.get("autonomous_dispatch_status") or "missing")
            in {"error", "missing", "skipped"}
            or str(execution_dispatch_context.get("autonomous_dispatch_post_completion_loop_state") or "")
            == "STALLED_LOOP"
        )
    ),
    "autonomous_dispatch_trace_path": execution_dispatch_context.get("autonomous_dispatch_trace_path"),
    "autonomous_dispatch_history_path": execution_dispatch_context.get("autonomous_dispatch_history_path"),
    "autonomous_dispatch_source_degraded": bool(execution_dispatch_context.get("autonomous_dispatch_source_degraded") is True),
    "supervisor_dispatch_intent_status": execution_supervisor_dispatch_intent.get("status"),
    "supervisor_dispatch_intent_decision": execution_supervisor_dispatch_intent.get("decision"),
    "supervisor_dispatch_intent_active": execution_supervisor_dispatch_intent.get("active"),
    "supervisor_dispatch_intent_fail_closed": execution_supervisor_dispatch_intent.get("fail_closed"),
    "supervisor_dispatch_intent_launch_mutation_allowed": execution_supervisor_dispatch_intent.get("launch_mutation_allowed"),
    "supervisor_dispatch_intent_ready_candidate_count": execution_supervisor_dispatch_intent.get("ready_candidate_count"),
    "supervisor_dispatch_intent_blocked_candidate_count": execution_supervisor_dispatch_intent.get("blocked_candidate_count"),
    "supervisor_dispatch_intent_ready_candidate_task_ids": execution_supervisor_dispatch_intent.get("ready_candidate_task_ids"),
    "supervisor_dispatch_intent_blocked_candidate_task_ids": execution_supervisor_dispatch_intent.get("blocked_candidate_task_ids"),
    "supervisor_dispatch_intent_decision_reasons": execution_supervisor_dispatch_intent.get("decision_reasons"),
    "supervisor_dispatch_intent_launch_readiness_state": execution_supervisor_dispatch_intent.get("launch_readiness_state"),
    "supervisor_dispatch_intent_launch_readiness_reason": execution_supervisor_dispatch_intent.get("launch_readiness_reason"),
    "supervisor_dispatch_intent_restore_pending_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_restore_pending_worker_count"),
    "supervisor_dispatch_intent_demoted_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_demoted_worker_count"),
    "supervisor_dispatch_intent_restored_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_restored_worker_count"),
    "supervisor_dispatch_intent_oldest_restore_pending_since": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_restore_pending_since"),
    "supervisor_dispatch_intent_oldest_restore_pending_worker": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_restore_pending_worker"),
    "supervisor_dispatch_intent_oldest_restore_pending_age_sec": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_restore_pending_age_sec"),
    "supervisor_dispatch_intent_oldest_demoted_at": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_demoted_at"),
    "supervisor_dispatch_intent_oldest_demoted_worker": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_demoted_worker"),
    "supervisor_dispatch_intent_oldest_demoted_age_sec": execution_supervisor_dispatch_intent.get("launch_readiness_oldest_demoted_age_sec"),
    "supervisor_dispatch_intent_latest_restored_at": execution_supervisor_dispatch_intent.get("launch_readiness_latest_restored_at"),
    "supervisor_dispatch_intent_latest_restored_worker": execution_supervisor_dispatch_intent.get("launch_readiness_latest_restored_worker"),
    "supervisor_dispatch_intent_latest_restored_age_sec": execution_supervisor_dispatch_intent.get("launch_readiness_latest_restored_age_sec"),
    "supervisor_dispatch_intent_demotion_action_priority": execution_supervisor_dispatch_intent.get("launch_readiness_demotion_action_priority"),
    "supervisor_dispatch_intent_blocked_probe_candidate_count": execution_supervisor_dispatch_intent.get("launch_readiness_blocked_probe_candidate_count"),
    "supervisor_dispatch_intent_probe_execution_status": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_status"),
    "supervisor_dispatch_intent_probe_execution_reason": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_reason"),
    "supervisor_dispatch_intent_probe_execution_action_priority": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_action_priority"),
    "supervisor_dispatch_intent_probe_execution_pending_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_pending_worker_count"),
    "supervisor_dispatch_intent_probe_execution_due_now_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_due_now_worker_count"),
    "supervisor_dispatch_intent_probe_execution_scheduled_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_scheduled_worker_count"),
    "supervisor_dispatch_intent_probe_execution_overdue_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_overdue_worker_count"),
    "supervisor_dispatch_intent_probe_execution_oldest_due_now_worker": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_due_now_worker"),
    "supervisor_dispatch_intent_probe_execution_oldest_due_now_started_at": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_due_now_started_at"),
    "supervisor_dispatch_intent_probe_execution_oldest_due_now_age_sec": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_due_now_age_sec"),
    "supervisor_dispatch_intent_probe_execution_oldest_overdue_worker": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_overdue_worker"),
    "supervisor_dispatch_intent_probe_execution_oldest_overdue_started_at": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_overdue_started_at"),
    "supervisor_dispatch_intent_probe_execution_oldest_overdue_age_sec": execution_supervisor_dispatch_intent.get("launch_readiness_probe_execution_oldest_overdue_age_sec"),
    "supervisor_dispatch_intent_launch_readiness_severity_state": execution_supervisor_dispatch_intent.get("launch_readiness_severity_state"),
    "supervisor_dispatch_intent_launch_readiness_severity_reason": execution_supervisor_dispatch_intent.get("launch_readiness_severity_reason"),
    "supervisor_dispatch_intent_launch_readiness_severity_active": execution_supervisor_dispatch_intent.get("launch_readiness_severity_active"),
    "supervisor_dispatch_intent_launch_readiness_severity_non_ready_ticks": execution_supervisor_dispatch_intent.get("launch_readiness_severity_non_ready_ticks"),
    "supervisor_dispatch_intent_launch_readiness_severity_threshold_ticks": execution_supervisor_dispatch_intent.get("launch_readiness_severity_threshold_ticks"),
    "supervisor_dispatch_intent_launch_readiness_severity_cohort_worker_count": execution_supervisor_dispatch_intent.get("launch_readiness_severity_cohort_worker_count"),
    "supervisor_dispatch_qualification_status": execution_supervisor_dispatch_qualification.get("status"),
    "supervisor_dispatch_qualification_decision": execution_supervisor_dispatch_qualification.get("decision"),
    "supervisor_dispatch_qualification_active": execution_supervisor_dispatch_qualification.get("active"),
    "supervisor_dispatch_qualification_fail_closed": execution_supervisor_dispatch_qualification.get("fail_closed"),
    "supervisor_dispatch_qualification_launch_mutation_allowed": execution_supervisor_dispatch_qualification.get("launch_mutation_allowed"),
    "supervisor_dispatch_qualification_ready_candidate_count": execution_supervisor_dispatch_qualification.get("ready_candidate_count"),
    "supervisor_dispatch_qualification_qualified_candidate_count": execution_supervisor_dispatch_qualification.get("qualified_candidate_count"),
    "supervisor_dispatch_qualification_blocked_candidate_count": execution_supervisor_dispatch_qualification.get("blocked_candidate_count"),
    "supervisor_dispatch_qualification_qualified_candidate_task_ids": execution_supervisor_dispatch_qualification.get("qualified_candidate_task_ids"),
    "supervisor_dispatch_qualification_blocked_candidate_task_ids": execution_supervisor_dispatch_qualification.get("blocked_candidate_task_ids"),
    "supervisor_dispatch_qualification_decision_reasons": execution_supervisor_dispatch_qualification.get("decision_reasons"),
    "supervisor_dispatch_qualification_launch_readiness_state": execution_supervisor_dispatch_qualification.get("launch_readiness_state"),
    "supervisor_dispatch_qualification_launch_readiness_reason": execution_supervisor_dispatch_qualification.get("launch_readiness_reason"),
    "supervisor_dispatch_qualification_restore_pending_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_restore_pending_worker_count"),
    "supervisor_dispatch_qualification_demoted_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_demoted_worker_count"),
    "supervisor_dispatch_qualification_restored_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_restored_worker_count"),
    "supervisor_dispatch_qualification_oldest_restore_pending_since": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_restore_pending_since"),
    "supervisor_dispatch_qualification_oldest_restore_pending_worker": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_restore_pending_worker"),
    "supervisor_dispatch_qualification_oldest_restore_pending_age_sec": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_restore_pending_age_sec"),
    "supervisor_dispatch_qualification_oldest_demoted_at": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_demoted_at"),
    "supervisor_dispatch_qualification_oldest_demoted_worker": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_demoted_worker"),
    "supervisor_dispatch_qualification_oldest_demoted_age_sec": execution_supervisor_dispatch_qualification.get("launch_readiness_oldest_demoted_age_sec"),
    "supervisor_dispatch_qualification_latest_restored_at": execution_supervisor_dispatch_qualification.get("launch_readiness_latest_restored_at"),
    "supervisor_dispatch_qualification_latest_restored_worker": execution_supervisor_dispatch_qualification.get("launch_readiness_latest_restored_worker"),
    "supervisor_dispatch_qualification_latest_restored_age_sec": execution_supervisor_dispatch_qualification.get("launch_readiness_latest_restored_age_sec"),
    "supervisor_dispatch_qualification_demotion_action_priority": execution_supervisor_dispatch_qualification.get("launch_readiness_demotion_action_priority"),
    "supervisor_dispatch_qualification_blocked_probe_candidate_count": execution_supervisor_dispatch_qualification.get("launch_readiness_blocked_probe_candidate_count"),
    "supervisor_dispatch_qualification_probe_execution_status": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_status"),
    "supervisor_dispatch_qualification_probe_execution_reason": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_reason"),
    "supervisor_dispatch_qualification_probe_execution_action_priority": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_action_priority"),
    "supervisor_dispatch_qualification_probe_execution_pending_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_pending_worker_count"),
    "supervisor_dispatch_qualification_probe_execution_due_now_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_due_now_worker_count"),
    "supervisor_dispatch_qualification_probe_execution_scheduled_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_scheduled_worker_count"),
    "supervisor_dispatch_qualification_probe_execution_overdue_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_overdue_worker_count"),
    "supervisor_dispatch_qualification_probe_execution_oldest_due_now_worker": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_due_now_worker"),
    "supervisor_dispatch_qualification_probe_execution_oldest_due_now_started_at": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_due_now_started_at"),
    "supervisor_dispatch_qualification_probe_execution_oldest_due_now_age_sec": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_due_now_age_sec"),
    "supervisor_dispatch_qualification_probe_execution_oldest_overdue_worker": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_overdue_worker"),
    "supervisor_dispatch_qualification_probe_execution_oldest_overdue_started_at": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_overdue_started_at"),
    "supervisor_dispatch_qualification_probe_execution_oldest_overdue_age_sec": execution_supervisor_dispatch_qualification.get("launch_readiness_probe_execution_oldest_overdue_age_sec"),
    "supervisor_dispatch_qualification_launch_readiness_severity_state": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_state"),
    "supervisor_dispatch_qualification_launch_readiness_severity_reason": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_reason"),
    "supervisor_dispatch_qualification_launch_readiness_severity_active": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_active"),
    "supervisor_dispatch_qualification_launch_readiness_severity_non_ready_ticks": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_non_ready_ticks"),
    "supervisor_dispatch_qualification_launch_readiness_severity_threshold_ticks": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_threshold_ticks"),
    "supervisor_dispatch_qualification_launch_readiness_severity_cohort_worker_count": execution_supervisor_dispatch_qualification.get("launch_readiness_severity_cohort_worker_count"),
    "supervisor_launch_readiness_severity_state": execution_supervisor_launch_readiness_severity_state,
    "supervisor_launch_readiness_severity_reason": execution_supervisor_launch_readiness_severity_reason,
    "supervisor_launch_readiness_severity_active": execution_supervisor_launch_readiness_severity_active,
    "supervisor_launch_readiness_severity_non_ready_ticks": execution_supervisor_launch_readiness_severity_non_ready_ticks,
    "supervisor_launch_readiness_severity_threshold_ticks": execution_supervisor_launch_readiness_severity_threshold_ticks,
    "supervisor_launch_readiness_severity_cohort_worker_count": execution_supervisor_launch_readiness_severity_cohort_worker_count,
    "supervisor_launch_readiness_restore_pending_worker_count": execution_supervisor_launch_readiness_restore_pending_worker_count,
    "supervisor_launch_readiness_demoted_worker_count": execution_supervisor_launch_readiness_demoted_worker_count,
    "supervisor_launch_readiness_restored_worker_count": execution_supervisor_launch_readiness_restored_worker_count,
    "supervisor_launch_readiness_oldest_restore_pending_since": execution_supervisor_launch_readiness_oldest_restore_pending_since,
    "supervisor_launch_readiness_oldest_restore_pending_worker": execution_supervisor_launch_readiness_oldest_restore_pending_worker,
    "supervisor_launch_readiness_oldest_restore_pending_age_sec": execution_supervisor_launch_readiness_oldest_restore_pending_age_sec,
    "supervisor_launch_readiness_oldest_demoted_at": execution_supervisor_launch_readiness_oldest_demoted_at,
    "supervisor_launch_readiness_oldest_demoted_worker": execution_supervisor_launch_readiness_oldest_demoted_worker,
    "supervisor_launch_readiness_oldest_demoted_age_sec": execution_supervisor_launch_readiness_oldest_demoted_age_sec,
    "supervisor_launch_readiness_latest_restored_at": execution_supervisor_launch_readiness_latest_restored_at,
    "supervisor_launch_readiness_latest_restored_worker": execution_supervisor_launch_readiness_latest_restored_worker,
    "supervisor_launch_readiness_latest_restored_age_sec": execution_supervisor_launch_readiness_latest_restored_age_sec,
    "supervisor_launch_readiness_demotion_action_priority": execution_supervisor_launch_readiness_demotion_action_priority,
    "supervisor_launch_readiness_probe_execution_status": execution_supervisor_launch_readiness_probe_execution_status,
    "supervisor_launch_readiness_probe_execution_reason": execution_supervisor_launch_readiness_probe_execution_reason,
    "supervisor_launch_readiness_probe_execution_action_priority": execution_supervisor_launch_readiness_probe_execution_action_priority,
    "supervisor_launch_readiness_probe_execution_action_priority_source": execution_supervisor_launch_readiness_probe_execution_action_priority_source,
    "supervisor_launch_readiness_probe_execution_pending_worker_count": execution_supervisor_launch_readiness_probe_execution_pending_worker_count,
    "supervisor_launch_readiness_probe_execution_due_now_worker_count": execution_supervisor_launch_readiness_probe_execution_due_now_worker_count,
    "supervisor_launch_readiness_probe_execution_overdue_worker_count": execution_supervisor_launch_readiness_probe_execution_overdue_worker_count,
    "supervisor_launch_readiness_probe_execution_oldest_due_now_worker": execution_supervisor_launch_readiness_probe_execution_oldest_due_now_worker,
    "supervisor_launch_readiness_probe_execution_oldest_due_now_started_at": execution_supervisor_launch_readiness_probe_execution_oldest_due_now_started_at,
    "supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec": execution_supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec,
    "supervisor_launch_readiness_probe_execution_oldest_overdue_worker": execution_supervisor_launch_readiness_probe_execution_oldest_overdue_worker,
    "supervisor_launch_readiness_probe_execution_oldest_overdue_started_at": execution_supervisor_launch_readiness_probe_execution_oldest_overdue_started_at,
    "supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec": execution_supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec,
    "last_signal_at": execution_last_signal_at,
    "last_signal_source": execution_last_signal_source,
    "last_signal_age_sec": execution_last_signal_age_sec,
}

core_roadmap_execution_queue_obj = load_json_if_exists(core_roadmap_execution_queue_path) or {}
if not core_roadmap_execution_queue_obj:
    core_roadmap_execution_queue_obj = load_json_if_exists(core_roadmap_slice_queue_path) or {}

librarian_promotions_rows: List[Dict[str, Any]] = []
if librarian_promotions_path.exists():
    try:
        with librarian_promotions_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                txt = str(line or "").strip()
                if not txt:
                    continue
                try:
                    row = json.loads(txt)
                except Exception:
                    continue
                if isinstance(row, dict):
                    librarian_promotions_rows.append(row)
    except Exception:
        librarian_promotions_rows = []

evidence_trace_viewer = build_evidence_trace_viewer_projection(
    queue_payload=core_roadmap_execution_queue_obj,
    execution_status_payload=execution_status,
    current_payload=current,
    promotions_rows=librarian_promotions_rows,
)
try:
    atomic_write(evidence_trace_viewer_export_path, evidence_trace_viewer)
except Exception as exc:
    if "evidence_trace_viewer_export_failed" not in warning_reasons:
        warning_reasons.append(f"evidence_trace_viewer_export_failed:{exc.__class__.__name__}")

meaningful_event_reporting_payload_present = bool(
    isinstance(execution_meaningful_event_reporting_obj, dict)
    and execution_meaningful_event_reporting_obj
)
meaningful_event_status_payload_present = bool(
    isinstance(execution_meaningful_event_reporting_status_obj, dict)
    and execution_meaningful_event_reporting_status_obj
)

meaningful_event_reporting = (
    execution_meaningful_event_reporting_obj
    if isinstance(execution_meaningful_event_reporting_obj, dict)
    else {}
)
meaningful_event_status = (
    execution_meaningful_event_reporting_status_obj
    if isinstance(execution_meaningful_event_reporting_status_obj, dict)
    else {}
)

critical_transition_code_set = {
    "WORKER_FAILED_OR_JUNK",
    "QUEUE_BLOCKED",
    "QUEUE_RELAUNCHED",
    "EXECUTOR_IDLE_TO_RELAUNCHED",
}

meaningful_event_status_fallback_used = False
if not meaningful_event_status:
    meaningful_event_status_fallback_used = True
    pending_codes_fallback = unique_preserve(
        [
            str(code).strip()
            for code in (meaningful_event_reporting.get("pending_required_event_codes") or [])
            if str(code).strip()
        ]
    )
    meaningful_event_status = {
        "status": "warning" if pending_codes_fallback else "clear",
        "operator_attention_required": bool(pending_codes_fallback),
        "pending_required_event_codes": pending_codes_fallback,
        "pending_required_event_count": len(pending_codes_fallback),
        "critical_pending_event_codes": [
            code
            for code in pending_codes_fallback
            if code in critical_transition_code_set
        ],
        "critical_pending_event_count": sum(
            1
            for code in pending_codes_fallback
            if code in critical_transition_code_set
        ),
        "new_event_count": _nonnegative_int(meaningful_event_reporting.get("new_event_count")),
        "new_event_codes": unique_preserve(
            [
                str((row or {}).get("event_code") or "").strip()
                for row in (meaningful_event_reporting.get("new_events") or [])
                if str((row or {}).get("event_code") or "").strip()
            ]
        ),
        "latest_event_code": None,
        "latest_event_detected_at": None,
        "attention_reasons": ["pending_required_event_codes_present"] if pending_codes_fallback else [],
        "checklist_projection": {
            "event_packet_minimum_fields_ok": (
                ((meaningful_event_reporting.get("checklist_status") or {}).get("event_packet_minimum_fields") or {}).get("ok")
                is True
            ),
            "event_packet_field_gap_count": len(
                ((meaningful_event_reporting.get("checklist_status") or {}).get("event_packet_minimum_fields") or {}).get("gaps")
                if isinstance((((meaningful_event_reporting.get("checklist_status") or {}).get("event_packet_minimum_fields") or {}).get("gaps")), list)
                else []
            ),
            "missed_trigger_recovery_required": (
                ((meaningful_event_reporting.get("checklist_status") or {}).get("missed_trigger_recovery") or {}).get("required")
                is True
            ),
            "action_before_narration_guard_required": False,
            "action_before_narration_guard_ok": True,
        },
    }

meaningful_event_reporting_pending_codes = unique_preserve(
    [
        str(code).strip()
        for code in (meaningful_event_reporting.get("pending_required_event_codes") or [])
        if str(code).strip()
    ]
)
meaningful_event_reporting_new_codes = unique_preserve(
    [
        str((row or {}).get("event_code") or "").strip()
        for row in (meaningful_event_reporting.get("new_events") or [])
        if str((row or {}).get("event_code") or "").strip()
    ]
)
meaningful_event_status_packet_pending_codes = unique_preserve(
    [
        str(code).strip()
        for code in (
            execution_meaningful_event_reporting_status_obj.get("pending_required_event_codes")
            if meaningful_event_status_payload_present
            and isinstance(execution_meaningful_event_reporting_status_obj, dict)
            and isinstance(execution_meaningful_event_reporting_status_obj.get("pending_required_event_codes"), list)
            else []
        )
        if str(code).strip()
    ]
)
meaningful_event_status_packet_generated_at = (
    str(execution_meaningful_event_reporting_status_obj.get("generated_at") or "").strip()
    if meaningful_event_status_payload_present and isinstance(execution_meaningful_event_reporting_status_obj, dict)
    else None
)
meaningful_event_status_packet_declared_pending_count = (
    _coerce_int(execution_meaningful_event_reporting_status_obj.get("pending_required_event_count"))
    if meaningful_event_status_payload_present and isinstance(execution_meaningful_event_reporting_status_obj, dict)
    else None
)

meaningful_event_pending_codes = unique_preserve(
    [
        str(code).strip()
        for code in (meaningful_event_status.get("pending_required_event_codes") or [])
        if str(code).strip()
    ]
)
meaningful_event_critical_pending_codes = unique_preserve(
    [
        str(code).strip()
        for code in (meaningful_event_status.get("critical_pending_event_codes") or [])
        if str(code).strip()
    ]
)
meaningful_event_new_codes = unique_preserve(
    [
        str(code).strip()
        for code in (meaningful_event_status.get("new_event_codes") or [])
        if str(code).strip()
    ]
)
meaningful_event_pending_count = _nonnegative_int(
    meaningful_event_status.get("pending_required_event_count"),
    default=len(meaningful_event_pending_codes),
)
if meaningful_event_pending_count <= 0 and meaningful_event_pending_codes:
    meaningful_event_pending_count = len(meaningful_event_pending_codes)
meaningful_event_critical_pending_count = _nonnegative_int(
    meaningful_event_status.get("critical_pending_event_count"),
    default=len(meaningful_event_critical_pending_codes),
)
if meaningful_event_critical_pending_count <= 0 and meaningful_event_critical_pending_codes:
    meaningful_event_critical_pending_count = len(meaningful_event_critical_pending_codes)
meaningful_event_new_count = _nonnegative_int(
    meaningful_event_status.get("new_event_count"),
    default=len(meaningful_event_new_codes),
)
if meaningful_event_new_count <= 0 and meaningful_event_new_codes:
    meaningful_event_new_count = len(meaningful_event_new_codes)
meaningful_event_status_label = str(meaningful_event_status.get("status") or "clear").strip() or "clear"
meaningful_event_attention_required = bool(meaningful_event_status.get("operator_attention_required") is True)
if not meaningful_event_attention_required and meaningful_event_pending_count > 0:
    meaningful_event_attention_required = True
meaningful_event_latest_code = str(meaningful_event_status.get("latest_event_code") or "").strip() or None
meaningful_event_latest_detected_at = str(meaningful_event_status.get("latest_event_detected_at") or "").strip() or None
if not meaningful_event_latest_code or not meaningful_event_latest_detected_at:
    latest_row: Optional[Dict[str, Any]] = None
    latest_row_detected: Optional[dt.datetime] = None
    for row in (meaningful_event_reporting.get("recent_events") or []):
        if not isinstance(row, dict):
            continue
        row_detected = parse_iso(row.get("detected_at") or row.get("event_at"))
        if row_detected is None:
            continue
        if latest_row_detected is None or row_detected >= latest_row_detected:
            latest_row_detected = row_detected
            latest_row = row
    if latest_row and not meaningful_event_latest_code:
        meaningful_event_latest_code = str(latest_row.get("event_code") or "").strip() or None
    if latest_row_detected is not None and not meaningful_event_latest_detected_at:
        meaningful_event_latest_detected_at = (
            latest_row_detected.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )

meaningful_event_attention_reasons = unique_preserve(
    [
        str(reason).strip()
        for reason in (meaningful_event_status.get("attention_reasons") or [])
        if str(reason).strip()
    ]
)
meaningful_event_checklist_projection = (
    meaningful_event_status.get("checklist_projection")
    if isinstance(meaningful_event_status.get("checklist_projection"), dict)
    else {}
)
meaningful_event_packet_fields_ok = bool(
    meaningful_event_checklist_projection.get("event_packet_minimum_fields_ok") is True
)
meaningful_event_missed_trigger_required = bool(
    meaningful_event_checklist_projection.get("missed_trigger_recovery_required") is True
)
meaningful_event_action_guard_required = bool(
    meaningful_event_checklist_projection.get("action_before_narration_guard_required") is True
)
meaningful_event_action_guard_ok = bool(
    meaningful_event_checklist_projection.get("action_before_narration_guard_ok") is True
)
if meaningful_event_action_guard_required and not meaningful_event_action_guard_ok:
    meaningful_event_attention_required = True
    if "action_before_narration_guard_failed" not in meaningful_event_attention_reasons:
        meaningful_event_attention_reasons.append("action_before_narration_guard_failed")
if meaningful_event_missed_trigger_required:
    meaningful_event_attention_required = True
    if "missed_trigger_recovery_required" not in meaningful_event_attention_reasons:
        meaningful_event_attention_reasons.append("missed_trigger_recovery_required")
if not meaningful_event_packet_fields_ok:
    meaningful_event_attention_required = True
    if "event_packet_field_gap" not in meaningful_event_attention_reasons:
        meaningful_event_attention_reasons.append("event_packet_field_gap")

meaningful_event_pending_codes_summary = summarize_codes_for_state(
    meaningful_event_pending_codes,
    max_items=4,
)
meaningful_event_critical_pending_codes_summary = summarize_codes_for_state(
    meaningful_event_critical_pending_codes,
    max_items=3,
)

meaningful_event_reporting_projection = {
    "source_path": "state/continuity/latest/execution_meaningful_event_reporting_latest.json",
    "status_path": "state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
    "status": meaningful_event_status_label,
    "attention_required": meaningful_event_attention_required,
    "pending_required_event_count": meaningful_event_pending_count,
    "pending_required_event_codes": meaningful_event_pending_codes,
    "pending_required_event_codes_summary": meaningful_event_pending_codes_summary,
    "critical_pending_event_count": meaningful_event_critical_pending_count,
    "critical_pending_event_codes": meaningful_event_critical_pending_codes,
    "critical_pending_event_codes_summary": meaningful_event_critical_pending_codes_summary,
    "new_event_count": meaningful_event_new_count,
    "new_event_codes": meaningful_event_new_codes,
    "latest_event_code": meaningful_event_latest_code,
    "latest_event_detected_at": meaningful_event_latest_detected_at,
    "attention_reasons": meaningful_event_attention_reasons,
    "checklist_projection": {
        "event_packet_minimum_fields_ok": meaningful_event_packet_fields_ok,
        "missed_trigger_recovery_required": meaningful_event_missed_trigger_required,
        "action_before_narration_guard_required": meaningful_event_action_guard_required,
        "action_before_narration_guard_ok": meaningful_event_action_guard_ok,
    },
    "generated_at": str(meaningful_event_status.get("generated_at") or meaningful_event_reporting.get("generated_at") or "").strip() or None,
}

meaningful_event_required_packet_expected = bool(
    meaningful_event_reporting_pending_codes
    or meaningful_event_reporting_new_codes
)
meaningful_event_status_packet_pending_codes_set = set(meaningful_event_status_packet_pending_codes)
meaningful_event_reporting_pending_codes_set = set(meaningful_event_reporting_pending_codes)
meaningful_event_missing_required_pending_codes = [
    code
    for code in meaningful_event_reporting_pending_codes
    if code not in meaningful_event_status_packet_pending_codes_set
]
meaningful_event_unexpected_pending_codes = [
    code
    for code in meaningful_event_status_packet_pending_codes
    if code not in meaningful_event_reporting_pending_codes_set
]

meaningful_event_status_declared_pending_count = meaningful_event_status_packet_declared_pending_count
meaningful_event_status_pending_count_mismatch = bool(
    meaningful_event_status_declared_pending_count is not None
    and meaningful_event_status_declared_pending_count >= 0
    and meaningful_event_status_declared_pending_count != len(meaningful_event_status_packet_pending_codes)
)

meaningful_event_reporting_generated_at = (
    str(meaningful_event_reporting.get("generated_at") or "").strip() or None
)
meaningful_event_status_generated_at = (
    str(meaningful_event_status_packet_generated_at or "").strip() or None
)
meaningful_event_reporting_generated_dt = parse_iso(meaningful_event_reporting_generated_at)
meaningful_event_status_generated_dt = parse_iso(meaningful_event_status_generated_at)
meaningful_event_status_stale_vs_reporting = bool(
    meaningful_event_required_packet_expected
    and meaningful_event_reporting_generated_dt is not None
    and meaningful_event_status_generated_dt is not None
    and meaningful_event_status_generated_dt < meaningful_event_reporting_generated_dt
)

meaningful_event_contract_failclose_reasons: List[str] = []
meaningful_event_contract_warning_reasons: List[str] = []

if meaningful_event_required_packet_expected and not meaningful_event_status_payload_present:
    meaningful_event_contract_failclose_reasons.append("meaningful_event_status_artifact_missing")

if meaningful_event_required_packet_expected and meaningful_event_missing_required_pending_codes:
    meaningful_event_contract_failclose_reasons.append("meaningful_event_status_missing_required_pending_codes")

if meaningful_event_required_packet_expected and meaningful_event_status_stale_vs_reporting:
    meaningful_event_contract_failclose_reasons.append("meaningful_event_status_stale_vs_reporting")

if meaningful_event_status_pending_count_mismatch:
    mismatch_reason = "meaningful_event_status_pending_count_mismatch"
    if meaningful_event_required_packet_expected:
        meaningful_event_contract_failclose_reasons.append(mismatch_reason)
    else:
        meaningful_event_contract_warning_reasons.append(mismatch_reason)

if meaningful_event_required_packet_expected and meaningful_event_unexpected_pending_codes:
    meaningful_event_contract_warning_reasons.append("meaningful_event_status_unexpected_pending_codes")

if meaningful_event_required_packet_expected and meaningful_event_status_fallback_used:
    meaningful_event_contract_warning_reasons.append("meaningful_event_status_fallback_projection_used")

meaningful_event_contract_failclose_reasons = unique_preserve(meaningful_event_contract_failclose_reasons)
meaningful_event_contract_warning_reasons = unique_preserve(meaningful_event_contract_warning_reasons)

meaningful_event_expected_codes_canonical = sorted(meaningful_event_reporting_pending_codes)
meaningful_event_delivered_codes_canonical = sorted(meaningful_event_status_packet_pending_codes)
meaningful_event_packet_checksum_basis = (
    f"required={1 if meaningful_event_required_packet_expected else 0};"
    f"expected_count={len(meaningful_event_expected_codes_canonical)};"
    f"expected_codes={','.join(meaningful_event_expected_codes_canonical)};"
    f"delivered_count={len(meaningful_event_delivered_codes_canonical)};"
    f"delivered_codes={','.join(meaningful_event_delivered_codes_canonical)};"
    f"missing_count={len(meaningful_event_missing_required_pending_codes)};"
    f"unexpected_count={len(meaningful_event_unexpected_pending_codes)};"
    f"stale_vs_reporting={1 if meaningful_event_status_stale_vs_reporting else 0};"
    f"fallback={1 if meaningful_event_status_fallback_used else 0}"
)
meaningful_event_packet_checksum_hash = hashlib.sha256(
    meaningful_event_packet_checksum_basis.encode("utf-8")
).hexdigest()[:12]
meaningful_event_packet_checksum_line = (
    f"exp={len(meaningful_event_expected_codes_canonical)} "
    f"del={len(meaningful_event_delivered_codes_canonical)} "
    f"miss={len(meaningful_event_missing_required_pending_codes)} "
    f"extra={len(meaningful_event_unexpected_pending_codes)} "
    f"stale={1 if meaningful_event_status_stale_vs_reporting else 0} "
    f"chk={meaningful_event_packet_checksum_hash}"
)

meaningful_event_contract_status = "ok"
if meaningful_event_contract_failclose_reasons:
    meaningful_event_contract_status = "failclose"
elif meaningful_event_contract_warning_reasons:
    meaningful_event_contract_status = "warning"

meaningful_event_contract_failclose_summary = summarize_codes_for_state(
    meaningful_event_contract_failclose_reasons,
    max_items=2,
)
meaningful_event_contract_warning_summary = summarize_codes_for_state(
    meaningful_event_contract_warning_reasons,
    max_items=2,
)
meaningful_event_contract_digest_basis = (
    f"status={meaningful_event_contract_status};"
    f"required={1 if meaningful_event_required_packet_expected else 0};"
    f"packet_checksum={meaningful_event_packet_checksum_hash};"
    f"failclose={','.join(sorted(meaningful_event_contract_failclose_reasons))};"
    f"warning={','.join(sorted(meaningful_event_contract_warning_reasons))}"
)
meaningful_event_contract_digest_hash = hashlib.sha256(
    meaningful_event_contract_digest_basis.encode("utf-8")
).hexdigest()[:12]
meaningful_event_contract_digest_token = (
    f"mrc_{meaningful_event_contract_status}_{meaningful_event_contract_digest_hash}"
)
meaningful_event_contract_digest_status = (
    "failclose"
    if meaningful_event_contract_failclose_reasons
    else ("warning" if meaningful_event_contract_warning_reasons else "ok")
)
meaningful_event_contract_escalation_level = (
    "p0"
    if meaningful_event_contract_failclose_reasons
    else (
        "p1"
        if meaningful_event_required_packet_expected and meaningful_event_contract_warning_reasons
        else ("p2" if meaningful_event_contract_warning_reasons else "none")
    )
)

meaningful_event_contract_failclose_digest_reason = (
    f"meaningful_event_reporting_contract_failclose_digest:{meaningful_event_contract_digest_token}"
    if meaningful_event_contract_failclose_reasons
    else None
)
meaningful_event_contract_warning_digest_reason = (
    f"meaningful_event_reporting_contract_warning_digest:{meaningful_event_contract_digest_token}"
    if meaningful_event_contract_warning_reasons
    else None
)

if meaningful_event_contract_failclose_digest_reason and meaningful_event_contract_failclose_digest_reason not in not_ready_reasons:
    not_ready_reasons.append(meaningful_event_contract_failclose_digest_reason)
if meaningful_event_contract_warning_digest_reason and meaningful_event_contract_warning_digest_reason not in warning_reasons:
    warning_reasons.append(meaningful_event_contract_warning_digest_reason)

meaningful_event_reporting_contract = {
    "status": meaningful_event_contract_status,
    "required_packet_expected": meaningful_event_required_packet_expected,
    "reporting_payload_present": meaningful_event_reporting_payload_present,
    "status_payload_present": meaningful_event_status_payload_present,
    "status_fallback_projection_used": meaningful_event_status_fallback_used,
    "reporting_pending_required_event_count": len(meaningful_event_reporting_pending_codes),
    "reporting_pending_required_event_codes": meaningful_event_reporting_pending_codes,
    "reporting_pending_required_event_codes_summary": summarize_codes_for_state(
        meaningful_event_reporting_pending_codes,
        max_items=4,
    ),
    "status_projection_pending_required_event_count": meaningful_event_pending_count,
    "status_projection_pending_required_event_codes": meaningful_event_pending_codes,
    "status_projection_pending_required_event_codes_summary": meaningful_event_pending_codes_summary,
    "status_pending_required_event_count": len(meaningful_event_status_packet_pending_codes),
    "status_pending_required_event_codes": meaningful_event_status_packet_pending_codes,
    "status_pending_required_event_codes_summary": summarize_codes_for_state(
        meaningful_event_status_packet_pending_codes,
        max_items=4,
    ),
    "missing_required_pending_codes": meaningful_event_missing_required_pending_codes,
    "unexpected_pending_codes": meaningful_event_unexpected_pending_codes,
    "status_pending_count_declared": meaningful_event_status_declared_pending_count,
    "status_pending_count_mismatch": meaningful_event_status_pending_count_mismatch,
    "status_stale_vs_reporting": meaningful_event_status_stale_vs_reporting,
    "failclose_reasons": meaningful_event_contract_failclose_reasons,
    "warning_reasons": meaningful_event_contract_warning_reasons,
    "digest_status": meaningful_event_contract_digest_status,
    "digest_token": meaningful_event_contract_digest_token,
    "digest_failclose_reason_count": len(meaningful_event_contract_failclose_reasons),
    "digest_failclose_reason_summary": meaningful_event_contract_failclose_summary,
    "digest_warning_reason_count": len(meaningful_event_contract_warning_reasons),
    "digest_warning_reason_summary": meaningful_event_contract_warning_summary,
    "expected_vs_delivered_packet_checksum_line": meaningful_event_packet_checksum_line,
    "expected_vs_delivered_packet_checksum_hash": meaningful_event_packet_checksum_hash,
    "expected_vs_delivered_packet_checksum_basis": meaningful_event_packet_checksum_basis,
    "escalation_level": meaningful_event_contract_escalation_level,
    "generated_at": meaningful_event_status_generated_at
    or meaningful_event_reporting_generated_at
    or meaningful_event_reporting_projection.get("generated_at"),
    "recommended_actions": (
        [
            "bash ops/openclaw/continuity.sh current --refresh --json",
            "cat state/continuity/latest/execution_meaningful_event_reporting_latest.json",
            "cat state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
        ]
        if meaningful_event_contract_failclose_reasons or meaningful_event_contract_warning_reasons
        else ["none"]
    ),
}

truth_strip = [
    {
        "id": "overall_ready_state",
        "state": effective_readiness,
        "freshness": current.get("generated_at"),
        "owner": "continuity",
        "next_safe_action": cmd_cont_current_refresh_json,
    },
    {
        "id": "mutation_gate",
        "state": (
            f"status={mutation_gate_status} "
            f"posture={mutation_gate_posture} "
            f"expected_in_flight_guard={mutation_gate_expected_in_flight_guard if isinstance(mutation_gate_expected_in_flight_guard, bool) else 'n/a'} "
            f"blocking={len(mutation_gate_blocking_reasons)} "
            f"concurrency={len(mutation_gate_concurrency_reasons)}"
        ),
        "freshness": current.get("generated_at"),
        "owner": "successor_guard",
        "next_safe_action": cmd_cont_verify_json,
    },
    {
        "id": "live_execution_status",
        "state": (
            f"posture={execution_status.get('posture') or 'idle'} "
            f"program_state={execution_status.get('program_state') or 'unknown'} "
            f"wave={execution_status.get('current_wave') if isinstance(execution_status.get('current_wave'), int) else 'n/a'} "
            f"frontier={execution_status.get('frontier_lane') or 'none'} "
            f"focus={execution_status.get('current_focus') or 'none'} "
            f"dispatch={execution_status.get('dispatch_status') or 'missing'} "
            f"auto_dispatch={execution_status.get('autonomous_dispatch_status') or 'missing'} "
            f"running={int(execution_status.get('running_tasks') or 0)} "
            f"target={execution_status.get('target_step_id') or 'none'} "
            f"launched={execution_status.get('launched_step_id') or 'none'} "
            f"last_signal_age_sec={execution_status.get('last_signal_age_sec') if isinstance(execution_status.get('last_signal_age_sec'), int) else 'n/a'}"
        ),
        "freshness": execution_status.get("last_signal_at") or current.get("generated_at") or now_obj.get("generated_at"),
        "owner": "continuity_current",
        "next_safe_action": cmd_cont_current_refresh_json,
    },
    {
        "id": "meaningful_event_reporting",
        "state": (
            f"status={meaningful_event_reporting_projection.get('status') or 'clear'} "
            f"attention={meaningful_event_reporting_projection.get('attention_required') is True} "
            f"pending={int(meaningful_event_reporting_projection.get('pending_required_event_count') or 0)} "
            f"critical_pending={int(meaningful_event_reporting_projection.get('critical_pending_event_count') or 0)} "
            f"new={int(meaningful_event_reporting_projection.get('new_event_count') or 0)} "
            f"latest={meaningful_event_reporting_projection.get('latest_event_code') or 'none'} "
            f"pending_codes={meaningful_event_reporting_projection.get('pending_required_event_codes_summary') or 'none'}"
        ),
        "freshness": meaningful_event_reporting_projection.get("generated_at")
        or meaningful_event_reporting_projection.get("latest_event_detected_at")
        or now_obj.get("generated_at")
        or current.get("generated_at"),
        "owner": "execution_meaningful_event_reporting",
        "next_safe_action": cmd_read_execution_meaningful_event_reporting_status_json,
    },
    {
        "id": "meaningful_event_reporting_contract",
        "state": (
            f"status={meaningful_event_reporting_contract.get('status') or 'ok'} "
            f"required_expected={meaningful_event_reporting_contract.get('required_packet_expected') is True} "
            f"missing_required={len(meaningful_event_reporting_contract.get('missing_required_pending_codes') or [])} "
            f"unexpected={len(meaningful_event_reporting_contract.get('unexpected_pending_codes') or [])} "
            f"stale_vs_reporting={meaningful_event_reporting_contract.get('status_stale_vs_reporting') is True} "
            f"digest={meaningful_event_reporting_contract.get('digest_token') or 'none'} "
            f"packet={meaningful_event_reporting_contract.get('expected_vs_delivered_packet_checksum_line') or 'none'}"
        ),
        "freshness": meaningful_event_reporting_contract.get("generated_at")
        or meaningful_event_reporting_projection.get("generated_at")
        or now_obj.get("generated_at")
        or current.get("generated_at"),
        "owner": "execution_meaningful_event_reporting",
        "next_safe_action": cmd_read_execution_meaningful_event_reporting_status_json,
    },
    {
        "id": "execution_frontier_supervisor",
        "state": (
            f"source_present={execution_frontier.get('source_present') is True} "
            f"selector={execution_frontier.get('selector_state') or 'unknown'} "
            f"close={execution_frontier.get('close_condition_met') if isinstance(execution_frontier.get('close_condition_met'), bool) else 'unknown'} "
            f"next={execution_frontier.get('next_candidate') or 'none'} "
            f"next_wave={execution_frontier.get('next_candidate_wave') if isinstance(execution_frontier.get('next_candidate_wave'), int) else 'n/a'} "
            f"reason={execution_frontier.get('transition_reason') or 'none'} "
            f"stalled={execution_frontier.get('stalled') if isinstance(execution_frontier.get('stalled'), bool) else 'unknown'} "
            f"supervisor={execution_frontier.get('supervisor_state') or 'unknown'} "
            f"auto_dispatch_eligible={execution_frontier.get('autonomous_dispatch_eligible') if isinstance(execution_frontier.get('autonomous_dispatch_eligible'), bool) else 'unknown'}"
        ),
        "freshness": execution_frontier.get("generated_at") or current.get("generated_at") or now_obj.get("generated_at"),
        "owner": "execution_frontier_ledger",
        "next_safe_action": cmd_cont_execution_frontier_refresh_json,
    },
    {
        "id": "execution_supervisor_dispatch_intent",
        "state": (
            f"source_present={execution_supervisor_dispatch_intent.get('source_present') is True} "
            f"status={execution_supervisor_dispatch_intent.get('status') or 'missing'} "
            f"decision={execution_supervisor_dispatch_intent.get('decision') or 'missing'} "
            f"fail_closed={execution_supervisor_dispatch_intent.get('fail_closed') if isinstance(execution_supervisor_dispatch_intent.get('fail_closed'), bool) else 'unknown'} "
            f"launch_mutation_allowed={execution_supervisor_dispatch_intent.get('launch_mutation_allowed') if isinstance(execution_supervisor_dispatch_intent.get('launch_mutation_allowed'), bool) else 'unknown'} "
            f"ready={int(execution_supervisor_dispatch_intent.get('ready_candidate_count') or 0)} "
            f"blocked={int(execution_supervisor_dispatch_intent.get('blocked_candidate_count') or 0)} "
            f"launch_readiness={execution_supervisor_dispatch_intent.get('launch_readiness_state') or 'unknown'} "
            f"severity={execution_supervisor_dispatch_intent.get('launch_readiness_severity_state') or 'none'} "
            f"severity_ticks={int(execution_supervisor_dispatch_intent.get('launch_readiness_severity_non_ready_ticks') or 0)}/{int(execution_supervisor_dispatch_intent.get('launch_readiness_severity_threshold_ticks') or 0)} "
            f"probe_due_now={int(execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_due_now_worker_count') or 0)} "
            f"probe_overdue={int(execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_overdue_worker_count') or 0)} "
            f"probe_due_age_sec={execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_oldest_due_now_age_sec') if execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_oldest_due_now_age_sec') is not None else 'n/a'} "
            f"probe_overdue_age_sec={execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_oldest_overdue_age_sec') if execution_supervisor_dispatch_intent.get('launch_readiness_probe_execution_oldest_overdue_age_sec') is not None else 'n/a'} "
            f"restore_pending={int(execution_supervisor_dispatch_intent.get('launch_readiness_restore_pending_worker_count') or 0)} "
            f"demoted={int(execution_supervisor_dispatch_intent.get('launch_readiness_demoted_worker_count') or 0)} "
            f"restored={int(execution_supervisor_dispatch_intent.get('launch_readiness_restored_worker_count') or 0)} "
            f"demotion_priority={execution_supervisor_dispatch_intent.get('launch_readiness_demotion_action_priority') or 'none'} "
            f"demotion_pending_age_sec={execution_supervisor_dispatch_intent.get('launch_readiness_oldest_restore_pending_age_sec') if execution_supervisor_dispatch_intent.get('launch_readiness_oldest_restore_pending_age_sec') is not None else 'n/a'} "
            f"demotion_oldest_age_sec={execution_supervisor_dispatch_intent.get('launch_readiness_oldest_demoted_age_sec') if execution_supervisor_dispatch_intent.get('launch_readiness_oldest_demoted_age_sec') is not None else 'n/a'} "
            f"latest_restore_age_sec={execution_supervisor_dispatch_intent.get('launch_readiness_latest_restored_age_sec') if execution_supervisor_dispatch_intent.get('launch_readiness_latest_restored_age_sec') is not None else 'n/a'} "
            f"reasons={','.join(execution_supervisor_dispatch_intent.get('decision_reasons') or []) or 'none'}"
        ),
        "freshness": execution_supervisor_dispatch_intent.get("generated_at") or current.get("generated_at") or now_obj.get("generated_at"),
        "owner": "execution_supervisor",
        "next_safe_action": cmd_read_execution_supervisor_dispatch_intent_json,
    },
    {
        "id": "execution_supervisor_dispatch_qualification",
        "state": (
            f"source_present={execution_supervisor_dispatch_qualification.get('source_present') is True} "
            f"status={execution_supervisor_dispatch_qualification.get('status') or 'missing'} "
            f"decision={execution_supervisor_dispatch_qualification.get('decision') or 'missing'} "
            f"fail_closed={execution_supervisor_dispatch_qualification.get('fail_closed') if isinstance(execution_supervisor_dispatch_qualification.get('fail_closed'), bool) else 'unknown'} "
            f"launch_mutation_allowed={execution_supervisor_dispatch_qualification.get('launch_mutation_allowed') if isinstance(execution_supervisor_dispatch_qualification.get('launch_mutation_allowed'), bool) else 'unknown'} "
            f"ready={int(execution_supervisor_dispatch_qualification.get('ready_candidate_count') or 0)} "
            f"qualified={int(execution_supervisor_dispatch_qualification.get('qualified_candidate_count') or 0)} "
            f"blocked={int(execution_supervisor_dispatch_qualification.get('blocked_candidate_count') or 0)} "
            f"launch_readiness={execution_supervisor_dispatch_qualification.get('launch_readiness_state') or 'unknown'} "
            f"severity={execution_supervisor_dispatch_qualification.get('launch_readiness_severity_state') or 'none'} "
            f"severity_ticks={int(execution_supervisor_dispatch_qualification.get('launch_readiness_severity_non_ready_ticks') or 0)}/{int(execution_supervisor_dispatch_qualification.get('launch_readiness_severity_threshold_ticks') or 0)} "
            f"probe_due_now={int(execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_due_now_worker_count') or 0)} "
            f"probe_overdue={int(execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_overdue_worker_count') or 0)} "
            f"probe_due_age_sec={execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_oldest_due_now_age_sec') if execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_oldest_due_now_age_sec') is not None else 'n/a'} "
            f"probe_overdue_age_sec={execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_oldest_overdue_age_sec') if execution_supervisor_dispatch_qualification.get('launch_readiness_probe_execution_oldest_overdue_age_sec') is not None else 'n/a'} "
            f"restore_pending={int(execution_supervisor_dispatch_qualification.get('launch_readiness_restore_pending_worker_count') or 0)} "
            f"demoted={int(execution_supervisor_dispatch_qualification.get('launch_readiness_demoted_worker_count') or 0)} "
            f"restored={int(execution_supervisor_dispatch_qualification.get('launch_readiness_restored_worker_count') or 0)} "
            f"demotion_priority={execution_supervisor_dispatch_qualification.get('launch_readiness_demotion_action_priority') or 'none'} "
            f"demotion_pending_age_sec={execution_supervisor_dispatch_qualification.get('launch_readiness_oldest_restore_pending_age_sec') if execution_supervisor_dispatch_qualification.get('launch_readiness_oldest_restore_pending_age_sec') is not None else 'n/a'} "
            f"demotion_oldest_age_sec={execution_supervisor_dispatch_qualification.get('launch_readiness_oldest_demoted_age_sec') if execution_supervisor_dispatch_qualification.get('launch_readiness_oldest_demoted_age_sec') is not None else 'n/a'} "
            f"latest_restore_age_sec={execution_supervisor_dispatch_qualification.get('launch_readiness_latest_restored_age_sec') if execution_supervisor_dispatch_qualification.get('launch_readiness_latest_restored_age_sec') is not None else 'n/a'} "
            f"reasons={','.join(execution_supervisor_dispatch_qualification.get('decision_reasons') or []) or 'none'}"
        ),
        "freshness": execution_supervisor_dispatch_qualification.get("generated_at") or current.get("generated_at") or now_obj.get("generated_at"),
        "owner": "execution_supervisor",
        "next_safe_action": cmd_read_execution_supervisor_dispatch_qualification_json,
    },
    {
        "id": "execution_frontier_controller_tick",
        "state": (
            f"status={execution_status.get('autonomous_dispatch_status') or 'missing'} "
            f"decision={execution_status.get('autonomous_dispatch_decision') or 'none'} "
            f"dispatch_decision={autopilot_execution_frontier_controller_dispatch_decision or 'none'} "
            f"dispatch_advance_applied={autopilot_execution_frontier_controller_dispatch_advance_applied} "
            f"post_completion_required={execution_status.get('autonomous_dispatch_post_completion_enforcement_required') is True} "
            f"post_completion_latched={execution_status.get('autonomous_dispatch_post_completion_enforcement_latched') is True} "
            f"loop_state={execution_status.get('autonomous_dispatch_post_completion_loop_state') or 'none'} "
            f"retry_state={((execution_status.get('autonomous_dispatch_retry_contract') or {}).get('state') if isinstance(execution_status.get('autonomous_dispatch_retry_contract'), dict) else 'none')} "
            f"cooldown_active={((execution_status.get('autonomous_dispatch_cooldown_policy') or {}).get('active') if isinstance(execution_status.get('autonomous_dispatch_cooldown_policy'), dict) else False)} "
            f"parity={((execution_status.get('autonomous_dispatch_queue_truth_vs_narrative_parity') or {}).get('status') if isinstance(execution_status.get('autonomous_dispatch_queue_truth_vs_narrative_parity'), dict) else 'unknown')} "
            f"intent_active={execution_status.get('autonomous_dispatch_intent_active') is True} "
            f"skip_reason={execution_status.get('autonomous_dispatch_skip_reason') or 'none'} "
            f"block_reason={execution_status.get('autonomous_dispatch_block_reason') or 'none'} "
            f"error={execution_status.get('autonomous_dispatch_error') or 'none'} "
            f"source_degraded={execution_status.get('autonomous_dispatch_source_degraded') is True}"
        ),
        "freshness": execution_status.get("autonomous_dispatch_updated_at") or now_obj.get("generated_at") or current.get("generated_at"),
        "owner": "run_no_nudge_continuity_watchdog",
        "next_safe_action": (
            cmd_watchdog_json
            if (
                autopilot_execution_frontier_controller_blocked
                or autopilot_execution_frontier_controller_error_state
                or autopilot_execution_frontier_controller_post_completion_stalled
                or autopilot_execution_frontier_controller_source_degraded
            )
            else cmd_read_execution_frontier_controller_trace_json
        ),
    },
    {
        "id": "action_token",
        "state": "ready" if action_token else "missing",
        "freshness": current.get("generated_at"),
        "owner": "truth_anchor_guard",
        "next_safe_action": cmd_cont_current_refresh_json,
    },
    {
        "id": "surface_freshness_contract",
        "state": (
            f"failclose={len(freshness_failclose_reasons)} "
            f"warning={len(freshness_warning_reasons)}"
        ),
        "freshness": now_obj.get("generated_at") or current.get("generated_at"),
        "owner": "operator_mission_control",
        "next_safe_action": cmd_cont_mission_refresh_json,
    },
    {
        "id": "generation_pointer_contract",
        "state": (
            f"present={bool(generation_pointer.get('present'))} "
            f"failclose={len(generation_failclose_reasons)} "
            f"current_gen={generation_pointer.get('current_generation_id') or 'n/a'} "
            f"pointer_gen={generation_pointer.get('pointer_generation_id') or 'n/a'}"
        ),
        "freshness": generation_pointer.get("pointer_current_generated_at") or current.get("generated_at"),
        "owner": "continuity_read_contract",
        "next_safe_action": cmd_cont_current_refresh_json,
    },
    {
        "id": "successor_proof_gate",
        "state": (
            f"enforced={proof_gate_enforced} "
            f"proof_state={proof_state} "
            f"top_blocker={proof_top_blocker or 'none'}"
        ),
        "freshness": handover.get("generated_at") or current.get("generated_at"),
        "owner": "successor_proof",
        "next_safe_action": cmd_read_proof_status_json,
    },
    {
        "id": "reset_ready_refresh",
        "state": (
            f"present={reset_ready_refresh_present} "
            f"ok={reset_ready_refresh_ok if reset_ready_refresh_ok is not None else 'unknown'} "
            f"phase={reset_ready_refresh_phase or 'unknown'} "
            f"error={reset_ready_refresh_error_code or 'none'} "
            f"partial_current={reset_ready_refresh_partial_current if reset_ready_refresh_partial_current is not None else 'unknown'} "
            f"partial_proof={reset_ready_refresh_partial_proof if reset_ready_refresh_partial_proof is not None else 'unknown'} "
            f"partial_handover={reset_ready_refresh_partial_handover if reset_ready_refresh_partial_handover is not None else 'unknown'}"
        ),
        "freshness": reset_ready_refresh_latest.get("generated_at") or now_obj.get("generated_at") or current.get("generated_at"),
        "owner": "reset_ready_refresh",
        "next_safe_action": cmd_cont_reset_ready_refresh_json if reset_ready_refresh_degraded else cmd_read_reset_ready_refresh_latest_json,
    },
    {
        "id": "queue_health",
        "state": f"running={contention.get('running_tasks')} active_locks={contention.get('active_locks')}",
        "freshness": now_obj.get("generated_at"),
        "owner": "queue_arbitrator",
        "next_safe_action": cmd_queue_remediate_json,
    },
    {
        "id": "queue_replay_projection",
        "state": (queue_replay.get("summary") or {}).get("status"),
        "freshness": queue_replay.get("generated_at"),
        "owner": "queue_replay_verify",
        "next_safe_action": cmd_cont_queue_replay_strict_json,
    },
    {
        "id": "wave2_replay_evidence",
        "state": (
            f"verdict={wave2_replay_verdict} "
            f"fresh={wave2_replay_fresh} "
            f"scenarios={wave2_replay_scenario_count} "
            f"soak_runs={wave2_replay_soak_runs_total} "
            f"soak_drift={wave2_replay_soak_drift_detected}"
        ),
        "freshness": wave2_replay_generated_at,
        "owner": "failover_replay_evidence",
        "next_safe_action": cmd_read_wave2_replay_evidence_index_json,
    },
    {
        "id": "failover_stress_soak",
        "state": (
            f"verdict={failover_stress_soak_verdict} "
            f"fresh={failover_stress_soak_fresh} "
            f"profiles={failover_stress_soak_profile_count} "
            f"cycles={failover_stress_soak_total_cycles} "
            f"convergence_fail={failover_stress_soak_convergence_fail_count} "
            f"drift={failover_stress_soak_drift_detected}"
        ),
        "freshness": failover_stress_soak_generated_at,
        "owner": "failover_stress_soak",
        "next_safe_action": cmd_read_failover_stress_soak_evidence_json,
    },
    {
        "id": "failover_stress_runtime_evidence",
        "state": (
            f"verdict={failover_stress_runtime_verdict} "
            f"publish={failover_stress_runtime_publish_verdict} "
            f"fresh={failover_stress_runtime_fresh} "
            f"assertions_failed={failover_stress_runtime_publish_assertions_failed} "
            f"repeatability={failover_stress_runtime_repeatability_status} "
            f"repeatability_match={failover_stress_runtime_repeatability_match if failover_stress_runtime_repeatability_match is not None else 'n/a'} "
            f"active_top_blocker={failover_stress_runtime_active_top_blocker or 'none'} "
            f"effective_top_blocker={failover_stress_runtime_effective_top_blocker or 'none'}"
        ),
        "freshness": failover_stress_runtime_generated_at,
        "owner": "failover_stress_runtime_evidence",
        "next_safe_action": cmd_read_failover_stress_runtime_evidence_json,
    },
    {
        "id": "gtc_handoff_binding_integrity",
        "state": (
            f"enabled={bool(gtc_now.get('enabled'))} "
            f"degraded={gtc_handoff_binding_degraded} "
            f"warnings={len(gtc_warning_reasons)}"
        ),
        "freshness": gtc_now.get("generated_at") or now_obj.get("generated_at"),
        "owner": "gtc_v2_sync",
        "next_safe_action": gtc_handoff_binding_next_safe_action,
    },
    {
        "id": "degraded_pending_backlog",
        "state": (
            f"active={bool(autopilot_degraded_pending_signal.get('active'))} "
            f"stale={int(autopilot_degraded_pending_signal.get('pending_stale_count') or 0)} "
            f"pending={int(autopilot_degraded_pending_signal.get('pending_total') or 0)}"
        ),
        "freshness": autopilot_degraded_pending_signal.get("last_emit_iso") or autopilot_degraded_pending_signal.get("active_since_iso") or now_obj.get("generated_at"),
        "owner": "queue_sync_from_autopilot_json",
        "next_safe_action": normalize_operator_command(autopilot_degraded_pending_signal.get("recovery_command") or cmd_cont_queue_sync_json),
    },
    {
        "id": "queue_stale_wave",
        "state": (
            f"active={bool(queue_stale_wave_signal.get('active'))} "
            f"reason={queue_stale_wave_signal.get('reason') or 'n/a'} "
            f"ready={int(queue_stale_wave_signal.get('ready_count') or 0)} "
            f"oldest_age={int(queue_stale_wave_signal.get('ready_oldest_age_sec') or 0)}"
        ),
        "freshness": queue_stale_wave_signal.get("ready_oldest_updated_at") or now_obj.get("generated_at"),
        "owner": "queue_arbitrator",
        "next_safe_action": normalize_operator_command(queue_stale_wave_signal.get("inspect_command") or cmd_queue_ready_list_json),
    },
    {
        "id": "load_shedding_lane",
        "state": (
            f"lane={load_shedding_lane_state} "
            f"warning={load_shedding_warning_tier} "
            f"critical={load_shedding_critical_tier} "
            f"escape={load_shedding_escape_triggered} "
            f"trigger={load_shedding_trigger_emitted or 'none'} "
            f"source_degraded={load_shedding_source_degraded}"
        ),
        "freshness": load_shedding_projection.get("evaluated_at") or now_obj.get("generated_at"),
        "owner": "load_shedding_policy",
        "next_safe_action": cmd_read_load_shedding_decision_json,
    },
    {
        "id": "idle_lane_autospawn",
        "state": (
            f"status={autopilot_idle_lane_status} "
            f"stalled={autopilot_idle_lane_stalled} "
            f"contradiction_latched={autopilot_idle_lane_contradiction_latched} "
            f"ready_work={autopilot_idle_lane_ready_work_exists} "
            f"idle_exceeded={autopilot_idle_lane_idle_threshold_exceeded} "
            f"idle_sec={int(autopilot_idle_lane_autospawn.get('idle_sec') or 0)} "
            f"abort_remaining_sec={autopilot_idle_lane_contradiction_abort_remaining_sec} "
            f"latch_repaired={autopilot_idle_lane_contradiction_latch_repaired} "
            f"latch_repair_reason={autopilot_idle_lane_contradiction_latch_repair_reason or 'none'}"
        ),
        "freshness": autopilot_idle_lane_autospawn.get("updated_at") or now_obj.get("generated_at"),
        "owner": "run_no_nudge_continuity_watchdog",
        "next_safe_action": cmd_watchdog_json,
    },
    {
        "id": "validators",
        "state": verify.get("status"),
        "freshness": verify.get("timestamp"),
        "owner": "verify",
        "next_safe_action": cmd_cont_verify_json,
    },
    {
        "id": "verify_gate_preflight",
        "state": (
            f"enabled={bool(verify_gate_preflight_strict.get('enabled') is True)} "
            f"source={str(verify_gate_preflight_strict.get('source') or 'disabled')} "
            f"required={verify_gate_preflight_strict.get('required') if verify_gate_preflight_strict.get('required') is not None else 'n/a'} "
            f"override={verify_gate_preflight_strict.get('override') or 'none'} "
            f"blocker={verify_gate_preflight_predicted.get('predicted_blocker_reason') or 'none'} "
            f"status_evidence={verify_status_evidence_failure_reason or 'ok'} "
            f"layered_health={verify_layered_health_failure_reason or ('ready' if verify_layered_health_closeout_ready is True else 'unknown')} "
            f"probe_gate={verify_probe_execution_gate_failure_reason or ('active' if verify_probe_execution_gate_active_blocker else 'ok')} "
            f"probe_due_now={verify_probe_execution_gate_due_now_worker_count} "
            f"probe_overdue={verify_probe_execution_gate_overdue_worker_count} "
            f"probe_due_age_sec={verify_probe_execution_gate_oldest_due_now_age_sec if verify_probe_execution_gate_oldest_due_now_age_sec is not None else 'n/a'} "
            f"probe_overdue_age_sec={verify_probe_execution_gate_oldest_overdue_age_sec if verify_probe_execution_gate_oldest_overdue_age_sec is not None else 'n/a'} "
            f"demotion_pending={verify_probe_execution_gate_demotion_restore_pending_worker_count} "
            f"demotion_priority={verify_probe_execution_gate_demotion_action_priority or 'none'} "
            f"demotion_pending_age_sec={verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec if verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec is not None else 'n/a'} "
            f"internal_bypass={verify_internal_bypass_closeout_failure_reason or ('ready' if verify_internal_bypass_closeout_ready is True else 'unknown')} "
            f"unknown_total={verify_internal_bypass_unknown_total} "
            f"break_glass_allow={verify_internal_bypass_break_glass_allow} "
            f"break_glass_denied={verify_internal_bypass_break_glass_denied}"
        ),
        "freshness": verify_gate_preflight.get("generated_at") or verify.get("timestamp") or now_obj.get("generated_at"),
        "owner": "verify_gate_status",
        "next_safe_action": cmd_cont_verify_gate_status_json,
    },
    {
        "id": "layered_health_gate",
        "state": (
            f"closeout_ready={verify_layered_health_closeout_ready} "
            f"failure={verify_layered_health_failure_reason or 'none'} "
            f"health_status={verify_layered_health_status} "
            f"health_layer={verify_layered_health_layer} "
            f"restore_slo={verify_layered_health_restore_slo_status} "
            f"missing={len(verify_layered_health_missing_required_lanes)} "
            f"failing={len(verify_layered_health_failing_required_lanes)} "
            f"layer_insufficient={len(verify_layered_health_layer_insufficient_required_lanes)}"
        ),
        "freshness": verify_gate_preflight.get("generated_at") or verify.get("timestamp") or now_obj.get("generated_at"),
        "owner": "layered_health_snapshot",
        "next_safe_action": normalize_operator_command(
            verify_gate_preflight_layered_health.get("inspect_layered_health_command")
            or verify_gate_preflight_layered_health.get("run_layered_health_command")
            or cmd_cont_verify_gate_status_json
        ),
    },
    {
        "id": "effective_routing",
        "state": (
            f"source={routing_preflight_source} "
            f"failure={routing_failure_reason or 'none'} "
            f"decision={routing_decision or 'unknown'} "
            f"route_class={routing_route_class or 'none'} "
            f"model={routing_selected_model or 'none'} "
            f"blocker={routing_block_reason or 'none'}"
        ),
        "freshness": routing_preflight_latest.get("evaluated_at") or now_obj.get("generated_at"),
        "owner": "session_topology_router",
        "next_safe_action": routing_next_safe_action,
    },
    {
        "id": "model_rollout_action_card",
        "state": (
            f"status={model_rollout_dashboard_status} "
            f"prompt={model_rollout_prompt_status} "
            f"reason={model_rollout_prompt_reason} "
            f"requires_approval={model_rollout_prompt_requires_approval} "
            f"commands={len(model_rollout_prompt_commands)} "
            f"scorecard={model_rollout_prompt_scorecard_ref or 'none'}"
        ),
        "freshness": model_rollout_dashboard.get("generated_at"),
        "owner": "model_rollout_dashboard",
        "next_safe_action": model_rollout_next_safe_action,
    },
    {
        "id": "model_rollout_operator_mistake_remediation",
        "state": (
            f"active={model_rollout_remediation_active} "
            f"status={model_rollout_remediation_status} "
            f"gate={model_rollout_remediation_reason_gate} "
            f"reason={model_rollout_remediation_reason_code} "
            f"commands={len(model_rollout_remediation_commands)} "
            f"log={model_rollout_remediation_correction_cycle_log_ref or 'none'}"
        ),
        "freshness": model_rollout_operator_mistake_remediation.get("recorded_at") or model_rollout_dashboard.get("generated_at"),
        "owner": "model_rollout_dashboard",
        "next_safe_action": model_rollout_remediation_first_command or cmd_read_model_rollout_dashboard_json,
    },
    {
        "id": "efficiency_kpi_baseline",
        "state": (
            f"status={efficiency_kpi_status} "
            f"validation={efficiency_kpi_validation_status} "
            f"kpis={efficiency_kpi_kpi_count} "
            f"measured={efficiency_kpi_measured_count} "
            f"no_signal={efficiency_kpi_no_signal_count}"
        ),
        "freshness": efficiency_kpi_baseline.get("generated_at") or efficiency_kpi_validation.get("generated_at"),
        "owner": "xe101_efficiency_kpi_baseline",
        "next_safe_action": efficiency_kpi_next_safe_action,
    },
    {
        "id": "evidence_trace_viewer",
        "state": (
            f"status={evidence_trace_viewer.get('status') or 'degraded'} "
            f"focus={evidence_trace_viewer.get('focus_task_id') or 'none'} "
            f"slice={evidence_trace_viewer.get('focus_slice_id') if isinstance(evidence_trace_viewer.get('focus_slice_id'), int) else 'n/a'} "
            f"trace={int(evidence_trace_viewer.get('trace_count') or 0)} "
            f"blocked={int(evidence_trace_viewer.get('blocked_reason_count') or 0)} "
            f"artifacts={int(evidence_trace_viewer.get('artifact_ref_count') or 0)}"
        ),
        "freshness": evidence_trace_viewer.get("generated_at") or now_obj.get("generated_at") or current.get("generated_at"),
        "owner": "b5_evidence_os",
        "next_safe_action": cmd_read_evidence_trace_viewer_json,
    },
    {
        "id": "parity", 
        "state": parity.get("status"),
        "freshness": parity.get("last_done_at"),
        "owner": "parity_harness",
        "next_safe_action": cmd_parity_dry_run_json,
    },
    {
        "id": "browser_artifact_pipeline",
        "state": browser_latest.get("status"),
        "freshness": browser_latest.get("captured_at"),
        "owner": "web_capture",
        "next_safe_action": cmd_web_capture_auto_json,
    },
    {
        "id": "web_domain_guard",
        "state": (
            f"domains={web_domain_guard.get('tracked_domains')} "
            f"cooldown={web_domain_guard.get('cooldown_active_domains')} "
            f"operator={web_domain_guard.get('operator_action_required_domains')} "
            f"actionable={web_domain_guard.get('actionable_incident_domains')}"
        ),
        "freshness": web_domain_guard.get("latest_updated_at"),
        "owner": "web_capture_domain_guard",
        "next_safe_action": cmd_cont_mission_refresh_json,
    },
    {
        "id": "web_scheduler",
        "state": (
            f"status={web_scheduler.get('selection_status') or 'unknown'} "
            f"eligible={web_scheduler.get('eligible_macros')}/{web_scheduler.get('total_macros')} "
            f"fresh={web_scheduler.get('fresh') if web_scheduler.get('fresh') is not None else 'n/a'} "
            f"contract_valid={web_scheduler.get('contract_state_valid') if web_scheduler.get('contract_state_valid') is not None else 'n/a'}"
        ),
        "freshness": web_scheduler.get("updated_at"),
        "owner": "web_capture_scheduler",
        "next_safe_action": cmd_web_capture_scheduler_dry_json,
    },
]

if continuity_current_publish_lock.get("surface_active") is True:
    publish_lock_row = {
        "id": "continuity_current_publish_lock",
        "state": (
            f"source={continuity_current_publish_lock.get('source') or 'unknown'} "
            f"status={continuity_current_publish_lock.get('status') or 'unknown'} "
            f"timeout={continuity_current_publish_lock.get('timeout_detected') is True} "
            f"wait_sec={continuity_current_publish_lock.get('wait_sec') if continuity_current_publish_lock.get('wait_sec') is not None else 'unknown'} "
            f"owner_pid={continuity_current_publish_lock.get('owner_pid') if continuity_current_publish_lock.get('owner_pid') is not None else 'unknown'} "
            f"owner_alive={continuity_current_publish_lock.get('owner_alive') if isinstance(continuity_current_publish_lock.get('owner_alive'), bool) else 'unknown'} "
            f"owner_age_sec={continuity_current_publish_lock.get('owner_age_sec') if continuity_current_publish_lock.get('owner_age_sec') is not None else 'unknown'} "
            f"hold_warn_sec={continuity_current_publish_lock.get('lock_hold_warn_sec') if continuity_current_publish_lock.get('lock_hold_warn_sec') is not None else 'unknown'} "
            f"exceeds_wait_budget={continuity_current_publish_lock.get('owner_exceeds_wait_budget') if isinstance(continuity_current_publish_lock.get('owner_exceeds_wait_budget'), bool) else 'unknown'} "
            f"exceeds_hold_warn={continuity_current_publish_lock.get('owner_exceeds_lock_hold_warn') if isinstance(continuity_current_publish_lock.get('owner_exceeds_lock_hold_warn'), bool) else 'unknown'} "
            f"source_degraded={continuity_current_publish_lock.get('source_degraded') is True} "
            f"source_fresh={continuity_current_publish_lock.get('source_current_fresh') if isinstance(continuity_current_publish_lock.get('source_current_fresh'), bool) else 'unknown'} "
            f"source_matches_current={continuity_current_publish_lock.get('source_current_matches_current_generated_at') if isinstance(continuity_current_publish_lock.get('source_current_matches_current_generated_at'), bool) else 'unknown'} "
            f"warnings={len(_reason_list(continuity_current_publish_lock.get('warning_reasons')))} "
            f"path={continuity_current_publish_lock.get('path') or 'unknown'}"
        ),
        "freshness": continuity_current_publish_lock.get("generated_at") or current.get("generated_at"),
        "owner": "blocker_registry" if continuity_current_publish_lock.get("source") == "blocker_registry" else "continuity_current",
        "next_safe_action": continuity_current_publish_lock.get("inspect_command") or cmd_read_current_publish_lock_owner_json,
    }
    surface_idx = next((idx for idx, row in enumerate(truth_strip) if str((row or {}).get("id") or "") == "surface_freshness_contract"), -1)
    if surface_idx >= 0:
        truth_strip.insert(surface_idx + 1, publish_lock_row)
    else:
        truth_strip.append(publish_lock_row)

truth_row_stale_reasons: Dict[str, str] = {}

for row_id in ["overall_ready_state", "mutation_gate", "action_token", "surface_freshness_contract"]:
    if freshness_failclose_reasons and row_id not in truth_row_stale_reasons:
        truth_row_stale_reasons[row_id] = freshness_failclose_reasons[0]
if generation_failclose_reasons and "generation_pointer_contract" not in truth_row_stale_reasons:
    truth_row_stale_reasons["generation_pointer_contract"] = generation_failclose_reasons[0]
if proof_failclose_reasons and "successor_proof_gate" not in truth_row_stale_reasons:
    truth_row_stale_reasons["successor_proof_gate"] = proof_failclose_reasons[0]
if _contract_source_degraded(queue_replay) and "queue_replay_projection" not in truth_row_stale_reasons:
    truth_row_stale_reasons["queue_replay_projection"] = "queue_replay_source_degraded"
if continuity_current_publish_lock.get("source_degraded") is True and "continuity_current_publish_lock" not in truth_row_stale_reasons:
    publish_lock_source_degraded_reasons = _reason_list(continuity_current_publish_lock.get("source_degraded_reasons"))
    truth_row_stale_reasons["continuity_current_publish_lock"] = (
        publish_lock_source_degraded_reasons[0]
        if publish_lock_source_degraded_reasons
        else "blocker_registry_publish_lock_source_degraded"
    )
if wave2_replay_evidence_source_degraded and "wave2_replay_evidence" not in truth_row_stale_reasons:
    truth_row_stale_reasons["wave2_replay_evidence"] = "wave2_replay_evidence_source_degraded"
elif isinstance(wave2_replay_evidence_obj, dict) and not wave2_replay_fresh and "wave2_replay_evidence" not in truth_row_stale_reasons:
    truth_row_stale_reasons["wave2_replay_evidence"] = "wave2_replay_evidence_stale"
if failover_stress_soak_source_degraded and "failover_stress_soak" not in truth_row_stale_reasons:
    truth_row_stale_reasons["failover_stress_soak"] = "failover_stress_soak_source_degraded"
elif isinstance(failover_stress_soak_obj, dict) and not failover_stress_soak_fresh and "failover_stress_soak" not in truth_row_stale_reasons:
    truth_row_stale_reasons["failover_stress_soak"] = "failover_stress_soak_stale"
if failover_stress_runtime_source_degraded and "failover_stress_runtime_evidence" not in truth_row_stale_reasons:
    truth_row_stale_reasons["failover_stress_runtime_evidence"] = "failover_stress_runtime_evidence_source_degraded"
elif isinstance(failover_stress_runtime_obj, dict) and not failover_stress_runtime_fresh and "failover_stress_runtime_evidence" not in truth_row_stale_reasons:
    truth_row_stale_reasons["failover_stress_runtime_evidence"] = "failover_stress_runtime_evidence_stale"
if load_shedding_source_degraded and "load_shedding_lane" not in truth_row_stale_reasons:
    truth_row_stale_reasons["load_shedding_lane"] = "load_shedding_source_degraded"
if bool(parity.get("due") is True) and "parity" not in truth_row_stale_reasons:
    truth_row_stale_reasons["parity"] = "parity_weekly_freshness_due"
if routing_failure_reason == "routing_decision_stale" and "effective_routing" not in truth_row_stale_reasons:
    truth_row_stale_reasons["effective_routing"] = "routing_decision_stale"
if verify_layered_health_failure_reason and "layered_health_gate" not in truth_row_stale_reasons:
    truth_row_stale_reasons["layered_health_gate"] = f"layered_health_gate_{verify_layered_health_failure_reason}"
if model_rollout_dashboard_status in {"error", "unknown"} and "model_rollout_action_card" not in truth_row_stale_reasons:
    truth_row_stale_reasons["model_rollout_action_card"] = "model_rollout_dashboard_unavailable"
if efficiency_kpi_status in {"missing", "degraded"} and "efficiency_kpi_baseline" not in truth_row_stale_reasons:
    truth_row_stale_reasons["efficiency_kpi_baseline"] = "efficiency_kpi_baseline_unavailable"
if efficiency_kpi_validation_status in {"missing", "fail"} and "efficiency_kpi_baseline" not in truth_row_stale_reasons:
    truth_row_stale_reasons["efficiency_kpi_baseline"] = "efficiency_kpi_validation_failed"
if evidence_trace_viewer.get("status") != "ready" and "evidence_trace_viewer" not in truth_row_stale_reasons:
    source_reasons = _reason_list(evidence_trace_viewer.get("source_degraded_reasons"))
    truth_row_stale_reasons["evidence_trace_viewer"] = source_reasons[0] if source_reasons else "evidence_trace_viewer_degraded"
if web_scheduler.get("fresh") is False and "web_scheduler" not in truth_row_stale_reasons:
    truth_row_stale_reasons["web_scheduler"] = "web_scheduler_stale"
if meaningful_event_reporting_projection.get("attention_required") is True and "meaningful_event_reporting" not in truth_row_stale_reasons:
    truth_row_stale_reasons["meaningful_event_reporting"] = (
        (meaningful_event_reporting_projection.get("attention_reasons") or ["meaningful_event_reporting_attention_required"])[0]
    )
if meaningful_event_contract_failclose_reasons and "meaningful_event_reporting_contract" not in truth_row_stale_reasons:
    truth_row_stale_reasons["meaningful_event_reporting_contract"] = (
        meaningful_event_contract_failclose_digest_reason
        or meaningful_event_contract_failclose_reasons[0]
    )
elif meaningful_event_contract_warning_reasons and "meaningful_event_reporting_contract" not in truth_row_stale_reasons:
    truth_row_stale_reasons["meaningful_event_reporting_contract"] = (
        meaningful_event_contract_warning_digest_reason
        or meaningful_event_contract_warning_reasons[0]
    )

truth_strip = [
    _truth_row_with_freshness_posture(
        row,
        stale_reason=truth_row_stale_reasons.get(str((row or {}).get("id") or "")),
    )
    for row in truth_strip
]

freshness_stale_count = sum(1 for row in truth_strip if str(row.get("freshness_posture") or "") == "stale")
freshness_unknown_count = sum(
    1
    for row in truth_strip
    if str(row.get("freshness_posture") or "") == "unknown" and str(row.get("freshness") or "").strip()
)
freshness_posture = "stale" if freshness_stale_count > 0 else ("unknown" if freshness_unknown_count > 0 else "fresh")

warning_reasons = unique_preserve(warning_reasons)
not_ready_reasons = unique_preserve(not_ready_reasons)

actions: List[Dict[str, Any]] = []
if mutation_gate_status != "allowed":
    actions.append({
        "priority": "p0",
        "action": "stay_read_only",
        "command": "export SUCCESSOR_MODE=1 MUTATION_ALLOWED=0",
    })
if generation_failclose_reasons:
    actions.append({
        "priority": "p0",
        "action": "refresh_generation_pointer_contract",
        "command": cmd_cont_current_refresh_json,
    })
    actions.append({
        "priority": "p1",
        "action": "inspect_generation_pointer_contract",
        "command": cmd_read_pointer_json,
    })
if freshness_failclose_reasons:
    actions.append({
        "priority": "p0",
        "action": "refresh_degraded_operator_sources",
        "command": cmd_cont_mission_refresh_json,
    })
actions.append({
    "priority": "p1" if evidence_trace_viewer.get("status") != "ready" else "p2",
    "action": "inspect_evidence_trace_viewer",
    "command": cmd_read_evidence_trace_viewer_json,
})
if meaningful_event_reporting_projection.get("attention_required") is True:
    actions.append({
        "priority": "p1" if int(meaningful_event_reporting_projection.get("critical_pending_event_count") or 0) > 0 else "p2",
        "action": "inspect_execution_meaningful_event_reporting_status",
        "command": cmd_read_execution_meaningful_event_reporting_status_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_meaningful_event_reporting_packet",
        "command": cmd_read_execution_meaningful_event_reporting_json,
    })
if meaningful_event_contract_failclose_reasons:
    actions.append({
        "priority": "p0",
        "action": "rebuild_execution_meaningful_event_reporting_status_surface",
        "command": cmd_cont_current_refresh_json,
    })
    actions.append({
        "priority": "p1",
        "action": "inspect_execution_meaningful_event_reporting_contract",
        "command": cmd_read_execution_meaningful_event_reporting_status_json,
    })
elif meaningful_event_contract_warning_reasons:
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_meaningful_event_reporting_contract",
        "command": cmd_read_execution_meaningful_event_reporting_status_json,
    })
if continuity_current_publish_lock.get("surface_active") is True:
    actions.append({
        "priority": "p1",
        "action": "inspect_current_publish_lock_owner",
        "command": continuity_current_publish_lock.get("inspect_command") or cmd_read_current_publish_lock_owner_json,
    })
if continuity_current_publish_lock.get("source_degraded") is True and not freshness_failclose_reasons:
    actions.append({
        "priority": "p2",
        "action": "refresh_publish_lock_surface_freshness",
        "command": cmd_cont_mission_refresh_json,
    })
if proof_failclose_reasons:
    actions.append({
        "priority": "p0",
        "action": "refresh_successor_proof_gate",
        "command": cmd_cont_current_refresh_json,
    })
    actions.append({
        "priority": "p1",
        "action": "inspect_successor_proof_status",
        "command": cmd_read_proof_status_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_successor_proof_artifact",
        "command": cmd_read_proof_json,
    })
if reset_ready_refresh_degraded:
    actions.append({
        "priority": "p1",
        "action": "rerun_reset_ready_refresh",
        "command": cmd_cont_reset_ready_refresh_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_reset_ready_refresh_result",
        "command": cmd_read_reset_ready_refresh_latest_json,
    })
if effective_readiness in {"NOT_READY", "RECONCILE_REQUIRED", "UNKNOWN"}:
    actions.append({
        "priority": "p0",
        "action": "reconcile_continuity",
        "command": f"{shell_cmd_for('ops/openclaw/continuity.sh')} --action-token {action_token_arg} reconcile --json",
    })
if contention.get("stale_active_locks", 0) > 0 or (contention.get("overdue_locks") or 0) > 0:
    actions.append({
        "priority": "p1",
        "action": "remediate_locks",
        "command": cmd_queue_remediate_apply_json,
    })
queue_replay_status = (queue_replay.get("summary") or {}).get("status")
if queue_replay_status == "fail":
    actions.append({
        "priority": "p1",
        "action": "reconcile_queue_journal_projection",
        "command": cmd_cont_queue_replay_strict_json,
    })
elif queue_replay_status == "warn":
    actions.append({
        "priority": "p2",
        "action": "inspect_queue_journal_discontinuities",
        "command": cmd_cont_queue_replay_json,
    })
if not isinstance(wave2_replay_evidence_obj, dict):
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_replay_evidence",
        "command": cmd_cont_failover_replay_evidence_json,
    })
elif wave2_replay_verdict != "PASS" or wave2_replay_soak_drift_detected or not wave2_replay_fresh:
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_replay_evidence",
        "command": cmd_cont_failover_replay_evidence_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_failover_replay_evidence_index",
        "command": cmd_read_wave2_replay_evidence_index_json,
    })
if not isinstance(failover_stress_soak_obj, dict):
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_stress_soak",
        "command": cmd_cont_failover_stress_soak_json,
    })
elif failover_stress_soak_verdict != "PASS" or failover_stress_soak_drift_detected or not failover_stress_soak_fresh or failover_stress_soak_convergence_fail_count > 0:
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_stress_soak",
        "command": cmd_cont_failover_stress_soak_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_failover_stress_soak_evidence",
        "command": cmd_read_failover_stress_soak_evidence_json,
    })
if not isinstance(failover_stress_runtime_obj, dict):
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_stress_runtime_evidence",
        "command": cmd_cont_failover_stress_runtime_evidence_json,
    })
elif (
    failover_stress_runtime_verdict != "PASS"
    or failover_stress_runtime_publish_verdict != "PASS"
    or not failover_stress_runtime_fresh
    or failover_stress_runtime_publish_assertions_failed > 0
    or failover_stress_runtime_repeatability_status == "mismatch"
):
    actions.append({
        "priority": "p1",
        "action": "refresh_failover_stress_runtime_evidence",
        "command": cmd_cont_failover_stress_runtime_evidence_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_failover_stress_runtime_evidence",
        "command": cmd_read_failover_stress_runtime_evidence_json,
    })
    if failover_stress_runtime_repeatability_status == "mismatch":
        actions.append({
            "priority": "p1",
            "action": "inspect_failover_stress_runtime_decision_log",
            "command": cmd_read_failover_stress_runtime_decision_log,
        })
elif failover_stress_runtime_repeatability_status in {"missing", "no_history", "previous_signature_missing"}:
    actions.append({
        "priority": "p2",
        "action": "inspect_failover_stress_runtime_decision_log",
        "command": cmd_read_failover_stress_runtime_decision_log,
    })
if gtc_handoff_binding_degraded:
    actions.append({
        "priority": "p1",
        "action": "inspect_gtc_handoff_binding_degradation",
        "command": gtc_handoff_binding_next_safe_action,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_recent_queue_handoffs",
        "command": cmd_queue_handoffs_json,
    })
if str(verify_gate_preflight_predicted.get("predicted_blocker_reason") or "").strip():
    actions.append({
        "priority": "p1",
        "action": "inspect_verify_gate_preflight",
        "command": cmd_cont_verify_gate_status_json,
    })
if (
    verify_probe_execution_gate_failure_reason
    or verify_probe_execution_gate_due_now_worker_count > 0
    or verify_probe_execution_gate_overdue_worker_count > 0
    or verify_probe_execution_gate_demotion_restore_pending_worker_count > 0
    or verify_probe_execution_gate_demotion_action_priority in {"p1", "p2"}
):
    actions.append({
        "priority": (
            verify_probe_execution_gate_action_priority
            if verify_probe_execution_gate_action_priority in {"p1", "p2"}
            else (
                verify_probe_execution_gate_demotion_action_priority
                if verify_probe_execution_gate_demotion_action_priority in {"p1", "p2"}
                else ("p1" if verify_probe_execution_gate_active_blocker or verify_probe_execution_gate_overdue_worker_count > 0 else "p2")
            )
        ),
        "action": "inspect_verify_gate_launch_readiness_probe_execution",
        "command": normalize_operator_command(
            verify_probe_execution_gate_first_actionable_command
            or verify_probe_execution_gate_inspect_probe_execution_plan_command
            or verify_probe_execution_gate_inspect_dispatch_qualification_command
            or cmd_cont_verify_gate_status_json
        ),
    })
if (
    (
        verify_probe_execution_gate_failure_reason
        or verify_probe_execution_gate_due_now_worker_count > 0
        or verify_probe_execution_gate_overdue_worker_count > 0
        or verify_probe_execution_gate_demotion_restore_pending_worker_count > 0
        or verify_probe_execution_gate_demotion_action_priority in {"p1", "p2"}
    )
    and verify_probe_execution_gate_refresh_command
):
    actions.append({
        "priority": "p2",
        "action": "refresh_verify_gate_launch_readiness_probe_execution",
        "command": verify_probe_execution_gate_refresh_command,
    })
if verify_status_evidence_failure_reason:
    actions.append({
        "priority": "p1",
        "action": "revalidate_verify_status_evidence",
        "command": cmd_cont_verify_json,
    })
if verify_internal_bypass_closeout_failure_reason or verify_internal_bypass_break_glass_allow > 0 or verify_internal_bypass_break_glass_denied > 0:
    actions.append({
        "priority": "p1",
        "action": "inspect_internal_bypass_stage_b",
        "command": cmd_cont_verify_gate_status_json,
    })
if verify_layered_health_failure_reason:
    actions.append({
        "priority": "p1",
        "action": "inspect_layered_health_gate",
        "command": normalize_operator_command(
            verify_gate_preflight_layered_health.get("inspect_layered_health_command")
            or cmd_cont_verify_gate_status_json
        ),
    })
    if normalize_operator_command(verify_gate_preflight_layered_health.get("run_slo_snapshot_command")):
        actions.append({
            "priority": "p1",
            "action": "refresh_slo_snapshot_for_layered_health",
            "command": normalize_operator_command(verify_gate_preflight_layered_health.get("run_slo_snapshot_command")),
        })
    if normalize_operator_command(verify_gate_preflight_layered_health.get("run_layered_health_command")):
        actions.append({
            "priority": "p1",
            "action": "refresh_layered_health_snapshot",
            "command": normalize_operator_command(verify_gate_preflight_layered_health.get("run_layered_health_command")),
        })
routing_actionable_failure_reasons = {
    "routing_blocked",
    "routing_decision_stale",
    "routing_decisions_unreadable",
    "routing_decisions_not_regular_file",
}
if routing_failure_reason in routing_actionable_failure_reasons:
    actions.append({
        "priority": "p1" if routing_failure_reason == "routing_blocked" else "p2",
        "action": "inspect_effective_routing",
        "command": normalize_operator_command(routing_preflight_effective.get("inspect_decisions_command") or cmd_cont_verify_gate_status_json),
    })
if routing_failure_reason in {"routing_blocked", "routing_decision_stale"}:
    actions.append({
        "priority": "p2",
        "action": "recheck_model_route_policy",
        "command": normalize_operator_command(routing_preflight_effective.get("recheck_policy_command") or cmd_cont_model_route_policy_lint_json),
    })
if routing_failure_reason == "routing_blocked" and routing_preflight_effective.get("first_actionable_command"):
    actions.append({
        "priority": "p1",
        "action": "unblock_effective_routing",
        "command": normalize_operator_command(routing_preflight_effective.get("first_actionable_command")),
    })
if model_rollout_prompt_status in {"blocked", "approval_required"}:
    actions.append({
        "priority": "p1" if model_rollout_prompt_status == "blocked" else "p2",
        "action": "inspect_model_rollout_action_card",
        "command": cmd_read_model_rollout_dashboard_json,
    })
if model_rollout_prompt_first_command:
    actions.append({
        "priority": "p1",
        "action": "run_model_rollout_action_card_command",
        "command": model_rollout_prompt_first_command,
    })
if model_rollout_prompt_status == "approval_required":
    actions.append({
        "priority": "p1",
        "action": "advance_model_rollout_controller",
        "command": cmd_cont_model_rollout_controller_json,
    })
if model_rollout_prompt_scorecard_ref:
    actions.append({
        "priority": "p2",
        "action": "inspect_model_rollout_bakeoff_scorecard",
        "command": cat_cmd_for(model_rollout_prompt_scorecard_ref),
    })
if model_rollout_remediation_active or model_rollout_remediation_status in {"active", "degraded"}:
    actions.append({
        "priority": "p1",
        "action": "inspect_model_rollout_operator_mistake_remediation",
        "command": cmd_read_model_rollout_dashboard_json,
    })
if model_rollout_remediation_first_command:
    actions.append({
        "priority": "p1",
        "action": "run_model_rollout_mistake_remediation_command",
        "command": model_rollout_remediation_first_command,
    })
if model_rollout_remediation_correction_cycle_log_ref:
    actions.append({
        "priority": "p2",
        "action": "inspect_model_rollout_correction_cycle_log",
        "command": cat_cmd_for(model_rollout_remediation_correction_cycle_log_ref),
    })
if model_rollout_dashboard_status in {"error", "unknown"} or not model_rollout_dashboard:
    actions.append({
        "priority": "p2",
        "action": "refresh_model_rollout_dashboard",
        "command": cmd_cont_model_rollout_dashboard_json,
    })
if efficiency_kpi_status in {"missing", "degraded"} or efficiency_kpi_validation_status in {"missing", "fail"}:
    actions.append({
        "priority": "p1" if efficiency_kpi_validation_status == "fail" else "p2",
        "action": "refresh_efficiency_kpi_baseline",
        "command": cmd_refresh_efficiency_kpi_baseline_json,
    })
if efficiency_kpi_baseline:
    actions.append({
        "priority": "p2",
        "action": "inspect_efficiency_kpi_baseline",
        "command": cmd_read_efficiency_kpi_baseline_json,
    })
if efficiency_kpi_validation:
    actions.append({
        "priority": "p2",
        "action": "inspect_efficiency_kpi_validation",
        "command": cmd_read_efficiency_kpi_validation_json,
    })
if bool(parity.get("due") is True) or ("parity_weekly_freshness_due" in warning_reasons):
    actions.append({
        "priority": "p2",
        "action": "refresh_parity_weekly_harness",
        "command": cmd_parity_force,
    })
if bool(autopilot_degraded_pending_signal.get("active") is True):
    recovery_cmd = normalize_operator_command(autopilot_degraded_pending_signal.get("recovery_command") or "")
    if not recovery_cmd:
        recovery_cmd = cmd_cont_queue_sync_json
    actions.append({
        "priority": "p1",
        "action": "recover_degraded_pending_backlog",
        "command": recovery_cmd,
    })
    inspect_cmd = normalize_operator_command(autopilot_degraded_pending_signal.get("inspect_command") or "")
    if inspect_cmd:
        actions.append({
            "priority": "p2",
            "action": "inspect_degraded_pending_backlog",
            "command": inspect_cmd,
        })
if bool(queue_stale_wave_signal.get("active") is True):
    wave_recovery_cmd = normalize_operator_command(queue_stale_wave_signal.get("recovery_command") or "")
    if not wave_recovery_cmd:
        wave_recovery_cmd = cmd_cont_queue_sync_json
    actions.append({
        "priority": "p1",
        "action": "recover_queue_stale_wave",
        "command": wave_recovery_cmd,
    })
    wave_inspect_cmd = normalize_operator_command(queue_stale_wave_signal.get("inspect_command") or "")
    if not wave_inspect_cmd:
        wave_inspect_cmd = cmd_queue_ready_list_json
    actions.append({
        "priority": "p2",
        "action": "inspect_queue_stale_wave",
        "command": wave_inspect_cmd,
    })
frontier_selector_state = str(execution_frontier.get("selector_state") or "unknown")
frontier_close_condition_met = execution_frontier.get("close_condition_met") is True
frontier_next_candidate = str(execution_frontier.get("next_candidate") or "").strip()
frontier_stalled = execution_frontier.get("stalled") is True
frontier_autonomous_dispatch_eligible = execution_frontier.get("autonomous_dispatch_eligible") is True
if execution_frontier.get("source_present") is True:
    if frontier_autonomous_dispatch_eligible:
        actions.append({
            "priority": "p1",
            "action": "autonomous_dispatch_execution_frontier_wave_close",
            "command": cmd_cont_execution_frontier_autonomous_dispatch_json,
        })
        actions.append({
            "priority": "p2",
            "action": "inspect_execution_frontier_supervisor",
            "command": cmd_cont_execution_frontier_show_json,
        })
    elif frontier_selector_state == "ready_for_dispatch" and frontier_close_condition_met and frontier_next_candidate:
        actions.append({
            "priority": "p1",
            "action": "advance_execution_frontier_wave_close",
            "command": cmd_cont_execution_frontier_advance_wave_close_json,
        })
        actions.append({
            "priority": "p2",
            "action": "inspect_execution_frontier_supervisor",
            "command": cmd_cont_execution_frontier_show_json,
        })
    elif frontier_selector_state in {"closed_blocked", "idle_no_candidate"} or frontier_stalled:
        actions.append({
            "priority": "p1" if frontier_stalled else "p2",
            "action": "inspect_execution_frontier_supervisor",
            "command": cmd_cont_execution_frontier_refresh_json,
        })
dispatch_intent_status = str(execution_supervisor_dispatch_intent.get("status") or "").strip()
if execution_supervisor_dispatch_intent.get("source_present") is True and dispatch_intent_status in {"dispatch_ready", "blocked", "degraded"}:
    actions.append({
        "priority": "p2" if dispatch_intent_status == "dispatch_ready" else "p1",
        "action": "inspect_execution_supervisor_dispatch_intent",
        "command": cmd_read_execution_supervisor_dispatch_intent_json,
    })
dispatch_qualification_status = str(execution_supervisor_dispatch_qualification.get("status") or "").strip()
if execution_supervisor_dispatch_qualification.get("source_present") is True and dispatch_qualification_status in {"qualified_ready", "blocked", "degraded"}:
    actions.append({
        "priority": "p2" if dispatch_qualification_status == "qualified_ready" else "p1",
        "action": "inspect_execution_supervisor_dispatch_qualification",
        "command": cmd_read_execution_supervisor_dispatch_qualification_json,
    })
dispatch_worker_health_gate_required = bool(
    execution_supervisor_dispatch_qualification.get("worker_health_gate_required") is True
)
dispatch_worker_health_canary_present = bool(
    execution_supervisor_dispatch_qualification.get("worker_health_canary_present") is True
)
verify_worker_health_canary_preflight_attention = bool(
    verify_worker_health_canary_preflight_active_blocker
    or verify_worker_health_canary_preflight_failure_reason
)
verify_worker_health_canary_preflight_priority = (
    verify_worker_health_canary_preflight_action_priority
    if verify_worker_health_canary_preflight_action_priority in {"p1", "p2"}
    else ("p1" if verify_worker_health_canary_preflight_active_blocker else "p2")
)

if verify_worker_health_canary_preflight_attention:
    if verify_worker_health_canary_preflight_refresh_command:
        actions.append({
            "priority": verify_worker_health_canary_preflight_priority,
            "action": "refresh_execution_supervisor_worker_health_canary_evidence",
            "command": verify_worker_health_canary_preflight_refresh_command,
        })
    inspect_worker_health_canary_command = (
        verify_worker_health_canary_preflight_first_actionable_command
        or verify_worker_health_canary_preflight_inspect_command
    )
    if inspect_worker_health_canary_command:
        actions.append({
            "priority": "p2",
            "action": "inspect_execution_supervisor_worker_health_canary_evidence",
            "command": inspect_worker_health_canary_command,
        })
elif execution_supervisor_dispatch_qualification.get("source_present") is True and dispatch_worker_health_gate_required:
    if dispatch_worker_health_canary_present:
        actions.append({
            "priority": "p2",
            "action": "inspect_execution_supervisor_worker_health_canary_evidence",
            "command": cmd_read_execution_supervisor_worker_health_canary_json,
        })
    else:
        actions.append({
            "priority": "p1" if dispatch_qualification_status in {"blocked", "degraded"} else "p2",
            "action": "refresh_execution_supervisor_worker_health_canary_evidence",
            "command": verify_worker_health_canary_preflight_refresh_command,
        })
if execution_supervisor_launch_readiness_severity_active:
    actions.append({
        "priority": "p1" if execution_supervisor_launch_readiness_severity_state == "critical" else "p2",
        "action": "inspect_execution_supervisor_launch_readiness_severity_gate",
        "command": cmd_read_execution_supervisor_dispatch_qualification_json,
    })
    actions.append({
        "priority": "p2",
        "action": "refresh_execution_supervisor_launch_readiness_projection",
        "command": cmd_cont_current_refresh_json,
    })
if (
    execution_supervisor_launch_readiness_probe_execution_due_now_worker_count > 0
    or execution_supervisor_launch_readiness_probe_execution_overdue_worker_count > 0
    or (
        execution_supervisor_launch_readiness_probe_execution_action_priority in {"p1", "p2"}
        and execution_supervisor_launch_readiness_probe_execution_action_priority_source != "demotion_restore_posture"
    )
):
    actions.append({
        "priority": (
            execution_supervisor_launch_readiness_probe_execution_action_priority
            if execution_supervisor_launch_readiness_probe_execution_action_priority in {"p1", "p2"}
            else ("p1" if execution_supervisor_launch_readiness_probe_execution_overdue_worker_count > 0 else "p2")
        ),
        "action": "inspect_execution_supervisor_probe_execution_plan",
        "command": cmd_read_execution_supervisor_probe_execution_plan_json,
    })
    actions.append({
        "priority": "p2",
        "action": "refresh_execution_supervisor_probe_execution_plan",
        "command": cmd_cont_current_refresh_json,
    })
if (
    execution_supervisor_launch_readiness_restore_pending_worker_count > 0
    or execution_supervisor_launch_readiness_demoted_worker_count > 0
    or execution_supervisor_launch_readiness_demotion_action_priority in {"p1", "p2"}
):
    actions.append({
        "priority": (
            execution_supervisor_launch_readiness_demotion_action_priority
            if execution_supervisor_launch_readiness_demotion_action_priority in {"p1", "p2"}
            else "p2"
        ),
        "action": "inspect_execution_supervisor_demotion_restore_posture",
        "command": (
            cmd_read_execution_supervisor_dispatch_qualification_json
            if execution_supervisor_dispatch_qualification.get("launch_readiness_demotion_posture_present") is True
            else (
                cmd_read_execution_supervisor_dispatch_intent_json
                if execution_supervisor_dispatch_intent.get("launch_readiness_demotion_posture_present") is True
                else cmd_cont_verify_gate_status_json
            )
        ),
    })
    actions.append({
        "priority": "p2",
        "action": "refresh_execution_supervisor_demotion_restore_projection",
        "command": cmd_cont_current_refresh_json,
    })
if (
    autopilot_execution_frontier_controller_blocked
    or autopilot_execution_frontier_controller_error_state
    or autopilot_execution_frontier_controller_post_completion_stalled
    or autopilot_execution_frontier_controller_loop_state in {"STALLED_LOOP", "BLOCKED_LOOP"}
    or bool(autopilot_execution_frontier_controller_cooldown_policy.get("active") is True)
    or str(autopilot_execution_frontier_controller_retry_contract.get("state") or "") == "retry_exhausted"
    or str(autopilot_execution_frontier_controller_parity.get("status") or "") == "mismatch"
):
    actions.append({
        "priority": "p1",
        "action": "rerun_execution_frontier_controller_tick",
        "command": cmd_watchdog_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_controller_tick_trace",
        "command": cat_cmd_for(autopilot_execution_frontier_controller_trace_path),
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_controller_tick_history",
        "command": cat_cmd_for(autopilot_execution_frontier_controller_history_path),
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_post_completion_latch",
        "command": cmd_read_execution_frontier_enforcement_latch_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_post_completion_latch_history",
        "command": cmd_read_execution_frontier_enforcement_latch_history_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_autonomous_execution_intent",
        "command": cmd_read_autonomous_execution_intent_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_autonomous_execution_intent_history",
        "command": cmd_read_autonomous_execution_intent_history_json,
    })
elif autopilot_execution_frontier_controller_applied:
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_controller_tick_trace",
        "command": cat_cmd_for(autopilot_execution_frontier_controller_trace_path),
    })
if autopilot_execution_frontier_controller_source_degraded:
    actions.append({
        "priority": "p2",
        "action": "inspect_execution_frontier_controller_contract_source",
        "command": cmd_read_execution_frontier_controller_trace_json,
    })
if load_shedding_critical_tier or load_shedding_escape_triggered:
    actions.append({
        "priority": "p1",
        "action": "stabilize_load_shedding_lane",
        "command": cmd_cont_queue_sync_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_load_shedding_decision",
        "command": cmd_read_load_shedding_decision_json,
    })
if load_shedding_source_degraded:
    actions.append({
        "priority": "p2",
        "action": "inspect_load_shedding_signal_snapshot",
        "command": cmd_read_load_shedding_signal_snapshot_json,
    })
if autopilot_idle_lane_stalled:
    actions.append({
        "priority": "p1",
        "action": "recover_idle_lane_autospawn_stall",
        "command": cmd_watchdog_json,
    })
    idle_lane_trace_cmd = cat_cmd_for(autopilot_idle_lane_trace_path)
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_autospawn_trace",
        "command": idle_lane_trace_cmd,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_watchdog_event_history",
        "command": cmd_idle_lane_watchdog_history_json,
    })
if autopilot_idle_lane_contradiction_latched:
    actions.append({
        "priority": "p1",
        "action": "recheck_idle_lane_contradiction_latch",
        "command": cmd_watchdog_json,
    })
    idle_lane_trace_cmd = cat_cmd_for(autopilot_idle_lane_trace_path)
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_autospawn_trace",
        "command": idle_lane_trace_cmd,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_contradiction_latch",
        "command": cmd_read_idle_lane_latch_json,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_watchdog_event_history",
        "command": cmd_idle_lane_watchdog_history_json,
    })
if autopilot_idle_lane_contradiction_latch_repaired:
    idle_lane_trace_cmd = cat_cmd_for(autopilot_idle_lane_trace_path)
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_latch_repair_event",
        "command": idle_lane_trace_cmd,
    })
    actions.append({
        "priority": "p2",
        "action": "inspect_idle_lane_watchdog_event_history",
        "command": cmd_idle_lane_watchdog_history_json,
    })
if (contention.get("overdue_locks") or 0) > 0:
    actions.append({
        "priority": "p1",
        "action": "audited_lock_break",
        "command": f"{shell_cmd_for('ops/openclaw/continuity.sh')} --action-token {action_token_arg} lock-break --task-id <task_id> --reason '<why>' --operator <operator> --json",
    })
if browser_latest.get("status") in {"blocked", "failed"}:
    actions.append({
        "priority": "p1",
        "action": "inspect_browser_failure_packet",
        "command": cmd_web_capture_fetch_json,
    })
if int(web_domain_guard.get("operator_action_required_domains") or 0) > 0:
    first_contract = None
    first_resume = None
    first_incident_cmd = None
    for row in web_domain_guard.get("domains") or []:
        if not row.get("operator_action_required"):
            continue
        first_contract = row.get("operator_contract_md") or row.get("operator_contract_json")
        first_resume = row.get("operator_resume_command")
        incident = row.get("incident_actionability") if isinstance(row.get("incident_actionability"), dict) else {}
        for cmd in (incident.get("recommended_commands") or []):
            if isinstance(cmd, str) and cmd.strip():
                first_incident_cmd = normalize_operator_command(cmd)
                break
        if not first_incident_cmd:
            for step in (incident.get("recommended_steps") or []):
                if not isinstance(step, dict):
                    continue
                cmd = normalize_operator_command(step.get("command") or "")
                if cmd:
                    first_incident_cmd = cmd
                    break
        break

    actions.append({
        "priority": "p1",
        "action": "complete_web_login_operator_contract",
        "command": cat_cmd_for(first_contract) if first_contract else cmd_cont_mission_refresh_json,
    })
    if first_resume:
        actions.append({
            "priority": "p1",
            "action": "resume_web_capture_after_login_assist",
            "command": normalize_operator_command(first_resume),
        })
    elif first_incident_cmd:
        actions.append({
            "priority": "p1",
            "action": "run_web_login_incident_first_command",
            "command": first_incident_cmd,
        })
if int(web_domain_guard.get("cooldown_active_domains") or 0) > 0 and int(web_domain_guard.get("operator_action_required_domains") or 0) == 0:
    actions.append({
        "priority": "p2",
        "action": "respect_web_domain_backoff_window",
        "command": cmd_web_capture_auto_dry_json,
    })
if web_scheduler.get("state_exists") is False:
    actions.append({
        "priority": "p1",
        "action": "materialize_web_scheduler_state",
        "command": cmd_web_capture_scheduler_dry_json,
    })
if web_scheduler.get("contract_state_valid") is False:
    actions.append({
        "priority": "p1",
        "action": "repair_web_scheduler_contract",
        "command": cmd_web_capture_scheduler_dry_json,
    })
if web_scheduler.get("fresh") is False:
    actions.append({
        "priority": "p2",
        "action": "refresh_web_scheduler_freshness",
        "command": cmd_web_capture_scheduler_dry_json,
    })
if int((queue_replay.get("summary") or {}).get("task_count") or 0) > 0:
    actions.append({
        "priority": "p2",
        "action": "run_librarian_hygiene",
        "command": cmd_cont_librarian_hygiene_json,
    })

raw_blocker_reasons = [str(x) for x in (now_obj.get("blocker_reasons") or []) if str(x).strip()]
raw_reconcile_only_reasons = [str(x) for x in (now_obj.get("reconcile_only_reasons") or []) if str(x).strip()]
drift_reason_set = set(_DRIFT_REASON_SET)
if raw_blocker_reasons:
    blocker_reasons = raw_blocker_reasons
else:
    blocker_reasons = [reason for reason in not_ready_reasons if reason not in drift_reason_set]
if raw_reconcile_only_reasons:
    reconcile_only_reasons = raw_reconcile_only_reasons
else:
    reconcile_only_reasons = [reason for reason in not_ready_reasons if reason in drift_reason_set]

# continuity_now can emit legacy/raw reason tokens in blocker/reconcile-only fields
# while top-level not_ready reasons may already be remapped (for example:
# ground_truth_capture_drift -> ground_truth_capture_drift_reconcile_only).
# Keep mission-control partition strictly aligned to the top-level not_ready domain.
not_ready_reason_set = set(not_ready_reasons)
blocker_reasons = [reason for reason in blocker_reasons if reason in not_ready_reason_set]
reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason in not_ready_reason_set]

if generation_failclose_reasons:
    reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason not in generation_failclose_reasons]
    for reason in generation_failclose_reasons:
        if reason not in blocker_reasons:
            blocker_reasons.append(reason)

if freshness_failclose_reasons:
    reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason not in freshness_failclose_reasons]
    for reason in freshness_failclose_reasons:
        if reason not in blocker_reasons:
            blocker_reasons.append(reason)

if proof_failclose_reasons:
    reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason not in proof_failclose_reasons]
    for reason in proof_failclose_reasons:
        if reason not in blocker_reasons:
            blocker_reasons.append(reason)

unclassified_non_drift_reasons = [
    reason
    for reason in not_ready_reasons
    if reason not in drift_reason_set
    and reason not in blocker_reasons
    and reason not in reconcile_only_reasons
]
for reason in unclassified_non_drift_reasons:
    blocker_reasons.append(reason)

validate_reason_partition(
    not_ready_reasons=not_ready_reasons,
    blocker_reasons=blocker_reasons,
    reconcile_only_reasons=reconcile_only_reasons,
)

hard_blocker_count = len(blocker_reasons)
warning_count = len(set(warning_reasons + reconcile_only_reasons + freshness_warning_reasons + publish_lock_warning_reasons))

payload = {
    "schema": "clawd.operator_mission_control.v1",
    "generated_at": now_iso(),
    "workspace_id": "clawd-architect",
    "truth_strip": truth_strip,
    "headline": {
        "readiness": effective_readiness,
        "mutation_gate": mutation_gate_status,
        "mutation_gate_posture": mutation_gate_posture,
        "mutation_gate_expected_in_flight_guard": mutation_gate_expected_in_flight_guard if isinstance(mutation_gate_expected_in_flight_guard, bool) else None,
        "mutation_gate_blocking_reason_count": len(mutation_gate_blocking_reasons),
        "mutation_gate_concurrency_reason_count": len(mutation_gate_concurrency_reasons),
        "hard_blockers": hard_blocker_count,
        "warnings": warning_count,
        "freshness_posture": freshness_posture,
        "freshness_stale_count": freshness_stale_count,
        "freshness_unknown_count": freshness_unknown_count,
        "freshness_failclose_reason_count": len(freshness_failclose_reasons),
        "continuity_current_publish_lock_timeout_detected": current_publish_lock_timeout_detected,
        "continuity_current_publish_lock_timeout_wait_sec": current_publish_lock_timeout_wait_sec,
        "continuity_current_publish_lock_timeout_owner_pid": (
            current_publish_lock_timeout.get("owner_pid")
            if isinstance(current_publish_lock_timeout.get("owner_pid"), int)
            else None
        ),
        "continuity_current_publish_lock_timeout_owner_alive": (
            current_publish_lock_timeout.get("owner_alive")
            if isinstance(current_publish_lock_timeout.get("owner_alive"), bool)
            else None
        ),
        "continuity_current_publish_lock_timeout_owner_age_sec": current_publish_lock_timeout_owner_age_sec,
        "continuity_current_publish_lock_timeout_owner_exceeds_hold_warn": current_publish_lock_timeout_hold_exceeded,
        "continuity_current_publish_lock_surface_active": continuity_current_publish_lock.get("surface_active") is True,
        "continuity_current_publish_lock_source": continuity_current_publish_lock.get("source"),
        "continuity_current_publish_lock_status": continuity_current_publish_lock.get("status"),
        "continuity_current_publish_lock_warning_count": len(publish_lock_warning_reasons),
        "continuity_current_publish_lock_source_degraded": continuity_current_publish_lock.get("source_degraded") is True,
        "continuity_current_publish_lock_source_degraded_reason_count": len(_reason_list(continuity_current_publish_lock.get("source_degraded_reasons"))),
        "continuity_current_publish_lock_source_current_fresh": continuity_current_publish_lock.get("source_current_fresh") if isinstance(continuity_current_publish_lock.get("source_current_fresh"), bool) else None,
        "continuity_current_publish_lock_source_current_matches_current_generated_at": continuity_current_publish_lock.get("source_current_matches_current_generated_at") if isinstance(continuity_current_publish_lock.get("source_current_matches_current_generated_at"), bool) else None,
        "continuity_current_publish_lock_source_current_age_sec": continuity_current_publish_lock.get("source_current_age_sec") if isinstance(continuity_current_publish_lock.get("source_current_age_sec"), (int, float)) else None,
        "successor_proof_gate_enforced": proof_gate_enforced,
        "successor_proof_state": proof_state,
        "successor_proof_top_blocker": proof_top_blocker,
        "successor_proof_fail_closed": proof_fail_closed,
        "queue_running": contention.get("running_tasks"),
        "active_locks": contention.get("active_locks"),
        "execution_posture": execution_status.get("posture"),
        "execution_in_flight": execution_status.get("in_flight"),
        "execution_running_tasks": execution_status.get("running_tasks"),
        "execution_active_locks": execution_status.get("active_locks"),
        "execution_dispatch_status": execution_status.get("dispatch_status"),
        "execution_target_step_id": execution_status.get("target_step_id"),
        "execution_launched_step_id": execution_status.get("launched_step_id"),
        "execution_last_signal_at": execution_status.get("last_signal_at"),
        "execution_last_signal_age_sec": execution_status.get("last_signal_age_sec"),
        "execution_program_state": execution_status.get("program_state"),
        "execution_current_wave": execution_status.get("current_wave"),
        "execution_frontier_lane": execution_status.get("frontier_lane"),
        "execution_current_focus": execution_status.get("current_focus"),
        "execution_last_progress_at": execution_status.get("last_progress_at"),
        "execution_last_progress_age_sec": execution_status.get("last_progress_age_sec"),
        "execution_frontier_selector_state": execution_frontier.get("selector_state"),
        "execution_frontier_close_condition_met": execution_frontier.get("close_condition_met"),
        "execution_frontier_next_candidate": execution_frontier.get("next_candidate"),
        "execution_frontier_next_candidate_wave": execution_frontier.get("next_candidate_wave"),
        "execution_frontier_transition_reason": execution_frontier.get("transition_reason"),
        "execution_frontier_stalled": execution_frontier.get("stalled"),
        "execution_frontier_supervisor_state": execution_frontier.get("supervisor_state"),
        "execution_frontier_autonomous_dispatch_eligible": execution_frontier.get("autonomous_dispatch_eligible"),
        "execution_supervisor_dispatch_intent_status": execution_status.get("supervisor_dispatch_intent_status"),
        "execution_supervisor_dispatch_intent_decision": execution_status.get("supervisor_dispatch_intent_decision"),
        "execution_supervisor_dispatch_intent_fail_closed": execution_status.get("supervisor_dispatch_intent_fail_closed"),
        "execution_supervisor_dispatch_intent_launch_mutation_allowed": execution_status.get("supervisor_dispatch_intent_launch_mutation_allowed"),
        "execution_supervisor_dispatch_intent_ready_candidate_count": execution_status.get("supervisor_dispatch_intent_ready_candidate_count"),
        "execution_supervisor_dispatch_intent_blocked_candidate_count": execution_status.get("supervisor_dispatch_intent_blocked_candidate_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_state": execution_status.get("supervisor_dispatch_intent_launch_readiness_state"),
        "execution_supervisor_dispatch_intent_launch_readiness_reason": execution_status.get("supervisor_dispatch_intent_launch_readiness_reason"),
        "execution_supervisor_dispatch_intent_restore_pending_worker_count": execution_status.get("supervisor_dispatch_intent_restore_pending_worker_count"),
        "execution_supervisor_dispatch_intent_demoted_worker_count": execution_status.get("supervisor_dispatch_intent_demoted_worker_count"),
        "execution_supervisor_dispatch_intent_restored_worker_count": execution_status.get("supervisor_dispatch_intent_restored_worker_count"),
        "execution_supervisor_dispatch_intent_oldest_restore_pending_since": execution_status.get("supervisor_dispatch_intent_oldest_restore_pending_since"),
        "execution_supervisor_dispatch_intent_oldest_restore_pending_worker": execution_status.get("supervisor_dispatch_intent_oldest_restore_pending_worker"),
        "execution_supervisor_dispatch_intent_oldest_restore_pending_age_sec": execution_status.get("supervisor_dispatch_intent_oldest_restore_pending_age_sec"),
        "execution_supervisor_dispatch_intent_oldest_demoted_at": execution_status.get("supervisor_dispatch_intent_oldest_demoted_at"),
        "execution_supervisor_dispatch_intent_oldest_demoted_worker": execution_status.get("supervisor_dispatch_intent_oldest_demoted_worker"),
        "execution_supervisor_dispatch_intent_oldest_demoted_age_sec": execution_status.get("supervisor_dispatch_intent_oldest_demoted_age_sec"),
        "execution_supervisor_dispatch_intent_latest_restored_at": execution_status.get("supervisor_dispatch_intent_latest_restored_at"),
        "execution_supervisor_dispatch_intent_latest_restored_worker": execution_status.get("supervisor_dispatch_intent_latest_restored_worker"),
        "execution_supervisor_dispatch_intent_latest_restored_age_sec": execution_status.get("supervisor_dispatch_intent_latest_restored_age_sec"),
        "execution_supervisor_dispatch_intent_demotion_action_priority": execution_status.get("supervisor_dispatch_intent_demotion_action_priority"),
        "execution_supervisor_dispatch_intent_blocked_probe_candidate_count": execution_status.get("supervisor_dispatch_intent_blocked_probe_candidate_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_status": execution_status.get("supervisor_dispatch_intent_probe_execution_status"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_reason": execution_status.get("supervisor_dispatch_intent_probe_execution_reason"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_action_priority": execution_status.get("supervisor_dispatch_intent_probe_execution_action_priority"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_pending_worker_count": execution_status.get("supervisor_dispatch_intent_probe_execution_pending_worker_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_due_now_worker_count": execution_status.get("supervisor_dispatch_intent_probe_execution_due_now_worker_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_scheduled_worker_count": execution_status.get("supervisor_dispatch_intent_probe_execution_scheduled_worker_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_overdue_worker_count": execution_status.get("supervisor_dispatch_intent_probe_execution_overdue_worker_count"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_due_now_worker": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_due_now_worker"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_due_now_started_at": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_due_now_started_at"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_due_now_age_sec": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_due_now_age_sec"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_overdue_worker": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_overdue_worker"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_overdue_started_at": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_overdue_started_at"),
        "execution_supervisor_dispatch_intent_launch_readiness_probe_execution_oldest_overdue_age_sec": execution_status.get("supervisor_dispatch_intent_probe_execution_oldest_overdue_age_sec"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_state": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_state"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_reason": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_reason"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_active": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_active"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_non_ready_ticks": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_non_ready_ticks"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_threshold_ticks": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_threshold_ticks"),
        "execution_supervisor_dispatch_intent_launch_readiness_severity_cohort_worker_count": execution_status.get("supervisor_dispatch_intent_launch_readiness_severity_cohort_worker_count"),
        "execution_supervisor_dispatch_qualification_status": execution_status.get("supervisor_dispatch_qualification_status"),
        "execution_supervisor_dispatch_qualification_decision": execution_status.get("supervisor_dispatch_qualification_decision"),
        "execution_supervisor_dispatch_qualification_fail_closed": execution_status.get("supervisor_dispatch_qualification_fail_closed"),
        "execution_supervisor_dispatch_qualification_launch_mutation_allowed": execution_status.get("supervisor_dispatch_qualification_launch_mutation_allowed"),
        "execution_supervisor_dispatch_qualification_ready_candidate_count": execution_status.get("supervisor_dispatch_qualification_ready_candidate_count"),
        "execution_supervisor_dispatch_qualification_qualified_candidate_count": execution_status.get("supervisor_dispatch_qualification_qualified_candidate_count"),
        "execution_supervisor_dispatch_qualification_blocked_candidate_count": execution_status.get("supervisor_dispatch_qualification_blocked_candidate_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_state": execution_status.get("supervisor_dispatch_qualification_launch_readiness_state"),
        "execution_supervisor_dispatch_qualification_launch_readiness_reason": execution_status.get("supervisor_dispatch_qualification_launch_readiness_reason"),
        "execution_supervisor_dispatch_qualification_restore_pending_worker_count": execution_status.get("supervisor_dispatch_qualification_restore_pending_worker_count"),
        "execution_supervisor_dispatch_qualification_demoted_worker_count": execution_status.get("supervisor_dispatch_qualification_demoted_worker_count"),
        "execution_supervisor_dispatch_qualification_restored_worker_count": execution_status.get("supervisor_dispatch_qualification_restored_worker_count"),
        "execution_supervisor_dispatch_qualification_oldest_restore_pending_since": execution_status.get("supervisor_dispatch_qualification_oldest_restore_pending_since"),
        "execution_supervisor_dispatch_qualification_oldest_restore_pending_worker": execution_status.get("supervisor_dispatch_qualification_oldest_restore_pending_worker"),
        "execution_supervisor_dispatch_qualification_oldest_restore_pending_age_sec": execution_status.get("supervisor_dispatch_qualification_oldest_restore_pending_age_sec"),
        "execution_supervisor_dispatch_qualification_oldest_demoted_at": execution_status.get("supervisor_dispatch_qualification_oldest_demoted_at"),
        "execution_supervisor_dispatch_qualification_oldest_demoted_worker": execution_status.get("supervisor_dispatch_qualification_oldest_demoted_worker"),
        "execution_supervisor_dispatch_qualification_oldest_demoted_age_sec": execution_status.get("supervisor_dispatch_qualification_oldest_demoted_age_sec"),
        "execution_supervisor_dispatch_qualification_latest_restored_at": execution_status.get("supervisor_dispatch_qualification_latest_restored_at"),
        "execution_supervisor_dispatch_qualification_latest_restored_worker": execution_status.get("supervisor_dispatch_qualification_latest_restored_worker"),
        "execution_supervisor_dispatch_qualification_latest_restored_age_sec": execution_status.get("supervisor_dispatch_qualification_latest_restored_age_sec"),
        "execution_supervisor_dispatch_qualification_demotion_action_priority": execution_status.get("supervisor_dispatch_qualification_demotion_action_priority"),
        "execution_supervisor_dispatch_qualification_blocked_probe_candidate_count": execution_status.get("supervisor_dispatch_qualification_blocked_probe_candidate_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_status": execution_status.get("supervisor_dispatch_qualification_probe_execution_status"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_reason": execution_status.get("supervisor_dispatch_qualification_probe_execution_reason"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_action_priority": execution_status.get("supervisor_dispatch_qualification_probe_execution_action_priority"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_pending_worker_count": execution_status.get("supervisor_dispatch_qualification_probe_execution_pending_worker_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_due_now_worker_count": execution_status.get("supervisor_dispatch_qualification_probe_execution_due_now_worker_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_scheduled_worker_count": execution_status.get("supervisor_dispatch_qualification_probe_execution_scheduled_worker_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_overdue_worker_count": execution_status.get("supervisor_dispatch_qualification_probe_execution_overdue_worker_count"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_due_now_worker": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_due_now_worker"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_due_now_started_at": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_due_now_started_at"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_due_now_age_sec": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_due_now_age_sec"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_overdue_worker": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_overdue_worker"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_overdue_started_at": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_overdue_started_at"),
        "execution_supervisor_dispatch_qualification_launch_readiness_probe_execution_oldest_overdue_age_sec": execution_status.get("supervisor_dispatch_qualification_probe_execution_oldest_overdue_age_sec"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_state": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_state"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_reason": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_reason"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_active": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_active"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_non_ready_ticks": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_non_ready_ticks"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_threshold_ticks": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_threshold_ticks"),
        "execution_supervisor_dispatch_qualification_launch_readiness_severity_cohort_worker_count": execution_status.get("supervisor_dispatch_qualification_launch_readiness_severity_cohort_worker_count"),
        "execution_supervisor_launch_readiness_severity_state": execution_status.get("supervisor_launch_readiness_severity_state"),
        "execution_supervisor_launch_readiness_severity_reason": execution_status.get("supervisor_launch_readiness_severity_reason"),
        "execution_supervisor_launch_readiness_severity_active": execution_status.get("supervisor_launch_readiness_severity_active"),
        "execution_supervisor_launch_readiness_severity_non_ready_ticks": execution_status.get("supervisor_launch_readiness_severity_non_ready_ticks"),
        "execution_supervisor_launch_readiness_severity_threshold_ticks": execution_status.get("supervisor_launch_readiness_severity_threshold_ticks"),
        "execution_supervisor_launch_readiness_severity_cohort_worker_count": execution_status.get("supervisor_launch_readiness_severity_cohort_worker_count"),
        "execution_supervisor_launch_readiness_restore_pending_worker_count": execution_status.get("supervisor_launch_readiness_restore_pending_worker_count"),
        "execution_supervisor_launch_readiness_demoted_worker_count": execution_status.get("supervisor_launch_readiness_demoted_worker_count"),
        "execution_supervisor_launch_readiness_restored_worker_count": execution_status.get("supervisor_launch_readiness_restored_worker_count"),
        "execution_supervisor_launch_readiness_oldest_restore_pending_since": execution_status.get("supervisor_launch_readiness_oldest_restore_pending_since"),
        "execution_supervisor_launch_readiness_oldest_restore_pending_worker": execution_status.get("supervisor_launch_readiness_oldest_restore_pending_worker"),
        "execution_supervisor_launch_readiness_oldest_restore_pending_age_sec": execution_status.get("supervisor_launch_readiness_oldest_restore_pending_age_sec"),
        "execution_supervisor_launch_readiness_oldest_demoted_at": execution_status.get("supervisor_launch_readiness_oldest_demoted_at"),
        "execution_supervisor_launch_readiness_oldest_demoted_worker": execution_status.get("supervisor_launch_readiness_oldest_demoted_worker"),
        "execution_supervisor_launch_readiness_oldest_demoted_age_sec": execution_status.get("supervisor_launch_readiness_oldest_demoted_age_sec"),
        "execution_supervisor_launch_readiness_latest_restored_at": execution_status.get("supervisor_launch_readiness_latest_restored_at"),
        "execution_supervisor_launch_readiness_latest_restored_worker": execution_status.get("supervisor_launch_readiness_latest_restored_worker"),
        "execution_supervisor_launch_readiness_latest_restored_age_sec": execution_status.get("supervisor_launch_readiness_latest_restored_age_sec"),
        "execution_supervisor_launch_readiness_demotion_action_priority": execution_status.get("supervisor_launch_readiness_demotion_action_priority"),
        "execution_supervisor_launch_readiness_probe_execution_status": execution_status.get("supervisor_launch_readiness_probe_execution_status"),
        "execution_supervisor_launch_readiness_probe_execution_reason": execution_status.get("supervisor_launch_readiness_probe_execution_reason"),
        "execution_supervisor_launch_readiness_probe_execution_action_priority": execution_status.get("supervisor_launch_readiness_probe_execution_action_priority"),
        "execution_supervisor_launch_readiness_probe_execution_action_priority_source": execution_status.get("supervisor_launch_readiness_probe_execution_action_priority_source"),
        "execution_supervisor_launch_readiness_probe_execution_pending_worker_count": execution_status.get("supervisor_launch_readiness_probe_execution_pending_worker_count"),
        "execution_supervisor_launch_readiness_probe_execution_due_now_worker_count": execution_status.get("supervisor_launch_readiness_probe_execution_due_now_worker_count"),
        "execution_supervisor_launch_readiness_probe_execution_overdue_worker_count": execution_status.get("supervisor_launch_readiness_probe_execution_overdue_worker_count"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_due_now_worker": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_due_now_worker"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_due_now_started_at": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_due_now_started_at"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_due_now_age_sec"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_overdue_worker": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_overdue_worker"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_overdue_started_at": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_overdue_started_at"),
        "execution_supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec": execution_status.get("supervisor_launch_readiness_probe_execution_oldest_overdue_age_sec"),
        "execution_frontier_controller_status": execution_status.get("autonomous_dispatch_status"),
        "execution_frontier_controller_decision": execution_status.get("autonomous_dispatch_decision"),
        "execution_frontier_controller_skip_reason": execution_status.get("autonomous_dispatch_skip_reason"),
        "execution_frontier_controller_block_reason": execution_status.get("autonomous_dispatch_block_reason"),
        "execution_frontier_controller_error": execution_status.get("autonomous_dispatch_error"),
        "execution_frontier_controller_source_degraded": execution_status.get("autonomous_dispatch_source_degraded"),
        "execution_frontier_controller_updated_at": execution_status.get("autonomous_dispatch_updated_at"),
        "execution_frontier_controller_selector_state": execution_status.get("autonomous_dispatch_selector_state"),
        "execution_frontier_post_completion_enforcement_required": execution_status.get("autonomous_dispatch_post_completion_enforcement_required"),
        "execution_frontier_post_completion_enforcement_latched": execution_status.get("autonomous_dispatch_post_completion_enforcement_latched"),
        "execution_frontier_post_completion_enforcement_blocked": execution_status.get("autonomous_dispatch_post_completion_blocked"),
        "execution_frontier_post_completion_enforcement_stalled": execution_status.get("autonomous_dispatch_post_completion_stalled"),
        "execution_frontier_post_completion_loop_state": execution_status.get("autonomous_dispatch_post_completion_loop_state"),
        "execution_frontier_post_completion_retry_state": (
            (execution_status.get("autonomous_dispatch_retry_contract") or {}).get("state")
            if isinstance(execution_status.get("autonomous_dispatch_retry_contract"), dict)
            else None
        ),
        "execution_frontier_post_completion_cooldown_active": (
            (execution_status.get("autonomous_dispatch_cooldown_policy") or {}).get("active")
            if isinstance(execution_status.get("autonomous_dispatch_cooldown_policy"), dict)
            else None
        ),
        "execution_frontier_queue_truth_vs_narrative_status": (
            (execution_status.get("autonomous_dispatch_queue_truth_vs_narrative_parity") or {}).get("status")
            if isinstance(execution_status.get("autonomous_dispatch_queue_truth_vs_narrative_parity"), dict)
            else None
        ),
        "execution_frontier_autonomous_intent_active": execution_status.get("autonomous_dispatch_intent_active"),
        "handover_stale": handover.get("stale"),
        "gate_failures": (gate_os.get("summary") or {}).get("fail"),
        "queue_replay_status": (queue_replay.get("summary") or {}).get("status"),
        "queue_replay_mismatches": int((queue_replay.get("summary") or {}).get("active_status_mismatch_count", (queue_replay.get("summary") or {}).get("status_mismatch_count")) or 0) + int((queue_replay.get("summary") or {}).get("role_mismatch_count") or 0),
        "wave2_replay_evidence_ready": wave2_replay_ready,
        "wave2_replay_evidence_verdict": wave2_replay_verdict,
        "wave2_replay_evidence_fresh": wave2_replay_fresh if isinstance(wave2_replay_evidence_obj, dict) else None,
        "wave2_replay_evidence_age_sec": wave2_replay_age_sec if isinstance(wave2_replay_evidence_obj, dict) else None,
        "wave2_replay_evidence_soak_drift_detected": wave2_replay_soak_drift_detected if isinstance(wave2_replay_evidence_obj, dict) else None,
        "failover_stress_soak_ready": failover_stress_soak_ready,
        "failover_stress_soak_verdict": failover_stress_soak_verdict,
        "failover_stress_soak_fresh": failover_stress_soak_fresh if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_soak_age_sec": failover_stress_soak_age_sec if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_soak_profile_count": failover_stress_soak_profile_count if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_soak_total_cycles": failover_stress_soak_total_cycles if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_soak_convergence_fail_count": failover_stress_soak_convergence_fail_count if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_soak_drift_detected": failover_stress_soak_drift_detected if isinstance(failover_stress_soak_obj, dict) else None,
        "failover_stress_runtime_evidence_ready": failover_stress_runtime_ready,
        "failover_stress_runtime_evidence_verdict": failover_stress_runtime_verdict,
        "failover_stress_runtime_publish_verdict": failover_stress_runtime_publish_verdict,
        "failover_stress_runtime_evidence_fresh": failover_stress_runtime_fresh if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_evidence_age_sec": failover_stress_runtime_age_sec if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_publish_assertions_failed": failover_stress_runtime_publish_assertions_failed if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_repeatability_status": failover_stress_runtime_repeatability_status if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_repeatability_match": failover_stress_runtime_repeatability_match if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_repeatability_mismatch_fields": failover_stress_runtime_repeatability_mismatch_fields if isinstance(failover_stress_runtime_obj, dict) else [],
        "failover_stress_runtime_active_top_blocker": failover_stress_runtime_active_top_blocker if isinstance(failover_stress_runtime_obj, dict) else None,
        "failover_stress_runtime_effective_top_blocker": failover_stress_runtime_effective_top_blocker if isinstance(failover_stress_runtime_obj, dict) else None,
        "gtc_handoff_binding_degraded": gtc_handoff_binding_degraded,
        "gtc_warning_reason_count": len(gtc_warning_reasons),
        "verify_gate_strict_mode_enabled": bool(verify_gate_preflight_strict.get("enabled") is True),
        "verify_gate_strict_mode_source": str(verify_gate_preflight_strict.get("source") or "disabled"),
        "verify_gate_strict_mode_required": verify_gate_preflight_strict.get("required") if isinstance(verify_gate_preflight_strict.get("required"), bool) else None,
        "verify_gate_strict_mode_override": verify_gate_preflight_strict.get("override"),
        "verify_gate_predicted_blocker_reason": verify_gate_preflight_predicted.get("predicted_blocker_reason"),
        "verify_gate_ready_to_run": verify_gate_preflight_predicted.get("ready_to_run") if isinstance(verify_gate_preflight_predicted.get("ready_to_run"), bool) else None,
        "verify_status_evidence_failure_reason": verify_status_evidence_failure_reason or None,
        "verify_status_evidence_fresh": verify_gate_preflight_status_evidence.get("fresh") if isinstance(verify_gate_preflight_status_evidence.get("fresh"), bool) else None,
        "verify_internal_bypass_closeout_ready": verify_internal_bypass_closeout_ready,
        "verify_internal_bypass_closeout_failure_reason": verify_internal_bypass_closeout_failure_reason or None,
        "verify_internal_bypass_unknown_callsite_total": verify_internal_bypass_unknown_total,
        "verify_internal_bypass_break_glass_allow_count": verify_internal_bypass_break_glass_allow,
        "verify_internal_bypass_break_glass_denied_count": verify_internal_bypass_break_glass_denied,
        "verify_layered_health_closeout_ready": verify_layered_health_closeout_ready,
        "verify_layered_health_failure_reason": verify_layered_health_failure_reason or None,
        "verify_layered_health_status": verify_layered_health_status,
        "verify_layered_health_layer": verify_layered_health_layer,
        "verify_layered_health_restore_slo_status": verify_layered_health_restore_slo_status,
        "verify_layered_health_missing_required_lane_count": len(verify_layered_health_missing_required_lanes),
        "verify_layered_health_failing_required_lane_count": len(verify_layered_health_failing_required_lanes),
        "verify_layered_health_layer_insufficient_required_lane_count": len(verify_layered_health_layer_insufficient_required_lanes),
        "verify_worker_health_canary_gate_failure_reason": verify_worker_health_canary_preflight_failure_reason or None,
        "verify_worker_health_canary_gate_active_blocker": verify_worker_health_canary_preflight_active_blocker,
        "verify_worker_health_canary_gate_dispatch_failure_reason": verify_worker_health_canary_preflight_dispatch_failure_reason or None,
        "verify_worker_health_canary_gate_source": verify_worker_health_canary_preflight_source,
        "verify_worker_health_canary_gate_action_priority": verify_worker_health_canary_preflight_action_priority,
        "verify_probe_execution_gate_failure_reason": verify_probe_execution_gate_failure_reason or None,
        "verify_probe_execution_gate_active_blocker": verify_probe_execution_gate_active_blocker,
        "verify_probe_execution_gate_pending_worker_count": verify_probe_execution_gate_pending_worker_count,
        "verify_probe_execution_gate_due_now_worker_count": verify_probe_execution_gate_due_now_worker_count,
        "verify_probe_execution_gate_overdue_worker_count": verify_probe_execution_gate_overdue_worker_count,
        "verify_probe_execution_gate_oldest_due_now_worker": verify_probe_execution_gate_oldest_due_now_worker,
        "verify_probe_execution_gate_oldest_due_now_started_at": verify_probe_execution_gate_oldest_due_now_started_at,
        "verify_probe_execution_gate_oldest_due_now_age_sec": verify_probe_execution_gate_oldest_due_now_age_sec,
        "verify_probe_execution_gate_oldest_overdue_worker": verify_probe_execution_gate_oldest_overdue_worker,
        "verify_probe_execution_gate_oldest_overdue_started_at": verify_probe_execution_gate_oldest_overdue_started_at,
        "verify_probe_execution_gate_oldest_overdue_age_sec": verify_probe_execution_gate_oldest_overdue_age_sec,
        "verify_probe_execution_gate_demotion_restore_pending_worker_count": verify_probe_execution_gate_demotion_restore_pending_worker_count,
        "verify_probe_execution_gate_demotion_demoted_worker_count": verify_probe_execution_gate_demotion_demoted_worker_count,
        "verify_probe_execution_gate_demotion_restored_worker_count": verify_probe_execution_gate_demotion_restored_worker_count,
        "verify_probe_execution_gate_demotion_oldest_restore_pending_since": verify_probe_execution_gate_demotion_oldest_restore_pending_since,
        "verify_probe_execution_gate_demotion_oldest_restore_pending_worker": verify_probe_execution_gate_demotion_oldest_restore_pending_worker,
        "verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec": verify_probe_execution_gate_demotion_oldest_restore_pending_age_sec,
        "verify_probe_execution_gate_demotion_oldest_demoted_at": verify_probe_execution_gate_demotion_oldest_demoted_at,
        "verify_probe_execution_gate_demotion_oldest_demoted_worker": verify_probe_execution_gate_demotion_oldest_demoted_worker,
        "verify_probe_execution_gate_demotion_oldest_demoted_age_sec": verify_probe_execution_gate_demotion_oldest_demoted_age_sec,
        "verify_probe_execution_gate_demotion_latest_restored_at": verify_probe_execution_gate_demotion_latest_restored_at,
        "verify_probe_execution_gate_demotion_latest_restored_worker": verify_probe_execution_gate_demotion_latest_restored_worker,
        "verify_probe_execution_gate_demotion_latest_restored_age_sec": verify_probe_execution_gate_demotion_latest_restored_age_sec,
        "verify_probe_execution_gate_demotion_action_priority": verify_probe_execution_gate_demotion_action_priority,
        "verify_probe_execution_gate_probe_plan_path": verify_probe_execution_gate_probe_plan_path,
        "verify_probe_execution_gate_probe_plan_present": verify_probe_execution_gate_probe_plan_present,
        "reset_ready_refresh_present": reset_ready_refresh_present,
        "reset_ready_refresh_ok": reset_ready_refresh_ok,
        "reset_ready_refresh_phase": reset_ready_refresh_phase,
        "reset_ready_refresh_error_code": reset_ready_refresh_error_code,
        "reset_ready_refresh_partial_failure": reset_ready_refresh_partial_failure,
        "routing_preflight_source": routing_preflight_source,
        "routing_preflight_failure_reason": routing_failure_reason or None,
        "routing_preflight_decision": routing_decision,
        "routing_preflight_route_class": routing_route_class,
        "routing_preflight_selected_model": routing_selected_model,
        "routing_preflight_block_gate": routing_block_gate,
        "routing_preflight_block_reason": routing_block_reason,
        "routing_preflight_blocked_fresh": routing_blocked_fresh,
        "model_rollout_dashboard_status": model_rollout_dashboard_status,
        "model_rollout_prompt_status": model_rollout_prompt_status,
        "model_rollout_prompt_reason": model_rollout_prompt_reason,
        "model_rollout_prompt_requires_approval": model_rollout_prompt_requires_approval,
        "model_rollout_prompt_command_count": len(model_rollout_prompt_commands),
        "model_rollout_prompt_scorecard_ref": model_rollout_prompt_scorecard_ref,
        "model_rollout_remediation_status": model_rollout_remediation_status,
        "model_rollout_remediation_active": model_rollout_remediation_active,
        "model_rollout_remediation_reason_gate": model_rollout_remediation_reason_gate,
        "model_rollout_remediation_reason_code": model_rollout_remediation_reason_code,
        "model_rollout_remediation_command_count": len(model_rollout_remediation_commands),
        "model_rollout_remediation_correction_cycle_log_ref": model_rollout_remediation_correction_cycle_log_ref,
        "dependency_policy_pack_status": dependency_policy_pack_projection.get("status"),
        "dependency_policy_pack_schema_ok": dependency_policy_pack_projection.get("schema_ok"),
        "dependency_policy_pack_slice_count": int(dependency_policy_pack_projection.get("slice_count") or 0),
        "dependency_policy_pack_required_slice_coverage": dependency_policy_pack_projection.get("required_slice_coverage"),
        "efficiency_kpi_baseline_status": efficiency_kpi_status,
        "efficiency_kpi_validation_status": efficiency_kpi_validation_status,
        "efficiency_kpi_kpi_count": efficiency_kpi_kpi_count,
        "efficiency_kpi_measured_count": efficiency_kpi_measured_count,
        "efficiency_kpi_no_signal_count": efficiency_kpi_no_signal_count,
        "degraded_pending_signal_active": bool(autopilot_degraded_pending_signal.get("active") is True),
        "degraded_pending_stale_count": int(autopilot_degraded_pending_signal.get("pending_stale_count") or 0),
        "degraded_pending_total": int(autopilot_degraded_pending_signal.get("pending_total") or 0),
        "degraded_pending_streak": int(autopilot_degraded_pending_signal.get("stale_ticks_consecutive") or 0),
        "evidence_trace_viewer_status": evidence_trace_viewer.get("status"),
        "evidence_trace_viewer_focus_task_id": evidence_trace_viewer.get("focus_task_id"),
        "evidence_trace_viewer_focus_slice_id": evidence_trace_viewer.get("focus_slice_id"),
        "evidence_trace_viewer_trace_count": int(evidence_trace_viewer.get("trace_count") or 0),
        "evidence_trace_viewer_artifact_ref_count": int(evidence_trace_viewer.get("artifact_ref_count") or 0),
        "evidence_trace_viewer_blocked_reason_count": int(evidence_trace_viewer.get("blocked_reason_count") or 0),
        "queue_stale_wave_active": bool(queue_stale_wave_signal.get("active") is True),
        "queue_stale_wave_ready_count": int(queue_stale_wave_signal.get("ready_count") or 0),
        "queue_stale_wave_ready_oldest_age_sec": int(queue_stale_wave_signal.get("ready_oldest_age_sec") or 0),
        "meaningful_event_reporting_status": meaningful_event_reporting_projection.get("status"),
        "meaningful_event_reporting_attention_required": meaningful_event_reporting_projection.get("attention_required"),
        "meaningful_event_reporting_pending_required_event_count": int(meaningful_event_reporting_projection.get("pending_required_event_count") or 0),
        "meaningful_event_reporting_pending_required_event_codes_summary": meaningful_event_reporting_projection.get("pending_required_event_codes_summary"),
        "meaningful_event_reporting_critical_pending_event_count": int(meaningful_event_reporting_projection.get("critical_pending_event_count") or 0),
        "meaningful_event_reporting_new_event_count": int(meaningful_event_reporting_projection.get("new_event_count") or 0),
        "meaningful_event_reporting_latest_event_code": meaningful_event_reporting_projection.get("latest_event_code"),
        "meaningful_event_reporting_latest_event_detected_at": meaningful_event_reporting_projection.get("latest_event_detected_at"),
        "meaningful_event_reporting_contract_status": meaningful_event_reporting_contract.get("status"),
        "meaningful_event_reporting_contract_required_packet_expected": meaningful_event_reporting_contract.get("required_packet_expected") is True,
        "meaningful_event_reporting_contract_failclose_reason_count": len(meaningful_event_reporting_contract.get("failclose_reasons") or []),
        "meaningful_event_reporting_contract_warning_reason_count": len(meaningful_event_reporting_contract.get("warning_reasons") or []),
        "meaningful_event_reporting_contract_digest_status": meaningful_event_reporting_contract.get("digest_status"),
        "meaningful_event_reporting_contract_digest_token": meaningful_event_reporting_contract.get("digest_token"),
        "meaningful_event_reporting_contract_escalation_level": meaningful_event_reporting_contract.get("escalation_level"),
        "meaningful_event_reporting_contract_expected_vs_delivered_checksum": meaningful_event_reporting_contract.get("expected_vs_delivered_packet_checksum_line"),
        "meaningful_event_reporting_contract_expected_vs_delivered_checksum_hash": meaningful_event_reporting_contract.get("expected_vs_delivered_packet_checksum_hash"),
        "load_shedding_lane_health_state": load_shedding_lane_state,
        "load_shedding_warning_tier": load_shedding_warning_tier,
        "load_shedding_critical_tier": load_shedding_critical_tier,
        "load_shedding_escape_triggered": load_shedding_escape_triggered,
        "load_shedding_trigger_emitted": load_shedding_trigger_emitted,
        "load_shedding_source_degraded": load_shedding_source_degraded,
        "idle_lane_autospawn_status": autopilot_idle_lane_status,
        "idle_lane_autospawn_stalled": autopilot_idle_lane_stalled,
        "idle_lane_autospawn_contradiction_latched": autopilot_idle_lane_contradiction_latched,
        "idle_lane_autospawn_contradiction_abort_remaining_sec": autopilot_idle_lane_contradiction_abort_remaining_sec,
        "idle_lane_autospawn_contradiction_latch_repaired": autopilot_idle_lane_contradiction_latch_repaired,
        "idle_lane_autospawn_contradiction_latch_repair_reason": autopilot_idle_lane_contradiction_latch_repair_reason or None,
        "last_lock_break_at": lock_break_latest.get("generated_at"),
        "web_domains_tracked": web_domain_guard.get("tracked_domains"),
        "web_domains_blocked": web_domain_guard.get("blocked_domains"),
        "web_operator_actions": web_domain_guard.get("operator_action_required_domains"),
        "web_actionable_incidents": web_domain_guard.get("actionable_incident_domains"),
        "web_scheduler_fresh": web_scheduler.get("fresh"),
        "web_scheduler_contract_valid": web_scheduler.get("contract_state_valid"),
    },
    "freshness": {
        "posture": freshness_posture,
        "stale_count": freshness_stale_count,
        "unknown_count": freshness_unknown_count,
        "failclose_reasons": freshness_failclose_reasons,
        "proof_failclose_reasons": proof_failclose_reasons,
        "warning_reasons": freshness_warning_reasons,
    },
    "execution_status": execution_status,
    "execution_frontier": execution_frontier,
    "execution_supervisor_dispatch_intent": execution_supervisor_dispatch_intent,
    "execution_supervisor_dispatch_qualification": execution_supervisor_dispatch_qualification,
    "continuity_current_publish_lock_timeout": current_publish_lock_timeout,
    "continuity_current_publish_lock": continuity_current_publish_lock,
    "queue": contention,
    "queue_stale_wave_signal": queue_stale_wave_signal,
    "meaningful_event_reporting": meaningful_event_reporting_projection,
    "meaningful_event_reporting_contract": meaningful_event_reporting_contract,
    "routing_preflight": routing_preflight,
    "layered_health_gate": verify_gate_preflight_layered_health,
    "launch_readiness_probe_execution_gate": verify_gate_preflight_launch_readiness_probe_execution,
    "load_shedding": load_shedding_projection,
    "queue_replay": queue_replay,
    "wave2_replay_evidence": {
        "path": str(wave2_replay_evidence_index_path.relative_to(root)),
        "present": wave2_replay_evidence_present,
        "source_degraded": wave2_replay_evidence_source_degraded,
        "ready": wave2_replay_ready,
        "verdict": wave2_replay_verdict,
        "generated_at": wave2_replay_generated_at,
        "age_sec": wave2_replay_age_sec,
        "max_age_sec": WAVE2_REPLAY_EVIDENCE_MAX_AGE_SEC,
        "scenario_count": wave2_replay_scenario_count,
        "soak_runs_total": wave2_replay_soak_runs_total,
        "soak_drift_detected": wave2_replay_soak_drift_detected,
    },
    "failover_stress_soak": {
        "path": str(failover_stress_soak_evidence_path.relative_to(root)),
        "present": failover_stress_soak_present,
        "source_degraded": failover_stress_soak_source_degraded,
        "ready": failover_stress_soak_ready,
        "verdict": failover_stress_soak_verdict,
        "generated_at": failover_stress_soak_generated_at,
        "age_sec": failover_stress_soak_age_sec,
        "max_age_sec": FAILOVER_STRESS_SOAK_MAX_AGE_SEC,
        "profile_count": failover_stress_soak_profile_count,
        "total_cycles": failover_stress_soak_total_cycles,
        "convergence_fail_count": failover_stress_soak_convergence_fail_count,
        "drift_detected": failover_stress_soak_drift_detected,
    },
    "failover_stress_runtime_evidence": {
        "path": str(failover_stress_runtime_evidence_path.relative_to(root)),
        "present": failover_stress_runtime_present,
        "source_degraded": failover_stress_runtime_source_degraded,
        "ready": failover_stress_runtime_ready,
        "verdict": failover_stress_runtime_verdict,
        "publish_verdict": failover_stress_runtime_publish_verdict,
        "generated_at": failover_stress_runtime_generated_at,
        "age_sec": failover_stress_runtime_age_sec,
        "max_age_sec": FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC,
        "publish_assertions_failed": failover_stress_runtime_publish_assertions_failed,
        "active_top_blocker": failover_stress_runtime_active_top_blocker,
        "effective_top_blocker": failover_stress_runtime_effective_top_blocker,
        "repeatability_status": failover_stress_runtime_repeatability_status,
        "repeatability_match": failover_stress_runtime_repeatability_match,
        "repeatability_mismatch_fields": failover_stress_runtime_repeatability_mismatch_fields,
        "decision_log_ref": failover_stress_runtime_decision_log_ref or None,
        "decision_log_exists": failover_stress_runtime_decision_log_exists,
    },
    "generation_pointer": generation_pointer,
    "handover": handover,
    "reset_ready_refresh": {
        "path": reset_ready_refresh_rel,
        "present": reset_ready_refresh_present,
        "ok": reset_ready_refresh_ok,
        "phase": reset_ready_refresh_phase,
        "error_code": reset_ready_refresh_error_code,
        "generated_at": reset_ready_refresh_projection.get("generated_at"),
        "partial_refresh": {
            "current_refreshed": reset_ready_refresh_partial_current,
            "proof_refreshed": reset_ready_refresh_partial_proof,
            "handover_refreshed": reset_ready_refresh_partial_handover,
        },
        "degraded": reset_ready_refresh_degraded,
    },
    "model_rollout_dashboard": model_rollout_dashboard,
    "dependency_policy_pack": dependency_policy_pack_projection,
    "model_rollout_operator_mistake_remediation": {
        "status": model_rollout_remediation_status,
        "active": model_rollout_remediation_active,
        "reason_gate": model_rollout_remediation_reason_gate,
        "reason_code": model_rollout_remediation_reason_code,
        "hint": model_rollout_remediation_hint,
        "operator_message": model_rollout_remediation_operator_message,
        "safe_remediation_commands": model_rollout_remediation_commands,
        "correction_cycle_log_ref": model_rollout_remediation_correction_cycle_log_ref,
        "source": "model_rollout_dashboard",
    },
    "efficiency_kpi_baseline": efficiency_kpi_baseline,
    "efficiency_kpi_validation": {
        "status": efficiency_kpi_validation_status,
        "error_count": int(efficiency_kpi_validation.get("error_count") or 0),
        "reason": efficiency_kpi_validation.get("reason"),
        "errors": efficiency_kpi_validation.get("errors") if isinstance(efficiency_kpi_validation.get("errors"), list) else [],
        "schema_ref": efficiency_kpi_validation.get("schema_ref"),
        "validated_snapshot": efficiency_kpi_validation.get("validated_snapshot"),
        "source": "xe101_efficiency_kpi_baseline",
    },
    "evidence_trace_viewer": evidence_trace_viewer,
    "browser": browser_latest,
    "web_domain_guard": web_domain_guard,
    "web_scheduler": web_scheduler,
    "lock_break_latest": lock_break_latest,
    "gate_os": gate_os,
    "actions": actions,
    "evidence_refs": unique_preserve(
        [
            "state/continuity/current.json",
            "state/continuity/latest/blocker_registry.json",
            "state/continuity/latest/execution_program_status.json",
            "state/continuity/latest/execution_frontier_ledger.json",
            "state/continuity/latest/execution_supervisor_dispatch_intent_latest.json",
            "state/continuity/history/execution_supervisor_dispatch_intent_history.jsonl",
            "state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
            "state/continuity/history/execution_supervisor_dispatch_qualification_history.jsonl",
            "state/continuity/latest/execution_meaningful_event_reporting_latest.json",
            "state/continuity/latest/execution_meaningful_event_reporting_status_latest.json",
            "state/continuity/latest/execution_frontier_transition_attempt_latest.json",
            "state/continuity/history/execution_frontier_transition_attempts.jsonl",
            current_publish_lock_owner_rel,
            "state/continuity/latest/continuity_read_pointer.json",
            "state/continuity/latest/successor_safe_handover_proof.json",
            "state/continuity/latest/successor_safe_handover_proof_status.json",
            "state/continuity/latest/reset_ready_refresh_latest.json",
            "state/handover/latest.json",
            continuity_now_evidence_path,
            "state/continuity/latest/gate_os_latest.json",
            "state/continuity/latest/queue_replay_verify.json",
            "state/continuity/session_topology_router/decisions.jsonl",
            "state/continuity/model_rollout_dashboard/latest.json",
            "state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json",
            "state/continuity/latest/efficiency_kpi_baseline_latest.json",
            "state/continuity/latest/efficiency_kpi_baseline_validation_latest.json",
            "state/continuity/latest/xe101_efficiency_kpi_schema_v1.json",
            model_rollout_remediation_correction_cycle_log_ref,
            "state/continuity/latest/wave2_replay_evidence_index.json", 
            "state/continuity/latest/failover_stress_soak_evidence.json",
            "state/continuity/latest/failover_stress_runtime_evidence.json",
            failover_stress_runtime_decision_log_ref,
            "state/continuity/latest/load_shedding_decision.json",
            "state/continuity/latest/load_shedding_signal_snapshot.json",
            "state/continuity/latest/evidence_trace_viewer_latest.json",
            "state/continuity/latest/core_roadmap_execution_queue.json",
            "state/continuity/latest/core_roadmap_slice_queue_2026-03-28.json",
            "state/continuity/librarian/promotions.jsonl",
            "state/continuity/latest/lock_break_last.json",
            "ops/web_capture/artifacts/latest_run.json",
        ]
        + gtc_evidence_refs
        + web_guard_paths[:12]
        + [
            str(execution_status.get("autonomous_dispatch_trace_path") or "").strip(),
            str(execution_status.get("autonomous_dispatch_history_path") or "").strip(),
        ]
        + (["state/continuity/latest/no_nudge_idle_lane_autospawn_contradiction_latch.json"] if autopilot_idle_lane_contradiction_latched else [])
        + [
            str(web_scheduler.get("state_path") or "").strip(),
            str(verify_gate_preflight_layered_health.get("layered_health_snapshot_path") or "").strip(),
            str(verify_gate_preflight_layered_health.get("slo_snapshot_path") or "").strip(),
        ],
    ),
}

try:
    validate_mission_control_contract(payload)
except Exception as exc:
    raise SystemExit(str(exc))

atomic_write(export_path, payload)

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(
        "MISSION CONTROL "
        f"readiness={payload['headline']['readiness']} "
        f"mutation_gate={payload['headline']['mutation_gate']} "
        f"posture={payload['headline'].get('mutation_gate_posture') or 'unknown'} "
        f"wave={payload['headline'].get('execution_current_wave') if isinstance(payload['headline'].get('execution_current_wave'), int) else 'n/a'} "
        f"frontier={payload['headline'].get('execution_frontier_lane') or 'none'} "
        f"focus={payload['headline'].get('execution_current_focus') or 'none'} "
        f"blockers={payload['headline']['hard_blockers']} "
        f"actions={len(actions)}"
    )
    for action in actions[:5]:
        print(f"- [{action.get('priority')}] {action.get('action')}: {action.get('command')}")
PY
