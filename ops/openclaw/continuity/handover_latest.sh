#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
REFRESH=0
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: handover_latest.sh [options]

Render successor-oriented handover/latest.{json,md} and compute staleness vs continuity/current.json.

Options:
  --refresh     Recompute continuity current + handover surfaces
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
import shlex
import signal
import subprocess
import sys
import uuid
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
refresh = bool(int(sys.argv[2]))
json_out = bool(int(sys.argv[3]))

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
        DEFAULT_HANDOVER_FRESHNESS_MAX_AGE_SEC as _DEFAULT_HANDOVER_FRESHNESS_MAX_AGE_SEC,
        DRIFT_REASON_SET as _DRIFT_REASON_SET,
        continuity_now_contract_declared as _continuity_now_contract_declared,
        continuity_now_contract_expected_fields as _continuity_now_contract_expected_fields,
        continuity_now_contract_failclose_reasons as _continuity_now_contract_failclose_reasons,
        generation_pointer_core_failclose_reasons as _generation_pointer_core_failclose_reasons,
        is_severe_verify_gate_preflight_blocker as _policy_is_severe_verify_gate_preflight_blocker,
        read_nonnegative_int_env as _read_nonnegative_int_env,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _DEFAULT_HANDOVER_FRESHNESS_MAX_AGE_SEC = 1800
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }

    def _read_nonnegative_int_env(name: str, *, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(int(default)))))
        except Exception:
            return int(default)

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

try:
    from continuity_now_paths import (
        CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON as _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON,
        DEFAULT_CONTINUITY_NOW_LATEST_REL as _CONTINUITY_NOW_LATEST_REL,
        continuity_now_contract_path_conflict_reason as _continuity_now_contract_path_conflict_reason,
        resolve_continuity_now_contract_path as _resolve_continuity_now_contract_path,
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

handover_json_path = root / "state" / "handover" / "latest.json"
handover_md_path = root / "state" / "handover" / "latest.md"
compat_handover_json_path = root / "state" / "continuity" / "latest" / "handover_latest.json"
compat_handover_md_path = root / "state" / "continuity" / "latest" / "handover_latest.md"
latest_pointer_path = root / "state" / "continuity" / "latest" / "latest_pointer.json"
current_path = root / "state" / "continuity" / "current.json"
handover_schema_path = root / "ops" / "openclaw" / "architecture" / "schemas" / "handover_latest.schema.json"
proof_path = root / "state" / "continuity" / "latest" / "successor_safe_handover_proof.json"
proof_status_path = root / "state" / "continuity" / "latest" / "successor_safe_handover_proof_status.json"
proof_status_legacy_path = root / "state" / "continuity" / "latest" / "proof_status.json"

handover_freshness_max_age_sec = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_HANDOVER_FRESHNESS_MAX_AGE_SEC",
    default=_DEFAULT_HANDOVER_FRESHNESS_MAX_AGE_SEC,
)
_handover_proactive_refresh_lead_sec_raw = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_HANDOVER_PROACTIVE_REFRESH_LEAD_SEC",
    default=300,
)
# Clamp lead below freshness max-age so default posture cannot accidentally force
# refresh-on-every-read when max-age is small.
if handover_freshness_max_age_sec <= 1:
    handover_proactive_refresh_lead_sec = 0
else:
    handover_proactive_refresh_lead_sec = min(
        _handover_proactive_refresh_lead_sec_raw,
        handover_freshness_max_age_sec - 1,
    )


def read_positive_int_env(name: str, *, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(int(default)))))
    except Exception:
        return int(default)


# continuity_current --refresh routinely performs bounded verify/refresh hooks and
# can legitimately take longer than a tight 45s budget under lock contention.
# Keep fail-closed timeout semantics, but default to a safer bound aligned with
# reset_ready_refresh's current phase budget.
refresh_current_timeout_sec = read_positive_int_env(
    "OPENCLAW_HANDOVER_LATEST_REFRESH_CURRENT_TIMEOUT_SEC",
    default=120,
)
refresh_kill_grace_sec = read_positive_int_env(
    "OPENCLAW_HANDOVER_LATEST_REFRESH_KILL_GRACE_SEC",
    default=3,
)


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_bounded_command(cmd: List[str], *, timeout_sec: int) -> Dict[str, Any]:
    started_at_epoch = int(dt.datetime.now(dt.timezone.utc).timestamp())
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return {
            "ok": proc.returncode == 0,
            "timed_out": False,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_sec": timeout_sec,
            "command": shell_join(cmd),
            "started_at_epoch": started_at_epoch,
            "finished_at_epoch": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            stdout, stderr = proc.communicate(timeout=refresh_kill_grace_sec)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            stdout, stderr = proc.communicate()

        return {
            "ok": False,
            "timed_out": True,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_sec": timeout_sec,
            "command": shell_join(cmd),
            "started_at_epoch": started_at_epoch,
            "finished_at_epoch": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        }


def run_current_command_or_die(cmd: List[str], *, phase: str) -> Dict[str, Any]:
    run = run_bounded_command(cmd, timeout_sec=refresh_current_timeout_sec)
    if run.get("timed_out") is True:
        raise SystemExit(
            f"handover_latest_current_timeout:phase={phase}:"
            f"timeout_sec={refresh_current_timeout_sec}:cmd={run.get('command')}"
        )
    if not run.get("ok"):
        raw_stdout = str(run.get("stdout") or "")
        if raw_stdout.strip():
            try:
                nonzero_payload = json.loads(raw_stdout)
            except Exception:
                nonzero_payload = None
            if isinstance(nonzero_payload, dict):
                # continuity_current fail-close contract may intentionally return a
                # structured degraded payload with nonzero exit status.
                return run

        err = (run.get("stderr") or run.get("stdout") or "continuity_current_failed").strip()[:240]
        raise SystemExit(f"handover_latest_current_failed:phase={phase}:{err}")
    return run


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def clock_now_dt() -> dt.datetime:
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc)


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_object_output(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "")
    if not text.strip():
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_current_payload_from_run(
    run: Dict[str, Any],
    *,
    phase: str,
    refresh_requested: bool,
) -> Dict[str, Any]:
    try:
        payload = json.loads(run.get("stdout") or "{}")
    except Exception as exc:
        if not current_path.exists():
            raise SystemExit(
                f"handover_latest_current_invalid_json:phase={phase}:source=stdout:cache=missing:"
                f"error={type(exc).__name__}"
            )

        if refresh_requested:
            try:
                cache_mtime_epoch = int(current_path.stat().st_mtime)
            except Exception as cache_exc:
                raise SystemExit(
                    f"handover_latest_current_invalid_json:phase={phase}:source=stdout:cache=stat_failed:"
                    f"error={type(cache_exc).__name__}"
                )

            started_at_epoch = int(run.get("started_at_epoch") or 0)
            if started_at_epoch > 0 and cache_mtime_epoch < started_at_epoch:
                raise SystemExit(
                    f"handover_latest_current_invalid_json:phase={phase}:source=stdout:cache=stale_preexisting:"
                    f"path={current_path.relative_to(root)}"
                )

        try:
            payload = load_json(current_path)
        except json.JSONDecodeError as cache_exc:
            raise SystemExit(
                f"handover_latest_current_invalid_json:phase={phase}:source=stdout:cache=invalid_json:"
                f"error={type(cache_exc).__name__}"
            )
        except Exception as cache_exc:
            raise SystemExit(
                f"handover_latest_current_invalid_json:phase={phase}:source=stdout:cache=unreadable:"
                f"error={type(cache_exc).__name__}"
            )

        if not isinstance(payload, dict):
            raise SystemExit(
                f"handover_latest_current_invalid_payload:phase={phase}:source=cache:detail=not_object"
            )
        return payload

    if not isinstance(payload, dict):
        raise SystemExit(
            f"handover_latest_current_invalid_payload:phase={phase}:source=stdout:detail=not_object"
        )
    return payload


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validate_handover_contract(payload: Dict[str, Any]) -> None:
    if _validate_contract_payload_schema is None:
        raise RuntimeError(
            _format_schema_helper_unavailable_error(
                contract_prefix="handover_contract",
                helper_path=schema_helper_path,
                import_error=_schema_helper_import_error,
            )
        )

    _validate_contract_payload_schema(
        payload,
        schema_path=handover_schema_path,
        contract_prefix="handover_contract",
    )


def validate_reason_partition(*, not_ready_reasons: List[str], blocker_reasons: List[str], reconcile_only_reasons: List[str]) -> None:
    not_ready_set = set(not_ready_reasons)
    blocker_set = set(blocker_reasons)
    reconcile_set = set(reconcile_only_reasons)

    overlap = sorted(blocker_set & reconcile_set)
    combined = blocker_set | reconcile_set
    missing_from_partition = sorted(not_ready_set - combined)
    # continuity_now may intentionally reclassify drift-only not-ready posture
    # into reconcile_only_reasons while emitting not_ready_reasons=[].
    # Keep fail-close semantics for blocker taxonomy (blockers must still be a
    # subset of not_ready_reasons) while allowing reconcile-only advisory
    # reasons to live outside the not-ready set.
    extras_outside_not_ready = sorted(blocker_set - not_ready_set)

    if overlap or missing_from_partition or extras_outside_not_ready:
        raise RuntimeError(
            "handover_contract_reason_partition_invalid:"
            f"overlap={json.dumps(overlap, ensure_ascii=False)}:"
            f"missing_from_partition={json.dumps(missing_from_partition, ensure_ascii=False)}:"
            f"extras_outside_not_ready={json.dumps(extras_outside_not_ready, ensure_ascii=False)}"
        )


def _normalized_string_list(raw: Any) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def default_operator_working_doctrine() -> Dict[str, Any]:
    return {
        "schema": "clawd.operator_working_doctrine.v1",
        "session_role": "main_session_control_plane",
        "main_session_default": "orchestrate_and_keep_truth",
        "default_worker_lane": "subagent_default",
        "main_session_exception_path": "main_session_tiny_exception",
        "execution_mode": {
            "required": True,
            "allowed_values": ["EXECUTE_NOW", "PLAN_ONLY"],
            "default_without_action": "PLAN_ONLY",
            "spawn_before_speak_required": True,
        },
        "execution_tuple": {
            "required": True,
            "required_fields": ["execution_mode", "worker_lane", "model_selection"],
        },
        "dispatch_contract": {
            "required": True,
            "required_fields": [
                "task_class",
                "risk_tier",
                "scope_shape",
                "verification_class",
                "worker_topology",
                "fold_in_target",
            ],
        },
        "active_assumptions": [
            "main_session_stays_lean",
            "subagent_first_for_non_trivial_work",
            "deliberate_model_selection_required",
        ],
        "source_refs": [
            "docs/ops/unified_operating_doctrine_v1.md",
            "WORKFLOW_AUTO.md",
            "WORKING_PROTOCOL.md",
            "docs/ops/subagent_slot_fill_protocol_v1.md",
        ],
    }


def normalize_operator_working_doctrine(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    execution_mode = raw.get("execution_mode") if isinstance(raw.get("execution_mode"), dict) else {}
    execution_tuple = raw.get("execution_tuple") if isinstance(raw.get("execution_tuple"), dict) else {}
    dispatch_contract = raw.get("dispatch_contract") if isinstance(raw.get("dispatch_contract"), dict) else {}

    return {
        "schema": str(raw.get("schema") or "").strip() or None,
        "session_role": str(raw.get("session_role") or "").strip() or None,
        "main_session_default": str(raw.get("main_session_default") or "").strip() or None,
        "default_worker_lane": str(raw.get("default_worker_lane") or "").strip() or None,
        "main_session_exception_path": str(raw.get("main_session_exception_path") or "").strip() or None,
        "execution_mode": {
            "required": execution_mode.get("required") if isinstance(execution_mode.get("required"), bool) else None,
            "allowed_values": _normalized_string_list(execution_mode.get("allowed_values")),
            "default_without_action": str(execution_mode.get("default_without_action") or "").strip() or None,
            "spawn_before_speak_required": (
                execution_mode.get("spawn_before_speak_required")
                if isinstance(execution_mode.get("spawn_before_speak_required"), bool)
                else None
            ),
        },
        "execution_tuple": {
            "required": execution_tuple.get("required") if isinstance(execution_tuple.get("required"), bool) else None,
            "required_fields": _normalized_string_list(execution_tuple.get("required_fields")),
        },
        "dispatch_contract": {
            "required": dispatch_contract.get("required") if isinstance(dispatch_contract.get("required"), bool) else None,
            "required_fields": _normalized_string_list(dispatch_contract.get("required_fields")),
        },
        "active_assumptions": _normalized_string_list(raw.get("active_assumptions")),
        "source_refs": _normalized_string_list(raw.get("source_refs")),
    }


def normalize_dispatch_context(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    autonomous_dispatch_close_condition_met = raw.get("autonomous_dispatch_close_condition_met")
    if not isinstance(autonomous_dispatch_close_condition_met, bool):
        autonomous_dispatch_close_condition_met = None

    return {
        "status": str(raw.get("status") or "").strip() or None,
        "source": str(raw.get("source") or "").strip() or None,
        "autopilot_status": str(raw.get("autopilot_status") or "").strip() or None,
        "ready_work_exists": bool(raw.get("ready_work_exists") is True),
        "idle_threshold_exceeded": bool(raw.get("idle_threshold_exceeded") is True),
        "idle_sec": nonnegative_int(raw.get("idle_sec")),
        "target_step_id": str(raw.get("target_step_id") or "").strip() or None,
        "launched_step_id": str(raw.get("launched_step_id") or "").strip() or None,
        "skip_reason": str(raw.get("skip_reason") or "").strip() or None,
        "trace_path": str(raw.get("trace_path") or "").strip() or None,
        "updated_at": str(raw.get("updated_at") or "").strip() or None,
        "autonomous_dispatch_status": str(raw.get("autonomous_dispatch_status") or "missing").strip() or "missing",
        "autonomous_dispatch_decision": str(raw.get("autonomous_dispatch_decision") or "").strip() or None,
        "autonomous_dispatch_skip_reason": str(raw.get("autonomous_dispatch_skip_reason") or "").strip() or None,
        "autonomous_dispatch_block_reason": str(raw.get("autonomous_dispatch_block_reason") or "").strip() or None,
        "autonomous_dispatch_block_reasons": [
            str(reason or "").strip()
            for reason in (raw.get("autonomous_dispatch_block_reasons") if isinstance(raw.get("autonomous_dispatch_block_reasons"), list) else [])
            if str(reason or "").strip()
        ],
        "autonomous_dispatch_error": str(raw.get("autonomous_dispatch_error") or "").strip() or None,
        "autonomous_dispatch_updated_at": str(raw.get("autonomous_dispatch_updated_at") or "").strip() or None,
        "autonomous_dispatch_selector_state": str(raw.get("autonomous_dispatch_selector_state") or "").strip() or None,
        "autonomous_dispatch_close_condition_met": autonomous_dispatch_close_condition_met,
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


def normalize_execution_context(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    expected_in_flight_guard = raw.get("expected_in_flight_guard")
    if not isinstance(expected_in_flight_guard, bool):
        expected_in_flight_guard = None

    return {
        "posture": str(raw.get("posture") or "").strip() or None,
        "source": str(raw.get("source") or "").strip() or None,
        "readiness": str(raw.get("readiness") or "").strip() or None,
        "in_flight": bool(raw.get("in_flight") is True),
        "running_tasks": max(0, int(raw.get("running_tasks") or 0)),
        "active_locks": max(0, int(raw.get("active_locks") or 0)),
        "mutation_gate_status": str(raw.get("mutation_gate_status") or "").strip() or None,
        "mutation_gate_posture": str(raw.get("mutation_gate_posture") or "").strip() or None,
        "expected_in_flight_guard": expected_in_flight_guard,
        "dispatch_status": str(raw.get("dispatch_status") or "").strip() or None,
    }


def build_dispatch_context_from_idle_lane(idle_lane: Dict[str, Any]) -> Dict[str, Any]:
    raw_status = str(idle_lane.get("status") or "missing")
    ready_work_exists = bool(idle_lane.get("ready_work_exists") is True)
    idle_threshold_exceeded = bool(idle_lane.get("idle_threshold_exceeded") is True)
    target_step_id = str(idle_lane.get("target_step_id") or "").strip() or None
    launched_step_id = str(idle_lane.get("launched_step_id") or "").strip() or None
    skip_reason = str(idle_lane.get("skip_reason") or "").strip() or None
    trace_path = str(idle_lane.get("trace_path") or "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json")
    idle_sec = max(0, int(idle_lane.get("idle_sec") or 0))
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
        "running_tasks": max(0, int(in_flight.get("running_tasks") or 0)),
        "active_locks": max(0, int(in_flight.get("active_locks") or 0)),
        "mutation_gate_status": str(mutation_gate.get("status") or "unknown"),
        "mutation_gate_posture": str(mutation_gate.get("posture") or "unknown"),
        "expected_in_flight_guard": expected_in_flight_guard,
        "dispatch_status": dispatch_status,
    }


def mission_is_drift_reconcile_like(raw: Any) -> bool:
    txt = str(raw or "").strip().lower()
    return bool(txt and "reconcile continuity drift" in txt)


def checkpoint_mission_text(checkpoint_obj: Dict[str, Any]) -> str:
    objective = checkpoint_obj.get("objective") if isinstance(checkpoint_obj.get("objective"), dict) else {}
    return str(objective.get("primary_goal") or checkpoint_obj.get("objective") or "").strip()


def handover_checkpoint_is_ready_steady_state(checkpoint_obj: Dict[str, Any]) -> bool:
    metadata = checkpoint_obj.get("metadata") if isinstance(checkpoint_obj.get("metadata"), dict) else {}
    objective = checkpoint_obj.get("objective") if isinstance(checkpoint_obj.get("objective"), dict) else {}
    trigger = str(metadata.get("trigger") or "").strip()
    status = str(objective.get("status") or "").strip()
    mission = checkpoint_mission_text(checkpoint_obj)
    if trigger != "post_completion_closeout":
        return False
    if status != "READY":
        return False
    if not mission or mission_is_drift_reconcile_like(mission):
        return False
    return True


def dispatch_context_is_quiet_idle_no_candidate(dispatch_context: Dict[str, Any]) -> bool:
    selector_state = str(dispatch_context.get("autonomous_dispatch_selector_state") or "").strip()
    if selector_state != "idle_no_candidate":
        return False
    if str(dispatch_context.get("status") or "").strip() not in {"", "idle"}:
        return False
    if dispatch_context.get("autonomous_dispatch_close_condition_met") is False:
        return False
    if bool(dispatch_context.get("ready_work_exists") is True):
        return False
    if bool(dispatch_context.get("autonomous_dispatch_post_completion_enforcement_required") is True):
        return False
    if bool(dispatch_context.get("autonomous_dispatch_post_completion_enforcement_latched") is True):
        return False
    if bool(dispatch_context.get("autonomous_dispatch_intent_active") is True):
        return False
    if str(dispatch_context.get("autonomous_dispatch_skip_reason") or "").strip() not in {"", "autonomous_dispatch_not_eligible"}:
        return False
    benign_reasons = {"selector_state_not_ready_for_dispatch", "next_candidate_missing"}
    block_reasons = [
        str(reason or "").strip()
        for reason in (dispatch_context.get("autonomous_dispatch_block_reasons") if isinstance(dispatch_context.get("autonomous_dispatch_block_reasons"), list) else [])
        if str(reason or "").strip()
    ]
    if any(reason not in benign_reasons for reason in block_reasons):
        return False
    return True


def choose_handover_mission(
    *,
    checkpoint_obj: Dict[str, Any],
    preserved_handover_checkpoint: Dict[str, Any],
    execution_context: Dict[str, Any],
    dispatch_context: Dict[str, Any],
) -> str:
    steady_state_fallback_mission = "Continuity-safe resume and operator visibility"
    checkpoint_mission = checkpoint_mission_text(checkpoint_obj)
    if checkpoint_mission and not mission_is_drift_reconcile_like(checkpoint_mission):
        return checkpoint_mission

    if str(execution_context.get("readiness") or "").strip() != "READY":
        return checkpoint_mission or steady_state_fallback_mission
    if bool(execution_context.get("in_flight") is True):
        return checkpoint_mission or steady_state_fallback_mission
    if str(execution_context.get("mutation_gate_status") or "").strip() != "allowed":
        return checkpoint_mission or steady_state_fallback_mission
    if not dispatch_context_is_quiet_idle_no_candidate(dispatch_context):
        return checkpoint_mission or steady_state_fallback_mission

    preserved_mission = checkpoint_mission_text(preserved_handover_checkpoint)
    if handover_checkpoint_is_ready_steady_state(preserved_handover_checkpoint) and preserved_mission:
        return preserved_mission

    return steady_state_fallback_mission


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


def nonnegative_int(raw: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(raw or 0))
    except Exception:
        return int(default)


def optional_nonnegative_int(raw: Any) -> Optional[int]:
    try:
        return max(0, int(raw))
    except Exception:
        return None


def _tail_jsonl_objects(path: pathlib.Path, *, max_lines: int = 128) -> List[Dict[str, Any]]:
    if max_lines <= 0 or not path.exists() or not path.is_file():
        return []

    tail: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                tail.append(raw_line)
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for raw_line in tail:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def execution_frontier_status_streak(
    *,
    history_path_value: Any,
    latest_status: str,
    target_statuses: Iterable[str],
) -> int:
    target_status_set = {
        str(status or "").strip().lower()
        for status in target_statuses
        if str(status or "").strip()
    }
    latest_status_normalized = str(latest_status or "").strip().lower()
    if not target_status_set or latest_status_normalized not in target_status_set:
        return 0

    history_path_txt = str(history_path_value or "").strip()
    if not history_path_txt:
        return 1

    history_path = pathlib.Path(history_path_txt)
    if not history_path.is_absolute():
        history_path = (root / history_path).resolve()
    else:
        history_path = history_path.resolve()

    rows = _tail_jsonl_objects(history_path, max_lines=256)
    if not rows:
        return 1

    streak = 0
    for row in reversed(rows):
        status = str((row or {}).get("status") or "").strip().lower()
        if status in target_status_set:
            streak += 1
        else:
            break

    return max(1, streak)



def execution_frontier_blocked_streak(*, history_path_value: Any, latest_status: str) -> int:
    return execution_frontier_status_streak(
        history_path_value=history_path_value,
        latest_status=latest_status,
        target_statuses=["blocked"],
    )



def execution_frontier_error_streak(*, history_path_value: Any, latest_status: str) -> int:
    return execution_frontier_status_streak(
        history_path_value=history_path_value,
        latest_status=latest_status,
        target_statuses=["error"],
    )


def read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    txt = str(raw).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def load_json_if_exists(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = load_json(path)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _looks_like_checkpoint_payload(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    if str(metadata.get("checkpoint_id") or "").strip():
        return True
    return bool(str(objective.get("primary_goal") or "").strip())


def resolve_preserved_handover_checkpoint(path: pathlib.Path) -> Dict[str, Any]:
    obj = load_json_if_exists(path) or {}
    if _looks_like_checkpoint_payload(obj):
        return obj

    preserved = obj.get("preserved_handover_checkpoint") if isinstance(obj.get("preserved_handover_checkpoint"), dict) else {}
    if _looks_like_checkpoint_payload(preserved):
        return preserved
    return {}


def resolve_checkpoint_relpath(*, checkpoint_id: str) -> Optional[str]:
    pointer = load_json_if_exists(latest_pointer_path) or {}
    if checkpoint_id and str(pointer.get("checkpoint_id") or "").strip() == checkpoint_id:
        rel = str(pointer.get("json_path") or "").strip()
        if rel:
            return rel
    if checkpoint_id:
        fallback = root / "state" / "continuity" / "checkpoints" / f"{checkpoint_id}.json"
        if fallback.exists():
            return str(fallback.relative_to(root))
    return None


def _load_successor_proof_module(proof_module_path: pathlib.Path):
    if not proof_module_path.exists():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "successor_safe_handover_proof_runtime",
            proof_module_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def maybe_refresh_successor_proof(
    *,
    current_obj: Dict[str, Any],
    generation_pointer_obj: Dict[str, Any],
    current_sha: str,
    trigger: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "attempted": False,
        "refreshed": False,
        "proof_id": None,
        "proof_state": None,
        "top_blocker": None,
        "error": None,
    }
    module = _load_successor_proof_module(root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py")
    if module is None:
        result["error"] = "successor_safe_handover_proof_module_unavailable"
        return result

    expected_generation_id = str(
        (((current_obj.get("coherence") or {}).get("build_generation_id") or generation_pointer_obj.get("current_generation_id") or "")).strip()
    )

    try:
        proof = module.build_successor_safe_handover_proof(root=root, trigger=trigger)
        proof_eval = module.evaluate_proof_consumability(
            proof,
            mode="reset",
            now=now_iso(),
            expected_generation_id=expected_generation_id or None,
            expected_pointer_sha256=current_sha or None,
        )
        projected_status = module.project_proof_status(
            proof=proof,
            evaluation=proof_eval,
            evaluated_at=now_iso(),
        )
        module.write_proof_artifact(root=root, proof=proof)
        proof_status_path.parent.mkdir(parents=True, exist_ok=True)
        proof_status_path.write_text(
            json.dumps(projected_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.update(
            {
                "attempted": True,
                "refreshed": True,
                "proof_id": projected_status.get("proof_id") or proof.get("proof_id"),
                "proof_state": projected_status.get("proof_state"),
                "top_blocker": projected_status.get("top_blocker"),
            }
        )
        return result
    except Exception as exc:
        result["attempted"] = True
        result["error"] = str(exc)
        return result


def evaluate_successor_proof_gate(
    *,
    current_obj: Dict[str, Any],
    generation_pointer_obj: Dict[str, Any],
    current_sha: str,
) -> Dict[str, Any]:
    proof_obj = load_json_if_exists(proof_path)
    proof_status_obj = load_json_if_exists(proof_status_path)
    if proof_status_obj is None:
        proof_status_obj = load_json_if_exists(proof_status_legacy_path)

    gate_required = read_bool_env("OPENCLAW_HANDOVER_REQUIRE_SUCCESSOR_PROOF", default=True) or read_bool_env(
        "OPENCLAW_WAVE2_PROOF_GATE_REQUIRED",
        default=False,
    )
    gate_auto_on_presence = read_bool_env("OPENCLAW_HANDOVER_PROOF_GATE_AUTO", default=True)
    proof_status_max_age_sec = _read_nonnegative_int_env(
        "OPENCLAW_HANDOVER_PROOF_STATUS_MAX_AGE_SEC",
        default=300,
    )
    proof_present = isinstance(proof_obj, dict)
    status_present = isinstance(proof_status_obj, dict)
    gate_enforced = bool(gate_required or (gate_auto_on_presence and (proof_present or status_present)))

    expected_generation_id = str(
        (((current_obj.get("coherence") or {}).get("build_generation_id") or generation_pointer_obj.get("current_generation_id") or "")).strip()
    )

    proof_state = "PROOF_MISSING"
    proof_id = None
    proof_expires_at = None
    proof_top_blocker = "BLK_PROOF_MISSING"
    proof_effective_top_blocker = "BLK_PROOF_MISSING"
    proof_reset_allowed = False
    proof_resume_allowed = False

    proof_eval_blockers: List[str] = []
    proof_source_path = str(proof_path.relative_to(root)) if proof_path.exists() else str(proof_path.relative_to(root))
    proof_status_source_path = (
        str(proof_status_path.relative_to(root))
        if proof_status_path.exists()
        else (str(proof_status_legacy_path.relative_to(root)) if proof_status_legacy_path.exists() else str(proof_status_path.relative_to(root)))
    )

    module = _load_successor_proof_module(root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py")
    if module is not None and isinstance(proof_obj, dict):
        try:
            proof_eval = module.evaluate_proof_consumability(
                proof_obj,
                mode="reset",
                now=now_iso(),
                expected_generation_id=expected_generation_id or None,
                expected_pointer_sha256=current_sha,
            )
            proof_status_projected = module.project_proof_status(
                proof=proof_obj,
                evaluation=proof_eval,
                evaluated_at=now_iso(),
            )
            proof_state = str(proof_status_projected.get("proof_state") or proof_state)
            proof_id = proof_status_projected.get("proof_id") or proof_obj.get("proof_id")
            proof_expires_at = proof_obj.get("expires_at")
            proof_top_blocker = proof_status_projected.get("top_blocker") or proof_top_blocker
            proof_effective_top_blocker = proof_status_projected.get("effective_top_blocker") or proof_effective_top_blocker
            proof_reset_allowed = bool(proof_status_projected.get("reset_allowed"))
            proof_resume_allowed = bool(proof_status_projected.get("resume_allowed"))
            proof_eval_blockers = [str(item).strip() for item in (proof_status_projected.get("blockers") or proof_eval.get("blockers") or []) if str(item).strip()]
        except Exception:
            pass

    status_stale = False
    if isinstance(proof_status_obj, dict):
        status_evaluated_at = parse_iso(proof_status_obj.get("evaluated_at"))
        if status_evaluated_at is None:
            status_stale = True
        elif proof_status_max_age_sec > 0:
            status_stale = max(0, int((clock_now_dt() - status_evaluated_at).total_seconds())) > proof_status_max_age_sec

        if proof_state == "PROOF_MISSING":
            proof_state = str(proof_status_obj.get("proof_state") or proof_state)
        if not proof_id:
            proof_id = proof_status_obj.get("proof_id")
        if not proof_expires_at:
            proof_expires_at = proof_status_obj.get("expires_at")
        status_top_blocker = str(proof_status_obj.get("top_blocker") or "").strip() or None
        if status_top_blocker and (proof_top_blocker in {None, "", "BLK_PROOF_MISSING"} or proof_state != "PROOF_MISSING"):
            proof_top_blocker = status_top_blocker
        status_effective_top_blocker = str(proof_status_obj.get("effective_top_blocker") or "").strip() or None
        if status_effective_top_blocker:
            proof_effective_top_blocker = status_effective_top_blocker
        status_blockers = [str(item).strip() for item in (proof_status_obj.get("blockers") or []) if str(item).strip()]
        if status_blockers:
            proof_eval_blockers = unique_preserve(status_blockers + proof_eval_blockers)

    if proof_state == "PROOF_MISSING" and isinstance(proof_obj, dict):
        status_raw = str(proof_obj.get("status") or "").strip().upper()
        proof_id = proof_id or proof_obj.get("proof_id")
        proof_expires_at = proof_expires_at or proof_obj.get("expires_at")
        if status_raw == "ACTIVE":
            expires_dt = parse_iso(proof_obj.get("expires_at"))
            if expires_dt is not None and clock_now_dt() <= expires_dt:
                proof_state = "PROOF_VALID_PASS"
                proof_reset_allowed = True
                proof_resume_allowed = True
                proof_top_blocker = None
                proof_effective_top_blocker = None
            else:
                proof_state = "PROOF_EXPIRED"
                proof_top_blocker = "BLK_PROOF_EXPIRED"
                proof_effective_top_blocker = "BLK_PROOF_EXPIRED"
        elif status_raw == "REFUSED":
            proof_state = "PROOF_REFUSED"
            proof_top_blocker = "BLK_PROOF_REFUSED"
            proof_effective_top_blocker = proof_effective_top_blocker or "BLK_PROOF_REFUSED"
        elif status_raw == "INVALIDATED":
            proof_state = "PROOF_INVALID"
            proof_top_blocker = "BLK_PROOF_INVALIDATED"
            proof_effective_top_blocker = proof_effective_top_blocker or "BLK_PROOF_INVALIDATED"
        elif status_raw == "EXPIRED":
            proof_state = "PROOF_EXPIRED"
            proof_top_blocker = "BLK_PROOF_EXPIRED"
            proof_effective_top_blocker = proof_effective_top_blocker or "BLK_PROOF_EXPIRED"
        elif status_raw:
            proof_state = "PROOF_INVALID"
            proof_top_blocker = "BLK_PROOF_STALE_OR_INVALID"
            proof_effective_top_blocker = proof_effective_top_blocker or "BLK_PROOF_STALE_OR_INVALID"

    if status_present and status_stale and not proof_present:
        proof_eval_blockers.append("BLK_PROOF_STALE_OR_INVALID")

    if not proof_present:
        proof_eval_blockers.append("BLK_PROOF_MISSING")
        if proof_state == "PROOF_VALID_PASS":
            proof_state = "PROOF_VALID_BLOCKED"

    proof_eval_blockers = unique_preserve(proof_eval_blockers)

    if proof_eval_blockers and not proof_top_blocker:
        proof_top_blocker = proof_eval_blockers[0]
    if not proof_top_blocker and proof_state != "PROOF_VALID_PASS":
        proof_top_blocker = "BLK_PROOF_STALE_OR_INVALID"
    if proof_eval_blockers and not proof_effective_top_blocker:
        proof_effective_top_blocker = proof_eval_blockers[0]
    if not proof_effective_top_blocker and proof_state != "PROOF_VALID_PASS":
        proof_effective_top_blocker = proof_top_blocker or "BLK_PROOF_STALE_OR_INVALID"

    if proof_state == "PROOF_VALID_PASS":
        proof_top_blocker = None
        proof_effective_top_blocker = None
    else:
        proof_reset_allowed = False
        proof_resume_allowed = False

    return {
        "gate_enforced": gate_enforced,
        "gate_required": gate_required,
        "proof_present": proof_present,
        "status_present": status_present,
        "status_stale": status_stale,
        "proof_state": proof_state,
        "proof_id": proof_id,
        "expires_at": proof_expires_at,
        "top_blocker": proof_top_blocker,
        "effective_top_blocker": proof_effective_top_blocker,
        "blockers": proof_eval_blockers,
        "reset_allowed": proof_reset_allowed,
        "resume_allowed": proof_resume_allowed,
        "proof_source_path": proof_source_path,
        "status_source_path": proof_status_source_path,
    }


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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

    failclose_reasons, actual_sha, now_obj = _continuity_now_contract_failclose_reasons(
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

    if contract_declared and isinstance(now_obj, dict):
        now_obj["contract_source"] = "continuity_now_pinned_current_contract"
        now_obj["contract_source_canonical"] = True
        now_obj["contract_source_degraded"] = bool(failclose_reasons)
        now_obj["contract_source_degraded_reason"] = ";".join(failclose_reasons) if failclose_reasons else None
        now_obj["contract_source_degraded_path"] = contract_info["path"]
        contract_info["source"] = "current_contract"
        return now_obj, contract_info, unique_preserve(failclose_reasons)

    cp_now = subprocess.run(
        [str(root / "ops" / "openclaw" / "continuity" / "continuity_now.sh"), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if cp_now.returncode == 0:
        now_obj: Dict[str, Any] = json.loads(cp_now.stdout or "{}")
    else:
        # continuity_now may intentionally emit a structured fail-closed JSON
        # payload with nonzero exit status.
        nonzero_payload = parse_json_object_output(cp_now.stdout)
        now_obj = nonzero_payload if isinstance(nonzero_payload, dict) else {}
    if not isinstance(now_obj, dict):
        now_obj = {}
    now_obj.setdefault("contract_source", "live")
    now_obj.setdefault("contract_source_canonical", not contract_declared)
    now_obj.setdefault("contract_source_degraded", bool(failclose_reasons))
    now_obj.setdefault("contract_source_degraded_reason", ";".join(failclose_reasons) if failclose_reasons else None)
    now_obj.setdefault("contract_source_degraded_path", contract_info["path"] if failclose_reasons else None)
    return now_obj, contract_info, unique_preserve(failclose_reasons)


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
        "current_generation_id": None,
        "continuity_now_generation_id": None,
        "pointer_current_sha256": None,
        "current_sha256": current_sha,
        "pointer_current_generated_at": None,
        "current_generated_at": str(current_obj.get("generated_at") or "").strip() or None,
        "continuity_now_generated_at": str(now_obj.get("generated_at") or "").strip() or None,
        "failclose_reasons": [],
    }

    if not pointer_path.exists():
        return out

    try:
        pointer_obj = load_json(pointer_path)
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


def stale_report(
    current: Dict[str, Any],
    handover: Dict[str, Any],
    *,
    now_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diffs: List[str] = []
    cur_anchor = current.get("truth_anchor") or {}
    ho_anchor = handover.get("truth_anchor") or {}
    if str(cur_anchor.get("snapshot_id") or "") != str(ho_anchor.get("snapshot_id") or ""):
        diffs.append("snapshot_id")
    if str(cur_anchor.get("journal_offset") or "") != str(ho_anchor.get("journal_offset") or ""):
        diffs.append("journal_offset")
    if str(cur_anchor.get("pointer_hash") or "") != str(ho_anchor.get("pointer_hash") or ""):
        diffs.append("pointer_hash")

    cur_coh = current.get("coherence") or {}
    ho_coh = handover.get("coherence") or {}
    if str(cur_coh.get("tuple_hash") or "") != str(ho_coh.get("tuple_hash") or ""):
        diffs.append("coherence_tuple_hash")
    cur_policy_sig = str(((cur_coh.get("policy") or {}).get("signature") or ""))
    ho_policy_sig = str(((ho_coh.get("policy") or {}).get("signature") or ""))
    if cur_policy_sig != ho_policy_sig:
        diffs.append("policy_signature")

    cur_readiness = str(current.get("readiness") or "UNKNOWN")
    cur_in_flight = bool((current.get("in_flight") or {}).get("value") is True)
    cur_gate = current.get("mutation_gate") if isinstance(current.get("mutation_gate"), dict) else {}
    ho_gate = handover.get("mutation_gate") if isinstance(handover.get("mutation_gate"), dict) else {}
    cur_gate_status = str(cur_gate.get("status") or "unknown")
    ho_gate_status = str(ho_gate.get("status") or "unknown")
    if cur_gate_status != ho_gate_status:
        diffs.append("mutation_gate_status")

    cur_gate_posture = str(cur_gate.get("posture") or "unknown")
    ho_gate_posture = str(ho_gate.get("posture") or "unknown")
    if cur_gate_posture != ho_gate_posture:
        diffs.append("mutation_gate_posture")

    cur_gate_expected_guard = cur_gate.get("expected_in_flight_guard")
    ho_gate_expected_guard = ho_gate.get("expected_in_flight_guard")
    if cur_gate_expected_guard != ho_gate_expected_guard:
        diffs.append("mutation_gate_expected_in_flight_guard")

    cur_gate_blocking = sorted({str(x).strip() for x in (cur_gate.get("blocking_reasons") or []) if str(x).strip()})
    ho_gate_blocking = sorted({str(x).strip() for x in (ho_gate.get("blocking_reasons") or []) if str(x).strip()})
    if cur_gate_blocking != ho_gate_blocking:
        diffs.append("mutation_gate_blocking_reasons")

    cur_gate_concurrency = sorted({str(x).strip() for x in (cur_gate.get("concurrency_reasons") or []) if str(x).strip()})
    ho_gate_concurrency = sorted({str(x).strip() for x in (ho_gate.get("concurrency_reasons") or []) if str(x).strip()})
    if cur_gate_concurrency != ho_gate_concurrency:
        diffs.append("mutation_gate_concurrency_reasons")

    expected_operator_working_doctrine = normalize_operator_working_doctrine(current.get("operator_working_doctrine"))
    if not expected_operator_working_doctrine:
        expected_operator_working_doctrine = default_operator_working_doctrine()
    handover_operator_working_doctrine = normalize_operator_working_doctrine(handover.get("operator_working_doctrine"))
    if handover_operator_working_doctrine != expected_operator_working_doctrine:
        diffs.append("operator_working_doctrine")

    expected_execution_context = normalize_execution_context(current.get("execution_context"))
    if expected_execution_context:
        handover_execution_context = normalize_execution_context(handover.get("execution_context"))
        if handover_execution_context != expected_execution_context:
            diffs.append("execution_context")

    expected_dispatch_context = normalize_dispatch_context(current.get("dispatch_context"))
    if expected_dispatch_context:
        handover_dispatch_context = normalize_dispatch_context(handover.get("dispatch_context"))
        if handover_dispatch_context != expected_dispatch_context:
            diffs.append("dispatch_context")

    safe_signals = handover.get("safe_signals") if isinstance(handover.get("safe_signals"), dict) else {}
    handover_proof_status = handover.get("proof_status") if isinstance(handover.get("proof_status"), dict) else {}

    live_proof_status = load_json_if_exists(proof_status_path)
    if live_proof_status is None:
        live_proof_status = load_json_if_exists(proof_status_legacy_path)

    def _norm_nullable_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        txt = str(value).strip()
        return txt or None

    if isinstance(live_proof_status, dict):
        for field in ("proof_id", "proof_state", "top_blocker", "effective_top_blocker"):
            if _norm_nullable_text(handover_proof_status.get(field)) != _norm_nullable_text(live_proof_status.get(field)):
                diffs.append(f"proof_status.{field}")

        for field in ("reset_allowed", "resume_allowed"):
            if handover_proof_status.get(field) != live_proof_status.get(field):
                diffs.append(f"proof_status.{field}")

        if _norm_nullable_text(safe_signals.get("proof_id")) != _norm_nullable_text(live_proof_status.get("proof_id")):
            diffs.append("safe_signals.proof_id")
        if _norm_nullable_text(safe_signals.get("proof_state")) != _norm_nullable_text(live_proof_status.get("proof_state")):
            diffs.append("safe_signals.proof_state")
        if safe_signals.get("proof_reset_allowed") != live_proof_status.get("reset_allowed"):
            diffs.append("safe_signals.proof_reset_allowed")
        if safe_signals.get("proof_resume_allowed") != live_proof_status.get("resume_allowed"):
            diffs.append("safe_signals.proof_resume_allowed")
        if _norm_nullable_text(safe_signals.get("proof_effective_top_blocker")) != _norm_nullable_text(live_proof_status.get("effective_top_blocker")):
            diffs.append("safe_signals.proof_effective_top_blocker")

    autopilot_now = now_obj.get("autopilot") if isinstance(now_obj, dict) and isinstance(now_obj.get("autopilot"), dict) else {}
    controller_projection = (
        autopilot_now.get("execution_frontier_controller")
        if isinstance(autopilot_now.get("execution_frontier_controller"), dict)
        else {}
    )
    controller_projection_contract = (
        autopilot_now.get("execution_frontier_controller_contract")
        if isinstance(autopilot_now.get("execution_frontier_controller_contract"), dict)
        else {}
    )
    degraded_pending_projection = (
        autopilot_now.get("degraded_pending_stale_signal")
        if isinstance(autopilot_now.get("degraded_pending_stale_signal"), dict)
        else {}
    )
    idle_lane_projection = (
        autopilot_now.get("idle_lane_autospawn")
        if isinstance(autopilot_now.get("idle_lane_autospawn"), dict)
        else {}
    )

    dispatch_projection = expected_dispatch_context if isinstance(expected_dispatch_context, dict) else {}
    expected_controller_status = str(
        dispatch_projection.get("autonomous_dispatch_status")
        or controller_projection.get("status")
        or "missing"
    )
    expected_controller_decision = str(
        dispatch_projection.get("autonomous_dispatch_decision")
        or controller_projection.get("decision")
        or ""
    )
    expected_controller_block_reason = str(
        dispatch_projection.get("autonomous_dispatch_block_reason")
        or controller_projection.get("block_reason")
        or ""
    )
    expected_controller_block_reasons = sorted(
        {
            str(x).strip()
            for x in (
                dispatch_projection.get("autonomous_dispatch_block_reasons")
                or controller_projection.get("block_reasons")
                or []
            )
            if str(x).strip()
        }
    )
    expected_controller_source_degraded = bool(
        dispatch_projection.get("autonomous_dispatch_source_degraded") is True
        or controller_projection.get("contract_source_degraded") is True
        or str(controller_projection_contract.get("status") or "").strip().lower() == "degraded"
    )

    expected_controller_fresh = controller_projection_contract.get("fresh")
    if not isinstance(expected_controller_fresh, bool):
        fallback_controller_fresh = controller_projection.get("fresh")
        expected_controller_fresh = fallback_controller_fresh if isinstance(fallback_controller_fresh, bool) else None

    expected_controller_updated_age_sec = optional_nonnegative_int(
        controller_projection_contract.get("updated_age_sec")
    )
    if expected_controller_updated_age_sec is None:
        expected_controller_updated_age_sec = optional_nonnegative_int(
            controller_projection.get("updated_age_sec")
        )
    expected_controller_max_age_sec = optional_nonnegative_int(
        controller_projection_contract.get("max_age_sec")
    )
    if expected_controller_max_age_sec is None:
        expected_controller_max_age_sec = optional_nonnegative_int(
            controller_projection.get("max_age_sec")
        )

    expected_controller_blocked_streak = execution_frontier_blocked_streak(
        history_path_value=controller_projection.get("history_path"),
        latest_status=expected_controller_status,
    )
    expected_controller_error_streak = execution_frontier_error_streak(
        history_path_value=controller_projection.get("history_path"),
        latest_status=expected_controller_status,
    )
    controller_repeated_block_threshold = optional_nonnegative_int(
        os.environ.get("OPENCLAW_HANDOVER_CONTROLLER_REPEATED_BLOCK_STREAK_THRESHOLD")
    )
    if controller_repeated_block_threshold is None:
        controller_repeated_block_threshold = 3
    expected_controller_repeated_block = bool(
        controller_repeated_block_threshold > 0
        and expected_controller_blocked_streak >= controller_repeated_block_threshold
    )

    if str(safe_signals.get("execution_frontier_controller_status") or "missing") != expected_controller_status:
        diffs.append("execution_frontier_controller_status")
    if str(safe_signals.get("execution_frontier_controller_decision") or "") != expected_controller_decision:
        diffs.append("execution_frontier_controller_decision")
    if str(safe_signals.get("execution_frontier_controller_block_reason") or "") != expected_controller_block_reason:
        diffs.append("execution_frontier_controller_block_reason")

    handover_controller_block_reasons = sorted(
        {str(x).strip() for x in (safe_signals.get("execution_frontier_controller_block_reasons") or []) if str(x).strip()}
    )
    if handover_controller_block_reasons != expected_controller_block_reasons:
        diffs.append("execution_frontier_controller_block_reasons")

    if bool(safe_signals.get("execution_frontier_controller_source_degraded") is True) != expected_controller_source_degraded:
        diffs.append("execution_frontier_controller_source_degraded")

    handover_controller_fresh = safe_signals.get("execution_frontier_controller_fresh")
    if isinstance(expected_controller_fresh, bool) and isinstance(handover_controller_fresh, bool):
        if handover_controller_fresh != expected_controller_fresh:
            diffs.append("execution_frontier_controller_fresh")

    handover_controller_updated_age_sec = nonnegative_int(
        safe_signals.get("execution_frontier_controller_updated_age_sec")
    )
    if (
        expected_controller_updated_age_sec is not None
        and handover_controller_updated_age_sec != expected_controller_updated_age_sec
    ):
        diffs.append("execution_frontier_controller_updated_age_sec")

    handover_controller_max_age_sec = nonnegative_int(
        safe_signals.get("execution_frontier_controller_max_age_sec")
    )
    if (
        expected_controller_max_age_sec is not None
        and handover_controller_max_age_sec != expected_controller_max_age_sec
    ):
        diffs.append("execution_frontier_controller_max_age_sec")

    handover_controller_blocked_streak = optional_nonnegative_int(
        safe_signals.get("execution_frontier_controller_blocked_streak")
    )
    if (
        handover_controller_blocked_streak is not None
        and handover_controller_blocked_streak != expected_controller_blocked_streak
    ):
        diffs.append("execution_frontier_controller_blocked_streak")

    handover_controller_error_streak = optional_nonnegative_int(
        safe_signals.get("execution_frontier_controller_error_streak")
    )
    if (
        handover_controller_error_streak is not None
        and handover_controller_error_streak != expected_controller_error_streak
    ):
        diffs.append("execution_frontier_controller_error_streak")

    if bool(safe_signals.get("execution_frontier_controller_repeated_block") is True) != expected_controller_repeated_block:
        diffs.append("execution_frontier_controller_repeated_block")

    handover_controller_repeated_block_threshold = optional_nonnegative_int(
        safe_signals.get("execution_frontier_controller_repeated_block_threshold")
    )
    if (
        handover_controller_repeated_block_threshold is not None
        and handover_controller_repeated_block_threshold != controller_repeated_block_threshold
    ):
        diffs.append("execution_frontier_controller_repeated_block_threshold")

    expected_degraded_pending_active = bool(degraded_pending_projection.get("active") is True)
    if bool(safe_signals.get("degraded_pending_signal_active") is True) != expected_degraded_pending_active:
        diffs.append("degraded_pending_signal_active")

    expected_degraded_pending_stale_count = nonnegative_int(
        degraded_pending_projection.get("pending_stale_count")
    )
    handover_degraded_pending_stale_count = optional_nonnegative_int(
        safe_signals.get("degraded_pending_stale_count")
    )
    if (
        handover_degraded_pending_stale_count is not None
        and handover_degraded_pending_stale_count != expected_degraded_pending_stale_count
    ):
        diffs.append("degraded_pending_stale_count")

    expected_degraded_pending_total = nonnegative_int(
        degraded_pending_projection.get("pending_total")
    )
    handover_degraded_pending_total = optional_nonnegative_int(
        safe_signals.get("degraded_pending_total")
    )
    if (
        handover_degraded_pending_total is not None
        and handover_degraded_pending_total != expected_degraded_pending_total
    ):
        diffs.append("degraded_pending_total")

    expected_degraded_pending_activate_after_ticks = nonnegative_int(
        degraded_pending_projection.get("activate_after_ticks")
    )
    handover_degraded_pending_activate_after_ticks = optional_nonnegative_int(
        safe_signals.get("degraded_pending_activate_after_ticks")
    )
    if (
        handover_degraded_pending_activate_after_ticks is not None
        and handover_degraded_pending_activate_after_ticks != expected_degraded_pending_activate_after_ticks
    ):
        diffs.append("degraded_pending_activate_after_ticks")

    expected_degraded_pending_oldest_age_sec = nonnegative_int(
        degraded_pending_projection.get("pending_oldest_age_sec")
    )
    handover_degraded_pending_oldest_age_sec = optional_nonnegative_int(
        safe_signals.get("degraded_pending_oldest_age_sec")
    )
    if (
        handover_degraded_pending_oldest_age_sec is not None
        and handover_degraded_pending_oldest_age_sec != expected_degraded_pending_oldest_age_sec
    ):
        diffs.append("degraded_pending_oldest_age_sec")

    expected_degraded_pending_last_emit_iso = (
        str(degraded_pending_projection.get("last_emit_iso") or "").strip() or None
    )
    handover_degraded_pending_last_emit_iso_raw = safe_signals.get("degraded_pending_last_emit_iso")
    handover_degraded_pending_last_emit_iso = (
        str(handover_degraded_pending_last_emit_iso_raw).strip()
        if handover_degraded_pending_last_emit_iso_raw is not None
        else None
    )
    if handover_degraded_pending_last_emit_iso == "":
        handover_degraded_pending_last_emit_iso = None
    if (
        handover_degraded_pending_last_emit_iso is not None
        and handover_degraded_pending_last_emit_iso != expected_degraded_pending_last_emit_iso
    ):
        diffs.append("degraded_pending_last_emit_iso")

    expected_degraded_pending_streak = nonnegative_int(
        degraded_pending_projection.get("stale_ticks_consecutive")
    )
    handover_degraded_pending_streak = optional_nonnegative_int(
        safe_signals.get("degraded_pending_streak")
    )
    if (
        handover_degraded_pending_streak is not None
        and handover_degraded_pending_streak != expected_degraded_pending_streak
    ):
        diffs.append("degraded_pending_streak")

    idle_lane_ready_work_exists = bool(idle_lane_projection.get("ready_work_exists") is True)
    idle_lane_idle_threshold_exceeded = bool(idle_lane_projection.get("idle_threshold_exceeded") is True)
    idle_lane_skip_reason = str(idle_lane_projection.get("skip_reason") or "")
    idle_lane_contract_source_degraded = bool(idle_lane_projection.get("contract_source_degraded") is True)
    idle_lane_contradiction_abort_active = bool(idle_lane_projection.get("contradiction_abort_active") is True)
    expected_idle_lane_contradiction_latched = bool(
        idle_lane_ready_work_exists
        and idle_lane_idle_threshold_exceeded
        and not idle_lane_contract_source_degraded
        and (
            idle_lane_skip_reason == "contradiction_latched_auto_abort"
            or idle_lane_contradiction_abort_active
        )
    )
    if bool(safe_signals.get("idle_lane_autospawn_contradiction_latched") is True) != expected_idle_lane_contradiction_latched:
        diffs.append("idle_lane_autospawn_contradiction_latched")

    expected_idle_lane_contradiction_abort_remaining_sec = nonnegative_int(
        idle_lane_projection.get("contradiction_abort_remaining_sec")
    )
    handover_idle_lane_contradiction_abort_remaining_sec = nonnegative_int(
        safe_signals.get("idle_lane_autospawn_contradiction_abort_remaining_sec")
    )
    if handover_idle_lane_contradiction_abort_remaining_sec != expected_idle_lane_contradiction_abort_remaining_sec:
        diffs.append("idle_lane_autospawn_contradiction_abort_remaining_sec")

    expected_idle_lane_contradiction_latch_repaired = bool(
        idle_lane_projection.get("contradiction_latch_repaired") is True
    )
    if bool(safe_signals.get("idle_lane_autospawn_contradiction_latch_repaired") is True) != expected_idle_lane_contradiction_latch_repaired:
        diffs.append("idle_lane_autospawn_contradiction_latch_repaired")

    proof_gate_enforced = bool(safe_signals.get("proof_gate_enforced") is True)
    proof_resume_allowed = safe_signals.get("proof_resume_allowed")
    proof_reset_allowed = safe_signals.get("proof_reset_allowed")
    expected_safe_to_resume_base = bool(
        cur_gate_status == "allowed"
        and cur_readiness == "READY"
        and not cur_in_flight
    )
    expected_safe_to_reset_base = bool(
        cur_gate_status == "allowed"
        and cur_readiness in {"READY", "READY_WITH_DEBT"}
    )
    if proof_gate_enforced:
        expected_safe_to_resume = bool(expected_safe_to_resume_base and proof_resume_allowed is True)
        expected_safe_to_reset = bool(expected_safe_to_reset_base and proof_reset_allowed is True)
    else:
        expected_safe_to_resume = expected_safe_to_resume_base
        expected_safe_to_reset = expected_safe_to_reset_base

    if isinstance(safe_signals.get("safe_to_resume"), bool) and safe_signals.get("safe_to_resume") != expected_safe_to_resume:
        diffs.append("safe_to_resume_contradiction")
    if isinstance(safe_signals.get("safe_to_reset"), bool) and safe_signals.get("safe_to_reset") != expected_safe_to_reset:
        diffs.append("safe_to_reset_contradiction")

    handover_generated = parse_iso(handover.get("generated_at"))
    handover_age_sec = None
    if handover_generated is not None:
        handover_age_sec = max(0, int((clock_now_dt() - handover_generated).total_seconds()))

    handover_freshness_remaining_sec = None
    if handover_freshness_max_age_sec > 0 and handover_age_sec is not None:
        handover_freshness_remaining_sec = max(0, handover_freshness_max_age_sec - handover_age_sec)

    freshness_breach = bool(
        handover_freshness_max_age_sec > 0
        and handover_age_sec is not None
        and handover_age_sec > handover_freshness_max_age_sec
    )
    proactive_refresh_due = bool(
        not freshness_breach
        and handover_freshness_remaining_sec is not None
        and handover_proactive_refresh_lead_sec > 0
        and handover_freshness_remaining_sec <= handover_proactive_refresh_lead_sec
    )

    if freshness_breach:
        diffs.append("handover_freshness_age_breach")
    elif proactive_refresh_due:
        diffs.append("handover_freshness_proactive_refresh_due")

    stale = len(diffs) > 0
    return {
        "stale": stale,
        "diffs": diffs,
        "freshness": {
            "handover_age_sec": handover_age_sec,
            "handover_freshness_max_age_sec": handover_freshness_max_age_sec,
            "handover_freshness_remaining_sec": handover_freshness_remaining_sec,
            "proactive_refresh_lead_sec": handover_proactive_refresh_lead_sec,
            "proactive_refresh_due": proactive_refresh_due,
            "age_breach": freshness_breach,
        },
    }


current_cmd = [str(root / "ops" / "openclaw" / "continuity" / "continuity_current.sh"), "--json"]
current_phase = "current_refresh" if refresh else "current_read"
if refresh:
    current_cmd.insert(1, "--refresh")
current_run = run_current_command_or_die(current_cmd, phase=current_phase)
current = load_current_payload_from_run(
    current_run,
    phase=current_phase,
    refresh_requested=refresh,
)

auto_refreshed = False
stale_before = None
preloaded_now_obj: Optional[Dict[str, Any]] = None
preloaded_continuity_now_contract: Optional[Dict[str, Any]] = None
preloaded_continuity_now_contract_failclose_reasons: Optional[List[str]] = None

if handover_json_path.exists() and not refresh:
    handover = load_json(handover_json_path)
    (
        preloaded_now_obj,
        preloaded_continuity_now_contract,
        preloaded_continuity_now_contract_failclose_reasons,
    ) = load_continuity_now_for_current(current)
    st = stale_report(current, handover, now_obj=preloaded_now_obj)
    stale_before = st
    if not st["stale"] and handover_md_path.exists():
        payload = {
            "ok": True,
            "handover_json": str(handover_json_path.relative_to(root)),
            "handover_md": str(handover_md_path.relative_to(root)),
            "stale": False,
            "diffs": [],
            "truth_anchor": handover.get("truth_anchor"),
            "auto_refreshed": False,
            "freshness": st.get("freshness") if isinstance(st, dict) else None,
        }
        if json_out:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("HANDOVER LATEST: stale=False diffs=none")
        raise SystemExit(0)

    auto_refreshed = True

if preloaded_now_obj is not None:
    now_obj = preloaded_now_obj
    continuity_now_contract = (
        preloaded_continuity_now_contract
        if isinstance(preloaded_continuity_now_contract, dict)
        else {}
    )
    continuity_now_contract_failclose_reasons = (
        preloaded_continuity_now_contract_failclose_reasons
        if isinstance(preloaded_continuity_now_contract_failclose_reasons, list)
        else []
    )
else:
    now_obj, continuity_now_contract, continuity_now_contract_failclose_reasons = load_continuity_now_for_current(current)
queue = now_obj.get("queue") or {}
verify = now_obj.get("verify") or {}
parity = now_obj.get("parity") or {}
checkpoint = now_obj.get("checkpoint") or {}
preserved_handover_checkpoint = resolve_preserved_handover_checkpoint(compat_handover_json_path)
autopilot_now = now_obj.get("autopilot") if isinstance(now_obj.get("autopilot"), dict) else {}
idle_lane_autospawn = autopilot_now.get("idle_lane_autospawn") if isinstance(autopilot_now.get("idle_lane_autospawn"), dict) else {}
verify_gate_preflight = verify.get("gate_preflight") if isinstance(verify.get("gate_preflight"), dict) else {}
verify_gate_preflight_strict = verify_gate_preflight.get("strict_autonomy") if isinstance(verify_gate_preflight.get("strict_autonomy"), dict) else {}
verify_gate_preflight_predicted = verify_gate_preflight.get("predicted_gate") if isinstance(verify_gate_preflight.get("predicted_gate"), dict) else {}
verify_gate_preflight_status_evidence = verify_gate_preflight.get("status_evidence_gate") if isinstance(verify_gate_preflight.get("status_evidence_gate"), dict) else {}
verify_status_evidence_failure_reason = str(verify_gate_preflight_status_evidence.get("failure_reason") or "").strip()
verify_gate_preflight_blocker = str(verify_gate_preflight_predicted.get("predicted_blocker_reason") or "").strip()
current_mutation_gate = current.get("mutation_gate") if isinstance(current.get("mutation_gate"), dict) else {}
projection_mutation_gate = now_obj.get("mutation_gate_projection") if isinstance(now_obj.get("mutation_gate_projection"), dict) else {}


def _reason_list(raw: Any) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def _is_severe_verify_gate_preflight_blocker(reason: Optional[str]) -> bool:
    return bool(_policy_is_severe_verify_gate_preflight_blocker(reason))


if current_path.exists():
    current_source_sha = sha256_file(current_path)
else:
    current_source_sha = hashlib.sha256(
        (json.dumps(current, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()

generation_pointer = evaluate_generation_pointer_contract(
    current_obj=current,
    now_obj=now_obj,
    current_sha=current_source_sha,
)
generation_pointer["continuity_now_contract"] = continuity_now_contract
generation_failclose_reasons = unique_preserve(
    _reason_list(generation_pointer.get("failclose_reasons"))
    + _reason_list(continuity_now_contract_failclose_reasons)
)
generation_pointer["failclose_reasons"] = generation_failclose_reasons

proof_refresh_result: Optional[Dict[str, Any]] = None
if refresh or auto_refreshed:
    proof_refresh_result = maybe_refresh_successor_proof(
        current_obj=current,
        generation_pointer_obj=generation_pointer,
        current_sha=current_source_sha,
        trigger="handover_refresh" if refresh else "handover_auto_refresh",
    )

mutation_gate_status = str(current_mutation_gate.get("status") or projection_mutation_gate.get("status") or "unknown")
mutation_gate_posture = str(current_mutation_gate.get("posture") or projection_mutation_gate.get("posture") or "unknown")
expected_in_flight_guard = current_mutation_gate.get("expected_in_flight_guard")
if not isinstance(expected_in_flight_guard, bool):
    expected_in_flight_guard = projection_mutation_gate.get("expected_in_flight_guard")
mutation_gate_view = {
    "status": mutation_gate_status,
    "posture": mutation_gate_posture,
    "reason": _reason_list(current_mutation_gate.get("reason") or projection_mutation_gate.get("reason")),
    "blocking_reasons": _reason_list(current_mutation_gate.get("blocking_reasons") or projection_mutation_gate.get("blocking_reasons")),
    "concurrency_reasons": _reason_list(current_mutation_gate.get("concurrency_reasons") or projection_mutation_gate.get("concurrency_reasons")),
    "expected_in_flight_guard": expected_in_flight_guard if isinstance(expected_in_flight_guard, bool) else None,
}

if generation_failclose_reasons:
    mutation_gate_status = "forbidden"
    mutation_gate_posture = "blocker"
    mutation_gate_view["status"] = "forbidden"
    mutation_gate_view["posture"] = "blocker"
    mutation_gate_view["reason"] = unique_preserve(_reason_list(mutation_gate_view.get("reason")) + generation_failclose_reasons)
    mutation_gate_view["blocking_reasons"] = unique_preserve(_reason_list(mutation_gate_view.get("blocking_reasons")) + generation_failclose_reasons)

operator_working_doctrine = normalize_operator_working_doctrine(current.get("operator_working_doctrine"))
if not operator_working_doctrine:
    operator_working_doctrine = default_operator_working_doctrine()

effective_readiness = str(current.get("readiness") or "UNKNOWN")
if generation_failclose_reasons:
    effective_readiness = "RECONCILE_REQUIRED"

cmd_cont_current_refresh_json = shell_cmd_for("ops/openclaw/continuity.sh", "current", "--refresh", "--json")
cmd_cont_verify_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify", "--json")
cmd_cont_mission_json = shell_cmd_for("ops/openclaw/continuity.sh", "mission-control", "--json")
cmd_cont_verify_gate_status_json = shell_cmd_for("ops/openclaw/continuity.sh", "verify-gate-status", "--json")
cmd_read_pointer_json = cat_cmd_for("state/continuity/latest/continuity_read_pointer.json")
cmd_read_idle_lane_latch_json = cat_cmd_for("state/continuity/latest/no_nudge_idle_lane_autospawn_contradiction_latch.json")
cmd_watchdog_json = shell_cmd_for("ops/openclaw/run_no_nudge_continuity_watchdog.sh", "--json")
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
cmd_cont_sync = shell_cmd_for("ops/openclaw/continuity.sh", "sync")
cmd_cont_current = shell_cmd_for("ops/openclaw/continuity.sh", "current")
cmd_read_proof_json = cat_cmd_for("state/continuity/latest/successor_safe_handover_proof.json")
cmd_read_proof_status_json = cat_cmd_for("state/continuity/latest/successor_safe_handover_proof_status.json")

proof_gate = evaluate_successor_proof_gate(
    current_obj=current,
    generation_pointer_obj=generation_pointer,
    current_sha=current_source_sha,
)

next_actions: List[str] = []
checkpoint_rel = str(checkpoint.get("path") or "")
if checkpoint_rel:
    cp_path = (root / checkpoint_rel).resolve()
    if cp_path.exists():
        try:
            cp_obj = load_json(cp_path)
            for item in (cp_obj.get("execution_plan") or {}).get("next_actions") or []:
                cmd = item.get("command") if isinstance(item, dict) else str(item)
                if cmd and cmd not in next_actions:
                    next_actions.append(normalize_operator_command(cmd))
        except Exception:
            pass

if not next_actions:
    next_actions = [
        cmd_cont_current_refresh_json,
        cmd_cont_verify_json,
        cmd_cont_mission_json,
    ]

verify_gate_status_cmd = cmd_cont_verify_gate_status_json
if verify_gate_preflight_blocker and verify_gate_status_cmd not in next_actions:
    next_actions = [verify_gate_status_cmd, *next_actions]
if verify_status_evidence_failure_reason:
    verify_cmd = cmd_cont_verify_json
    if verify_cmd not in next_actions:
        next_actions = [verify_cmd, *next_actions]
    if verify_gate_status_cmd not in next_actions:
        next_actions = [verify_gate_status_cmd, *next_actions]

if generation_failclose_reasons:
    refresh_cmd = cmd_cont_current_refresh_json
    inspect_cmd = cmd_read_pointer_json
    if refresh_cmd not in next_actions:
        next_actions = [refresh_cmd, *next_actions]
    if inspect_cmd not in next_actions:
        next_actions.append(inspect_cmd)

proof_gate_enforced = bool(proof_gate.get("gate_enforced") is True)
proof_state = str(proof_gate.get("proof_state") or "PROOF_MISSING")
proof_top_blocker = str(proof_gate.get("top_blocker") or "").strip() or None
proof_reset_allowed = bool(proof_gate.get("reset_allowed") is True)
proof_resume_allowed = bool(proof_gate.get("resume_allowed") is True)
proof_blocked = bool(proof_state != "PROOF_VALID_PASS")

if proof_gate_enforced:
    if cmd_read_proof_json not in next_actions:
        next_actions.append(cmd_read_proof_json)
    if cmd_read_proof_status_json not in next_actions:
        next_actions.append(cmd_read_proof_status_json)
    if proof_blocked and cmd_cont_current_refresh_json not in next_actions:
        next_actions = [cmd_cont_current_refresh_json, *next_actions]

not_ready_reasons = [str(x) for x in (now_obj.get("not_ready_reasons") or []) if str(x).strip()]
for reason in generation_failclose_reasons:
    if reason not in not_ready_reasons:
        not_ready_reasons.append(reason)
proof_reason: Optional[str] = None
if proof_gate_enforced and proof_blocked:
    proof_reason = f"successor_proof_gate:{proof_state}"
    if proof_reason not in not_ready_reasons:
        not_ready_reasons.append(proof_reason)
warning_reasons = [str(x) for x in (now_obj.get("warning_reasons") or []) if str(x).strip()]
if (not proof_gate_enforced) and proof_blocked and proof_state != "PROOF_MISSING":
    proof_warning = f"successor_proof_observed:{proof_state}"
    if proof_warning not in warning_reasons:
        warning_reasons.append(proof_warning)
raw_blocker_reasons = [str(x) for x in (now_obj.get("blocker_reasons") or []) if str(x).strip()]
raw_reconcile_only_reasons = [str(x) for x in (now_obj.get("reconcile_only_reasons") or []) if str(x).strip()]
idle_lane_status = str(idle_lane_autospawn.get("status") or "missing")
idle_lane_ready_work_exists = bool(idle_lane_autospawn.get("ready_work_exists") is True)
idle_lane_idle_threshold_exceeded = bool(idle_lane_autospawn.get("idle_threshold_exceeded") is True)
idle_lane_trace_path = str(idle_lane_autospawn.get("trace_path") or "state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json")
idle_lane_target_step_id = str(idle_lane_autospawn.get("target_step_id") or "")
idle_lane_launched_step_id = str(idle_lane_autospawn.get("launched_step_id") or "")
idle_lane_skip_reason = str(idle_lane_autospawn.get("skip_reason") or "")
idle_lane_contract_source_degraded = bool(idle_lane_autospawn.get("contract_source_degraded") is True)
idle_lane_contradiction_abort_active = bool(idle_lane_autospawn.get("contradiction_abort_active") is True)
idle_lane_contradiction_abort_remaining_sec = int(idle_lane_autospawn.get("contradiction_abort_remaining_sec") or 0)
idle_lane_contradiction_latch_repaired = bool(idle_lane_autospawn.get("contradiction_latch_repaired") is True)
idle_lane_contradiction_latch_repair_reason = str(idle_lane_autospawn.get("contradiction_latch_repair_reason") or "")
autopilot_degraded_pending_signal = (
    autopilot_now.get("degraded_pending_stale_signal")
    if isinstance(autopilot_now.get("degraded_pending_stale_signal"), dict)
    else {}
)
autopilot_degraded_pending_signal_active = bool(autopilot_degraded_pending_signal.get("active") is True)
autopilot_degraded_pending_signal_stale_count = nonnegative_int(
    autopilot_degraded_pending_signal.get("pending_stale_count")
)
autopilot_degraded_pending_signal_total = nonnegative_int(
    autopilot_degraded_pending_signal.get("pending_total")
)
autopilot_degraded_pending_signal_streak = nonnegative_int(
    autopilot_degraded_pending_signal.get("stale_ticks_consecutive")
)
autopilot_degraded_pending_signal_activate_after_ticks = nonnegative_int(
    autopilot_degraded_pending_signal.get("activate_after_ticks")
)
autopilot_degraded_pending_signal_oldest_age_sec = nonnegative_int(
    autopilot_degraded_pending_signal.get("pending_oldest_age_sec")
)
autopilot_degraded_pending_signal_last_emit_iso = (
    str(autopilot_degraded_pending_signal.get("last_emit_iso") or "").strip() or None
)
autopilot_degraded_pending_signal_recovery_command = normalize_operator_command(
    autopilot_degraded_pending_signal.get("recovery_command") or ""
) or None
autopilot_degraded_pending_signal_inspect_command = normalize_operator_command(
    autopilot_degraded_pending_signal.get("inspect_command") or ""
) or None
autopilot_execution_frontier_controller = (
    autopilot_now.get("execution_frontier_controller")
    if isinstance(autopilot_now.get("execution_frontier_controller"), dict)
    else {}
)
autopilot_execution_frontier_controller_contract = (
    autopilot_now.get("execution_frontier_controller_contract")
    if isinstance(autopilot_now.get("execution_frontier_controller_contract"), dict)
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
autopilot_execution_frontier_controller_block_reasons = unique_preserve(
    [str(item).strip() for item in (autopilot_execution_frontier_controller.get("block_reasons") or []) if str(item).strip()]
)
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
    or str(autopilot_execution_frontier_controller_contract.get("status") or "").strip().lower() == "degraded"
)
autopilot_execution_frontier_controller_fresh = autopilot_execution_frontier_controller_contract.get("fresh")
if not isinstance(autopilot_execution_frontier_controller_fresh, bool):
    autopilot_execution_frontier_controller_fresh = (
        autopilot_execution_frontier_controller.get("fresh")
        if isinstance(autopilot_execution_frontier_controller.get("fresh"), bool)
        else None
    )
autopilot_execution_frontier_controller_updated_age_sec = optional_nonnegative_int(
    autopilot_execution_frontier_controller_contract.get("updated_age_sec")
)
if autopilot_execution_frontier_controller_updated_age_sec is None:
    autopilot_execution_frontier_controller_updated_age_sec = optional_nonnegative_int(
        autopilot_execution_frontier_controller.get("updated_age_sec")
    )
autopilot_execution_frontier_controller_max_age_sec = optional_nonnegative_int(
    autopilot_execution_frontier_controller_contract.get("max_age_sec")
)
if autopilot_execution_frontier_controller_max_age_sec is None:
    autopilot_execution_frontier_controller_max_age_sec = optional_nonnegative_int(
        autopilot_execution_frontier_controller.get("max_age_sec")
    )
autopilot_execution_frontier_controller_blocked = bool(
    autopilot_execution_frontier_controller_status == "blocked"
)
autopilot_execution_frontier_controller_error_state = bool(
    autopilot_execution_frontier_controller_status == "error"
)
autopilot_execution_frontier_controller_blocked_streak = execution_frontier_blocked_streak(
    history_path_value=autopilot_execution_frontier_controller_history_path,
    latest_status=autopilot_execution_frontier_controller_status,
)
autopilot_execution_frontier_controller_error_streak = execution_frontier_error_streak(
    history_path_value=autopilot_execution_frontier_controller_history_path,
    latest_status=autopilot_execution_frontier_controller_status,
)
controller_repeated_block_threshold = optional_nonnegative_int(
    os.environ.get("OPENCLAW_HANDOVER_CONTROLLER_REPEATED_BLOCK_STREAK_THRESHOLD")
)
if controller_repeated_block_threshold is None:
    controller_repeated_block_threshold = 3
autopilot_execution_frontier_controller_repeated_block = bool(
    controller_repeated_block_threshold > 0
    and autopilot_execution_frontier_controller_blocked_streak >= controller_repeated_block_threshold
)
cmd_read_execution_frontier_controller_trace_json = cat_cmd_for(autopilot_execution_frontier_controller_trace_path)
cmd_read_execution_frontier_controller_history_json = cat_cmd_for(autopilot_execution_frontier_controller_history_path)
dispatch_context = normalize_dispatch_context(current.get("dispatch_context"))
if not dispatch_context:
    dispatch_context = build_dispatch_context_from_idle_lane(idle_lane_autospawn)
execution_context = normalize_execution_context(current.get("execution_context"))
if not execution_context:
    execution_context = build_execution_context(
        readiness=effective_readiness,
        in_flight=current.get("in_flight") if isinstance(current.get("in_flight"), dict) else {},
        mutation_gate=mutation_gate_view,
        dispatch_context=dispatch_context,
    )
handover_mission = choose_handover_mission(
    checkpoint_obj=checkpoint,
    preserved_handover_checkpoint=preserved_handover_checkpoint,
    execution_context=execution_context,
    dispatch_context=dispatch_context,
)
idle_lane_contradiction_latched = bool(
    idle_lane_ready_work_exists
    and idle_lane_idle_threshold_exceeded
    and not idle_lane_contract_source_degraded
    and (
        idle_lane_skip_reason == "contradiction_latched_auto_abort"
        or idle_lane_contradiction_abort_active
    )
)
idle_lane_failure_like = idle_lane_status in {"tick_failed", "attempted_no_launch", "error"}
idle_lane_stalled = bool(
    idle_lane_failure_like
    and idle_lane_ready_work_exists
    and idle_lane_idle_threshold_exceeded
)
drift_reason_set = set(_DRIFT_REASON_SET)
if raw_blocker_reasons:
    blocker_reasons = raw_blocker_reasons
else:
    blocker_reasons = [reason for reason in not_ready_reasons if reason not in drift_reason_set]
if raw_reconcile_only_reasons:
    reconcile_only_reasons = raw_reconcile_only_reasons
else:
    reconcile_only_reasons = [reason for reason in not_ready_reasons if reason in drift_reason_set]

if generation_failclose_reasons:
    reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason not in generation_failclose_reasons]
    for reason in generation_failclose_reasons:
        if reason not in blocker_reasons:
            blocker_reasons.append(reason)

if proof_reason:
    reconcile_only_reasons = [reason for reason in reconcile_only_reasons if reason != proof_reason]
    if proof_reason not in blocker_reasons:
        blocker_reasons.append(proof_reason)

validate_reason_partition(
    not_ready_reasons=not_ready_reasons,
    blocker_reasons=blocker_reasons,
    reconcile_only_reasons=reconcile_only_reasons,
)

blockers: List[Dict[str, Any]] = []
for reason in blocker_reasons:
    blockers.append({"severity": "blocker", "owner": "sre_watchdog", "reason": reason, "next_check": "next_cycle"})
for reason in reconcile_only_reasons[:5]:
    blockers.append({"severity": "warn", "owner": "sre_watchdog", "reason": reason, "next_check": "after_reconcile"})

if mutation_gate_status == "forbidden" and mutation_gate_posture == "concurrency_guard":
    blockers.append(
        {
            "severity": "warn",
            "owner": "successor_guard",
            "reason": "mutation_gate:concurrency_guard",
            "next_check": "after_in_flight_clear",
        }
    )

if proof_blocked:
    blockers.append(
        {
            "severity": "blocker" if proof_gate_enforced else "warn",
            "owner": "successor_proof",
            "reason": f"successor_safe_handover_proof:{proof_state}",
            "next_check": "after_proof_refresh",
            "top_blocker": proof_top_blocker,
        }
    )

if idle_lane_stalled:
    blockers.append(
        {
            "severity": "warn",
            "owner": "autopilot_watchdog",
            "reason": "idle_lane_autospawn_stalled",
            "next_check": "run_no_nudge_watchdog",
        }
    )
    stalled_priority_cmds = [
        cmd_watchdog_json,
        cat_cmd_for(idle_lane_trace_path),
        cmd_idle_lane_watchdog_history_json,
    ]
    for cmd in reversed(stalled_priority_cmds):
        if cmd not in next_actions:
            next_actions.insert(0, cmd)

if idle_lane_contradiction_latched:
    blockers.append(
        {
            "severity": "warn",
            "owner": "autopilot_watchdog",
            "reason": "idle_lane_autospawn_contradiction_latched",
            "next_check": "run_no_nudge_watchdog",
        }
    )
    contradiction_priority_cmds = [
        cmd_watchdog_json,
        cat_cmd_for(idle_lane_trace_path),
        cmd_read_idle_lane_latch_json,
        cmd_idle_lane_watchdog_history_json,
    ]
    for cmd in reversed(contradiction_priority_cmds):
        if cmd not in next_actions:
            next_actions.insert(0, cmd)

if idle_lane_contradiction_latch_repaired:
    repair_priority_cmds = [
        cat_cmd_for(idle_lane_trace_path),
        cmd_idle_lane_watchdog_history_json,
    ]
    for cmd in reversed(repair_priority_cmds):
        if cmd not in next_actions:
            next_actions.insert(0, cmd)

execution_frontier_controller_degraded = bool(
    autopilot_execution_frontier_controller_blocked
    or autopilot_execution_frontier_controller_error_state
    or autopilot_execution_frontier_controller_source_degraded
    or autopilot_execution_frontier_controller_fresh is False
)
if execution_frontier_controller_degraded:
    controller_reason = (
        autopilot_execution_frontier_controller_block_reason
        or autopilot_execution_frontier_controller_error
        or (
            "execution_frontier_controller_trace_stale"
            if autopilot_execution_frontier_controller_fresh is False
            else "execution_frontier_controller_source_degraded"
        )
    )
    blockers.append(
        {
            "severity": "warn",
            "owner": "autopilot_watchdog",
            "reason": f"execution_frontier_controller:{controller_reason}",
            "next_check": "run_no_nudge_watchdog",
        }
    )
    controller_priority_cmds = [
        cmd_watchdog_json,
        cmd_read_execution_frontier_controller_trace_json,
        cmd_read_execution_frontier_controller_history_json,
    ]
    for cmd in reversed(controller_priority_cmds):
        if cmd not in next_actions:
            next_actions.insert(0, cmd)

if autopilot_degraded_pending_signal_active:
    blockers.append(
        {
            "severity": "warn",
            "owner": "queue_sync",
            "reason": "degraded_pending_stale_signal_active",
            "next_check": "queue_sync",
        }
    )
    degraded_pending_priority_cmds = [
        autopilot_degraded_pending_signal_recovery_command,
        autopilot_degraded_pending_signal_inspect_command,
    ]
    for cmd in reversed([x for x in degraded_pending_priority_cmds if x]):
        if cmd not in next_actions:
            next_actions.insert(0, cmd)

if verify_gate_preflight_blocker:
    blockers.append(
        {
            "severity": "blocker" if _is_severe_verify_gate_preflight_blocker(verify_gate_preflight_blocker) else "warn",
            "owner": "verify_gate",
            "reason": f"verify_gate_preflight:{verify_gate_preflight_blocker}",
            "next_check": "verify_gate_status",
        }
    )
if verify_status_evidence_failure_reason:
    blockers.append(
        {
            "severity": "warn",
            "owner": "verify_gate",
            "reason": f"verify_status_evidence:{verify_status_evidence_failure_reason}",
            "next_check": "verify",
        }
    )

for reason in warning_reasons[:5]:
    if reason == "verify_gate_preflight_blocker_predicted" and verify_gate_preflight_blocker:
        continue
    if reason.startswith("verify_status_evidence_") and verify_status_evidence_failure_reason:
        continue
    if reason == "idle_lane_autospawn_stalled" and not idle_lane_stalled:
        # Idle-lane stall warning is only actionable when failure-like + readiness preconditions are met.
        # When those preconditions are false, suppress this warning from blocker surfaces.
        continue
    if reason == "idle_lane_autospawn_contradiction_latched" and not idle_lane_contradiction_latched:
        # Contradiction-latched warning is only actionable when the auto-abort tuple is still active.
        continue
    blockers.append({"severity": "warn", "owner": "sre_watchdog", "reason": reason, "next_check": "next_cycle"})

status_counts = queue.get("status_counts") or {}
running_count = int(status_counts.get("RUNNING") or 0)
active_locks_count = int(queue.get("active_file_lock_count") or 0)

current_readiness = effective_readiness
current_in_flight = bool((current.get("in_flight") or {}).get("value") is True)
mutation_gate_allows = mutation_gate_status == "allowed"
safe_to_resume_base = bool(mutation_gate_allows and current_readiness == "READY" and not current_in_flight)
safe_to_reset_base = bool(mutation_gate_allows and current_readiness in {"READY", "READY_WITH_DEBT"})
if proof_gate_enforced:
    safe_to_resume = bool(safe_to_resume_base and proof_resume_allowed)
    safe_to_reset = bool(safe_to_reset_base and proof_reset_allowed)
else:
    safe_to_resume = safe_to_resume_base
    safe_to_reset = safe_to_reset_base

if safe_to_resume:
    forbidden_until = None
elif not mutation_gate_allows:
    forbidden_until = "mutation_gate_allowed"
elif current_readiness != "READY":
    forbidden_until = "readiness_green"
elif current_in_flight:
    forbidden_until = "in_flight_clear"
elif proof_gate_enforced and proof_blocked:
    forbidden_until = "successor_proof_valid"
else:
    forbidden_until = "resume_gates_green"

handover = {
    "schema": "claw.handover.v1",
    "generated_at": now_iso(),
    "handover_id": f"ho_{uuid.uuid4().hex[:16]}",
    "workspace_id": "clawd-architect",
    "producer": {
        "kind": "system",
        "session_id": os.environ.get("OPENCLAW_TARGET_SESSION_KEY") or "unknown",
        "agent_id": os.environ.get("OPENCLAW_AGENT_ID") or "codex-orchestrator-pro",
    },
    "truth_anchor": current.get("truth_anchor") or {},
    "coherence": current.get("coherence") or {},
    "generation_pointer": generation_pointer,
    "objective": {
        "mission": handover_mission,
        "success_criteria": "readiness=READY and mutation_gate=allowed",
        "deadline": None,
    },
    "mutation_gate": mutation_gate_view,
    "operator_working_doctrine": operator_working_doctrine,
    "checkpoint": {
        "id": checkpoint.get("id"),
        "path": resolve_checkpoint_relpath(checkpoint_id=str(checkpoint.get("id") or "")),
        "objective": checkpoint.get("objective"),
        "trigger": checkpoint.get("trigger"),
        "status": checkpoint.get("status"),
    },
    "preserved_handover_checkpoint": preserved_handover_checkpoint or None,
    "execution_context": execution_context,
    "dispatch_context": dispatch_context,
    "queue_state": {
        "status_counts": status_counts,
        "ready_count": int(queue.get("ready_count") or 0),
        "running_count": running_count,
        "dependency_blocked_count": int(queue.get("dependency_blocked_count") or 0),
    },
    "autopilot": {
        "idle_lane_autospawn": {
            "status": idle_lane_status,
            "stalled": idle_lane_stalled,
            "contradiction_latched": idle_lane_contradiction_latched,
            "contradiction_abort_active": idle_lane_contradiction_abort_active,
            "contradiction_abort_remaining_sec": idle_lane_contradiction_abort_remaining_sec,
            "contradiction_latch_repaired": idle_lane_contradiction_latch_repaired,
            "contradiction_latch_repair_reason": idle_lane_contradiction_latch_repair_reason or None,
            "ready_work_exists": idle_lane_ready_work_exists,
            "idle_threshold_exceeded": idle_lane_idle_threshold_exceeded,
            "idle_sec": int(idle_lane_autospawn.get("idle_sec") or 0),
            "target_step_id": idle_lane_target_step_id or None,
            "launched_step_id": idle_lane_launched_step_id or None,
            "skip_reason": idle_lane_skip_reason or None,
            "trace_path": idle_lane_trace_path,
            "updated_at": idle_lane_autospawn.get("updated_at"),
        },
        "degraded_pending_stale_signal": {
            "active": autopilot_degraded_pending_signal_active,
            "pending_stale_count": autopilot_degraded_pending_signal_stale_count,
            "pending_total": autopilot_degraded_pending_signal_total,
            "stale_ticks_consecutive": autopilot_degraded_pending_signal_streak,
            "activate_after_ticks": autopilot_degraded_pending_signal_activate_after_ticks,
            "pending_oldest_age_sec": autopilot_degraded_pending_signal_oldest_age_sec,
            "last_emit_iso": autopilot_degraded_pending_signal_last_emit_iso,
            "inspect_command": autopilot_degraded_pending_signal_inspect_command,
            "recovery_command": autopilot_degraded_pending_signal_recovery_command,
        },
        "execution_frontier_controller": {
            "status": autopilot_execution_frontier_controller_status,
            "decision": autopilot_execution_frontier_controller_decision,
            "skip_reason": autopilot_execution_frontier_controller_skip_reason,
            "block_reason": autopilot_execution_frontier_controller_block_reason,
            "block_reasons": autopilot_execution_frontier_controller_block_reasons,
            "error": autopilot_execution_frontier_controller_error,
            "source_degraded": autopilot_execution_frontier_controller_source_degraded,
            "fresh": autopilot_execution_frontier_controller_fresh,
            "recorded_at": autopilot_execution_frontier_controller_recorded_at,
            "updated_age_sec": autopilot_execution_frontier_controller_updated_age_sec,
            "max_age_sec": autopilot_execution_frontier_controller_max_age_sec,
            "trace_path": autopilot_execution_frontier_controller_trace_path,
            "history_path": autopilot_execution_frontier_controller_history_path,
            "blocked_streak": autopilot_execution_frontier_controller_blocked_streak,
            "error_streak": autopilot_execution_frontier_controller_error_streak,
            "repeated_block": autopilot_execution_frontier_controller_repeated_block,
            "repeated_block_threshold": controller_repeated_block_threshold,
        },
    },
    "blockers": blockers,
    "active_locks": {
        "count": active_locks_count,
        "stale_count": int(queue.get("stale_active_file_lock_count") or 0),
        "examples": queue.get("stale_active_lock_examples") or [],
    },
    "validators": {
        "status": verify.get("status") or (current.get("validators") or {}).get("status"),
        "timestamp": verify.get("timestamp"),
        "age_sec": verify.get("age_sec"),
        "verify_gate_ready_to_run": verify_gate_preflight_predicted.get("ready_to_run") if isinstance(verify_gate_preflight_predicted.get("ready_to_run"), bool) else None,
        "verify_gate_predicted_blocker_reason": verify_gate_preflight_blocker or None,
        "verify_status_evidence_failure_reason": verify_status_evidence_failure_reason or None,
        "verify_status_evidence_fresh": verify_gate_preflight_status_evidence.get("fresh") if isinstance(verify_gate_preflight_status_evidence.get("fresh"), bool) else None,
    },
    "verify_gate_preflight": verify_gate_preflight,
    "parity": {
        "status": parity.get("status"),
        "fresh": parity.get("fresh"),
        "due": parity.get("due"),
        "last_done_at": parity.get("last_done_at"),
        "last_done_age_sec": parity.get("last_done_age_sec"),
    },
    "proof_status": {
        "proof_gate_enforced": proof_gate_enforced,
        "proof_gate_required": bool(proof_gate.get("gate_required") is True),
        "proof_state": proof_state,
        "proof_id": proof_gate.get("proof_id"),
        "expires_at": proof_gate.get("expires_at"),
        "top_blocker": proof_top_blocker,
        "effective_top_blocker": proof_gate.get("effective_top_blocker"),
        "blockers": proof_gate.get("blockers") or [],
        "reset_allowed": proof_reset_allowed,
        "resume_allowed": proof_resume_allowed,
        "proof_source_path": proof_gate.get("proof_source_path"),
        "status_source_path": proof_gate.get("status_source_path"),
    },
    "safe_signals": {
        "safe_to_resume": safe_to_resume,
        "safe_to_reset": safe_to_reset,
        "forbidden_until": forbidden_until,
        "proof_gate_enforced": proof_gate_enforced,
        "proof_state": proof_state,
        "proof_id": proof_gate.get("proof_id"),
        "proof_expires_at": proof_gate.get("expires_at"),
        "proof_top_blocker": proof_top_blocker,
        "proof_effective_top_blocker": proof_gate.get("effective_top_blocker"),
        "proof_blockers": proof_gate.get("blockers") or [],
        "proof_reset_allowed": proof_reset_allowed,
        "proof_resume_allowed": proof_resume_allowed,
        "mutation_gate_posture": mutation_gate_posture,
        "mutation_gate_expected_in_flight_guard": mutation_gate_view.get("expected_in_flight_guard"),
        "execution_frontier_controller_status": autopilot_execution_frontier_controller_status,
        "execution_frontier_controller_decision": autopilot_execution_frontier_controller_decision,
        "execution_frontier_controller_skip_reason": autopilot_execution_frontier_controller_skip_reason,
        "execution_frontier_controller_block_reason": autopilot_execution_frontier_controller_block_reason,
        "execution_frontier_controller_block_reasons": autopilot_execution_frontier_controller_block_reasons,
        "execution_frontier_controller_error": autopilot_execution_frontier_controller_error,
        "execution_frontier_controller_source_degraded": autopilot_execution_frontier_controller_source_degraded,
        "execution_frontier_controller_fresh": autopilot_execution_frontier_controller_fresh,
        "execution_frontier_controller_recorded_at": autopilot_execution_frontier_controller_recorded_at,
        "execution_frontier_controller_updated_age_sec": autopilot_execution_frontier_controller_updated_age_sec,
        "execution_frontier_controller_max_age_sec": autopilot_execution_frontier_controller_max_age_sec,
        "execution_frontier_controller_blocked_streak": autopilot_execution_frontier_controller_blocked_streak,
        "execution_frontier_controller_error_streak": autopilot_execution_frontier_controller_error_streak,
        "execution_frontier_controller_repeated_block": autopilot_execution_frontier_controller_repeated_block,
        "execution_frontier_controller_repeated_block_threshold": controller_repeated_block_threshold,
        "degraded_pending_signal_active": autopilot_degraded_pending_signal_active,
        "degraded_pending_stale_count": autopilot_degraded_pending_signal_stale_count,
        "degraded_pending_total": autopilot_degraded_pending_signal_total,
        "degraded_pending_streak": autopilot_degraded_pending_signal_streak,
        "degraded_pending_activate_after_ticks": autopilot_degraded_pending_signal_activate_after_ticks,
        "degraded_pending_oldest_age_sec": autopilot_degraded_pending_signal_oldest_age_sec,
        "degraded_pending_last_emit_iso": autopilot_degraded_pending_signal_last_emit_iso,
        "idle_lane_autospawn_status": idle_lane_status,
        "idle_lane_autospawn_stalled": idle_lane_stalled,
        "idle_lane_autospawn_contradiction_latched": idle_lane_contradiction_latched,
        "idle_lane_autospawn_contradiction_abort_remaining_sec": idle_lane_contradiction_abort_remaining_sec,
        "idle_lane_autospawn_contradiction_latch_repaired": idle_lane_contradiction_latch_repaired,
    },
    "replay": {
        "last_good_checkpoint": checkpoint.get("id"),
        "from_checkpoint": {
            "cmd": cmd_cont_sync,
            "args": ["--checkpoint", checkpoint.get("id")] if checkpoint.get("id") else [],
        },
        "to_truth_anchor": {
            "cmd": cmd_cont_current,
            "args": ["--refresh", "--json"],
        },
    },
    "recommended_next_actions": [
        {
            "step": idx + 1,
            "precondition": "mutation_gate=forbidden" if idx == 0 else "previous_step_done",
            "command": cmd,
        }
        for idx, cmd in enumerate(next_actions[:7])
    ],
    "notes": (
        "Generated from canonical continuity surfaces; advisory only."
        if not generation_failclose_reasons
        else "Generated from canonical continuity surfaces; fail-closed on generation-pointer contract mismatch."
    ),
}

st = stale_report(current, handover, now_obj=now_obj)

md_lines = [
    "## Objective",
    f"- Mission: `{handover['objective']['mission']}`",
    "- Success condition: `readiness=READY and mutation_gate=allowed`",
    "",
    "## Current truth anchor",
    f"- snapshot_id: `{(handover.get('truth_anchor') or {}).get('snapshot_id') or 'n/a'}`",
    f"- journal_offset: `{(handover.get('truth_anchor') or {}).get('journal_offset') or 'n/a'}`",
    f"- pointer_hash: `{(handover.get('truth_anchor') or {}).get('pointer_hash') or 'n/a'}`",
    f"- coherence_tuple_hash: `{(handover.get('coherence') or {}).get('tuple_hash') or 'n/a'}`",
    "",
    "## Generation pointer contract",
    f"- present: `{bool((handover.get('generation_pointer') or {}).get('present'))}`",
    f"- failclose_reasons: `{(handover.get('generation_pointer') or {}).get('failclose_reasons') or []}`",
    f"- current_generation: `{(handover.get('generation_pointer') or {}).get('current_generation_id') or 'n/a'}`",
    f"- pointer_generation: `{(handover.get('generation_pointer') or {}).get('pointer_generation_id') or 'n/a'}`",
    "",
    "## What is in flight",
    f"- running_tasks: `{running_count}`",
    f"- active_locks: `{active_locks_count}`",
    "",
    "## Mutation gate posture",
    f"- status: `{mutation_gate_status}`",
    f"- posture: `{mutation_gate_posture}`",
    f"- expected_in_flight_guard: `{mutation_gate_view.get('expected_in_flight_guard')}`",
    f"- blocking_reasons: `{mutation_gate_view.get('blocking_reasons') or []}`",
    f"- concurrency_reasons: `{mutation_gate_view.get('concurrency_reasons') or []}`",
    "",
    "## Operator working doctrine",
    f"- session_role: `{operator_working_doctrine.get('session_role') or 'n/a'}`",
    f"- main_session_default: `{operator_working_doctrine.get('main_session_default') or 'n/a'}`",
    f"- default_worker_lane: `{operator_working_doctrine.get('default_worker_lane') or 'n/a'}`",
    f"- main_session_exception_path: `{operator_working_doctrine.get('main_session_exception_path') or 'n/a'}`",
    f"- execution_modes: `{((operator_working_doctrine.get('execution_mode') or {}).get('allowed_values')) or []}` default_without_action=`{((operator_working_doctrine.get('execution_mode') or {}).get('default_without_action')) or 'n/a'}` spawn_before_speak_required=`{((operator_working_doctrine.get('execution_mode') or {}).get('spawn_before_speak_required'))}`",
    f"- required_execution_tuple_fields: `{((operator_working_doctrine.get('execution_tuple') or {}).get('required_fields')) or []}`",
    f"- required_dispatch_fields: `{((operator_working_doctrine.get('dispatch_contract') or {}).get('required_fields')) or []}`",
    f"- active_assumptions: `{operator_working_doctrine.get('active_assumptions') or []}`",
    "",
    "## Live execution context",
    f"- posture: `{execution_context.get('posture') or 'n/a'}`",
    f"- readiness: `{execution_context.get('readiness') or 'n/a'}`",
    f"- in_flight: `{execution_context.get('in_flight')}` running_tasks=`{execution_context.get('running_tasks')}` active_locks=`{execution_context.get('active_locks')}`",
    f"- mutation_gate: `status={execution_context.get('mutation_gate_status') or 'unknown'} posture={execution_context.get('mutation_gate_posture') or 'unknown'} expected_in_flight_guard={execution_context.get('expected_in_flight_guard')}`",
    "",
    "## Live dispatch context",
    f"- status: `{dispatch_context.get('status') or 'n/a'}` raw_autopilot_status=`{dispatch_context.get('autopilot_status') or 'n/a'}`",
    f"- target_step_id: `{dispatch_context.get('target_step_id') or 'none'}` launched_step_id=`{dispatch_context.get('launched_step_id') or 'none'}`",
    f"- ready_work_exists: `{dispatch_context.get('ready_work_exists')}` idle_threshold_exceeded=`{dispatch_context.get('idle_threshold_exceeded')}` idle_sec=`{dispatch_context.get('idle_sec')}`",
    f"- skip_reason: `{dispatch_context.get('skip_reason') or 'none'}` trace_path=`{dispatch_context.get('trace_path') or 'n/a'}`",
    "",
    "## Successor packet freshness",
    f"- handover_age_sec: `{(st.get('freshness') or {}).get('handover_age_sec') if isinstance(st, dict) else 'n/a'}`",
    f"- handover_freshness_max_age_sec: `{(st.get('freshness') or {}).get('handover_freshness_max_age_sec') if isinstance(st, dict) else 'n/a'}`",
    f"- handover_freshness_remaining_sec: `{(st.get('freshness') or {}).get('handover_freshness_remaining_sec') if isinstance(st, dict) else 'n/a'}`",
    f"- proactive_refresh_lead_sec: `{(st.get('freshness') or {}).get('proactive_refresh_lead_sec') if isinstance(st, dict) else 'n/a'}`",
    f"- proactive_refresh_due: `{(st.get('freshness') or {}).get('proactive_refresh_due') if isinstance(st, dict) else 'n/a'}` age_breach=`{(st.get('freshness') or {}).get('age_breach') if isinstance(st, dict) else 'n/a'}`",
    "",
    "## Successor proof gate",
    f"- enforced: `{proof_gate_enforced}`",
    f"- proof_state: `{proof_state}`",
    f"- proof_id: `{proof_gate.get('proof_id') or 'n/a'}`",
    f"- expires_at: `{proof_gate.get('expires_at') or 'n/a'}`",
    f"- top_blocker: `{proof_top_blocker or 'none'}`",
    f"- reset_allowed: `{proof_reset_allowed}` resume_allowed: `{proof_resume_allowed}`",
    "",
    "## Idle-lane autospawn",
    f"- status: `{idle_lane_status}`",
    f"- stalled: `{idle_lane_stalled}`",
    f"- contradiction_latched: `{idle_lane_contradiction_latched}`",
    f"- contradiction_abort_remaining_sec: `{idle_lane_contradiction_abort_remaining_sec}`",
    f"- contradiction_latch_repaired: `{idle_lane_contradiction_latch_repaired}`",
    f"- contradiction_latch_repair_reason: `{idle_lane_contradiction_latch_repair_reason or 'none'}`",
    f"- ready_work_exists: `{idle_lane_ready_work_exists}`",
    f"- idle_threshold_exceeded: `{idle_lane_idle_threshold_exceeded}`",
    f"- target_step_id: `{idle_lane_target_step_id or 'none'}`",
    f"- launched_step_id: `{idle_lane_launched_step_id or 'none'}`",
    "",
    "## Execution-frontier controller signal",
    f"- status: `{autopilot_execution_frontier_controller_status}` decision=`{autopilot_execution_frontier_controller_decision or 'none'}`",
    f"- block_reason: `{autopilot_execution_frontier_controller_block_reason or 'none'}` block_reasons=`{autopilot_execution_frontier_controller_block_reasons}`",
    f"- source_degraded: `{autopilot_execution_frontier_controller_source_degraded}` fresh=`{autopilot_execution_frontier_controller_fresh}`",
    f"- blocked_streak: `{autopilot_execution_frontier_controller_blocked_streak}` error_streak=`{autopilot_execution_frontier_controller_error_streak}` repeated_block=`{autopilot_execution_frontier_controller_repeated_block}` threshold=`{controller_repeated_block_threshold}` updated_age_sec=`{autopilot_execution_frontier_controller_updated_age_sec if autopilot_execution_frontier_controller_updated_age_sec is not None else 'n/a'}` max_age_sec=`{autopilot_execution_frontier_controller_max_age_sec if autopilot_execution_frontier_controller_max_age_sec is not None else 'n/a'}`",
    f"- trace_path: `{autopilot_execution_frontier_controller_trace_path}`",
    "",
    "## Degraded pending backlog signal",
    f"- active: `{autopilot_degraded_pending_signal_active}`",
    f"- stale_count: `{autopilot_degraded_pending_signal_stale_count}` total=`{autopilot_degraded_pending_signal_total}`",
    f"- streak: `{autopilot_degraded_pending_signal_streak}` activate_after_ticks=`{autopilot_degraded_pending_signal_activate_after_ticks}`",
    f"- oldest_age_sec: `{autopilot_degraded_pending_signal_oldest_age_sec}` last_emit_iso=`{autopilot_degraded_pending_signal_last_emit_iso or 'none'}`",
    "",
    "## Verify-gate preflight",
    f"- available: `{bool(verify_gate_preflight.get('available') is True)}`",
    f"- strict_mode: `enabled={bool(verify_gate_preflight_strict.get('enabled') is True)} source={verify_gate_preflight_strict.get('source') or 'disabled'} required={verify_gate_preflight_strict.get('required') if verify_gate_preflight_strict.get('required') is not None else 'n/a'} override={verify_gate_preflight_strict.get('override') or 'none'}`",
    f"- predicted_blocker: `{verify_gate_preflight_blocker or 'none'}`",
    f"- status_evidence: `{verify_status_evidence_failure_reason or 'ok'}`",
    "",
    "## Blockers",
]
if blockers:
    for b in blockers[:7]:
        md_lines.append(f"- [{b.get('severity')}] {b.get('reason')} (owner: {b.get('owner')})")
else:
    md_lines.append("- none")

md_lines += ["", "## Next actions"]
for item in handover.get("recommended_next_actions") or []:
    md_lines.append(f"- `{item.get('command')}`")

md_lines += [
    "",
    "## Risk notes",
    f"- readiness: `{current.get('readiness')}`",
    f"- mutation_gate: `status={mutation_gate_status} posture={mutation_gate_posture}`",
]

try:
    validate_handover_contract(handover)
except Exception as exc:
    raise SystemExit(str(exc))

handover_json_text = json.dumps(handover, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
handover_md_text = "\n".join(md_lines) + "\n"

atomic_write_text(handover_json_path, handover_json_text)
atomic_write_text(handover_md_path, handover_md_text)
atomic_write_text(compat_handover_json_path, handover_json_text)
atomic_write_text(compat_handover_md_path, handover_md_text)

payload = {
    "ok": True,
    "handover_json": str(handover_json_path.relative_to(root)),
    "handover_md": str(handover_md_path.relative_to(root)),
    "stale": st["stale"],
    "diffs": st["diffs"],
    "freshness": st.get("freshness") if isinstance(st, dict) else None,
    "truth_anchor": handover.get("truth_anchor"),
    "coherence_tuple_hash": ((handover.get("coherence") or {}).get("tuple_hash")),
    "generation_pointer": handover.get("generation_pointer") or {},
    "proof_status": handover.get("proof_status") or {},
    "blockers": handover.get("blockers") or [],
    "auto_refreshed": auto_refreshed,
    "stale_before_refresh": stale_before,
}

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(f"HANDOVER LATEST: stale={payload['stale']} diffs={','.join(payload['diffs']) or 'none'}")
PY
