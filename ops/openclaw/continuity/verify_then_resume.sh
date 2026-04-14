#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"

python3 - "$ROOT" "$@" <<'PY'
import argparse
import atexit
import datetime as dt
import fcntl
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
argv = sys.argv[2:]

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from coherence_tuple import compute_policy_freshness
except Exception:  # pragma: no cover - fail-soft if local module unavailable
    compute_policy_freshness = None

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover - fail-soft if local module unavailable
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from strict_required_check_contracts import (
        compute_contract_fingerprint as _shared_compute_contract_fingerprint,
        strict_required_contracts_for_verify_then_resume,
    )
except Exception as exc:
    raise SystemExit(f"strict required-check contracts unavailable: {exc}")


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


parser = argparse.ArgumentParser(description="Verify latest continuity checkpoint before any mutate step.")
parser.add_argument("--checkpoint", help="Checkpoint JSON path or checkpoint id")
parser.add_argument("--execute", action="store_true", help="Run first next_action command after verification passes")
parser.add_argument("--run-rollback", action="store_true", help="Run rollback commands if verification fails")
parser.add_argument(
    "--allow-unsafe-verification",
    action="store_true",
    help="Allow potentially mutating verification commands",
)
parser.add_argument(
    "--skip-baseline-checks",
    action="store_true",
    help="Skip baseline continuity/architecture invariants (not recommended)",
)
parser.add_argument(
    "--status-evidence-repair",
    action="store_true",
    help=(
        "Run a narrow stale-evidence repair verification pass: skip A6 observability and "
        "strict-autonomy regression gates while still enforcing connector health and "
        "checkpoint verification commands"
    ),
)
parser.add_argument("--action-token", "--truth-anchor", dest="action_token", help="Mutation token for --execute path")
parser.add_argument(
    "--allow-legacy-anchor",
    action="store_true",
    help="Allow legacy anchor format when validating --action-token",
)
parser.add_argument("--mutation-ticket", help="Lane authority mutation ticket for enforced ingress modes")
parser.add_argument(
    "--attestation",
    action="append",
    default=[],
    help="Structured attestation payload for enforced ingress modes (repeatable)",
)
parser.add_argument(
    "--attestation-object",
    action="append",
    default=[],
    help="Path to attestation object JSON for enforced ingress modes (repeatable)",
)
strict_autonomy_group = parser.add_mutually_exclusive_group()
strict_autonomy_group.add_argument(
    "--strict-autonomy-regressions",
    action="store_true",
    help=(
        "Force-enable strict autonomy regression cluster for this run "
        "(default mode is already strict-on unless explicitly disabled via "
        "--no-strict-autonomy-regressions or OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS=0)"
    ),
)
strict_autonomy_group.add_argument(
    "--no-strict-autonomy-regressions",
    action="store_true",
    help=(
        "Disable strict autonomy regression gating for this run "
        "unless OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED=1 forces it"
    ),
)
args = parser.parse_args(argv)
status_evidence_repair_mode = bool(args.status_evidence_repair)
if status_evidence_repair_mode:
    args.skip_baseline_checks = True

latest_dir = root / "state" / "continuity" / "latest"
verify_report_path = latest_dir / "verify_last.json"


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


VERIFY_PUBLISH_LOCK_PATH = latest_dir / "locks" / "verify_then_resume.publish.lock"


def _verify_publish_lock_wait_seconds() -> float:
    raw = str(os.environ.get("OPENCLAW_VERIFY_THEN_RESUME_LOCK_WAIT_SEC", "30")).strip()
    try:
        return max(0.0, float(raw))
    except Exception:
        return 30.0


def _verify_publish_lock_hold_test_seconds() -> float:
    raw = str(os.environ.get("OPENCLAW_VERIFY_THEN_RESUME_TEST_HOLD_LOCK_SEC", "0")).strip()
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


_verify_publish_lock_fd: Optional[Any] = None


def _release_verify_publish_lock() -> None:
    global _verify_publish_lock_fd
    fd = _verify_publish_lock_fd
    _verify_publish_lock_fd = None
    if fd is None:
        return
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        fd.close()
    except Exception:
        pass


def acquire_verify_publish_lock() -> tuple[bool, Dict[str, Any]]:
    global _verify_publish_lock_fd

    VERIFY_PUBLISH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    wait_sec = _verify_publish_lock_wait_seconds()
    started = time.monotonic()
    fd = VERIFY_PUBLISH_LOCK_PATH.open("a+")

    while True:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _verify_publish_lock_fd = fd
            atexit.register(_release_verify_publish_lock)
            hold_sec = _verify_publish_lock_hold_test_seconds()
            if hold_sec > 0:
                time.sleep(hold_sec)
            return True, {
                "path": str(VERIFY_PUBLISH_LOCK_PATH),
                "wait_sec": wait_sec,
                "held_for_test_sec": hold_sec,
            }
        except BlockingIOError:
            if (time.monotonic() - started) >= wait_sec:
                try:
                    fd.close()
                except Exception:
                    pass
                return False, {
                    "path": str(VERIFY_PUBLISH_LOCK_PATH),
                    "wait_sec": wait_sec,
                    "status": "timeout",
                }
            time.sleep(0.05)


def run_execute_mutator_ingress_guard(parsed_args: argparse.Namespace) -> Dict[str, Any]:
    guard_script = (root / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh").resolve()
    command: List[str] = [
        "bash",
        str(guard_script),
        "--script",
        "verify_then_resume.sh",
        "--risk-tier",
        "medium",
        "--mutation-operation",
        "verify_then_resume:execute",
    ]

    if parsed_args.action_token:
        command.extend(["--action-token", str(parsed_args.action_token)])
    if parsed_args.allow_legacy_anchor:
        command.append("--allow-legacy-anchor")
    if parsed_args.mutation_ticket:
        command.extend(["--mutation-ticket", str(parsed_args.mutation_ticket)])

    for att in parsed_args.attestation or []:
        if str(att).strip():
            command.extend(["--attestation", str(att)])

    for att_obj in parsed_args.attestation_object or []:
        if str(att_obj).strip():
            command.extend(["--attestation-object", str(att_obj)])

    cp = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(root),
        env={**os.environ, "OPENCLAW_ROOT": str(root)},
    )
    return {
        "ok": cp.returncode == 0,
        "returncode": cp.returncode,
        "stderr_tail": tail_text(cp.stderr or ""),
        "stdout_tail": tail_text(cp.stdout or ""),
        "command": command,
    }


def resolve_checkpoint_path(raw: Optional[str]) -> pathlib.Path:
    if raw:
        candidate = pathlib.Path(raw)
        if not candidate.is_absolute():
            candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
        # treat as checkpoint id under checkpoints/
        cid = raw.strip()
        by_id = root / "state" / "continuity" / "checkpoints" / f"{cid}.json"
        if by_id.exists():
            return by_id
        raise SystemExit(f"checkpoint not found: {raw}")

    surface_path = latest_dir / "handover_latest.json"
    if surface_path.exists():
        try:
            surface_obj = json.loads(surface_path.read_text(encoding="utf-8"))
            checkpoint = surface_obj.get("checkpoint") if isinstance(surface_obj.get("checkpoint"), dict) else {}
            json_rel = str(checkpoint.get("path") or "").strip()
            if json_rel:
                p = (root / json_rel).resolve()
                if p.exists():
                    return p
        except Exception:
            pass
        return surface_path.resolve()

    pointer_path = latest_dir / "latest_pointer.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        json_rel = str(pointer.get("json_path") or "")
        if json_rel:
            p = (root / json_rel).resolve()
            if p.exists():
                return p

    raise SystemExit("no checkpoint found (missing handover_latest.json / latest_pointer.json)")


def is_potentially_mutating(command: str) -> bool:
    c = command.strip().lower()
    if not c:
        return False
    patterns = [
        r"\brm\b",
        r"\bmv\b",
        r"\bcp\b",
        r"\btruncate\b",
        r"\bchmod\b",
        r"\bchown\b",
        r"\bsed\s+-i\b",
        r"\btee\b",
        r"\bopenclaw\s+cron\s+(add|edit|rm|enable|disable)\b",
        r"\bopenclaw\s+gateway\s+(restart|start|stop)\b",
        r"\bsystemctl\b.*\b(start|stop|restart)\b",
        r"\bgit\b.*\b(reset|checkout|revert|clean|merge)\b",
        r"\bkill\b",
        r"\bpkill\b",
    ]
    return any(re.search(p, c) for p in patterns)


def run_shell(command: str) -> Dict[str, Any]:
    cp = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "OPENCLAW_VERIFY_THEN_RESUME_ACTIVE": "1",
            "OPENCLAW_CONTINUITY_SKIP_PUBLISH_LOCK": "1",
        },
    )
    return {
        "command": command,
        "returncode": cp.returncode,
        "ok": cp.returncode == 0,
        "stdout": (cp.stdout or "")[:1200],
        "stderr": (cp.stderr or "")[:1200],
    }


def truthy_raw(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def env_truthy(name: str) -> bool:
    return truthy_raw(str(os.environ.get(name, "")))


def env_is_set(name: str) -> bool:
    return name in os.environ


def resolve_wrapper_strict_autonomy_effective() -> Dict[str, Any]:
    source = str(os.environ.get("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_SOURCE", "")).strip()
    enabled_raw = str(os.environ.get("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_ENABLED", "")).strip()
    required_raw = str(os.environ.get("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_REQUIRED", "")).strip()

    if not source and not enabled_raw and not required_raw:
        return {}

    payload: Dict[str, Any] = {}
    if source:
        payload["source"] = source
    if enabled_raw:
        payload["enabled"] = truthy_raw(enabled_raw)
    if required_raw:
        payload["required"] = truthy_raw(required_raw)
    return payload


def resolve_strict_autonomy_regressions_mode(parsed_args: argparse.Namespace) -> Dict[str, Any]:
    required = env_truthy("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED")

    if parsed_args.no_strict_autonomy_regressions:
        if required:
            return {
                "enabled": True,
                "required": True,
                "source": "verify_gate_required_env",
                "override": "disable",
                "override_denied": True,
            }
        return {
            "enabled": False,
            "required": False,
            "source": "cli_flag_disable",
            "override": "disable",
            "override_denied": False,
        }

    if required:
        return {"enabled": True, "required": True, "source": "verify_gate_required_env"}

    if parsed_args.strict_autonomy_regressions:
        return {"enabled": True, "required": False, "source": "cli_flag"}

    if env_is_set("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS"):
        return {
            "enabled": env_truthy("OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS"),
            "required": False,
            "source": "verify_gate_policy_env",
        }

    if env_is_set("OPENCLAW_STRICT_AUTONOMY_REGRESSIONS"):
        return {
            "enabled": env_truthy("OPENCLAW_STRICT_AUTONOMY_REGRESSIONS"),
            "required": False,
            "source": "legacy_env",
        }

    return {"enabled": True, "required": False, "source": "default_on"}


def tail_text(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    rows = (text or "").splitlines()
    if len(rows) > max_lines:
        rows = rows[-max_lines:]
    trimmed = "\n".join(rows)
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    return trimmed


def run_command(command: List[str], *, cwd: pathlib.Path) -> Dict[str, Any]:
    start = time.monotonic()
    cp = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(cwd),
        env={
            **os.environ,
            "OPENCLAW_ROOT": str(cwd),
            "OPENCLAW_CONTINUITY_SKIP_PUBLISH_LOCK": "1",
        },
    )
    duration = round(time.monotonic() - start, 3)
    stdout = cp.stdout or ""
    stderr = cp.stderr or ""
    result: Dict[str, Any] = {
        "command": command,
        "returncode": cp.returncode,
        "ok": cp.returncode == 0,
        "duration_sec": duration,
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }
    try:
        result["summary"] = json.loads(stdout)
    except Exception:
        if stdout.strip():
            result["summary_parse_error"] = "stdout_not_json"
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compute_contract_fingerprint(contract_inputs: Dict[str, Any]) -> str:
    return _shared_compute_contract_fingerprint(contract_inputs)


REQUIRED_AUTONOMY_CLUSTER_CHECK_CONTRACTS: tuple[Dict[str, Any], ...] = strict_required_contracts_for_verify_then_resume()


def _normalize_command_tokens(command_value: Any) -> List[str]:
    if isinstance(command_value, list):
        return [str(item).strip() for item in command_value if str(item).strip()]
    if isinstance(command_value, str) and command_value.strip():
        try:
            return [str(item).strip() for item in shlex.split(command_value) if str(item).strip()]
        except Exception:
            return [command_value.strip()]
    return []


def _command_matches_suffix(command_value: Any, suffix: str) -> bool:
    if not suffix:
        return True
    normalized_suffix = str(suffix).strip().replace("\\", "/")
    if not normalized_suffix:
        return True
    for token in _normalize_command_tokens(command_value):
        if token.replace("\\", "/").endswith(normalized_suffix):
            return True
    return False


def _extract_summary_json(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    for marker in ("\n{", "{"):
        idx = raw.rfind(marker)
        if idx < 0:
            continue
        candidate = raw[idx + 1 :] if marker == "\n{" else raw[idx:]
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def _extract_required_check_summary(row: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    stdout_json = row.get("stdout_json")
    if isinstance(stdout_json, dict):
        return stdout_json, "stdout_json"

    parsed_from_tail = _extract_summary_json(row.get("stdout_tail"))
    if isinstance(parsed_from_tail, dict):
        return parsed_from_tail, "stdout_tail"

    return None, None


def validate_required_autonomy_cluster_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    required_check_ids = [str(item.get("id") or "") for item in REQUIRED_AUTONOMY_CLUSTER_CHECK_CONTRACTS if str(item.get("id") or "")]
    required_check_id_set = set(required_check_ids)
    payload: Dict[str, Any] = {
        "required_check_ids": required_check_ids,
        "required_check_contracts": [dict(item) for item in REQUIRED_AUTONOMY_CLUSTER_CHECK_CONTRACTS],
        "ok": True,
    }

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else None
    if not isinstance(summary, dict):
        payload["ok"] = False
        payload["error"] = "summary_missing_or_unreadable"
        return payload

    selected_ids_raw = summary.get("selected_ids")
    selected_ids: List[str] = []
    if isinstance(selected_ids_raw, list):
        selected_ids = [str(item) for item in selected_ids_raw]

    rows_raw = summary.get("results")
    rows = rows_raw if isinstance(rows_raw, list) else []
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_required_result_row_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            continue

        if row_id in rows_by_id:
            if row_id in required_check_id_set:
                duplicate_required_result_row_counts[row_id] = duplicate_required_result_row_counts.get(row_id, 1) + 1
            continue

        rows_by_id[row_id] = row

    missing_from_selected: List[str] = []
    missing_from_results: List[str] = []
    failed_required: List[Dict[str, Any]] = []
    command_contract_violations: List[Dict[str, Any]] = []
    scenario_contract_violations: List[Dict[str, Any]] = [
        {
            "id": check_id,
            "error": "duplicate_required_check_result_rows",
            "result_row_count": row_count,
        }
        for check_id, row_count in sorted(duplicate_required_result_row_counts.items())
    ]

    for contract in REQUIRED_AUTONOMY_CLUSTER_CHECK_CONTRACTS:
        check_id = str(contract.get("id") or "").strip()
        if not check_id:
            continue
        if check_id not in selected_ids:
            missing_from_selected.append(check_id)

        row = rows_by_id.get(check_id)
        if not isinstance(row, dict):
            missing_from_results.append(check_id)
            continue

        if row.get("ok") is not True:
            failed_required.append(
                {
                    "id": check_id,
                    "returncode": row.get("returncode"),
                }
            )

        expected_command_suffix = str(contract.get("command_suffix") or "").strip()
        if expected_command_suffix and not _command_matches_suffix(row.get("command"), expected_command_suffix):
            command_contract_violations.append(
                {
                    "id": check_id,
                    "expected_command_suffix": expected_command_suffix,
                    "actual_command": row.get("command"),
                }
            )

        expected_harness = str(contract.get("expected_harness") or "").strip()
        expected_summary_source = str(contract.get("expected_summary_source") or "").strip()
        expected_summary_schema_version = str(contract.get("expected_summary_schema_version") or "").strip()
        expected_summary_fields_raw = contract.get("expected_summary_fields")
        expected_summary_fields = expected_summary_fields_raw if isinstance(expected_summary_fields_raw, dict) else {}
        expected_provenance_schema_version = str(contract.get("expected_provenance_schema_version") or "").strip()
        expected_contract_fingerprint_inputs = contract.get("expected_contract_fingerprint_inputs")
        require_provenance_contract_inputs = bool(contract.get("require_provenance_contract_inputs"))
        minimum_result_count = contract.get("minimum_result_count")
        required_scenario_names = [
            str(item).strip()
            for item in (contract.get("required_scenario_names") or [])
            if str(item).strip()
        ]
        if (
            not expected_harness
            and not expected_summary_source
            and not expected_summary_schema_version
            and not expected_summary_fields
            and not expected_provenance_schema_version
            and not isinstance(expected_contract_fingerprint_inputs, dict)
            and not require_provenance_contract_inputs
            and minimum_result_count is None
            and not required_scenario_names
        ):
            continue

        check_summary, check_summary_source = _extract_required_check_summary(row)
        if not isinstance(check_summary, dict):
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "error": "required_harness_summary_missing",
                }
            )
            continue

        if expected_harness and str(check_summary.get("harness") or "").strip() != expected_harness:
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "summary_source": check_summary_source,
                    "expected_harness": expected_harness,
                    "actual_harness": check_summary.get("harness"),
                }
            )

        if expected_summary_source and str(check_summary.get("source") or "").strip() != expected_summary_source:
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "summary_source": check_summary_source,
                    "error": "summary_source_mismatch",
                    "expected_summary_source": expected_summary_source,
                    "actual_summary_source": check_summary.get("source"),
                }
            )

        if expected_summary_schema_version and str(check_summary.get("summary_schema_version") or "").strip() != expected_summary_schema_version:
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "summary_source": check_summary_source,
                    "error": "summary_schema_version_mismatch",
                    "expected_summary_schema_version": expected_summary_schema_version,
                    "actual_summary_schema_version": check_summary.get("summary_schema_version"),
                }
            )

        for summary_field_name, expected_summary_field_value in expected_summary_fields.items():
            actual_summary_field_value = check_summary.get(summary_field_name)
            if _canonical_json(actual_summary_field_value) != _canonical_json(expected_summary_field_value):
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "error": "summary_field_mismatch",
                        "field": summary_field_name,
                        "expected_value": expected_summary_field_value,
                        "actual_value": actual_summary_field_value,
                    }
                )

        provenance = check_summary.get("required_check_provenance")
        if not isinstance(provenance, dict):
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "summary_source": check_summary_source,
                    "error": "required_check_provenance_missing",
                }
            )
        else:
            if expected_provenance_schema_version and str(provenance.get("schema_version") or "").strip() != expected_provenance_schema_version:
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "error": "required_check_provenance_schema_mismatch",
                        "expected_provenance_schema_version": expected_provenance_schema_version,
                        "actual_provenance_schema_version": provenance.get("schema_version"),
                    }
                )

            provenance_check_id = str(provenance.get("check_id") or "").strip()
            if provenance_check_id != check_id:
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "error": "required_check_provenance_check_id_mismatch",
                        "expected_provenance_check_id": check_id,
                        "actual_provenance_check_id": provenance.get("check_id"),
                    }
                )

            actual_contract_fingerprint = str(provenance.get("contract_fingerprint") or "").strip()
            provenance_contract_inputs = provenance.get("contract_inputs")
            provenance_contract_inputs_dict = provenance_contract_inputs if isinstance(provenance_contract_inputs, dict) else None

            if require_provenance_contract_inputs and not isinstance(provenance_contract_inputs_dict, dict):
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "error": "required_check_contract_inputs_missing",
                    }
                )

            if isinstance(expected_contract_fingerprint_inputs, dict):
                expected_contract_fingerprint = _compute_contract_fingerprint(expected_contract_fingerprint_inputs)
                if not actual_contract_fingerprint:
                    scenario_contract_violations.append(
                        {
                            "id": check_id,
                            "summary_source": check_summary_source,
                            "error": "required_check_contract_fingerprint_missing",
                            "expected_contract_fingerprint": expected_contract_fingerprint,
                        }
                    )
                elif actual_contract_fingerprint != expected_contract_fingerprint:
                    scenario_contract_violations.append(
                        {
                            "id": check_id,
                            "summary_source": check_summary_source,
                            "error": "required_check_contract_fingerprint_mismatch",
                            "expected_contract_fingerprint": expected_contract_fingerprint,
                            "actual_contract_fingerprint": provenance.get("contract_fingerprint"),
                        }
                    )

                if isinstance(provenance_contract_inputs_dict, dict):
                    if _canonical_json(provenance_contract_inputs_dict) != _canonical_json(expected_contract_fingerprint_inputs):
                        scenario_contract_violations.append(
                            {
                                "id": check_id,
                                "summary_source": check_summary_source,
                                "error": "required_check_contract_inputs_mismatch",
                                "expected_contract_inputs": expected_contract_fingerprint_inputs,
                                "actual_contract_inputs": provenance_contract_inputs_dict,
                            }
                        )

                    observed_contract_fingerprint = _compute_contract_fingerprint(provenance_contract_inputs_dict)
                    if actual_contract_fingerprint and actual_contract_fingerprint != observed_contract_fingerprint:
                        scenario_contract_violations.append(
                            {
                                "id": check_id,
                                "summary_source": check_summary_source,
                                "error": "required_check_contract_inputs_fingerprint_mismatch",
                                "actual_contract_fingerprint": provenance.get("contract_fingerprint"),
                                "observed_contract_fingerprint": observed_contract_fingerprint,
                            }
                        )

        scenario_rows = check_summary.get("results") if isinstance(check_summary.get("results"), list) else []
        scenario_names = [
            str(item.get("name") or "").strip()
            for item in scenario_rows
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]

        if isinstance(minimum_result_count, int):
            total_raw = check_summary.get("total")
            if isinstance(total_raw, int):
                observed_result_count = total_raw
            elif scenario_rows:
                observed_result_count = len(scenario_rows)
            else:
                observed_result_count = None

            if not isinstance(observed_result_count, int):
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "error": "scenario_count_missing",
                        "minimum_result_count": minimum_result_count,
                    }
                )
            elif observed_result_count < minimum_result_count:
                scenario_contract_violations.append(
                    {
                        "id": check_id,
                        "summary_source": check_summary_source,
                        "minimum_result_count": minimum_result_count,
                        "actual_result_count": observed_result_count,
                    }
                )

        missing_scenarios = [name for name in required_scenario_names if name not in scenario_names]
        if missing_scenarios:
            scenario_contract_violations.append(
                {
                    "id": check_id,
                    "summary_source": check_summary_source,
                    "missing_required_scenarios": missing_scenarios,
                }
            )

    if missing_from_selected:
        payload["missing_from_selected_ids"] = missing_from_selected
    if missing_from_results:
        payload["missing_from_results"] = missing_from_results
    if failed_required:
        payload["failed_required_checks"] = failed_required
    if command_contract_violations:
        payload["command_contract_violations"] = command_contract_violations
    if scenario_contract_violations:
        payload["scenario_contract_violations"] = scenario_contract_violations

    payload["ok"] = not (
        missing_from_selected
        or missing_from_results
        or failed_required
        or command_contract_violations
        or scenario_contract_violations
    )
    return payload


def _probe_connector_blockers(root_path: pathlib.Path, continuity_script: pathlib.Path) -> Dict[str, Any]:
    cp = subprocess.run(
        ["bash", str(continuity_script), "--strict", "--json"],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(root_path),
        env={**os.environ, "OPENCLAW_ROOT": str(root_path)},
    )
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        if cp.returncode != 0:
            return {
                "ok": False,
                "reason": "continuity_now_failed",
                "stderr": (cp.stderr or "")[:600],
                "returncode": cp.returncode,
            }
        return {"ok": False, "reason": "continuity_now_invalid_json"}

    blocking_raw = data.get("coherence", {}).get("connector_blocking_reasons", [])
    if isinstance(blocking_raw, list):
        blocking_reasons = [str(item).strip() for item in blocking_raw if str(item).strip()]
    elif isinstance(blocking_raw, str) and blocking_raw.strip():
        blocking_reasons = [blocking_raw.strip()]
    else:
        blocking_reasons = []

    blocker_reasons_raw = data.get("blocker_reasons", [])
    if isinstance(blocker_reasons_raw, list):
        non_connector_blockers = [str(item).strip() for item in blocker_reasons_raw if str(item).strip()]
    elif isinstance(blocker_reasons_raw, str) and blocker_reasons_raw.strip():
        non_connector_blockers = [blocker_reasons_raw.strip()]
    else:
        non_connector_blockers = []

    if cp.returncode != 0 and not blocking_reasons:
        if non_connector_blockers:
            return {
                "ok": True,
                "reason": "connectors_ok_non_connector_blockers_present",
                "blockers": [],
                "generated_at": data.get("generated_at"),
                "returncode": cp.returncode,
                "continuity_now_blocker_reasons": list(non_connector_blockers),
            }
        return {
            "ok": False,
            "reason": "continuity_now_failed",
            "stderr": (cp.stderr or "")[:600],
            "returncode": cp.returncode,
            "generated_at": data.get("generated_at"),
        }

    return {
        "ok": not bool(blocking_reasons),
        "reason": "connectors_ok" if not blocking_reasons else "connector_blocker_present",
        "blockers": list(blocking_reasons or []),
        "generated_at": data.get("generated_at"),
        "returncode": cp.returncode,
    }


def _only_validation_gate_self_blockers(blockers: Any) -> bool:
    rows = [str(item).strip() for item in (blockers or []) if str(item).strip()]
    if not rows:
        return False
    return all("validation.gates::core" in row for row in rows)


def check_connector_health(root_path: pathlib.Path) -> Dict[str, Any]:
    """Runs continuity_now.sh and checks for blocking connectors."""
    continuity_script = root_path / "ops" / "openclaw" / "continuity" / "continuity_now.sh"
    if not continuity_script.exists():
        return {"ok": False, "reason": "continuity_now_script_missing"}

    first_probe = _probe_connector_blockers(root_path, continuity_script)
    if first_probe.get("ok"):
        return first_probe
    if first_probe.get("reason") != "connector_blocker_present":
        return first_probe

    # One bounded freshness retry: sync GTC connector evidence and probe again.
    # This avoids latching transient pre-sync connector expiry residue while preserving fail-closed behavior.
    gtc_sync_script = root_path / "ops" / "openclaw" / "continuity" / "gtc_v2_sync.sh"
    if not gtc_sync_script.exists():
        return first_probe

    gtc_sync = subprocess.run(
        ["bash", str(gtc_sync_script)],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(root_path),
        env={
            **os.environ,
            "OPENCLAW_ROOT": str(root_path),
            "OPENCLAW_INTERNAL_MUTATION": "1",
            "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "verify_then_resume.sh:check_connector_health:gtc_v2_sync_retry",
        },
    )

    second_probe = _probe_connector_blockers(root_path, continuity_script)
    if second_probe.get("ok"):
        second_probe["reason"] = "connectors_ok_after_gtc_sync_retry"
        second_probe["retry"] = {
            "attempted": True,
            "gtc_v2_sync_returncode": gtc_sync.returncode,
            "initial_blockers": first_probe.get("blockers") or [],
        }
        return second_probe

    if second_probe.get("reason") == "connector_blocker_present":
        second_blockers = second_probe.get("blockers") or []
        if _only_validation_gate_self_blockers(second_blockers):
            return {
                "ok": True,
                "reason": "connectors_ok_validation_gate_self_refresh_pending",
                "generated_at": second_probe.get("generated_at"),
                "retry": {
                    "attempted": True,
                    "gtc_v2_sync_returncode": gtc_sync.returncode,
                    "initial_blockers": first_probe.get("blockers") or [],
                    "suppressed_blockers": second_blockers,
                },
            }

        second_probe["retry"] = {
            "attempted": True,
            "gtc_v2_sync_returncode": gtc_sync.returncode,
            "initial_blockers": first_probe.get("blockers") or [],
        }
    return second_probe


checkpoint_path = resolve_checkpoint_path(args.checkpoint)
checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

checkpoint_id = (((checkpoint.get("metadata") or {}).get("checkpoint_id")) or checkpoint_path.stem)
verification_commands: List[str] = list(((checkpoint.get("execution_plan") or {}).get("verification_commands") or []))
rollback_commands: List[str] = list(((checkpoint.get("execution_plan") or {}).get("rollback_commands") or []))
next_actions = list(((checkpoint.get("execution_plan") or {}).get("next_actions") or []))
strict_autonomy_mode = resolve_strict_autonomy_regressions_mode(args)
strict_autonomy_regressions = bool(strict_autonomy_mode.get("enabled"))
strict_autonomy_required = bool(strict_autonomy_mode.get("required"))
strict_autonomy_override = str(strict_autonomy_mode.get("override") or "").strip() or None
strict_autonomy_override_denied = bool(strict_autonomy_mode.get("override_denied"))
wrapper_strict_autonomy_effective = resolve_wrapper_strict_autonomy_effective()
strict_autonomy_effective_source = strict_autonomy_mode.get("source")
if strict_autonomy_mode.get("source") == "cli_flag" and wrapper_strict_autonomy_effective.get("source"):
    strict_autonomy_effective_source = wrapper_strict_autonomy_effective.get("source")
strict_autonomy_regressions_effective = bool(
    strict_autonomy_regressions and not status_evidence_repair_mode
)

mutator_ingress_result: Optional[Dict[str, Any]] = None
if args.execute:
    mutator_ingress_result = run_execute_mutator_ingress_guard(args)
    if mutator_ingress_result.get("ok") is not True:
        report = {
            "schema_version": "continuity.verify_report.v1",
            "timestamp": now_iso(),
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_path),
            "status": "BLOCKER",
            "reason": "mutator_ingress_denied",
            "execute_requested": True,
            "mutator_ingress": mutator_ingress_result,
            "strict_autonomy_regressions": {
                "enabled": strict_autonomy_regressions,
                "required": strict_autonomy_required,
                "source": strict_autonomy_mode.get("source"),
                "effective_source": strict_autonomy_effective_source,
                "wrapper_effective": wrapper_strict_autonomy_effective or None,
                "skipped": True,
                "reason": "mutator_ingress_denied",
            },
        }
        atomic_write(verify_report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print("BLOCKER: verify_then_resume execute path denied by mutator ingress guard")
        raise SystemExit(1)

publish_lock_ok, publish_lock_payload = acquire_verify_publish_lock()
if not publish_lock_ok:
    report = {
        "schema_version": "continuity.verify_report.v1",
        "timestamp": now_iso(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "status": "BLOCKER",
        "reason": "verify_publish_lock_timeout",
        "execute_requested": args.execute,
        "mutator_ingress": mutator_ingress_result,
        "publish_lock": publish_lock_payload,
        "strict_autonomy_regressions": {
            "enabled": strict_autonomy_regressions,
            "required": strict_autonomy_required,
            "source": strict_autonomy_mode.get("source"),
            "effective_source": strict_autonomy_effective_source,
            "wrapper_effective": wrapper_strict_autonomy_effective or None,
            "skipped": True,
            "reason": "verify_publish_lock_timeout",
        },
    }
    atomic_write(verify_report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        "BLOCKER: verify_then_resume publish lock timeout "
        f"path={publish_lock_payload.get('path')} wait_sec={publish_lock_payload.get('wait_sec')}"
    )
    raise SystemExit(1)

baseline_commands: List[str] = []
if not args.skip_baseline_checks:
    db_check_script = (root / "ops" / "openclaw" / "continuity" / "db_integrity_check.sh").resolve()
    contract_check_script = (root / "ops" / "openclaw" / "architecture" / "validate_contracts.sh").resolve()
    swarm_check_script = (root / "ops" / "openclaw" / "architecture" / "check_swarm_operability.sh").resolve()
    swarm_runtime_check_script = (root / "ops" / "openclaw" / "continuity" / "swarm_runtime_check.sh").resolve()
    slot_fill_check_script = (root / "ops" / "openclaw" / "continuity" / "check_slot_fill_protocol.sh").resolve()
    gtc_schema_check_script = (root / "ops" / "openclaw" / "continuity" / "gtc_latest_schema_check.sh").resolve()
    publish_transaction_check_script = (root / "ops" / "openclaw" / "continuity" / "check_gtc_publish_transaction_regressions.py").resolve()
    queue_cooldown_check_script = (root / "ops" / "openclaw" / "continuity" / "check_queue_cooldown_authority_regressions.py").resolve()
    baseline_commands = [
        f"bash {db_check_script} --strict --json",
        f"bash {contract_check_script} --json",
        f"bash {swarm_check_script} --json",
        f"bash {swarm_runtime_check_script} --strict --json",
        f"bash {slot_fill_check_script} --json",
        f"bash {gtc_schema_check_script} --strict --json",
        f"{sys.executable} {publish_transaction_check_script}",
        f"{sys.executable} {queue_cooldown_check_script}",
    ]

if not verification_commands:
    raise SystemExit("checkpoint has no verification_commands")

unsafe = [cmd for cmd in verification_commands if is_potentially_mutating(cmd)]
if unsafe and not args.allow_unsafe_verification:
    report = {
        "schema_version": "continuity.verify_report.v1",
        "timestamp": now_iso(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "status": "BLOCKER",
        "reason": "unsafe_verification_command",
        "unsafe_commands": unsafe,
        "executed": False,
        "execute_requested": args.execute,
        "mutator_ingress": mutator_ingress_result,
        "publish_lock": publish_lock_payload,
        "strict_autonomy_regressions": {
            "enabled": strict_autonomy_regressions,
            "required": strict_autonomy_required,
            "source": strict_autonomy_mode.get("source"),
            "effective_source": strict_autonomy_effective_source,
            "wrapper_effective": wrapper_strict_autonomy_effective or None,
            "skipped": True,
            "reason": "unsafe_verification_command",
        },
    }
    atomic_write(verify_report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("BLOCKER: unsafe verification command detected; rerun with --allow-unsafe-verification if intentional")
    raise SystemExit(1)

if strict_autonomy_override_denied:
    strict_payload: Dict[str, Any] = {
        "enabled": strict_autonomy_regressions,
        "required": strict_autonomy_required,
        "source": strict_autonomy_mode.get("source"),
        "effective_source": strict_autonomy_effective_source,
        "skipped": True,
        "reason": "strict_autonomy_required_override_denied",
        "override_denied": True,
    }
    if strict_autonomy_override:
        strict_payload["override"] = strict_autonomy_override
    if wrapper_strict_autonomy_effective:
        strict_payload["wrapper_effective"] = wrapper_strict_autonomy_effective

    report = {
        "schema_version": "continuity.verify_report.v1",
        "timestamp": now_iso(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "status": "BLOCKER",
        "reason": "strict_autonomy_required_override_denied",
        "executed": False,
        "execute_requested": args.execute,
        "mutator_ingress": mutator_ingress_result,
        "publish_lock": publish_lock_payload,
        "strict_autonomy_regressions": strict_payload,
    }
    atomic_write(verify_report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("BLOCKER: strict autonomy regressions required; explicit disable override denied")
    raise SystemExit(1)

# A6 Ops Reliability Gates
slo_eval_path = (root / "ops" / "openclaw" / "continuity" / "slo_evaluator_snapshot.sh").resolve()
health_snap_path = (root / "ops" / "openclaw" / "continuity" / "layered_health_snapshot.sh").resolve()
multi_host_jitter_harness_path = (root / "ops" / "openclaw" / "continuity" / "a6_multi_host_jitter_harness.py").resolve()
observability_commands = []
if not status_evidence_repair_mode:
    if slo_eval_path.exists():
        observability_commands.append(f"bash {str(slo_eval_path)}")
    if health_snap_path.exists():
        observability_commands.append(f"bash {str(health_snap_path)}")
    if multi_host_jitter_harness_path.exists():
        observability_commands.append(f"{sys.executable} {str(multi_host_jitter_harness_path)} --json")
print("Starting verification.")

observability_results = [run_shell(cmd) for cmd in observability_commands]
observability_failed = [row for row in observability_results if not row.get("ok")]

print("Observability checks complete.")

baseline_results = [run_shell(cmd) for cmd in baseline_commands]
baseline_failed = [row for row in baseline_results if not row.get("ok")]

print("Baseline checks complete.")

autonomy_regressions_payload: Dict[str, Any] = {
    "enabled": strict_autonomy_regressions_effective,
    "policy_enabled": strict_autonomy_regressions,
    "required": strict_autonomy_required,
    "source": strict_autonomy_mode.get("source"),
    "effective_source": strict_autonomy_effective_source,
}
if strict_autonomy_override:
    autonomy_regressions_payload["override"] = strict_autonomy_override
if wrapper_strict_autonomy_effective:
    autonomy_regressions_payload["wrapper_effective"] = wrapper_strict_autonomy_effective
    hinted_enabled = wrapper_strict_autonomy_effective.get("enabled")
    if isinstance(hinted_enabled, bool) and hinted_enabled != strict_autonomy_regressions_effective:
        autonomy_regressions_payload["wrapper_hint_mismatch"] = True
    hinted_required = wrapper_strict_autonomy_effective.get("required")
    if isinstance(hinted_required, bool) and hinted_required != strict_autonomy_required:
        autonomy_regressions_payload["wrapper_required_hint_mismatch"] = True
autonomy_regressions_failed = False
if status_evidence_repair_mode:
    autonomy_regressions_payload["skipped"] = True
    autonomy_regressions_payload["reason"] = "status_evidence_repair_mode"
elif strict_autonomy_regressions_effective:
    print("Running autonomy regressions.")
    harness_path = root / "ops" / "openclaw" / "continuity" / "check_autonomy_continuity_regressions.py"
    command = [sys.executable, str(harness_path), "--json"]
    autonomy_regressions_payload["command"] = command
    if not harness_path.exists():
        autonomy_regressions_payload["result"] = {
            "command": command,
            "ok": None,
            "skipped": True,
            "reason": "harness_missing",
        }
    else:
        result = run_command(command, cwd=root)
        autonomy_regressions_payload["result"] = result
        required_checks = validate_required_autonomy_cluster_checks(result)
        autonomy_regressions_payload["required_checks"] = required_checks
        autonomy_regressions_failed = (not result.get("ok")) or (required_checks.get("ok") is not True)

print("Autonomy regressions complete.")

verification_results = [] if (observability_failed or baseline_failed or autonomy_regressions_failed) else [run_shell(cmd) for cmd in verification_commands]
verify_failed = [row for row in verification_results if not row.get("ok")]

print("Verification commands complete.")

rollback_results: List[Dict[str, Any]] = []
next_action_result: Optional[Dict[str, Any]] = None

status = "READY"
reason = "verification_passed"

connector_health_check = check_connector_health(root)
if not connector_health_check.get("ok"):
    status = "BLOCKER"
    reason = connector_health_check.get("reason", "connector_check_failed")
elif autonomy_regressions_failed:
    status = "BLOCKER"
    reason = "autonomy_regressions_failed"
elif baseline_failed:
    status = "BLOCKER"
    reason = "baseline_check_failed"
elif observability_failed:
    status = "BLOCKER"
    reason = "a6_observability_failed"
elif verify_failed:
    status = "BLOCKER"
    reason = "verification_failed"
    if args.run_rollback and rollback_commands:
        rollback_results = [run_shell(cmd) for cmd in rollback_commands]
else:
    if args.execute and next_actions:
        first = next_actions[0]
        command = first.get("command") if isinstance(first, dict) else str(first)
        next_action_result = run_shell(command)
        if not next_action_result.get("ok"):
            status = "BLOCKER"
            reason = "next_action_failed"

freshness_policy: Dict[str, Any] = {}
if compute_policy_freshness is not None:
    try:
        pf = compute_policy_freshness(root, update_epoch=False)
        freshness_policy = {
            "policy_epoch": pf.get("policy_epoch"),
            "policy_digest": pf.get("policy_digest"),
            "evaluator_hash": pf.get("evaluator_hash"),
            "signature": pf.get("signature"),
            "missing_paths": sorted(set((pf.get("policy_missing_paths") or []) + (pf.get("evaluator_missing_paths") or []))),
            "computed_at": pf.get("computed_at"),
        }
    except Exception as exc:
        freshness_policy = {"error": f"policy_freshness_compute_failed:{exc}"}

report = {
    "schema_version": "continuity.verify_report.v1",
    "timestamp": now_iso(),
    "checkpoint_id": checkpoint_id,
    "checkpoint_path": str(checkpoint_path),
    "status": status,
    "reason": reason,
    "a6_observability_results": observability_results,
    "baseline_checks_enabled": not args.skip_baseline_checks,
    "baseline_results": baseline_results,
    "verification_results": verification_results,
    "strict_autonomy_regressions": autonomy_regressions_payload,
    "status_evidence_repair": {
        "enabled": status_evidence_repair_mode,
        "skip_baseline_checks": bool(args.skip_baseline_checks),
        "skip_observability_checks": bool(status_evidence_repair_mode),
        "skip_strict_autonomy_regressions": bool(status_evidence_repair_mode),
    },
    "connector_health_check": connector_health_check,
    "rollback_requested": args.run_rollback,
    "rollback_results": rollback_results,
    "execute_requested": args.execute,
    "mutator_ingress": mutator_ingress_result,
    "publish_lock": publish_lock_payload,
    "next_action_result": next_action_result,
    "freshness": {
        "policy": freshness_policy,
    },
}
atomic_write(verify_report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

if status == "BLOCKER":
    print(f"BLOCKER: verify_then_resume status=BLOCKER checkpoint_id={checkpoint_id} reason={reason}")
    raise SystemExit(1)

if args.execute:
    print(f"READY: verification passed and first next_action executed for checkpoint_id={checkpoint_id}")
else:
    print(f"READY: verification passed (dry-run; no mutation executed) checkpoint_id={checkpoint_id}")
PY
