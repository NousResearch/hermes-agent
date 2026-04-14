#!/usr/bin/env python3
"""Bounded A3 failover runtime-evidence summary wrapper.

Purpose:
- run the guarded A3 failover runtime-evidence command,
- normalize its evidence packet into a compact check summary,
- preserve explicit phase/artifact verdicts for later cluster/required-check promotion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strict_required_check_contracts import (
    REQUIRED_CHECK_PROVENANCE_SCHEMA_VERSION,
    compute_contract_fingerprint,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_COMMAND = "ops/openclaw/continuity/failover_stress_runtime_evidence.py"
DEFAULT_STRESS_TIMEOUT_SEC = 45
DEFAULT_REFRESH_TIMEOUT_SEC = 45
DEFAULT_RUNTIME_TIMEOUT_GUARD_BAND_SEC = 15
MAX_AUTO_RUNTIME_TIMEOUT_SEC = 180
CHECK_ID = "a3_failover_runtime_evidence"
HARNESS_ID = "a3_failover_runtime_evidence"
SUMMARY_SOURCE = "check_a3_failover_runtime_evidence.py"
SUMMARY_SCHEMA_VERSION = "openclaw.continuity.a3_failover_runtime_evidence_summary.v1"
LEGACY_SCHEMA_VERSION = "a3.failover.runtime_evidence.check.v1"
DEFAULT_EXECUTION_PROFILE = "strict_live"
SHADOW_EXECUTION_PROFILE = "shadow_baseline"
STRICT_CLUSTER_SAFE_EXECUTION_PROFILE = "strict_cluster_safe"
STRICT_CLUSTER_SAFE_FIXTURE_TOP_BLOCKER = "BLK_PROOF_VERIFY_GATE_NOT_PASS"
SHADOW_TOLERATED_PUBLISH_REASONS = {
    "active_top_blocker_not_covered_by_stress_linkage",
}
REQUIRED_SCENARIO_NAMES = (
    "runtime_command_json_contract",
    "repeated_stress_evidence_phase",
    "stress_soak_phase",
    "blocked_reset_successor_proof_phase",
    "successor_takeover_stability_phase",
    "refresh_publish_chain_phase",
    "publish_assertion_verification_phase",
    "runtime_evidence_artifacts_written",
    "runtime_decision_log_evidence",
)
MINIMUM_RESULT_COUNT = len(REQUIRED_SCENARIO_NAMES)

REQUIRED_BLOCKED_RESET_PROOF_ROWS = (
    {
        "failure_class": "generation_mismatch",
        "top_blocker": "BLK_PROOF_GENERATION_MISMATCH",
        "required_profile": "successor_generation_race_recovery_under_load",
        "required_step_label": "successor_validation_generation_mismatch",
    },
    {
        "failure_class": "read_pointer_mismatch",
        "top_blocker": "BLK_PROOF_READ_POINTER_MISMATCH",
        "required_profile": "successor_pointer_drift_recovery_under_load",
        "required_step_label": "successor_validation_pointer_drift_mismatch",
    },
    {
        "failure_class": "proof_invalidated",
        "top_blocker": "BLK_PROOF_INVALIDATED",
        "required_profile": "blocker_clear_recovery_under_load",
        "required_step_label": "successor_validation_fail",
    },
    {
        "failure_class": "verify_gate_not_pass",
        "top_blocker": "BLK_PROOF_VERIFY_GATE_NOT_PASS",
        "required_profile": "blocker_clear_recovery_under_load",
        "required_step_label": "successor_validation_verify_gate_not_pass",
    },
)

REQUIRED_SUCCESSOR_TAKEOVER_STABILITY_PROFILES = (
    "steady_recovery_under_load",
    "blocker_clear_recovery_under_load",
    "successor_generation_race_recovery_under_load",
    "successor_pointer_drift_recovery_under_load",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run bounded A3 failover runtime-evidence summary check")
    ap.add_argument("--repo-root", default=str(ROOT), help="Repository root for the target runtime command")
    ap.add_argument("--python-bin", default=sys.executable, help="Python executable for the runtime command")
    ap.add_argument(
        "--runtime-command",
        default=DEFAULT_RUNTIME_COMMAND,
        help="Repo-relative path to failover_stress_runtime_evidence.py",
    )
    ap.add_argument("--cycles", type=int, default=2, help="Cycles per failover stress profile (must be >=2 for repeated-stress evidence)")
    ap.add_argument("--stress-timeout-sec", type=int, default=DEFAULT_STRESS_TIMEOUT_SEC, help="Stress phase timeout")
    ap.add_argument("--refresh-timeout-sec", type=int, default=DEFAULT_REFRESH_TIMEOUT_SEC, help="Refresh phase timeout")
    ap.add_argument(
        "--runtime-timeout-sec",
        type=int,
        default=None,
        help="Wrapper runtime budget for the runtime evidence command (default: stress+refresh budget + guard band)",
    )
    ap.add_argument("--skip-refresh", action="store_true", help="Skip refresh/publish chain phase")
    ap.add_argument(
        "--require-live-assertions",
        action="store_true",
        help="Fail closed when no active top blocker is available for assertion verification",
    )
    ap.add_argument(
        "--execution-profile",
        default=DEFAULT_EXECUTION_PROFILE,
        choices=[DEFAULT_EXECUTION_PROFILE, SHADOW_EXECUTION_PROFILE, STRICT_CLUSTER_SAFE_EXECUTION_PROFILE],
        help=(
            "Execution profile. strict_live preserves fail-close runtime semantics; "
            "shadow_baseline auto-enables --skip-refresh and tolerates only known live-surface "
            "blocker-mapping drift for bounded shadow evidence collection; "
            "strict_cluster_safe auto-enables --skip-refresh, requires live assertions, and seeds "
            "ephemeral deterministic baseline surfaces for strict-safe cluster runs"
        ),
    )
    return ap.parse_args()


def _tail(text: str, *, max_chars: int = 400) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3] + "..."


def _parse_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise RuntimeError("runtime_command_empty_stdout")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runtime_command_stdout_not_json:{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"runtime_command_payload_not_object:{type(payload).__name__}")
    return payload


def _parse_iso8601(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _strict_cluster_safe_surface_payloads() -> dict[str, dict[str, Any]]:
    top = STRICT_CLUSTER_SAFE_FIXTURE_TOP_BLOCKER
    return {
        "state/handover/latest.json": {
            "proof_status": {
                "top_blocker": top,
            },
            "safe_signals": {
                "proof_top_blocker": top,
                "safe_to_resume": False,
                "safe_to_reset": False,
            },
        },
        "state/continuity/latest/reset_ready_refresh_latest.json": {
            "handover_proof_status": {
                "top_blocker": top,
            },
            "handover_safe_signals": {
                "safe_to_resume": False,
                "safe_to_reset": False,
            },
        },
        "state/continuity/current.json": {
            "reset_ready_refresh": {
                "path": "state/continuity/latest/reset_ready_refresh_latest.json",
            }
        },
        "state/continuity/latest/successor_safe_handover_proof_status.json": {
            "top_blocker": top,
        },
    }


def _seed_strict_cluster_safe_surfaces(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads = _strict_cluster_safe_surface_payloads()
    restore_entries: list[dict[str, Any]] = []

    for rel, payload in payloads.items():
        path = (repo_root / rel).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(f"strict_cluster_safe_seed_path_outside_repo:{rel}") from exc

        existed = path.exists()
        previous_bytes = path.read_bytes() if existed else None
        restore_entries.append(
            {
                "path": path,
                "existed": existed,
                "previous_bytes": previous_bytes,
            }
        )
        _write_json(path, payload)

    return {
        "applied": True,
        "seeded_top_blocker": STRICT_CLUSTER_SAFE_FIXTURE_TOP_BLOCKER,
        "seeded_paths": sorted(payloads.keys()),
    }, restore_entries


def _restore_seeded_surfaces(restore_entries: list[dict[str, Any]]) -> None:
    for row in reversed(restore_entries):
        path = row.get("path")
        if not isinstance(path, Path):
            continue
        existed = bool(row.get("existed"))
        previous_bytes = row.get("previous_bytes")

        if existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(previous_bytes, bytes):
                path.write_bytes(previous_bytes)
            else:
                path.write_text("", encoding="utf-8")
        else:
            if path.exists():
                path.unlink()


def _result(*, name: str, ok: bool, expectation: str, details: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "ok": bool(ok),
        "expectation": expectation,
    }
    if isinstance(details, dict) and details:
        row["details"] = details
    if error:
        row["error"] = error
    return row


def _required_check_contract_inputs() -> dict[str, Any]:
    return {
        "check_id": CHECK_ID,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "minimum_result_count": MINIMUM_RESULT_COUNT,
        "scenario_names": list(REQUIRED_SCENARIO_NAMES),
    }


def _required_check_provenance() -> dict[str, Any]:
    contract_inputs = _required_check_contract_inputs()
    return {
        "schema_version": REQUIRED_CHECK_PROVENANCE_SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "contract_fingerprint": compute_contract_fingerprint(contract_inputs),
        "contract_inputs": contract_inputs,
    }


def _runtime_budget_components(
    args: argparse.Namespace,
    *,
    effective_skip_refresh: bool,
) -> dict[str, Any]:
    stress_timeout_sec = int(args.stress_timeout_sec)
    refresh_timeout_sec = 0 if bool(effective_skip_refresh) else int(args.refresh_timeout_sec)
    guard_band_sec = DEFAULT_RUNTIME_TIMEOUT_GUARD_BAND_SEC
    base_budget_sec = stress_timeout_sec + refresh_timeout_sec
    derived_uncapped_timeout_sec = max(base_budget_sec + guard_band_sec, guard_band_sec)
    return {
        "stress_timeout_sec": stress_timeout_sec,
        "refresh_timeout_sec": refresh_timeout_sec,
        "skip_refresh": bool(effective_skip_refresh),
        "guard_band_sec": guard_band_sec,
        "base_budget_sec": base_budget_sec,
        "derived_uncapped_timeout_sec": derived_uncapped_timeout_sec,
        "max_auto_runtime_timeout_sec": MAX_AUTO_RUNTIME_TIMEOUT_SEC,
    }


def _summary(
    *,
    ok: bool,
    results: list[dict[str, Any]],
    command: list[str],
    payload: dict[str, Any] | None = None,
    error: str | None = None,
    runtime_timeout_sec: int | None = None,
    runtime_timeout_mode: str | None = None,
    runtime_budget_components: dict[str, Any] | None = None,
    execution_profile: str | None = None,
    effective_skip_refresh: bool | None = None,
    effective_require_live_assertions: bool | None = None,
    profile_fixture_seeding: dict[str, Any] | None = None,
    runtime_command_returncode: int | None = None,
    runtime_command_ok: bool | None = None,
    operator_cadence_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(payload or {})
    evidence_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "ok": bool(ok),
        "check_id": CHECK_ID,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "command": list(command),
        "execution_profile": execution_profile,
        "effective_skip_refresh": effective_skip_refresh,
        "effective_require_live_assertions": effective_require_live_assertions,
        "profile_fixture_seeding": dict(profile_fixture_seeding or {}),
        "runtime_timeout_sec": runtime_timeout_sec,
        "runtime_timeout_mode": runtime_timeout_mode,
        "runtime_budget_components": dict(runtime_budget_components or {}),
        "runtime_command_returncode": payload.get("_runtime_returncode") if runtime_command_returncode is None else runtime_command_returncode,
        "runtime_command_ok": payload.get("_runtime_ok") if runtime_command_ok is None else runtime_command_ok,
        "runtime_command_error": error,
        "runtime_overall_verdict": evidence_summary.get("overall_verdict"),
        "active_top_blocker": evidence_summary.get("active_top_blocker"),
        "repeatability": (
            dict(evidence_summary.get("repeatability") or {})
            if isinstance(evidence_summary.get("repeatability"), dict)
            else {}
        ),
        "blocked_reasons": list(evidence_summary.get("blocked_reasons") or []),
        "operator_cadence_evidence": dict(operator_cadence_evidence or {}),
        "total": len(results),
        "passed": sum(1 for row in results if bool(row.get("ok"))),
        "failed": sum(1 for row in results if not bool(row.get("ok"))),
        "results": results,
    }


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    runtime_script = (repo_root / args.runtime_command).resolve()
    execution_profile = str(args.execution_profile or DEFAULT_EXECUTION_PROFILE)
    effective_skip_refresh = bool(args.skip_refresh) or execution_profile in {
        SHADOW_EXECUTION_PROFILE,
        STRICT_CLUSTER_SAFE_EXECUTION_PROFILE,
    }
    effective_require_live_assertions = bool(args.require_live_assertions) or execution_profile == STRICT_CLUSTER_SAFE_EXECUTION_PROFILE
    command = [
        str(args.python_bin),
        str(runtime_script),
        "--repo-root",
        str(repo_root),
        "--cycles",
        str(int(args.cycles)),
        "--stress-timeout-sec",
        str(int(args.stress_timeout_sec)),
        "--refresh-timeout-sec",
        str(int(args.refresh_timeout_sec)),
        "--json",
    ]
    if effective_skip_refresh:
        command.append("--skip-refresh")
    if effective_require_live_assertions:
        command.append("--require-live-assertions")

    profile_fixture_seeding: dict[str, Any] = {}

    runtime_timeout_sec_raw = args.runtime_timeout_sec
    runtime_timeout_mode = "derived_default" if runtime_timeout_sec_raw is None else "explicit"
    runtime_budget_components = _runtime_budget_components(args, effective_skip_refresh=effective_skip_refresh)
    if runtime_timeout_sec_raw is None:
        runtime_timeout_sec = int(runtime_budget_components["derived_uncapped_timeout_sec"])
        auto_budget_cap = int(runtime_budget_components["max_auto_runtime_timeout_sec"])
        if runtime_timeout_sec > auto_budget_cap:
            summary = _summary(
                ok=False,
                results=[
                    _result(
                        name="runtime_command_runtime_budget",
                        ok=False,
                        expectation=(
                            "derived default wrapper runtime budget stays within auto safety cap; "
                            "set --runtime-timeout-sec explicitly for longer manual runs"
                        ),
                        details={
                            "runtime_timeout_sec": runtime_timeout_sec,
                            "max_auto_runtime_timeout_sec": auto_budget_cap,
                            "stress_timeout_sec": runtime_budget_components.get("stress_timeout_sec"),
                            "refresh_timeout_sec": runtime_budget_components.get("refresh_timeout_sec"),
                            "skip_refresh": runtime_budget_components.get("skip_refresh"),
                        },
                        error="derived_runtime_timeout_exceeds_auto_budget_cap",
                    )
                ],
                command=command,
                error="derived_runtime_timeout_exceeds_auto_budget_cap",
                runtime_timeout_sec=runtime_timeout_sec,
                runtime_timeout_mode=runtime_timeout_mode,
                runtime_budget_components=runtime_budget_components,
                execution_profile=execution_profile,
                effective_skip_refresh=effective_skip_refresh,
                effective_require_live_assertions=effective_require_live_assertions,
                profile_fixture_seeding=profile_fixture_seeding,
                runtime_command_ok=False,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
    else:
        runtime_timeout_sec = int(runtime_timeout_sec_raw)

    if runtime_timeout_sec < 1:
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="runtime_command_runtime_budget",
                    ok=False,
                    expectation="wrapper runtime timeout budget is a positive integer",
                    details={"runtime_timeout_sec": runtime_timeout_sec},
                    error="invalid_runtime_timeout_sec",
                )
            ],
            command=command,
            error="invalid_runtime_timeout_sec",
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    if not runtime_script.exists():
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="runtime_command_present",
                    ok=False,
                    expectation="runtime evidence command script exists under repo root",
                    error=f"missing_runtime_command:{runtime_script}",
                )
            ],
            command=command,
            error="runtime_command_missing",
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    restore_entries: list[dict[str, Any]] = []
    if execution_profile == STRICT_CLUSTER_SAFE_EXECUTION_PROFILE:
        try:
            profile_fixture_seeding, restore_entries = _seed_strict_cluster_safe_surfaces(repo_root)
        except Exception as exc:
            profile_fixture_seeding = {
                "applied": False,
                "error": str(exc),
                "seeded_top_blocker": STRICT_CLUSTER_SAFE_FIXTURE_TOP_BLOCKER,
            }
            summary = _summary(
                ok=False,
                results=[
                    _result(
                        name="strict_cluster_safe_fixture_seed",
                        ok=False,
                        expectation=(
                            "strict_cluster_safe profile seeds deterministic baseline surfaces before runtime execution"
                        ),
                        details=dict(profile_fixture_seeding),
                        error="strict_cluster_safe_fixture_seed_failed",
                    )
                ],
                command=command,
                error="strict_cluster_safe_fixture_seed_failed",
                runtime_timeout_sec=runtime_timeout_sec,
                runtime_timeout_mode=runtime_timeout_mode,
                runtime_budget_components=runtime_budget_components,
                execution_profile=execution_profile,
                effective_skip_refresh=effective_skip_refresh,
                effective_require_live_assertions=effective_require_live_assertions,
                profile_fixture_seeding=profile_fixture_seeding,
                runtime_command_ok=False,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 1

    cp: subprocess.CompletedProcess[str] | None = None
    timeout_exc: subprocess.TimeoutExpired | None = None
    try:
        cp = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=str(repo_root),
            check=False,
            timeout=runtime_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_exc = exc

    restore_error: str | None = None
    if restore_entries:
        try:
            _restore_seeded_surfaces(restore_entries)
        except Exception as exc:
            restore_error = str(exc)

    if restore_error:
        profile_fixture_seeding["restore_error"] = restore_error
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="strict_cluster_safe_fixture_restore",
                    ok=False,
                    expectation=(
                        "strict_cluster_safe profile restores pre-run baseline surfaces after runtime execution"
                    ),
                    details={
                        "restore_error": restore_error,
                        "runtime_timeout_sec": runtime_timeout_sec,
                        "runtime_command_returncode": int(cp.returncode) if isinstance(cp, subprocess.CompletedProcess) else None,
                    },
                    error="strict_cluster_safe_fixture_restore_failed",
                )
            ],
            command=command,
            error="strict_cluster_safe_fixture_restore_failed",
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_returncode=int(cp.returncode) if isinstance(cp, subprocess.CompletedProcess) else None,
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    if timeout_exc is not None:
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="runtime_command_runtime_budget",
                    ok=False,
                    expectation="runtime evidence command completes within wrapper runtime budget",
                    details={
                        "runtime_timeout_sec": runtime_timeout_sec,
                        "stdout_tail": _tail(str(timeout_exc.stdout or "")),
                        "stderr_tail": _tail(str(timeout_exc.stderr or "")),
                    },
                    error="runtime_command_timeout",
                )
            ],
            command=command,
            error="runtime_command_timeout",
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    if cp is None:
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="runtime_command_execution",
                    ok=False,
                    expectation="runtime evidence command returns a process result or timeout",
                    error="runtime_command_execution_missing_result",
                )
            ],
            command=command,
            error="runtime_command_execution_missing_result",
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    try:
        payload = _parse_json(cp.stdout)
    except Exception as exc:
        summary = _summary(
            ok=False,
            results=[
                _result(
                    name="runtime_command_json_contract",
                    ok=False,
                    expectation="runtime evidence command emits JSON object summary",
                    details={
                        "stdout_tail": _tail(cp.stdout),
                        "stderr_tail": _tail(cp.stderr),
                        "returncode": int(cp.returncode),
                    },
                    error=str(exc),
                )
            ],
            command=command,
            error=str(exc),
            runtime_timeout_sec=runtime_timeout_sec,
            runtime_timeout_mode=runtime_timeout_mode,
            runtime_budget_components=runtime_budget_components,
            execution_profile=execution_profile,
            effective_skip_refresh=effective_skip_refresh,
            effective_require_live_assertions=effective_require_live_assertions,
            profile_fixture_seeding=profile_fixture_seeding,
            runtime_command_returncode=int(cp.returncode),
            runtime_command_ok=False,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    payload["_runtime_returncode"] = int(cp.returncode)
    payload["_runtime_ok"] = cp.returncode == 0

    evidence_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    repeatability_summary = (
        evidence_summary.get("repeatability")
        if isinstance(evidence_summary.get("repeatability"), dict)
        else {}
    )
    phases = payload.get("phases") if isinstance(payload.get("phases"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    results: list[dict[str, Any]] = []

    results.append(
        _result(
            name="runtime_command_json_contract",
            ok=str(payload.get("object_type") or "") == "clawd.a3_failover_stress_runtime_evidence.v1",
            expectation="runtime evidence command emits the bounded A3 evidence object",
            details={
                "object_type": payload.get("object_type"),
                "run_id": payload.get("run_id"),
                "returncode": int(cp.returncode),
            },
            error=None if str(payload.get("object_type") or "") == "clawd.a3_failover_stress_runtime_evidence.v1" else "unexpected_object_type",
        )
    )

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    stress_evidence = payload.get("stress_evidence") if isinstance(payload.get("stress_evidence"), dict) else {}
    stress_summary = stress_evidence.get("summary") if isinstance(stress_evidence.get("summary"), dict) else {}

    configured_cycles = max(0, _safe_int(config.get("cycles_per_profile"), _safe_int(args.cycles, 0)))
    stress_profile_count = max(0, _safe_int(stress_summary.get("profile_count"), 0))
    stress_total_cycles = max(0, _safe_int(stress_summary.get("total_cycles"), 0))
    minimum_repeat_cycles = max(2, stress_profile_count * 2)
    expected_total_cycles = stress_profile_count * configured_cycles
    repeated_stress_ok = configured_cycles >= 2 and stress_total_cycles >= minimum_repeat_cycles
    if expected_total_cycles > 0:
        repeated_stress_ok = repeated_stress_ok and stress_total_cycles >= expected_total_cycles

    repeated_stress_error: str | None = None
    if configured_cycles < 2:
        repeated_stress_error = "stress_cycles_below_repeatability_floor"
    elif stress_total_cycles < minimum_repeat_cycles:
        repeated_stress_error = "stress_total_cycles_below_repeatability_floor"
    elif expected_total_cycles > 0 and stress_total_cycles < expected_total_cycles:
        repeated_stress_error = "stress_total_cycles_below_configured_expectation"

    results.append(
        _result(
            name="repeated_stress_evidence_phase",
            ok=repeated_stress_ok,
            expectation="stress evidence proves repeated execution (>=2 cycles/profile) before runtime packet can pass",
            details={
                "configured_cycles_per_profile": configured_cycles,
                "stress_profile_count": stress_profile_count,
                "stress_total_cycles": stress_total_cycles,
                "minimum_repeat_cycles": minimum_repeat_cycles,
                "expected_total_cycles": expected_total_cycles,
            },
            error=repeated_stress_error,
        )
    )

    phase_stress = phases.get("stress_soak") if isinstance(phases.get("stress_soak"), dict) else {}
    stress_ok = str(phase_stress.get("status") or "").upper() == "PASS" and str(evidence_summary.get("stress_verdict") or "").upper() == "PASS"
    results.append(
        _result(
            name="stress_soak_phase",
            ok=stress_ok,
            expectation="stress-soak phase passes and preserves deterministic A3 soak evidence",
            details={
                "status": phase_stress.get("status"),
                "duration_ms": phase_stress.get("duration_ms"),
                "stress_verdict": evidence_summary.get("stress_verdict"),
                "stress_profile_count": stress_profile_count,
                "stress_total_cycles": stress_total_cycles,
            },
            error=None if stress_ok else str((phase_stress.get("error") or {}).get("code") or "stress_soak_not_pass"),
        )
    )

    stress_latest_ref = str(artifacts.get("stress_latest_ref") or "").strip()
    stress_latest_details: dict[str, Any] = {
        "stress_latest_ref": stress_latest_ref or None,
        "stress_latest_inside_repo": None,
        "stress_latest_exists": None,
    }
    stress_latest_payload: dict[str, Any] = {}

    if stress_latest_ref:
        stress_latest_path = (repo_root / stress_latest_ref).resolve()
        stress_latest_details["stress_latest_path"] = str(stress_latest_path)
        try:
            stress_latest_path.relative_to(repo_root)
            stress_latest_inside_repo = True
        except ValueError:
            stress_latest_inside_repo = False
        stress_latest_details["stress_latest_inside_repo"] = stress_latest_inside_repo

        if stress_latest_inside_repo:
            stress_latest_exists = stress_latest_path.exists()
            stress_latest_details["stress_latest_exists"] = stress_latest_exists
            if stress_latest_exists:
                try:
                    loaded = json.loads(stress_latest_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        stress_latest_payload = loaded
                    else:
                        stress_latest_details["stress_latest_error"] = "stress_latest_payload_not_object"
                except Exception as exc:
                    stress_latest_details["stress_latest_error"] = f"stress_latest_unreadable:{exc}"

    linkage = (
        stress_latest_payload.get("live_surface_linkage")
        if isinstance(stress_latest_payload.get("live_surface_linkage"), dict)
        else stress_evidence.get("live_surface_linkage")
        if isinstance(stress_evidence.get("live_surface_linkage"), dict)
        else {}
    )
    blocked_rows = [
        row
        for row in (linkage.get("blocked_reset_successor_failures") or [])
        if isinstance(row, dict)
    ]
    blocked_rows_by_key = {
        (str(row.get("failure_class") or "").strip(), str(row.get("top_blocker") or "").strip()): row
        for row in blocked_rows
    }

    required_blocked_count = max(1, configured_cycles)
    blocked_reset_failures: list[str] = []
    for required in REQUIRED_BLOCKED_RESET_PROOF_ROWS:
        key = (required["failure_class"], required["top_blocker"])
        row = blocked_rows_by_key.get(key)
        if not isinstance(row, dict):
            blocked_reset_failures.append(f"missing_failure_row:{required['failure_class']}:{required['top_blocker']}")
            continue

        row_count = max(0, _safe_int(row.get("count"), 0))
        profiles = {str(x).strip() for x in (row.get("profiles") or []) if str(x).strip()}
        step_labels = {str(x).strip() for x in (row.get("step_labels") or []) if str(x).strip()}
        projected_live_assertions = [x for x in (row.get("projected_live_assertions") or []) if isinstance(x, dict)]

        if row_count < required_blocked_count:
            blocked_reset_failures.append(
                f"failure_row_count_below_cycles:{required['failure_class']}:{required['top_blocker']}:{row_count}:{required_blocked_count}"
            )
        if required["required_profile"] not in profiles:
            blocked_reset_failures.append(
                f"failure_row_missing_profile:{required['failure_class']}:{required['required_profile']}"
            )
        if required["required_step_label"] not in step_labels:
            blocked_reset_failures.append(
                f"failure_row_missing_step_label:{required['failure_class']}:{required['required_step_label']}"
            )
        if not projected_live_assertions:
            blocked_reset_failures.append(
                f"failure_row_missing_projected_assertions:{required['failure_class']}:{required['top_blocker']}"
            )

    blocked_reset_ok = bool(linkage) and not blocked_reset_failures
    results.append(
        _result(
            name="blocked_reset_successor_proof_phase",
            ok=blocked_reset_ok,
            expectation=(
                "blocked-reset successor failure linkage covers canonical blocker classes with per-cycle evidence "
                "and projected live assertions"
            ),
            details={
                **stress_latest_details,
                "required_blocked_count": required_blocked_count,
                "blocked_failure_row_count": len(blocked_rows),
                "required_failure_row_count": len(REQUIRED_BLOCKED_RESET_PROOF_ROWS),
                "missing_or_failed_rows": blocked_reset_failures,
            },
            error=None if blocked_reset_ok else (blocked_reset_failures[0] if blocked_reset_failures else "blocked_reset_successor_proof_missing"),
        )
    )

    stress_profiles = [
        row
        for row in (stress_latest_payload.get("profiles") or [])
        if isinstance(row, dict)
    ]
    stress_profiles_by_id = {
        str(row.get("profile_id") or "").strip(): row
        for row in stress_profiles
        if str(row.get("profile_id") or "").strip()
    }

    successor_stability_failures: list[str] = []
    successor_profile_details: dict[str, Any] = {}
    for profile_id in REQUIRED_SUCCESSOR_TAKEOVER_STABILITY_PROFILES:
        row = stress_profiles_by_id.get(profile_id)
        if not isinstance(row, dict):
            successor_stability_failures.append(f"missing_takeover_profile:{profile_id}")
            successor_profile_details[profile_id] = None
            continue

        pass_count = max(0, _safe_int(row.get("pass_count"), 0))
        fail_count = max(0, _safe_int(row.get("fail_count"), 0))
        deterministic_signature_count = max(0, _safe_int(row.get("deterministic_signature_count"), 0))
        drift_detected = bool(row.get("drift_detected"))
        terminal_state_counts = row.get("terminal_state_counts") if isinstance(row.get("terminal_state_counts"), dict) else {}
        healthy_terminal_count = max(0, _safe_int(terminal_state_counts.get("HEALTHY"), 0))

        successor_profile_details[profile_id] = {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "deterministic_signature_count": deterministic_signature_count,
            "drift_detected": drift_detected,
            "healthy_terminal_count": healthy_terminal_count,
            "cycles": _safe_int(row.get("cycles"), 0),
        }

        if pass_count < required_blocked_count:
            successor_stability_failures.append(f"takeover_profile_pass_count_below_cycles:{profile_id}:{pass_count}:{required_blocked_count}")
        if fail_count != 0:
            successor_stability_failures.append(f"takeover_profile_fail_count_nonzero:{profile_id}:{fail_count}")
        if deterministic_signature_count != 1:
            successor_stability_failures.append(
                f"takeover_profile_signature_drift:{profile_id}:{deterministic_signature_count}"
            )
        if drift_detected:
            successor_stability_failures.append(f"takeover_profile_drift_detected:{profile_id}")
        if healthy_terminal_count < required_blocked_count:
            successor_stability_failures.append(
                f"takeover_profile_healthy_terminal_count_below_cycles:{profile_id}:{healthy_terminal_count}:{required_blocked_count}"
            )

    successor_takeover_ok = bool(stress_profiles_by_id) and not successor_stability_failures
    results.append(
        _result(
            name="successor_takeover_stability_phase",
            ok=successor_takeover_ok,
            expectation=(
                "successor takeover profiles remain deterministic and fully convergent across repeated stress cycles"
            ),
            details={
                **stress_latest_details,
                "required_profiles": list(REQUIRED_SUCCESSOR_TAKEOVER_STABILITY_PROFILES),
                "profile_checks": successor_profile_details,
                "failures": successor_stability_failures,
            },
            error=None if successor_takeover_ok else (successor_stability_failures[0] if successor_stability_failures else "successor_takeover_stability_not_proven"),
        )
    )

    phase_refresh = phases.get("refresh_publish_chain") if isinstance(phases.get("refresh_publish_chain"), dict) else {}
    expected_refresh_status = {"SKIPPED"} if effective_skip_refresh else {"PASS"}
    refresh_status = str(phase_refresh.get("status") or "").upper()
    refresh_ok = refresh_status in expected_refresh_status and str(evidence_summary.get("refresh_verdict") or "").upper() == "PASS"
    results.append(
        _result(
            name="refresh_publish_chain_phase",
            ok=refresh_ok,
            expectation=(
                "refresh/publish phase is skipped only when explicitly requested or when shadow_baseline profile is selected"
                if effective_skip_refresh
                else "refresh/publish phase completes successfully"
            ),
            details={
                "status": phase_refresh.get("status"),
                "refresh_verdict": evidence_summary.get("refresh_verdict"),
                "latest_surface_ref": phase_refresh.get("latest_surface_ref"),
            },
            error=None if refresh_ok else str((phase_refresh.get("error") or {}).get("code") or "refresh_publish_not_pass"),
        )
    )

    phase_publish = phases.get("publish_assertion_verification") if isinstance(phases.get("publish_assertion_verification"), dict) else {}
    publish_reason = str(phase_publish.get("reason") or "")
    publish_ok = str(phase_publish.get("verdict") or "").upper() == "PASS" and str(evidence_summary.get("publish_chain_verdict") or "").upper() == "PASS"
    publish_shadow_tolerated = False
    if not publish_ok and execution_profile == SHADOW_EXECUTION_PROFILE:
        publish_shadow_tolerated = publish_reason in SHADOW_TOLERATED_PUBLISH_REASONS
        if publish_shadow_tolerated:
            publish_ok = True

    publish_details: dict[str, Any] = {
        "verdict": phase_publish.get("verdict"),
        "reason": phase_publish.get("reason"),
        "active_top_blocker": phase_publish.get("active_top_blocker"),
        "assertions_checked": phase_publish.get("assertions_checked"),
        "assertions_failed": phase_publish.get("assertions_failed"),
        "repeatability_status": repeatability_summary.get("status"),
        "repeatability_match": repeatability_summary.get("match"),
    }
    if publish_shadow_tolerated:
        publish_details["shadow_profile_tolerated"] = True
        publish_details["shadow_profile_tolerated_reason"] = publish_reason

    results.append(
        _result(
            name="publish_assertion_verification_phase",
            ok=publish_ok,
            expectation=(
                "projected live assertions verify against current publish-chain surfaces"
                if execution_profile == DEFAULT_EXECUTION_PROFILE
                else "projected live assertions verify, or a known live-blocker mapping drift is tolerated in shadow mode"
            ),
            details=publish_details,
            error=None if publish_ok else str(phase_publish.get("reason") or "publish_assertion_verification_not_pass"),
        )
    )

    latest_ref = str(artifacts.get("latest_ref") or "").strip()
    evidence_ref = str(artifacts.get("evidence_ref") or "").strip()
    latest_ok = bool(latest_ref and evidence_ref)
    latest_details: dict[str, Any] = {
        "latest_ref": latest_ref or None,
        "evidence_ref": evidence_ref or None,
    }
    if latest_ok:
        latest_path = (repo_root / latest_ref).resolve()
        evidence_path = (repo_root / evidence_ref).resolve()
        latest_exists = latest_path.exists()
        evidence_exists = evidence_path.exists()
        latest_details["latest_exists"] = latest_exists
        latest_details["evidence_exists"] = evidence_exists
        latest_ok = latest_exists and evidence_exists
    results.append(
        _result(
            name="runtime_evidence_artifacts_written",
            ok=latest_ok,
            expectation="runtime evidence writes both latest and run-scoped evidence artifacts",
            details=latest_details,
            error=None if latest_ok else "runtime_evidence_artifacts_missing",
        )
    )

    run_id = str(payload.get("run_id") or "").strip()
    runtime_verdict = str(evidence_summary.get("overall_verdict") or "").upper()
    decision_log_ref = str(artifacts.get("decision_log_ref") or "").strip()
    cadence_ok = bool(run_id and decision_log_ref)
    cadence_error: str | None = None
    cadence_details: dict[str, Any] = {
        "run_id": run_id or None,
        "runtime_overall_verdict": runtime_verdict or None,
        "decision_log_ref": decision_log_ref or None,
        "repeatability_status": repeatability_summary.get("status"),
        "repeatability_match": repeatability_summary.get("match"),
        "repeatability_mismatch_fields": list(repeatability_summary.get("mismatch_fields") or []),
    }

    def _mark_cadence_error(error_code: str) -> None:
        nonlocal cadence_ok, cadence_error
        cadence_ok = False
        if cadence_error is None:
            cadence_error = str(error_code)

    if repeatability_summary.get("status") == "mismatch":
        _mark_cadence_error("runtime_repeatability_mismatch")

    if not run_id:
        _mark_cadence_error("runtime_run_id_missing")
    elif not decision_log_ref:
        _mark_cadence_error("runtime_decision_log_ref_missing")
    else:
        decision_log_path = (repo_root / decision_log_ref).resolve()
        cadence_details["decision_log_path"] = str(decision_log_path)

        try:
            decision_log_path.relative_to(repo_root)
            decision_log_inside_repo = True
        except ValueError:
            decision_log_inside_repo = False

        cadence_details["decision_log_inside_repo"] = decision_log_inside_repo
        if not decision_log_inside_repo:
            _mark_cadence_error("runtime_decision_log_outside_repo")
        elif not decision_log_path.exists():
            _mark_cadence_error("runtime_decision_log_missing")
        else:
            try:
                rows: list[dict[str, Any]] = []
                for line_no, line in enumerate(decision_log_path.read_text(encoding="utf-8").splitlines(), start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise RuntimeError(f"runtime_decision_log_row_not_object:line_{line_no}")
                    rows.append(row)
            except Exception as exc:
                _mark_cadence_error(f"runtime_decision_log_invalid_jsonl:{exc}")
                rows = []

            if rows:
                cadence_details["decision_log_entry_count"] = len(rows)
                matching_indexes = [
                    idx for idx, row in enumerate(rows) if str(row.get("run_id") or "").strip() == run_id
                ]
                cadence_details["run_id_entry_count"] = len(matching_indexes)
                cadence_details["run_entry_present"] = bool(matching_indexes)

                if not matching_indexes:
                    _mark_cadence_error("runtime_decision_log_run_entry_missing")
                else:
                    current_index = matching_indexes[-1]
                    current_row = rows[current_index]
                    run_entry_is_latest = current_index == (len(rows) - 1)
                    cadence_details["run_entry_is_latest"] = run_entry_is_latest

                    if len(matching_indexes) > 1:
                        _mark_cadence_error("runtime_decision_log_duplicate_run_entries")
                    if not run_entry_is_latest:
                        _mark_cadence_error("runtime_decision_log_run_entry_not_latest")

                    row_verdict = str(current_row.get("verdict") or "").upper()
                    cadence_details["run_entry_verdict"] = row_verdict or None
                    cadence_details["run_entry_generated_at"] = current_row.get("generated_at")
                    current_generated_at = _parse_iso8601(current_row.get("generated_at"))
                    cadence_details["run_entry_generated_at_parseable"] = current_generated_at is not None
                    if current_generated_at is None:
                        _mark_cadence_error("runtime_decision_log_generated_at_unparseable")

                    previous_row = rows[current_index - 1] if current_index > 0 else None
                    if isinstance(previous_row, dict):
                        cadence_details["previous_run_id"] = previous_row.get("run_id")
                        cadence_details["previous_run_generated_at"] = previous_row.get("generated_at")

                        previous_generated_at = _parse_iso8601(previous_row.get("generated_at"))
                        cadence_details["previous_run_generated_at_parseable"] = previous_generated_at is not None
                        if current_generated_at and previous_generated_at:
                            delta_sec = int((current_generated_at - previous_generated_at).total_seconds())
                            cadence_details["since_previous_run_sec_raw"] = delta_sec
                            cadence_details["since_previous_run_sec"] = max(delta_sec, 0)
                            cadence_details["generated_at_monotonic"] = delta_sec >= 0
                            if delta_sec < 0:
                                _mark_cadence_error("runtime_decision_log_non_monotonic_generated_at")

                    if runtime_verdict and row_verdict and row_verdict != runtime_verdict:
                        _mark_cadence_error("runtime_decision_log_verdict_mismatch")
            else:
                _mark_cadence_error("runtime_decision_log_empty")

    operator_cadence_evidence = {
        "ok": cadence_ok,
        **cadence_details,
    }
    if cadence_error:
        operator_cadence_evidence["error"] = cadence_error

    results.append(
        _result(
            name="runtime_decision_log_evidence",
            ok=cadence_ok,
            expectation=(
                "runtime evidence appends a unique latest run-scoped decision-log row with verdict/timestamp integrity "
                "for operator cadence audits"
            ),
            details=cadence_details,
            error=cadence_error,
        )
    )

    summary = _summary(
        ok=all(bool(row.get("ok")) for row in results),
        results=results,
        command=command,
        payload=payload,
        runtime_timeout_sec=runtime_timeout_sec,
        runtime_timeout_mode=runtime_timeout_mode,
        runtime_budget_components=runtime_budget_components,
        execution_profile=execution_profile,
        effective_skip_refresh=effective_skip_refresh,
        effective_require_live_assertions=effective_require_live_assertions,
        profile_fixture_seeding=profile_fixture_seeding,
        operator_cadence_evidence=operator_cadence_evidence,
    )
    for row in results:
        prefix = "PASS" if bool(row.get("ok")) else "FAIL"
        print(f"{prefix}: {row.get('name')}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
