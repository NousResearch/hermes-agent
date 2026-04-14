#!/usr/bin/env python3
"""Guarded driver for cron session-card lifecycle reconciliation writes.

This wrapper operationalizes the existing authoritative write-path script with
strict input validation, optional input collection from live ops surfaces, and
concise machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENCLAW_DIR = Path(__file__).resolve().parent
DEFAULT_RECONCILE_SCRIPT = OPENCLAW_DIR / "reconcile_cron_session_card_lifecycle.py"
DEFAULT_GUARD_SCRIPT = OPENCLAW_DIR / "no_llm_watchdog_cron_authority_guard.sh"


def _emit(payload: Dict[str, Any], code: int) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return code


def _normalize_json_obj(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label}_not_found:{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label}_invalid_json:{path}:{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label}_not_object:{path}")
    return raw


def _validate_summary_payload(summary: Dict[str, Any]) -> None:
    reconciliation = summary.get("session_surface_reconciliation")
    if not isinstance(reconciliation, dict):
        raise ValueError("summary_missing_session_surface_reconciliation")
    rows = reconciliation.get("historical_failed_session_rows")
    if not isinstance(rows, list):
        raise ValueError("summary_missing_historical_failed_session_rows")


def _validate_sessions_payload(sessions: Dict[str, Any], extra_store_paths: Sequence[str]) -> None:
    paths = sessions.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("sessions_list_missing_paths_map")

    valid_paths = [
        str(v).strip()
        for k, v in paths.items()
        if str(k or "").strip() and str(v or "").strip()
    ]
    if not valid_paths and not list(extra_store_paths):
        raise ValueError("sessions_list_paths_empty")


def _parse_guard_output(stdout: str) -> tuple[str, Dict[str, Any]]:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("guard_output_empty")

    status_line = lines[0]
    if not status_line.startswith("READY:"):
        raise RuntimeError(f"guard_not_ready:{status_line}")

    summary_obj: Optional[Dict[str, Any]] = None
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(candidate, dict):
            summary_obj = candidate
            break

    if summary_obj is None:
        raise ValueError("guard_summary_json_missing")
    return status_line, summary_obj


def _collect_inputs(
    *,
    openclaw_bin: str,
    session_active_minutes: int,
    guard_script: Path,
    expected_names: str,
    timeout_sec: int,
) -> tuple[str, Path, Path, Path]:
    if not guard_script.is_file():
        raise ValueError(f"guard_script_not_found:{guard_script}")

    temp_root = Path(tempfile.mkdtemp(prefix="cron_session_card_lifecycle_driver_inputs_"))
    sessions_list_path = temp_root / "sessions_list.json"
    guard_summary_path = temp_root / "guard_summary.json"

    sessions_cp = subprocess.run(
        [
            openclaw_bin,
            "sessions",
            "--all-agents",
            "--active",
            str(max(1, int(session_active_minutes))),
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(5, int(timeout_sec)),
    )
    if sessions_cp.returncode != 0:
        err = (sessions_cp.stderr or sessions_cp.stdout or "").strip()
        raise RuntimeError(f"sessions_collect_failed:rc={sessions_cp.returncode};err={err[:220]}")

    sessions_list_path.write_text(sessions_cp.stdout, encoding="utf-8")

    guard_cmd = [str(guard_script), "--sessions-json", str(sessions_list_path), "--json"]
    if expected_names:
        guard_cmd.extend(["--expected-names", expected_names])

    guard_cp = subprocess.run(
        guard_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(5, int(timeout_sec)),
    )
    if guard_cp.returncode != 0:
        err = (guard_cp.stderr or "").strip()
        raise RuntimeError(f"guard_command_failed:rc={guard_cp.returncode};err={err[:220]}")

    guard_status_line, guard_summary_obj = _parse_guard_output(guard_cp.stdout)
    guard_summary_path.write_text(json.dumps(guard_summary_obj, ensure_ascii=False), encoding="utf-8")

    return guard_status_line, guard_summary_path, sessions_list_path, temp_root


def _run_reconcile(
    *,
    python_bin: str,
    reconcile_script: Path,
    summary_json: Path,
    sessions_list_json: Path,
    store_paths: Sequence[str],
    apply_changes: bool,
    timeout_sec: int,
) -> Dict[str, Any]:
    if not reconcile_script.is_file():
        raise ValueError(f"reconcile_script_not_found:{reconcile_script}")

    cmd = [
        python_bin,
        str(reconcile_script),
        "--summary-json",
        str(summary_json),
        "--sessions-list-json",
        str(sessions_list_json),
    ]
    for store_path in store_paths:
        cmd.extend(["--store-path", store_path])
    if apply_changes:
        cmd.append("--apply")

    cp = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(5, int(timeout_sec)),
    )

    payload: Dict[str, Any]
    try:
        payload_raw = json.loads((cp.stdout or "").strip() or "{}")
        payload = payload_raw if isinstance(payload_raw, dict) else {"ok": False, "error": "reconcile_output_not_object"}
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error": f"reconcile_output_invalid_json:{exc}",
            "stderr": (cp.stderr or "")[:400],
            "stdout": (cp.stdout or "")[:400],
        }

    if cp.returncode != 0 or payload.get("ok") is not True:
        err = str(payload.get("error") or "reconcile_failed")
        raise RuntimeError(f"reconcile_failed:rc={cp.returncode};error={err}")

    return payload


def _normalize_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001
        return 0
    return parsed if parsed >= 0 else 0


def _normalize_reason_token(value: str) -> str:
    txt = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    chars = []
    for char in txt:
        if char.isalnum() or char == "_":
            chars.append(char)
    normalized = "".join(chars).strip("_")
    return normalized or "unknown"


def _build_single_run_promotion_evidence(
    *,
    requested_mode: str,
    state: str,
    reason: str,
    candidate_count: int,
    max_candidates: int,
    min_age_sec: int,
    age_known_count: int,
    age_unknown_count: int,
    oldest_age_sec: Optional[int],
    newest_age_sec: Optional[int],
    apply_enabled: bool,
) -> str:
    oldest = str(_normalize_non_negative_int(oldest_age_sec)) if oldest_age_sec is not None else "unknown"
    newest = str(_normalize_non_negative_int(newest_age_sec)) if newest_age_sec is not None else "unknown"
    return (
        f"v1|requested={requested_mode}|state={state}|reason={_normalize_reason_token(reason)}"
        f"|candidate={_normalize_non_negative_int(candidate_count)}|max_candidates={_normalize_non_negative_int(max_candidates)}"
        f"|min_age_sec={_normalize_non_negative_int(min_age_sec)}|age_known={_normalize_non_negative_int(age_known_count)}"
        f"|age_unknown={_normalize_non_negative_int(age_unknown_count)}|oldest_age_sec={oldest}|newest_age_sec={newest}"
        f"|apply_enabled={'1' if apply_enabled else '0'}"
    )


def _evaluate_single_run_promotion(
    *,
    requested_mode: str,
    rows: Sequence[Any],
    enabled: bool,
    max_candidates: int,
    min_age_sec: int,
    apply_enabled: bool,
) -> Dict[str, Any]:
    candidate_count = len(rows)
    now_ms = int(time.time() * 1000)
    candidate_age_secs: List[int] = []

    for row in rows:
        updated_at_ms = getattr(row, "updated_at_ms", None)
        try:
            updated_at_ms_int = int(updated_at_ms)
        except Exception:  # noqa: BLE001
            continue
        if updated_at_ms_int < 0:
            continue
        candidate_age_secs.append(max(0, int((now_ms - updated_at_ms_int) / 1000)))

    age_known_count = len(candidate_age_secs)
    age_unknown_count = max(0, candidate_count - age_known_count)
    oldest_age_sec = max(candidate_age_secs) if candidate_age_secs else None
    newest_age_sec = min(candidate_age_secs) if candidate_age_secs else None
    threshold_candidates = max(0, int(max_candidates))
    threshold_age_sec = max(0, int(min_age_sec))

    if requested_mode != "dry_run":
        state = "not_applicable"
        reason = "requested_mode_apply"
    elif not enabled:
        state = "disabled"
        reason = "single_run_auto_apply_disabled"
    elif candidate_count <= 0:
        state = "demoted"
        reason = "no_reconcilable_candidates"
    elif candidate_count > threshold_candidates:
        state = "demoted"
        reason = "candidate_count_exceeds_threshold"
    elif age_unknown_count > 0 or newest_age_sec is None:
        state = "demoted"
        reason = "candidate_age_metadata_unavailable"
    elif newest_age_sec < threshold_age_sec:
        state = "demoted"
        reason = "candidate_too_recent"
    elif not apply_enabled:
        state = "demoted"
        reason = "apply_gate_disabled"
    else:
        state = "promoted"
        reason = "eligible_small_stale_residue"

    return {
        "requested_mode": requested_mode,
        "state": state,
        "reason": reason,
        "candidate_count": candidate_count,
        "max_candidates": threshold_candidates,
        "min_age_sec": threshold_age_sec,
        "age_known_count": age_known_count,
        "age_unknown_count": age_unknown_count,
        "oldest_age_sec": oldest_age_sec,
        "newest_age_sec": newest_age_sec,
        "apply_enabled": bool(apply_enabled),
        "effective_mode": "apply" if state == "promoted" else requested_mode,
        "evidence": _build_single_run_promotion_evidence(
            requested_mode=requested_mode,
            state=state,
            reason=reason,
            candidate_count=candidate_count,
            max_candidates=threshold_candidates,
            min_age_sec=threshold_age_sec,
            age_known_count=age_known_count,
            age_unknown_count=age_unknown_count,
            oldest_age_sec=oldest_age_sec,
            newest_age_sec=newest_age_sec,
            apply_enabled=apply_enabled,
        ),
    }


def _build_reconcile_rollup(
    *,
    status: str,
    mode: str,
    apply_enable_policy: str,
    result_mode: str,
    verification_state: str,
    candidate_count: int,
    mutated_count: int,
    verified_mutation_count: int,
    verification_mismatch_count: int,
    verification_store_load_error_count: int,
    skipped_active_running_count: int,
    skipped_non_failed_like_count: int,
    skipped_missing_session_count: int,
    skipped_missing_store_count: int,
    store_load_error_count: int,
) -> str:
    skipped_total = (
        _normalize_non_negative_int(skipped_active_running_count)
        + _normalize_non_negative_int(skipped_non_failed_like_count)
        + _normalize_non_negative_int(skipped_missing_session_count)
        + _normalize_non_negative_int(skipped_missing_store_count)
    )
    error_total = (
        _normalize_non_negative_int(verification_mismatch_count)
        + _normalize_non_negative_int(verification_store_load_error_count)
        + _normalize_non_negative_int(store_load_error_count)
    )

    if status == "dry_run_ok":
        outcome = "dry_run_projected"
    elif status == "apply_ok":
        outcome = "apply_verified_mutated" if _normalize_non_negative_int(mutated_count) > 0 else "apply_verified_noop"
    else:
        outcome = status

    return (
        f"v1|status={status}|mode={mode}|policy={apply_enable_policy}|result={result_mode}"
        f"|verification={verification_state}|outcome={outcome}|candidate={_normalize_non_negative_int(candidate_count)}"
        f"|mutated={_normalize_non_negative_int(mutated_count)}|verified={_normalize_non_negative_int(verified_mutation_count)}"
        f"|skipped_total={skipped_total}|error_total={error_total}"
    )


def _summarize_reconcile_payload(payload: Dict[str, Any], *, promotion: Dict[str, Any]) -> Dict[str, Any]:
    mutated = payload.get("mutated") if isinstance(payload.get("mutated"), list) else []
    store_paths_touched = payload.get("store_paths_touched") if isinstance(payload.get("store_paths_touched"), list) else []

    mode = str(payload.get("mode") or "dry_run")
    candidate_count = int(payload.get("candidate_count") or 0)
    mutated_count = int(payload.get("mutated_count") or 0)
    verification_state = str(payload.get("verification_state") or "unknown")
    verified_mutation_count = int(payload.get("verified_mutation_count") or 0)
    verification_mismatch_count = int(payload.get("verification_mismatch_count") or 0)
    verification_store_load_error_count = int(payload.get("verification_store_load_error_count") or 0)
    skipped_active_running_count = len(payload.get("skipped_active_running") or [])
    skipped_non_failed_like_count = len(payload.get("skipped_non_failed_like") or [])
    skipped_missing_session_count = len(payload.get("skipped_missing_session") or [])
    skipped_missing_store_count = len(payload.get("skipped_missing_store") or [])
    store_load_error_count = len(payload.get("store_load_errors") or [])

    status = "apply_ok" if mode == "apply" else "dry_run_ok"
    apply_enable_policy = "allowed" if mode == "apply" else "not_applicable"
    rollup = _build_reconcile_rollup(
        status=status,
        mode=mode,
        apply_enable_policy=apply_enable_policy,
        result_mode=mode,
        verification_state=verification_state,
        candidate_count=candidate_count,
        mutated_count=mutated_count,
        verified_mutation_count=verified_mutation_count,
        verification_mismatch_count=verification_mismatch_count,
        verification_store_load_error_count=verification_store_load_error_count,
        skipped_active_running_count=skipped_active_running_count,
        skipped_non_failed_like_count=skipped_non_failed_like_count,
        skipped_missing_session_count=skipped_missing_session_count,
        skipped_missing_store_count=skipped_missing_store_count,
        store_load_error_count=store_load_error_count,
    )

    return {
        "status": status,
        "mode": mode,
        "apply_enable_policy": apply_enable_policy,
        "single_run_promotion_state": str(promotion.get("state") or "unknown"),
        "single_run_promotion_reason": str(promotion.get("reason") or "unknown"),
        "single_run_promotion_evidence": str(promotion.get("evidence") or ""),
        "candidate_count": candidate_count,
        "mutated_count": mutated_count,
        "verification_state": verification_state,
        "verified_mutation_count": verified_mutation_count,
        "verification_mismatch_count": verification_mismatch_count,
        "verification_store_load_error_count": verification_store_load_error_count,
        "mutated_session_keys": [
            str(row.get("session_key") or "")
            for row in mutated
            if isinstance(row, dict) and str(row.get("session_key") or "").strip()
        ][:50],
        "store_paths_touched": [str(p) for p in store_paths_touched][:50],
        "skipped_active_running_count": skipped_active_running_count,
        "skipped_non_failed_like_count": skipped_non_failed_like_count,
        "skipped_missing_session_count": skipped_missing_session_count,
        "skipped_missing_store_count": skipped_missing_store_count,
        "store_load_error_count": store_load_error_count,
        "rollup": rollup,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded driver for ops/openclaw/reconcile_cron_session_card_lifecycle.py",
    )
    parser.add_argument("--summary-json", help="Precomputed guard summary JSON path")
    parser.add_argument("--sessions-list-json", help="Precomputed openclaw sessions --json payload path")
    parser.add_argument("--collect-inputs", action="store_true", help="Collect guard+sessions inputs live before reconcile")
    parser.add_argument("--guard-script", default=str(DEFAULT_GUARD_SCRIPT), help="Guard script path used in collect mode")
    parser.add_argument(
        "--expected-names",
        default=os.environ.get("OPENCLAW_NO_LLM_AUTHORITY_EXPECTED_NAMES", ""),
        help="Optional expected names CSV passed to guard script in collect mode",
    )
    parser.add_argument(
        "--session-active-minutes",
        type=int,
        default=int(os.environ.get("OPENCLAW_NO_LLM_AUTHORITY_SESSION_ACTIVE_MINUTES", "10080")),
        help="Session activity horizon when collecting openclaw sessions",
    )
    parser.add_argument("--openclaw-bin", default="openclaw", help="OpenClaw binary name/path for collect mode")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for reconcile invocation")
    parser.add_argument("--reconcile-script", default=str(DEFAULT_RECONCILE_SCRIPT), help="Reconcile script path")
    parser.add_argument("--store-path", action="append", default=[], help="Additional explicit session store path(s)")
    parser.add_argument("--apply", action="store_true", help="Apply writes (default dry-run)")
    parser.add_argument(
        "--single-run-auto-apply",
        action="store_true",
        help="When requested mode is dry-run, allow one authoritative auto-promotion to apply if narrow eligibility rules pass",
    )
    parser.add_argument(
        "--single-run-apply-enabled",
        action="store_true",
        help="Explicitly allow single-run auto-promotion to execute apply when eligibility rules pass",
    )
    parser.add_argument(
        "--single-run-max-candidates",
        type=int,
        default=3,
        help="Maximum candidate count eligible for single-run auto-promotion",
    )
    parser.add_argument(
        "--single-run-min-age-sec",
        type=int,
        default=172800,
        help="Minimum age in seconds for every candidate eligible for single-run auto-promotion",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Command timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    summary_arg = str(args.summary_json or "").strip()
    sessions_arg = str(args.sessions_list_json or "").strip()

    explicit_inputs_provided = bool(summary_arg or sessions_arg)
    if explicit_inputs_provided and not (summary_arg and sessions_arg):
        return _emit(
            {
                "ok": False,
                "error": "both_summary_json_and_sessions_list_json_required",
            },
            2,
        )

    input_mode = "provided"
    guard_status_line = "provided_inputs"
    collected_temp_root: Optional[Path] = None

    try:
        if explicit_inputs_provided:
            summary_path = Path(summary_arg)
            sessions_path = Path(sessions_arg)
        else:
            input_mode = "collected"
            if not args.collect_inputs:
                return _emit(
                    {
                        "ok": False,
                        "error": "inputs_missing_enable_collect_inputs_or_provide_paths",
                    },
                    2,
                )
            guard_status_line, summary_path, sessions_path, collected_temp_root = _collect_inputs(
                openclaw_bin=str(args.openclaw_bin),
                session_active_minutes=int(args.session_active_minutes),
                guard_script=Path(str(args.guard_script)),
                expected_names=str(args.expected_names or "").strip(),
                timeout_sec=int(args.timeout_sec),
            )

        summary_obj = _normalize_json_obj(summary_path, "summary_json")
        sessions_obj = _normalize_json_obj(sessions_path, "sessions_list_json")
        _validate_summary_payload(summary_obj)
        _validate_sessions_payload(sessions_obj, args.store_path)

        from reconcile_cron_session_card_lifecycle import _extract_reconcile_rows

        requested_mode = "apply" if bool(args.apply) else "dry_run"
        promotion = _evaluate_single_run_promotion(
            requested_mode=requested_mode,
            rows=_extract_reconcile_rows(summary_obj),
            enabled=bool(args.single_run_auto_apply),
            max_candidates=int(args.single_run_max_candidates),
            min_age_sec=int(args.single_run_min_age_sec),
            apply_enabled=bool(args.single_run_apply_enabled),
        )

        reconcile_payload = _run_reconcile(
            python_bin=str(args.python_bin),
            reconcile_script=Path(str(args.reconcile_script)),
            summary_json=summary_path,
            sessions_list_json=sessions_path,
            store_paths=args.store_path,
            apply_changes=str(promotion.get("effective_mode") or requested_mode) == "apply",
            timeout_sec=int(args.timeout_sec),
        )

        return _emit(
            {
                "ok": True,
                "driver": "ops/openclaw/run_cron_session_card_lifecycle_reconcile.py",
                "input_mode": input_mode,
                "guard": {
                    "ready": guard_status_line.startswith("READY:") or guard_status_line == "provided_inputs",
                    "status_line": guard_status_line,
                },
                "inputs": {
                    "summary_json": str(summary_path),
                    "sessions_list_json": str(sessions_path),
                    "store_path_count": len(args.store_path),
                },
                "result": _summarize_reconcile_payload(reconcile_payload, promotion=promotion),
            },
            0,
        )
    except RuntimeError as exc:
        err = str(exc)
        code = 3 if err.startswith("guard_not_ready:") else 5
        return _emit({"ok": False, "error": err}, code)
    except Exception as exc:  # noqa: BLE001
        return _emit({"ok": False, "error": str(exc)}, 2)
    finally:
        if collected_temp_root and collected_temp_root.exists():
            for child in sorted(collected_temp_root.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    try:
                        child.unlink()
                    except Exception:  # noqa: BLE001
                        pass
                elif child.is_dir():
                    try:
                        child.rmdir()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                collected_temp_root.rmdir()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
