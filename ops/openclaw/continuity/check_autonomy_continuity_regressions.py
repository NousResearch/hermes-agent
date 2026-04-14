#!/usr/bin/env python3
"""Unified autonomy/continuity regression cluster harness.

Purpose:
- reduce fragmented fixed-now/autonomy checks into one operator command.
- provide a deterministic summary for readiness gating and fast diagnosis.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strict_required_check_contracts import strict_required_cluster_command_map

ROOT = Path(__file__).resolve().parents[3]

SMOKE_CHECK_IDS = [
    "action_token_regressions",
    "gtc_latest_schema_failclose",
    "swarm_operability_contract_regressions",
    "slot_fill_protocol_contract_regressions",
]

CRITICAL_PATH_CHECK_IDS = [
    "gtc_latest_schema_failclose",
    "gtc_publish_manifest_auth_dual_mode",
    "gtc_incident_replay_verify_gate_posture",
    "gtc_publish_transaction_regressions",
    "queue_cooldown_authority_regressions",
    "no_nudge_reminder_runtime_hardening",
    "swarm_operability_contract_regressions",
    "slot_fill_protocol_contract_regressions",
]

BENCHMARK_BASELINE_SCHEMA = "openclaw.continuity.autonomy_regression_benchmark_baseline.v1"
BENCHMARK_SCORECARD_SCHEMA = "openclaw.continuity.autonomy_regression_benchmark_scorecard.v1"
BENCHMARK_HISTORY_ENTRY_SCHEMA = "openclaw.continuity.autonomy_regression_benchmark_history_entry.v1"
BENCHMARK_TREND_SCHEMA = "openclaw.continuity.autonomy_regression_benchmark_trend.v1"
BENCHMARK_IMPROVEMENT_MARGIN_RATIO = 0.05
DEFAULT_BENCHMARK_HISTORY_PATH = Path("state/continuity/history/autonomy_regression_critical_path_benchmark_history.jsonl")


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    description: str
    command: list[str]


def _tail(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    rows = (text or "").splitlines()
    if len(rows) > max_lines:
        rows = rows[-max_lines:]
    trimmed = "\n".join(rows)
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    return trimmed


def _extract_summary_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
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


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _resolve_optional_path(raw: str | None, *, root: Path) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path


def _scorecard_digest(scorecard: dict[str, Any]) -> str:
    canonical = json.dumps(scorecard, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append_benchmark_history(
    *,
    history_path: Path,
    scorecard: dict[str, Any],
    baseline_path: Path,
    selected_ids: list[str],
    benchmark_output_path: Path | None,
) -> dict[str, Any]:
    summary_obj = scorecard.get("summary") if isinstance(scorecard.get("summary"), dict) else {}
    hard_answer_obj = scorecard.get("hard_answer") if isinstance(scorecard.get("hard_answer"), dict) else {}

    entry = {
        "schema_version": BENCHMARK_HISTORY_ENTRY_SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_id": str(scorecard.get("suite_id") or ""),
        "profile": str(scorecard.get("profile") or ""),
        "decision": str(scorecard.get("decision") or "UNKNOWN"),
        "hard_answer": str(hard_answer_obj.get("did_this_get_better_or_worse") or "unknown"),
        "selected_check_count": int(summary_obj.get("selected_check_count") or len(selected_ids)),
        "total_duration_sec": _float_or_none(summary_obj.get("total_duration_sec")),
        "baseline_target_total_duration_sec": _float_or_none(summary_obj.get("baseline_target_total_duration_sec")),
        "baseline_max_total_duration_sec": _float_or_none(summary_obj.get("baseline_max_total_duration_sec")),
        "failed_check_count": int(summary_obj.get("failed_check_count") or 0),
        "duration_regression_check_count": int(summary_obj.get("duration_regression_check_count") or 0),
        "improved_check_count": int(summary_obj.get("improved_check_count") or 0),
        "failed_check_ids": [str(item) for item in (summary_obj.get("failed_check_ids") or [])],
        "duration_regressed_check_ids": [str(item) for item in (summary_obj.get("duration_regressed_check_ids") or [])],
        "improved_check_ids": [str(item) for item in (summary_obj.get("improved_check_ids") or [])],
        "baseline_path": str(baseline_path),
        "scorecard_output_path": None if benchmark_output_path is None else str(benchmark_output_path),
        "scorecard_schema_version": str(scorecard.get("schema_version") or ""),
        "scorecard_digest": _scorecard_digest(scorecard),
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _iter_jsonl_dict_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _summarize_benchmark_trend(
    *,
    history_path: Path,
    trend_window: int,
    suite_id: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    window = max(2, int(trend_window or 0))
    recent: deque[dict[str, Any]] = deque(maxlen=window)
    total_rows = 0
    matching_rows = 0
    suite_filter = str(suite_id or "").strip()
    profile_filter = str(profile or "").strip()

    for row in _iter_jsonl_dict_rows(history_path):
        total_rows += 1
        if suite_filter and str(row.get("suite_id") or "").strip() != suite_filter:
            continue
        if profile_filter and str(row.get("profile") or "").strip() != profile_filter:
            continue
        matching_rows += 1
        recent.append(row)

    rows = list(recent)
    decision_counts = dict(Counter(str((row.get("decision") or "UNKNOWN")) for row in rows))
    hard_answer_counts = dict(Counter(str((row.get("hard_answer") or "unknown")) for row in rows))

    latest_row = rows[-1] if rows else {}
    previous_row = rows[-2] if len(rows) >= 2 else {}
    first_row = rows[0] if rows else {}

    latest_duration = _float_or_none(latest_row.get("total_duration_sec"))
    previous_duration = _float_or_none(previous_row.get("total_duration_sec"))
    first_duration = _float_or_none(first_row.get("total_duration_sec"))

    latest_vs_previous_delta = None
    latest_vs_previous_delta_pct = None
    if latest_duration is not None and previous_duration is not None:
        latest_vs_previous_delta = round(latest_duration - previous_duration, 3)
        if previous_duration > 0:
            latest_vs_previous_delta_pct = round((latest_duration - previous_duration) / previous_duration, 4)

    window_delta = None
    window_delta_pct = None
    if latest_duration is not None and first_duration is not None:
        window_delta = round(latest_duration - first_duration, 3)
        if first_duration > 0:
            window_delta_pct = round((latest_duration - first_duration) / first_duration, 4)

    trend_status = "insufficient_history"
    trend_reason = "fewer_than_two_history_entries"
    latest_decision = str(latest_row.get("decision") or "UNKNOWN")
    if len(rows) >= 2:
        if latest_decision == "BLOCK":
            trend_status = "regressing"
            trend_reason = "latest_decision_block"
        elif latest_vs_previous_delta_pct is None:
            trend_status = "stable"
            trend_reason = "missing_duration_data"
        elif latest_vs_previous_delta_pct <= -BENCHMARK_IMPROVEMENT_MARGIN_RATIO:
            trend_status = "improving"
            trend_reason = "latest_duration_below_previous_by_margin"
        elif latest_vs_previous_delta_pct >= BENCHMARK_IMPROVEMENT_MARGIN_RATIO:
            trend_status = "regressing"
            trend_reason = "latest_duration_above_previous_by_margin"
        else:
            trend_status = "stable"
            trend_reason = "latest_duration_within_margin"

    return {
        "schema_version": BENCHMARK_TREND_SCHEMA,
        "history_path": str(history_path),
        "trend_window": window,
        "history_entry_count_total": total_rows,
        "history_entry_count_matching": matching_rows,
        "history_entry_count_used": len(rows),
        "latest_recorded_at": latest_row.get("recorded_at"),
        "latest_decision": latest_decision,
        "latest_hard_answer": latest_row.get("hard_answer"),
        "decision_counts": decision_counts,
        "hard_answer_counts": hard_answer_counts,
        "latest_total_duration_sec": None if latest_duration is None else round(latest_duration, 3),
        "previous_total_duration_sec": None if previous_duration is None else round(previous_duration, 3),
        "latest_vs_previous_duration_delta_sec": latest_vs_previous_delta,
        "latest_vs_previous_duration_delta_pct": latest_vs_previous_delta_pct,
        "window_first_total_duration_sec": None if first_duration is None else round(first_duration, 3),
        "window_duration_delta_sec": window_delta,
        "window_duration_delta_pct": window_delta_pct,
        "status": trend_status,
        "reason": trend_reason,
        "improvement_margin_ratio": BENCHMARK_IMPROVEMENT_MARGIN_RATIO,
    }


def _build_benchmark_scorecard(
    *,
    profile: str,
    selected_ids: list[str],
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_schema = str(baseline.get("schema_version") or "").strip()
    baseline_checks = baseline.get("checks") if isinstance(baseline.get("checks"), list) else []
    baseline_by_id: dict[str, dict[str, Any]] = {}
    duplicate_baseline_ids: set[str] = set()

    for row in baseline_checks:
        if not isinstance(row, dict):
            continue
        check_id = str(row.get("id") or "").strip()
        if not check_id:
            continue
        if check_id in baseline_by_id:
            duplicate_baseline_ids.add(check_id)
        baseline_by_id[check_id] = row

    selected_id_set = set(selected_ids)
    result_map = {str(row.get("id") or ""): row for row in results if isinstance(row, dict)}
    missing_results = sorted(check_id for check_id in selected_ids if check_id not in result_map)
    missing_baseline_ids = sorted(check_id for check_id in selected_ids if check_id not in baseline_by_id)

    check_rows: list[dict[str, Any]] = []
    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    failed_ids: list[str] = []

    total_duration_sec = 0.0
    baseline_target_total = 0.0
    baseline_max_total = 0.0
    baseline_target_count = 0
    baseline_max_count = 0

    for check_id in selected_ids:
        result_row = result_map.get(check_id) or {}
        baseline_row = baseline_by_id.get(check_id) or {}

        ok = bool(result_row.get("ok") is True)
        duration_sec = _float_or_none(result_row.get("duration_sec"))
        if duration_sec is None:
            duration_sec = 0.0

        baseline_target = _float_or_none(baseline_row.get("target_duration_sec"))
        baseline_max = _float_or_none(baseline_row.get("max_duration_sec"))

        total_duration_sec += duration_sec
        if baseline_target is not None:
            baseline_target_total += baseline_target
            baseline_target_count += 1
        if baseline_max is not None:
            baseline_max_total += baseline_max
            baseline_max_count += 1

        status = "pass"
        reasons: list[str] = []
        if not ok:
            status = "failed"
            reasons.append("check_failed")
            failed_ids.append(check_id)
            regressed_ids.append(check_id)
        elif baseline_max is not None and duration_sec > baseline_max:
            status = "duration_regressed"
            reasons.append("duration_exceeds_max")
            regressed_ids.append(check_id)
        else:
            if baseline_target is not None and duration_sec <= (baseline_target * (1.0 - BENCHMARK_IMPROVEMENT_MARGIN_RATIO)):
                status = "improved"
                improved_ids.append(check_id)

        check_rows.append(
            {
                "id": check_id,
                "status": status,
                "ok": ok,
                "duration_sec": round(duration_sec, 3),
                "baseline_target_duration_sec": None if baseline_target is None else round(baseline_target, 3),
                "baseline_max_duration_sec": None if baseline_max is None else round(baseline_max, 3),
                "duration_vs_target_delta_sec": None
                if baseline_target is None
                else round(duration_sec - baseline_target, 3),
                "duration_vs_max_delta_sec": None
                if baseline_max is None
                else round(duration_sec - baseline_max, 3),
                "reasons": reasons,
            }
        )

    aggregate = baseline.get("aggregate") if isinstance(baseline.get("aggregate"), dict) else {}
    aggregate_target_total = _float_or_none(aggregate.get("target_total_duration_sec"))
    aggregate_max_total = _float_or_none(aggregate.get("max_total_duration_sec"))
    if aggregate_target_total is None and baseline_target_count:
        aggregate_target_total = baseline_target_total
    if aggregate_max_total is None and baseline_max_count:
        aggregate_max_total = baseline_max_total

    aggregate_status = "pass"
    aggregate_reasons: list[str] = []
    if aggregate_max_total is not None and total_duration_sec > aggregate_max_total:
        aggregate_status = "duration_regressed"
        aggregate_reasons.append("total_duration_exceeds_max")

    baseline_integrity_errors: list[str] = []
    if baseline_schema != BENCHMARK_BASELINE_SCHEMA:
        baseline_integrity_errors.append("baseline_schema_version_invalid")
    if duplicate_baseline_ids:
        baseline_integrity_errors.append("duplicate_baseline_check_id")
    if missing_baseline_ids:
        baseline_integrity_errors.append("missing_baseline_check_id")
    if missing_results:
        baseline_integrity_errors.append("missing_result_for_selected_check")

    blocked = bool(
        baseline_integrity_errors
        or regressed_ids
        or failed_ids
        or aggregate_status != "pass"
    )
    hard_answer = "worse"
    if not blocked:
        hard_answer = "better" if improved_ids else "same"

    scorecard = {
        "schema_version": BENCHMARK_SCORECARD_SCHEMA,
        "suite_id": str(baseline.get("suite_id") or "autonomy_critical_path_v1"),
        "profile": profile,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "BLOCK" if blocked else "PASS",
        "hard_answer": {
            "did_this_get_better_or_worse": hard_answer,
            "explanation": (
                "regression detected in critical-path checks"
                if blocked
                else "at least one check beat baseline target"
                if improved_ids
                else "no regression and no measurable improvement beyond baseline targets"
            ),
        },
        "summary": {
            "selected_check_count": len(selected_ids),
            "result_count": len(results),
            "improvement_margin_ratio": BENCHMARK_IMPROVEMENT_MARGIN_RATIO,
            "failed_check_count": len(failed_ids),
            "duration_regression_check_count": len(regressed_ids) - len(failed_ids),
            "improved_check_count": len(improved_ids),
            "total_duration_sec": round(total_duration_sec, 3),
            "baseline_target_total_duration_sec": None
            if aggregate_target_total is None
            else round(aggregate_target_total, 3),
            "baseline_max_total_duration_sec": None
            if aggregate_max_total is None
            else round(aggregate_max_total, 3),
            "total_duration_vs_target_delta_sec": None
            if aggregate_target_total is None
            else round(total_duration_sec - aggregate_target_total, 3),
            "total_duration_vs_max_delta_sec": None
            if aggregate_max_total is None
            else round(total_duration_sec - aggregate_max_total, 3),
            "failed_check_ids": failed_ids,
            "duration_regressed_check_ids": [row_id for row_id in regressed_ids if row_id not in failed_ids],
            "improved_check_ids": improved_ids,
        },
        "aggregate": {
            "status": aggregate_status,
            "reasons": aggregate_reasons,
        },
        "baseline_integrity": {
            "status": "fail" if baseline_integrity_errors else "pass",
            "errors": baseline_integrity_errors,
            "missing_result_check_ids": missing_results,
            "missing_baseline_check_ids": missing_baseline_ids,
            "duplicate_baseline_check_ids": sorted(duplicate_baseline_ids),
            "unused_baseline_check_ids": sorted(check_id for check_id in baseline_by_id if check_id not in selected_id_set),
        },
        "checks": check_rows,
    }
    return scorecard


def _resolve_pytest_command(pytest_bin: str | None) -> list[str]:
    if pytest_bin:
        return [pytest_bin]

    venv_pytest = ROOT / ".venv" / "bin" / "pytest"
    if venv_pytest.exists() and os.access(venv_pytest, os.X_OK):
        return [str(venv_pytest)]

    raise FileNotFoundError(
        "repo venv pytest is required for deterministic execution: "
        f"missing executable {venv_pytest}. "
        "Install pytest in the workspace venv or pass --pytest-bin <path> explicitly."
    )


def _build_checks(*, python_bin: str, pytest_cmd: list[str]) -> list[CheckSpec]:
    continuity_dir = ROOT / "ops" / "openclaw" / "continuity"
    tests_dir = ROOT / "tests"
    strict_required_command_map = strict_required_cluster_command_map(
        python_bin=python_bin,
        continuity_dir=continuity_dir,
    )
    checks = [
        CheckSpec(
            check_id="action_token_regressions",
            description="Coherence-aware action-token guard mismatch/expiry matrix.",
            command=[python_bin, str(continuity_dir / "check_action_token_regressions.py")],
        ),
        CheckSpec(
            check_id="verify_gate_strict_autonomy",
            description="verify_then_resume strict autonomy gate + shared wrapper toggle/required-policy semantics.",
            command=pytest_cmd
            + [
                "-q",
                str(tests_dir / "test_verify_then_resume_autonomy_strict.py"),
                str(tests_dir / "test_verify_gate_strict_autonomy_toggle.py"),
                str(tests_dir / "test_verify_gate_status_command.py"),
            ],
        ),
        CheckSpec(
            check_id="gtc_latest_schema_failclose",
            description="GTC latest schema gate negative-regression fail-close semantics (strict + non-strict).",
            command=strict_required_command_map["gtc_latest_schema_failclose"],
        ),
        CheckSpec(
            check_id="gtc_publish_manifest_auth_dual_mode",
            description="Publish-manifest authenticity dual-mode regression harness (compat hmac + default ed25519 + tamper fail-close).",
            command=strict_required_command_map["gtc_publish_manifest_auth_dual_mode"],
        ),
        CheckSpec(
            check_id="gtc_incident_replay_verify_gate_posture",
            description="Replay bundle/surface verify-gate posture projection + incident-scoped evidence integrity.",
            command=strict_required_command_map["gtc_incident_replay_verify_gate_posture"],
        ),
        CheckSpec(
            check_id="gtc_publish_transaction_regressions",
            description="GTC publish transaction lock/crash/recovery fail-close semantics (contention + crash windows + journaled recovery).",
            command=strict_required_command_map["gtc_publish_transaction_regressions"],
        ),
        CheckSpec(
            check_id="swarm_operability_contract_regressions",
            description="Swarm contract/runbook operability fail-close and warning-boundary regression harness.",
            command=[python_bin, str(continuity_dir / "check_swarm_operability_regressions.py")],
        ),
        CheckSpec(
            check_id="slot_fill_protocol_contract_regressions",
            description="Subagent slot-fill spawn-before-speak protocol/runbook fail-close and warning-boundary regression harness.",
            command=strict_required_command_map["slot_fill_protocol_contract_regressions"],
        ),
        CheckSpec(
            check_id="autopilot_entrypoints_fixed_now_parity",
            description="Autopilot tick/ctl fixed-now parity across helper and fallback paths.",
            command=pytest_cmd + ["-q", str(tests_dir / "test_autopilot_entrypoints_fixed_now_parity.py")],
        ),
        CheckSpec(
            check_id="queue_cooldown_authority_regressions",
            description="Queue cooldown projection + fixed-now claim gating for autopilot retry backoff tasks.",
            command=strict_required_command_map["queue_cooldown_authority_regressions"],
        ),
        CheckSpec(
            check_id="continuity_refresh_fixed_now_authority",
            description="Continuity now/current refresh hooks and fixed-now cache authority semantics.",
            command=pytest_cmd + ["-q", str(tests_dir / "test_continuity_refresh_pipeline.py")],
        ),
        CheckSpec(
            check_id="operator_surfaces_fixed_now",
            description="Mission-control/gate/handover/history/operator surfaces fixed-now authority coverage.",
            command=pytest_cmd + ["-q", str(tests_dir / "test_operator_surfaces_fixed_now.py")],
        ),
        CheckSpec(
            check_id="mission_control_degraded_pending_signal",
            description="No-nudge degraded pending backlog signal propagation through continuity/operator surfaces.",
            command=pytest_cmd + ["-q", str(tests_dir / "test_operator_mission_control_degraded_pending_signal.py")],
        ),
        CheckSpec(
            check_id="no_nudge_reminder_runtime_hardening",
            description="No-nudge reminder runtime blocker-only contract hardening + READY/PROGRESS fail-close regression harness.",
            command=strict_required_command_map["no_nudge_reminder_runtime_hardening"],
        ),
    ]

    registry_ids = {item.check_id for item in checks}
    missing_required_ids = sorted(check_id for check_id in strict_required_command_map if check_id not in registry_ids)
    if missing_required_ids:
        raise RuntimeError(
            "required strict checks missing from autonomy cluster registry: "
            + ", ".join(missing_required_ids)
        )

    return checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified autonomy/continuity regression cluster")
    parser.add_argument("--only", action="append", default=[], help="Run only selected check_id (repeatable)")
    parser.add_argument(
        "--profile",
        choices=["full", "smoke", "critical-path"],
        default="full",
        help="Check profile selection (full=all checks, smoke=minimal high-signal subset, critical-path=strict-required benchmark subset)",
    )
    parser.add_argument("--list", action="store_true", help="List available checks and exit")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failing check")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for python-based checks")
    parser.add_argument(
        "--pytest-bin",
        default=None,
        help="Pytest executable override (default requires repo .venv/bin/pytest; no system fallback)",
    )
    parser.add_argument(
        "--benchmark-baseline",
        default=None,
        help="Optional benchmark baseline JSON fixture path. When set, emits hard better/worse verdict and blocks on regressions.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Optional path to write benchmark scorecard JSON (requires --benchmark-baseline).",
    )
    parser.add_argument(
        "--benchmark-history-path",
        default=str(DEFAULT_BENCHMARK_HISTORY_PATH),
        help="Benchmark scorecard history jsonl path (used only when --benchmark-baseline is set; set empty string to disable).",
    )
    parser.add_argument(
        "--benchmark-trend-window",
        type=int,
        default=10,
        help="Rolling history window size used for benchmark trend summary (minimum 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        pytest_cmd = _resolve_pytest_command(args.pytest_bin)
    except FileNotFoundError as exc:
        payload = {
            "ok": False,
            "error": "pytest_venv_required",
            "message": str(exc),
            "expected_pytest": str(ROOT / ".venv" / "bin" / "pytest"),
            "hint": "Install pytest into ./.venv or pass --pytest-bin <absolute-path>.",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"ERROR {payload['error']}: {payload['message']}", file=sys.stderr)
            print(payload["hint"], file=sys.stderr)
        return 2

    checks = _build_checks(python_bin=args.python_bin, pytest_cmd=pytest_cmd)
    check_map = {item.check_id: item for item in checks}

    selected_ids = [c.check_id for c in checks]
    selection_mode = "profile_full"
    if args.profile == "smoke":
        selected_ids = [check_id for check_id in SMOKE_CHECK_IDS if check_id in check_map]
        selection_mode = "profile_smoke"
    elif args.profile == "critical-path":
        selected_ids = [check_id for check_id in CRITICAL_PATH_CHECK_IDS if check_id in check_map]
        selection_mode = "profile_critical_path"

    if args.only:
        unknown = [item for item in args.only if item not in check_map]
        if unknown:
            payload = {
                "ok": False,
                "error": "unknown_check_id",
                "unknown": unknown,
                "known": sorted(check_map.keys()),
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"ERROR unknown check_id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected_ids = args.only
        selection_mode = "explicit_only"

    selected = [check_map[item] for item in selected_ids]

    if args.list:
        payload = {
            "ok": True,
            "schema_version": "openclaw.continuity.autonomy_regression_cluster.list.v1",
            "profile": args.profile,
            "selection_mode": selection_mode,
            "smoke_check_ids": SMOKE_CHECK_IDS,
            "critical_path_check_ids": CRITICAL_PATH_CHECK_IDS,
            "total": len(selected),
            "checks": [
                {
                    "id": item.check_id,
                    "description": item.description,
                    "command": item.command,
                }
                for item in selected
            ],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for row in payload["checks"]:
                print(f"- {row['id']}: {row['description']}")
                print(f"  command: {' '.join(row['command'])}")
        return 0

    results: list[dict[str, Any]] = []
    failed = 0
    for item in selected:
        start = time.monotonic()
        cp = subprocess.run(
            item.command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ},
        )
        duration = round(time.monotonic() - start, 3)
        ok = cp.returncode == 0
        if not ok:
            failed += 1

        row = {
            "id": item.check_id,
            "ok": ok,
            "returncode": cp.returncode,
            "command": item.command,
            "duration_sec": duration,
            "stdout_tail": _tail(cp.stdout or ""),
            "stderr_tail": _tail(cp.stderr or ""),
        }
        stdout_json = _extract_summary_json(cp.stdout or "")
        if isinstance(stdout_json, dict):
            row["stdout_json"] = stdout_json
        results.append(row)

        if not args.json:
            status = "PASS" if ok else "FAIL"
            print(f"{status} {item.check_id} ({duration:.3f}s)")
            if not ok:
                if row["stdout_tail"]:
                    print("--- stdout (tail) ---")
                    print(row["stdout_tail"])
                if row["stderr_tail"]:
                    print("--- stderr (tail) ---", file=sys.stderr)
                    print(row["stderr_tail"], file=sys.stderr)

        if failed and args.fail_fast:
            break

    summary = {
        "ok": failed == 0,
        "schema_version": "openclaw.continuity.autonomy_regression_cluster.v2",
        "profile": args.profile,
        "selection_mode": selection_mode,
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "selected_ids": [item.check_id for item in selected],
        "results": results,
    }

    benchmark_blocked = False
    if args.benchmark_output and not args.benchmark_baseline:
        error_payload = {
            "ok": False,
            "error": "benchmark_output_requires_baseline",
            "detail": "--benchmark-output requires --benchmark-baseline",
        }
        if args.json:
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print("ERROR: --benchmark-output requires --benchmark-baseline", file=sys.stderr)
        return 2

    if args.benchmark_baseline:
        baseline_path = Path(args.benchmark_baseline).expanduser().resolve()
        baseline_obj = _load_json_file(baseline_path)
        if baseline_obj is None:
            error_payload = {
                "ok": False,
                "error": "benchmark_baseline_unreadable",
                "baseline_path": str(baseline_path),
            }
            if args.json:
                print(json.dumps(error_payload, ensure_ascii=False, indent=2))
            else:
                print(f"ERROR unreadable benchmark baseline: {baseline_path}", file=sys.stderr)
            return 2

        scorecard = _build_benchmark_scorecard(
            profile=args.profile,
            selected_ids=[item.check_id for item in selected],
            results=results,
            baseline=baseline_obj,
        )
        summary["benchmark_baseline_path"] = str(baseline_path)
        summary["benchmark_scorecard"] = scorecard
        benchmark_blocked = str(scorecard.get("decision") or "") == "BLOCK"
        summary["ok"] = summary["ok"] and (not benchmark_blocked)

        benchmark_output_path: Path | None = None
        if args.benchmark_output:
            benchmark_output_path = Path(args.benchmark_output).expanduser().resolve()
            benchmark_output_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_output_path.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["benchmark_output_path"] = str(benchmark_output_path)

        benchmark_history_path = _resolve_optional_path(args.benchmark_history_path, root=ROOT)
        if benchmark_history_path is not None:
            history_entry = _append_benchmark_history(
                history_path=benchmark_history_path,
                scorecard=scorecard,
                baseline_path=baseline_path,
                selected_ids=[item.check_id for item in selected],
                benchmark_output_path=benchmark_output_path,
            )
            summary["benchmark_history"] = {
                "enabled": True,
                "path": str(benchmark_history_path),
                "entry_schema_version": BENCHMARK_HISTORY_ENTRY_SCHEMA,
                "appended": True,
                "entry": history_entry,
            }
            summary["benchmark_trend"] = _summarize_benchmark_trend(
                history_path=benchmark_history_path,
                trend_window=args.benchmark_trend_window,
                suite_id=str(scorecard.get("suite_id") or ""),
                profile=str(scorecard.get("profile") or ""),
            )
        else:
            summary["benchmark_history"] = {
                "enabled": False,
                "path": None,
                "entry_schema_version": BENCHMARK_HISTORY_ENTRY_SCHEMA,
                "appended": False,
            }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"SUMMARY {summary['passed']}/{summary['total']} checks passed")
        if args.benchmark_baseline:
            scorecard = summary.get("benchmark_scorecard") if isinstance(summary.get("benchmark_scorecard"), dict) else {}
            hard_answer = ((scorecard.get("hard_answer") or {}).get("did_this_get_better_or_worse") or "unknown")
            decision = scorecard.get("decision") or "UNKNOWN"
            print(f"BENCHMARK decision={decision} hard_answer={hard_answer}")
            trend_obj = summary.get("benchmark_trend") if isinstance(summary.get("benchmark_trend"), dict) else {}
            if trend_obj:
                print(
                    "BENCHMARK_TREND "
                    f"status={trend_obj.get('status') or 'unknown'} "
                    f"reason={trend_obj.get('reason') or 'unknown'}"
                )

    return 0 if (failed == 0 and not benchmark_blocked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
