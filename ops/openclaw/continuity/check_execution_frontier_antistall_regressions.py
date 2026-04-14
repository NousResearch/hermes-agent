#!/usr/bin/env python3
"""EXB-04 execution-frontier anti-stall sustained-soak regression check.

Fail-closed validator focused on one contract:
- execution-frontier transition attempts must not silently APPLY while stalled/block conditions are active.
- latest execution_program_status dispatch context must not present idle/pending when anti-stall signals are active.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "clawd.execution_frontier_antistall_regression_check.v1"
DEFAULT_HISTORY_PATH = Path("state/continuity/history/execution_frontier_transition_attempts.jsonl")
DEFAULT_STATUS_PATH = Path("state/continuity/latest/execution_program_status.json")
STALL_SIGNAL_FIELDS = (
    "execution_supervisor_idle_no_candidate_stall_signal",
    "execution_supervisor_probe_due_now_idle_no_dispatch_candidate_signal",
    "execution_supervisor_closed_blocked_stagnation_with_clear_next_slice_signal",
    "execution_supervisor_provider_quota_lane_exhausted_signal",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        return token in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"parse_error:{exc}"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def _load_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append("history_missing")
        return rows, issues
    except Exception as exc:
        issues.append(f"history_read_error:{exc}")
        return rows, issues

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            issues.append(f"history_parse_error:line_{line_no}")
            continue
        if not isinstance(row, dict):
            issues.append(f"history_non_object_row:line_{line_no}")
            continue
        rows.append(row)

    return rows, issues


def _collect_block_reasons(row: dict[str, Any]) -> list[str]:
    values = row.get("block_reasons")
    if not isinstance(values, list):
        values = []
    result = [str(item).strip() for item in values if str(item).strip()]
    block_reason = str(row.get("block_reason") or "").strip()
    if block_reason and block_reason not in result:
        result.append(block_reason)
    return result


def _validate_attempt_rows(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    apply_count = 0
    block_count = 0

    for idx, row in enumerate(rows, start=1):
        decision = str(row.get("decision") or "").strip().upper()
        advance_applied = _as_bool(row.get("advance_applied"))
        block_reasons = _collect_block_reasons(row)
        autonomous_dispatch_eligible = row.get("autonomous_dispatch_eligible")
        next_candidate_dependency_blocked = row.get("next_candidate_dependency_blocked")
        selector_state = str(row.get("selector_state") or "").strip()

        if decision == "APPLY":
            apply_count += 1
            if not advance_applied:
                issues.append(f"attempt_{idx}:apply_without_advance_applied")
            if block_reasons:
                issues.append(f"attempt_{idx}:apply_with_block_reasons")
            if autonomous_dispatch_eligible is False:
                issues.append(f"attempt_{idx}:apply_when_autonomous_dispatch_not_eligible")
            if next_candidate_dependency_blocked is True:
                issues.append(f"attempt_{idx}:apply_with_dependency_blocked_candidate")
            if selector_state and selector_state != "ready_for_dispatch":
                issues.append(f"attempt_{idx}:apply_selector_state_not_ready_for_dispatch")
        elif decision == "BLOCK":
            block_count += 1
            if advance_applied:
                issues.append(f"attempt_{idx}:block_with_advance_applied_true")
            if not block_reasons and not str(row.get("error") or "").strip():
                issues.append(f"attempt_{idx}:block_without_reason")
        else:
            issues.append(f"attempt_{idx}:unknown_decision")

        if "stalled_detection_active" in block_reasons and decision != "BLOCK":
            issues.append(f"attempt_{idx}:stalled_detection_active_not_blocked")

    summary = {
        "attempts_total": len(rows),
        "apply_total": apply_count,
        "block_total": block_count,
        "latest_recorded_at": (rows[-1].get("recorded_at") if rows else None),
    }
    return issues, summary


def _validate_dispatch_status_surface(status_payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []

    dispatch_context = status_payload.get("dispatch_context")
    if isinstance(dispatch_context, dict):
        signal_source = dispatch_context
        dispatch_status = str(dispatch_context.get("status") or "").strip().lower() or None
        signal_surface = "dispatch_context"
    else:
        signal_source = status_payload
        dispatch_status = str(status_payload.get("dispatch_status") or "").strip().lower() or None
        signal_surface = "execution_program_status_top_level"

    active_signals = [field for field in STALL_SIGNAL_FIELDS if signal_source.get(field) is True]
    signal_surface_available = any(field in signal_source for field in STALL_SIGNAL_FIELDS)

    if active_signals and dispatch_status != "stalled":
        issues.append("dispatch_status_not_stalled_despite_active_antistall_signal")

    return issues, {
        "dispatch_status": dispatch_status,
        "active_stall_signals": active_signals,
        "signal_surface": signal_surface,
        "signal_surface_available": signal_surface_available,
    }


def _resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def run_check(*, root: Path, history_path: Path, status_path: Path, min_attempts: int, window_size: int) -> dict[str, Any]:
    issues: list[str] = []

    history_rows, history_load_issues = _load_jsonl_rows(history_path)
    issues.extend(history_load_issues)

    if len(history_rows) < min_attempts:
        issues.append("insufficient_history_attempts")

    attempt_issues, attempt_summary = _validate_attempt_rows(history_rows)
    issues.extend(attempt_issues)

    window_rows = history_rows[-window_size:] if window_size > 0 else history_rows
    window_issues, _ = _validate_attempt_rows(window_rows)
    if window_issues:
        issues.append("recent_window_regression_detected")

    status_payload, status_error = _load_json(status_path)
    status_summary: dict[str, Any] = {
        "dispatch_status": None,
        "active_stall_signals": [],
        "signal_surface": "unknown",
        "signal_surface_available": False,
    }
    if status_error is not None:
        issues.append(f"status_load_error:{status_error}")
    else:
        status_issues, status_summary = _validate_dispatch_status_surface(status_payload or {})
        issues.extend(status_issues)

    unique_issues = sorted(set(issues))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": len(unique_issues) == 0,
        "issues": unique_issues,
        "issue_count": len(unique_issues),
        "inputs": {
            "root": str(root),
            "history_path": str(history_path),
            "status_path": str(status_path),
            "min_attempts": int(min_attempts),
            "window_size": int(window_size),
        },
        "attempt_summary": {
            **attempt_summary,
            "window_attempts": len(window_rows),
            "window_issue_count": len(window_issues),
        },
        "status_surface": status_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check execution-frontier anti-stall sustained-soak regressions.")
    parser.add_argument("--root", default=None, help="OpenClaw root (defaults to OPENCLAW_ROOT or repo root)")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Path to execution-frontier attempts JSONL")
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH), help="Path to execution program status JSON")
    parser.add_argument("--min-attempts", type=int, default=5, help="Minimum required attempt rows")
    parser.add_argument("--window-size", type=int, default=5, help="Recent window size to enforce regression-free streak")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[3]
    if args.root:
        root = _resolve_path(Path.cwd(), args.root)
    else:
        root = Path(os.environ.get("OPENCLAW_ROOT", str(default_root))).resolve()

    history_path = _resolve_path(root, args.history_path)
    status_path = _resolve_path(root, args.status_path)

    summary = run_check(
        root=root,
        history_path=history_path,
        status_path=status_path,
        min_attempts=max(1, int(args.min_attempts)),
        window_size=max(1, int(args.window_size)),
    )

    if not args.json:
        status = "PASS" if summary.get("ok") else "FAIL"
        print(f"{status}: execution-frontier anti-stall regressions")
        print(
            f"attempts={summary.get('attempt_summary', {}).get('attempts_total', 0)} "
            f"window={summary.get('attempt_summary', {}).get('window_attempts', 0)} "
            f"issues={summary.get('issue_count', 0)}"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
