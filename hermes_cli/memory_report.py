from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home

REPORT_LOGDIR = get_hermes_home() / "logs" / "memory-maintenance"
TIMER_UNIT = "hermes-memory-maintenance-report.timer"
TIMEZONE = ZoneInfo("America/Sao_Paulo")


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_summary_path(logdir: Path = REPORT_LOGDIR) -> Path | None:
    candidates = [path for path in logdir.glob("summary-*.txt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _parse_summary(summary_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        text = summary_path.read_text(encoding="utf-8")
    except OSError:
        return data

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key in {"doctor_exit_code", "maintain_exit_code", "maintain_actions_proposed"}:
            parsed = _parse_int(value)
            if parsed is not None:
                data[key] = parsed
            continue
        data[key] = value
    return data


def _report_date_from_summary(summary_path: Path, summary_data: dict[str, Any]) -> str:
    stem = summary_path.stem
    match = re.fullmatch(r"summary-(\d{4}-\d{2}-\d{2})", stem)
    if match:
        return match.group(1)

    timestamp = summary_data.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return summary_path.stem.replace("summary-", "")


def _best_next_action(maintain_report: dict[str, Any]) -> str:
    actions = maintain_report.get("actions", []) if isinstance(maintain_report, dict) else []
    if not isinstance(actions, list):
        actions = []
    best = next((action for action in actions if action.get("severity") == "high"), None)
    if best is None:
        best = next((action for action in actions if action.get("proposed_action") != "NOOP"), None)
    if best:
        return f"Review {best.get('scope')} proposal for {best.get('target_id')} ({best.get('proposed_action')})"
    return "No changes needed"


def _format_usec(value: Any) -> str | None:
    if value in (None, "", "n/a"):
        return None
    try:
        ts = int(str(value).strip()) / 1_000_000
    except (TypeError, ValueError, AttributeError):
        return None
    try:
        return datetime.fromtimestamp(ts, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    except (OverflowError, OSError, ValueError):
        return None


def _collect_timer_status() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                TIMER_UNIT,
                "-p",
                "ActiveState",
                "-p",
                "UnitFileState",
                "-p",
                "SubState",
                "-p",
                "NextElapseUSecRealtime",
                "-p",
                "LastTriggerUSec",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"status": "unknown", "available": False, "error": str(exc)}

    fields: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    if not fields and result.returncode != 0:
        return {"status": "unknown", "available": False, "error": f"systemctl exit {result.returncode}"}

    active_state = fields.get("ActiveState", "unknown") or "unknown"
    unit_file_state = fields.get("UnitFileState", "unknown") or "unknown"
    sub_state = fields.get("SubState", "unknown") or "unknown"
    next_run = _format_usec(fields.get("NextElapseUSecRealtime"))
    last_trigger = _format_usec(fields.get("LastTriggerUSec"))
    return {
        "status": active_state,
        "available": True,
        "active_state": active_state,
        "unit_file_state": unit_file_state,
        "sub_state": sub_state,
        "next_run": next_run,
        "last_trigger": last_trigger,
    }


def build_memory_report_latest(logdir: Path = REPORT_LOGDIR) -> dict[str, Any]:
    summary_path = _latest_summary_path(logdir)
    if summary_path is None:
        raise FileNotFoundError(f"no memory maintenance summaries found in {logdir}")

    summary_data = _parse_summary(summary_path)
    doctor_report_path_text = str(summary_data.get("doctor_report", "")).strip()
    maintain_report_path_text = str(summary_data.get("maintain_report", "")).strip()
    doctor_path = Path(doctor_report_path_text) if doctor_report_path_text else None
    maintain_path = Path(maintain_report_path_text) if maintain_report_path_text else None
    doctor_report = _load_json(doctor_path) if doctor_path and doctor_path.exists() else {}
    maintain_report = _load_json(maintain_path) if maintain_path and maintain_path.exists() else {}
    maintain_actions = maintain_report.get("actions", []) if isinstance(maintain_report, dict) else []
    if not isinstance(maintain_actions, list):
        maintain_actions = []

    proposal_counts = {}
    summary_counts = maintain_report.get("summary", {}).get("proposal_counts") if isinstance(maintain_report, dict) else None
    if isinstance(summary_counts, dict) and summary_counts:
        proposal_counts = dict(summary_counts)
    else:
        for action in maintain_actions:
            key = action.get("proposed_action") or action.get("action") or "UNKNOWN"
            proposal_counts[key] = proposal_counts.get(key, 0) + 1

    overall_status = (
        maintain_report.get("overall_status")
        if isinstance(maintain_report, dict)
        else None
    ) or (
        doctor_report.get("overall_status")
        if isinstance(doctor_report, dict)
        else None
    ) or summary_data.get("maintain_overall_status") or summary_data.get("doctor_overall_status") or "unknown"

    return {
        "mode": "latest",
        "report_date": _report_date_from_summary(summary_path, summary_data),
        "overall_status": overall_status,
        "doctor_exit_code": summary_data.get("doctor_exit_code", "unknown"),
        "maintain_exit_code": summary_data.get("maintain_exit_code", "unknown"),
        "maintain_actions_proposed": summary_data.get("maintain_actions_proposed", len(maintain_actions)),
        "best_next_action": _best_next_action(maintain_report),
        "paths": {
            "summary": str(summary_path),
            "doctor_json": str(doctor_path) if doctor_path else "",
            "maintain_json": str(maintain_path) if maintain_path else "",
        },
        "timer": _collect_timer_status(),
        "summary": {
            "doctor_overall_status": summary_data.get("doctor_overall_status", "unknown"),
            "maintain_overall_status": summary_data.get("maintain_overall_status", "unknown"),
            "maintain_actions_proposed": summary_data.get("maintain_actions_proposed", len(maintain_actions)),
            "proposal_counts": proposal_counts,
        },
        "warnings": ["No changes applied"],
    }


def format_memory_report_latest(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Hermes memory report latest")
    lines.append(f"Report date: {report.get('report_date', 'unknown')}")
    lines.append(f"Overall status: {report.get('overall_status', 'unknown')}")
    lines.append("")

    lines.append("Execution")
    lines.append(f"  doctor_exit_code: {report.get('doctor_exit_code', 'unknown')}")
    lines.append(f"  maintain_exit_code: {report.get('maintain_exit_code', 'unknown')}")
    lines.append(f"  maintain_actions_proposed: {report.get('maintain_actions_proposed', 0)}")
    lines.append(f"  best_next_action: {report.get('best_next_action', 'unknown')}")
    lines.append("")

    paths = report.get("paths", {}) if isinstance(report.get("paths", {}), dict) else {}
    lines.append("Paths")
    lines.append(f"  summary: {paths.get('summary', 'unknown')}")
    lines.append(f"  doctor_json: {paths.get('doctor_json', 'unknown')}")
    lines.append(f"  maintain_json: {paths.get('maintain_json', 'unknown')}")
    lines.append("")

    timer = report.get("timer", {}) if isinstance(report.get("timer", {}), dict) else {}
    lines.append("Timer")
    lines.append(f"  status: {timer.get('status', 'unknown')}")
    if timer.get("unit_file_state"):
        lines.append(f"  unit_file_state: {timer.get('unit_file_state')}")
    if timer.get("sub_state"):
        lines.append(f"  sub_state: {timer.get('sub_state')}")
    if timer.get("next_run"):
        lines.append(f"  next_run: {timer.get('next_run')}")
    if timer.get("last_trigger"):
        lines.append(f"  last_trigger: {timer.get('last_trigger')}")
    lines.append("")

    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    lines.append("Summary")
    lines.append(f"  doctor_overall_status: {summary.get('doctor_overall_status', 'unknown')}")
    lines.append(f"  maintain_overall_status: {summary.get('maintain_overall_status', 'unknown')}")
    lines.append(f"  proposal_counts: {summary.get('proposal_counts', {})}")
    lines.append("")

    lines.append("No changes applied.")
    return "\n".join(lines)


def cmd_memory_report_latest(args: argparse.Namespace) -> dict[str, Any]:
    report = build_memory_report_latest()
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_memory_report_latest(report))
    return report


def cmd_memory_report(args: argparse.Namespace) -> dict[str, Any]:
    sub = getattr(args, "report_command", None)
    if sub in (None, "latest"):
        return cmd_memory_report_latest(args)
    raise ValueError(f"unsupported memory report subcommand: {sub}")
