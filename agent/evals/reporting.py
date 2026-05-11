"""Compact reporting helpers for eval runs."""

from __future__ import annotations

import time
from typing import Any, Optional

from .types import CaseStatus, RunSummary


def format_run_summary(summary: RunSummary) -> str:
    """Format a run summary as a compact terminal-friendly string."""
    lines = [
        f"Eval run: {summary.run_id}  suite={summary.suite_name}",
        f"  cases: {summary.case_count}  passed: {summary.passed_count}  "
        f"failed: {summary.failed_count}  avg_score: {summary.avg_score:.2f}",
    ]
    if summary.label:
        lines[0] += f"  label={summary.label}"

    # Per-case detail
    for cr in summary.case_results:
        status_str = cr.status.value if isinstance(cr.status, CaseStatus) else cr.status
        icon = "✓" if status_str == "passed" else "✗"
        line = f"  {icon} {cr.case_id}  score={cr.total_score:.2f}  {cr.duration_ms}ms"
        if cr.failure_summary:
            line += f"  — {cr.failure_summary[:80]}"
        lines.append(line)

    return "\n".join(lines)


def format_run_row(run: dict[str, Any]) -> str:
    """Format a single run dict (from storage) as one summary line."""
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.get("created_at", 0)))
    passed = run.get("passed_count", 0)
    total = run.get("case_count", 0)
    score = run.get("avg_score", 0.0)
    suite = run.get("suite_name", "?")
    label = run.get("label", "")
    label_part = f"  [{label}]" if label else ""
    return f"{run['id']}  {ts}  {suite}  {passed}/{total} passed  avg={score:.2f}{label_part}"


def format_recent_runs(runs: list[dict[str, Any]]) -> str:
    """Format a list of recent runs for terminal display."""
    if not runs:
        return "No eval runs found."
    lines = ["Recent eval runs:", ""]
    for run in runs:
        lines.append("  " + format_run_row(run))
    return "\n".join(lines)


def format_run_detail(run: dict[str, Any]) -> str:
    """Format a run with nested case results for detailed display."""
    lines = [
        f"Eval run: {run['id']}",
        f"  suite: {run.get('suite_name', '?')}",
        f"  created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.get('created_at', 0)))}",
        f"  cases: {run.get('case_count', 0)}  passed: {run.get('passed_count', 0)}  "
        f"failed: {run.get('failed_count', 0)}",
        f"  avg_score: {run.get('avg_score', 0.0):.2f}",
    ]
    if run.get("label"):
        lines.append(f"  label: {run['label']}")

    case_results = run.get("case_results", [])
    if case_results:
        lines.append("")
        lines.append("  Case results:")
        for cr in case_results:
            status = cr.get("status", "?")
            icon = "✓" if status == "passed" else "✗"
            line = (
                f"    {icon} {cr.get('case_id', '?')}  "
                f"score={cr.get('total_score', 0.0):.2f}  "
                f"{cr.get('duration_ms', 0)}ms"
            )
            fail = cr.get("failure_summary", "")
            if fail:
                line += f"  — {fail[:80]}"
            lines.append(line)

    return "\n".join(lines)
