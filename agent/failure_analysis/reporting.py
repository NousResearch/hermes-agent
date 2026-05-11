"""CLI-facing formatters for failure analysis output."""

from __future__ import annotations

import time
from typing import Any


def _ts(epoch: float) -> str:
    """Format epoch timestamp as compact local time."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
    except Exception:
        return "?"


def _severity_icon(severity: str) -> str:
    icons = {"critical": "!!", "high": "!", "medium": "~", "low": "."}
    return icons.get(severity, "?")


def format_recent_failures(failures: list[dict[str, Any]]) -> str:
    """Format a list of recent failures for CLI display."""
    if not failures:
        return "  No failures recorded."
    lines = ["  Recent failures:", "  " + "-" * 72]
    for f in failures:
        sev = _severity_icon(f.get("severity", ""))
        ts = _ts(f.get("created_at", 0))
        ftype = f.get("failure_type", "?")
        fsub = f.get("failure_subtype", "?")
        summary = (f.get("summary", "") or "")[:60]
        fp = f.get("fingerprint", "")[:8]
        lines.append(f"  [{sev}] {ts}  {ftype}.{fsub}  [{fp}]  {summary}")
    return "\n".join(lines)


def format_top_failures(patterns: list[dict[str, Any]]) -> str:
    """Format top recurring failure patterns for CLI display."""
    if not patterns:
        return "  No failure patterns found in this window."
    lines = ["  Top failure patterns (7d):", "  " + "-" * 72]
    for i, p in enumerate(patterns, 1):
        count = p.get("count", 0)
        ftype = p.get("failure_type", "?")
        fsub = p.get("failure_subtype", "?")
        fp = p.get("fingerprint", "")[:12]
        summary = (p.get("summary", "") or "")[:50]
        latest = _ts(p.get("latest_at", 0))
        tool = p.get("tool_name") or ""
        tool_part = f"  tool={tool}" if tool else ""
        lines.append(
            f"  {i:>2}. [{count:>3}x] {ftype}.{fsub}  {fp}  "
            f"{summary}{tool_part}  (latest: {latest})"
        )
    return "\n".join(lines)


def format_fingerprint_detail(
    fingerprint: str, occurrences: list[dict[str, Any]]
) -> str:
    """Format detailed view of a specific failure fingerprint."""
    if not occurrences:
        return f"  No failures found for fingerprint: {fingerprint}"
    first = occurrences[0]
    lines = [
        f"  Fingerprint: {fingerprint}",
        f"  Type: {first.get('failure_type', '?')}.{first.get('failure_subtype', '?')}",
        f"  Severity: {first.get('severity', '?')}",
        f"  Occurrences: {len(occurrences)}",
        f"  Latest: {_ts(first.get('created_at', 0))}",
        f"  Summary: {first.get('summary', '')}",
    ]
    if first.get("tool_name"):
        lines.append(f"  Tool: {first['tool_name']}")
    if first.get("model"):
        lines.append(f"  Model: {first['model']}")
    lines.append("")
    lines.append("  Recent occurrences:")
    for occ in occurrences[:10]:
        ts = _ts(occ.get("created_at", 0))
        src = occ.get("source_surface", "?")
        eid = occ.get("eval_run_id") or occ.get("session_id") or ""
        lines.append(f"    {ts}  src={src}  ref={eid}")
    return "\n".join(lines)


def format_failure_summary(summary: dict[str, Any]) -> str:
    """Format the compact failure summary for CLI display."""
    lines = [
        "  Failure summary:",
        f"    Last 24h: {summary.get('total_24h', 0)} failures",
        f"    Last 7d:  {summary.get('total_7d', 0)} failures",
        f"    All time: {summary.get('total_all', 0)} failures",
    ]
    top = summary.get("top_patterns", [])
    if top:
        lines.append("")
        lines.append("  Top patterns (7d):")
        for p in top[:5]:
            count = p.get("count", 0)
            ftype = p.get("failure_type", "?")
            fsub = p.get("failure_subtype", "?")
            summary_text = (p.get("summary", "") or "")[:40]
            lines.append(f"    [{count:>3}x] {ftype}.{fsub}  {summary_text}")
    return "\n".join(lines)
