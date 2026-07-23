"""Non-destructive skill evolution reports.

The report is the first measurement layer for Hermes' skill evolution loop. It
summarises existing ``tools.skill_usage`` telemetry and writes an auditable
markdown/JSON pair. It never edits, archives, consolidates, or otherwise mutates
skills; callers can use the report to decide what the next measured loop should
be.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools import skill_usage

_REPORT_LIMIT = 10


def _int(row: Dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _row_name(row: Dict[str, Any]) -> str:
    return str(row.get("name") or "")


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable subset of row fields useful in reports."""
    return {
        "name": _row_name(row),
        "provenance": row.get("provenance"),
        "use_count": _int(row, "use_count"),
        "view_count": _int(row, "view_count"),
        "patch_count": _int(row, "patch_count"),
        "activity_count": _int(row, "activity_count"),
        "success_count": _int(row, "success_count"),
        "failure_count": _int(row, "failure_count"),
        "last_activity_at": row.get("last_activity_at"),
        "last_patched_at": row.get("last_patched_at"),
        "last_evaluated_at": row.get("last_evaluated_at"),
        "last_evaluation": row.get("last_evaluation"),
    }


def _compact_run_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": str(row.get("run_id") or ""),
        "mission": row.get("mission") or "default",
        "final_status": row.get("final_status") or "unknown",
        "summary": row.get("summary") or "",
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "success_count": len(row.get("success_signals") or []),
        "failure_count": len(row.get("failure_signals") or []),
        "evidence_count": len(row.get("evidence_sources") or []),
        "gate_count": len(row.get("gate_results") or []),
        "decision_count": len(row.get("decisions") or []),
        "context_pack_count": len(row.get("context_packs") or []),
        "change_packet_count": len(row.get("change_packets") or []),
    }


def build_report(
    rows: Iterable[Dict[str, Any]],
    *,
    run_traces: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a deterministic report structure from skill usage rows and run traces."""
    compact = [_compact_row(r) for r in rows]
    compact = [r for r in compact if r["name"]]

    def by_usage(row: Dict[str, Any]) -> tuple:
        return (-row["use_count"], -row["activity_count"], row["name"])

    def by_patch_recency(row: Dict[str, Any]) -> tuple:
        return (row.get("last_patched_at") or "", row["name"])

    evaluated = [r for r in compact if r["last_evaluated_at"]]
    unevaluated = [r for r in compact if not r["last_evaluated_at"]]
    evaluation_gaps = sorted(
        [r for r in unevaluated if r["activity_count"] > 0 or r["patch_count"] > 0],
        key=lambda r: (-r["patch_count"], -r["activity_count"], r["name"]),
    )
    failure_heavy = [
        r for r in compact
        if r["failure_count"] > 0 and r["failure_count"] >= r["success_count"]
    ]
    review_candidates = sorted(
        {r["name"]: r for r in [*failure_heavy, *evaluation_gaps]}.values(),
        key=lambda r: (-r["failure_count"], -r["patch_count"], -r["activity_count"], r["name"]),
    )
    recent_runs = [_compact_run_trace(r) for r in (run_traces or [])]
    recent_runs = [r for r in recent_runs if r["run_id"]]
    recent_runs = sorted(
        recent_runs,
        key=lambda r: (r.get("completed_at") or r.get("started_at") or "", r["run_id"]),
        reverse=True,
    )[:_REPORT_LIMIT]

    return {
        "summary": {
            "total_skills": len(compact),
            "evaluated_skills": len(evaluated),
            "unevaluated_skills": len(unevaluated),
            "total_successes": sum(r["success_count"] for r in compact),
            "total_failures": sum(r["failure_count"] for r in compact),
        },
        "most_used": sorted(compact, key=by_usage)[:_REPORT_LIMIT],
        "recently_patched": sorted(
            [r for r in compact if r.get("last_patched_at")],
            key=by_patch_recency,
            reverse=True,
        )[:_REPORT_LIMIT],
        "evaluation_gaps": evaluation_gaps[:_REPORT_LIMIT],
        "review_candidates": review_candidates[:_REPORT_LIMIT],
        "recent_runs": recent_runs,
    }


def _table(rows: List[Dict[str, Any]], *, empty: str) -> str:
    if not rows:
        return empty
    lines = ["| Skill | Use | Activity | Patch | Success | Failure | Last evaluation |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| {name} | {use} | {activity} | {patch} | {success} | {failure} | {last_eval} |".format(
                name=row["name"],
                use=row["use_count"],
                activity=row["activity_count"],
                patch=row["patch_count"],
                success=row["success_count"],
                failure=row["failure_count"],
                last_eval=row.get("last_evaluation") or row.get("last_evaluated_at") or "—",
            )
        )
    return "\n".join(lines)


def _run_table(rows: List[Dict[str, Any]], *, empty: str) -> str:
    if not rows:
        return empty
    lines = ["| Run | Mission | Status | Success | Failure | Evidence | Gates | Decisions | Context Packs | Change Packets | Summary |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| {run_id} | {mission} | {status} | {success} | {failure} | {evidence} | {gates} | {decisions} | {context_packs} | {change_packets} | {summary} |".format(
                run_id=row["run_id"],
                mission=row.get("mission") or "default",
                status=row.get("final_status") or "unknown",
                success=row.get("success_count") or 0,
                failure=row.get("failure_count") or 0,
                evidence=row.get("evidence_count") or 0,
                gates=row.get("gate_count") or 0,
                decisions=row.get("decision_count") or 0,
                context_packs=row.get("context_pack_count") or 0,
                change_packets=row.get("change_packet_count") or 0,
                summary=row.get("summary") or "—",
            )
        )
    return "\n".join(lines)


def render_markdown(report: Dict[str, Any], *, generated_at: Optional[str] = None) -> str:
    """Render a human-readable markdown report."""
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    summary = report.get("summary") or {}
    return "\n\n".join([
        "# Hermes Evolution Report",
        f"Generated: {generated_at}",
        "This is report-only telemetry. It does not edit skills, invoke mutation tools, archive, consolidate, or change runtime behavior.",
        "## Summary\n"
        f"- Total skills: {summary.get('total_skills', 0)}\n"
        f"- Evaluated skills: {summary.get('evaluated_skills', 0)}\n"
        f"- Unevaluated skills: {summary.get('unevaluated_skills', 0)}\n"
        f"- Recorded successes: {summary.get('total_successes', 0)}\n"
        f"- Recorded failures: {summary.get('total_failures', 0)}",
        "## Most used\n" + _table(report.get("most_used") or [], empty="No skill usage telemetry yet."),
        "## Recently patched\n" + _table(report.get("recently_patched") or [], empty="No recently patched skills recorded."),
        "## Evaluation gaps\n" + _table(report.get("evaluation_gaps") or [], empty="No active skills currently lack evaluation telemetry."),
        "## Review candidates\n" + _table(report.get("review_candidates") or [], empty="No failure-heavy or unmeasured patched skills found."),
        "## Recent evolution runs\n" + _run_table(report.get("recent_runs") or [], empty="No evolution run traces recorded yet."),
        "## Recommended next loop\n"
        "1. Pick one review candidate.\n"
        "2. Define one observable improvement target.\n"
        "3. Make one focused change.\n"
        "4. Re-run a relevant task/test.\n"
        "5. Record the outcome with skill evaluation telemetry.",
    ]) + "\n"


def generate_report() -> Dict[str, Any]:
    """Write REPORT.md and run.json under HERMES_HOME/logs/evolution/."""
    generated_at = datetime.now(timezone.utc)
    base_report_id = generated_at.strftime("%Y%m%d-%H%M%S")
    report_root = get_hermes_home() / "logs" / "evolution"
    report_root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        report_id = base_report_id if suffix == 1 else f"{base_report_id}-{suffix}"
        report_dir = report_root / report_id
        try:
            report_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    try:
        from agent import evolution_trace
        run_traces = evolution_trace.list_recent_traces(limit=_REPORT_LIMIT)
    except Exception:
        run_traces = []
    report = build_report(skill_usage.usage_report(), run_traces=run_traces)
    markdown = render_markdown(report, generated_at=generated_at.isoformat())

    markdown_path = report_dir / "REPORT.md"
    json_path = report_dir / "run.json"
    payload = {
        "report_id": report_id,
        "generated_at": generated_at.isoformat(),
        **report,
    }
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    return {
        "report_id": report_id,
        "report_dir": str(report_dir),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "summary": report["summary"],
    }
