"""Markdown report rendering and publish-summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_run_report(
    *,
    project: dict[str, Any],
    family: dict[str, Any],
    run: dict[str, Any],
    output_path: Path,
) -> str:
    """Render and write one interesting-run report."""
    champion = run.get("selector", {}).get("champion") or {}
    kept = run.get("selector", {}).get("kept_candidates") or []
    anchors = run.get("anchors") or []
    lines = [
        f"# AutoResearch Run: {project['project_id']} / {family['family_id']}",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Created: `{run['created_at']}`",
        f"- Project root: `{run['project_root']}`",
        f"- Thesis: {family.get('thesis') or 'n/a'}",
        "",
        "## Summary",
        "",
    ]

    if champion:
        lines.extend(
            [
                f"- Champion: `{champion.get('candidate_id')}`",
                f"- Parent: `{champion.get('parent_candidate_id') or 'n/a'}`",
                f"- Primary metric (`{run['selection']['primary_metric']}`): `{champion.get('primary_metric')}`",
                f"- Improvement vs parent: `{champion.get('primary_delta')}`",
            ]
        )
    else:
        lines.append("- No champion selected.")

    lines.extend(
        [
            "",
            "## Anchors",
            "",
        ]
    )
    for anchor in anchors:
        lines.append(
            f"- `{anchor['candidate_id']}`: primary metric `{anchor.get('primary_metric')}`"
        )

    lines.extend(
        [
            "",
            "## Shortlisted Candidates",
            "",
        ]
    )
    if kept:
        for candidate in kept:
            lines.append(
                f"- `{candidate['candidate_id']}`: parent `{candidate.get('parent_candidate_id') or 'n/a'}`, primary `{candidate.get('primary_metric')}`, delta `{candidate.get('primary_delta')}`"
            )
    else:
        lines.append("- No shortlisted candidates survived selector checks.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Run metadata: `{Path(run['run_path']) / 'run.json'}`",
        ]
    )
    if champion and champion.get("result_json"):
        lines.append(f"- Champion result JSON: `{champion['result_json']}`")
    if champion and champion.get("workspace_path"):
        lines.append(f"- Champion workspace: `{champion['workspace_path']}`")

    lines.extend(
        [
            "",
            "## Interestingness",
            "",
            f"- Verdict: `{run.get('interesting', {}).get('verdict')}`",
            f"- Reason: {run.get('interesting', {}).get('reason') or 'n/a'}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def build_publish_summary(run: dict[str, Any]) -> str:
    """Build a short messaging-friendly summary for a completed run."""
    selector = run.get("selector", {})
    champion = selector.get("champion")
    base = f"AutoResearch {run.get('project_id')}/{run.get('family_id')} run `{run.get('run_id')}`"
    if not champion:
        return f"{base}: no champion selected."

    report_path = run.get("report_path")
    message = (
        f"{base}: champion `{champion['candidate_id']}` "
        f"hit `{champion.get('primary_metric')}` on `{run['selection']['primary_metric']}` "
        f"(delta `{champion.get('primary_delta')}` vs parent `{champion.get('parent_candidate_id') or 'n/a'}`)."
    )
    if report_path:
        message += f" Report: `{report_path}`."
    return message
