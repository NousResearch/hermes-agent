"""Markdown rendering helpers for research artifacts."""

from __future__ import annotations

from typing import Any

from research.models import ResearchProject


def render_iteration_report(
    project: ResearchProject,
    status_summary: dict[str, Any],
    summary: str,
    next_steps: list[str] | None = None,
    run_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Render an iteration report as markdown."""
    next_steps = next_steps or []
    run_rows = run_rows or []
    lines = [
        f"# Iteration Report: {project.name}",
        "",
        f"- Project ID: `{project.project_id}`",
        f"- Phase: `{project.phase}`",
        f"- Spec level: `{project.spec_level}`",
        "",
        "## Summary",
        "",
        summary.strip() or "No summary provided.",
        "",
        "## State Snapshot",
        "",
        f"- Ideas recorded: {status_summary.get('ideas', 0)}",
        f"- Literature notes: {status_summary.get('literature_notes', 0)}",
        f"- Hypotheses: {status_summary.get('hypotheses', 0)}",
        f"- Experiments: {status_summary.get('experiments', 0)}",
        f"- Active runs: {', '.join(status_summary.get('active_run_ids', [])) or 'none'}",
        "",
    ]

    if run_rows:
        lines.extend(["## Runs", ""])
        for run in run_rows:
            lines.append(
                f"- `{run.get('run_id')}`: {run.get('status', 'unknown')} | "
                f"{run.get('method', 'n/a')} | {run.get('base_model', 'n/a')}"
            )
        lines.append("")

    lines.extend(["## Next Steps", ""])
    if next_steps:
        for step in next_steps:
            lines.append(f"- {step}")
    else:
        lines.append("- No next steps recorded.")

    return "\n".join(lines).rstrip() + "\n"


def render_comparison_table(
    experiments: list[dict[str, Any]],
    eval_results: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> str:
    """Render a markdown table comparing experiments and their eval results."""
    if not experiments:
        return "No experiments to compare.\n"

    # Gather all metric names if not specified
    if not metrics:
        metric_set: set[str] = set()
        for result in eval_results:
            metric_set.update(result.get("metrics", {}).keys())
        metrics = sorted(metric_set) if metric_set else ["(no metrics)"]

    header = "| Experiment | Benchmark | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 2)) + "|"
    rows: list[str] = []

    exp_map = {e.get("experiment_id", ""): e for e in experiments}
    for result in eval_results:
        exp_id = result.get("experiment_id", result.get("run_id", ""))
        exp = exp_map.get(exp_id, {})
        label = exp.get("title", exp_id)
        benchmark = result.get("benchmark", "")
        values = []
        for m in metrics:
            val = result.get("metrics", {}).get(m)
            values.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val or "—"))
        rows.append(f"| {label} | {benchmark} | " + " | ".join(values) + " |")

    if not rows:
        return "No evaluation results to compare.\n"

    return "\n".join([header, separator, *rows]) + "\n"


def render_eval_summary(eval_results: list[dict[str, Any]], benchmark: str | None = None) -> str:
    """Render an evaluation summary report as markdown."""
    filtered = eval_results
    if benchmark:
        filtered = [r for r in filtered if r.get("benchmark") == benchmark]
    if not filtered:
        return f"No evaluation results{' for ' + benchmark if benchmark else ''}.\n"

    lines = [f"# Evaluation Summary{': ' + benchmark if benchmark else ''}", ""]
    for result in filtered:
        lines.append(f"## {result.get('eval_id', 'unknown')} — {result.get('benchmark', 'n/a')}")
        lines.append(f"- Status: `{result.get('status', 'unknown')}`")
        lines.append(f"- Run ID: `{result.get('run_id', 'n/a')}`")
        lines.append(f"- Few-shot: {result.get('num_fewshot', 0)}")
        m = result.get("metrics", {})
        if m:
            lines.append("- Metrics:")
            for key, val in m.items():
                lines.append(f"  - {key}: {val}")
        if result.get("error"):
            lines.append(f"- Error: {result['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_session_changelog(
    project: ResearchProject,
    checkpoint: dict[str, Any],
    changes: dict[str, Any],
) -> str:
    """Render a changelog entry for a session checkpoint."""
    lines = [
        f"# Session Changelog: {project.name}",
        "",
        f"- Checkpoint: `{checkpoint.get('checkpoint_id', 'n/a')}`",
        f"- Reason: {checkpoint.get('reason', 'checkpoint')}",
        "",
        "## Changes this session",
        "",
    ]
    for key, delta in changes.items():
        if isinstance(delta, dict):
            added = delta.get("added", 0)
            removed = delta.get("removed", 0)
            if added or removed:
                lines.append(f"- **{key}**: +{added} / -{removed}")
        else:
            lines.append(f"- **{key}**: {delta}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_failure_report(
    run: dict[str, Any],
    diagnosis: dict[str, Any],
    recovery_options: list[str],
) -> str:
    """Render a failure diagnosis report as markdown."""
    lines = [
        f"# Failure Report: {run.get('run_id', 'unknown')}",
        "",
        f"- Status: `{run.get('status', 'unknown')}`",
        f"- Method: `{run.get('method', 'n/a')}`",
        f"- Base model: `{run.get('base_model', 'n/a')}`",
        "",
        "## Diagnosis",
        "",
        f"- Error type: {diagnosis.get('error_type', 'unknown')}",
        f"- Pattern: {diagnosis.get('pattern', 'none detected')}",
        f"- Message: {diagnosis.get('message', '')}",
        "",
        "## Recovery Options",
        "",
    ]
    for idx, option in enumerate(recovery_options, 1):
        lines.append(f"{idx}. {option}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_inbox_markdown(title: str, summary: str, details: str = "", next_steps: list[str] | None = None) -> str:
    """Render an inbox summary as markdown."""
    next_steps = next_steps or []
    lines = [f"# {title}", "", summary.strip()]
    if details.strip():
        lines.extend(["", "## Details", "", details.strip()])
    if next_steps:
        lines.extend(["", "## Next Steps", ""])
        for step in next_steps:
            lines.append(f"- {step}")
    return "\n".join(lines).rstrip() + "\n"


def render_literature_brief(
    literature_summary: dict[str, Any],
    top_k: int = 5,
) -> str:
    """Render a concise literature triage summary."""
    ranked = list(literature_summary.get("ranked_papers") or [])[:top_k]
    gaps = list(literature_summary.get("gap_candidates") or [])[: min(top_k, 3)]
    focus_areas = list(literature_summary.get("focus_areas") or [])
    lines = ["## Literature Brief", ""]
    if focus_areas:
        lines.append(f"- Focus areas: {', '.join(focus_areas)}")
    if ranked:
        lines.extend(["", "### Top Papers", ""])
        for paper in ranked:
            lines.append(
                f"- `{paper.get('paper_id', 'unknown')}` {paper.get('title', 'Untitled')} "
                f"(score={paper.get('score', 0):.3f}, label={paper.get('label', 'related')})"
            )
    if gaps:
        lines.extend(["", "### Gap Candidates", ""])
        for paper in gaps:
            lines.append(f"- {paper.get('title', 'Untitled')} ({paper.get('year', 'n/a')})")
    return "\n".join(lines).rstrip() + "\n"


def render_dataset_quality_report(dataset_reports: list[dict[str, Any]]) -> str:
    """Render dataset provenance and quality summaries."""
    lines = ["## Dataset Quality", ""]
    if not dataset_reports:
        lines.append("- No dataset assessments recorded.")
        return "\n".join(lines).rstrip() + "\n"
    for report in dataset_reports:
        lines.append(
            f"- `{report.get('name') or report.get('dataset_id') or report.get('path', 'dataset')}`: "
            f"quality={report.get('quality_score', 0):.1f} | rows={report.get('num_rows', 0)} | "
            f"schema_valid_ratio={report.get('schema_valid_ratio', 0):.2f} | duplicates={report.get('duplicate_rows', 0)}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_research_memo(
    project: ResearchProject,
    status_summary: dict[str, Any],
    decision: dict[str, Any],
    ranked_runs: list[dict[str, Any]] | None = None,
    literature_summary: dict[str, Any] | None = None,
    dataset_reports: list[dict[str, Any]] | None = None,
    recovery_notes: list[dict[str, Any]] | None = None,
) -> str:
    """Render a decision memo for the current research state."""
    ranked_runs = ranked_runs or []
    literature_summary = literature_summary or {}
    dataset_reports = dataset_reports or []
    recovery_notes = recovery_notes or []
    lines = [
        f"# Research Memo: {project.name}",
        "",
        f"- Project ID: `{project.project_id}`",
        f"- Phase: `{project.phase}`",
        f"- Spec level: `{project.spec_level}`",
        f"- Recommended next action: `{decision.get('next_action', 'continue')}`",
        "",
        "## Executive Summary",
        "",
        decision.get("summary", "No decision summary available."),
        "",
        "## Decision Rationale",
        "",
    ]
    reasons = list(decision.get("reasons") or [])
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- No rationale recorded.")

    lines.extend([
        "",
        "## Project Snapshot",
        "",
        f"- Ideas: {status_summary.get('ideas', 0)}",
        f"- Literature notes: {status_summary.get('literature_notes', 0)}",
        f"- Hypotheses: {status_summary.get('hypotheses', 0)}",
        f"- Experiments: {status_summary.get('experiments', 0)}",
        f"- Runs: {status_summary.get('run_count', 0)}",
        f"- Evaluations: {status_summary.get('evaluations', 0)}",
        f"- Datasets: {status_summary.get('datasets', 0)}",
        "",
    ])

    if literature_summary:
        lines.append(render_literature_brief(literature_summary).rstrip())
        lines.append("")

    if ranked_runs:
        lines.extend(["## Ranked Runs", ""])
        for row in ranked_runs[:5]:
            regressions = row.get("regressions") or []
            regression_suffix = f" | regressions={len(regressions)}" if regressions else ""
            lines.append(
                f"- `{row.get('run_id', 'unknown')}` score={row.get('score', 0):.4f} "
                f"status={row.get('status', 'unknown')} benchmark_count={row.get('eval_count', 0)}{regression_suffix}"
            )
        lines.append("")

    lines.append(render_dataset_quality_report(dataset_reports).rstrip())
    lines.append("")

    if recovery_notes:
        lines.extend(["## Recovery Notes", ""])
        for note in recovery_notes:
            lines.append(
                f"- `{note.get('run_id', 'unknown')}` -> {note.get('recommended_action', 'review')}: "
                f"{note.get('summary', '')}"
            )
        lines.append("")

    lines.extend(["## Next Steps", ""])
    next_steps = list(decision.get("next_steps") or [])
    if next_steps:
        for step in next_steps:
            lines.append(f"- {step}")
    else:
        lines.append("- No next steps recorded.")
    return "\n".join(lines).rstrip() + "\n"
