"""Markdown reporting for eval-lab runs."""

from __future__ import annotations

from pathlib import Path

from agent.eval_lab.redaction import redact_secrets
from agent.eval_lab.schemas import EvalScore, TrajectoryGroup


def _line(text: object) -> str:
    return str(redact_secrets(str(text)))


def render_markdown_report(
    run_id: str,
    groups: list[TrajectoryGroup],
    scores: list[EvalScore],
    artifact_paths: list[str],
) -> str:
    """Render a safe markdown report for an eval run."""
    scores_by_attempt = {score.attempt_id: score for score in scores}
    lines = [f"# Hermes Eval Lab Run: {_line(run_id)}", ""]
    lines.append("## Artefakti")
    if artifact_paths:
        for path in artifact_paths:
            lines.append(f"- `{_line(path)}`")
    else:
        lines.append("- Nema zapisanih artefakata.")
    lines.append("")
    lines.append("## Scenariji i pokušaji")
    for group in groups:
        lines.append(f"### Scenario: {_line(group.scenario_id)}")
        lines.append(f"- Group: `{_line(group.group_id)}`")
        for attempt in group.attempts:
            score = scores_by_attempt.get(attempt.attempt_id)
            total = f"{score.total:.4f}" if score else "n/a"
            lines.append(f"- Attempt `{_line(attempt.attempt_id)}`: status={_line(attempt.status)}, score={total}")
            if attempt.final_response:
                lines.append(f"  - Final: {_line(attempt.final_response)}")
            if score and score.notes:
                lines.append(f"  - Notes: {_line('; '.join(score.notes))}")
        lines.append("")
    lines.append("## Rangiranje")
    for rank, score in enumerate(sorted(scores, key=lambda item: (-item.total, item.attempt_id)), start=1):
        lines.append(f"{rank}. `{_line(score.attempt_id)}` — {score.total:.4f}")
    if not scores:
        lines.append("Nema score podataka.")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    output_path: str | Path,
    run_id: str,
    groups: list[TrajectoryGroup],
    scores: list[EvalScore],
    artifact_paths: list[str],
) -> Path:
    """Write a redacted markdown report and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(run_id=run_id, groups=groups, scores=scores, artifact_paths=artifact_paths),
        encoding="utf-8",
    )
    return path
