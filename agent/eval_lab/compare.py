"""Run comparison metrics for longitudinal eval-lab checks."""

from __future__ import annotations

from agent.eval_lab.schemas import EvalScore


def _avg(scores: list[EvalScore]) -> float:
    if not scores:
        return 0.0
    return round(sum(score.total for score in scores) / len(scores), 4)


def _failure_count(scores: list[EvalScore], criterion: str) -> int:
    return sum(1 for score in scores if score.criteria.get(criterion, 1.0) <= 0.0)


def compare_runs(
    baseline_scores: list[EvalScore],
    candidate_scores: list[EvalScore],
    *,
    regression_threshold: float = 0.1,
) -> dict[str, object]:
    """Compare two score lists and return deterministic longitudinal metrics."""

    baseline_by_id = {score.attempt_id: score for score in baseline_scores}
    candidate_by_id = {score.attempt_id: score for score in candidate_scores}
    regressions: list[dict[str, object]] = []
    for attempt_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id.get(attempt_id)
        if not candidate:
            continue
        delta = round(candidate.total - baseline.total, 4)
        if delta <= -abs(regression_threshold):
            regressions.append(
                {
                    "attempt_id": attempt_id,
                    "baseline": baseline.total,
                    "candidate": candidate.total,
                    "delta": delta,
                }
            )

    baseline_avg = _avg(baseline_scores)
    candidate_avg = _avg(candidate_scores)
    return {
        "baseline_avg_score": baseline_avg,
        "candidate_avg_score": candidate_avg,
        "score_delta": round(candidate_avg - baseline_avg, 4),
        "baseline_completion_count": len(baseline_scores),
        "candidate_completion_count": len(candidate_scores),
        "candidate_safety_failures": _failure_count(candidate_scores, "secret_safety"),
        "candidate_tool_discipline_failures": _failure_count(candidate_scores, "tool_use_required"),
        "regressions": sorted(regressions, key=lambda item: str(item["attempt_id"])),
    }
