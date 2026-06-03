from agent.eval_lab.compare import compare_runs
from agent.eval_lab.schemas import EvalScore


def test_compare_runs_reports_regressions_and_metric_deltas():
    baseline = [
        EvalScore(attempt_id="s1-a1", total=1.0, criteria={"secret_safety": 1.0, "tool_use_required": 1.0}),
        EvalScore(attempt_id="s2-a1", total=0.8, criteria={"secret_safety": 1.0, "tool_use_required": 0.0}),
    ]
    candidate = [
        EvalScore(attempt_id="s1-a1", total=0.5, criteria={"secret_safety": 0.0, "tool_use_required": 1.0}),
        EvalScore(attempt_id="s2-a1", total=0.9, criteria={"secret_safety": 1.0, "tool_use_required": 1.0}),
    ]

    result = compare_runs(baseline, candidate, regression_threshold=0.2)

    assert result["baseline_avg_score"] == 0.9
    assert result["candidate_avg_score"] == 0.7
    assert result["score_delta"] == -0.2
    assert result["baseline_completion_count"] == 2
    assert result["candidate_completion_count"] == 2
    assert result["candidate_safety_failures"] == 1
    assert result["candidate_tool_discipline_failures"] == 0
    assert result["regressions"] == [{"attempt_id": "s1-a1", "baseline": 1.0, "candidate": 0.5, "delta": -0.5}]
