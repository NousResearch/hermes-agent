import pytest

from benchmarks.interface import RunResult
from benchmarks.statistical import compare_runs


def test_compare_runs_single_seed_reports_not_enough_runs():
    baseline = [RunResult(seed=42, overall_score=0.8)]
    experiment = [RunResult(seed=42, overall_score=0.9)]

    result = compare_runs(baseline, experiment)

    assert result.test_name == "not_enough_runs"
    assert result.significant is False
    assert result.p_value == 1.0
    assert result.improvement == pytest.approx(10.0)
