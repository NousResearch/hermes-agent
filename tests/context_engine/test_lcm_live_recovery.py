from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lcm_live_recovery as harness


def _trial(index: int, *, correct: bool) -> harness.TrialRecord:
    return harness.TrialRecord(
        prompt_id=f"exact-{index:03d}",
        arm="exact",
        buried_fact=f"LCM-SENTINEL-{index:03d}",
        tool_calls=[{"name": "lcm_grep", "arguments": {"query": f"LCM-SENTINEL-{index:03d}"}}],
        answer=f"LCM-SENTINEL-{index:03d}" if correct else "wrong answer",
        correct=correct,
        confidence_wrong=False,
        spend=harness.SpendRecord(
            estimated_prompt_tokens=int("3000"),
            estimated_completion_tokens=int("128"),
            observed_prompt_tokens=int("2900"),
            observed_completion_tokens=int("64"),
            estimated_cost_usd=float("0.01"),
            observed_cost_usd=float("0.008"),
        ),
    )


def test_dry_run_records_required_trial_fields_and_writes_report(tmp_path: Path) -> None:
    out_path = tmp_path / "lcm-live-recovery.md"

    run = harness.run_recovery_gate(
        mode="dry-run",
        n=int("12"),
        out_path=out_path,
        seed=int("4242"),
        budget=harness.BudgetPolicy(max_usd=float("10")),
    )

    assert run.gate.passed, "\n".join(run.gate.failures)
    assert len(run.trials) == int("12")
    first = run.trials[0]
    assert first.prompt_id
    assert first.buried_fact
    assert first.tool_calls
    assert first.answer
    assert first.correct is True
    assert first.confidence_wrong is False
    assert first.spend.estimated_prompt_tokens > 0
    assert first.spend.observed_cost_usd >= 0

    report = out_path.read_text(encoding="utf-8")
    assert "Wilson 95% lower bound" in report
    assert "Arithmetic" in report
    assert "estimated spend" in report
    assert "observed spend" in report
    assert "| prompt_id | buried_fact | tool_calls | answer | correct | confidence_wrong |" in report


def test_wilson_lower_bound_is_binding_when_point_recall_passes() -> None:
    total = int("180")
    successes = int("171")
    trials = [_trial(i, correct=i < successes) for i in range(total)]

    gate = harness.evaluate_gate(
        trials,
        thresholds=harness.GateThresholds(
            min_trials=total,
            recall_point_min=float("0.95"),
            wilson_lower_min=float("0.95"),
        ),
        mode="live",
        semantic_calibration=harness.JudgeCalibrationResult(
            precision=float("1"),
            recall=float("1"),
            passed=True,
            details="unit fixture",
        ),
    )

    assert gate.point_recall == pytest.approx(float("0.95"))
    assert gate.wilson_lower < float("0.95")
    assert not gate.passed
    assert any("Wilson lower bound" in failure for failure in gate.failures)


def test_semantic_judge_calibration_fails_before_driver_is_called(tmp_path: Path) -> None:
    class CountingDriver(harness.RecoveryDriver):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run_trial(self, fixture: harness.TrialFixture, sampling: harness.SamplingParams) -> harness.DriverResponse:
            self.calls.append(fixture.prompt_id)
            return harness.DriverResponse(answer=fixture.expected_answer, tool_calls=[{"name": "lcm_grep"}])

    class AlwaysPositiveJudge:
        def contains_expected(self, answer: str, expected: list[str]) -> bool:
            return True

    driver = CountingDriver()

    run = harness.run_recovery_gate(
        mode="dry-run",
        n=int("12"),
        out_path=tmp_path / "failed-calibration.md",
        driver=driver,
        judge=AlwaysPositiveJudge(),
    )

    assert not run.gate.passed
    assert run.trials == []
    assert driver.calls == []
    assert any("judge calibration" in failure for failure in run.gate.failures)


def test_budget_preflight_aborts_before_crossing_cap(tmp_path: Path) -> None:
    class CountingDriver(harness.RecoveryDriver):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run_trial(self, fixture: harness.TrialFixture, sampling: harness.SamplingParams) -> harness.DriverResponse:
            self.calls.append(fixture.prompt_id)
            return harness.DriverResponse(answer=fixture.expected_answer, tool_calls=[{"name": "lcm_grep"}])

    driver = CountingDriver()

    run = harness.run_recovery_gate(
        mode="dry-run",
        n=int("12"),
        out_path=tmp_path / "budget-abort.md",
        budget=harness.BudgetPolicy(
            max_usd=float("0.000001"),
            estimated_input_usd_per_mtok=float("500"),
            estimated_output_usd_per_mtok=float("500"),
            max_completion_tokens=int("4096"),
        ),
        driver=driver,
    )

    assert not run.gate.passed
    assert run.aborted_reason is not None
    assert "budget cap" in run.aborted_reason
    assert driver.calls == []
    assert any("budget cap" in failure for failure in run.gate.failures)


def test_live_mode_requires_phase_two_minimum_sample_size() -> None:
    with pytest.raises(ValueError, match="N=180 minimum"):
        harness.validate_run_config(mode="live", n=int("179"))

    harness.validate_run_config(mode="live", n=int("180"))


def test_default_gate_thresholds_match_prd6_spec():
    """PRD-6: N=180, point recall >=0.95, Wilson 95% lower bound >=0.90 (binding),
    judge precision/recall >=0.95. Lock the spec numbers as the shipped defaults so a
    future edit can't silently re-tighten Wilson to 0.95 (unreachable at N=180) or
    loosen the point/judge gates."""
    gt = harness.GateThresholds()
    assert gt.min_trials == 180
    assert gt.recall_point_min == pytest.approx(0.95)
    assert gt.wilson_lower_min == pytest.approx(0.90)
    assert gt.judge_precision_min == pytest.approx(0.95)
    assert gt.judge_recall_min == pytest.approx(0.95)
    assert gt.require_tool_call_evidence is True
