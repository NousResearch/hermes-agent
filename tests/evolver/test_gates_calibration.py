from __future__ import annotations

import pytest

from evolver.battery import BatteryScore
from evolver.calibration import CalibrationCase, CalibrationPolicy, calibrate
from evolver.gates import ActivationEvidence, CreditPolicy, PairedScore, ValidityEvidence, evaluate_credit


def _scores(candidate_passed: bool) -> tuple[PairedScore, ...]:
    return tuple(
        PairedScore(
            task_id=f"sealed-{index}",
            baseline=BatteryScore(False, 2.0),
            candidate=BatteryScore(candidate_passed, 1.0),
        )
        for index in range(8)
    )


def test_calibration_separates_good_and_bad_patches_and_enforces_kill_criterion():
    policy = CalibrationPolicy(
        CreditPolicy("2026-Q3", bootstrap_samples=500, seed=4, max_cost_ratio=1.0),
        min_good_acceptance=1.0,
        min_bad_rejection=1.0,
    )
    good = CalibrationCase(
        "merged-fix",
        "known_good",
        ValidityEvidence(True, True, True),
        ActivationEvidence(("trace-a",), ("trace-a",)),
        _scores(True),
    )
    bad = CalibrationCase(
        "rejected-fix",
        "known_bad",
        ValidityEvidence(False, True, True),
        ActivationEvidence(("trace-b",), ("trace-b",)),
        _scores(True),
    )

    separated = calibrate((good, bad), policy)
    assert separated["metrics"] == {
        "known_good": 1,
        "known_bad": 1,
        "good_acceptance_rate": 1.0,
        "bad_rejection_rate": 1.0,
    }
    assert separated["phase_1_allowed"] is True
    assert separated["kill_criterion_triggered"] is False
    assert separated["cases"][1]["gates"]["credit"]["reason"] == "activation gate failed"

    miscalibrated = calibrate((good, CalibrationCase(
        "bad-that-slipped-through",
        "known_bad",
        ValidityEvidence(True, True, True),
        ActivationEvidence(("trace-c",), ("trace-c",)),
        _scores(True),
    )), policy)
    assert miscalibrated["kill_criterion_triggered"] is True
    assert miscalibrated["phase_1_allowed"] is False


def test_credit_gate_charges_candidate_cost_for_newly_solved_tasks():
    pairs = tuple(
        PairedScore(
            task_id=f"improved-{index}",
            baseline=BatteryScore(False, 1.0),
            candidate=BatteryScore(True, 1_000_000.0),
        )
        for index in range(8)
    )

    decision = evaluate_credit(
        pairs,
        CreditPolicy("2026-Q3", bootstrap_samples=500, seed=4, max_cost_ratio=1.0),
    )

    assert decision.metrics["ci_lower"] == 1.0
    assert decision.metrics["cost_ratio_all_pairs"] == 1_000_000.0
    assert decision.passed is False
    assert decision.reason == "candidate exceeded the pre-registered cost bound"


@pytest.mark.parametrize(
    ("passed", "cost"),
    ((1, 1.0), (True, True), (True, float("nan")), (True, float("inf")), (True, -1.0)),
)
def test_battery_score_rejects_malformed_evidence(passed, cost):
    with pytest.raises(ValueError):
        BatteryScore(passed, cost)


@pytest.mark.parametrize(
    "validity",
    (
        {"patch_applies": "false", "tests_pass": True, "typecheck_pass": True},
        {"patch_applies": True, "tests_pass": 1, "typecheck_pass": True},
        {"patch_applies": True, "tests_pass": True, "typecheck_pass": 0},
    ),
)
def test_calibration_rejects_non_boolean_validity_evidence(validity):
    raw = {
        "case_id": "bad-validity",
        "label": "known_bad",
        "validity": validity,
        "activation": {"replayed_trace_ids": [], "activated_trace_ids": []},
        "scores": [],
    }

    with pytest.raises(ValueError, match="validity evidence fields must be booleans"):
        CalibrationCase.from_dict(raw)


def test_calibration_rejects_malformed_battery_score():
    raw = {
        "case_id": "bad-evidence",
        "label": "known_bad",
        "validity": {"patch_applies": True, "tests_pass": True, "typecheck_pass": True},
        "activation": {"replayed_trace_ids": [], "activated_trace_ids": []},
        "scores": [{"task_id": "sealed-1", "baseline": {"passed": False, "cost": 1.0}, "candidate": {"passed": True, "cost": float("nan")}}],
    }

    with pytest.raises(ValueError):
        CalibrationCase.from_dict(raw)
