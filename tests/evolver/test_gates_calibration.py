from __future__ import annotations

from evolver.battery import BatteryScore
from evolver.calibration import CalibrationCase, CalibrationPolicy, calibrate
from evolver.gates import ActivationEvidence, CreditPolicy, PairedScore, ValidityEvidence


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
