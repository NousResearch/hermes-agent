"""Calibrate the three credit gates against labeled human-authored patches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .battery import BatteryScore
from .gates import ActivationEvidence, CreditPolicy, PairedScore, ValidityEvidence, evaluate_gates

_LABELS = {"known_good", "known_bad"}


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    label: str
    validity: ValidityEvidence
    activation: ActivationEvidence
    scores: tuple[PairedScore, ...]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationCase":
        scores = tuple(
            PairedScore(
                task_id=item["task_id"],
                baseline=BatteryScore(**item["baseline"]),
                candidate=BatteryScore(**item["candidate"]),
            )
            for item in raw["scores"]
        )
        activation = raw["activation"]
        return cls(
            case_id=raw["case_id"],
            label=raw["label"],
            validity=ValidityEvidence(**raw["validity"]),
            activation=ActivationEvidence(
                tuple(activation["replayed_trace_ids"]), tuple(activation["activated_trace_ids"])
            ),
            scores=scores,
        )


@dataclass(frozen=True)
class CalibrationPolicy:
    credit: CreditPolicy
    min_good_acceptance: float = 0.8
    min_bad_rejection: float = 0.8

    def __post_init__(self) -> None:
        if not 0 <= self.min_good_acceptance <= 1 or not 0 <= self.min_bad_rejection <= 1:
            raise ValueError("calibration thresholds must be between zero and one")


def calibrate(cases: Iterable[CalibrationCase], policy: CalibrationPolicy) -> dict[str, Any]:
    cases = tuple(cases)
    labels = {case.label for case in cases}
    if labels != _LABELS:
        raise ValueError("calibration requires at least one known_good and one known_bad case")
    results = []
    for case in cases:
        report = evaluate_gates(case.validity, case.activation, case.scores, policy.credit)
        results.append({"case_id": case.case_id, "label": case.label, "accepted": report.passed, "gates": report.to_dict()})
    good = [row for row in results if row["label"] == "known_good"]
    bad = [row for row in results if row["label"] == "known_bad"]
    good_acceptance = sum(row["accepted"] for row in good) / len(good)
    bad_rejection = sum(not row["accepted"] for row in bad) / len(bad)
    killed = good_acceptance < policy.min_good_acceptance or bad_rejection < policy.min_bad_rejection
    return {
        "schema_version": 1,
        "battery_version": policy.credit.battery_version,
        "pre_registered_policy": {
            "credit": asdict(policy.credit),
            "min_good_acceptance": policy.min_good_acceptance,
            "min_bad_rejection": policy.min_bad_rejection,
        },
        "metrics": {
            "known_good": len(good),
            "known_bad": len(bad),
            "good_acceptance_rate": good_acceptance,
            "bad_rejection_rate": bad_rejection,
        },
        "kill_criterion_triggered": killed,
        "phase_1_allowed": not killed,
        "cases": results,
    }


def _read_cases(path: Path) -> tuple[CalibrationCase, ...]:
    with path.open(encoding="utf-8") as handle:
        return tuple(CalibrationCase.from_dict(json.loads(line)) for line in handle if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL labeled human patch evidence")
    parser.add_argument("--output", type=Path, required=True, help="Calibration report JSON")
    parser.add_argument("--battery-version", required=True, help="Frozen sealed-battery version")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pass-rate-delta", type=float, default=0.0)
    parser.add_argument("--max-cost-ratio", type=float, default=1.0)
    parser.add_argument("--min-good-acceptance", type=float, default=0.8)
    parser.add_argument("--min-bad-rejection", type=float, default=0.8)
    args = parser.parse_args(argv)
    policy = CalibrationPolicy(
        credit=CreditPolicy(
            battery_version=args.battery_version,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            min_pass_rate_delta=args.min_pass_rate_delta,
            max_cost_ratio=args.max_cost_ratio,
        ),
        min_good_acceptance=args.min_good_acceptance,
        min_bad_rejection=args.min_bad_rejection,
    )
    report = calibrate(_read_cases(args.input), policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if report["kill_criterion_triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
