"""Validity, activation, and statistically calibrated credit gates."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean

from .battery import BatteryScore


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    metrics: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class ValidityEvidence:
    patch_applies: bool
    tests_pass: bool
    typecheck_pass: bool


@dataclass(frozen=True)
class ActivationEvidence:
    replayed_trace_ids: tuple[str, ...]
    activated_trace_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedScore:
    task_id: str
    baseline: BatteryScore
    candidate: BatteryScore


@dataclass(frozen=True)
class CreditPolicy:
    battery_version: str
    confidence: float = 0.95
    bootstrap_samples: int = 10_000
    seed: int = 0
    min_pass_rate_delta: float = 0.0
    max_cost_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not self.battery_version:
            raise ValueError("battery_version must identify a frozen battery")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.max_cost_ratio <= 0:
            raise ValueError("max_cost_ratio must be positive")


@dataclass(frozen=True)
class GateReport:
    validity: GateDecision
    activation: GateDecision
    credit: GateDecision
    policy: CreditPolicy

    @property
    def passed(self) -> bool:
        return self.validity.passed and self.activation.passed and self.credit.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "validity": asdict(self.validity),
            "activation": asdict(self.activation),
            "credit": asdict(self.credit),
            "policy": asdict(self.policy),
        }


def evaluate_validity(evidence: ValidityEvidence) -> GateDecision:
    checks = asdict(evidence)
    failed = [name for name, passed in checks.items() if not passed]
    return GateDecision(not failed, "all validity checks passed" if not failed else f"failed: {', '.join(failed)}", checks)


def evaluate_activation(evidence: ActivationEvidence) -> GateDecision:
    replayed = set(evidence.replayed_trace_ids)
    activated = set(evidence.activated_trace_ids)
    unknown = activated - replayed
    passed = bool(activated) and not unknown
    reason = "candidate activated in a replayed failing trace" if passed else (
        "activation evidence references traces that were not replayed" if unknown else "candidate never activated"
    )
    return GateDecision(passed, reason, {"replayed": len(replayed), "activated": len(activated), "unknown": len(unknown)})


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.floor(probability * len(ordered))))
    return ordered[index]


def evaluate_credit(pairs: tuple[PairedScore, ...], policy: CreditPolicy) -> GateDecision:
    if not pairs:
        return GateDecision(False, "sealed battery returned no paired scores", {"pairs": 0})
    task_ids = [pair.task_id for pair in pairs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("paired battery task IDs must be unique")
    deltas = [float(pair.candidate.passed) - float(pair.baseline.passed) for pair in pairs]
    rng = random.Random(policy.seed)
    draws = [mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(policy.bootstrap_samples)]
    alpha = (1.0 - policy.confidence) / 2.0
    lower = _percentile(draws, alpha)
    upper = _percentile(draws, 1.0 - alpha)
    both_passed = [pair for pair in pairs if pair.baseline.passed and pair.candidate.passed]
    baseline_cost = sum(pair.baseline.cost for pair in both_passed)
    candidate_cost = sum(pair.candidate.cost for pair in both_passed)
    cost_ratio = candidate_cost / baseline_cost if baseline_cost else (0.0 if not candidate_cost else math.inf)
    effect_passed = lower > policy.min_pass_rate_delta
    cost_passed = cost_ratio <= policy.max_cost_ratio
    passed = effect_passed and cost_passed
    reason = "paired credit interval and cost bound passed" if passed else (
        "paired confidence interval did not clear the pre-registered effect" if not effect_passed else "candidate exceeded the pre-registered cost bound"
    )
    return GateDecision(
        passed,
        reason,
        {
            "pairs": len(pairs),
            "mean_pass_rate_delta": mean(deltas),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence": policy.confidence,
            "cost_ratio_on_joint_passes": cost_ratio,
        },
    )


def evaluate_gates(
    validity: ValidityEvidence,
    activation: ActivationEvidence,
    pairs: tuple[PairedScore, ...],
    policy: CreditPolicy,
) -> GateReport:
    validity_result = evaluate_validity(validity)
    activation_result = evaluate_activation(activation) if validity_result.passed else GateDecision(False, "validity gate failed", {})
    credit_result = evaluate_credit(pairs, policy) if activation_result.passed else GateDecision(False, "activation gate failed", {})
    return GateReport(validity_result, activation_result, credit_result, policy)
