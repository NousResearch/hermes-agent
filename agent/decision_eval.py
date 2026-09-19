"""Replay evaluation for typed decision families and candidate providers."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from agent.decision_provider import DecisionLabel, DecisionQuestion, DecisionStatus
from agent.decision_runtime import DecisionRuntime


@dataclass(frozen=True)
class ReplayCase:
    task: str
    state: Mapping[str, Any]
    questions: Mapping[str, DecisionQuestion]
    expected: Mapping[str, DecisionLabel]


@dataclass(frozen=True)
class ReplayMetrics:
    examples: int
    coverage: float
    top1_error: Optional[float]
    brier_score: Optional[float]
    log_loss: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    usage: Mapping[str, float]


def replay_decisions(
    runtime: DecisionRuntime,
    corpus: Sequence[ReplayCase],
    providers: Sequence[str],
    *,
    timeout: float = 30.0,
    acceptance_threshold: float = 0.0,
) -> Dict[str, Dict[str, ReplayMetrics]]:
    """Replay labeled cases and score every provider independently per decision family."""
    if not 0 <= acceptance_threshold <= 1:
        raise ValueError("acceptance_threshold must be between 0 and 1")
    grouped: Dict[str, list[ReplayCase]] = {}
    for case in corpus:
        if set(case.expected) != set(case.questions):
            raise ValueError("replay expected labels must exactly match question keys")
        for key, expected in case.expected.items():
            if expected not in case.questions[key].labels:
                raise ValueError(f"expected label for {key!r} is outside the question domain")
        grouped.setdefault(case.task, []).append(case)

    reports: Dict[str, Dict[str, ReplayMetrics]] = {}
    for provider in providers:
        reports[provider] = {}
        for task, cases in grouped.items():
            reports[provider][task] = _score_family(
                runtime, provider, cases, timeout=timeout, acceptance_threshold=acceptance_threshold,
            )
    return reports


def _score_family(
    runtime: DecisionRuntime,
    provider: str,
    cases: Sequence[ReplayCase],
    *,
    timeout: float,
    acceptance_threshold: float,
) -> ReplayMetrics:
    total = sum(len(case.questions) for case in cases)
    accepted = 0
    wrong = 0
    brier_total = 0.0
    log_total = 0.0
    latencies: list[float] = []
    usage: Dict[str, float] = {}

    for case in cases:
        response = runtime.evaluate(
            task=case.task,
            state=case.state,
            questions=case.questions,
            provider=provider,
            mode="replay",
            timeout=timeout,
        )
        latencies.append(response.latency_ms)
        for key, value in response.usage.items():
            usage[key] = usage.get(key, 0.0) + float(value)
        if response.status not in {DecisionStatus.AVAILABLE, DecisionStatus.ABSTAINED}:
            continue
        for key, answer in response.answers.items():
            if answer.abstained or answer.selected is None:
                continue
            selected_probability = float(answer.probabilities[answer.selected])
            if selected_probability < acceptance_threshold:
                continue
            expected = case.expected[key]
            if expected not in answer.probabilities:
                raise ValueError(f"expected label for {key!r} is outside the question domain")
            accepted += 1
            wrong += int(answer.selected != expected)
            brier_total += sum(
                (float(probability) - float(label == expected)) ** 2
                for label, probability in answer.probabilities.items()
            )
            log_total += -math.log(max(float(answer.probabilities[expected]), 1e-15))

    return ReplayMetrics(
        examples=total,
        coverage=accepted / total if total else 0.0,
        top1_error=wrong / accepted if accepted else None,
        brier_score=brier_total / accepted if accepted else None,
        log_loss=log_total / accepted if accepted else None,
        latency_p50_ms=statistics.median(latencies) if latencies else None,
        latency_p95_ms=_percentile(latencies, 0.95) if latencies else None,
        usage=usage,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
