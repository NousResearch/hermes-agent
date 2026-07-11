"""Deterministic Block Logic continuity-pressure evaluation.

The evaluator is pure and side-effect free. Host integrations decide whether to
log, display a notice, or create a temporary noncanonical rescue artifact.
Canonical Continuity Block writes are never authorized here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Mapping


class ThresholdLevel(str, Enum):
    NORMAL = "normal"
    QUIET = "quiet_indicator"
    RECOMMEND = "block_logic_recommended"
    PROMINENT = "prominent_warning"
    RESCUE = "temporary_rescue_snapshot"


@dataclass(frozen=True)
class ThresholdConfig:
    quiet_threshold: float = 0.70
    recommend_threshold: float = 0.82
    prominent_threshold: float = 0.90
    rescue_threshold: float = 0.95

    elapsed_hours_full: float = 24.0
    message_count_full: int = 180
    decision_count_full: int = 25
    changed_files_full: int = 20
    compression_events_full: int = 3
    model_handoffs_full: int = 4

    context_weight: float = 0.72
    elapsed_weight: float = 0.08
    messages_weight: float = 0.07
    decisions_weight: float = 0.05
    files_weight: float = 0.03
    compression_weight: float = 0.03
    handoffs_weight: float = 0.02

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ThresholdConfig":
        raw = raw if isinstance(raw, Mapping) else {}
        allowed = {field.name for field in fields(cls)}
        values = {key: raw[key] for key in allowed if key in raw}
        config = cls(**values)
        thresholds = (
            config.quiet_threshold,
            config.recommend_threshold,
            config.prominent_threshold,
            config.rescue_threshold,
        )
        if any(not 0.0 < value <= 1.0 for value in thresholds):
            raise ValueError("Block Logic thresholds must be in the range (0, 1]")
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("Block Logic thresholds must be monotonically increasing")
        return config


@dataclass(frozen=True)
class SessionMetrics:
    context_used: int
    context_limit: int
    elapsed_hours: float = 0.0
    message_count: int = 0
    decision_count: int = 0
    changed_files: int = 0
    compression_events: int = 0
    model_handoffs: int = 0

    @property
    def context_ratio(self) -> float:
        if self.context_limit <= 0:
            raise ValueError("context_limit must be greater than zero")
        if self.context_used < 0:
            raise ValueError("context_used cannot be negative")
        return min(self.context_used / self.context_limit, 1.0)


@dataclass(frozen=True)
class ThresholdDecision:
    level: ThresholdLevel
    context_ratio: float
    operational_pressure: float
    effective_pressure: float
    should_notify: bool
    should_offer_run_now: bool
    should_offer_checkpoint: bool
    should_create_temporary_rescue: bool
    canonical_write_allowed: bool
    headline: str
    message: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["reasons"] = list(self.reasons)
        return payload


def _norm(value: float, full_value: float) -> float:
    if full_value <= 0:
        raise ValueError("normalization full value must be greater than zero")
    return max(0.0, min(value / full_value, 1.0))


def operational_pressure(metrics: SessionMetrics, config: ThresholdConfig) -> float:
    pressure = (
        metrics.context_ratio * config.context_weight
        + _norm(metrics.elapsed_hours, config.elapsed_hours_full) * config.elapsed_weight
        + _norm(metrics.message_count, config.message_count_full) * config.messages_weight
        + _norm(metrics.decision_count, config.decision_count_full) * config.decisions_weight
        + _norm(metrics.changed_files, config.changed_files_full) * config.files_weight
        + _norm(metrics.compression_events, config.compression_events_full) * config.compression_weight
        + _norm(metrics.model_handoffs, config.model_handoffs_full) * config.handoffs_weight
    )
    return min(max(pressure, 0.0), 1.0)


def evaluate(
    metrics: SessionMetrics,
    config: ThresholdConfig | None = None,
) -> ThresholdDecision:
    config = config or ThresholdConfig()
    context_ratio = metrics.context_ratio
    pressure = operational_pressure(metrics, config)
    effective = max(context_ratio, pressure)

    reasons: list[str] = []
    if context_ratio >= config.quiet_threshold:
        reasons.append(f"context usage is {context_ratio:.1%}")
    if metrics.elapsed_hours >= 12:
        reasons.append(f"session has run for {metrics.elapsed_hours:.1f} hours")
    if metrics.message_count >= 100:
        reasons.append(f"message count is {metrics.message_count}")
    if metrics.decision_count >= 12:
        reasons.append(f"tracked decisions total {metrics.decision_count}")
    if metrics.changed_files >= 10:
        reasons.append(f"changed files total {metrics.changed_files}")
    if metrics.compression_events:
        reasons.append(f"compression events total {metrics.compression_events}")
    if metrics.model_handoffs >= 2:
        reasons.append(f"model handoffs total {metrics.model_handoffs}")

    power_sprint_recommend = (
        context_ratio >= 0.75
        and metrics.elapsed_hours >= 24
        and metrics.changed_files >= 10
        and (metrics.message_count >= 150 or metrics.decision_count >= 18)
    )
    if power_sprint_recommend:
        reasons.append("power-sprint continuity override triggered")

    rescue = (
        context_ratio >= config.rescue_threshold
        or (
            context_ratio >= config.prominent_threshold
            and metrics.compression_events >= 1
        )
        or (pressure >= 0.97 and metrics.compression_events >= 1)
    )

    if rescue:
        level = ThresholdLevel.RESCUE
        headline = "Block Logic rescue threshold reached"
        message = (
            "Create a temporary noncanonical rescue snapshot before further "
            "compression, then offer Block Logic or Continue."
        )
    elif effective >= config.prominent_threshold:
        level = ThresholdLevel.PROMINENT
        headline = "Block Logic strongly recommended"
        message = "This session is at high continuity risk."
    elif effective >= config.recommend_threshold or power_sprint_recommend:
        level = ThresholdLevel.RECOMMEND
        headline = "Block Logic recommended"
        message = "This session is approaching its continuity threshold."
    elif effective >= config.quiet_threshold:
        level = ThresholdLevel.QUIET
        headline = "Block Logic threshold approaching"
        message = "Continue monitoring without interrupting the user."
    else:
        level = ThresholdLevel.NORMAL
        headline = "Block Logic not needed"
        message = "Continue monitoring."
        reasons = []

    return ThresholdDecision(
        level=level,
        context_ratio=round(context_ratio, 6),
        operational_pressure=round(pressure, 6),
        effective_pressure=round(effective, 6),
        should_notify=level not in {ThresholdLevel.NORMAL, ThresholdLevel.QUIET},
        should_offer_run_now=level in {
            ThresholdLevel.RECOMMEND,
            ThresholdLevel.PROMINENT,
            ThresholdLevel.RESCUE,
        },
        should_offer_checkpoint=level in {
            ThresholdLevel.RECOMMEND,
            ThresholdLevel.PROMINENT,
            ThresholdLevel.RESCUE,
        },
        should_create_temporary_rescue=level is ThresholdLevel.RESCUE,
        canonical_write_allowed=False,
        headline=headline,
        message=message,
        reasons=tuple(reasons),
    )
