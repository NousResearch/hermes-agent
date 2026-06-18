"""Source-health policy helpers for monitor-style cron jobs.

Monitor scripts often use a primary source plus weaker fallbacks.  This module
makes that state explicit so monitors can suppress delivery when a result came
only from discovery-only or unverified sources instead of accidentally reporting
weak evidence as news.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal

ConfidenceLevel = Literal["high", "medium", "low", "none"]
VerificationState = Literal["verified", "single_source", "unverified", "discovery_only"]
MONITOR_STATUS_PREFIX = "HERMES_MONITOR_STATUS:"

_CONFIDENCE_RANK: dict[ConfidenceLevel, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass(slots=True)
class SourceObservation:
    """One attempted source in a monitor run.

    `access_ok` means the source was reached and parsed. `verification` records
    whether the resulting items can support delivery. Discovery-only sources are
    useful for finding candidates, but do not independently verify a report.
    """

    name: str
    access_ok: bool | None
    verification: VerificationState = "unverified"
    confidence: ConfidenceLevel = "low"
    fallback: bool = False
    items_seen: int | None = None
    failure_reason: str | None = None
    notes: str | None = None

    @property
    def deliverable(self) -> bool:
        return (
            self.access_ok is True
            and self.verification in {"verified", "single_source"}
            and _CONFIDENCE_RANK[self.confidence] >= _CONFIDENCE_RANK["medium"]
        )

    @property
    def discovery_only(self) -> bool:
        return self.verification == "discovery_only"


@dataclass(slots=True)
class SourceHealthStatus:
    """Canonical source-health status emitted by monitor runs.

    `source_access_ok` is raw access/parsing health: true if at least one
    source was reached, even if delivery is later suppressed because that
    source is discovery-only, unverified, or low-confidence.
    """

    source_access_ok: bool | None
    fallback_used: bool
    confidence: ConfidenceLevel
    delivery_allowed: bool
    delivery_suppressed: bool
    suppression_reason: str | None = None
    failure_reason: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_monitor_status_line(self) -> str:
        return MONITOR_STATUS_PREFIX + json.dumps(self.to_json_dict(), sort_keys=True)


def evaluate_source_health(
    observations: list[SourceObservation],
    *,
    require_verified_delivery: bool = True,
) -> SourceHealthStatus:
    """Classify monitor source health and whether delivery is allowed.

    Policy:
    - `source_access_ok` is true when at least one source was reached and parsed.
    - Delivery requires at least one accessible, non-discovery source with
      `medium` or `high` confidence.
    - If all accessible sources are discovery-only or unverified, delivery is
      suppressed even when candidate items were discovered.
    - `fallback_used` is true when any fallback source was accessed after the
      monitor attempted at least one non-fallback source.
    """

    if not observations:
        return SourceHealthStatus(
            source_access_ok=False,
            fallback_used=False,
            confidence="none",
            delivery_allowed=False,
            delivery_suppressed=True,
            suppression_reason="no_sources_attempted",
            failure_reason="monitor did not report any attempted sources",
            sources=[],
        )

    sources = [asdict(obs) for obs in observations]
    accessible = [obs for obs in observations if obs.access_ok is True]
    attempted_primary = any(not obs.fallback for obs in observations)
    fallback_used = attempted_primary and any(obs.fallback and obs.access_ok is True for obs in observations)

    deliverable = [obs for obs in accessible if obs.deliverable]
    best_confidence: ConfidenceLevel = "none"
    for obs in accessible:
        if _CONFIDENCE_RANK[obs.confidence] > _CONFIDENCE_RANK[best_confidence]:
            best_confidence = obs.confidence

    if deliverable or not require_verified_delivery:
        return SourceHealthStatus(
            source_access_ok=True,
            fallback_used=fallback_used,
            confidence=best_confidence,
            delivery_allowed=True,
            delivery_suppressed=False,
            sources=sources,
        )

    if not accessible:
        failures = [obs.failure_reason for obs in observations if obs.failure_reason]
        failure_reason = "; ".join(failures) if failures else "no source was accessible"
        return SourceHealthStatus(
            source_access_ok=False,
            fallback_used=fallback_used,
            confidence="none",
            delivery_allowed=False,
            delivery_suppressed=True,
            suppression_reason="source_access_failed",
            failure_reason=failure_reason,
            sources=sources,
        )

    if all(obs.discovery_only for obs in accessible):
        suppression_reason = "discovery_only_source"
    elif all(obs.verification == "unverified" for obs in accessible):
        suppression_reason = "unverified_source"
    else:
        suppression_reason = "low_confidence_source"

    return SourceHealthStatus(
        source_access_ok=True,
        fallback_used=fallback_used,
        confidence=best_confidence,
        delivery_allowed=False,
        delivery_suppressed=True,
        suppression_reason=suppression_reason,
        failure_reason="accessible sources did not meet verified delivery policy",
        sources=sources,
    )


def monitor_status_line(observations: list[SourceObservation], *, require_verified_delivery: bool = True) -> str:
    """Return a `HERMES_MONITOR_STATUS:{...}` line for script output."""

    return evaluate_source_health(
        observations,
        require_verified_delivery=require_verified_delivery,
    ).to_monitor_status_line()
