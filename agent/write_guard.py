"""Scope-aware write guard utilities.

This is a deterministic preflight helper for any future write surface that needs
to compare source content scope against target audience scope.  It deliberately
keeps a tiny API so individual tools can adopt it incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


class Bucket(str, Enum):
    RESTRICTED = "restricted"
    PRIVATE = "private"
    SHARED_LIMITED = "shared_limited"
    TEAMSPACE = "teamspace"
    WORKSPACE = "workspace"
    PUBLIC = "public"
    EXTERNAL = "external"


class Confidence(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    TENTATIVE = "tentative"


RANK = {
    Bucket.RESTRICTED: 0,
    Bucket.PRIVATE: 1,
    Bucket.SHARED_LIMITED: 2,
    Bucket.TEAMSPACE: 3,
    Bucket.WORKSPACE: 4,
    Bucket.PUBLIC: 5,
    Bucket.EXTERNAL: 6,
}


@dataclass(frozen=True)
class ScopedContent:
    content_ref: str
    bucket: Bucket
    bucket_confidence: Literal["metadata", "inferred", "unknown"] = "metadata"
    labels: tuple[str, ...] = ()

    @property
    def effective_bucket(self) -> Bucket:
        if self.bucket_confidence == "unknown":
            return Bucket.RESTRICTED
        if any(label in {"credentials", "hr", "legal"} for label in self.labels):
            return Bucket.RESTRICTED
        return self.bucket


@dataclass(frozen=True)
class Target:
    content_ref: str
    bucket: Bucket
    audience_description: str = ""


@dataclass(frozen=True)
class WriteRequest:
    sources: list[ScopedContent]
    target: Target
    operation: str = "write"
    injection_signals: list[str] = field(default_factory=list)
    memory_confidences: list[Confidence] = field(default_factory=list)


@dataclass(frozen=True)
class WriteGuardDecision:
    verdict: Literal["allow", "confirm", "block"]
    source_min_bucket: Bucket
    target_bucket: Bucket
    triggered_rules: list[str]
    evaluated_at: str


def _narrowest_source_bucket(sources: list[ScopedContent]) -> Bucket:
    if not sources:
        return Bucket.RESTRICTED
    return min((source.effective_bucket for source in sources), key=lambda bucket: RANK[bucket])


def evaluate_write_guard(request: WriteRequest) -> WriteGuardDecision:
    """Return allow/confirm/block for a proposed write.

    Retrieved content is monotonic: it can only make the decision stricter
    through injection_signals; it can never loosen a scope broadening rule.
    """
    source_min = _narrowest_source_bucket(request.sources)
    target = request.target.bucket
    rules: list[str] = []

    def decision(verdict: Literal["allow", "confirm", "block"], rules_: list[str]) -> WriteGuardDecision:
        return WriteGuardDecision(
            verdict=verdict,
            source_min_bucket=source_min,
            target_bucket=target,
            triggered_rules=rules_,
            evaluated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )

    if source_min == Bucket.RESTRICTED and target in {Bucket.PUBLIC, Bucket.EXTERNAL}:
        return decision("block", [f"restricted->{target.value}"])

    if (
        RANK[target] >= RANK[Bucket.TEAMSPACE]
        and Confidence.TENTATIVE in request.memory_confidences
    ):
        return decision("block", ["tentative_memory_to_shared"])

    if RANK[source_min] < RANK[target]:
        rules.append(f"broadening:{source_min.value}->{target.value}")

    rules.extend(f"injection_signal:{signal}" for signal in request.injection_signals)

    if rules:
        return decision("confirm", rules)
    return decision("allow", [])
