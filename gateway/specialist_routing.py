"""Fail-closed natural-language routing for guarded Kanban specialists.

This module deliberately owns no Discord, database, or provider state. It
turns a bounded auxiliary-model answer into a typed decision; the caller owns
all side effects. Invalid answers always fall through to normal chat.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional


DEFAULT_SPECIALIST_PROFILES: dict[str, str] = {
    "task-orchestrator": "broad actionable work needing a plan, specialist handoffs, and final verification",
    "patch-steward": "narrow corrective patches with focused regression evidence",
    "acceptance-verifier": "acceptance evidence and release-gate verification",
    "safety-reviewer": "security, privacy, and operational boundary review",
    "data-quality-auditor": "data quality, freshness, and provenance review",
    "execution-boundary-auditor": "side-effect and execution-boundary verification",
    "dependency-health-sentinel": "dependency and development-tooling health",
    "learning-steward": "governed learning and memory maintenance",
    "ux-auditor": "operator experience and interface evidence",
    "research-scout": "read-only research and evidence gathering",
    "performance-sentinel": "performance and latency diagnostics",
}

_RESPONSE_FIELDS = frozenset({"kind", "profile", "confidence", "reason", "title"})
_MAX_REQUEST_CHARS = 4_000
_MAX_REASON_CHARS = 500
_MAX_TITLE_CHARS = 160


class RouteKind(str, Enum):
    SPECIALIST = "specialist"
    GENERAL = "general"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class SpecialistRouteDecision:
    """One fail-closed classifier decision suitable for gateway audit logs."""

    kind: RouteKind
    profile: Optional[str] = None
    confidence: Optional[float] = None
    reason: str = ""
    title: str = ""
    audit_reason: str = ""

    @property
    def dispatches(self) -> bool:
        return self.kind is RouteKind.SPECIALIST and self.profile is not None


def _general(audit_reason: str) -> SpecialistRouteDecision:
    return SpecialistRouteDecision(kind=RouteKind.GENERAL, audit_reason=audit_reason)


def _profiles(value: Optional[dict[str, str]]) -> dict[str, str]:
    return dict(value or DEFAULT_SPECIALIST_PROFILES)


def build_classifier_messages(
    request: str, *, profiles: Optional[dict[str, str]] = None
) -> list[dict[str, str]]:
    """Build a cache-independent, tool-free JSON-only auxiliary request."""
    routes = "\n".join(
        f"- {name}: {description}" for name, description in _profiles(profiles).items()
    )
    system = (
        "Classify whether the authorized user's message is a bounded task for one "
        "specialist route. Do not use tools and do not answer the request. Return exactly "
        "one JSON object with only these fields: kind, profile, confidence, reason, "
        "title. kind must be specialist, general, or clarify. Use specialist only "
        "when exactly one listed profile clearly owns an actionable task. Route broad "
        "multi-step work with no narrower owner to task-orchestrator. Keep ordinary "
        "conversation, questions, and requests lacking an actionable outcome as general. "
        "For specialist, title "
        "must be a short non-empty task title. For general or clarify "
        "set profile to null, confidence to 0, and title to an empty string. "
        "Do not infer missing scope or turn ordinary conversation into a task.\n\n"
        f"Available specialists:\n{routes}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": request[:_MAX_REQUEST_CHARS]},
    ]


def parse_specialist_response(
    raw: str,
    *,
    threshold: float = 0.80,
    fallback_title: str = "",
    profiles: Optional[dict[str, str]] = None,
) -> SpecialistRouteDecision:
    """Validate an untrusted classifier answer without repair or coercion."""
    if not isinstance(raw, str):
        return _general("invalid_classifier_output")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _general("malformed_json")
    if not isinstance(value, dict):
        return _general("invalid_classifier_output")
    if set(value) != _RESPONSE_FIELDS:
        return _general("unexpected_fields")

    kind_value = value.get("kind")
    try:
        kind = RouteKind(kind_value)
    except (TypeError, ValueError):
        return _general("invalid_kind")

    profile = value.get("profile")
    confidence = value.get("confidence")
    reason = value.get("reason")
    title = value.get("title")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > _MAX_REASON_CHARS:
        return _general("invalid_reason")
    if not isinstance(title, str) or len(title) > _MAX_TITLE_CHARS:
        return _general("invalid_title")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return _general("invalid_confidence")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return _general("invalid_confidence")

    if kind is RouteKind.SPECIALIST:
        if not isinstance(profile, str) or profile not in _profiles(profiles):
            return _general("unknown_profile")
        if not title.strip():
            title = " ".join(fallback_title.split())[:_MAX_TITLE_CHARS]
            if not title:
                return _general("invalid_title")
        if confidence < threshold:
            return _general("low_confidence")
        return SpecialistRouteDecision(
            kind=kind,
            profile=profile,
            confidence=confidence,
            reason=reason.strip(),
            title=title.strip(),
            audit_reason="specialist",
        )

    if profile is not None or confidence != 0.0 or title:
        return _general("inconsistent_non_specialist")
    return SpecialistRouteDecision(
        kind=kind,
        reason=reason.strip(),
        confidence=confidence,
        audit_reason=kind.value,
    )


ClassifierCall = Callable[[list[dict[str, str]]], Awaitable[str]]


async def classify_specialist_request(
    request: str,
    classifier: ClassifierCall,
    *,
    threshold: float = 0.80,
    timeout: float = 12.0,
    profiles: Optional[dict[str, str]] = None,
) -> SpecialistRouteDecision:
    """Run one bounded classifier call and turn every failure into fallback."""
    if not isinstance(request, str) or not request.strip():
        return _general("empty_request")
    if not callable(classifier):
        return _general("classifier_unavailable")
    try:
        pending = classifier(build_classifier_messages(request, profiles=profiles))
        if not inspect.isawaitable(pending):
            return _general("invalid_classifier_output")
        raw = await asyncio.wait_for(pending, timeout=max(0.01, float(timeout)))
    except asyncio.TimeoutError:
        return _general("classifier_timeout")
    except Exception:
        return _general("classifier_error")
    if not isinstance(raw, str):
        return _general("invalid_classifier_output")
    return parse_specialist_response(
        raw,
        threshold=threshold,
        fallback_title=request,
        profiles=profiles,
    )
