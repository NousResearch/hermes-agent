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
import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol

from gateway.capability_registry import CapabilitySignature, RegistryResolution


SPECIALIST_PROFILES: dict[str, str] = {
    "task-orchestrator": "broad actionable work needing a plan, specialist handoffs, and final verification",
    "burndown-patch-steward": "exception burndowns and narrow corrective patches",
    "acceptance-gate-verifier": "acceptance evidence and release-gate verification",
    "paper-safety-guardian": "paper-trading safety and live-boundary review",
    "market-data-authority-auditor": "market-data authority, freshness, and provenance",
    "route-execution-boundary-auditor": "route and execution-boundary verification",
    "dependency-tooling-health-sentinel": "dependency and development-tooling health",
    "copilot-learning-steward": "governed Luna copilot learning and memory",
    "mission-control-ux-auditor": "Mission Control operator UX evidence",
    "research-scout": "read-only research and evidence gathering",
    "performance-sentinel": "performance and latency diagnostics",
}

# Classifier output chooses only from ``SPECIALIST_PROFILES``. This separately
# maps each fixed profile to an explicit, read-only capability signature for
# the local registry lookup; no model-provided domain, action, or permission
# becomes part of the lookup.
_FIXED_PROFILE_CAPABILITIES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "task-orchestrator": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "burndown-patch-steward": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "acceptance-gate-verifier": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "paper-safety-guardian": (
        "financial-analysis",
        ("audit", "inspect", "read", "review", "validate"),
        "financial-analysis:read",
    ),
    "market-data-authority-auditor": (
        "market-data",
        ("audit", "inspect", "read", "review", "validate"),
        "market-data:read",
    ),
    "route-execution-boundary-auditor": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "dependency-tooling-health-sentinel": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "copilot-learning-steward": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "mission-control-ux-auditor": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
    "research-scout": (
        "research",
        ("audit", "read", "research", "review"),
        "research:read",
    ),
    "performance-sentinel": (
        "repository-evidence",
        ("audit", "inspect", "read", "review", "validate"),
        "repository-evidence:read",
    ),
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


def capability_signature_for_profile(profile: str | None) -> CapabilitySignature | None:
    """Return the fixed local lookup scope for one classifier-approved profile.

    The mapping is intentionally closed over ``SPECIALIST_PROFILES``. Unknown
    or generated profile names therefore cannot smuggle a scope, permission,
    or registry lookup through the Discord ingress.
    """
    if not isinstance(profile, str):
        return None
    capability = _FIXED_PROFILE_CAPABILITIES.get(profile)
    if capability is None:
        return None
    domain, actions, permission = capability
    return CapabilitySignature(
        domain=domain,
        actions=actions,
        evidence_class="diagnostic-only",
        requested_permissions=(permission,),
    )


def classify_explicit_burndown_patch_request(request: str) -> Optional[SpecialistRouteDecision]:
    """Route the one unambiguous exception-burndown instruction without an LLM.

    This is intentionally narrower than natural-language routing in general.
    When the local classifier is occupied by a long-running model request, an
    explicit request to perform an exception burndown *and* patch confirmed
    failures must not fall through to the general chat agent and duplicate the
    work.  Ordinary questions about exceptions, burndowns, or patches remain
    model-classified (or normal chat).
    """
    if not isinstance(request, str):
        return None
    normalized = " ".join(request.casefold().split())
    if (
        "exception" not in normalized
        or "burndown" not in normalized
        or re.search(r"\bpatch(?:es|ed|ing)?\b", normalized) is None
    ):
        return None
    return SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="burndown-patch-steward",
        confidence=1.0,
        reason="explicit exception burndown and patch request",
        title="Narrow Exception Burndown and Patching",
        audit_reason="deterministic_burndown_patch",
    )


def build_classifier_messages(request: str) -> list[dict[str, str]]:
    """Build a cache-independent, tool-free JSON-only auxiliary request."""
    routes = "\n".join(
        f"- {name}: {description}" for name, description in SPECIALIST_PROFILES.items()
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
    raw: str, *, threshold: float = 0.80, fallback_title: str = ""
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
        if not isinstance(profile, str) or profile not in SPECIALIST_PROFILES:
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


class CapabilityResolver(Protocol):
    """Minimal local registry dependency for active-profile routing."""

    def resolve(self, signature: CapabilitySignature) -> RegistryResolution:
        """Return the locally verified resolution for one exact signature."""


def _inactive_profile_decision() -> SpecialistRouteDecision:
    return _general("inactive_profile")


def _active_registry_decision(
    resolution: RegistryResolution, fallback: SpecialistRouteDecision | None
) -> SpecialistRouteDecision:
    """Build a dispatch only from a local active-resolution receipt."""
    if not isinstance(resolution.profile, str) or not resolution.profile.strip():
        return _general("invalid_active_registry_profile")
    return SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile=resolution.profile,
        confidence=fallback.confidence if fallback is not None else None,
        reason=resolution.reason,
        title=(fallback.title if fallback is not None and fallback.title else "Specialist task"),
        audit_reason="active_registry_match",
    )


def apply_registry_resolution(
    resolution: RegistryResolution, *, fallback: SpecialistRouteDecision | None = None
) -> SpecialistRouteDecision:
    """Compose one trusted local resolution with an optional classifier fallback.

    Only an ``active_match`` can select a profile outside the fixed baseline.
    A no-match or ambiguity preserves a fixed-profile classifier result so the
    handoff owner can open the inert Task-3 candidate request and use the
    existing orchestrator fallback. Candidate names and malformed registry
    output are never dispatchable.
    """
    if not isinstance(resolution, RegistryResolution):
        return _general("registry_unavailable")
    if resolution.status == "active_match":
        return _active_registry_decision(resolution, fallback)
    if resolution.status == "unavailable":
        return _general("registry_unavailable")
    if resolution.status not in {"no_match", "ambiguous"}:
        return _general("registry_unavailable")
    if fallback is None:
        return _general(f"registry_{resolution.status}")
    if not fallback.dispatches or fallback.profile not in SPECIALIST_PROFILES:
        return _inactive_profile_decision()
    return fallback


def resolve_registry(
    signature: CapabilitySignature, registry: CapabilityResolver
) -> RegistryResolution:
    """Resolve locally and turn registry faults into a typed no-dispatch result."""
    if not isinstance(signature, CapabilitySignature) or not hasattr(registry, "resolve"):
        return RegistryResolution(
            status="unavailable", profile=None, reason="local capability registry is unavailable"
        )
    try:
        resolution = registry.resolve(signature)
    except Exception:
        return RegistryResolution(
            status="unavailable", profile=None, reason="local capability registry is unavailable"
        )
    if not isinstance(resolution, RegistryResolution):
        return RegistryResolution(
            status="unavailable", profile=None, reason="local capability registry returned invalid data"
        )
    return resolution


def resolve_route(
    signature: CapabilitySignature,
    registry: CapabilityResolver,
    *,
    fallback: SpecialistRouteDecision | None = None,
) -> SpecialistRouteDecision:
    """Resolve an active specialist before using the fixed classifier baseline.

    This function owns no provider call and cannot activate profiles. It only
    composes an already-local capability lookup into a typed route decision.
    """
    return apply_registry_resolution(resolve_registry(signature, registry), fallback=fallback)


async def classify_specialist_request(
    request: str,
    classifier: ClassifierCall,
    *,
    threshold: float = 0.80,
    timeout: float = 12.0,
) -> SpecialistRouteDecision:
    """Run one bounded classifier call and turn every failure into fallback."""
    if not isinstance(request, str) or not request.strip():
        return _general("empty_request")
    explicit_burndown = classify_explicit_burndown_patch_request(request)
    if explicit_burndown is not None:
        return explicit_burndown
    if not callable(classifier):
        return _general("classifier_unavailable")
    try:
        pending = classifier(build_classifier_messages(request))
        if not inspect.isawaitable(pending):
            return _general("invalid_classifier_output")
        raw = await asyncio.wait_for(pending, timeout=max(0.01, float(timeout)))
    except asyncio.TimeoutError:
        return _general("classifier_timeout")
    except Exception:
        return _general("classifier_error")
    if not isinstance(raw, str):
        return _general("invalid_classifier_output")
    return parse_specialist_response(raw, threshold=threshold, fallback_title=request)
