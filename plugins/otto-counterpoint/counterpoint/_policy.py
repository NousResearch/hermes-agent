# -*- coding: utf-8 -*-
"""Route eligibility and counterpoint trigger policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

_VALID_REASONING = frozenset({"high", "xhigh", "ultra"})
_VALID_RISKS = frozenset({"low", "medium", "high", "critical"})
_VALID_COMPLEXITIES = frozenset({"simple", "moderate", "complex", "frontier"})


class RoutePolicyError(ValueError):
    """A route cannot satisfy the workflow's independence policy."""


class NoIndependentRoute(RoutePolicyError):
    """No authenticated, accessible route met the requested independence gate."""

    def __init__(self, role: str, excluded_families: Sequence[str]) -> None:
        self.role = role
        self.excluded_families = tuple(sorted(set(excluded_families)))
        self.requires_human_review = True
        super().__init__(
            f"no independent {role} route after excluding families: "
            f"{', '.join(self.excluded_families) or 'none'}"
        )


@dataclass(frozen=True)
class RouteIdentity:
    """An exact, already-resolved model route used by one workflow role."""

    vendor: str
    family: str
    provider: str
    model: str
    reasoning_effort: str
    authenticated: bool = True
    accessible: bool = True
    smoke_tested: bool = True
    relative_load: float = 1.0

    def __post_init__(self) -> None:
        for field in ("vendor", "family", "provider", "model", "reasoning_effort"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise RoutePolicyError(f"{field} must be a non-empty string")
        if self.reasoning_effort not in _VALID_REASONING:
            raise RoutePolicyError(
                f"reasoning_effort must be one of {sorted(_VALID_REASONING)}"
            )
        for field in ("authenticated", "accessible", "smoke_tested"):
            if not isinstance(getattr(self, field), bool):
                raise RoutePolicyError(f"{field} must be a boolean")
        if isinstance(self.relative_load, bool) or not isinstance(self.relative_load, (int, float)):
            raise RoutePolicyError("relative_load must be numeric")
        if self.relative_load < 0:
            raise RoutePolicyError("relative_load must be non-negative")

    @property
    def route_key(self) -> str:
        return f"{self.provider}/{self.model}@{self.reasoning_effort}"


@dataclass(frozen=True)
class CounterpointDecision:
    """Policy output deciding whether a second model must be called."""

    required: bool
    reason: str


def decide_counterpoint(
    *,
    risk: str,
    complexity: str,
    external_effect: bool = False,
    evidence_conflict: bool = False,
    uncertainty: bool = False,
    sampling_selected: bool = False,
) -> CounterpointDecision:
    """Apply the fail-closed trigger policy without calling a model."""
    normalized_risk = str(risk).lower()
    normalized_complexity = str(complexity).lower()
    if normalized_risk not in _VALID_RISKS:
        raise RoutePolicyError(f"unknown risk: {risk!r}")
    if normalized_complexity not in _VALID_COMPLEXITIES:
        raise RoutePolicyError(f"unknown complexity: {complexity!r}")
    if normalized_risk in {"high", "critical"}:
        return CounterpointDecision(True, "high_risk")
    if external_effect:
        return CounterpointDecision(True, "external_effect")
    if evidence_conflict:
        return CounterpointDecision(True, "evidence_conflict")
    if uncertainty or normalized_complexity == "frontier":
        return CounterpointDecision(True, "material_uncertainty")
    if sampling_selected:
        return CounterpointDecision(True, "audit_sampling")
    return CounterpointDecision(False, "deterministic_checks_sufficient")


def eligible_routes(
    routes: Iterable[RouteIdentity],
    *,
    role: str,
    generator_route: RouteIdentity | None = None,
    counterpoint_route: RouteIdentity | None = None,
    require_independence: bool = True,
    allowed_families: Sequence[str] = (),
    excluded_families: Sequence[str] = (),
    require_smoke_tested: bool = True,
) -> list[RouteIdentity]:
    """Return routes allowed for a role without silently falling back.

    Independence is based on the resolved model family, not merely the provider
    string. Thus ``openai-codex`` and ``openai-api`` remain correlated when both
    serve OpenAI models. If a critic or judge has no independent route, the
    function raises instead of returning the generator route.
    """
    if role not in {"generator", "counterpoint", "adjudicator"}:
        raise RoutePolicyError(f"unknown workflow role: {role!r}")

    excluded = {str(item) for item in excluded_families if str(item).strip()}
    if require_independence and role in {"counterpoint", "adjudicator"}:
        if generator_route is not None:
            excluded.add(generator_route.family)
        if role == "adjudicator" and counterpoint_route is not None:
            excluded.add(counterpoint_route.family)

    allowed = {str(item) for item in allowed_families if str(item).strip()}
    output: list[RouteIdentity] = []
    for route in routes:
        if not isinstance(route, RouteIdentity):
            raise RoutePolicyError("routes must contain RouteIdentity values")
        if not route.authenticated or not route.accessible:
            continue
        if require_smoke_tested and not route.smoke_tested:
            continue
        if route.family in excluded:
            continue
        if allowed and route.family not in allowed:
            continue
        output.append(route)

    output.sort(key=lambda item: (item.relative_load, item.provider, item.model))
    if require_independence and role in {"counterpoint", "adjudicator"} and not output:
        raise NoIndependentRoute(role, tuple(sorted(excluded)))
    return output


__all__ = [
    "CounterpointDecision",
    "NoIndependentRoute",
    "RouteIdentity",
    "RoutePolicyError",
    "decide_counterpoint",
    "eligible_routes",
]
