"""Pure routing policy for output-only Gemini leaf delegations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["gemini", "sol"]
    reason: str
    eligible_for_daily_review: bool


def decide_delegation_route(
    *, task: Mapping[str, Any], role: str, profile: str, config: Mapping[str, Any]
) -> RouteDecision:
    """Choose Gemini for eligible leaf work unless a hard exclusion applies."""
    requested_route = task.get("route") or config.get("default_route", "gemini")
    explicit_gemini = task.get("route") == "gemini"

    def sol(reason: str) -> RouteDecision:
        if explicit_gemini:
            reason = f"Gemini route denied; falling back to Sol: {reason}"
        return RouteDecision(
            route="sol",
            reason=reason,
            eligible_for_daily_review=False,
        )

    if not config.get("enabled", False):
        return sol("Gemini routing is disabled")

    profiles = config.get("profiles", ())
    if profile not in profiles:
        return sol(f"profile {profile!r} is not enabled for Gemini routing")

    if role == "orchestrator":
        return sol("the orchestrator role retains routing and execution authority")

    if requested_route == "sol":
        return sol("Sol was explicitly selected by route policy")

    data_classification = task.get("data_classification") or config.get(
        "default_data_classification", "restricted"
    )
    if data_classification == "restricted":
        return sol("restricted data cannot use the Gemini subscription lane")

    if requested_route not in {"auto", "gemini"}:
        return sol(f"unsupported route {requested_route!r}")

    return RouteDecision(
        route="gemini",
        reason="eligible output-only leaf delegation",
        eligible_for_daily_review=True,
    )
