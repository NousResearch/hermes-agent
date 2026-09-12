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
    task_route = task.get("route") if "route" in task else None
    requested_route = (
        task_route if task_route is not None else config.get("default_route", "gemini")
    )
    explicit_gemini = task.get("route") == "gemini"

    def sol(reason: str) -> RouteDecision:
        if explicit_gemini:
            reason = f"Gemini route denied; falling back to Sol: {reason}"
        return RouteDecision(
            route="sol",
            reason=reason,
            eligible_for_daily_review=False,
        )

    if config.get("enabled", False) is not True:
        if config.get("enabled", False) is not False:
            return sol("invalid Gemini routing enabled flag")
        return sol("Gemini routing is disabled")

    profiles = config.get("profiles", [])
    if not isinstance(profiles, list) or not all(
        isinstance(value, str) for value in profiles
    ):
        return sol("invalid Gemini routing profiles list")
    if profile not in profiles:
        return sol(f"profile {profile!r} is not enabled for Gemini routing")

    if role == "orchestrator":
        return sol("the orchestrator role retains routing and execution authority")

    if not isinstance(requested_route, str) or requested_route not in {
        "auto",
        "gemini",
        "sol",
    }:
        return sol(f"invalid Gemini routing default route {requested_route!r}")

    if requested_route == "sol":
        return sol("Sol was explicitly selected by route policy")

    task_classification = (
        task.get("data_classification") if "data_classification" in task else None
    )
    data_classification = (
        task_classification
        if task_classification is not None
        else config.get("default_data_classification", "restricted")
    )
    if not isinstance(data_classification, str) or data_classification not in {
        "standard",
        "restricted",
    }:
        return sol(f"invalid Gemini data classification {data_classification!r}")
    if data_classification == "restricted":
        return sol("restricted data cannot use the Gemini subscription lane")

    return RouteDecision(
        route="gemini",
        reason="eligible output-only leaf delegation",
        eligible_for_daily_review=True,
    )
