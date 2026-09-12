"""Pure routing policy for output-only Gemini leaf delegations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal

from agent.gemini_routing_contract import (
    parse_exact_slack_target,
    validate_antigravity_extra_args,
)


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["gemini", "sol"]
    reason: str
    eligible_for_daily_review: bool


def enabled_routing_config_error(config: Mapping[str, Any]) -> str | None:
    """Return a fail-closed reason when an enabled Gemini route is malformed."""
    enabled = config.get("enabled", False)
    if enabled is not True:
        return None

    profiles = config.get("profiles")
    if not isinstance(profiles, list) or not all(
        isinstance(value, str) and value.strip() for value in profiles
    ):
        return "invalid Gemini routing profiles list"

    enum_fields = {
        "default_route": {"auto", "gemini", "sol"},
        "default_data_classification": {"standard", "restricted"},
        "effort": {"ultra", "max", "xhigh", "high", "medium", "low", "minimal", "none"},
    }
    for field, allowed in enum_fields.items():
        value = config.get(field)
        if not isinstance(value, str) or value not in allowed:
            return f"invalid Gemini routing {field}"

    for field in ("command", "model"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"invalid Gemini routing {field}"

    for field in ("timeout_seconds", "max_input_bytes", "max_output_bytes"):
        value = config.get(field)
        if type(value) is not int or value <= 0:
            return f"invalid Gemini routing {field}"

    if type(config.get("fallback_to_delegation_model")) is not bool:
        return "invalid Gemini routing fallback_to_delegation_model"

    receipt_db = config.get("receipt_db")
    if not isinstance(receipt_db, str) or not receipt_db.strip():
        return "invalid Gemini routing receipt_db"
    receipt_path = PurePath(receipt_db)
    if receipt_path.is_absolute() or ".." in receipt_path.parts:
        return "invalid Gemini routing receipt_db"

    try:
        validate_antigravity_extra_args(config.get("extra_args", []))
    except ValueError:
        return "invalid Gemini routing extra_args"

    retention = config.get("retention")
    if not isinstance(retention, Mapping):
        return "invalid Gemini routing retention"
    for field in ("raw_days", "aggregate_days"):
        value = retention.get(field)
        if type(value) is not int or value <= 0:
            return f"invalid Gemini routing retention.{field}"

    review = config.get("review")
    if not isinstance(review, Mapping):
        return "invalid Gemini routing review"
    if type(review.get("enabled")) is not bool:
        return "invalid Gemini routing review.enabled"
    if review.get("timezone") != "America/Los_Angeles":
        return "invalid Gemini routing review.timezone"
    if type(review.get("sample_size")) is not int or review.get("sample_size") != 5:
        return "invalid Gemini routing review.sample_size"
    if not isinstance(review.get("not_before_local"), str) or not re.fullmatch(
        r"(?:[01]\d|2[0-3]):[0-5]\d", review["not_before_local"]
    ):
        return "invalid Gemini routing review.not_before_local"
    for field in ("review_provider", "review_model", "alert_target"):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"invalid Gemini routing review.{field}"
    if review.get("review_provider") != "openai-codex":
        return "invalid Gemini routing review.review_provider"
    if review.get("review_model") != "gpt-5.6-sol":
        return "invalid Gemini routing review.review_model"
    if review.get("enabled") is True:
        try:
            parse_exact_slack_target(review["alert_target"])
        except ValueError:
            return "invalid Gemini routing review.alert_target"
        workspace_id = review.get("alert_workspace_id")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            return "invalid Gemini routing review.alert_workspace_id"
    return None


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

    config_error = enabled_routing_config_error(config)
    if config_error is not None:
        return sol(config_error)

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
