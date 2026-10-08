"""Request-local /v1/runs limits. They can only shrink the profile already resolved."""

from __future__ import annotations

from typing import Any, Optional

_LIMIT_FIELDS = {"max_turns", "run_budget_seconds", "toolsets"}
_MAX_TURNS = 10000
_MAX_BUDGET_SECONDS = 10800


def parse_shrink_only_limits(body: Any) -> dict:
    """Return the shrink fields, or {} when the request sets none.

    Unknown execution_policy keys are rejected. ``owner`` is ignored: 0.21.5
    labels a turn with ``author`` and does not grant from owner metadata.
    """
    if not isinstance(body, dict):
        return {}
    policy = body.get("execution_policy")
    if policy is None:
        return {}
    if not isinstance(policy, dict) or set(policy) - _LIMIT_FIELDS:
        raise ValueError("Unsupported execution_policy fields")
    for name, maximum in (("max_turns", _MAX_TURNS), ("run_budget_seconds", _MAX_BUDGET_SECONDS)):
        if name not in policy:
            continue
        value = policy[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Invalid execution_policy {name}")
    if "toolsets" in policy:
        toolsets = policy["toolsets"]
        if (not isinstance(toolsets, list) or len(toolsets) > 128
                or any(not isinstance(item, str) or not item or len(item) > 128 for item in toolsets)):
            raise ValueError("Invalid execution_policy toolsets")
    return {key: policy[key] for key in _LIMIT_FIELDS if key in policy}


def shrink_iterations(max_iterations: int, limits: dict) -> int:
    if "max_turns" not in limits or isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        return max_iterations
    return min(max_iterations, limits["max_turns"])


def shrink_toolsets(enabled_toolsets: list, limits: dict) -> list:
    if "toolsets" not in limits:
        return list(enabled_toolsets)
    allowed = set(limits["toolsets"])
    return sorted(set(enabled_toolsets) & allowed)


def shrink_run_budget(profile_budget: Optional[float], limits: dict) -> Optional[float]:
    """Positive profile budget wins the floor. No profile cap keeps the request value."""
    if "run_budget_seconds" not in limits:
        return None
    requested = float(limits["run_budget_seconds"])
    if profile_budget is not None and profile_budget > 0:
        return min(float(profile_budget), requested)
    return requested
