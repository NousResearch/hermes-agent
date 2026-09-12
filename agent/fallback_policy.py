"""Fallback-entry request policy helpers.

Fallback entries may opt out of the primary runtime's premium service tier
without mutating the session-wide request overrides.  The active entry policy
is applied to a deep copy immediately before transport kwargs are built.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

_SUPPORTED_SERVICE_TIER_OVERRIDES = {"normal"}
_TIER_FIELDS = ("service_tier", "speed")


def normalize_fallback_service_tier_override(value: Any) -> str | None:
    """Normalize one configured fallback tier override.

    ``None`` disables the policy.  Unknown or malformed values are rejected so
    callers can warn once at activation and then fail closed to the unchanged
    request policy.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("service_tier_override must be 'normal' or null")

    normalized = value.strip().lower()
    if normalized not in _SUPPORTED_SERVICE_TIER_OVERRIDES:
        raise ValueError("service_tier_override must be 'normal' or null")
    return normalized


def activate_fallback_service_tier_override(agent: Any, entry: Mapping[str, Any], *, log: Any) -> str | None:
    """Attach a validated entry policy to the active fallback runtime."""

    raw_value = entry.get("service_tier_override")
    try:
        normalized = normalize_fallback_service_tier_override(raw_value)
    except ValueError:
        log.warning(
            "Ignoring unsupported fallback service_tier_override=%r; "
            "supported value: normal",
            raw_value,
        )
        normalized = None

    agent._active_fallback_service_tier_override = normalized
    return normalized


def apply_fallback_service_tier_override(
    request_overrides: Mapping[str, Any] | None,
    active_override: str | None,
) -> dict[str, Any]:
    """Return transport overrides for the active fallback entry.

    Always deep-copies the caller's mapping.  ``normal`` removes only known
    premium-tier selectors from both the top level and ``extra_body`` while
    preserving unrelated provider-specific fields.
    """

    effective = deepcopy(dict(request_overrides or {}))
    if active_override != "normal":
        return effective

    for field in _TIER_FIELDS:
        effective.pop(field, None)

    extra_body = effective.get("extra_body")
    if isinstance(extra_body, dict):
        for field in _TIER_FIELDS:
            extra_body.pop(field, None)

    return effective
