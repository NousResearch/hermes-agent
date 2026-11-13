"""Provider-profile predicates the auxiliary client routes on."""

from __future__ import annotations

from typing import Any


PROFILE_REASONING_KEYS = {
    "reasoning", "reasoning_effort", "thinking", "thinking_config", "thinkingconfig",
    "thinking_budget", "thinkingbudget", "enable_thinking", "think", "verbosity",
}


def contains_profile_reasoning_fields(value: Any) -> bool:
    """Return whether a profile payload contains a reasoning wire control (recursive)."""
    if not isinstance(value, dict):
        return False
    return any(
        str(key).strip().lower() in PROFILE_REASONING_KEYS or contains_profile_reasoning_fields(nested)
        for key, nested in value.items()
    )


def profile_forwards_max_tokens(provider_norm: str) -> bool:
    """Whether the provider's profile declares that its wire honors an explicit ``max_tokens``."""
    try:
        from hermes_agent.providers import get_provider_profile
        profile = get_provider_profile(provider_norm)
    except Exception:  # health: allow BLE001 -- lazy plugin discovery must not break cap routing
        return False
    return profile is not None and bool(profile.forwards_max_tokens)
