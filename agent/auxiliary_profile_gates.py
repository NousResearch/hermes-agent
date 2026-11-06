"""Provider-profile predicates the auxiliary client routes on.

Split out of ``agent/auxiliary_client.py`` (which sits at its code-health line cap) so the
profile seams stay importable and independently testable.
"""

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
    """Whether the provider's profile opts its wire into explicit ``max_tokens`` forwarding.

    Plugin external-process routes (``process://<id>``) match none of the built-in
    ``_forwards_max_tokens`` branches, so an auxiliary cap was dropped silently (#139778);
    a profile that verified its endpoint honors the kwarg declares ``forwards_max_tokens``.
    A miss keeps the omit-the-cap default; so does a broken plugin — profile lookup spans
    lazy plugin discovery, and cap routing must survive it.
    """
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider_norm)
    except Exception:  # health: allow BLE001 -- lazy plugin discovery must not break cap routing
        return False
    return profile is not None and bool(profile.forwards_max_tokens)
