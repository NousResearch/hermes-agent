"""Premium escalation gate for local/premium lane safety.

Small helper only; it stores no prompt/response content and performs no I/O.
"""
from __future__ import annotations

from typing import Any

PREMIUM_PROVIDERS = {
    "openai",
    "openai-codex",
    "anthropic",
    "openrouter",
    "nous",
    "xai",
    "xai-oauth",
}
LOCAL_PROVIDERS = {"ollama", "lmstudio", "local"}


def is_premium_provider(provider: str | None) -> bool:
    p = (provider or "").strip().lower()
    if not p:
        return False
    if p in LOCAL_PROVIDERS:
        return False
    return p in PREMIUM_PROVIDERS or p.startswith("custom:")


def premium_escalation_approved(agent: Any) -> bool:
    return bool(
        getattr(agent, "_premium_escalation_approved", False)
        or getattr(agent, "premium_escalation_approved", False)
    )


def premium_fallback_allowed(agent: Any, provider: str | None = None) -> bool:
    candidate = provider if provider is not None else getattr(agent, "provider", "")
    if not is_premium_provider(candidate):
        return True
    return premium_escalation_approved(agent)


def premium_gate_reason(provider: str | None) -> str:
    return (
        f"Premium fallback to {provider or 'unknown'} requires explicit approval; "
        "local draft failures remain draft-only until approved escalation."
    )
