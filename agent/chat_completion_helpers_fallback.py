"""Fallback-chain bookkeeping split from :mod:`agent.chat_completion_helpers`:
the activation log line and the post-exhaustion cooldown are self-contained
fallback concerns. ``chat_completion_helpers`` re-exports them, so callers and
tests keep importing from the parent module."""

from __future__ import annotations

import logging
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

# Fallback chain exhausted on a non-rate-limit failure (#24996): arm a short
# cooldown so the NEXT turn's restore_primary_runtime stays gated instead of
# resetting _fallback_index=0 and re-marshaling the whole context across every
# provider again (memory/swap exhaustion on constrained hosts). Rate-limit /
# billing reasons keep their own longer cooldown.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0


def _log_fallback_activated(agent, reason, old_model, old_provider, fb_model, fb_provider) -> None:
    """A billing switch is a WARNING naming the profile, both models and the remedy: the gateway
    persists the turn as a transient failure otherwise, and nothing in the log says the paid
    model was refused for credits or how to fix it (#115702). Other reasons stay INFO."""
    if reason != FailoverReason.billing:
        logger.info("Fallback activated: %s → %s (%s)", old_model, fb_model, fb_provider)
        return
    from hermes_constants import get_hermes_home, profile_name_for_home
    profile = profile_name_for_home(get_hermes_home()) or "default"
    remedy = "hermes model" if profile == "default" else f"hermes -p {profile} model"
    logger.warning(
        "Profile %s: %s via %s refused for billing/credits — using fallback %s via %s. "
        "Top up credits, or run `%s` to pick a model this account can use.",
        profile, old_model, old_provider, fb_model, fb_provider, remedy,
    )


def _fallback_chain_exhausted(agent, reason: FailoverReason | None) -> bool:
    """Chain exhausted (always False). A non-empty chain walked on a non-rate-limit failure arms a
    short cooldown so next turn's restore_primary_runtime stays gated instead of replaying the whole
    context across every provider again."""
    from agent.fallback_cooldown import _RATE_LIMIT_FAILOVER_REASONS
    if agent._fallback_chain and reason not in _RATE_LIMIT_FAILOVER_REASONS:
        agent._rate_limited_until = max(
            getattr(agent, "_rate_limited_until", 0) or 0, time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S)
    return False
