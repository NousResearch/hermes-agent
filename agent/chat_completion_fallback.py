"""Fallback-chain exhaustion cooldown, split out of ``chat_completion_helpers``.

The helpers here are the #24996 anti-storm gate: a chain walked on a non-rate-limit failure arms a
short cooldown so the NEXT turn's ``restore_primary_runtime`` stays gated instead of resetting
``_fallback_index = 0`` and re-marshaling the whole context across every provider again.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.error_classifier import FailoverReason

# Fallback chain exhausted on a non-rate-limit failure (#24996): arm a short
# cooldown so the NEXT turn's restore_primary_runtime stays gated instead of
# resetting _fallback_index=0 and re-marshaling the whole context across every
# provider again (memory/swap exhaustion on constrained hosts). Rate-limit /
# billing reasons keep their own longer cooldown.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0

def _fallback_chain_exhausted(agent, reason: FailoverReason | None) -> bool:
    """Chain exhausted (always False). A non-empty chain walked on a non-rate-limit failure arms a
    short cooldown so next turn's restore_primary_runtime stays gated instead of replaying the whole
    context across every provider again."""
    from agent.fallback_cooldown import _RATE_LIMIT_FAILOVER_REASONS
    if agent._fallback_chain and reason not in _RATE_LIMIT_FAILOVER_REASONS:
        _existing_cooldown = getattr(agent, "_rate_limited_until", 0) or 0
        _now = time.monotonic()
        # Only arm a fresh exhaustion cooldown when the existing window has
        # already expired (first exhaustion or expired cooldown). This prevents
        # repeated calls on an already-exhausted chain from re-arming the gate
        # every sub-window turn and locking out the primary restore permanently.
        # Keeps the bounded-replay guarantee from #24996 (at most one full chain
        # walk per cooldown window) while allowing the throttle to naturally
        # expire so the chain resets for unrelated later failures (#57582).
        if _now >= _existing_cooldown:
            agent._rate_limited_until = _now + _FALLBACK_EXHAUSTED_COOLDOWN_S
    return False
