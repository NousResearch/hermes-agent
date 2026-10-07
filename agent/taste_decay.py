"""Decay model for the corroboration engine.

Pure stdlib. Offline. Deterministic (injectable clock) so it can be unit-tested.

Design rationale (why this shape)
---------------------------------
1. Weights must always be non-negative and normalisable, so the base growth
   function is monotonic and bounded.  ``math.expm1`` grows ~linearly for small
   inputs and saturates toward 1.0 as ``t`` -> inf, which gives us a natural
   "recent evidence weighs more" curve without a magic breakpoint.
2. Half-life is the domain language that matches Command Code's own
   ``half-life`` concept, so we accept it and convert to a decay constant.
3. The decayed score returned here is the *only* thing written to disk as the
   human-facing ``confidence`` token, guaranteeing format compatibility with
   CC's ``confidence:\\s*([0-9.]+)`` parser. The raw (undecayed) weight is the
   machine-readable companion field.
"""

import math
from dataclasses import dataclass


# The weight at t=0 is expm1(0) = 0, so a brand-new preference starts at 0 and
# only gains weight from observed corroboration. That is intentional: an
# unconfirmed candidate preference is not yet a confident taste until it is seen.
GROWTH_SCALE = 0.25  # per corroboration observation (CC default half-life ~= 14d)


def _half_life_to_decay_constant(half_life_days):
    """Convert a physical half-life into a per-day exponential decay rate."""
    if half_life_days is None:
        # No half-life configured -> no decay. The weight only ever grows.
        return 0.0
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    # We want w(t) to lose 50% of its value every half_life_days.
    # decayed = base * 2^(-t / half_life) = base * exp(-lambda * t)
    # => lambda = ln(2) / half_life.
    return math.log(2.0) / half_life_days


def growth_weight(n_obs, scale=GROWTH_SCALE):
    """Bounded, monotonic growth weight from n_obs corroboration observations.

    Uses the classic saturating curve ``1 - exp(-scale * n)``: starts at 0, is
    strictly increasing, and asymptotes to (but never reaches) 1.0. The earlier
    ``expm1`` normalisation in this module grew past 1.0 (it hit ~2.28 at n=2),
    which violated the [0,1] bound this score is contractually required to hold.
    """
    if n_obs < 0:
        raise ValueError("n_obs must be >= 0")
    return 1.0 - math.exp(-scale * n_obs)


def decay_weight(weight, age_days, half_life_days):
    """Apply age-based decay to a raw weight (>= 0)."""
    if age_days < 0:
        raise ValueError("age_days must be >= 0")
    if weight < 0:
        raise ValueError("weight must be >= 0")
    if age_days == 0:
        return weight
    lam = _half_life_to_decay_constant(half_life_days)
    if lam == 0.0:
        # No decay configured.
        return weight
    return weight * math.exp(-lam * age_days)


def decayed_score(weight, age_days, half_life_days):
    """The human-facing confidence score = decayed weight, clamped to [0, 1]."""
    s = decay_weight(weight, age_days, half_life_days)
    # Clamp defensively: growth_weight -> 1 at infinity, decay can't push >1,
    # but floating point near-saturations are clamped for stable disk output.
    return max(0.0, min(1.0, s))


@dataclass
class DecayConfig:
    """Half-life, in days, applied to corroboration weights.

    ``None`` disables decay entirely (weight only grows), which mirrors the
    current Command Code behaviour (inert confidence) and is a safe baseline
    for the very first rollout of this engine.
    """

    half_life_days: float = 14.0
    enabled: bool = True

    def effective_decay_constant(self):
        if not self.enabled:
            return 0.0
        return _half_life_to_decay_constant(self.half_life_days)

    def clone(self):
        return DecayConfig(half_life_days=self.half_life_days, enabled=self.enabled)
