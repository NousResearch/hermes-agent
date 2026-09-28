"""Corroboration engine: the decaying, corroborated score that upgrades
Command Code's inert ``confidence`` into a living, self-correcting signal.

Offline. Pure stdlib. Deterministic (injectable clock) for testing.

How it upgrades CC
------------------
Command Code stores ``confidence: X`` once, parses it back with a regex, and
never ages it. That means a confidence earned from one early session survives
indefinitely and never drops, and conflicting evidence from two sessions just
silently overwrites. Our engine fixes both:

  * accumulation  -- repeated corroboration raises the weight (bounded growth).
  * decay         -- stale evidence loses weight (age-based, half-life).
  * staleness     -- when a candidate hasn't been corroborated in a long time
                     it is *escalated* to a human review queue (a signal CC
                     has), but with a threshold + auto-ack, not a hard wall.
  * conflict      -- two sessions disagree -> the conflict is tracked, and if
                     the two most recent high-weight confidences disagree by
                     more than EPSILON, the candidate is escalated.

The output is always a score in [0, 1] that we write into the CC-compatible
``confidence:`` token, so disk format stays interoperable.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

from .taste_decay import DecayConfig, decayed_score, growth_weight


# --- thresholds -------------------------------------------------------------
# A candidate that has been corroborated at least this many times is
# "established" and eligible to be written back to the shared taste.md.
MIN_OBSERVATIONS_FOR_WRITE = 3
# Escalate to a human when a candidate has not been corroborated in this many
# days (staleness), or when two sessions disagree strongly (conflict).
STALENESS_DAYS = 21.0
CONFLICT_EPSILON = 0.15  # confidences within 0.15 of each other agree.
# Auto-ack: a candidate that has been observed this many times is trusted to be
# written without a human in the loop.
AUTO_ACK_OBSERVATIONS = 10

# Escalation reason constants.
ESCALATION_STALENESS = "staleness"
ESCALATION_CONFLICT = "conflict"


@dataclass
class Candidate:
    """A learned preference under supervision, tracked before it earns its spot
    in the shared ``taste.md``. This is the machine-readable state we persist
    alongside the human-facing score."""

    id: str
    label: str
    weight: float = 0.0
    n_obs: int = 0
    last_obs_epoch: float = field(default_factory=time.time)
    first_obs_epoch: Optional[float] = None
    conflicts: int = 0
    last_conflict_epoch: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=2))  # (epoch, raw_score)

    def to_dict(self):
        d = asdict(self)
        d["history"] = list(self.history)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            label=d.get("label", d["id"]),
            weight=d.get("weight", 0.0),
            n_obs=d.get("n_obs", 0),
            last_obs_epoch=d.get("last_obs_epoch", 0.0),
            first_obs_epoch=d.get("first_obs_epoch", None),
            conflicts=d.get("conflicts", 0),
            last_conflict_epoch=d.get("last_conflict_epoch", 0.0),
            history=deque(d.get("history", []), maxlen=2),
        )


@dataclass
class Escalation:
    """A candidate flagged for human review (staleness or conflict).

    This is a soft signal: the candidate still exists and keeps accumulating,
    but we surface it so a human (or a later auto-ack) can resolve it. Mirrors
    CC's escalation idea without the hard wall of ``escalated: true``.
    """

    id: str
    reason: str  # ESCALATION_STALENESS | ESCALATION_CONFLICT
    epoch: float
    note: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def staleness(cls, c: "Candidate", age_days: float, epoch: float):
        return cls(c.id, ESCALATION_STALENESS, epoch, f"not corroborated in {age_days:.1f}d")

    @classmethod
    def conflict(cls, c: "Candidate", a: float, b: float, epoch: float):
        return cls(c.id, ESCALATION_CONFLICT, epoch, f"confidences {a:.2f} vs {b:.2f}")


@dataclass
class CorroborationResult:
    """Outcome of one observation."""

    id: str
    label: str
    raw_weight: float
    decayed_weight: float
    score: float  # the human-facing confidence to write (0..1, decayed)
    established: bool  # has MIN_OBSERVATIONS_FOR_WRITE been met?
    auto_ack: bool  # has AUTO_ACK_OBSERVATIONS been met?
    escalated: Optional[Escalation]
    conflict: bool
    n_obs: int
    age_days: float


class CorroborationEngine:
    """Accumulates corroboration for a single candidate and computes its
    decaying, corroborated score. One instance per candidate; state can be
    (de)serialised via :meth:`snapshot` / :meth:`restore`."""

    def __init__(
        self,
        id: str,
        label: str,
        decay: Optional[DecayConfig] = None,
        clock=time.time,
    ):
        if not id or not label:
            raise ValueError("id and label are required")
        self.id = id
        self.label = label
        self.decay = decay or DecayConfig()
        self._clock = clock
        self._state = Candidate(id=id, label=label)

    # -- clock access (injectable, default real time) ------------------------
    @property
    def now(self):
        return self._clock()

    def advance(self, seconds: float) -> float:
        """TEST-ONLY: advance the injected clock by ``seconds``.

        Never call this outside tests. The clock must expose a mutable
        ``t`` attribute (a FakeClock in tests). A bare callable clock such
        as the real :func:`time.time` cannot be advanced and raises, which
        is correct: you can only move time forward in a fake clock.
        """
        clock = self._clock
        if not hasattr(clock, "t"):
            raise TypeError("advance() requires a clock with a mutable 't' "
                            "attribute (a FakeClock); cannot advance a bare callable clock")
        clock.t += seconds
        return clock.t

    # -- decay helper (respects the enabled flag) ----------------------------
    def _apply_decay(self, weight: float, age_days: float) -> float:
        """Decay ``weight`` by ``age_days``, unless decay is disabled."""
        if not self.decay.enabled:
            return weight
        return decayed_score(weight, age_days, self.decay.half_life_days)

    def _build_result(self, escalation, conflict: bool, age_days: float) -> CorroborationResult:
        c = self._state
        score_out = self._apply_decay(c.weight, 0.0)
        return CorroborationResult(
            id=self.id,
            label=self.label,
            raw_weight=c.weight,
            decayed_weight=score_out,
            score=round(score_out, 4),
            established=c.n_obs >= MIN_OBSERVATIONS_FOR_WRITE,
            auto_ack=c.n_obs >= AUTO_ACK_OBSERVATIONS,
            escalated=escalation,
            conflict=conflict,
            n_obs=c.n_obs,
            age_days=round(age_days, 3),
        )

    # -- core observation ----------------------------------------------------
    def observe(self, score: float = 0.5) -> CorroborationResult:
        """Feed one new corroborated observation (a score in [0,1]).

        Returns a result describing the freshly computed score and whether the
        candidate has crossed the write / auto-ack / escalation thresholds.

        The observation is *accepted even on conflict* -- we track the conflict
        but never discard evidence (a lost conflict in CC is worse than an
        open one). Conflict resolution is a separate pass (:meth:`resolve`).
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")

        c = self._state
        age_days = (self.now - c.last_obs_epoch) / 86400.0
        if age_days < 0:
            age_days = 0.0

        # Decay the *previous* weight by the gap since last obs, then fold in
        # the new evidence as a weighted blend. The blend factor alpha grows
        # with n_obs, so repeated agreement raises the score toward the
        # evidence strength -- but the actual input magnitude always matters
        # (no growth curve can dominate regardless of input).
        prev_weight = c.weight
        c.n_obs += 1
        c.last_obs_epoch = self.now
        if c.first_obs_epoch is None:
            c.first_obs_epoch = self.now

        effective_prev = self._apply_decay(prev_weight, age_days) if c.n_obs > 1 else 0.0
        alpha = growth_weight(c.n_obs)
        c.weight = alpha * effective_prev + (1.0 - alpha) * score

        # Store the raw input score so conflict detection compares evidence
        # against evidence (not the merged running weight against raw input).
        c.history.append((c.last_obs_epoch, score))

        # Conflict check against the previous raw score.
        conflict = False
        escalation = None
        if c.n_obs >= 2:
            prev_score = c.history[-2][1]  # history entries are (epoch, raw_score)
            if abs(prev_score - score) > CONFLICT_EPSILON:
                conflict = True
                c.conflicts += 1
                c.last_conflict_epoch = self.now
                escalation = Escalation.conflict(c, prev_score, score, self.now)

        # NOTE: no staleness escalation here. The observation itself cures the
        # staleness (last_obs_epoch was just refreshed); staleness is reported
        # only on the peek path (to_result_now / is_stale).
        return self._build_result(escalation, conflict, age_days)

    # -- conflict resolution -------------------------------------------------
    def resolve_conflict(self, confidence: float = 0.5) -> CorroborationResult:
        """Apply a human/authoritative resolution confidence (a tie-breaker).

        Sets the running weight directly to the resolved value and clears the
        conflict count. Unlike :meth:`observe`, this does NOT re-run the
        conflict detector, so a resolution can never immediately re-conflict.
        Returns a fresh result with conflict=False.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        c = self._state
        c.weight = confidence
        c.conflicts = 0
        c.last_conflict_epoch = self.now
        c.last_obs_epoch = self.now
        if c.first_obs_epoch is None:
            c.first_obs_epoch = self.now
        c.n_obs += 1
        c.history.append((self.now, confidence))
        return self._build_result(None, False, 0.0)

    # -- lifecycle -----------------------------------------------------------
    def is_stale(self):
        if self._state.n_obs < 2:
            return False
        age_days = (self.now - self._state.last_obs_epoch) / 86400.0
        return age_days > STALENESS_DAYS

    def is_conflicting(self):
        return self._state.conflicts > 0

    def should_write(self):
        """A candidate is eligible for the shared taste.md once established
        AND free of escalation. Conflicted or stale candidates never reach
        taste.md without an ack (see doc section 4.5).

        Returns False when is_conflicting() or is_stale() is true, even if
        n_obs >= MIN_OBSERVATIONS_FOR_WRITE.
        """
        if self.is_conflicting() or self.is_stale():
            return False
        return self._state.n_obs >= MIN_OBSERVATIONS_FOR_WRITE

    def snapshot(self):
        """Return a serialisable dict for the in-memory cache / disk sidecar."""
        return {
            "id": self._state.id,
            "label": self._state.label,
            "weight": self._state.weight,
            "n_obs": self._state.n_obs,
            "last_obs_epoch": self._state.last_obs_epoch,
            "first_obs_epoch": self._state.first_obs_epoch,
            "conflicts": self._state.conflicts,
            "last_conflict_epoch": self._state.last_conflict_epoch,
            "history": list(self._state.history),
        }

    def restore(self, state: dict):
        self._state = Candidate.from_dict(state)
        return self

    def to_result_now(self) -> CorroborationResult:
        """Return the current decaying score without a new observation (peek)."""
        age_days = 0.0
        if self._state.n_obs >= 1:
            age_days = max(0.0, (self.now - self._state.last_obs_epoch) / 86400.0)
        score_out = self._apply_decay(self._state.weight, age_days)
        return CorroborationResult(
            id=self.id,
            label=self.label,
            raw_weight=self._state.weight,
            decayed_weight=score_out,
            score=round(score_out, 4),
            established=self._state.n_obs >= MIN_OBSERVATIONS_FOR_WRITE,
            auto_ack=self._state.n_obs >= AUTO_ACK_OBSERVATIONS,
            escalated=(
                Escalation.staleness(self._state, age_days, self.now)
                if self.is_stale() else None
            ),
            conflict=self.is_conflicting(),
            n_obs=self._state.n_obs,
            age_days=round(age_days, 3),
        )
