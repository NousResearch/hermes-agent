"""Per-turn OpenRouter service-tier escalation based on streaming TTFT.

Turn-local state lives on ``agent._service_tier_escalation``. Canonical
``agent.service_tier`` and ``agent.request_overrides`` are the baseline and
are never mutated by this module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from hermes_constants import (
    DEFAULT_SERVICE_TIER_ESCALATION,
    SERVICE_TIER_BOUNDED_VALUES,
    ServiceTierEscalationConfig,
    parse_service_tier,
    resolve_service_tier_escalation_config,
)

logger = logging.getLogger("run_agent")

# * Ladder: flex → default (omit field) → priority. No in-turn downgrade.
_TIER_LADDER: tuple[str | None, ...] = ("flex", None, "priority")

# * Surfaces that must not escalate even if a caller passes enabled config.
_DISABLED_PLATFORMS = frozenset({"cron", "subagent"})

# * Distinguishes ``begin_turn()`` (keep current base) from ``begin_turn(None)``
# (explicit default / omit-on-wire baseline after a model switch).
_KEEP_BASE_TIER = object()
_WIRE_TIERS = frozenset({"flex", "priority"})


def _ladder_tier(value) -> str | None:
    """Map a stored preference onto the TTFT ladder (flex / default / priority).

    ``auto`` / ``cold`` are bounded windows, not wire rungs — they sit at default.
    """
    parsed = parse_service_tier(value)
    return parsed if parsed in _WIRE_TIERS else None


class TtftObservation:
    """Request-local TTFT sample. Not stored on the agent across calls."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.clock = clock or time.monotonic
        self.t_send: float | None = None
        self.t_first: float | None = None
        self.open_count = 0

    def mark_send(self) -> None:
        self.open_count += 1
        if self.t_send is None:
            self.t_send = self.clock()

    def mark_first_delta(self) -> None:
        if self.t_first is None:
            self.t_first = self.clock()

    def ttft_seconds(self) -> float | None:
        if self.t_send is None or self.t_first is None:
            return None
        return max(0.0, self.t_first - self.t_send)

    def was_retried(self) -> bool:
        return self.open_count > 1


class ServiceTierEscalationState:
    """Turn-scoped ladder, streak, and current effective tier."""

    def __init__(
        self,
        config: ServiceTierEscalationConfig | None = None,
        base_tier: str | None = None,
    ) -> None:
        self.config = config or DEFAULT_SERVICE_TIER_ESCALATION
        self.base_tier = _ladder_tier(base_tier)
        self.effective_tier: str | None = self.base_tier
        self.streak = 0
        # * Rungs climbed from base this turn. Stored separately so a cap at
        # priority does not forget the height when failover rebases.
        self.climbed_rungs = 0
        self.last_ttft_seconds: float | None = None
        # * Pre-attempt wire snapshot: outer-retry of one logical request
        # must not pick up a tier climbed from a not-yet-accepted sample.
        self.wire_locked = False
        self.wire_tier: str | None = self.base_tier
        self.pending_ttft: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def begin_turn(self, base_tier: Any = _KEEP_BASE_TIER) -> None:
        """Reset streak and adopt *base_tier* when the caller passes one.

        Omitted *base_tier* keeps the current baseline. Explicit ``None``
        is the default/normal rung (omit ``service_tier`` on the wire).
        """
        if base_tier is not _KEEP_BASE_TIER:
            self.base_tier = _ladder_tier(base_tier)
        self.effective_tier = self.base_tier
        self.streak = 0
        self.climbed_rungs = 0
        self.last_ttft_seconds = None
        self.wire_locked = False
        self.wire_tier = self.base_tier
        self.pending_ttft = None

    def reset_for_model_switch(self, base_tier: str | None = None) -> None:
        """Drop turn state after ``switch_model``; adopt the new model's base."""
        self.begin_turn(base_tier)

    def note_non_observation(self) -> None:
        """Retry / error / interrupt / non-streaming: drop streak, keep tier."""
        self.streak = 0
        self.pending_ttft = None

    def observe_ttft(self, seconds: float, *, model: str = "") -> None:
        """Record a successful streaming TTFT and maybe climb one ladder step."""
        if not self.enabled:
            return
        try:
            ttft = float(seconds)
        except (TypeError, ValueError):
            self.streak = 0
            return
        self.last_ttft_seconds = ttft
        if ttft <= self.config.ttft_threshold_seconds:
            self.streak = 0
            return
        self.streak += 1
        if self.streak < self.config.consecutive_slow_requests:
            return
        previous = self.effective_tier
        nxt = _next_tier(previous)
        self.streak = 0
        if nxt == previous:
            return
        self.effective_tier = nxt
        self.climbed_rungs = _climbed_rungs(self.base_tier, self.effective_tier)
        logger.info(
            "service_tier_escalation: model=%s %s → %s (streak_trigger=%s last_ttft=%.3fs)",
            model or "-",
            _tier_label(previous),
            _tier_label(nxt),
            self.config.consecutive_slow_requests,
            ttft,
        )


def _tier_label(tier: str | None) -> str:
    return "default" if tier is None else str(tier)


def _next_tier(current: str | None) -> str | None:
    try:
        idx = _TIER_LADDER.index(current)
    except ValueError:
        return current
    if idx >= len(_TIER_LADDER) - 1:
        return _TIER_LADDER[-1]
    return _TIER_LADDER[idx + 1]


def _rung_index(tier: str | None) -> int:
    parsed = _ladder_tier(tier)
    try:
        return _TIER_LADDER.index(parsed)
    except ValueError:
        return _TIER_LADDER.index(None)


def _climbed_rungs(base_tier: str | None, effective_tier: str | None) -> int:
    return max(0, _rung_index(effective_tier) - _rung_index(base_tier))


def _climb_from(base_tier: str | None, rungs: int) -> str | None:
    """Walk *rungs* steps up from *base_tier*, capped at priority."""
    base = _ladder_tier(base_tier)
    try:
        steps = int(rungs)
    except (TypeError, ValueError):
        steps = 0
    if steps <= 0:
        return base
    idx = min(_rung_index(base) + steps, len(_TIER_LADDER) - 1)
    return _TIER_LADDER[idx]


def _openrouter_service_tier_route(agent: Any) -> bool:
    """Same OpenRouter predicate used when overlaying ``service_tier`` on the wire."""
    try:
        from hermes_cli.models import _is_openrouter_service_tier_route

        return bool(
            _is_openrouter_service_tier_route(
                getattr(agent, "provider", None),
                getattr(agent, "base_url", None),
            )
        )
    except Exception:
        return False


def _surface_blocks_escalation(agent: Any) -> bool:
    platform = str(getattr(agent, "platform", None) or "").strip().lower()
    if platform in _DISABLED_PLATFORMS:
        return True
    if getattr(agent, "_persist_disabled", False):
        return True
    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0:
        return True
    if getattr(agent, "is_subagent", False):
        return True
    # * Hard gate for batch_runner and gateway background tasks: even an
    # accidentally-enabled config must not climb the ladder.
    if getattr(agent, "_block_service_tier_escalation", False):
        return True
    return False


def escalation_is_active(agent: Any) -> bool:
    """True when this agent may climb the ladder and overlay request kwargs."""
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState) or not state.enabled:
        return False
    if getattr(agent, "_service_tier_session_pinned", False):
        return False
    if _surface_blocks_escalation(agent):
        return False
    return True


def bind_service_tier_escalation(agent: Any, raw_config: Any = None) -> ServiceTierEscalationState:
    """Attach turn-local escalation state. ``None`` / invalid → disabled."""
    if isinstance(raw_config, ServiceTierEscalationConfig):
        config = raw_config
    elif isinstance(raw_config, dict) and (
        "enabled" in raw_config
        or "ttft_threshold_seconds" in raw_config
        or "consecutive_slow_requests" in raw_config
    ) and "service_tier_escalation" not in raw_config:
        config = resolve_service_tier_escalation_config(
            {"service_tier_escalation": raw_config},
        )
    else:
        config = resolve_service_tier_escalation_config(
            raw_config if isinstance(raw_config, dict) else {},
        )
    state = ServiceTierEscalationState(
        config=config,
        base_tier=getattr(agent, "service_tier", None),
    )
    agent._service_tier_escalation = state
    return state


def begin_escalation_turn(agent: Any) -> None:
    """Entry of ``run_conversation``: always reset to configured base tier."""
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    state.begin_turn(getattr(agent, "service_tier", None))


def reset_escalation_for_model_switch(agent: Any) -> None:
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    state.reset_for_model_switch(getattr(agent, "service_tier", None))


def rebase_escalation_runtime(agent: Any, new_base_tier: str | None) -> None:
    """Re-anchor the TTFT ladder on a fallback/restore model's base tier.

    Preserves climbed rungs and the slow-streak. Recalculates ``effective_tier``
    and ``wire_tier`` as ``climb(new_base, rungs)`` capped at priority.
    Unlocks the wire snapshot so the next logical request pins the rebased
    wire — a failover is a new model, not an outer-retry of the old one.
    Does not reset the ladder (unlike ``begin_turn`` / ``switch_model``).
    """
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    stored = getattr(state, "climbed_rungs", 0)
    try:
        stored_rungs = int(stored)
    except (TypeError, ValueError):
        stored_rungs = 0
    rungs = max(stored_rungs, _climbed_rungs(state.base_tier, state.effective_tier))
    new_base = _ladder_tier(new_base_tier)
    state.base_tier = new_base
    state.climbed_rungs = rungs
    state.effective_tier = _climb_from(new_base, rungs)
    state.wire_tier = state.effective_tier
    state.wire_locked = False
    state.pending_ttft = None


def note_non_observation(agent: Any) -> None:
    state = getattr(agent, "_service_tier_escalation", None)
    if isinstance(state, ServiceTierEscalationState):
        state.note_non_observation()


def begin_request_ttft(
    agent: Any,
    clock: Callable[[], float] | None = None,
) -> TtftObservation:
    obs = TtftObservation(clock=clock)
    stack = getattr(agent, "_ttft_obs_stack", None)
    if not isinstance(stack, list):
        stack = []
        agent._ttft_obs_stack = stack
    stack.append(obs)
    return obs


def end_request_ttft(agent: Any, obs: TtftObservation) -> None:
    stack = getattr(agent, "_ttft_obs_stack", None)
    if isinstance(stack, list) and stack and stack[-1] is obs:
        stack.pop()


def _current_ttft_obs(agent: Any) -> TtftObservation | None:
    stack = getattr(agent, "_ttft_obs_stack", None)
    if isinstance(stack, list) and stack:
        top = stack[-1]
        if isinstance(top, TtftObservation):
            return top
    return None


def mark_ttft_send(agent: Any) -> None:
    obs = _current_ttft_obs(agent)
    if obs is not None:
        obs.mark_send()


def mark_ttft_first_delta(agent: Any) -> None:
    obs = _current_ttft_obs(agent)
    if obs is not None:
        obs.mark_first_delta()


def finish_request_ttft(
    agent: Any,
    obs: TtftObservation,
    *,
    interrupted: bool = False,
) -> None:
    """Park a TTFT sample until the outer loop accepts this logical request.

    Inner stream retries (``open_count > 1``), interrupts, and missing
    samples are non-observations. Outer-loop retries must call
    ``begin_logical_request`` again so the parked sample is dropped
    without climbing the ladder.
    """
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    # * Non-OpenRouter samples must not park pending or climb the ladder;
    # a later OpenRouter failover in the same turn would inherit them.
    if not _openrouter_service_tier_route(agent):
        return
    if not escalation_is_active(agent):
        note_non_observation(agent)
        return
    if interrupted or obs.was_retried():
        note_non_observation(agent)
        return
    ttft = obs.ttft_seconds()
    if ttft is None:
        note_non_observation(agent)
        return
    state.pending_ttft = ttft


def begin_logical_request(agent: Any) -> None:
    """Pin wire-tier for this logical request; drop a parked sample on retry.

    First call locks ``wire_tier`` to the current effective tier. Further
    calls while still locked are outer-retries of the same request: the
    parked observation is discarded and the locked tier stays on the wire.
    """
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    if state.wire_locked:
        state.note_non_observation()
        return
    state.wire_tier = state.effective_tier
    state.wire_locked = True
    state.pending_ttft = None


def accept_logical_request(agent: Any) -> None:
    """Commit a parked TTFT after the outer loop accepts the response.

    Unlocks the wire snapshot so the next tool-loop iteration overlays
    the (possibly climbed) effective tier.
    """
    state = getattr(agent, "_service_tier_escalation", None)
    if not isinstance(state, ServiceTierEscalationState):
        return
    pending = state.pending_ttft
    state.pending_ttft = None
    state.wire_locked = False
    if pending is None or not escalation_is_active(agent):
        return
    state.observe_ttft(pending, model=str(getattr(agent, "model", "") or ""))


def apply_escalation_to_overrides(agent: Any, overrides: dict) -> dict:
    """Last-step overlay: OpenRouter ``service_tier`` only; default omits the key."""
    if not escalation_is_active(agent):
        return overrides
    state: ServiceTierEscalationState = agent._service_tier_escalation
    if not _openrouter_service_tier_route(agent):
        return overrides

    target = state.wire_tier if state.wire_locked else state.effective_tier
    # * auto/cold windows own the per-request field until the ladder climbs.
    if target is None and getattr(agent, "service_tier", None) in SERVICE_TIER_BOUNDED_VALUES:
        return overrides
    present = parse_service_tier(overrides.get("service_tier"))
    if target is None and "service_tier" not in overrides and "speed" not in overrides:
        return overrides
    if target is not None and present == target:
        if state.effective_tier != state.base_tier:
            logger.debug(
                "service_tier_escalation applying %s to request (model=%s)",
                _tier_label(target),
                getattr(agent, "model", None),
            )
        return overrides

    if target in ("flex", "priority"):
        overrides["service_tier"] = target
        overrides.pop("speed", None)
    else:
        overrides.pop("service_tier", None)
        overrides.pop("speed", None)
    if state.effective_tier != state.base_tier:
        logger.debug(
            "service_tier_escalation applying %s to request (model=%s)",
            _tier_label(target),
            getattr(agent, "model", None),
        )
    return overrides
