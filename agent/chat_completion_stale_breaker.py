"""Cross-turn stale-call circuit breaker (#58962).

A session wedged against an unresponsive provider would otherwise hit the
stale detector on every call forever. ``agent._consecutive_stale_streams``
is bumped on every stale kill and reset only when a call completes or the
provider is swapped (switch_model / try_activate_fallback /
restore_primary_runtime — the streak measured the OLD provider). Past the
give-up threshold, calls abort immediately with an actionable error.

Half-open recovery (#89587): an unattended single-provider session has no
human to switch models and no fallback chain to swap to, so a tripped
breaker used to latch failed forever even after the provider recovered.
Once per ``agent.stream_stale_probe_interval`` window (config.yaml), one real
probe attempt is allowed through instead of raising; success clears the streak
via the existing completed-call resets, a stale probe re-bumps it and the
window re-arms at probe-attempt time.
"""

import contextlib
import logging
import threading
import time

from utils import env_int

logger = logging.getLogger(__name__)


def _stale_streak(agent) -> int:
    try:
        return int(getattr(agent, "_consecutive_stale_streams", 0) or 0)
    except Exception:
        return 0


def _bump_stale_streak(agent) -> None:
    with contextlib.suppress(Exception):
        agent._consecutive_stale_streams = _stale_streak(agent) + 1


def _reset_stale_streak(agent) -> None:
    with contextlib.suppress(Exception):
        agent._consecutive_stale_streams = 0
        # A future fresh trip should start un-armed: its first over-threshold
        # call insta-fails and arms the probe window, rather than inheriting a
        # long-expired window and burning a stale-timeout wait immediately.
        agent._stale_probe_after = 0.0


def _is_responsive_provider_error(error) -> bool:
    """True when *error* carries an authoritative HTTP response from the
    provider — a real status code (429/503/401/…) proves the provider is
    reachable, however unhappy. Timeouts, socket drops, and the locally
    built TimeoutErrors of the stale detectors carry no status and stay
    False. Shapes covered: openai/anthropic ``APIStatusError``
    (``.status_code``), ``httpx.HTTPStatusError`` (``.response.status_code``),
    botocore ``ClientError`` (dict ``.response`` with ResponseMetadata)."""
    if error is None:
        return False
    if isinstance(getattr(error, "status_code", None), int):
        return True
    resp = getattr(error, "response", None)
    if resp is None:
        return False
    if isinstance(getattr(resp, "status_code", None), int):
        return True
    if isinstance(resp, dict):
        try:
            meta = resp.get("ResponseMetadata") or {}
            return isinstance(meta.get("HTTPStatusCode"), int)
        except (AttributeError, TypeError):  # ResponseMetadata is not a mapping
            return False
    return False


def _resolve_stale_on_responsive_error(agent, error) -> None:
    """Any authoritative provider response resolves the stale/no-response
    condition this breaker represents: a prompt 429/503/401 proves the
    provider is reachable again, so the breaker must not mask that newer,
    more accurate state by short-circuiting the normal retry/rate-limit/
    fallback machinery's next attempt as "provider unresponsive". Streak
    and probe window both clear; subsequent handling belongs to the normal
    HTTP error policy. A timeout / no-byte / stale outcome carries no
    status, so it leaves the breaker open (the stale kill paths bump the
    streak themselves)."""
    try:
        if _stale_streak(agent) and _is_responsive_provider_error(error):
            logger.info("Stale give-up breaker cleared: provider answered with a real HTTP response (%s) — "
                        "no longer unresponsive; normal retry policy owns the error.", type(error).__name__)
            _reset_stale_streak(agent)
    except Exception:
        logger.debug("stale resolve-on-response failed", exc_info=True)


# Half-open probe window for the tripped give-up breaker (#89587). Worst
# case a probe against a still-wedged LOCAL endpoint burns up to the 900s
# local stale ceiling, so 300s bounds the duty cycle to one wedged wait per
# ~15 min while cloud providers (180s stale timeout) recover within ~5 min.
_STALE_PROBE_INTERVAL_S = 300.0


def _stale_probe_interval() -> float:
    """``agent.stream_stale_probe_interval`` (seconds, default 300; ``0`` keeps the tripped
    breaker latched). Read only once the breaker has tripped, like
    ``agent.local_stream_stale_timeout``."""
    interval = _STALE_PROBE_INTERVAL_S
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()  # read-only consumer — no deepcopy
        agent_cfg = cfg.get("agent") if isinstance(cfg, dict) else None
        value = agent_cfg.get("stream_stale_probe_interval") if isinstance(agent_cfg, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            interval = float(value)
    return interval

# Serializes the check-elapsed → arm-next-window → admit-probe transition in
# _check_stale_giveup: concurrent callers reaching an expired window boundary
# must not each observe it elapsed and all claim the same probe opportunity —
# at most one call per window reaches provider dispatch. Module-level (not
# per-agent) because lazy per-agent lock creation would itself race; the
# critical section is a few attribute reads/writes, so cross-agent contention
# is negligible.
_stale_probe_claim_lock = threading.Lock()


def _check_stale_giveup(agent, *, allow_probe: bool = True) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait.

    Half-open recovery (#89587): when ``allow_probe`` is True (the
    pre-network entry guards), one real probe attempt per
    ``agent.stream_stale_probe_interval`` window is allowed through
    instead of raising, so an unattended session can self-heal once the
    provider recovers. The window is armed at probe-ATTEMPT time, so a
    probe that itself wedges (a full stale-timeout wait) cannot be
    followed by another until the next window. ``allow_probe=False``
    (the Bedrock mid-call escalation) keeps the unconditional raise —
    that site is already ending the call as stale, so an admitted probe
    there could never dispatch: it would only consume the window's single
    claim and push real recovery out by another full cooldown.
    """
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup <= 0 or _streak < _giveup:
        # NOTE: keep this early-out ahead of any ``_stale_probe_after``
        # access — below-threshold agents (including bare-mock test agents)
        # must never touch the probe state.
        return
    _extra = ""
    _interval = _stale_probe_interval()
    if allow_probe and _interval > 0:
        try:
            _admitted = False
            _now = time.monotonic()
            # The read-decide-write on ``_stale_probe_after`` is a claim:
            # under the lock, exactly one caller at an expired-window
            # boundary observes it elapsed, re-arms it, and is admitted —
            # every concurrent loser sees the freshly armed window and
            # falls through to the fail-fast raise below.
            with _stale_probe_claim_lock:
                _probe_after = float(getattr(agent, "_stale_probe_after", 0.0) or 0.0)
                if _probe_after and _now >= _probe_after:
                    agent._stale_probe_after = _now + _interval
                    _admitted = True
                elif not _probe_after:
                    # First short-circuit past the threshold (the trip call,
                    # or an interrupt-counted overshoot that never ran this
                    # guard): arm the window so the probe fires only after a
                    # cooldown, never as an immediate re-wait of the stale
                    # timeout.
                    _probe_after = _now + _interval
                    agent._stale_probe_after = _probe_after
            if _admitted:
                logger.warning("Stale give-up breaker half-open: allowing one probe attempt "
                               "(streak=%d, next window in %.0fs).", _streak, _interval)
                return
            _extra = f" Next automatic probe in ~{max(0, int(_probe_after - _now))}s."
        except Exception:
            # Probe bookkeeping must never mask the breaker: fall back to the plain latch.
            logger.debug("stale probe window bookkeeping failed", exc_info=True)
    raise RuntimeError(
        "Provider has been unresponsive (no response received) for "
        f"{_streak} consecutive stale attempts — aborting this call to "
        "avoid an indefinite stall. Switch models or start a new session, then retry." + _extra
    )
