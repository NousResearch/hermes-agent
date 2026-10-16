"""Cross-turn stale-call circuit breaker (#58962).

Extracted from :mod:`agent.chat_completion_helpers` to keep both files under the
2000-line ratchet cap. The parent module re-exports every public symbol so
existing callers are unaffected.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Optional

from utils import env_float, env_int

import logging

logger = logging.getLogger(__name__)


# ── StaleGiveupError ────────────────────────────────────────────────────
# Raised by _check_stale_giveup when the consecutive-stale streak reaches the
# give-up threshold. Every consumer that needs to detect this specific case
# checks ``isinstance(exc, StaleGiveupError)`` or the class name, never a
# substring match on the message.


class StaleGiveupError(RuntimeError):
    """Raised by ``_check_stale_giveup`` when the consecutive-stale streak
    reaches the give-up threshold. Every consumer that needs to detect this
    specific case checks ``isinstance(exc, StaleGiveupError)`` or the class
    name, never a substring match on the message."""


# ── Cross-turn stale-call circuit breaker (#58962) ─────────────────────
# A session wedged against an unresponsive provider would otherwise hit the
# stale detector on every call forever. ``agent._consecutive_stale_streams``
# is bumped on every stale kill and reset only when a call completes or the
# provider is swapped (switch_model / try_activate_fallback /
# restore_primary_runtime — the streak measured the OLD provider). Past the
# give-up threshold, calls abort immediately with an actionable error.


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


_INTERRUPTED_WAIT_STALE_SECONDS = 30.0


def _record_interrupted_provider_wait(agent, elapsed: float, *, response_started: bool) -> bool:
    """Count a user-aborted pre-response stall toward the stale breaker: past the
    wait-notice interval an interrupt is evidence of an unresponsive attempt.
    Mid-response and early interrupts stay neutral."""
    if response_started or elapsed < _INTERRUPTED_WAIT_STALE_SECONDS:
        return False
    _bump_stale_streak(agent)
    logger.warning("Interrupted provider wait counted as stale after %.0fs with no output; "
        "consecutive stale attempts=%d.", elapsed, _stale_streak(agent))
    return True


def _report_stale_nonstream_kill(agent, api_kwargs: dict, elapsed: float, stale_timeout: float, *,
    inline: bool = False, hint: Optional[str] = None) -> None:
    """Log + status message for a stale non-streaming kill, shared by the worker
    poll loop and the inline ``direct_api_call`` watchdog (their kill/state
    sequences differ deliberately: different locking models)."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    model = api_kwargs.get("model", "unknown")
    logger.warning("%son-streaming API call stale for %.0fs (threshold %.0fs). "
        "model=%s context=~%s tokens. Killing connection.", "Inline n" if inline else "N", elapsed,
        stale_timeout, model, f"{estimate_request_context_tokens(api_kwargs):,}")
    try:
        agent._buffer_diagnostic_status(
            f"⚠️ No response from provider for {int(elapsed)}s (non-streaming, model: {model}). {hint or 'Aborting call.'}")
    except Exception:
        logger.debug("stale status buffering failed", exc_info=True)


def _touch_stale_kill_activity(agent, elapsed: float) -> None:
    try:
        agent._touch_activity(f"stale non-streaming call killed after {int(elapsed)}s")
    except Exception:
        logger.debug("stale activity touch failed", exc_info=True)


def _check_stale_giveup(agent) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait."""
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup > 0 and _streak >= _giveup:
        raise StaleGiveupError(
            "Provider has been unresponsive (no response received) for "
            f"{_streak} consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Send another message to try again, "
            "switch models with /model, or start a new session with /new."
        )
