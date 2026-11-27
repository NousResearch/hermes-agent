"""Opt-in total-lifetime cap for one streaming response attempt.

The stale-stream detector bounds the gap BETWEEN chunks and resets on every
chunk, so a stream that keeps yielding data but never finishes (a gateway
trickling chunks, a runaway generation) is never cut. ``agent.stream_max_lifetime``
caps the wall-clock time since the current attempt started; it never resets on
chunk arrival. Default ``0`` = off, so a
legitimately long reasoning stream is never cut unless the user asks for it.
"""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)


def resolve_stream_max_lifetime(stale_timeout: float | None) -> float:
    """Cap in seconds, ``0`` when disabled. Floored at a finite stale timeout so the
    cap can never cut a stream the stale detector would still be waiting on."""
    configured = 0.0
    try:
        from hermes_cli.config import load_config_readonly

        agent_cfg = (load_config_readonly() or {}).get("agent")
        value = agent_cfg.get("stream_max_lifetime") if isinstance(agent_cfg, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            configured = float(value)
    except Exception:  # unreadable config = the documented default (off)
        logger.debug("stream_max_lifetime: config unreadable, cap off", exc_info=True)
    if configured <= 0 or not math.isfinite(configured):
        return 0.0
    if stale_timeout is not None and math.isfinite(stale_timeout):
        return max(configured, stale_timeout)
    return configured


class StreamLifetimeExceeded(TimeoutError):
    """A stream attempt cut by ``agent.stream_max_lifetime``: the stream opened and was still
    delivering chunks, so the user-facing error names the setting, not the endpoint."""

    def __init__(self, elapsed: float, cap: float):
        self.cap = cap
        super().__init__(
            f"Response still streaming after {elapsed:.0f}s, cut by agent.stream_max_lifetime ({cap:g}s). "
            f"Raise agent.stream_max_lifetime, or set it to 0 to turn the cap off, if this model "
            f"legitimately streams this long.")


class StreamLifetime:
    """Per-call clock: restarts whenever a new stream attempt is dispatched, so each
    retry gets the full cap, and after each kill (the killed attempt may unwind slowly)."""

    def __init__(self, cap: float):
        self.cap = cap
        self.attempt = None
        self.started = time.time()

    def exceeded(self, attempt) -> float | None:
        """Elapsed seconds when the cap is exceeded for ``attempt``, else ``None``."""
        if not self.cap:
            return None
        now = time.time()
        if attempt != self.attempt:
            self.attempt, self.started = attempt, now
            return None
        elapsed = now - self.started
        if elapsed <= self.cap:
            return None
        self.started = now
        return elapsed

