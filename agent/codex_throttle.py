"""Codex outbound-call burst limiter (HEMP 2026-07-12).

Proactively paces ``openai-codex`` Responses API calls so a short burst cannot
exhaust the provider's 5-hour rolling request window — the failure mode behind
the 2026-07-12 13:14 429 credential cascade (been-main-2/3 + tom -> luigi).

The cap is a rolling-window request count: while the number of codex calls in
the trailing ``window_seconds`` stays below ``calls_per_hour`` the limiter is a
no-op; once at the cap the next call *waits* (never dropped, FIFO order
preserved) until the window drains.  Same total work, spread over time.

Async by construction
---------------------
The wait is ``await asyncio.sleep`` inside the gateway event loop, so a waiting
codex call never blocks the loop — every other channel keeps flowing.

The codex chokepoint (:func:`agent.codex_runtime.run_codex_stream`) runs
*synchronously* inside a gateway executor thread (``max_workers=10``), below the
event loop.  It reaches this async limiter through
:func:`throttle_codex_call_blocking`, which hops onto the gateway loop via
``asyncio.run_coroutine_threadsafe`` (the same cross-thread pattern already used
in ``gateway/run.py``).  Because the *wait* happens in the loop rather than the
thread, the loop is never blocked; the executor thread simply blocks on the
future until admitted, which is the desired per-call back-pressure.

Waiting in the loop (not spinning threads) is deliberate: blocking the 10 shared
executor threads on ``time.sleep`` during a burst would wedge every channel —
exactly the class of gateway freeze this project has fought before.

Scope / off switch
-------------------
When no gateway loop is registered (CLI, cron, tests) the bridge is a no-op, so
only the interactive gateway path is paced.  That matches the contract scope:
cron / subagents get their own caps (out of scope here).

``throttle.codex.enabled: false`` in config.yaml (or ``calls_per_hour <= 0``)
makes both :meth:`CodexRateLimiter.acquire` and the bridge complete-through with
zero interference.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Callable, Deque, Mapping, Optional

logger = logging.getLogger(__name__)

# Code defaults for the two knobs that stay out of config.yaml (the contract
# keeps the config surface to ``enabled`` + ``calls_per_hour`` only).
_DEFAULT_WINDOW_SECONDS = 3600.0
# Safety valve: bounds how long a single codex call may block its executor
# thread under *pathological sustained* overload.  Never reached by normal
# traffic (which stays under the cap and therefore never waits at all).
_DEFAULT_MAX_WAIT_SECONDS = 120.0


class CodexRateLimiter:
    """Async rolling-window limiter for codex Responses API calls.

    ``acquire`` admits immediately while under the cap; at the cap it awaits
    (drop-free, FIFO) until the oldest in-window call ages out.  The internal
    ``asyncio.Lock`` is held across the wait so admissions are strictly ordered
    — the "queue" is just callers awaiting the lock.
    """

    def __init__(
        self,
        *,
        calls_per_hour: int,
        enabled: bool = True,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        max_wait_seconds: Optional[float] = _DEFAULT_MAX_WAIT_SECONDS,
        time_func: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.calls_per_hour = int(calls_per_hour)
        self.window_seconds = float(window_seconds)
        self.max_wait_seconds = (
            None if max_wait_seconds is None else float(max_wait_seconds)
        )
        # Injectable clock/sleep so the probe can drive deterministic bursts
        # without real time.  Defaults: monotonic wall clock + asyncio.sleep.
        self._time: Callable[[], float] = time_func or time.monotonic
        self._sleep: Callable[[float], Any] = sleep_func or asyncio.sleep
        self._events: Deque[float] = collections.deque()  # admitted call times
        self._lock: Optional[asyncio.Lock] = None
        self._waiting = 0  # callers currently trying to get through (queue depth)

    @property
    def active(self) -> bool:
        """True when the limiter should actually gate (enabled + positive cap)."""
        return self.enabled and self.calls_per_hour > 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        events = self._events
        while events and events[0] <= cutoff:
            events.popleft()

    def window_count(self, now: Optional[float] = None) -> int:
        """Current number of admitted calls inside the rolling window."""
        if now is None:
            now = self._time()
        self._prune(now)
        return len(self._events)

    async def acquire(self) -> None:
        """Block (async) until this codex call may proceed.

        No-op when the limiter is inactive (off switch / non-positive cap).
        """
        if not self.active:
            return

        # Lazily bind the lock to whatever loop is running now.  Safe without a
        # guard: check-and-assign has no await between the two statements, so on
        # a single-threaded event loop it cannot interleave.
        if self._lock is None:
            self._lock = asyncio.Lock()

        self._waiting += 1
        try:
            # Strict FIFO: waiters acquire the lock in arrival order, so
            # admissions preserve call order (drop-free queueing).
            async with self._lock:
                waited = 0.0
                while True:
                    now = self._time()
                    self._prune(now)
                    if len(self._events) < self.calls_per_hour:
                        self._events.append(now)
                        return

                    # At the cap: wait until the oldest in-window call ages out.
                    wait = (self._events[0] + self.window_seconds) - now
                    if wait <= 0:
                        continue

                    if self.max_wait_seconds is not None:
                        remaining = self.max_wait_seconds - waited
                        if remaining <= 0:
                            # Safety valve: proceed drop-free rather than pin the
                            # executor thread indefinitely.  Only reachable under
                            # sustained > cap load; normal traffic never waits.
                            logger.warning(
                                "codex rate cap, waiting ~0.0s queue_depth=%d "
                                "(max_wait %.0fs reached, proceeding drop-free)",
                                self._waiting,
                                self.max_wait_seconds,
                            )
                            self._events.append(self._time())
                            return
                        wait = min(wait, remaining)

                    logger.info(
                        "codex rate cap, waiting ~%.1fs queue_depth=%d",
                        wait,
                        self._waiting,
                    )
                    await self._sleep(wait)
                    waited += wait
        finally:
            self._waiting -= 1


# --------------------------------------------------------------------------- #
# Process-wide singleton + gateway wiring
# --------------------------------------------------------------------------- #

_limiter: Optional[CodexRateLimiter] = None
_gateway_loop: Optional[asyncio.AbstractEventLoop] = None


def configure(
    *,
    enabled: bool,
    calls_per_hour: int,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    max_wait_seconds: Optional[float] = _DEFAULT_MAX_WAIT_SECONDS,
) -> CodexRateLimiter:
    """Install the process-wide codex limiter and return it."""
    global _limiter
    _limiter = CodexRateLimiter(
        calls_per_hour=calls_per_hour,
        enabled=enabled,
        window_seconds=window_seconds,
        max_wait_seconds=max_wait_seconds,
    )
    logger.info(
        "codex throttle configured: enabled=%s calls_per_hour=%s window=%.0fs "
        "max_wait=%s",
        _limiter.enabled,
        _limiter.calls_per_hour,
        _limiter.window_seconds,
        _limiter.max_wait_seconds,
    )
    return _limiter


def configure_from_config(cfg: Optional[Mapping[str, Any]]) -> CodexRateLimiter:
    """Build the limiter from the raw config mapping's ``throttle.codex`` block.

    Missing / malformed values fail safe to *disabled* so a config typo can
    never wedge codex calls.
    """
    throttle = (cfg or {}).get("throttle") or {}
    codex = throttle.get("codex") or {}

    enabled = bool(codex.get("enabled", False))
    try:
        calls_per_hour = int(codex.get("calls_per_hour", 0) or 0)
    except (TypeError, ValueError):
        calls_per_hour = 0
    try:
        window_seconds = float(codex.get("window_seconds", _DEFAULT_WINDOW_SECONDS))
    except (TypeError, ValueError):
        window_seconds = _DEFAULT_WINDOW_SECONDS
    raw_max_wait = codex.get("max_wait_seconds", _DEFAULT_MAX_WAIT_SECONDS)
    try:
        max_wait_seconds = None if raw_max_wait is None else float(raw_max_wait)
    except (TypeError, ValueError):
        max_wait_seconds = _DEFAULT_MAX_WAIT_SECONDS

    return configure(
        enabled=enabled,
        calls_per_hour=calls_per_hour,
        window_seconds=window_seconds,
        max_wait_seconds=max_wait_seconds,
    )


def set_gateway_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Register the gateway event loop the cross-thread bridge submits onto."""
    global _gateway_loop
    _gateway_loop = loop


def get_limiter() -> Optional[CodexRateLimiter]:
    return _limiter


def _running_loop_is(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def throttle_codex_call_blocking(timeout: Optional[float] = None) -> None:
    """Synchronous gate for the codex chokepoint (runs in an executor thread).

    Hops onto the gateway loop and blocks the calling thread until the async
    limiter admits the call.  No-op — completes through immediately — when:

      * the limiter is unconfigured / disabled / non-positive cap,
      * no gateway loop is registered (CLI, cron, tests), or
      * we are already running on the gateway loop thread (avoids deadlock).

    Fails open on any error: throttling must never break a real codex call.
    """
    limiter = _limiter
    if limiter is None or not limiter.active:
        return

    loop = _gateway_loop
    if loop is None or not loop.is_running():
        return
    if _running_loop_is(loop):
        return

    try:
        future = asyncio.run_coroutine_threadsafe(limiter.acquire(), loop)
        future.result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        logger.debug("codex throttle bridge skipped (fail-open): %s", exc)
