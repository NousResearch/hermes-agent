"""
Idle auto-compression timer — triggers context compression after user inactivity.

Inspired by OpenClacky's IdleCompressionTimer. When the user steps away,
the timer fires a compression call that reuses the existing prompt cache
(Insert-then-Compress pattern). The delay is kept under the 5-minute cache
TTL so the compression call itself hits the cached prefix.

Usage in CLI and gateway:
    timer = IdleCompressionTimer(agent, delay_seconds=300, min_tokens=20000)
    timer.start()   # after each agent run
    timer.cancel()  # on new user input
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from run_agent import AIAgent

logger = logging.getLogger(__name__)

# Defaults — overridable via config.yaml compression.idle.*
DEFAULT_IDLE_DELAY_SECONDS = 300  # 5 minutes, under cache TTL
DEFAULT_MIN_TOKENS = 20_000      # skip small sessions


class IdleCompressionTimer:
    """Background timer that triggers context compression after inactivity.

    Thread-safe. The timer thread only schedules work; the actual compression
    API call runs on a separate thread so the timer can be cancelled quickly.

    Attributes:
        delay_seconds: Seconds of inactivity before triggering.
        min_tokens: Minimum estimated token count — sessions below this
                    are skipped (compressing a 5K-token session is waste).
    """

    def __init__(
        self,
        agent: AIAgent,
        delay_seconds: float = DEFAULT_IDLE_DELAY_SECONDS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        on_compress: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._agent = agent
        self._delay_seconds = delay_seconds
        self._min_tokens = min_tokens
        self._on_compress = on_compress

        self._timer_thread: Optional[threading.Thread] = None
        self._compress_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    # ── public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start (or restart) the idle countdown.

        Cancels any existing timer first, then waits ``delay_seconds``
        before triggering compression on a background thread.
        """
        self.cancel()

        self._cancelled.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            name="idle-compression-timer",
            daemon=True,
        )
        self._timer_thread.start()

    def cancel(self) -> None:
        """Cancel the idle countdown and any in-flight compression.

        Safe to call multiple times. Blocks briefly (up to 2s) until
        the compression thread acknowledges the cancellation.
        """
        self._cancelled.set()

        with self._lock:
            timer = self._timer_thread
            compress = self._compress_thread
            self._timer_thread = None
            self._compress_thread = None

        # Wait for compression to finish rolling back (non-blocking join)
        if compress is not None and compress.is_alive():
            compress.join(timeout=2.0)

        # Timer thread will exit on its own (daemon, or sees cancelled flag)

    @property
    def active(self) -> bool:
        """True if the timer or compression is currently running."""
        with self._lock:
            return (
                (self._timer_thread is not None and self._timer_thread.is_alive())
                or (self._compress_thread is not None and self._compress_thread.is_alive())
            )

    @property
    def compressing(self) -> bool:
        """True only when the compression API call is in flight."""
        with self._lock:
            c = self._compress_thread
            return c is not None and c.is_alive()

    @property
    def delay_seconds(self) -> float:
        return self._delay_seconds

    @delay_seconds.setter
    def delay_seconds(self, value: float) -> None:
        self._delay_seconds = max(10.0, float(value))

    @property
    def min_tokens(self) -> int:
        return self._min_tokens

    @min_tokens.setter
    def min_tokens(self, value: int) -> None:
        self._min_tokens = max(1000, int(value))

    # ── internals ───────────────────────────────────────────────────────

    def _timer_loop(self) -> None:
        """Countdown loop. When time elapses, spawn compression thread."""
        # Wait with periodic wake-up so we can detect cancellation
        remaining = self._delay_seconds
        tick = 1.0  # check every second
        while remaining > 0 and not self._cancelled.is_set():
            sleep_for = min(tick, remaining)
            self._cancelled.wait(timeout=sleep_for)
            remaining -= sleep_for

        if self._cancelled.is_set():
            logger.debug("Idle compression timer cancelled during countdown")
            return

        # Check token floor
        est_tokens = self._estimate_tokens()
        if est_tokens < self._min_tokens:
            logger.debug(
                "Idle compression skipped: %d tokens < %d min",
                est_tokens, self._min_tokens,
            )
            return

        # Check compression is enabled
        if not getattr(self._agent, "compression_enabled", True):
            logger.debug("Idle compression skipped: compression disabled")
            return

        # Spawn compression work thread
        with self._lock:
            if self._cancelled.is_set():
                return
            self._compress_thread = threading.Thread(
                target=self._run_compression,
                name="idle-compression-work",
                daemon=True,
            )
            self._compress_thread.start()

    def _run_compression(self) -> None:
        """Execute the compression call on a background thread."""
        success = False
        try:
            logger.info("Idle compression started")
            self._agent._emit_status(
                "💤 Idle detected — compacting context to save tokens..."
            )

            # Trigger compression via the agent's existing path
            messages = self._agent.messages
            sp = self._agent._build_system_prompt("")
            compressed_msgs, new_sp = self._agent._compress_context(
                messages, sp, force=True,
            )

            if compressed_msgs is not messages:  # actual compression happened
                self._agent.messages = compressed_msgs
                # Reset the cached system prompt so next turn rebuilds
                self._agent._invalidate_system_prompt()
                success = True
                logger.info("Idle compression completed successfully")

        except Exception as exc:
            logger.warning("Idle compression failed: %s", exc, exc_info=True)
            if hasattr(self._agent, "_emit_warning"):
                self._agent._emit_warning(
                    f"⚠ Idle compression failed: {exc}. "
                    "Context may be large — use /compress to retry."
                )
        finally:
            if self._on_compress:
                try:
                    self._on_compress(success)
                except Exception:
                    pass

            with self._lock:
                self._compress_thread = None

    def _estimate_tokens(self) -> int:
        """Rough token estimate of current conversation."""
        try:
            from agent.model_metadata import estimate_messages_tokens_rough
            return estimate_messages_tokens_rough(self._agent.messages)
        except Exception:
            # Fallback: rough char-based estimate
            total = 0
            for m in self._agent.messages:
                content = m.get("content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, str):
                            total += len(p)
                        elif isinstance(p, dict):
                            total += len(p.get("text", ""))
            return total // 4  # ~4 chars per token
