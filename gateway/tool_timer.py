"""Tool-timer animation state machine for native stream bubbles.

Extracted from ``stream_consumer.py`` for bounded ownership.  The
``ToolTimerMixin`` provides the per-tool elapsed-time spinner that ticks
every second in the WeCom native stream bubble, plus the completion
history overlay.

Public symbols re-exported for callers:
- ``_TIMER_TICK`` sentinel
- ``_SPINNER_CHARS``
- ``_TOOL_NAME_RE``, ``_parse_tool_name``

Host requirements (must be present on ``self``):
- ``_use_native_streaming: bool``
- ``_native_stream_opened: bool``
- ``_queue: queue.Queue``  (stdlib thread-safe queue)
- ``_tool_progress_lines: list[str]``
- ``_tool_progress_active: bool``
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel for the tool-timer tick — a no-op wake-up for the drain loop.
# The tick callback already updated ``_tool_progress_lines`` and set
# ``_tool_progress_active``; this just unblocks the loop so it pushes a frame.
_TIMER_TICK = object()

# Braille-dot spinner characters for the tool timer animation.
_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Pattern to extract a tool name from progress lines emitted by run.py.
# Examples: "🔧 Running terminal..." → "terminal"
#           "⚙️ Calling web_search..." → "web_search"
#           "🔍 Searching..." → "Searching"  (fallback: first word after emoji)
_TOOL_NAME_RE = re.compile(
    r"^[^\w]*"              # leading emoji / punctuation
    r"(?:Running|Calling|Using)?\s*"  # optional verb
    r"(\w+)",               # capture the tool name
    re.UNICODE,
)


def _parse_tool_name(line: str) -> str:
    """Extract a tool name from a progress line like '🔧 Running terminal...'."""
    m = _TOOL_NAME_RE.search(line)
    return m.group(1) if m else "tool"


class ToolTimerMixin:
    """Mixin providing tool-timer animation for native stream bubbles.

    Initialise timer state by calling ``_init_tool_timer()`` from the host
    ``__init__``.  The host must also provide the attributes listed in the
    module docstring.
    """

    def _init_tool_timer(self) -> None:
        """Initialise tool-timer mutable state.  Call from host ``__init__``."""
        self._tool_timer_handle: Optional[asyncio.TimerHandle] = None
        self._tool_timer_loop: Optional[asyncio.AbstractEventLoop] = None
        self._tool_start_times: dict[str, float] = {}  # key -> monotonic start
        self._tool_timer_labels: dict[str, str] = {}  # key -> original progress line
        self._tool_timer_tick_count: int = 0  # for spinner rotation
        self._timer_lock = threading.Lock()  # guards ALL timer mutable state
        self._tool_completed_lines: list[str] = []  # completed tool history (max 5)

    # ── Public API ───────────────────────────────────────────────────────

    def on_tool_progress(self, line: str, tool_call_id: str | None = None) -> None:
        """Inject a tool-progress status line into the native stream bubble.

        Thread-safe (called from agent worker thread via queue.Queue). Only
        effective when native streaming is active for this consumer.

        The line is displayed as an overlay until the next text delta arrives,
        at which point real content overwrites the tool-progress lines.

        Also starts the tool-timer animation (1s ticks with spinner + elapsed)
        if not already running.

        ``tool_call_id``, when provided, is used as the dict key instead of
        the parsed tool name.  This allows two concurrent calls to the same
        tool (e.g. two ``terminal`` invocations) to track independently.
        """
        from gateway.stream_consumer import _TOOL_PROGRESS
        if line:
            self._queue.put((_TOOL_PROGRESS, line))
            # Start/join the timer for this tool, preserving the original
            # progress line as the display label for animated ticks.
            tool_name = _parse_tool_name(line)
            key = tool_call_id if tool_call_id is not None else tool_name
            # Don't clear other running tools — they may be parallel.
            # on_tool_completed() handles moving finished tools to
            # _tool_completed_lines when tool.completed fires.
            with self._timer_lock:
                self._tool_timer_labels[key] = line
            self._start_tool_timer(key)

    def on_tool_completed(self, tool_name: str, duration: float, tool_call_id: str | None = None) -> None:
        """Record a completed tool in the history overlay.

        Thread-safe: called from the agent worker thread.

        ``tool_call_id``, when provided, is used as the dict key for looking
        up the matching timer entry.  Falls back to *tool_name* when absent.
        """
        key = tool_call_id if tool_call_id is not None else tool_name
        with self._timer_lock:
            label = self._tool_timer_labels.pop(key, tool_name)
            self._tool_start_times.pop(key, None)
            completion_line = f"✓ {label} ({int(duration)}s)"
            self._tool_completed_lines.append(completion_line)
            # Keep max 5 entries
            if len(self._tool_completed_lines) > 5:
                self._tool_completed_lines = self._tool_completed_lines[-5:]
        self._tool_progress_active = True
        self._queue.put(_TIMER_TICK)

    def on_llm_thinking(self, label: "str | None" = None) -> None:
        """Signal that an LLM API call has started — show thinking animation.

        Thread-safe: called from the agent worker thread.  Only activates
        when the native stream is already open (the bubble is visible).

        ``label`` is an optional display string (e.g. "claude-4.6-opus (API call #3)")
        shown alongside the thinking timer for richer context.
        """
        if not self._use_native_streaming:
            return
        if not self._native_stream_opened:
            return
        # LLM thinking means all tools are done — move remaining tool entries
        # to completed history, then start the thinking timer.
        with self._timer_lock:
            stale = [k for k in self._tool_start_times if k != "_thinking"]
            now = time.monotonic()
            for k in stale:
                start = self._tool_start_times.pop(k)
                tool_label = self._tool_timer_labels.pop(k, k)
                elapsed = int(now - start)
                completion_line = f"✓ {tool_label} ({elapsed}s)"
                self._tool_completed_lines.append(completion_line)
            # Trim to max 5 completed entries
            if len(self._tool_completed_lines) > 5:
                self._tool_completed_lines = self._tool_completed_lines[-5:]
            if "_thinking" not in self._tool_start_times:
                self._tool_start_times["_thinking"] = time.monotonic()
            # Store the label for display in _tool_timer_tick
            if label:
                self._tool_timer_labels["_thinking"] = label
        # Arm the timer if not already running
        with self._timer_lock:
            need_arm = self._tool_timer_handle is None and self._tool_timer_loop is not None
        if need_arm:
            self._tool_timer_loop.call_soon_threadsafe(self._arm_tool_timer)

    # ── Frame composition helper ─────────────────────────────────────────

    def _compose_tool_overlay(self) -> list[str]:
        """Return tool-progress lines including completed history.

        Called by the host's ``_compose_frame_content`` to build the tool
        overlay section of the stream frame.
        """
        tool_lines = self._tool_progress_lines
        if not tool_lines:
            with self._timer_lock:
                if self._tool_completed_lines:
                    tool_lines = list(self._tool_completed_lines)
        return tool_lines

    # ── Internal timer machinery ─────────────────────────────────────────

    def _start_tool_timer(self, tool_name: str) -> None:
        """Start (or join) the 1-second tool-timer animation.

        Records *tool_name*'s start time and arms the periodic tick if not
        already running.  Only arms when ``_use_native_streaming`` is True
        (non-native platforms don't benefit from sub-second bubble updates).

        Thread-safe: called from the agent worker thread.  Uses
        call_soon_threadsafe to schedule the first tick on the event loop.
        """
        if not self._use_native_streaming:
            return
        with self._timer_lock:
            if tool_name not in self._tool_start_times:
                self._tool_start_times[tool_name] = time.monotonic()
            # Arm the periodic tick if not already running.
            # Use call_soon_threadsafe because this method is called from the
            # agent worker thread, not the event loop thread.
            need_arm = self._tool_timer_handle is None and self._tool_timer_loop is not None
        if need_arm:
            self._tool_timer_loop.call_soon_threadsafe(self._arm_tool_timer)

    def _arm_tool_timer(self) -> None:
        """Arm the 1s periodic tick.  Must run on the event loop thread."""
        with self._timer_lock:
            if self._tool_timer_handle is None:
                self._tool_timer_handle = self._tool_timer_loop.call_later(
                    1.0, self._tool_timer_tick,
                )
                logger.debug("[timer] armed")

    def _stop_tool_timer(self) -> None:
        """Cancel the tool-timer animation and clear associated state."""
        with self._timer_lock:
            was_running = self._tool_timer_handle is not None
            if self._tool_timer_handle is not None:
                self._tool_timer_handle.cancel()
                self._tool_timer_handle = None
            self._tool_start_times.clear()
            self._tool_timer_labels.clear()
            self._tool_completed_lines.clear()
            self._tool_timer_tick_count = 0
        logger.debug("[timer] stopped (was_running=%s)", was_running)

    def _tool_timer_tick(self) -> None:
        """Periodic tick: rebuild tool-progress lines with spinner + elapsed.

        Runs on the asyncio event loop thread (via ``call_later``).
        """
        with self._timer_lock:
            if not self._tool_start_times:
                # All tools cleared — don't re-arm
                self._tool_timer_handle = None
                return

            self._tool_timer_tick_count += 1
            logger.debug("[timer] tick #%d, entries=%d", self._tool_timer_tick_count, len(self._tool_start_times))
            now = time.monotonic()
            lines: list[str] = list(self._tool_completed_lines)  # completed history first
            for tool_name, start in self._tool_start_times.items():
                elapsed = int(now - start)
                spinner = _SPINNER_CHARS[self._tool_timer_tick_count % len(_SPINNER_CHARS)]
                if tool_name == "_thinking":
                    thinking_label = self._tool_timer_labels.get("_thinking")
                    if thinking_label:
                        lines.append(f"{spinner} 💭 {thinking_label} ({elapsed}s)")
                    else:
                        lines.append(f"{spinner} 💭 Thinking ({elapsed}s)")
                else:
                    # Use the original progress line (full summary) as label,
                    # stripping any trailing "..." and appending elapsed time.
                    label = self._tool_timer_labels.get(tool_name, f"{tool_name}...")
                    lines.append(f"{spinner} {label} ({elapsed}s)")

            self._tool_progress_lines = lines
        self._tool_progress_active = True
        # Wake the drain loop so it pushes a frame
        self._queue.put(_TIMER_TICK)

        # Re-arm for the next tick
        with self._timer_lock:
            if self._tool_timer_loop is not None:
                self._tool_timer_handle = self._tool_timer_loop.call_later(
                    1.0, self._tool_timer_tick,
                )
