"""No-progress watchdog for streamed chat-completions auxiliary calls (#100501).

The consumer blocks inside the SDK's chunk iterator, so a silent stream is only noticed when the
httpx read timeout (the full auxiliary request budget, >= 300s for compression) expires; the host
compression wait gives up first and the attempt dies with no fallback. The window is enforced from a
Timer that only ``shutdown()``s this attempt's socket: that is FD-safe from a stranger thread
(#70773), wakes the owner's blocked read, and the owner still closes the stream in its ``finally``
so the pool drops the dead connection instead of recycling it.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional


def _stream_socket(stream: Any) -> Any:
    """The raw socket under an SDK chunk stream (``stream.response`` is the ``httpx.Response``)."""
    from agent.agent_runtime_helpers import _socket_from_response
    response = getattr(stream, "response", None)
    return _socket_from_response(response) if response is not None else None


class ChatStreamWatchdog:
    """Re-armed by substantive chunks only; ``first_token=False`` leaves time-to-first-token to the
    request timeout (a local server's prefill of a large prompt is silent but alive)."""

    def __init__(self, stream: Any, window: float, *, first_token: bool = True):
        self._stream = stream
        self.window = window
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._deadline: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._done = False
        self.saw_progress = False
        self.fired = False
        if first_token:
            self._rearm()

    def _rearm(self) -> None:
        self._deadline = time.monotonic() + self.window
        if self._timer is None:
            self._schedule(self.window)

    def _schedule(self, delay: float) -> None:
        self._timer = threading.Timer(max(delay, 0.0), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def progress(self) -> None:
        with self._lock:
            self.saw_progress = True
            if not self._done:
                self._rearm()

    def _fire(self) -> None:
        with self._lock:
            if self._done or self._deadline is None:
                return
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                self._schedule(remaining)
                return
            self.fired = True
        sock = _stream_socket(self._stream)
        if sock is not None:
            from agent.agent_runtime_helpers import _shutdown_socket
            _shutdown_socket(sock)

    def finish(self) -> None:
        with self._lock:
            self._done = True
            timer = self._timer
        if timer is not None:
            timer.cancel()

    def timeout_error(self) -> TimeoutError:
        """Zero-output stalls say "no-progress timeout" (same-provider retry stays allowed, see
        ``_should_skip_same_provider_retry``); a mid-stream stall goes straight to fallback."""
        elapsed = time.monotonic() - self._started
        if not self.saw_progress:
            return TimeoutError(
                f"Auxiliary chat stream produced no output within {self.window:.1f}s "
                f"(no-progress timeout, {elapsed:.1f}s elapsed)")
        return TimeoutError(
            f"Auxiliary chat stream stalled: no new output for {self.window:.1f}s "
            f"({elapsed:.1f}s elapsed, timed out)")
