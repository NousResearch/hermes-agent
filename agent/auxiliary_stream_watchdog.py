"""No-progress watchdog for streamed chat-completions auxiliary calls (#100501).

The consumer blocks inside the SDK's chunk iterator, so a silent stream is only noticed when the
httpx read timeout (the full auxiliary request budget, >= 300s for compression) expires; the host
compression wait gives up first and the attempt dies with no fallback. The window is enforced from a
Timer that only ``shutdown()``s this attempt's socket: that is FD-safe from a stranger thread
(#70773), wakes the owner's blocked read, and the owner still closes the stream in its ``finally``
so the pool drops the dead connection instead of recycling it.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any, Optional


def _stream_socket(stream: Any) -> Any:
    """The raw socket under an SDK chunk stream (``stream.response`` is the ``httpx.Response``)."""
    from agent.agent_runtime_helpers import _socket_from_response
    response = getattr(stream, "response", None)
    return _socket_from_response(response) if response is not None else None


def _close_chunk_stream(chunks: Any, *, allow_aclose: bool = False) -> Any:
    """Best-effort ``close()`` (or ``aclose()``); returns a pending awaitable or None."""
    close_fn = getattr(chunks, "close", None) or (
        getattr(chunks, "aclose", None) if allow_aclose else None)
    if not callable(close_fn):
        return None
    try:
        result = close_fn()
    except Exception:
        return None
    return result if inspect.isawaitable(result) else None


class ChatStreamWatchdog:
    """Re-armed by substantive chunks only. Silence before the first token is bounded by
    *first_token_window* (the main loop's stale patience: reasoning models think silently for
    minutes); ``None`` leaves it to the request timeout (a local server's silent prefill)."""

    def __init__(self, stream: Any, window: float, *, first_token_window: Optional[float] = None):
        self.window = window
        self.first_token_window = first_token_window
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._deadline: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._generation = 0
        self._done = False
        self.saw_progress = False
        self.fired = False
        # Pin THIS attempt's socket now and retire the watchdog before the response hands its
        # connection back to the pool: a timer callback that already woke must never shut down a
        # pooled connection a later request has acquired (keepalive reuse).
        self._sock = _stream_socket(stream)
        self._fence_release(getattr(getattr(stream, "response", None), "stream", None))
        if first_token_window is not None:
            self._rearm(first_token_window)

    def _fence_release(self, body: Any) -> None:
        release = getattr(body, "close", None)
        if self._sock is None or not callable(release):
            return

        def fenced_release() -> None:
            with self._lock:
                self._done = True
            release()

        try:
            body.close = fenced_release
        except (AttributeError, TypeError):
            self._sock = None  # cannot fence the release: never shut down a socket we may not own

    def _rearm(self, window: float) -> None:
        # A pending timer re-checks the deadline when it fires; replace it only when the new
        # deadline is earlier (the first token swaps the long first-token window for the short one).
        previous, self._deadline = self._deadline, time.monotonic() + window
        if self._timer is None or (previous is not None and self._deadline < previous):
            if self._timer is not None:
                self._timer.cancel()
            self._schedule(window)

    def _schedule(self, delay: float) -> None:
        self._generation += 1
        self._timer = threading.Timer(max(delay, 0.0), self._fire, args=(self._generation,))
        self._timer.daemon = True
        self._timer.start()

    def progress(self) -> None:
        with self._lock:
            self.saw_progress = True
            if not self._done:
                self._rearm(self.window)

    def _fire(self, generation: int) -> None:
        # Decision and shutdown happen under the lock the fenced release takes, so a retired
        # attempt (or a timer superseded by ``_rearm`` after it already woke) never acts.
        with self._lock:
            if self._done or self._deadline is None or generation != self._generation:
                return
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                self._schedule(remaining)
                return
            self.fired = True
            if self._sock is not None:
                from agent.agent_runtime_helpers import _shutdown_socket
                _shutdown_socket(self._sock)

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
                f"Auxiliary chat stream produced no output within {self.first_token_window or 0:.1f}s "
                f"(no-progress timeout, {elapsed:.1f}s elapsed)")
        return TimeoutError(
            f"Auxiliary chat stream stalled: no new output for {self.window:.1f}s "
            f"({elapsed:.1f}s elapsed, timed out)")


def chat_stream_windows(client: Any, kwargs: dict, task: Optional[str]) -> "tuple[float, Optional[float]]":
    """(inter-chunk window, first-token window) for a streamed chat-completions attempt. Between
    chunks: the Codex guard's window (``auxiliary.<task>.no_progress_timeout``, 60s default). Before
    the first token: the main loop's cloud stale patience (context-scaled, reasoning-model floor), so
    a model thinking silently on a large prompt is not cut; local servers keep their silent prefill
    on the request timeout (None). Both are capped at the request timeout."""
    from agent.auxiliary_client import _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS, _get_task_no_progress_timeout
    from agent.chat_completion_helpers import _cloud_stale_timeout, _stream_env_stale_base
    from agent.model_metadata import is_local_endpoint
    window = _get_task_no_progress_timeout(task or "") or _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
    first = None
    if not is_local_endpoint(str(getattr(client, "base_url", "") or "")):
        first = max(window, _cloud_stale_timeout(_stream_env_stale_base()[0], kwargs))
    timeout = kwargs.get("timeout")
    if isinstance(timeout, (int, float)) and timeout > 0:
        window = min(window, float(timeout))
        first = min(first, float(timeout)) if first is not None else None
    return window, first


def consume_chat_stream(chunks: Any, acc: Any, no_progress: "Optional[tuple[float, Optional[float]]]") -> None:
    """Feed *chunks* into the accumulator *acc*. Under a ``(window, first_token_window)`` watchdog a
    silent stream raises TimeoutError, except after the terminal chunk: with finish_reason (and any
    usage) in hand a stall is teardown, so the completed response and its billed usage are kept."""
    if not no_progress:
        for chunk in chunks:
            acc.feed(chunk)
        return
    watchdog = ChatStreamWatchdog(chunks, no_progress[0], first_token_window=no_progress[1])
    try:
        for chunk in chunks:
            if acc.feed(chunk):
                watchdog.progress()
    except Exception as exc:
        if not watchdog.fired:
            raise
        if not acc.finish_reason:
            raise watchdog.timeout_error() from exc
    finally:
        watchdog.finish()
    if watchdog.fired and not acc.finish_reason:
        raise watchdog.timeout_error()
