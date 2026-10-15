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
from types import SimpleNamespace
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


def chat_stream_windows(client: Any, kwargs: dict, task: Optional[str]) -> tuple[float, Optional[float]]:
    """(inter-chunk window, first-token window) for a streamed chat-completions attempt. Between
    chunks: the Codex guard's window (``auxiliary.<task>.no_progress_timeout``, 60s default). Before
    the first token: the main loop's cloud stale patience for the routed provider (its explicit
    ``providers.<id>.stale_timeout_seconds``, else context-scaled with the reasoning-model floor), so
    a model thinking silently on a large prompt is not cut; local servers keep their silent prefill
    on the request timeout (None). Both are capped at the request timeout."""
    from agent.auxiliary_client import (
        _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS, _RELAY_AUX_CALL_CONTEXT, _get_task_no_progress_timeout)
    from agent.chat_completion_helpers import _cloud_stale_timeout_for
    from agent.model_metadata import is_local_endpoint
    window = _get_task_no_progress_timeout(task or "") or _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
    first = None
    if not is_local_endpoint(str(getattr(client, "base_url", "") or "")):
        route = SimpleNamespace(provider=(_RELAY_AUX_CALL_CONTEXT.get() or {}).get("stream_provider") or "",
                                model=kwargs.get("model"))
        first = max(window, _cloud_stale_timeout_for(route, kwargs))
    timeout = kwargs.get("timeout")
    if isinstance(timeout, (int, float)) and timeout > 0:
        window = min(window, float(timeout))
        first = min(first, float(timeout)) if first is not None else None
    return window, first


def consume_chat_stream(chunks: Any, acc: Any, no_progress: Optional[tuple[float, Optional[float]]]) -> None:
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

_REASONING_DETAIL_TEXT_FIELDS = ("summary", "thinking", "content", "text")


class _ChatStreamAccumulator:
    """Shared per-chunk accumulation so sync and async aggregation cannot drift."""

    def __init__(self, model: str = "", total_ceiling: Optional[float] = None,
                 host_deadline: Optional[float] = None):
        self._started = time.monotonic()
        self._total_ceiling = total_ceiling
        # Absolute instant the waiting host gives up; checked alongside (not instead of) the
        # ceiling, and unaffected by pre-construction dispatch/TTFT.
        # Checked as well as (not instead of) the ceiling above: the ceiling still bounds callers with no
        # host deadline, and the host deadline is absolute, so it is unaffected by however long dispatch and
        # TTFT took before this accumulator was constructed. See #99692.
        self._host_deadline = host_deadline
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.reasoning_details: list[Any] = []
        self.tool_calls_acc: dict[int, dict[str, Any]] = {}
        self.finish_reason = self.usage = None
        self.resp_id = ""
        self.resp_model = model or ""

    def _check_deadlines(self) -> None:
        """Raise TimeoutError past the total ceiling or the host deadline."""
        now = time.monotonic()
        if self._total_ceiling is not None and (now - self._started) >= self._total_ceiling:
            raise TimeoutError(f"Auxiliary streamed call timed out after {self._total_ceiling:.0f}s "
                               "total ceiling (stream still open but over budget)")
        if self._host_deadline is not None and now >= self._host_deadline:
            raise TimeoutError("Auxiliary streamed call timed out at the host compression "
                               f"deadline after {time.monotonic() - self._started:.0f}s "
                               "(the caller already stopped waiting; streaming on would only "
                               "pin its session lease)")

    def _feed_reasoning_details(self, delta: Any) -> bool:
        """Collect ``reasoning_details`` (OpenRouter-style thinking); True only when a detail
        carries text, so structural/signed envelopes can't keep a stall alive."""
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details is None:
            model_extra = getattr(delta, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_details = model_extra.get("reasoning_details")
        if not isinstance(reasoning_details, list):
            return False
        made_progress = False
        for detail in reasoning_details:
            self.reasoning_details.append(detail)
            if isinstance(detail, dict) and any(
                isinstance(detail.get(f), str) and detail[f] for f in _REASONING_DETAIL_TEXT_FIELDS):
                made_progress = True
        return made_progress

    def _feed_tool_calls(self, delta: Any) -> bool:
        """Merge tool-call fragments by index; True when any fragment carried data."""
        made_progress = False
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            acc = self.tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": []})
            if getattr(tc, "id", None):
                acc["id"] = tc.id
                made_progress = True
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                    made_progress = True
                if getattr(fn, "arguments", None):
                    acc["arguments"].append(fn.arguments)
                    made_progress = True
        return made_progress

    def feed(self, chunk: Any) -> bool:
        # Every frame records transport timing (TTFP); only a substantive payload ticks the
        # forward-progress hook that keeps compression alive (and is reported as True).
        from agent.auxiliary_client import _notify_aux_timing_response, _notify_aux_progress
        _notify_aux_timing_response()
        self._check_deadlines()
        self.resp_id = getattr(chunk, "id", None) or self.resp_id
        self.resp_model = getattr(chunk, "model", None) or self.resp_model
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            self.usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return False
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            return False
        made_progress = False
        from agent.message_content import flatten_message_text

        piece = flatten_message_text(getattr(delta, "content", None), sep="")
        if piece:
            self.content_parts.append(piece)
            made_progress = True
        reasoning_piece = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning_piece is None and isinstance(getattr(delta, "model_extra", None), dict):
            reasoning_piece = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")
        reasoning_piece = flatten_message_text(reasoning_piece, sep="")
        if reasoning_piece:
            self.reasoning_parts.append(reasoning_piece)
            made_progress = True
        # Evaluate both unconditionally: they accumulate state, not just progress.
        made_progress |= self._feed_reasoning_details(delta)
        made_progress |= self._feed_tool_calls(delta)
        if made_progress:
            _notify_aux_progress()
        return made_progress

    def finish(self) -> Any:
        tool_calls = None
        if self.tool_calls_acc:
            tool_calls = [
                SimpleNamespace(id=acc["id"], type="function", function=SimpleNamespace(
                    name=acc["name"], arguments="".join(acc["arguments"])))
                for _idx, acc in sorted(self.tool_calls_acc.items())]
        message = SimpleNamespace(
            role="assistant", content="".join(self.content_parts), tool_calls=tool_calls,
            reasoning="".join(self.reasoning_parts) or None,
            reasoning_details=self.reasoning_details or None,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason=self.finish_reason or "stop")
        return SimpleNamespace(id=self.resp_id, model=self.resp_model, object="chat.completion",
                               choices=[choice], usage=self.usage)
