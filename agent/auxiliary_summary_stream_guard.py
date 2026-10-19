"""Runaway-output guard and abort errors for streamed auxiliary calls (``_ChatStreamAccumulator``).

The summary call deliberately carries NO max_tokens wire cap: Anthropic/NIM forward it and a hard
cap truncates summaries (thinking models burn it on reasoning); see the rule comment at
context_compressor.py's _SUMMARY_INPUT_MAX_CHARS. A summary stream stuck in a repetition loop
therefore keeps producing chunks, so neither the no-progress watchdog nor the stall checks fire,
and it runs until the compression ceiling. The guard bounds the STREAM instead (never the wire
request). Counting reasoning and content chars (~4 chars/token):

- past ``_repetition_check_from_chars()`` the tail of each stream is checked every
  ``_REPETITION_CHECK_EVERY_CHARS`` with ``repetition_guard.is_runaway_repetition`` (the detector
  the main loop uses), and a verbatim loop stops the stream;
- anything at all stops at ``_RUNAWAY_HARD_TOKENS``, well past any real summary.

Size alone is not a loop: a single bound of 4 x _SUMMARY_TOKENS_CEILING tokens aborted two
legitimate summaries of a ~100K-token conversation by a thinking model at exactly that size.

Every stop raises TimeoutError phrased "timed out", so ``_is_timeout_error`` matches and the
attempt routes through the existing failed-attempt fallback and cooldown machinery. The
accumulator's total-ceiling and host-deadline errors are built here too, so every abort of a
summary stream (runaway or deadline) is logged at INFO with which check fired and a short tail
of the content and reasoning streams. Other tasks get the same errors without the log line.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_RUNAWAY_HARD_TOKENS = 100_000
_REPETITION_CHECK_EVERY_CHARS = 16_000
_REPETITION_TAIL_CHARS = 32_000
# Tail of each stream (content / reasoning) logged at INFO when a summary stream is aborted.
_ABORT_LOG_TAIL_CHARS = 600
_TIMED_OUT = "Auxiliary streamed call timed out on runaway output: "
_NO_CAP = "(the summary call carries no max_tokens wire cap by design, so the STREAM is bounded instead)"


def _repetition_check_from_chars() -> int:
    """Stream size (chars) from which the summary stream's tail is checked for a loop."""
    from agent.context_compressor import _SUMMARY_TOKENS_CEILING

    return 4 * _SUMMARY_TOKENS_CEILING * 4


def summary_runaway_output_bound_chars(task: Optional[str]) -> Optional[int]:
    """HARD output bound (chars) for the compression summary stream; None for other tasks."""
    if task != "compression":
        return None
    return _RUNAWAY_HARD_TOKENS * 4


def tail_text(parts: list[str], limit: int) -> str:
    """The last ``limit`` chars of ``"".join(parts)`` without joining the whole stream."""
    tail: list[str] = []
    size = 0
    for piece in reversed(parts):
        tail.append(piece)
        size += len(piece)
        if size >= limit:
            break
    return "".join(reversed(tail))[-limit:]


class SummaryStreamGuard:
    """Per-stream state over the accumulator's own ``content_parts`` / ``reasoning_parts`` lists.
    With ``bound_chars=None`` (every task but the compression summary) ``check`` does nothing and
    ``abort`` returns a plain TimeoutError without logging."""

    def __init__(self, bound_chars: Optional[int], content_parts: list[str], reasoning_parts: list[str]):
        self._started = time.monotonic()
        self.bound_chars = bound_chars
        self.chars = 0
        self._streams = (("content", content_parts), ("reasoning", reasoning_parts))
        self._seen = [0, 0]  # parts already counted, per stream
        self._next_repetition_check = _repetition_check_from_chars() if bound_chars is not None else None

    def abort(self, kind: str, message: str) -> TimeoutError:
        """The error to raise. For the summary stream, first log which check fired plus a short
        tail of each stream, so a run's log shows WHY a ``summary_generation_aborted`` happened
        (loop vs size vs time) and what the model was emitting."""
        if self.bound_chars is not None:
            (_, content), (_, reasoning) = self._streams
            logger.info(
                "Compression summary stream aborted: check=%s after %.1fs, %s content + %s "
                "reasoning chars; reason: %s; content tail=%r; reasoning tail=%r",
                kind, time.monotonic() - self._started, f"{sum(len(p) for p in content):,}",
                f"{sum(len(p) for p in reasoning):,}", message,
                tail_text(content, _ABORT_LOG_TAIL_CHARS), tail_text(reasoning, _ABORT_LOG_TAIL_CHARS),
            )
        return TimeoutError(message)

    def total_ceiling_error(self, ceiling: float) -> TimeoutError:
        return self.abort("total_ceiling", f"Auxiliary streamed call timed out after {ceiling:.0f}s "
                          "total ceiling (stream still open but over budget)")

    def host_deadline_error(self, elapsed: float) -> TimeoutError:
        return self.abort("host_deadline", "Auxiliary streamed call timed out at the host compression "
                          f"deadline after {elapsed:.0f}s "
                          "(the caller already stopped waiting; streaming on would only "
                          "pin its session lease)")

    def check(self) -> None:
        """Count new parts; raise past the hard bound, or on a verbatim loop past the soft size."""
        if self.bound_chars is None:
            return
        for i, (_, parts) in enumerate(self._streams):
            self.chars += sum(len(p) for p in parts[self._seen[i]:])
            self._seen[i] = len(parts)
        if self.chars > self.bound_chars:
            raise self.abort("hard_size_bound", (
                f"{_TIMED_OUT}the stream produced {self.chars:,} chars, past the "
                f"{self.bound_chars:,}-char hard runaway bound {_NO_CAP}"))
        if self.chars < self._next_repetition_check:
            return
        self._next_repetition_check = self.chars + _REPETITION_CHECK_EVERY_CHARS
        from agent.repetition_guard import is_runaway_repetition

        for label, parts in self._streams:
            if parts and is_runaway_repetition(tail_text(parts, _REPETITION_TAIL_CHARS)):
                raise self.abort(f"repetition_loop:{label}", (
                    f"{_TIMED_OUT}the summary {label} stream is a repetition loop after "
                    f"{self.chars:,} chars {_NO_CAP}"))
