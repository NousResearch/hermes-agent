"""Streaming output repetition guard.

The guard is deliberately small and deterministic: it watches visible text
while it streams and raises once a non-trivial line repeats too many times.
This catches local-model runaway tails before the user watches minutes of
duplicated prose.
"""

from __future__ import annotations

from collections import Counter, deque
import math
import os
import re
from typing import Any, Optional


FAILED_REPETITION_LOOP = "FAILED_REPETITION_LOOP"

# Marker appended when a degenerate tail is cut from a stored message so the
# next reader (user, model, compaction) knows the text was bounded on purpose.
REPETITION_CUT_MARKER = (
    "[System note: output cut here — the model degenerated into repeating "
    "the same text.]"
)


def normalize_guard_line(line: str, min_line_chars: int = 40) -> str:
    """Normalize one line for repetition counting ('' = not countable)."""
    line = re.sub(r"\s+", " ", (line or "").strip())
    if len(line) < min_line_chars:
        return ""
    if line in {"```", "---", "***"}:
        return ""
    if set(line) <= {"-", "|", ":", " "}:
        return ""
    return line


def _smallest_period(text: str) -> int:
    """Return the smallest exact period length for ``text`` using KMP."""
    n = len(text)
    if n <= 1:
        return n
    failure = [0] * n
    k = 0
    for i in range(1, n):
        while k and text[i] != text[k]:
            k = failure[k - 1]
        if text[i] == text[k]:
            k += 1
        failure[i] = k
    period = n - failure[-1]
    return period if period and n % period == 0 else n


def _shannon_entropy(text: str) -> float:
    if not text:
        return 8.0
    counts = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


class StreamRepetitionLoopError(RuntimeError):
    """Raised when a streaming response degenerates into repeated text."""

    def __init__(self, repeated_line: str, repeat_count: int):
        self.repeated_line = repeated_line
        self.repeat_count = repeat_count
        preview = repeated_line[:160]
        super().__init__(
            f"{FAILED_REPETITION_LOOP}: repeated line {repeat_count}x: {preview!r}"
        )


class StreamingRepetitionGuard:
    """Detect a *degenerate* repeated tail in an assistant response.

    Discrimination is by LOCAL DENSITY, not a global count.  A real runaway
    loop drives the tail of the output to consist of nothing but a handful of
    lines cycling (period 1 for a single-line loop, period k for a k-line
    block).  Legitimate text that merely *contains* a repeated line — a digest
    with one identical status line per section, a quoted log with a recurring
    error line, a table, a song refrain — keeps that line interleaved with
    unique content, so it never dominates a trailing window.

    The guard therefore fires only when, within the sliding window of the last
    ``window`` counted lines, the most frequent line reaches ``repeat_threshold``
    occurrences AND the window holds at most ``max_distinct`` distinct lines
    (i.e. the window is dominated by a few cycling lines).  A pure global count
    caught the 2026-07-02 incident but also killed the legitimate spaced-repeat
    shapes above; windowed dominance keeps the incident and spares them.
    """

    def __init__(
        self,
        *,
        min_total_chars: int = 1200,
        min_line_chars: int = 40,
        repeat_threshold: int = 8,
        window: Optional[int] = None,
        max_distinct: int = 6,
        tail_check_min_chars: int = 96,
        tail_max_period: int = 80,
        tail_min_repeats: int = 5,
        heavy_window_chars: int = 4000,
        heavy_check_every_chars: int = 1024,
        ngram_size: int = 48,
        distinct_ngram_ratio_threshold: float = 0.15,
        entropy_threshold: float = 2.5,
    ) -> None:
        self.min_total_chars = max(0, int(min_total_chars))
        self.min_line_chars = max(1, int(min_line_chars))
        self.repeat_threshold = max(2, int(repeat_threshold))
        # Window must span enough lines for a k-line block loop to accumulate
        # ``repeat_threshold`` copies of each of its lines: k*threshold.  With
        # the default max_distinct=6 the largest catchable block is 6 lines, so
        # threshold*6 is the natural span.  Callers may override.
        if window is None:
            window = self.repeat_threshold * 6
        self.window = max(self.repeat_threshold, int(window))
        self.max_distinct = max(1, int(max_distinct))
        self.tail_check_min_chars = max(16, int(tail_check_min_chars))
        self.tail_max_period = max(2, int(tail_max_period))
        self.tail_min_repeats = max(3, int(tail_min_repeats))
        self.heavy_window_chars = max(256, int(heavy_window_chars))
        self.heavy_check_every_chars = max(128, int(heavy_check_every_chars))
        self.ngram_size = max(8, int(ngram_size))
        self.distinct_ngram_ratio_threshold = float(distinct_ngram_ratio_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self._window_lines: deque[str] = deque()
        self._window_counts: Counter[str] = Counter()
        self._char_window: deque[str] = deque(maxlen=self.heavy_window_chars)
        self._pending = ""
        self._total_chars = 0
        self._chars_since_heavy_check = 0

    @classmethod
    def from_env(cls) -> Optional["StreamingRepetitionGuard"]:
        raw_enabled = os.getenv("HERMES_STREAM_REPETITION_GUARD", "1").strip().lower()
        if raw_enabled in {"0", "false", "off", "no"}:
            return None

        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        _threshold = _int("HERMES_STREAM_REPETITION_THRESHOLD", 8)
        return cls(
            min_total_chars=_int("HERMES_STREAM_REPETITION_MIN_CHARS", 1200),
            min_line_chars=_int("HERMES_STREAM_REPETITION_MIN_LINE_CHARS", 40),
            repeat_threshold=_threshold,
            window=_int("HERMES_STREAM_REPETITION_WINDOW", _threshold * 6),
            max_distinct=_int("HERMES_STREAM_REPETITION_MAX_DISTINCT", 6),
            tail_check_min_chars=_int(
                "HERMES_STREAM_REPETITION_TAIL_MIN_CHARS", 96
            ),
            tail_max_period=_int("HERMES_STREAM_REPETITION_TAIL_MAX_PERIOD", 80),
            tail_min_repeats=_int("HERMES_STREAM_REPETITION_TAIL_MIN_REPEATS", 5),
            heavy_window_chars=_int(
                "HERMES_STREAM_REPETITION_HEAVY_WINDOW_CHARS", 4000
            ),
            heavy_check_every_chars=_int(
                "HERMES_STREAM_REPETITION_HEAVY_CHECK_EVERY_CHARS", 1024
            ),
            ngram_size=_int("HERMES_STREAM_REPETITION_NGRAM_SIZE", 48),
            distinct_ngram_ratio_threshold=_float(
                "HERMES_STREAM_REPETITION_NGRAM_RATIO", 0.15
            ),
            entropy_threshold=_float("HERMES_STREAM_REPETITION_ENTROPY", 2.5),
        )

    @classmethod
    def reasoning_from_env(cls) -> Optional["StreamingRepetitionGuard"]:
        """Guard instance for the reasoning/thinking channel.

        Reasoning loops burn the whole output budget with zero visible text
        (finish_reason='length', content empty) — the incident signature the
        content-only guard cannot see.  Same thresholds as the content guard;
        disabled by either the main switch or its own.
        """
        raw = os.getenv(
            "HERMES_STREAM_REPETITION_GUARD_REASONING", "1"
        ).strip().lower()
        if raw in {"0", "false", "off", "no"}:
            return None
        return cls.from_env()

    def feed(self, text: str) -> None:
        if not isinstance(text, str) or not text:
            return
        self._total_chars += len(text)
        self._chars_since_heavy_check += len(text)
        self._char_window.extend(text)
        self._pending += text
        parts = self._pending.splitlines(keepends=True)
        if parts and not (parts[-1].endswith("\n") or parts[-1].endswith("\r")):
            self._pending = parts.pop()
        else:
            self._pending = ""
        for part in parts:
            self._record_line(part)
        self._check_periodic_tail()
        self._check_low_entropy_window()

    def flush(self) -> None:
        """Count the trailing unterminated line (for completed-text checks)."""
        pending, self._pending = self._pending, ""
        if pending:
            self._record_line(pending)

    def _record_line(self, line: str) -> None:
        normalized = self._normalize_line(line)
        if not normalized:
            return
        # Slide the window: append the new line, evict the oldest past capacity.
        self._window_lines.append(normalized)
        self._window_counts[normalized] += 1
        if len(self._window_lines) > self.window:
            evicted = self._window_lines.popleft()
            if self._window_counts[evicted] <= 1:
                del self._window_counts[evicted]
            else:
                self._window_counts[evicted] -= 1

        if self._total_chars < self.min_total_chars:
            return
        top_line, top_count = self._window_counts.most_common(1)[0]
        # Fire only when a few lines DOMINATE the trailing window: the most
        # frequent hits the threshold and the window holds few distinct lines.
        # Spaced legitimate repeats keep the window diverse (many distinct
        # lines) so top_count rises but distinct-count stays high → no fire.
        if (
            top_count >= self.repeat_threshold
            and len(self._window_counts) <= self.max_distinct
        ):
            raise StreamRepetitionLoopError(top_line, top_count)

    def _normalize_line(self, line: str) -> str:
        return normalize_guard_line(line, self.min_line_chars)

    def _check_periodic_tail(self) -> None:
        if self._total_chars < self.min_total_chars:
            return
        tail_len = self.tail_max_period * self.tail_min_repeats
        if len(self._char_window) < max(self.tail_check_min_chars, tail_len):
            return
        tail = "".join(self._char_window)[-tail_len:]
        period = _smallest_period(tail)
        repeats = len(tail) // max(1, period)
        if 2 <= period <= self.tail_max_period and repeats >= self.tail_min_repeats:
            unit = tail[:period]
            if unit.strip():
                raise StreamRepetitionLoopError(
                    f"periodic tail: {unit[:160]!r}",
                    repeats,
                )

    def _check_low_entropy_window(self) -> None:
        if self._total_chars < self.min_total_chars:
            return
        if self._chars_since_heavy_check < self.heavy_check_every_chars:
            return
        self._chars_since_heavy_check = 0
        if len(self._char_window) < max(self.ngram_size * 4, self.heavy_window_chars // 2):
            return
        window = "".join(self._char_window)
        if len(window) <= self.ngram_size:
            return
        total = len(window) - self.ngram_size
        distinct = len({window[i : i + self.ngram_size] for i in range(total)})
        ratio = distinct / max(1, total)
        if ratio > self.distinct_ngram_ratio_threshold:
            return
        entropy = _shannon_entropy(window)
        if entropy <= self.entropy_threshold:
            raise StreamRepetitionLoopError(
                (
                    "low-entropy repetitive window: "
                    f"ngram_ratio={ratio:.3f}, entropy={entropy:.2f}"
                ),
                max(2, int(1 / max(ratio, 0.001))),
            )


def find_repeated_line_from_env(
    text: Optional[str],
) -> Optional[StreamRepetitionLoopError]:
    """Run the guard over already-completed text (non-streaming check).

    Returns the would-be error (carrying ``repeated_line``/``repeat_count``)
    instead of raising, or None when the text is clean, empty, or the guard
    is disabled via HERMES_STREAM_REPETITION_GUARD.
    """
    if not isinstance(text, str) or not text:
        return None
    guard = StreamingRepetitionGuard.from_env()
    if guard is None:
        return None
    try:
        guard.feed(text)
        guard.flush()
    except StreamRepetitionLoopError as exc:
        return exc
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _countable_lines(text: Any, *, min_line_chars: int) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    return {
        line
        for line in (
            normalize_guard_line(part, min_line_chars=min_line_chars)
            for part in text.splitlines()
        )
        if line
    }


def find_cross_turn_repeated_line_from_env(
    messages: list[dict[str, Any]],
    current_text: Optional[str],
    *,
    threshold_override: Optional[int] = None,
) -> Optional[StreamRepetitionLoopError]:
    """Detect the same non-trivial assistant line recurring across turns.

    ``StreamingRepetitionGuard`` catches loops *inside* one response.  The TUI
    marathon failure mode repeats a short-but-nontrivial closure/status line
    across several assistant turns, often separated by tool calls, so every
    individual stream looks legal.  This check fires only on a consecutive
    assistant-turn streak and is disabled by the main repetition kill switch.
    """
    raw_enabled = os.getenv("HERMES_STREAM_REPETITION_GUARD", "1").strip().lower()
    if raw_enabled in {"0", "false", "off", "no"}:
        return None
    raw_cross_enabled = os.getenv(
        "HERMES_CROSS_TURN_REPETITION_GUARD", "1"
    ).strip().lower()
    if raw_cross_enabled in {"0", "false", "off", "no"}:
        return None

    min_line_chars = max(
        1, _env_int("HERMES_CROSS_TURN_REPETITION_MIN_LINE_CHARS", 40)
    )
    threshold = max(
        2,
        threshold_override
        if threshold_override is not None
        else _env_int("HERMES_CROSS_TURN_REPETITION_THRESHOLD", 4),
    )
    max_assistant_turns = max(
        threshold, _env_int("HERMES_CROSS_TURN_REPETITION_LOOKBACK", 8)
    )
    current_lines = _countable_lines(current_text, min_line_chars=min_line_chars)
    if not current_lines:
        return None

    for line in current_lines:
        streak = 1
        seen_assistant_turns = 0
        for msg in reversed(messages or []):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if msg.get("_compressed_summary"):
                continue
            seen_assistant_turns += 1
            if seen_assistant_turns > max_assistant_turns:
                break
            prior_text = "\n".join(
                part
                for part in (
                    msg.get("content"),
                    msg.get("reasoning"),
                    msg.get("reasoning_content"),
                )
                if isinstance(part, str) and part
            )
            if line not in _countable_lines(prior_text, min_line_chars=min_line_chars):
                break
            streak += 1
            if streak >= threshold:
                return StreamRepetitionLoopError(line, streak)
    return None


def truncate_at_repetition(text: str, repeated_line: str) -> str:
    """Cut *text* at the second occurrence of the degenerate line.

    Keeps the first occurrence (context for the reader), drops the repeated
    tail, and appends REPETITION_CUT_MARKER.  Returns *text* unchanged when
    the line repeats fewer than twice (e.g. the loop was in another channel).
    """
    if not isinstance(text, str) or not text or not repeated_line:
        return text
    seen = 0
    kept_len = 0
    for line in text.splitlines(keepends=True):
        if normalize_guard_line(line, min_line_chars=1) == repeated_line:
            seen += 1
            if seen >= 2:
                return text[:kept_len].rstrip() + "\n\n" + REPETITION_CUT_MARKER
        kept_len += len(line)
    return text


__all__ = [
    "FAILED_REPETITION_LOOP",
    "REPETITION_CUT_MARKER",
    "StreamRepetitionLoopError",
    "StreamingRepetitionGuard",
    "find_cross_turn_repeated_line_from_env",
    "find_repeated_line_from_env",
    "normalize_guard_line",
    "truncate_at_repetition",
]
