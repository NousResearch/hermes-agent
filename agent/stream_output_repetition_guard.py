"""Bound degenerate repetition while an assistant response is streaming."""

from __future__ import annotations

from collections import Counter, deque
import re


FAILED_REPETITION_LOOP = "FAILED_REPETITION_LOOP"
REPETITION_CUT_MARKER = "[Output stopped: repetitive generation detected.]"


def _normalize_line(line: str, min_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", (line or "").strip())
    if len(normalized) < min_chars:
        return ""
    if normalized in {"```", "---", "***"}:
        return ""
    if set(normalized) <= {"-", "|", ":", " "}:
        return ""
    return normalized


class StreamOutputRepetitionError(RuntimeError):
    """Raised when the tail of one streamed response stops making progress."""

    def __init__(
        self,
        repeated_unit: str,
        repeat_count: int,
        *,
        normalized_line: str | None = None,
    ) -> None:
        self.repeated_unit = repeated_unit
        self.repeat_count = repeat_count
        self.normalized_line = normalized_line
        super().__init__(
            f"{FAILED_REPETITION_LOOP}: {repeated_unit[:160]!r} "
            f"repeated {repeat_count} times"
        )


class StreamOutputRepetitionGuard:
    """Detect a repeated line block or an exact periodic character tail.

    A repeated line only trips the guard when a small set of lines dominates a
    trailing window. This excludes reports and logs where a recurring status is
    separated by varied content. The periodic check covers streams that never
    emit newlines.
    """

    def __init__(
        self,
        *,
        min_total_chars: int = 1200,
        min_line_chars: int = 40,
        repeat_threshold: int = 8,
        line_window: int | None = None,
        max_distinct_lines: int = 6,
        periodic_max_chars: int = 80,
        periodic_min_repeats: int = 5,
        periodic_check_every_chars: int = 256,
    ) -> None:
        self.min_total_chars = max(0, int(min_total_chars))
        self.min_line_chars = max(1, int(min_line_chars))
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.line_window = max(
            self.repeat_threshold,
            int(line_window or self.repeat_threshold * 6),
        )
        self.max_distinct_lines = max(1, int(max_distinct_lines))
        self.periodic_max_chars = max(2, int(periodic_max_chars))
        self.periodic_min_repeats = max(3, int(periodic_min_repeats))
        self.periodic_check_every_chars = max(64, int(periodic_check_every_chars))

        self._lines: deque[str] = deque()
        self._line_counts: Counter[str] = Counter()
        self._pending_line = ""
        self._total_chars = 0
        self._chars_since_periodic_check = 0
        periodic_window = self.periodic_max_chars * self.periodic_min_repeats
        self._character_tail: deque[str] = deque(maxlen=periodic_window)

    def feed(self, text: str) -> None:
        if not isinstance(text, str) or not text:
            return

        self._total_chars += len(text)
        self._chars_since_periodic_check += len(text)
        self._character_tail.extend(text)
        self._pending_line += text
        parts = self._pending_line.splitlines(keepends=True)
        if parts and not parts[-1].endswith(("\n", "\r")):
            self._pending_line = parts.pop()
        else:
            self._pending_line = ""

        for part in parts:
            self._record_line(part)
        if (
            self._pending_line
            and self._chars_since_periodic_check >= self.periodic_check_every_chars
        ):
            self._chars_since_periodic_check = 0
            self._check_periodic_tail()

    def flush(self) -> None:
        pending, self._pending_line = self._pending_line, ""
        if pending:
            self._record_line(pending)

    def _record_line(self, line: str) -> None:
        normalized = _normalize_line(line, self.min_line_chars)
        if not normalized:
            return

        self._lines.append(normalized)
        self._line_counts[normalized] += 1
        if len(self._lines) > self.line_window:
            evicted = self._lines.popleft()
            if self._line_counts[evicted] == 1:
                del self._line_counts[evicted]
            else:
                self._line_counts[evicted] -= 1

        if self._total_chars < self.min_total_chars:
            return
        repeated_line, repeat_count = self._line_counts.most_common(1)[0]
        if (
            repeat_count >= self.repeat_threshold
            and len(self._line_counts) <= self.max_distinct_lines
        ):
            raise StreamOutputRepetitionError(
                repeated_line,
                repeat_count,
                normalized_line=repeated_line,
            )

    def _check_periodic_tail(self) -> None:
        if self._total_chars < self.min_total_chars:
            return
        tail = "".join(self._character_tail)
        for period in range(2, self.periodic_max_chars + 1):
            sample_length = period * self.periodic_min_repeats
            if len(tail) < sample_length:
                break
            unit = tail[-period:]
            if unit.strip() and tail[-sample_length:] == unit * self.periodic_min_repeats:
                raise StreamOutputRepetitionError(
                    f"periodic tail: {unit!r}",
                    self.periodic_min_repeats,
                )


def truncate_repeated_tail(text: str, repeated_line: str | None) -> str:
    """Keep one repeated line, remove its repeated tail, and mark the cut."""
    if not isinstance(text, str) or not text:
        return REPETITION_CUT_MARKER
    if not repeated_line:
        return text.rstrip() + "\n\n" + REPETITION_CUT_MARKER

    seen = 0
    kept_chars = 0
    for line in text.splitlines(keepends=True):
        if _normalize_line(line, 1) == repeated_line:
            seen += 1
            if seen == 2:
                return text[:kept_chars].rstrip() + "\n\n" + REPETITION_CUT_MARKER
        kept_chars += len(line)
    return text.rstrip() + "\n\n" + REPETITION_CUT_MARKER


__all__ = [
    "FAILED_REPETITION_LOOP",
    "REPETITION_CUT_MARKER",
    "StreamOutputRepetitionError",
    "StreamOutputRepetitionGuard",
    "truncate_repeated_tail",
]
