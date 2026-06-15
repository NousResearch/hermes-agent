"""Stateful scrubber for provider-native tool-call markup in text streams.

Some OpenAI-compatible providers fall back to text-encoded tool calls when
their tool parser misses a call.  DeepSeek-style DSML is one such format:

    <｜DSML｜tool_calls> <｜DSML｜invoke name="write_file"> ...

That markup is internal control data, not assistant prose.  If it reaches the
gateway, long JSON arguments get split into dozens of chat messages with
``(n/N)`` suffixes.  This scrubber drops those spans before they reach users.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["StreamingToolCallScrubber", "strip_tool_call_markup"]


class StreamingToolCallScrubber:
    """Remove text-encoded tool-call spans across streaming boundaries."""

    _OPEN_MARKERS: Tuple[str, ...] = (
        "<|DSML|tool_calls>",
        "<｜DSML｜tool_calls>",
    )
    _CLOSE_MARKERS: Tuple[str, ...] = (
        "<|DSML|/tool_calls>",
        "<｜DSML｜/tool_calls>",
        "<|DSML|end_tool_calls>",
        "<｜DSML｜end_tool_calls>",
        "</tool_calls>",
    )
    _MAX_MARKER_LEN: int = max(len(m) for m in _OPEN_MARKERS + _CLOSE_MARKERS)

    def __init__(self) -> None:
        self._in_tool_call = False
        self._buf = ""

    def reset(self) -> None:
        self._in_tool_call = False
        self._buf = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_tool_call:
                close_idx, close_len = self._find_first(buf, self._CLOSE_MARKERS)
                if close_idx == -1:
                    held = self._max_partial_suffix(buf, self._CLOSE_MARKERS)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                buf = buf[close_idx + close_len :]
                self._in_tool_call = False
                continue

            open_idx, open_len = self._find_first(buf, self._OPEN_MARKERS)
            if open_idx == -1:
                held = self._max_partial_suffix(buf, self._OPEN_MARKERS)
                if held:
                    out.append(buf[:-held])
                    self._buf = buf[-held:]
                else:
                    out.append(buf)
                return "".join(out)

            if open_idx:
                out.append(buf[:open_idx])
            buf = buf[open_idx + open_len :]
            self._in_tool_call = True

        return "".join(out)

    def flush(self) -> str:
        """Flush benign marker-prefix tails, dropping unfinished tool spans."""
        if self._in_tool_call:
            self._in_tool_call = False
            self._buf = ""
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    @staticmethod
    def _find_first(buf: str, markers: Tuple[str, ...]) -> tuple[int, int]:
        best_idx = -1
        best_len = 0
        for marker in markers:
            idx = buf.find(marker)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_len = len(marker)
        return best_idx, best_len

    @classmethod
    def _max_partial_suffix(cls, buf: str, markers: Tuple[str, ...]) -> int:
        max_check = min(len(buf), cls._MAX_MARKER_LEN - 1)
        for size in range(max_check, 0, -1):
            suffix = buf[-size:]
            if any(marker.startswith(suffix) for marker in markers if len(marker) > size):
                return size
        return 0


def strip_tool_call_markup(text: str) -> str:
    """One-shot DSML tool-call markup removal for completed responses."""
    if not isinstance(text, str) or not text:
        return text
    scrubber = StreamingToolCallScrubber()
    return scrubber.feed(text) + scrubber.flush()
