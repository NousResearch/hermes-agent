"""SSE ``[DONE]`` sentinel normalization for OpenAI-compatible proxies.

Strict OpenAI clients treat a stream without ``data: [DONE]`` as truncated. The tracker watches the
forwarded SSE bytes and says whether to append ONE ``[DONE]`` after a *clean* upstream EOF: only after
a terminal choice (``finish_reason`` non-null) or ``lastOne: true``; never after an error event, an
interrupted stream, or when the upstream already sent ``[DONE]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


DONE_SSE_FRAME = b"data: [DONE]\n\n"


@dataclass
class SseDoneTracker:
    """Incremental scanner over forwarded SSE chunks."""

    saw_done: bool = False
    saw_terminal_finish: bool = False
    saw_last_one: bool = False
    saw_error_event: bool = False
    saw_malformed_event: bool = False
    interrupted: bool = False
    _buf: bytearray = field(default_factory=bytearray, repr=False)
    _data_lines: list = field(default_factory=list, repr=False)

    def feed(self, chunk: bytes) -> None:
        """Observe a forwarded chunk (bytes are not modified)."""
        if not chunk:
            return
        self._buf.extend(chunk)
        while (nl := self._buf.find(b"\n")) >= 0:
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            self._consume_line(line)

    def mark_interrupted(self) -> None:
        """Upstream stream ended via error/cancel — do not synthesize DONE."""
        self.interrupted = True

    def _blocked(self) -> bool:
        return self.saw_done or self.saw_error_event or self.saw_malformed_event

    def should_append_done(self) -> bool:
        """True when a single terminal ``[DONE]`` should be appended."""
        if self.interrupted or self._blocked():
            return False
        # Flush a trailing line without a final newline (rare but valid), then dispatch a final
        # event that never saw its blank-line boundary.
        if self._buf:
            self._consume_line(bytes(self._buf))
            self._buf.clear()
        self._dispatch_event()
        if self._blocked():
            return False
        return self.saw_terminal_finish or self.saw_last_one

    def _consume_line(self, line: bytes) -> None:
        if line.endswith(b"\r"):  # CRLF-delimited SSE
            line = line[:-1]
        if not line:  # blank line = event boundary
            self._dispatch_event()
            return
        if not line.startswith(b"data:"):
            return
        # One event may span several ``data:`` lines joined with "\n" at dispatch; parsing each
        # line alone would misread a split JSON event as two malformed fragments.
        self._data_lines.append(line[5:].strip())

    def _dispatch_event(self) -> None:
        if not self._data_lines:
            return
        payload = b"\n".join(self._data_lines).strip()
        self._data_lines = []
        if payload == b"[DONE]":
            self.saw_done = True
            return
        if not payload:
            return
        try:
            event = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.saw_malformed_event = True
            return
        if not isinstance(event, dict):
            return
        if event.get("error") is not None:
            self.saw_error_event = True
            return
        # Relabelled upstreams have been observed sending ``"lastOne": 1`` / ``"true"``.
        if event.get("lastOne") in (True, 1, "true"):
            self.saw_last_one = True
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            fr = choice.get("finish_reason")
            if fr is not None:
                self.saw_terminal_finish = True
            # OpenAI error-shaped finish reasons should not unlock DONE.
            if isinstance(fr, str) and fr.lower() in {"error", "provider_error"}:
                self.saw_error_event = True


def content_type_is_sse(headers) -> bool:
    """True when response headers advertise an SSE body."""
    try:
        value = headers.get("Content-Type") or headers.get("content-type") or ""
    except Exception:
        value = ""
    return "text/event-stream" in str(value).lower()


__all__ = ["DONE_SSE_FRAME", "SseDoneTracker", "content_type_is_sse"]
