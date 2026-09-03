"""SSE ``[DONE]`` sentinel normalization for OpenAI-compatible proxies.

Some upstreams (notably Nous Portal for certain free models) deliver a
complete chat-completions stream — content deltas, a non-null
``finish_reason``, and often a ``lastOne: true`` usage frame — then close
the connection without the conventional OpenAI terminal event::

    data: [DONE]

Strict OpenAI-compatible clients treat that shape as a truncated stream.
This module watches the forwarded SSE byte stream and reports whether the
proxy should append a single ``data: [DONE]`` frame after a *clean*
upstream EOF.

Rules (issue #90848):
- Retain every original delta unchanged (this helper never rewrites bytes).
- Append ``[DONE]`` only after a complete terminal choice
  (``finish_reason`` non-null) **or** an upstream ``lastOne: true`` marker.
- Never synthesize ``[DONE]`` after an error event, or when the stream was
  interrupted before clean EOF.
- Never emit a second ``[DONE]`` when the upstream already sent one.
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
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            self._consume_line(line)

    def mark_interrupted(self) -> None:
        """Upstream stream ended via error/cancel — do not synthesize DONE."""
        self.interrupted = True

    def should_append_done(self) -> bool:
        """True when a single terminal ``[DONE]`` should be appended."""
        if (
            self.interrupted
            or self.saw_done
            or self.saw_error_event
            or self.saw_malformed_event
        ):
            return False
        # Flush any trailing line without a final newline (rare but valid),
        # then dispatch a final event that never saw its blank-line boundary.
        if self._buf:
            self._consume_line(bytes(self._buf))
            self._buf.clear()
        self._dispatch_event()
        if self.saw_done or self.saw_error_event or self.saw_malformed_event:
            return False
        return self.saw_terminal_finish or self.saw_last_one

    def _consume_line(self, line: bytes) -> None:
        # Strip CR from CRLF-delimited SSE.
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            # Blank line = SSE event boundary: dispatch accumulated data.
            self._dispatch_event()
            return
        if not line.startswith(b"data:"):
            return
        # Per the SSE spec one event may span several consecutive ``data:``
        # lines whose payloads are joined with "\n" at dispatch time.
        # Parsing each line independently would misread a split JSON event
        # as two malformed fragments.
        self._data_lines.append(line[5:].strip())

    def _dispatch_event(self) -> None:
        if not self._data_lines:
            return
        payload = b"\n".join(self._data_lines)
        self._data_lines = []
        payload = payload.strip()
        if payload == b"[DONE]":
            self.saw_done = True
            return
        if not payload:
            return
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            self.saw_malformed_event = True
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            self.saw_malformed_event = True
            return
        if not isinstance(event, dict):
            return
        if event.get("error") is not None:
            self.saw_error_event = True
            return
        # Accept integer-truthy sentinels too — relabelled upstreams have
        # been observed sending ``"lastOne": 1`` / ``"true"``.
        if event.get("lastOne") in (True, 1, "true"):
            self.saw_last_one = True
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                self.saw_terminal_finish = True
            # OpenAI error-shaped finish reasons should not unlock DONE.
            fr = choice.get("finish_reason")
            if isinstance(fr, str) and fr.lower() in {"error", "provider_error"}:
                self.saw_error_event = True


def content_type_is_sse(headers) -> bool:
    """Return True when response headers advertise an SSE body."""
    try:
        value = headers.get("Content-Type") or headers.get("content-type") or ""
    except Exception:
        value = ""
    return "text/event-stream" in str(value).lower()


__all__ = [
    "DONE_SSE_FRAME",
    "SseDoneTracker",
    "content_type_is_sse",
]
