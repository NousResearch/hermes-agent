"""Stateful scrubber for Harmony tags in streamed assistant text.

Allows only content in the 'final' channel to pass through, stripping the surrounding
Harmony tags (<|channel|>final<|message|>, <|end|>, <|endoftext|>). For all other channels,
both the tags and the payloads are suppressed entirely to avoid leaking tool calls, thoughts,
or formatting instructions in real-time.

Usage:
    scrubber = StreamingHarmonyScrubber()
    for delta in stream:
        visible = scrubber.feed(delta)
        if visible:
            emit(visible)
    tail = scrubber.flush()
    if tail:
        emit(tail)
"""

from __future__ import annotations

import re

__all__ = ["StreamingHarmonyScrubber"]


class StreamingHarmonyScrubber:
    """Stateful scrubber for Harmony tags.

    State machine states:
      - "SCANNING": Outside any channel. Searching for '<|channel|>' or orphan '<|end|>'/'<|endoftext|>'.
      - "IN_HEADER": Found '<|channel|>' but haven't resolved '<|message|>' yet.
      - "IN_CHANNEL_FINAL": Inside a 'final' channel. Emitting all content until '<|end|>' or '<|endoftext|>'.
      - "IN_CHANNEL_SUPPRESSED": Inside a non-final channel (e.g. commentary, analysis, thought). Discarding content.
    """

    def __init__(self) -> None:
        self._state: str = "SCANNING"
        self._buf: str = ""
        self._channel_name: str | None = None

    def reset(self) -> None:
        """Reset all state. Call at the top of every new turn."""
        self._state = "SCANNING"
        self._buf = ""
        self._channel_name = None

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion of the stream."""
        if not text:
            return ""

        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._state == "SCANNING":
                # Find the earliest occurrence of any relevant tag in SCANNING state
                tags = ("<|channel|>", "<|end|>", "<|endoftext|>")
                earliest_idx = -1
                matched_tag = None
                buf_lower = buf.lower()

                for tag in tags:
                    idx = buf_lower.find(tag)
                    if idx != -1:
                        if earliest_idx == -1 or idx < earliest_idx:
                            earliest_idx = idx
                            matched_tag = tag

                if matched_tag is not None:
                    # Emit everything before the tag
                    pre = buf[:earliest_idx]
                    if pre:
                        out.append(pre)

                    if matched_tag == "<|channel|>":
                        self._state = "IN_HEADER"
                        buf = buf[earliest_idx:]
                    else:
                        # Orphan close tag: discard it
                        buf = buf[earliest_idx + len(matched_tag):]
                else:
                    # No tag matched. Hold back any partial suffix at the tail
                    partial_len = 0
                    for tag in tags:
                        for l in range(min(len(buf), len(tag) - 1), 0, -1):
                            suffix = buf_lower[-l:]
                            if tag.startswith(suffix):
                                if l > partial_len:
                                    partial_len = l

                    if partial_len > 0:
                        self._buf = buf[-partial_len:]
                        emit_text = buf[:-partial_len]
                    else:
                        emit_text = buf

                    if emit_text:
                        out.append(emit_text)
                    buf = ""

            elif self._state == "IN_HEADER":
                # Search for '<|message|>' to complete the header
                idx = buf.lower().find("<|message|>")
                if idx != -1:
                    header_content = buf[:idx]
                    header_payload = header_content[len("<|channel|>"):].strip()
                    parts = header_payload.split(None, 1)
                    channel_name = parts[0].lower() if parts else ""

                    self._channel_name = channel_name
                    if channel_name == "final":
                        self._state = "IN_CHANNEL_FINAL"
                    else:
                        self._state = "IN_CHANNEL_SUPPRESSED"

                    # Discard header and the '<|message|>' tag
                    buf = buf[idx + len("<|message|>"):]
                else:
                    # Keep buffering until we find '<|message|>'
                    if len(buf) > 1000:
                        # Fallback for abnormally long pseudo-headers
                        out.append(buf)
                        self._state = "SCANNING"
                        buf = ""
                    else:
                        self._buf = buf
                        buf = ""

            elif self._state in ("IN_CHANNEL_FINAL", "IN_CHANNEL_SUPPRESSED"):
                # Search for end tags
                close_tags = ("<|end|>", "<|endoftext|>")
                close_idx = -1
                close_len = 0
                buf_lower = buf.lower()

                for tag in close_tags:
                    idx = buf_lower.find(tag)
                    if idx != -1:
                        if close_idx == -1 or idx < close_idx:
                            close_idx = idx
                            close_len = len(tag)

                if close_idx != -1:
                    content = buf[:close_idx]
                    if self._state == "IN_CHANNEL_FINAL" and content:
                        out.append(content)

                    self._state = "SCANNING"
                    self._channel_name = None
                    buf = buf[close_idx + close_len:]
                else:
                    # Hold back partial suffix
                    partial_len = 0
                    for tag in close_tags:
                        for l in range(min(len(buf), len(tag) - 1), 0, -1):
                            suffix = buf_lower[-l:]
                            if tag.startswith(suffix):
                                if l > partial_len:
                                    partial_len = l

                    if partial_len > 0:
                        self._buf = buf[-partial_len:]
                        content = buf[:-partial_len]
                    else:
                        content = buf

                    if self._state == "IN_CHANNEL_FINAL" and content:
                        out.append(content)
                    buf = ""

        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush."""
        tail = self._buf
        self._buf = ""
        state = self._state
        self._state = "SCANNING"
        self._channel_name = None

        if state in ("IN_CHANNEL_FINAL", "SCANNING"):
            return tail
        return ""
