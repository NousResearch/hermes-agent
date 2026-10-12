"""Keep one-shot model routing guidance out of assistant output and compaction replay.

Only a complete, leading Hermes notice is scaffolding. Quoted examples, ordinary
bracketed notes, tool output and incomplete notices remain user-visible data.
"""

from __future__ import annotations

import re
from typing import Any


_PREFIX = "[Note: model was just switched from "
_NOTICE = re.compile(
    r"\A\s*\[Note: model was just switched from [^\]\r\n]*"
    r"Adjust your self-identification accordingly\.\)?\]\s*"
)
_MAX_NOTICE_CHARS = 4096


def strip_model_switch_note(text: str, *, role: str = "assistant") -> str:
    """Remove leading routing notices, preserving the remainder byte-for-byte."""
    if role not in ("assistant", "user"):
        return text
    while match := _NOTICE.match(text):
        text = text[match.end():]
    return text


def _clean_content(content: Any) -> Any:
    if isinstance(content, str):
        return strip_model_switch_note(content)
    if not isinstance(content, list):
        return content
    parts = list(content)
    # A multimodal notice may occupy its own text part before the real request.
    for index, part in enumerate(parts):
        text = part if isinstance(part, str) else part.get("text") if isinstance(part, dict) else None
        if not isinstance(text, str):
            break
        clean = strip_model_switch_note(text)
        parts[index] = clean if isinstance(part, str) else {**part, "text": clean}
        if clean.strip():
            break
    return parts


def without_model_switch_notes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean compaction copies, including API sidecars, without touching cached rows."""
    result = []
    for message in messages:
        if message.get("role") not in ("assistant", "user"):
            result.append(message)
            continue
        clean = dict(message)
        for key in ("content", "api_content"):
            if key in clean:
                clean[key] = _clean_content(clean[key])
        result.append(clean)
    return result


class StreamingModelSwitchNoteFilter:
    """Hold a possible leading notice until its closing bracket arrives.

    Normal prose passes through immediately. A bounded candidate buffer prevents a
    malformed notice from indefinitely hiding an otherwise useful answer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._passthrough = False
        self._stripped_notice = False

    def feed(self, text: str) -> str:
        if self._passthrough:
            return text
        self._buffer += text
        while self._buffer:
            candidate = self._buffer.lstrip()
            if not candidate:
                return ""
            if _PREFIX.startswith(candidate):
                return ""
            if candidate.startswith(_PREFIX) and "]" not in candidate and len(candidate) <= _MAX_NOTICE_CHARS:
                return ""
            match = _NOTICE.match(self._buffer)
            if match:
                self._stripped_notice = True
                self._buffer = self._buffer[match.end():]
                continue
            self._passthrough = True
            text, self._buffer = self._buffer, ""
            return text
        return ""

    def flush(self) -> str:
        """Release an incomplete/non-notice candidate and start a fresh response."""
        tail = "" if self._stripped_notice and not self._buffer.strip() else self._buffer
        self.reset()
        return tail
