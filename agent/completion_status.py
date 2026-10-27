"""Private, optional end-of-turn marker handling for local AI Server responses."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

COMPLETION_STATUS_MARKER = "[[HERMES_TURN_COMPLETE]]"
COMPLETION_STATUS_SIGNATURE = "# Local model turn completion protocol v1"
COMPLETION_STATUS_GUIDANCE = (
    f"{COMPLETION_STATUS_SIGNATURE}\n"
    "When convenient, you may end your user-facing reply with this exact standalone line:\n"
    f"{COMPLETION_STATUS_MARKER}\n"
    "This optional line is a private control signal: do not explain, quote, or format it as code. "
    "Never withhold or continue a finished answer just to add this line. "
    "If more work remains, make the next tool call instead of sending a final reply. "
    "If you are blocked or need the user's input, say that plainly."
)
_TERMINAL_MARKER_RE = re.compile(
    r"(?:^|\r?\n)\[\[HERMES_TURN_COMPLETE\]\][ \t]*(?:\r?\n[ \t]*)*\Z"
)
_TERMINAL_MARKER_PREFIXES = (
    COMPLETION_STATUS_MARKER,
    "\n" + COMPLETION_STATUS_MARKER,
    "\r\n" + COMPLETION_STATUS_MARKER,
)


def is_local_ai_server_gateway_url(base_url: Any) -> bool:
    """Match the loopback AI Server OpenAI endpoint, without classifying every local service."""
    raw = str(base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:
        if parsed.port != 8080:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def completion_status_enabled(system_prompt: Any, *, base_url: Any = None) -> bool:
    """Enable private-marker prompting/filtering for local routes and opted-in prompts."""
    return COMPLETION_STATUS_SIGNATURE in str(system_prompt or "") or is_local_ai_server_gateway_url(base_url)


def extract_completion_status(text: Any) -> tuple[Any, bool]:
    """Remove the exact terminal marker and report whether it was present."""
    if not isinstance(text, str):
        return text, False
    match = _TERMINAL_MARKER_RE.search(text)
    if match is None:
        return text, False
    return text[:match.start()].rstrip(), True


class CompletionStatusStreamFilter:
    """Hold a possible end marker until the stream proves whether it is terminal."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        combined = self._pending + text
        terminal = _TERMINAL_MARKER_RE.search(combined)
        if terminal is not None:
            emitted, self._pending = combined[:terminal.start()], combined[terminal.start():]
            return emitted
        keep = 0
        for candidate in _TERMINAL_MARKER_PREFIXES:
            limit = min(len(candidate), len(combined))
            while limit > keep and not combined.endswith(candidate[:limit]):
                limit -= 1
            keep = max(keep, limit)
        emitted = combined[:-keep] if keep else combined
        self._pending = combined[-keep:] if keep else ""
        return emitted

    def finish(self, *, marker_present: bool) -> str:
        pending, self._pending = self._pending, ""
        if marker_present:
            visible, found = extract_completion_status(pending)
            return visible if found else ""
        return pending


def apply_completion_status(
    agent: Any, *, assistant_message: Any, final_response: Any, promoted: bool,
) -> tuple[Any, bool]:
    """Strip a private marker when present; a missing marker never reopens a finished turn."""
    visible, marked = extract_completion_status(final_response)
    if marked and not promoted:
        assistant_message.content = visible
    finish_stream = getattr(agent, "_finish_completion_status_stream", None)
    if callable(finish_stream):
        finish_stream(marker_present=marked)
    return visible, marked
