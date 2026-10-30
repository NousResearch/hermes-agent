"""Incremental transcript rendering for the adviser.

Port of oh-my-pi's ``advisor/runtime.ts`` render half plus
``advisor/message-fingerprint.ts`` and ``advisor/delta-split.ts``, narrowed to the
message shapes Hermes actually sends (OpenAI-style ``system | user | assistant |
tool`` dicts). Each boundary renders only the tail past the adviser's cursor; a
changed prefix (compaction, undo, branch) rewinds the cursor so the next update
replays the rewritten transcript.

The renderer is deliberately append-only and byte-stable: prompt caches are
prefix-based, so a rendered boundary never depends on anything beyond the messages
it contains. The ``[in progress — more steps follow]`` marker stays at the END of
a batch so a wip→final flip never changes the batch prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

WIP_MARKER = "\n\n---\n\n[in progress — more steps follow]"
SESSION_UPDATE_HEADING = "### Session update"
#: A single rendered message is capped so one giant tool result cannot poison every
#: later adviser request; the primary agent already saw the full text.
MAX_MESSAGE_CHARS = 24_000

_FINGERPRINT_FIELDS = (
    "role", "content", "tool_calls", "tool_call_id", "name", "reasoning",
    "display_kind", "finish_reason", "is_error",
)


def fingerprint_message(message: Any) -> Optional[str]:
    """Stable identity of the fields the renderer reads (timestamps/usage excluded)."""
    if not isinstance(message, dict):
        return None
    try:
        payload = json.dumps(
            {field: message.get(field) for field in _FINGERPRINT_FIELDS},
            sort_keys=True, default=str, ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _clip(text: str) -> str:
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 1].rstrip() + "…"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str) and (
                    block.get("type") == "text" or block.get("type") is None
                ):
                    parts.append(block["text"])
        return "\n".join(part for part in parts if part)
    return str(content)


def _tool_calls_text(tool_calls: Any) -> str:
    lines: list[str] = []
    for call in tool_calls or ():
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "tool")
        arguments = function.get("arguments") if function else call.get("arguments")
        if isinstance(arguments, str):
            rendered = arguments
        else:
            rendered = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
        lines.append(f"**tool call — {name}**\n```json\n{rendered}\n```")
    return "\n\n".join(lines)


def _render_message(message: dict[str, Any], *, include_reasoning: bool = True) -> str:
    role = str(message.get("role") or "unknown")
    if role == "assistant":
        parts = ["**assistant**"]
        reasoning = message.get("reasoning")
        if include_reasoning and isinstance(reasoning, str) and reasoning.strip():
            parts.append("_reasoning_\n" + _clip(reasoning.strip()))
        text = _content_text(message.get("content"))
        if text.strip():
            parts.append(_clip(text.strip()))
        calls = _tool_calls_text(message.get("tool_calls"))
        if calls:
            parts.append(calls)
        return "\n\n".join(parts)
    if role == "tool":
        name = str(message.get("name") or "tool")
        body = _clip(_content_text(message.get("content")).strip())
        return f"**tool result — {name}**\n{body}" if body else f"**tool result — {name}**"
    if role == "user":
        body = _clip(_content_text(message.get("content")).strip())
        return f"**user**\n{body}" if body else ""
    body = _clip(_content_text(message.get("content")).strip())
    return f"**{role}**\n{body}" if body else ""


def _adviser_visible(message: Any) -> bool:
    """Skip rows the adviser itself injected: re-feeding them causes advice loops."""
    if not isinstance(message, dict):
        return False
    if message.get("role") == "system":
        return False
    return message.get("display_kind") != "adviser"


def render_delta(messages: Iterable[Any], *, wip: bool = False,
                 include_reasoning: bool = True) -> Optional[str]:
    """Render a batch of messages as one Session update block, or None when empty."""
    blocks = [
        block for block in (
            _render_message(message, include_reasoning=include_reasoning)
            for message in messages if _adviser_visible(message)
        ) if block
    ]
    if not blocks:
        return None
    text = f"{SESSION_UPDATE_HEADING}\n\n" + "\n\n".join(blocks)
    return text + WIP_MARKER if wip else text


@dataclass
class PendingDelta:
    text: str
    wip: bool
    turns: int = 1
    reset: bool = False
    terminal: bool = False


class DeltaTracker:
    """Cursor + delivered-prefix identities; detects rewritten history."""

    def __init__(self) -> None:
        self._cursor = 0
        self._prefix: list[Optional[str]] = []
        self._render_revision = 0

    @property
    def render_revision(self) -> int:
        return self._render_revision

    @property
    def cursor(self) -> int:
        return self._cursor

    def reset(self, *, cursor: int = 0) -> None:
        self._cursor = cursor
        self._prefix = []
        self._render_revision += 1

    def seed(self, messages: list[Any]) -> None:
        self._cursor = len(messages)
        self._prefix = [fingerprint_message(message) for message in messages]
        self._render_revision += 1

    def _prefix_changed(self, messages: list[Any]) -> bool:
        if len(messages) < self._cursor:
            return True
        for index in range(self._cursor):
            if fingerprint_message(messages[index]) != self._prefix[index]:
                return True
        return False

    def capture(self, messages: list[Any], *, wip: bool,
                include_reasoning: bool = True) -> Optional[PendingDelta]:
        """Render the tail and advance the cursor; a rewritten prefix replays everything."""
        reset = self._prefix_changed(messages)
        if reset:
            self.reset(cursor=0)
        tail = messages[self._cursor:]
        self._cursor = len(messages)
        self._prefix = [fingerprint_message(message) for message in messages]
        text = render_delta(tail, wip=wip, include_reasoning=include_reasoning)
        if text is None:
            return None
        return PendingDelta(text=text, wip=wip, turns=1, reset=reset)


__all__ = [
    "MAX_MESSAGE_CHARS",
    "SESSION_UPDATE_HEADING",
    "WIP_MARKER",
    "DeltaTracker",
    "PendingDelta",
    "fingerprint_message",
    "render_delta",
]
