"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI gpt-5.x and Responses-API relays onto the chat wire) emit one
``reasoning_content`` delta per *completed* summary part, each opening with a bold heading.
The chat wire lacks the Responses API's ``summary_index`` delimiter (verified live on Nous
Portal ``openai/gpt-5.6-sol``), so plain concatenation glues ``**One****Two**`` into one
half-bold paragraph. The boundary is re-derived from a delta opening a bold heading, matching
the blank-line join Hermes' own Responses adapter does.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, Optional

from agent.message_content import flatten_message_text
from agent.message_sanitization import _sanitize_surrogates
from agent.think_scrubber import THINK_TAG_NAMES


__all__ = [
    "ReasoningDeltaAccumulator", "append_streamed_reasoning_detail", "assistant_reasoning_text",
    "extract_reasoning", "normalize_reasoning_delta", "separate_glued_reasoning_blocks",
]

_INLINE_REASONING_PATTERNS = tuple(
    re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in THINK_TAG_NAMES
)
_logger = logging.getLogger("agent.chat_completion_helpers")


def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """Reasoning text from structured fields, else inline thinking blocks in content."""
    parts: list[str] = []

    def _add(text, *, detail: bool = False) -> None:
        text = flatten_message_text(text, sep="")
        # Structured detail records often repeat the public reasoning summary
        # as smaller blocks. Preserve exact dedup between independent fields,
        # while dropping a detail only when its whole text is already present.
        duplicate = any(text in part for part in parts) if detail else text in parts
        if text and not duplicate:
            parts.append(text)

    _add(getattr(assistant_message, "reasoning", None))
    _add(getattr(assistant_message, "reasoning_content", None))
    for detail in getattr(assistant_message, "reasoning_details", None) or []:
        if isinstance(detail, dict):
            _add(
                detail.get("summary") or detail.get("thinking") or detail.get("content") or detail.get("text"),
                detail=True,
            )
    content = getattr(assistant_message, "content", None)
    if not parts and isinstance(content, list):
        # DeepSeek V4 Pro returns typed thinking blocks; replaying them is required on
        # the next turn. Custom backends can return the value as a JSON array.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                _add(flatten_message_text(block.get("thinking") or block.get("text") or "", sep="").strip())
    if not parts and isinstance(content, str) and content:
        for pattern in _INLINE_REASONING_PATTERNS:
            for block in pattern.findall(content):
                _add(block.strip())
    return "\n\n".join(parts) if parts else None


def assistant_reasoning_text(agent, assistant_message) -> Optional[str]:
    """Resolve stored reasoning and deliver it once when no stream sink received it."""
    reasoning_text = agent._extract_reasoning(assistant_message)
    if not reasoning_text:
        content = flatten_message_text(getattr(assistant_message, "content", None))
        think_blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if think_blocks:
            reasoning_text = "\n\n".join(b.strip() for b in think_blocks if b.strip()) or None
    if reasoning_text and agent.verbose_logging:
        _logger.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")
    # A reasoning-only display can receive native deltas without text-stream callbacks.
    # Latch before attempting the sink so a callback that accepted text and raised is
    # not replayed during storage (repeated materialization is also safe).
    if (reasoning_text and agent.reasoning_callback
            and not getattr(agent, "_reasoning_streamed_this_response", False)
            and not agent.stream_delta_callback and not agent._stream_callback):
        agent._reasoning_streamed_this_response = True
        with contextlib.suppress(Exception):
            agent.reasoning_callback(reasoning_text)
    return _sanitize_surrogates(reasoning_text) if reasoning_text else reasoning_text


def separate_glued_reasoning_blocks(previous: str, delta: Any) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    A break is inserted when *delta* opens a *closed* bold heading and *previous* is mid-line
    (heading butting heading, or prose butting heading). Token-streamed reasoning is left
    alone: its deltas carry their own whitespace, and a fragment that merely opens emphasis
    (``**`` alone) is not a part boundary — summary parts carry the whole heading in one delta.
    """
    # Relays also emit content-part lists/dicts; fragments carry their own whitespace.
    delta = flatten_message_text(delta, sep="")
    glued = previous and delta and not previous[-1].isspace() and delta.startswith("**") and "**" in delta[2:]
    return f"\n\n{delta}" if glued else delta


# reasoning_details entry types whose consecutive fragments are ONE logical block.
_MERGEABLE_DETAIL_TEXT_KEYS = {"reasoning.text": "text", "reasoning.summary": "summary"}
_BACKFILL_DETAIL_KEYS = ("signature", "id", "format", "index")


# Cumulative snapshots and long reconnect echoes are trimmed to their new tail. Short
# repeated tokens can be real model text, so only overlaps of at least 24 chars trim.
_MIN_REASONING_OVERLAP = 24


def normalize_reasoning_delta(accumulated: str, delta: str) -> str:
    """The part of ``delta`` not already present at the end of ``accumulated``."""
    if not delta or not accumulated:
        return delta
    if len(accumulated) >= _MIN_REASONING_OVERLAP and delta.startswith(accumulated):
        return delta[len(accumulated):]
    for overlap in range(min(len(accumulated), len(delta)), _MIN_REASONING_OVERLAP - 1, -1):
        if accumulated.endswith(delta[:overlap]):
            return delta[overlap:]
    return delta


class ReasoningDeltaAccumulator:
    """Per-request cumulative/suffix echo normalizer for main and Relay collectors."""

    def __init__(self) -> None:
        self._raw_parts: list[str] = []
        self.parts: list[str] = []

    def feed(self, delta: Any) -> str:
        delta = flatten_message_text(delta, sep="")
        if len(delta) >= _MIN_REASONING_OVERLAP:
            delta = normalize_reasoning_delta("".join(self._raw_parts), delta)
        if not delta:
            return ""
        self._raw_parts.append(delta)
        text = separate_glued_reasoning_blocks(self.parts[-1] if self.parts else "", delta)
        self.parts.append(text)
        return text


def streamed_reasoning_detail_text(detail: Any) -> str:
    """Readable text from a detail delta; never expose opaque replay material."""
    dtype = detail.get("type") if isinstance(detail, dict) else getattr(detail, "type", None)
    key = _MERGEABLE_DETAIL_TEXT_KEYS.get(dtype) if isinstance(dtype, str) else None
    if key is None:
        return ""
    text = detail.get(key) if isinstance(detail, dict) else getattr(detail, key, None)
    return text if isinstance(text, str) else ""


def append_streamed_reasoning_detail(details_acc: list, detail: Any) -> None:
    """Accumulate one streamed ``reasoning_details`` delta entry into *details_acc*.

    OpenRouter streams ``reasoning_details`` as word-level deltas: consecutive
    ``reasoning.text`` / ``reasoning.summary`` entries are fragments of one logical
    block and are merged (later fragments backfill ``signature``/``id`` the first
    omitted); encrypted/opaque entries stay discrete. Unmerged, a long thought
    replays as hundreds of one-word entries and providers that validate the
    sequence shape on the next turn reject it. SDK objects are normalized to dicts.
    """
    if not isinstance(detail, dict):
        if hasattr(detail, "model_dump"):
            detail = detail.model_dump(warnings=False)
        elif hasattr(detail, "__dict__"):
            detail = dict(detail.__dict__)
        else:
            return
    dtype = detail.get("type")
    merge_key = _MERGEABLE_DETAIL_TEXT_KEYS.get(dtype)
    last = details_acc[-1] if details_acc else None
    if last is not None and merge_key and last.get("type") == dtype and isinstance(detail.get(merge_key), str):
        last[merge_key] = (last.get(merge_key) or "") + detail[merge_key]
        for k in _BACKFILL_DETAIL_KEYS:
            if last.get(k) in (None, "") and detail.get(k) not in (None, ""):
                last[k] = detail[k]
        return
    details_acc.append(dict(detail))
