"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

import re

from typing import Any, Dict, Optional

from agent.context_compressor import (
    _LEAN_RECOVERY_HEADING, _LEAN_USER_MESSAGES_HEADING, ContextCompressor, is_compaction_summary_message,
)
from agent.message_content import flatten_message_text


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    # Provider replay/metadata fields that ride the wire on every request but are invisible to
    # ``msg["content"]``/``msg["tool_calls"]`` accounting. Codex Responses sessions in particular carry
    # ``codex_reasoning_items`` blobs of ``encrypted_content`` that can dominate the serialized session (a
    # measured 214-turn session held ~115K tokens / 27% of its payload there — #55572).
    # ``reasoning_details`` is handled separately (see ``_reasoning_details_text_chars``): its signed/base64
    # envelope is excluded from the budget, mirroring the preflight estimator's exclusion in
    # ``model_metadata._estimate_message_tokens_without_images`` (#73298).
    # An assistant turn may carry only reasoning/thinking content with no visible text (extended-thinking
    # turns, thinking-only recovery responses). Such a turn is persisted with its reasoning fields and is
    # recallable from the transcript, but dropping it here as "empty" makes it vanish from the
    # resumed/reloaded session view while the desktop's reasoning disclosure has nothing to render. Keep it
    # when it carries reasoning so the "Thinking…" block still shows. (#44022)
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def project_compaction_message_for_display(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier.
    """
    if not isinstance(message, dict):
        return None
    if not is_compaction_summary_message(message):
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    # A tool-call carrier survives the strip for the model with no text of its own; with its
    # tool state removed below there is nothing left to show.
    if projected is None or not projected.get("content"):
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    # A safety-refused ask embedded in a composite carrier keeps its ``content_refused`` tag through
    # the unwrap, or a reopened chat would redraw it as an ordinary user bubble with no refused notice
    # (#149708 review): the model-facing handoff is gone, the user's live ask survives as a greyed
    # refused block. Every other carrier drops its internal ``display_kind`` as before.
    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND
    if projected.get("display_kind") != CONTENT_REFUSED_DISPLAY_KIND:
        projected.pop("display_kind", None)
    return projected



# Lean-mode sections written for the model, not for a reader: the compacted turns' user messages
# quoted again, and the footer naming ``session_search`` (``ContextCompressor._augment_summary_lean``).
_MODEL_ONLY_SECTIONS = (_LEAN_USER_MESSAGES_HEADING, _LEAN_RECOVERY_HEADING)
_HISTORICAL_ONLY_LINE = "Historical only; newer protected-tail messages after this summary win."
_COMPACTED_COUNT = re.compile(r"The (\d+) compacted message\(s\) remain fully preserved")


def compaction_summary(message: Any) -> Optional[dict[str, Any]]:
    """What a compaction handoff tells a reader: ``{summary, compacted_messages}``, or ``None`` for any
    other message. ``summary`` is the summary as the compressor wrote it, without the model-facing
    preamble and end marker, prior-tail content or live ask merged into the carrier, the lean
    sections above and the snapshot's ranking line; ``compacted_messages`` is the count the recovery
    footer names (``None`` when it has none)."""
    if not is_compaction_summary_message(message):
        return None
    text = ContextCompressor._strip_summary_prefix(flatten_message_text(message.get("content")))
    sections = {}
    for heading in _MODEL_ONLY_SECTIONS:
        start = text.find(heading)
        if start >= 0:
            end = text.find("\n## ", start + len(heading))
            sections[heading] = text[start:end if end >= 0 else len(text)]
            text = text[:start] + (text[end + 1:] if end >= 0 else "")
    # The footer's count, never one a quoted user message happens to spell out (the quotes are cut first).
    counts = _COMPACTED_COUNT.findall(sections.get(_LEAN_RECOVERY_HEADING, text))
    text = "\n".join(line for line in text.splitlines() if line.strip() != _HISTORICAL_ONLY_LINE)
    return {"summary": re.sub(r"\n{3,}", "\n\n", text).strip(),
            "compacted_messages": int(counts[-1]) if counts else None}
