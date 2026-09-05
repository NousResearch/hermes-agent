"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.context_compressor import (
    ContextCompressor, is_compaction_summary_message, user_originated_turn_view,
)


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


def project_compaction_message_for_display(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier. User rows carry the canonical human-turn
    classification derived from the original row, before display unwrapping.
    """
    if not isinstance(message, dict):
        return None
    if is_compaction_summary_message(message):
        projected = ContextCompressor._strip_context_summary_handoff_message(message)
        if projected is None:
            return None
        projected = projected.copy()
        for key in _COMPACTION_INTERNAL_FIELDS:
            projected.pop(key, None)
        projected.pop("display_kind", None)
    else:
        projected = message.copy()

    if message.get("role") == "user":
        # Classify the physical row before unwrapping its display content. A
        # composite carrier can contain a real ask even with a hidden wrapper;
        # a typed runtime notice can contain ordinary-looking prose. The same
        # canonical view owns backend retry/undo ordinals. This field belongs
        # only to the fresh display copy, never model history or persistence.
        projected["user_originated"] = user_originated_turn_view(message) is not None
    return projected
