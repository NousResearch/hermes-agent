"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.context_compressor import (
    ContextCompressor,
    _INFLIGHT_TASK_REPLAY_HEADER,
    is_compaction_summary_message,
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
    embedded in the carrier.
    """
    if not isinstance(message, dict):
        return None
    if not is_compaction_summary_message(message):
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    projected.pop("display_kind", None)
    return projected


def _without_inflight_replay_header(content: Any) -> Optional[Any]:
    """Copy replay content without its synthetic header, or ``None`` when absent."""
    if isinstance(content, str):
        leading = content.lstrip()
        if not leading.startswith(_INFLIGHT_TASK_REPLAY_HEADER):
            return None
        return leading[len(_INFLIGHT_TASK_REPLAY_HEADER):].lstrip()
    if not isinstance(content, list) or not content:
        return None
    first, *tail = content
    text = first if isinstance(first, str) else first.get("text") if isinstance(first, dict) else None
    if not isinstance(text, str):
        return None
    leading = text.lstrip()
    if not leading.startswith(_INFLIGHT_TASK_REPLAY_HEADER):
        return None
    remainder = leading[len(_INFLIGHT_TASK_REPLAY_HEADER):].lstrip()
    if not remainder:
        return tail
    rewritten = remainder if isinstance(first, str) else {**first, "text": remainder}
    return [rewritten, *tail]


def inflight_replay_content_for_display(message: Dict[str, Any]) -> Optional[Any]:
    """Return task content restated by an in-flight compaction replay.

    The replay is model-facing recovery state, not a second authored user
    turn. Display projections use this helper to suppress it when the
    authentic still-open turn is present, while retaining a clean fallback
    when history paging omitted that original row.
    """
    projected = project_compaction_message_for_display(message)
    if projected is None or projected.get("role") != "user":
        return None
    return _without_inflight_replay_header(projected.get("content"))
