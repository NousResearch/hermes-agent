"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from agent.context_compressor import ContextCompressor, is_compaction_summary_message


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

_PROJECTION_PERSISTENCE_FIELDS = {
    "id",
    "session_id",
    "active",
    "compacted",
    "_compressed_summary",
    "api_content",
    "_row_id",
}


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


def suppress_redundant_compaction_projection_sources(
    messages: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Drop only a compacted row proven redundant by its display carrier.

    Raw transcript identity remains untouched. This projection-only rule is
    deliberately narrow: an active structured carrier must immediately follow
    an archived row in the same session, retain its timestamp, and project to
    the same complete client-visible payload. Distinct reasoning, tool state,
    metadata, or a non-adjacent/equal message prevents suppression.
    """
    rows = list(messages)
    suppressed_indices: set[int] = set()

    def _payload(message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in message.items()
            if key not in _PROJECTION_PERSISTENCE_FIELDS
            and value is not None
        }

    for carrier_index in range(1, len(rows)):
        source = rows[carrier_index - 1]
        carrier = rows[carrier_index]
        if (
            not source.get("compacted")
            or source.get("active")
            or not carrier.get("active")
            or source.get("session_id") != carrier.get("session_id")
            or source.get("timestamp") is None
            or source.get("timestamp") != carrier.get("timestamp")
            or not is_compaction_summary_message(carrier)
        ):
            continue
        projected = project_compaction_message_for_display(carrier)
        if projected is None or _payload(source) != _payload(projected):
            continue
        suppressed_indices.add(carrier_index - 1)

    return [
        message
        for index, message in enumerate(rows)
        if index not in suppressed_indices
    ]
