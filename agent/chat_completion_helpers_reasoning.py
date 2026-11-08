"""Reasoning-only helpers for chat-completions response materialization and streaming."""

from __future__ import annotations

from typing import Any

from agent.message_content import flatten_message_text

from agent.reasoning_summaries import (
    ReasoningDeltaAccumulator,
    append_streamed_reasoning_detail,
    normalize_reasoning_delta,
    separate_glued_reasoning_blocks,
    streamed_reasoning_detail_text,
)


def stream_reasoning_delta(
    owner: Any,
    delta: Any,
    reasoning: ReasoningDeltaAccumulator,
    detail_display_parts: list[str],
    reasoning_details: list,
    reasoning_watch: Any,
    detail_watch: Any,
    carriers: Any,
) -> bool:
    """Accumulate and display one chunk's canonical reasoning representation."""
    reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    # Same model_extra fallback as the non-streaming path, for clients that park unknown
    # reasoning fields instead of exposing them as attributes.
    model_extra = getattr(delta, "model_extra", None)
    if reasoning_text is None and isinstance(model_extra, dict):
        reasoning_text = model_extra.get("reasoning_content") or model_extra.get("reasoning")
    # Carrier capture sees even signature-only deltas. Watches consume the raw
    # channels before display dedup can hide a model that keeps emitting a loop.
    reasoning_text = carriers.feed(delta, reasoning_text)
    plain_text = flatten_message_text(reasoning_text, sep="")
    runaway_reasoning = reasoning_watch.feed(plain_text)
    plain_display = ""
    if reasoning_text:
        plain_display = reasoning.feed(reasoning_text)

    rd_delta = getattr(delta, "reasoning_details", None)
    if rd_delta is None and isinstance(model_extra, dict):
        rd_delta = model_extra.get("reasoning_details")
    detail_text_parts = []
    for detail in rd_delta if isinstance(rd_delta, (list, tuple)) else ():
        detail_text_parts.append(streamed_reasoning_detail_text(detail))
        append_streamed_reasoning_detail(reasoning_details, detail)
    raw_detail_text = "".join(detail_text_parts)
    runaway_details = detail_watch.feed(raw_detail_text)
    # The plain accumulator is for persistence; details can have already advanced
    # the live display beyond it. Compare both readable channels with emitted text.
    emitted = "".join(detail_display_parts)
    plain_new = normalize_reasoning_delta(emitted, plain_text) if plain_display else ""
    detail_new = normalize_reasoning_delta(emitted, raw_detail_text)
    display_raw = detail_new if len(detail_new) > len(plain_new) else plain_new
    if display_raw:
        display_reasoning = separate_glued_reasoning_blocks(
            detail_display_parts[-1] if detail_display_parts else "", display_raw)
        detail_display_parts.append(display_raw)
        owner._emit_reasoning(display_reasoning)
    return runaway_reasoning or runaway_details
