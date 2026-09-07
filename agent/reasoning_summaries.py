"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI gpt-5.x and Responses-API relays onto the chat wire) emit one
``reasoning_content`` delta per *completed* summary part, each opening with a bold heading.
The chat wire lacks the Responses API's ``summary_index`` delimiter (verified live on Nous
Portal ``openai/gpt-5.6-sol``), so plain concatenation glues ``**One****Two**`` into one
half-bold paragraph. The boundary is re-derived from a delta opening a bold heading, matching
the blank-line join Hermes' own Responses adapter does.
"""

from __future__ import annotations

from typing import Any

from agent.message_content import flatten_message_text

__all__ = ["separate_glued_reasoning_blocks"]


def _as_reasoning_text(value: Any) -> str:
    """Providers sometimes emit reasoning as a list of content parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return flatten_message_text(value, sep="")


def separate_glued_reasoning_blocks(previous: Any, delta: Any) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    A break is inserted when *delta* opens a *closed* bold heading and *previous* is mid-line
    (heading butting heading, or prose butting heading). Token-streamed reasoning is left
    alone: its deltas carry their own whitespace, and a fragment that merely opens emphasis
    (``**`` alone) is not a part boundary — summary parts carry the whole heading in one delta.

    Some OpenAI-compatible relays (Grok / custom gateways) emit ``reasoning_content`` as a
    list of content parts instead of a string. Flatten those shapes first so a list delta
    cannot crash the stream loop with ``'list' object has no attribute 'startswith'``.
    """
    previous = _as_reasoning_text(previous)
    delta = _as_reasoning_text(delta)
    glued = previous and delta and not previous[-1].isspace() and delta.startswith("**") and "**" in delta[2:]
    return f"\n\n{delta}" if glued else delta
