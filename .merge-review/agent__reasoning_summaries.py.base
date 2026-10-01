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

__all__ = ["append_streamed_reasoning_detail", "separate_glued_reasoning_blocks"]


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
