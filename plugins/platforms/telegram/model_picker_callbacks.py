#!/usr/bin/env python3
"""Selection payloads for the Telegram ``/model`` picker.

Telegram never retires an inline keyboard: every button of every picker message
a chat has shown stays tappable for as long as that message exists. The picker
meanwhile rewrites the model sub-list it has in scope on each vendor
drill-down, each Back, and each new ``/model``.

A payload naming only a position in that scoped sub-list therefore resolved
against whichever list happened to be current when the tap arrived — the Amazon
page's ``nova-lite-v1:0`` button switched the session to
``global.anthropic.claude-opus-5`` after one Back, and a button from a
superseded picker selected out of a catalog the user never opened (#94990
review).

So a payload carries two fields: the *listing* it was drawn from, and a
position in that listing's **full** model list, which the vendor step only ever
slices and never reorders. The listing id makes a tap from an older listing
refusable; the canonical position keeps a still-valid button on the exact ID it
displayed, whatever the picker has in scope now.

Pure functions — no Telegram objects, no adapter state — so the wire format and
its refusals are testable without a bot.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# ``<listing>:<canonical index>``. Telegram caps callback_data at 64 bytes; two
# small integers plus the ``mm:``/``mc:`` prefix stay far inside it.
_SEPARATOR = ":"

# Shown when a tap cannot be resolved to the model its button displayed. Both
# refusals name the remedy, because a silent ``answer()`` looks like the tap worked.
STALE_LISTING_NOTICE = "This list is out of date — use /model again."
UNKNOWN_SELECTION_NOTICE = "Invalid selection — use /model again."


def selection_payload(listing: int, index: int) -> str:
    """Encode *listing* + canonical *index* for an ``mm:``/``mc:`` button."""
    return f"{int(listing)}{_SEPARATOR}{int(index)}"


def canonical_indices(count: int) -> List[int]:
    """Identity mapping for an unscoped listing of *count* models."""
    return list(range(count))


def parse_selection(raw: str) -> Optional[Tuple[int, int]]:
    """Decode a payload into ``(listing, index)``; ``None`` when it is not one.

    A bare index is the pre-fix format, still live on any keyboard drawn before
    the gateway reloaded; it decodes to ``None`` because there is no listing to
    check it against.
    """
    listing_raw, separator, index_raw = raw.partition(_SEPARATOR)
    if not separator:
        return None
    try:
        return int(listing_raw), int(index_raw)
    except ValueError:
        return None


def resolve_selection(
    raw: str, listing: Optional[int], models: Sequence[str]
) -> Tuple[Optional[str], str]:
    """``(model_id, refusal)`` for the payload *raw* against the full *models*.

    Exactly one side is set. A payload from another listing, or one carrying no
    listing at all, is refused rather than resolved: with nothing to check it
    against, resolving it is the silent mis-selection this format exists to
    prevent.
    """
    parsed = parse_selection(raw)
    if parsed is None:
        # No listing to check, or a non-numeric field: either way the index has
        # no meaning here. Both read as "that list is gone" to the user.
        return None, STALE_LISTING_NOTICE
    tapped_listing, index = parsed
    if listing is None or tapped_listing != listing:
        return None, STALE_LISTING_NOTICE
    if index < 0 or index >= len(models):
        return None, UNKNOWN_SELECTION_NOTICE
    return models[index], ""
