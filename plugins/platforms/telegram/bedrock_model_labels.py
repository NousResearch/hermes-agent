#!/usr/bin/env python3
"""Display-only labels and vendor grouping for Bedrock model IDs in inline pickers.

Bedrock advertises the same model several times: as a plain foundation-model ID
(``anthropic.claude-opus-5``) and behind one or more routing namespaces
(``us.``, ``global.``, ...). Rendered raw in a two-column Telegram keyboard those
IDs truncate to indistinguishable buttons (issue #94986).

Pure functions on purpose — no Telegram objects, no adapter state — so the label
contract is testable without a bot. **Model IDs are never rewritten**: callers
select through positional indices into their own list, so the exact ID the
provider advertised is what reaches Bedrock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Vendor segments are matched against this map rather than inferred from position:
# model names contain dots too (``openai.gpt-5.6-terra``, ``zai.glm-4.7``), and a
# positional guess reads the version as a vendor ("Gpt-5"). Values are display labels.
VENDOR_LABELS = {
    "ai21": "AI21",
    "amazon": "Amazon",
    "anthropic": "Anthropic",
    "cohere": "Cohere",
    "deepseek": "DeepSeek",
    "google": "Google",
    "meta": "Meta",
    "minimax": "MiniMax",
    "mistral": "Mistral",
    "moonshot": "Moonshot AI",
    "moonshotai": "Moonshot AI",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "qwen": "Qwen",
    "stability": "Stability",
    "twelvelabs": "TwelveLabs",
    "writer": "Writer",
    "xai": "xAI",
    "zai": "Z.ai",
}

# Distinct segments that denote one vendor, folded so no label appears twice.
VENDOR_ALIASES = {"moonshot": "moonshotai"}

# Telegram truncates long button text; keep labels inside the usable width.
_MAX_LABEL = 38


def split_bedrock_id(model_id: str) -> Tuple[str, str, str]:
    """Split a Bedrock ID into ``(geo, vendor, model)``.

    Recognises ``<vendor>.<model>`` and ``<geo>.<vendor>.<model>``; the vendor is
    read from :data:`VENDOR_LABELS` at those two positions only. Returns
    ``("", "", <id>)`` for anything else, which is how callers keep non-Bedrock
    providers untouched and free of a vendor drill-down.
    """
    short = model_id.split("/")[-1] if "/" in model_id else model_id
    parts = short.split(".")
    # A trailing empty segment means there is no model name to show, so such a
    # degenerate ID must fall through to the verbatim branch, never to an empty label.
    if parts[0] in VENDOR_LABELS and len(parts) >= 2 and parts[1]:
        return "", parts[0], ".".join(parts[1:])
    if len(parts) >= 3 and parts[1] in VENDOR_LABELS and parts[2]:
        return parts[0], parts[1], ".".join(parts[2:])
    return "", "", short


def canonical_vendor(vendor: str) -> str:
    """Fold alias spellings onto one vendor key."""
    return VENDOR_ALIASES.get(vendor, vendor)


def _display_short(vendor: str, short: str) -> str:
    # Inside the Anthropic page the vendor is already named, so the repeated
    # ``claude-`` prefix only costs width.
    return short.removeprefix("claude-") if vendor == "anthropic" else short


def model_button_labels(models: List[str]) -> List[str]:
    """Readable, pairwise-distinct labels for *models*, one per entry, same order.

    The routing namespace is dropped when the model name alone is unambiguous and
    kept as a short ``geo:`` prefix when the same model appears more than once —
    a bare foundation-model ID keeps no prefix, which is what distinguishes it
    from its routed twins. Collisions are counted over the list the caller
    renders, so labels do not change as the user pages.
    """
    parsed = [split_bedrock_id(m) for m in models]
    counts: Dict[str, int] = {}
    for _geo, vendor, short in parsed:
        key = _display_short(vendor, short)
        counts[key] = counts.get(key, 0) + 1

    labels: List[str] = []
    for model_id, (geo, vendor, short) in zip(models, parsed):
        label = _display_short(vendor, short)
        if vendor and geo and counts.get(label, 0) > 1:
            # ``G`` keeps the useful global/regional distinction without eating a
            # whole button's width.
            label = f"{'G' if geo == 'global' else geo}: {label}"
        if not label:  # never emit blank text: Telegram rejects the whole message
            label = model_id
        if len(label) > _MAX_LABEL:
            label = label[: _MAX_LABEL - 3] + "..."
        labels.append(label)
    return labels


def group_models_by_vendor(models: List[str]) -> List[Dict[str, Any]]:
    """Vendor-sorted ``[{vendor, label, indices}]`` for the Bedrock IDs in *models*.

    ``indices`` are positions in *models*, so a caller scopes a sub-list without
    ever rewriting an ID. Empty when the list carries no Bedrock-shaped IDs —
    the signal not to insert the drill-down step at all.
    """
    groups: Dict[str, List[int]] = {}
    for i, model_id in enumerate(models):
        _geo, vendor, _short = split_bedrock_id(model_id)
        if vendor:
            groups.setdefault(canonical_vendor(vendor), []).append(i)
    return [
        {"vendor": vendor, "label": VENDOR_LABELS[vendor], "indices": indices}
        for vendor, indices in sorted(groups.items())
    ]
