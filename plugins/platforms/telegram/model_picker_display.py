#!/usr/bin/env python3
"""Display-only shaping for Bedrock IDs in the Telegram ``/model`` picker.

Bedrock advertises the same model several times: as a plain foundation-model ID
(``anthropic.claude-opus-5``) and behind one or more routing namespaces
(``us.``, ``global.``, ...). Rendered raw in a two-column inline keyboard those
IDs truncate to indistinguishable buttons, which makes picking a specific model
guesswork rather than a choice (issue #94986).

Pure functions on purpose — no Telegram objects, no adapter state — so the
display contract is testable without a bot. **Model IDs are never rewritten**:
callers select through positional indices into their own list, so the exact ID
the provider advertised is what reaches Bedrock.
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

# Vendor bucket for Bedrock-shaped IDs whose vendor segment we do not know, and
# for IDs that are not Bedrock-shaped at all. Never a key of VENDOR_LABELS, so a
# real vendor can never land in it by accident.
OTHER_VENDOR = "other"
OTHER_VENDOR_LABEL = "Other"

# Two columns of a mobile inline keyboard show roughly this much before the
# label is ellipsized — past it a button gets a full-width row of its own so a
# trailing model count stays readable (#94986).
TWO_COLUMN_BUDGET = 18


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


def _clamp(label: str, tail_len: int = 0, limit: int = _MAX_LABEL) -> str:
    """Fit *label* in *limit* characters, keeping both ends.

    Bedrock's long IDs differ in their TAIL (``…-preview-20260101-v1:0`` vs
    ``…-20260202-v1:0``), so a head-only clamp deletes exactly the part that
    tells two buttons apart and re-creates the collision these labels exist to
    remove. Eliding the middle keeps the head, which says what the model is, and
    the tail, which says which variant it is. *tail_len* forces a wider tail when
    the caller knows the discriminating text sits further left.
    """
    if len(label) <= limit:
        return label
    keep = limit - 1  # one character for the ellipsis
    tail = min(keep, max(tail_len, keep // 2))
    return f"{label[:keep - tail]}…{label[len(label) - tail:]}"


def _clamp_distinct(labels: List[str], limit: int = _MAX_LABEL) -> List[str]:
    """Clamp *labels* to *limit*, keeping already-distinct entries distinct.

    Clamping is the last step before a button is rendered, so it is where an
    otherwise-correct label set can still collapse into identical buttons. For
    each colliding group the tail is widened until it covers the first character
    where the members differ, which is the minimum the user needs to tell them
    apart. Order is preserved: callbacks are positional.
    """
    clamped = [_clamp(label, limit=limit) for label in labels]
    groups: Dict[str, List[int]] = {}
    for i, label in enumerate(clamped):
        groups.setdefault(label, []).append(i)

    for idx in groups.values():
        if len(idx) < 2:
            continue
        originals = [labels[i] for i in idx]
        if len(set(originals)) < len(originals):
            continue  # genuinely identical input: no truncation can separate them
        # First position where the group's members stop agreeing.
        shortest = min(len(o) for o in originals)
        diff_at = next((p for p in range(shortest) if len({o[p] for o in originals}) > 1), shortest)
        for i in idx:
            clamped[i] = _clamp(labels[i], tail_len=len(labels[i]) - diff_at, limit=limit)
    return clamped


def _geo_prefix(geo: str) -> str:
    # ``G`` keeps the global/regional distinction without eating a whole button.
    return "G" if geo == "global" else geo


def model_button_labels(models: List[str]) -> List[str]:
    """Readable, pairwise-distinct labels for *models*, one per entry, same order.

    The routing namespace is dropped when the model name alone is unambiguous and
    kept as a short ``geo:`` prefix when the same model is advertised more than
    once. A bare foundation-model ID keeps NO prefix: it does not carry a
    namespace, and lending it the configured region would make it identical to
    its regional twin (``us.amazon.nova-lite-v1:0`` vs ``amazon.nova-lite-v1:0``
    both rendering ``us: nova-lite-v1:0``). :func:`routing_legend` names that
    unprefixed shape in the message body instead, where it costs no button width.

    Collisions are counted over the list the caller renders, on the label as
    displayed, so labels neither change as the user pages nor collide after the
    vendor prefix is stripped.
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
            label = f"{_geo_prefix(geo)}: {label}"
        if not label:  # never emit blank text: Telegram rejects the whole message
            label = model_id
        labels.append(label)
    # Clamped last, and collision-aware: the width limit is itself a way to turn
    # two distinct labels into one indistinguishable button.
    return _clamp_distinct(labels)


def routing_legend(models: List[str], region_geo: str = "") -> str:
    """One-line explanation of the ``geo:`` prefixes, or ``""`` when unneeded.

    A bare label is only self-explanatory next to a legend, so one is offered
    exactly when the rendered list mixes a namespaced ID with the plain
    foundation-model ID of the same model — the case where the user must choose
    between routing scopes rather than between models.
    """
    parsed = [split_bedrock_id(m) for m in models]
    bare = {_display_short(v, s) for g, v, s in parsed if v and not g}
    geos = sorted({g for g, v, s in parsed if v and g and _display_short(v, s) in bare})
    if not geos:
        return ""
    scopes = ", ".join(f"{_geo_prefix(g)}: = {g}" for g in geos)
    in_region = f"in-region ({region_geo})" if region_geo else "in-region"
    return f"{scopes}, no prefix = {in_region}"


def configured_region_geo() -> str:
    """Short geography of the Bedrock endpoint this runtime calls (``""`` unknown).

    Follows the runtime's own region precedence so the legend describes the
    endpoint Hermes will actually reach, and deliberately shows the geography
    rather than the full region name, which would not fit.
    """
    try:
        from agent.bedrock_adapter import resolve_bedrock_runtime_region
        from hermes_cli.model_setup_flows_bedrock import bedrock_region_geo_prefix

        return bedrock_region_geo_prefix(resolve_bedrock_runtime_region()).rstrip(".")
    except Exception:
        return ""


def group_models_by_vendor(models: List[str]) -> List[Dict[str, Any]]:
    """Vendor-sorted ``[{vendor, label, indices}]`` for the Bedrock IDs in *models*.

    ``indices`` are positions in *models*, so a caller scopes a sub-list without
    ever rewriting an ID. Empty when the list carries no Bedrock-shaped IDs —
    the signal not to insert the drill-down step at all.

    When at least one known vendor IS present the result **partitions** the whole
    list: everything else lands in one trailing ``Other`` group. ``indices`` is
    the only route from a vendor button to a model, so an ID left out of every
    group becomes unselectable — which is what would happen to an unreleased
    Bedrock vendor, or to a plain ID sitting beside namespaced ones.
    """
    groups: Dict[str, List[int]] = {}
    unknown: List[int] = []
    for i, model_id in enumerate(models):
        _geo, vendor, _short = split_bedrock_id(model_id)
        if vendor:
            groups.setdefault(canonical_vendor(vendor), []).append(i)
        else:
            unknown.append(i)
    if not groups:
        # No known vendor: this is not a Bedrock-shaped catalog, so it gets no
        # drill-down rather than a single pointless ``Other`` button.
        return []
    out = [
        {"vendor": vendor, "label": VENDOR_LABELS[vendor], "indices": indices}
        for vendor, indices in sorted(groups.items())
    ]
    if unknown:
        out.append({"vendor": OTHER_VENDOR, "label": OTHER_VENDOR_LABEL, "indices": unknown})
    return out


def pack_rows(labels: List[str], budget: int = TWO_COLUMN_BUDGET) -> List[List[int]]:
    """Row layout for *labels* as index groups: pairs, or a full-width row.

    Two columns halve the readable width, which is what ellipsized
    ``✓ AWS Bedrock (1…`` and hid the model count. A label that does not fit a
    column takes a row of its own instead of being truncated; order is preserved
    so a caller's positional callbacks stay valid.
    """
    rows: List[List[int]] = []
    pending: List[int] = []
    for i, label in enumerate(labels):
        if len(label) > budget:
            if pending:
                rows.append(pending)
                pending = []
            rows.append([i])
            continue
        pending.append(i)
        if len(pending) == 2:
            rows.append(pending)
            pending = []
    if pending:
        rows.append(pending)
    return rows
