"""Deterministic trait-blending index (Phase 3).

Blends two or more rendering cards by merging their trait vectors with
explicit conflict-resolution rules, producing a neutral prompt_fragment plus
the chosen traits and a list of resolution notes. Fully deterministic for a
given input and seed (same inputs + same seed -> identical output).

Conflict-resolution rules (locked decisions):
  * medium conflicts -> dominant wins (higher confidence; tie -> earlier input).
  * incompatible palettes -> weighted blend of hex swatches + joined mood.
  * opposing tags -> higher-confidence style's tag wins, conflict recorded.
  * otherwise -> union / conjunction of compatible traits.
"""
from __future__ import annotations

import random
import re
from typing import Any

# Families of medium descriptors. Two mediums from different families are
# treated as a conflict (dominant wins). Same family -> merged descriptor.
_MEDIUM_FAMILIES = {
    "photography": ["photography", "photo", "film", "still"],
    "painting": ["paint", "watercolour", "watercolor", "gouache", "acrylic",
                  "oil", "fresco", "impasto", "encaustic"],
    "printmaking": ["print", "linocut", "woodcut", "woodblock", "etching",
                     "lithograph", "screenprint", "risograph", "engraving",
                     "monotype", "mezzotint", "letterpress"],
    "drawing": ["draw", "sketch", "pencil", "charcoal", "pastel", "ink",
                 "crayon", "graphite", "marker", "doodle", "pen"],
    "graphic-design": ["graphic", "design", "poster", "typograph", "layout",
                        "logo", "branding", "infographic", "diagram", "chart",
                        "blueprint", "wireframe", "schematic", "plan"],
    "digital": ["digital", "render", "3d", "cgi", "pixel", "low-poly"],
    "illustration": ["illustration", "comic", "cartoon", "manga", "anime",
                      "animation", "still"],
    "sculpture": ["sculpt", "installation", "relief", "carving", "ceramic"],
    "textile": ["textile", "embroidery", "weaving", "tapestry", "lace",
                 "knitted", "crocheted", "quilted", "macrame"],
    "other": [],
}

# Tag opposition pairs: if both sides appear across the blended inputs, the
# higher-confidence side wins and the other is dropped (recorded as a note).
_TAG_CONFLICTS = [
    ({"monochrome", "black-and-white", "bw-monochrome"},
     {"vivid-palette", "psychedelic-palette", "vivid", "colorful"}),
    ({"realistic", "photorealism"}, {"abstract", "surrealism", "geometric"}),
    ({"minimalist", "minimalism"}, {"ornate", "baroque", "rococo", "detailed"}),
    ({"dark", "moody-palette", "moody"}, {"light-palette", "pastel-palette",
      "cute", "bright"}),
    ({"muted-palette", "subdued"}, {"vivid-palette", "psychedelic-palette"}),
]


def _norm(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().lower().replace("-", " "))


def _family(medium: str) -> str:
    m = medium.lower()
    for fam, keys in _MEDIUM_FAMILIES.items():
        if any(k in m for k in keys):
            return fam
    return "other"


def trait_vector(card: dict[str, Any]) -> dict[str, Any]:
    """Extract a trait vector from a rendering card (tags + fields)."""
    palette = card.get("palette") or {}
    return {
        "slug": card["slug"],
        "medium": card.get("medium") or "",
        "composition": card.get("composition") or "",
        "line_and_shape": card.get("line_and_shape") or "",
        "palette_mood": palette.get("mood") or "",
        "palette_hex": palette.get("hex") or [],
        "lighting_and_texture": card.get("lighting_and_texture") or "",
        "mood_and_era": card.get("mood_and_era") or "",
        "tags": set(card.get("tags") or []),
        "confidence": card.get("confidence") or "medium",
    }


_CONF_RANK = {"high": 2, "medium": 1, "low": 0}


def _dominant(inputs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Highest-confidence input; tie-break by earlier position (stable)."""
    best = inputs[0]
    for inp in inputs[1:]:
        r = _CONF_RANK.get(inp["trait"][key], 0) - _CONF_RANK.get(best["trait"][key], 0)
        if r > 0:
            best = inp
    return best


def _merge_medium(inputs: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Merge mediums; conflict -> dominant wins. Return (medium, notes)."""
    fams = {_family(i["trait"]["medium"]) for i in inputs}
    notes: list[str] = []
    if len(fams) == 1 and fams != {"other"}:
        # same family -> join descriptors with 'with'
        mediums = list(dict.fromkeys(i["trait"]["medium"] for i in inputs))
        merged = " with ".join(mediums)
        return merged, notes
    # conflict across families -> dominant (highest confidence) wins
    dom = _dominant(inputs, "confidence")
    losers = [i for i in inputs if i is not dom]
    if losers:
        notes.append(
            f"medium conflict ({', '.join(i['trait']['medium'] for i in inputs)}) "
            f"-> dominant wins: {dom['trait']['medium']!r}"
        )
    return dom["trait"]["medium"], notes


def _merge_palette(inputs: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], list[str]]:
    """Weighted blend of hex swatches; joined mood. Returns (palette, notes)."""
    notes: list[str] = []
    rng = random.Random(seed)
    # weighted interleave: higher confidence contributes more swatches
    weighted: list[str] = []
    for inp in inputs:
        hexes = inp["trait"]["palette_hex"]
        w = _CONF_RANK.get(inp["trait"]["confidence"], 0) + 1
        weighted.extend(hexes * w)
    rng.shuffle(weighted)
    merged_hex: list[str] = []
    for h in weighted:
        if h not in merged_hex:
            merged_hex.append(h)
        if len(merged_hex) >= 8:
            break
    moods = [i["trait"]["palette_mood"] for i in inputs if i["trait"]["palette_mood"]]
    # strip a leading "palette of "/"palette" and trailing " palette" from each mood
    stripped = []
    for m in moods:
        mm = re.sub(r"^(palette of|palette)\s+", "", m.strip(), flags=re.IGNORECASE)
        mm = re.sub(r"\s+palette$", "", mm, flags=re.IGNORECASE)
        if mm and mm not in stripped:
            stripped.append(mm)
    mood = " blended with ".join(stripped) if stripped else ""
    if len(stripped) > 1:
        notes.append(
            "palette conflict -> weighted hex blend, moods joined ("
            + "; ".join(moods) + ")"
        )
    return {"mood": mood, "hex": merged_hex}, notes


def _merge_tags(inputs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Union tags with conflict resolution. Returns (tags, notes)."""
    notes: list[str] = []
    tags: set[str] = set()
    for inp in inputs:
        tags |= inp["trait"]["tags"]
    for left, right in _TAG_CONFLICTS:
        l_present = {t for t in tags if _norm(t) in {_norm(x) for x in left}}
        r_present = {t for t in tags if _norm(t) in {_norm(x) for x in right}}
        if l_present and r_present:
            # resolve by confidence: which style contributed the higher conf side
            l_conf = max(_CONF_RANK.get(i["trait"]["confidence"], 0)
                         for i in inputs if i["trait"]["tags"] & l_present)
            r_conf = max(_CONF_RANK.get(i["trait"]["confidence"], 0)
                         for i in inputs if i["trait"]["tags"] & r_present)
            winner, loser = (l_present, r_present) if l_conf >= r_conf else (r_present, l_present)
            tags -= loser
            notes.append(
                f"tag conflict {sorted(l_present)} vs {sorted(r_present)} -> "
                f"{'kept ' + str(sorted(winner)) + ' dropped ' + str(sorted(loser))}"
            )
    return sorted(tags), notes


def _merge_text(inputs: list[dict[str, Any]], key: str, joiner: str = "; ") -> str:
    parts = [i["trait"][key] for i in inputs if i["trait"].get(key)]
    return joiner.join(dict.fromkeys(parts))


def blend(cards: list[dict[str, Any]], seed: int = 42,
          weights: list[float] | None = None) -> dict[str, Any]:
    """Blend 2+ rendering cards into one neutral fragment + trait report.

    Deterministic: same cards + same seed -> identical dict.
    """
    if len(cards) < 2:
        raise ValueError("blend requires at least 2 cards")
    inputs = [{"card": c, "trait": trait_vector(c)} for c in cards]
    notes: list[str] = []

    medium, n1 = _merge_medium(inputs)
    notes += n1
    palette, n2 = _merge_palette(inputs, seed)
    notes += n2
    tags, n3 = _merge_tags(inputs)
    notes += n3

    composition = _merge_text(inputs, "composition")
    line = _merge_text(inputs, "line_and_shape")
    lighting = _merge_text(inputs, "lighting_and_texture")
    era = _merge_text(inputs, "mood_and_era")
    best_for = sorted({b for i in inputs for b in (i["card"].get("best_for") or [])})
    avoid_for = sorted({a for i in inputs for a in (i["card"].get("avoid_for") or [])})

    # confidence of the blend = min of inputs (conservative)
    blend_conf = min((_CONF_RANK[i["trait"]["confidence"]] for i in inputs), default=0)
    confidence = {0: "low", 1: "medium", 2: "high"}[blend_conf]

    prompt_fragment = ", ".join(
        p for p in (medium, composition, line, palette["mood"], lighting, era) if p
    )

    return {
        "seed": seed,
        "inputs": [{"slug": c["slug"], "style_label": _neutral_label(c),
                    "confidence": c.get("confidence")} for c in cards],
        "chosen_traits": {
            "medium": medium,
            "composition": composition,
            "line_and_shape": line,
            "palette": palette,
            "lighting_and_texture": lighting,
            "mood_and_era": era,
            "tags": tags,
            "best_for": best_for,
            "avoid_for": avoid_for,
            "confidence": confidence,
        },
        "resolution_notes": notes,
        "prompt_fragment": prompt_fragment,
    }


def _neutral_label(card: dict[str, Any]) -> str:
    """Reuse the card's neutral registry label if present, else medium+mood."""
    label = (card.get("style_label") or "").strip()
    if label:
        return label
    medium = (card.get("medium") or "").strip()
    mood = (card.get("mood_and_era") or "").strip()
    return medium if not mood else f"{medium}; {mood}"
