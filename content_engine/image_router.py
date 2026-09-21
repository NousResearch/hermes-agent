"""Editorial Decision Engine — Visual Content Pipeline 2.0 routing layer.

Classifies content into (intent, narrative, layout, composition, studio) and
maps the result into existing Baoyu generation parameters.

Sits ABOVE the existing generation pipeline. Does NOT replace it.

Usage:
    from image_router import decide

    result = decide({
        "title": "Mixture of Experts vs Mixture of Agents",
        "body_text": "...",
        "brand": "sahil_twitter",
        "platform": "twitter",
        "category": "ai-research",
        "pillar": "comparison",
    })
    # result = {
    #   "intent": "comparison",
    #   "narrative": "systems",
    #   "layout": "network",
    #   "composition": "split_narrative",
    #   "studio": "chromatic-institute",
    #   "baoyu_type": "comparison",
    #   "baoyu_style": "network-map",
    #   "palette": "chromatic-research",
    #   "mood": "analytical",
    #   "aspect": "16:9",
    # }
"""

from __future__ import annotations
from typing import Optional


# ── Intent classification ──

_INTENT_CUES = [
    ("journey", ("how i built", "how we built", "my journey", "our journey", "from scratch", "i built a", "we built a")),
    ("comparison", (" vs ", " vs.", "versus", "comparison", "choose between", "better than", "compared to", "a vs b", "x vs y")),
    ("timeline", ("timeline", "history of", "evolution", "roadmap", "journey so far", "progress")),
    ("profile", ("who is", "meet ", "profile of", "founder", "creator", "behind ")),
    ("infographic", ("reasons", "ways to", "tips", "checklist", "top ", "metrics", "stats", "numbers")),
    ("social_card", ("thread:", "tip:", "quick ", "short ", "tweet")),
    ("cover", ("issue ", "edition ", "volume ", "cover story")),
    ("divider", ("part ", "chapter ", "section ", "interlude")),
    ("hero", ("introducing", "announcing", "launching", "presenting", "welcome to")),
    ("inline_explanation", ("how", "what is", "why", "explain", "understand", "means")),
]

_NARRATIVE_CUES = [
    ("discovery", ("discover", "explore", "uncover", "reveal", "find", "new")),
    ("observation", ("i noticed", "i observed", "i realised", "what i learned", "reflection")),
    ("mythology", ("legend", "epic", "myth", "saga", "hero", "destiny")),
    ("exploration", ("future", "frontier", "horizon", "next", "beyond", "what if")),
    ("knowledge", ("guide", "tutorial", "reference", "documentation", "how to")),
    ("emotion", ("felt", "struggled", "failed", "overcame", "lonely", "proud")),
    ("systems", ("architecture", "system", "network", "pipeline", "stack", "ecosystem", "infrastructure")),
    ("conflict", ("vs", "versus", "competition", "tradeoff", "battle", "war", "fight")),
    ("transformation", ("before", "after", "evolved", "transformed", "changed", "became")),
    ("construction", ("built", "shipped", "deployed", "implemented", "coded", "engineered")),
    ("conversation", ("interview", "conversation", "discussion", "debate", "roundtable")),
]

# ── Studio mapping ──

# Each studio entry: (studio_name, best_for_narratives, best_for_intents, brand_affinity)
STUDIO_REGISTRY = [
    {
        "name": "mythic-tech-codex-illustration",
        "theme": "Knowledge",
        "best_narratives": {"knowledge", "discovery", "observation"},
        "best_intents": {"hero", "inline_explanation", "profile"},
        "brand_affinity": {"sahil_linkedin", "sahil_twitter"},
    },
    {
        "name": "cosmic-postcard-atelier",
        "theme": "Exploration",
        "best_narratives": {"exploration", "discovery", "transformation"},
        "best_intents": {"hero", "cover", "journey"},
        "brand_affinity": {"sahil_twitter", "sahil_linkedin"},
    },
    {
        "name": "ink-ember-studio",
        "theme": "Emotion",
        "best_narratives": {"emotion", "observation", "conversation"},
        "best_intents": {"profile", "journey", "hero"},
        "brand_affinity": {"sahil_linkedin", "sahil_twitter"},
    },
    {
        "name": "saga-noir-studio",
        "theme": "Legend",
        "best_narratives": {"mythology", "conflict", "transformation"},
        "best_intents": {"hero", "comparison", "cover"},
        "brand_affinity": {"sahil_twitter", "sahil_linkedin"},
    },
    {
        "name": "ninth-observatory",
        "theme": "Place",
        "best_narratives": {"systems", "construction", "exploration"},
        "best_intents": {"inline_explanation", "hero", "journey"},
        "brand_affinity": {"sahil_twitter", "sahil_linkedin"},
    },
    {
        "name": "chromatic-institute",
        "theme": "Pattern",
        "best_narratives": {"systems", "knowledge", "discovery"},
        "best_intents": {"infographic", "comparison", "inline_explanation"},
        "brand_affinity": {"sahil_twitter", "sahil_linkedin"},
    },
    {
        "name": "dark-cyberpunk-hud",
        "theme": "Technical",
        "best_narratives": {"systems", "construction", "knowledge"},
        "best_intents": {"inline_explanation", "social_card", "infographic"},
        "brand_affinity": {"sahil_twitter"},
    },
]

# ── Layout mapping ──

LAYOUT_REGISTRY = {
    "poster": {"best_intents": {"hero", "cover"}, "best_narratives": {"mythology", "conflict"}},
    "multi_panel": {"best_intents": {"inline_explanation", "infographic"}, "best_narratives": {"knowledge", "systems"}},
    "triptych": {"best_intents": {"comparison", "timeline"}, "best_narratives": {"transformation", "conflict"}},
    "diptych": {"best_intents": {"comparison"}, "best_narratives": {"transformation", "conflict"}},
    "blueprint": {"best_intents": {"inline_explanation"}, "best_narratives": {"construction", "systems"}},
    "cross_section": {"best_intents": {"inline_explanation"}, "best_narratives": {"systems", "knowledge"}},
    "flow_diagram": {"best_intents": {"inline_explanation", "journey"}, "best_narratives": {"construction", "systems"}},
    "timeline": {"best_intents": {"timeline", "journey"}, "best_narratives": {"transformation", "discovery"}},
    "council": {"best_intents": {"comparison", "hero"}, "best_narratives": {"conversation", "conflict"}},
    "ecosystem": {"best_intents": {"infographic", "inline_explanation"}, "best_narratives": {"systems", "knowledge"}},
    "network": {"best_intents": {"infographic", "inline_explanation"}, "best_narratives": {"systems", "discovery"}},
    "atlas": {"best_intents": {"journey", "hero"}, "best_narratives": {"exploration", "discovery"}},
    "gallery_plate": {"best_intents": {"profile", "infographic"}, "best_narratives": {"knowledge", "observation"}},
    "collection": {"best_intents": {"infographic", "profile"}, "best_narratives": {"knowledge", "observation"}},
    "hero_environment": {"best_intents": {"hero", "cover"}, "best_narratives": {"exploration", "mythology"}},
    "portrait": {"best_intents": {"profile", "hero"}, "best_narratives": {"emotion", "observation"}},
    "architectural_study": {"best_intents": {"inline_explanation", "hero"}, "best_narratives": {"systems", "exploration"}},
}

# ── Composition mapping ──

COMPOSITION_REGISTRY = {
    "central_hero": {"best_intents": {"hero", "cover", "profile"}, "best_narratives": {"mythology", "conflict"}},
    "wide_landscape": {"best_intents": {"hero", "cover"}, "best_narratives": {"exploration", "discovery"}},
    "interior": {"best_intents": {"profile", "journey"}, "best_narratives": {"emotion", "observation"}},
    "bird_eye": {"best_intents": {"infographic", "inline_explanation"}, "best_narratives": {"systems", "knowledge"}},
    "worm_eye": {"best_intents": {"hero", "cover"}, "best_narratives": {"mythology", "conflict"}},
    "symmetrical": {"best_intents": {"comparison", "infographic"}, "best_narratives": {"knowledge", "systems"}},
    "asymmetrical": {"best_intents": {"hero", "journey"}, "best_narratives": {"emotion", "discovery"}},
    "silhouette": {"best_intents": {"hero", "cover"}, "best_narratives": {"mythology", "conflict"}},
    "collage": {"best_intents": {"infographic", "profile"}, "best_narratives": {"knowledge", "observation"}},
    "environment_first": {"best_intents": {"hero", "inline_explanation"}, "best_narratives": {"exploration", "systems"}},
    "character_first": {"best_intents": {"profile", "hero"}, "best_narratives": {"emotion", "observation"}},
    "radial": {"best_intents": {"infographic", "inline_explanation"}, "best_narratives": {"systems", "knowledge"}},
    "journey_path": {"best_intents": {"journey", "timeline"}, "best_narratives": {"transformation", "discovery"}},
    "split_narrative": {"best_intents": {"comparison", "timeline"}, "best_narratives": {"conflict", "transformation"}},
}

# ── Baoyu mapping ──

# Maps (studio, intent) -> (baoyu_type, baoyu_style, palette, mood)
# Falls back to generic mapping when no specific entry exists.
_BAOYU_MAP = {
    # Mythic Tech Codex
    ("mythic-tech-codex-illustration", "hero"): ("scene", "editorial", "warm", "contemplative"),
    ("mythic-tech-codex-illustration", "inline_explanation"): ("infographic", "scientific", "mono-ink", "analytical"),
    ("mythic-tech-codex-illustration", "infographic"): ("infographic", "scientific", "mono-ink", "analytical"),
    ("mythic-tech-codex-illustration", "comparison"): ("comparison", "ink-notes", "mono-ink", "analytical"),
    ("mythic-tech-codex-illustration", "timeline"): ("timeline", "elegant", "warm", "reflective"),
    ("mythic-tech-codex-illustration", "profile"): ("scene", "editorial", "warm", "contemplative"),
    ("mythic-tech-codex-illustration", "journey"): ("timeline", "elegant", "warm", "reflective"),
    # Cosmic Postcard Atelier
    ("cosmic-postcard-atelier", "hero"): ("scene", "screen-print", "warm", "optimistic"),
    ("cosmic-postcard-atelier", "cover"): ("scene", "screen-print", "warm", "optimistic"),
    ("cosmic-postcard-atelier", "journey"): ("timeline", "retro", "warm", "hopeful"),
    ("cosmic-postcard-atelier", "inline_explanation"): ("infographic", "vector-illustration", "warm", "curious"),
    # Ink & Ember Studio
    ("ink-ember-studio", "hero"): ("scene", "sketch", "mono-ink", "introspective"),
    ("ink-ember-studio", "profile"): ("scene", "sketch", "mono-ink", "introspective"),
    ("ink-ember-studio", "journey"): ("scene", "sketch", "mono-ink", "reflective"),
    ("ink-ember-studio", "inline_explanation"): ("infographic", "ink-notes", "mono-ink", "contemplative"),
    # Saga Noir Studio
    ("saga-noir-studio", "hero"): ("scene", "screen-print", "mono-ink", "epic"),
    ("saga-noir-studio", "comparison"): ("comparison", "ink-notes", "mono-ink", "conflict"),
    ("saga-noir-studio", "cover"): ("scene", "screen-print", "mono-ink", "epic"),
    ("saga-noir-studio", "timeline"): ("timeline", "elegant", "mono-ink", "mythic"),
    # Ninth Observatory
    ("ninth-observatory", "hero"): ("scene", "blueprint", "mono-ink", "monumental"),
    ("ninth-observatory", "inline_explanation"): ("infographic", "blueprint", "mono-ink", "analytical"),
    ("ninth-observatory", "journey"): ("timeline", "elegant", "warm", "contemplative"),
    # Chromatic Institute
    ("chromatic-institute", "infographic"): ("infographic", "vector-illustration", "neon", "analytical"),
    ("chromatic-institute", "comparison"): ("comparison", "vector-illustration", "neon", "analytical"),
    ("chromatic-institute", "inline_explanation"): ("infographic", "vector-illustration", "neon", "curious"),
    ("chromatic-institute", "hero"): ("scene", "vector-illustration", "neon", "emergent"),
    # Dark Cyberpunk HUD
    ("dark-cyberpunk-hud", "inline_explanation"): ("infographic", "dark-cyberpunk-hud", "neon", "technical"),
    ("dark-cyberpunk-hud", "infographic"): ("infographic", "dark-cyberpunk-hud", "neon", "technical"),
    ("dark-cyberpunk-hud", "social_card"): ("infographic", "dark-cyberpunk-hud", "neon", "technical"),
}

# Default fallback per studio
_STUDIO_DEFAULTS = {
    "mythic-tech-codex-illustration": ("scene", "editorial", "warm", "contemplative"),
    "cosmic-postcard-atelier": ("scene", "screen-print", "warm", "optimistic"),
    "ink-ember-studio": ("scene", "sketch", "mono-ink", "introspective"),
    "saga-noir-studio": ("scene", "screen-print", "mono-ink", "epic"),
    "ninth-observatory": ("scene", "blueprint", "mono-ink", "monumental"),
    "chromatic-institute": ("infographic", "vector-illustration", "neon", "analytical"),
    "dark-cyberpunk-hud": ("infographic", "dark-cyberpunk-hud", "neon", "technical"),
}

_ASPECT_MAP = {
    "twitter": "square",
    "linkedin": "landscape",
    "instagram": "portrait_4_5",
    "tiktok": "portrait_9_16",
    "blog": "landscape",
    "web": "landscape",
}


# ── Public API ──


def classify_intent(text: str) -> str:
    """Classify editorial intent from content text."""
    t = text.lower()
    for intent, cues in _INTENT_CUES:
        if any(c in t for c in cues):
            return intent
    return "hero"  # default


def classify_narrative(text: str) -> str:
    """Classify narrative mode from content text."""
    t = text.lower()
    for narrative, cues in _NARRATIVE_CUES:
        if any(c in t for c in cues):
            return narrative
    return "observation"  # default


def pick_studio(intent: str, narrative: str, brand: str,
                recent_history: Optional[list] = None) -> str:
    """Pick the best studio for the given intent + narrative + brand.

    Args:
        intent: Classified editorial intent.
        narrative: Classified narrative mode.
        brand: Brand key (e.g. 'sahil_twitter').
        recent_history: List of recent studio names to avoid repeating.

    Returns:
        Studio name string.
    """
    # Score each studio
    scores = []
    for studio in STUDIO_REGISTRY:
        score = 0
        if narrative in studio["best_narratives"]:
            score += 3
        if intent in studio["best_intents"]:
            score += 2
        if brand in studio["brand_affinity"]:
            score += 1
        # Penalise recently used studios
        if recent_history and studio["name"] in recent_history:
            penalty = recent_history.count(studio["name"])
            score -= penalty * 2
        scores.append((score, studio["name"]))

    # Sort by score descending, then by name for determinism
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][1]


def pick_layout(intent: str, narrative: str,
                recent_history: Optional[list] = None) -> str:
    """Pick the best layout for the given intent + narrative."""
    scores = []
    for name, meta in LAYOUT_REGISTRY.items():
        score = 0
        if intent in meta["best_intents"]:
            score += 2
        if narrative in meta["best_narratives"]:
            score += 1
        if recent_history and name in recent_history:
            penalty = recent_history.count(name)
            score -= penalty
        scores.append((score, name))
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][1]


def pick_composition(intent: str, narrative: str,
                     recent_history: Optional[list] = None) -> str:
    """Pick the best composition for the given intent + narrative."""
    scores = []
    for name, meta in COMPOSITION_REGISTRY.items():
        score = 0
        if intent in meta["best_intents"]:
            score += 2
        if narrative in meta["best_narratives"]:
            score += 1
        if recent_history and name in recent_history:
            penalty = recent_history.count(name)
            score -= penalty
        scores.append((score, name))
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][1]


def map_to_baoyu(studio: str, intent: str) -> tuple:
    """Map (studio, intent) -> (baoyu_type, baoyu_style, palette, mood)."""
    key = (studio, intent)
    if key in _BAOYU_MAP:
        return _BAOYU_MAP[key]
    # Fall back to studio default
    return _STUDIO_DEFAULTS.get(studio, ("scene", "editorial", "warm", "contemplative"))


def decide(content: dict, recent_history: Optional[list] = None) -> dict:
    """Full editorial decision for a piece of content.

    Args:
        content: Dict with keys: title, body_text, brand, platform, category, pillar.
        recent_history: Optional list of recent (studio, layout, composition) tuples
                        for rotation enforcement.

    Returns:
        Decision dict with all editorial + baoyu parameters.
    """
    title = content.get("title", "")
    body = content.get("body_text", "")
    brand = (content.get("brand") or "").lower()
    platform = (content.get("platform") or "").lower()
    pillar = (content.get("pillar") or "").lower()
    category = (content.get("category") or "").lower()

    text = f"{title} {body} {pillar} {category}"

    intent = classify_intent(text)
    narrative = classify_narrative(text)

    # Extract recent studio/layout/composition names from history
    recent_studios = [h[0] for h in (recent_history or [])]
    recent_layouts = [h[1] for h in (recent_history or [])]
    recent_compositions = [h[2] for h in (recent_history or [])]

    studio = pick_studio(intent, narrative, brand, recent_studios)
    layout = pick_layout(intent, narrative, recent_layouts)
    composition = pick_composition(intent, narrative, recent_compositions)

    baoyu_type, baoyu_style, palette, mood = map_to_baoyu(studio, intent)

    aspect = _ASPECT_MAP.get(platform, "square")

    return {
        "intent": intent,
        "narrative": narrative,
        "layout": layout,
        "composition": composition,
        "studio": studio,
        "baoyu_type": baoyu_type,
        "baoyu_style": baoyu_style,
        "palette": palette,
        "mood": mood,
        "aspect": aspect,
    }
