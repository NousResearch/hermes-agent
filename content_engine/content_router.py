"""Decide whether a draft becomes an AI infographic (Track B, Seedream) or an
illustrated/photographic image (Track A, krea + overlay).

Infographic when the post is structured/data-led (comparison, list, framework,
stats, how-to with steps); otherwise an illustrated scene. An explicit
draft['format'] always wins.

Source-aware routing (editorial_router) preferred over static mapping when a
real signal is available.
"""
from __future__ import annotations
from typing import Optional

_IG_PILLARS = {"data", "data_driven", "comparison", "framework", "list", "tips", "stats", "howto"}
_IG_CUES = (" vs ", " vs.", "do and don", "do/don", "don't", "step 1", "steps to",
            "reasons", "ways to", "framework", "comparison", "checklist", "top ", "x vs y")


def suggest_layout(draft: dict):
    """Pick a layout that matches the content's natural shape, else None (let the
    brand's weighted rotation choose)."""
    t = ((draft.get("title") or "") + " " + (draft.get("body_text") or "")).lower()
    if " vs " in t or " vs." in t or "versus" in t or "comparison" in t:
        return "binary-comparison"
    if "do/don" in t or "do and don" in t or "don't" in t or "do's" in t:
        return "do-dont"
    if any(k in t for k in ("steps", "ways to", "reasons", "tips", "checklist", "top ")):
        return "feature-list"
    if "funnel" in t or "stages" in t or "pipeline" in t:
        return "funnel"
    if "framework" in t or "matrix" in t or "quadrant" in t:
        return "priority-quadrants"
    return None


_TYPE_CUES = [
    ("comparison", (" vs ", " vs.", "versus", " vs ")),
    ("flowchart", ("step 1", "steps to", "how i ", "how to", "process", "pipeline")),
    ("framework", ("framework", "matrix", "quadrant", "the 6-minute", "system", "stack")),
    ("timeline", ("timeline", "history of", "evolution", "roadmap", "week ")),
    ("infographic", ("reasons", "ways to", "tips", "checklist", "top ", "metrics", "stats")),
]
_STORY_TYPES = {"hero", "scene", "metaphor", "typography"}
_VALID_TYPES = {"infographic", "comparison", "framework", "flowchart", "timeline"} | _STORY_TYPES


def content_type_for(draft: dict) -> str:
    """Map a draft to a baoyu Type. Explicit draft['format'] wins. Structured
    cues -> data types; otherwise -> 'scene' (story/atmospheric)."""
    fmt = (draft.get("format") or "").lower()
    if fmt in _VALID_TYPES:
        return fmt
    text = ((draft.get("title") or "") + " " + (draft.get("body_text") or "")
            + " " + (draft.get("pillar") or "")).lower()
    pillar = (draft.get("pillar") or "").lower()
    if pillar in _IG_PILLARS or is_infographic(draft):
        for t, cues in _TYPE_CUES:
            if any(c in text for c in cues):
                return t
        return "infographic"
    return "scene"


def is_infographic(draft: dict) -> bool:
    fmt = (draft.get("format") or "").lower()
    if fmt == "infographic":
        return True
    if fmt in ("illustration", "photo", "scene", "hero"):
        return False
    pillar = (draft.get("pillar") or "").lower()
    ctype = (draft.get("content_type") or "").lower()
    if pillar in _IG_PILLARS or ctype in _IG_PILLARS:
        return True
    text = ((draft.get("title") or "") + " " + (draft.get("body_text") or "")).lower()
    return any(c in text for c in _IG_CUES)


def source_aware_content_type(draft: dict, signal: Optional[dict] = None) -> str:
    """Determine content type with source awareness.

    When a real signal is available (has signal_type), the editorial router
    determines the content type based on signal properties (screenshot →
    text+image, build note → text+image, etc). Falls back to the static
    content_type_for when no signal is available (app brand static topics).

    Args:
        draft: Draft dict with brand, pillar, format, title, body_text.
        signal: Optional signal dict from activity_collector / topic source.

    Returns:
        Content type string: text | text+image | article | blog | scene | ...
    """
    has_signal_context = bool(
        signal and (
            signal.get("signal_type")
            or (signal.get("topic_id") and signal.get("title_hint") and (
                signal.get("curated") is True
                or signal.get("evidence")
                or signal.get("source_notes")
                or signal.get("context")
                or signal.get("source_override")
            ))
        )
    )
    if signal and signal.get("signal_type"):
        try:
            from editorial_router import choose_content_type
            ctype = choose_content_type(
                signal,
                brand=draft.get("brand", ""),
                pillar=draft.get("pillar", ""),
                platform=draft.get("platform", "twitter"),
            )
            if ctype:
                return ctype
        except Exception:
            pass
    return content_type_for(draft)


# ── Claim/provenance gate for app-brand drafts ──

_INVENTED_CLAIM_PATTERNS = [
    # Test group sizes
    r"\b\d{2,}\s*(household|test|user|family|families)\b",
    # Beta tester names (specific people)
    r"\b(beta tester|maya|lucy)\b",
    # Fabricated quotes with specific numbers
    r"(?:saved|save)\s*(?::)?\s*[£]\d+\.\d{2}",
    r"[£]\d+\.\d{2}\s*(?:saved|save|cheaper|less)",
    # Specific receipt totals
    r"[Rr]eceipt\s*says\s*[£]\d+",
    # Time savings
    r"\b\d{2,}\s*(minute|hour|second)\s*(?:saved|per|less|down|from)",
]


def gate_claims(draft: dict) -> tuple[bool, list[str]]:
    """Check a draft body for unsupported/invented claims.

    Returns (passes_gate, issues) where passes_gate is True if no
    invented claims found, or all claims have source provenance.

    Applies to all brands, but especially app brands (Plenishd,
    CoachOS, MatchdayMaestro) where invented test groups, fake
    quotes, and unsupported savings are highest risk.
    """
    body = (draft.get("body_text") or draft.get("full_post", "") or "")
    body_lower = body.lower()
    topics_data = draft.get("topic", "")
    issues = []

    import re
    for pat in _INVENTED_CLAIM_PATTERNS:
        match = re.search(pat, body_lower)
        if match:
            # Check if source provenance exists
            prov = draft.get("source_provenance") or {}
            if not prov or prov.get("confidence") in ("manual", ""):
                issues.append(f"Unsupported claim: '{match.group()[:40]}'"
                              f" (no source provenance)")
            # Verified/claimed provenance: flag but don't block
            # (allows drafts with real source data through review)

    return (len(issues) == 0, issues)
