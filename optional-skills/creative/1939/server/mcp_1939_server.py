#!/usr/bin/env python3
"""1939 Perceptual Color Engine — MCP Server.

Exposes three tools for voice-queryable palette access:
  - palette_lookup: Get a specific palette by name or slug
  - palette_search: Search palettes by mood, color, use case, or character
  - palette_recommend: Get recommendations for a specific use case

Also exposes one resource:
  - palettes://flagship/list — JSON list of all 29 flagship themes

And one prompt:
  - apply_palette_to_document: Generate application instructions for a palette

Data sources (all relative to this script's directory):
  - Brand JSON files from flagship/*.brand.json
  - Voice descriptions from flagship_descriptions.json
  - Memes index from memes/index.json

Setup:
  pip install mcp fastmcp
  python3 mcp_1939_server.py

Configure in your agent's config:
  mcp_servers:
    1939:
      command: python3
      args: ["path/to/1939/palettes/mcp_1939_server.py"]
      timeout: 30
"""

import json
import os
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Optional phonetic/Levenshtein — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    from metaphone import doublemetaphone
    _HAS_METAPHONE = True
except ImportError:
    _HAS_METAPHONE = False

try:
    from Levenshtein import distance as _lev_distance
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False

# ---------------------------------------------------------------------------
# Configuration — paths are relative to this script's directory
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: data lives in ../palettes/ relative to this server script
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "palettes")
DATA_DIR = os.environ.get("NINETEEN_DATA_DIR", DEFAULT_DATA_DIR)
FLAGSHIP_DIR = os.path.join(DATA_DIR, "flagship")
MEMES_INDEX_PATH = os.path.join(DATA_DIR, "memes", "index.json")
DESCRIPTIONS_PATH = os.path.join(SCRIPT_DIR, "flagship_descriptions.json")

mcp = FastMCP("1939")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

_flagship_cache: Dict[str, dict] = {}
_descriptions_cache: List[dict] = []
_memes_cache: List[dict] = []

# ---------------------------------------------------------------------------
# STT Aliases — known speech-to-text mishearings mapped to palette slugs
# ---------------------------------------------------------------------------
# Keys are lowercased spoken forms; values are canonical slugs.
_STT_ALIASES: Dict[str, str] = {
    # $1.50 — STT sees dollar signs as "dollar", numbers spoken out
    "one fifty": "one-fifty-1985",
    "dollar fifty": "one-fifty-1985",
    "a dollar fifty": "one-fifty-1985",
    "one dollar fifty": "one-fifty-1985",
    "dollar one fifty": "one-fifty-1985",
    "buck fifty": "one-fifty-1985",
    # No. 5 — STT expands "No." as "number" or drops it
    "number five": "chanel-no-5",
    "channel number five": "chanel-no-5",
    "channel five": "chanel-no-5",
    "chanel number five": "chanel-no-5",
    "chanel five": "chanel-no-5",
    "chanel": "chanel-no-5",
    "no five": "chanel-no-5",
    # Hugo's Mom — apostrophe variants
    "hugos mom": "hugos-mom-1974",
    "hugo's mom": "hugos-mom-1974",
    "hugo mom": "hugos-mom-1974",
    # 1939 — spoken form of the Wizard of Oz theme
    "nineteen thirty nine": "wizard-of-oz-1939",
    "nineteen thirty nine": "wizard-of-oz-1939",
    "wizard of oz": "wizard-of-oz-1939",
    "wizard of oz nineteen thirty nine": "wizard-of-oz-1939",
    # WKW — initials get spelled out
    "w k w": "in-the-mood-for-love-2000",
    "wong kar wai": "in-the-mood-for-love-2000",
    "in the mood for love": "in-the-mood-for-love-2000",
    "mood for love": "in-the-mood-for-love-2000",
    # DTF XLB — initialisms get spelled out
    "d t f x l b": "dtf-xlb-1958",
    "dtf": "dtf-xlb-1958",
    "xlb": "dtf-xlb-1958",
    # L'Air de Panache — French article gets garbled
    "l air de panache": "lair-de-panache-2014",
    "laire de panache": "lair-de-panache-2014",
    "lair de panache": "lair-de-panache-2014",
    "air de panache": "lair-de-panache-2014",
    "panache": "lair-de-panache-2014",
    # Display name aliases (many palettes have short names + long descriptive slugs)
    "smaug": "conversation-with-smaug-1937",
    "conversation with smaug": "conversation-with-smaug-1937",
    "lucy": "sgt-peppers-lonely-hearts-club-band-1967",
    "sgt pepper": "sgt-peppers-lonely-hearts-club-band-1967",
    "sgt peppers": "sgt-peppers-lonely-hearts-club-band-1967",
    "sergeant pepper": "sgt-peppers-lonely-hearts-club-band-1967",
    "sergeant peppers": "sgt-peppers-lonely-hearts-club-band-1967",
    "peppers lonely hearts": "sgt-peppers-lonely-hearts-club-band-1967",
    "melting": "persistence-of-memory-1931",
    "persistence of memory": "persistence-of-memory-1931",
    "dali": "persistence-of-memory-1931",
    "el aurens": "lawrence-of-arabia-1962",
    "lawrence of arabia": "lawrence-of-arabia-1962",
    "mumtaz": "taj-mahal-interior-1632",
    "taj mahal": "taj-mahal-interior-1632",
    "summer jpg": "summer-2021",
    "baiwei": "buidl-value",
    "build value": "buidl-value",
    "buidl value": "buidl-value",
    "star gate": "star-gate-1968",
    "stargate": "star-gate-1968",
    "starg ate": "star-gate-1968",
    "close up": "close-up-1950",
    "pussy wagon": "pussy-wagon-2003",
    "slice": "slice",
    "blue": "blue",
    "gutenberg": "gutenberg",
    "intrepid": "intrepid-2023",
    "sobol": "sobol",
    "soup": "soup-1962",
    "thriller": "thriller-1983",
    "moon river": "moon-river-2019",
    "laugh tango": "laugh-tango-1985",
    "slings and arrows": "slings-and-arrows",
}

# Use-case aliases for palette_recommend
_USE_CASE_ALIASES: Dict[str, str] = {
    "presentation": "presentation",
    "deck": "presentation",
    "slide": "presentation",
    "slides": "presentation",
    "ppt": "presentation",
    "pptx": "presentation",
    "dashboard": "dashboard",
    "dash": "dashboard",
    "report": "document",
    "doc": "document",
    "document": "document",
    "word": "document",
    "website": "dashboard",
    "web": "dashboard",
    "data viz": "data visualization",
    "data_viz": "data visualization",
    "visualization": "data visualization",
    "chart": "data visualization",
    "pitch deck": "presentation",
    "pitch": "presentation",
    "portfolio": "photography",
    "mobile app": "mobile",
    "app": "mobile",
}

_MOOD_ALIASES: Dict[str, str] = {
    "warm": "warm",
    "cool": "cool",
    "cold": "cool",
    "bold": "bold",
    "minimal": "minimal",
    "clean": "minimal",
    "cinematic": "cinematic",
    "movie": "cinematic",
    "film": "cinematic",
    "intimate": "intimate",
    "dramatic": "dramatic",
    "neutral": "neutral",
    "muted": "neutral",
    "dark": "dark",
    "light": "light",
    "bright": "light",
    "vibrant": "bold",
    "soft": "intimate",
    "professional": "neutral",
}


def _load_flagship():
    """Load all 29 flagship brand JSONs into cache."""
    global _flagship_cache, _descriptions_cache
    if _flagship_cache:
        return

    for fname in os.listdir(FLAGSHIP_DIR):
        if not fname.endswith(".brand.json"):
            continue
        slug = fname.replace(".brand.json", "")
        with open(os.path.join(FLAGSHIP_DIR, fname)) as f:
            _flagship_cache[slug] = json.load(f)

    if os.path.exists(DESCRIPTIONS_PATH):
        with open(DESCRIPTIONS_PATH) as f:
            _descriptions_cache = json.load(f)

    # Build phonetic index after loading all data
    _build_phonetic_index()


def _load_memes():
    """Load memes index into cache."""
    global _memes_cache
    if _memes_cache:
        return
    if os.path.exists(MEMES_INDEX_PATH):
        with open(MEMES_INDEX_PATH) as f:
            _memes_cache = json.load(f)


def _get_description(slug: str) -> Optional[str]:
    """Get voice-ready description for a flagship theme."""
    for d in _descriptions_cache:
        if d["slug"] == slug:
            return d.get("description", "")
    return None


# ---------------------------------------------------------------------------
# Phonetic index (Layer 4)
# ---------------------------------------------------------------------------
_phonetic_index: Dict[str, str] = {}  # metaphone_code → slug


def _build_phonetic_index():
    """Build double-metaphone index from all palette names + aliases."""
    global _phonetic_index
    if not _HAS_METAPHONE:
        return

    # Index from all names
    for slug, data in _flagship_cache.items():
        name = data.get("name", "")
        # Index the display name
        for token in name.split():
            primary, alternate = doublemetaphone(token)
            if primary:
                _phonetic_index[primary] = slug
            if alternate and alternate != primary:
                _phonetic_index[alternate] = slug
        # Index the full name as one string
        primary, alternate = doublemetaphone(name)
        if primary:
            _phonetic_index[primary] = slug
        if alternate and alternate != primary:
            _phonetic_index[alternate] = slug
        # Index slug tokens (these contain descriptive words)
        for token in slug.split("-"):
            if len(token) <= 2:  # skip year fragments, "of", "the", etc.
                continue
            primary, alternate = doublemetaphone(token)
            if primary:
                _phonetic_index[primary] = slug
            if alternate and alternate != primary:
                _phonetic_index[alternate] = slug

    # Index from aliases
    for alias, slug in _STT_ALIASES.items():
        for token in alias.split():
            primary, alternate = doublemetaphone(token)
            if primary:
                _phonetic_index[primary] = slug
            if alternate and alternate != primary:
                _phonetic_index[alternate] = slug
        primary, alternate = doublemetaphone(alias)
        if primary:
            _phonetic_index[primary] = slug
        if alternate and alternate != primary:
            _phonetic_index[alternate] = slug


# ---------------------------------------------------------------------------
# 6-Layer Fuzzy Matching (mirrors sf-atlas/sf-curriculum pattern)
# ---------------------------------------------------------------------------
# Layer 1: Exact / case-insensitive / slug match        (score 100)
# Layer 2: STT alias (exact or longest substring)       (score 95/90)
# Layer 3: Token subset match                             (score 60-80)
# Layer 4: Metaphone phonetic match                      (score 60/token)
# Layer 5: Levenshtein distance (normalised threshold)   (score up to 50)
# Layer 6: difflib get_close_matches fallback             (score 30)


def _match_palette(name: str) -> Optional[str]:
    """Find a flagship palette slug by name using 6-layer fuzzy matching.

    Returns the slug if found, None if no match.
    """
    key = name.strip()
    key_lower = key.lower()

    # --- Layer 1: Exact match (case-insensitive + slug conversion) ---
    # Try slug conversion
    slug_guess = key_lower.replace(" ", "-").replace("'", "").replace(".", "").replace(",", "").replace("$", "")
    if slug_guess in _flagship_cache:
        return slug_guess

    # Try exact name match
    for slug, data in _flagship_cache.items():
        if data.get("name", "").lower() == key_lower:
            return slug

    # Try slug exact
    if key_lower in _flagship_cache:
        return key_lower

    # --- Layer 2: STT alias (exact match or longest substring) ---
    # Exact alias match
    if key_lower in _STT_ALIASES:
        return _STT_ALIASES[key_lower]

    # Substring match — find longest alias contained in the input
    best_alias = None
    best_len = 0
    for alias, slug in _STT_ALIASES.items():
        if alias in key_lower and len(alias) > best_len:
            best_alias = slug
            best_len = len(alias)
    if best_alias:
        return best_alias

    # --- Layer 3: Token subset match ---
    # "Miramar" matches "Miramar Equity Partners" if all query tokens are in name tokens
    key_tokens = set(re.findall(r'\w+', key_lower))
    for slug, data in _flagship_cache.items():
        name = data.get("name", "").lower()
        slug_text = slug.replace("-", " ")
        # Check tokens in both display name AND slug words
        all_tokens = set(re.findall(r'\w+', f"{name} {slug_text}"))
        if key_tokens and key_tokens.issubset(all_tokens):
            return slug

    # --- Layer 4: Metaphone phonetic match ---
    if _HAS_METAPHONE and _phonetic_index:
        # Try full input first
        primary, alternate = doublemetaphone(key)
        for mp in [primary, alternate]:
            if mp and mp in _phonetic_index:
                return _phonetic_index[mp]
        # Try each token
        for token in key_tokens:
            primary, alternate = doublemetaphone(token)
            for mp in [primary, alternate]:
                if mp and mp in _phonetic_index:
                    return _phonetic_index[mp]

    # --- Layer 5: Levenshtein distance ---
    if _HAS_LEVENSHTEIN:
        best_slug = None
        best_dist = float('inf')
        # Threshold scales with input length
        threshold = max(1, len(key) // 3)
        threshold = min(threshold, 3)

        for slug, data in _flagship_cache.items():
            name = data.get("name", "")
            # Check against display name and slug
            for candidate in [name, slug]:
                dist = _lev_distance(key_lower, candidate.lower())
                if dist < best_dist and dist <= threshold:
                    best_dist = dist
                    best_slug = slug
        if best_slug:
            return best_slug

    # --- Layer 6: difflib get_close_matches fallback ---
    all_names = [data.get("name", "") for data in _flagship_cache.values()]
    all_slugs = list(_flagship_cache.keys())

    matches = get_close_matches(key, all_names + all_slugs, n=1, cutoff=0.5)
    if matches:
        best = matches[0]
        for slug, data in _flagship_cache.items():
            if data.get("name") == best or slug == best:
                return slug

    # Also try description keyword search as final resort
    hits = _keyword_search(name, _descriptions_cache)
    if hits:
        slug = hits[0]["slug"]
        if slug in _flagship_cache:
            return slug

    return None


def _keyword_search(query: str, descriptions: List[dict]) -> List[dict]:
    """Search descriptions by keywords. Returns matching entries sorted by relevance."""
    q = query.lower()
    # Tokenize the query
    q_tokens = set(re.findall(r'\w+', q))

    scored = []
    for d in descriptions:
        text = d.get("description", "").lower()
        name = d.get("name", "").lower()
        slug = d.get("slug", "").lower()

        # Check for direct name/slug match first
        if q in name or q in slug:
            scored.append((d, 100))
            continue

        # Keyword matching
        d_tokens = set(re.findall(r'\w+', f"{text} {name} {slug}"))
        overlap = len(q_tokens & d_tokens)
        if overlap > 0:
            # Boost if keywords match high-value fields
            name_overlap = len(q_tokens & set(re.findall(r'\w+', name)))
            score = overlap + (name_overlap * 3)
            scored.append((d, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored if s[1] > 0]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def palette_lookup(name: str) -> str:
    """Look up a specific 1939 palette by name or slug.

    Returns the full brand-ready JSON with all 8 roles, 10 tints per role,
    contrast ratios, semantic mappings, and PPT/Word document mappings.
    Also includes a natural-language description for voice queryability.

    Uses 6-layer fuzzy matching: exact → STT alias → token subset →
    metaphone phonetic → Levenshtein → difflib/description fallback.

    Args:
        name: Palette name (e.g., "Hugo's Mom", "wizard-of-oz-1939") or slug.
              Supports fuzzy matching — "hugo" will find "Hugo's Mom",
              "one fifty" will find "$1.50", "w k w" will find WKW.
    """
    _load_flagship()

    slug = _match_palette(name)
    if slug and slug in _flagship_cache:
        result = _flagship_cache[slug].copy()
        result["description"] = _get_description(slug)
        return json.dumps(result, indent=2, ensure_ascii=False)

    available = sorted([d.get("name", s) for s, d in _flagship_cache.items()])
    return json.dumps({
        "error": f"Palette '{name}' not found.",
        "available": available
    }, indent=2)


@mcp.tool()
def palette_search(query: str, limit: int = 5) -> str:
    """Search 1939 palettes by mood, color, use case, or character.

    Searches across both flagship themes (29) and memes (496) using
    natural language descriptions and keyword matching. Returns matching
    palettes with their voice-ready descriptions.

    Examples:
        "warm dark cinematic palette"
        "bold presentation theme"
        "palette with coral highlight"
        "minimal monochrome"
        "good for data visualization"

    Args:
        query: Natural language search query describing mood, color, use case, etc.
        limit: Maximum results to return (default 5, max 20).
    """
    _load_flagship()
    _load_memes()

    limit = min(limit, 20)
    results = []

    # Search flagship descriptions
    flagship_hits = _keyword_search(query, _descriptions_cache)
    for hit in flagship_hits[:limit]:
        slug = hit["slug"]
        if slug in _flagship_cache:
            entry = {
                "slug": slug,
                "name": hit.get("name", ""),
                "year": hit.get("year"),
                "collection": "1939-flagship",
                "description": hit.get("description", ""),
                "pv": hit.get("pv"),
                "is_dark": hit.get("is_dark"),
                "highlight": hit.get("highlight_hex"),
                "support": hit.get("support_hex"),
                "background": hit.get("background_hex"),
                "canvas": hit.get("canvas_hex"),
                "url": f"/api/themes/{slug}",
                "brand_file": f"palettes/flagship/{slug}.brand.json"
            }
            results.append(entry)

    # Also search memes index
    q = query.lower()
    q_tokens = set(re.findall(r'\w+', q))
    for card in _memes_cache:
        # Keyword search on name, artist, and character
        searchable = f"{card.get('name', '')} {card.get('artist', '')} season {card.get('season', '')}".lower()
        s_tokens = set(re.findall(r'\w+', searchable))
        overlap = len(q_tokens & s_tokens)
        if overlap > 0:
            pv = card.get("pv", 0)
            cc = card.get("center_colors", {})
            results.append({
                "slug": card.get("slug", ""),
                "name": card.get("name", ""),
                "year": card.get("year"),
                "collection": "6529-memes",
                "season": card.get("season"),
                "artist": card.get("artist", ""),
                "pv": pv,
                "is_dark": True,  # most 6529 cards are dark palettes
                "highlight": cc.get("Highlight", ""),
                "support": cc.get("Support", ""),
                "background": cc.get("Background", ""),
                "canvas": cc.get("Canvas", ""),
                "url": card.get("api_url", f"/api/cards/{card.get('slug', '')}")
            })

    # Sort flagship hits first, then by relevance
    results.sort(key=lambda x: (
        0 if x.get("collection") == "1939-flagship" else 1,
        -len(set(re.findall(r'\w+', q)) & set(re.findall(r'\w+', x.get("description", x.get("name", "")))))
    ))

    return json.dumps(results[:limit], indent=2, ensure_ascii=False)


@mcp.tool()
def palette_recommend(use_case: str, mood: str = "", limit: int = 3) -> str:
    """Get palette recommendations for a specific use case.

    Returns the best palettes matching your use case (presentation,
    dashboard, document, website, data visualization) and optional mood
    (warm, cool, bold, minimal, cinematic, etc.).

    Each recommendation includes the full brand-ready JSON and a
    natural-language description explaining why it's a good match.

    Args:
        use_case: What you're creating. One of: presentation, dashboard,
                  document, website, data_viz, pitch_deck, portfolio,
                  report, mobile_app. Or describe it in your own words.
        mood: Optional mood preference: warm, cool, bold, minimal,
              cinematic, intimate, dramatic, neutral. Leave empty for
              best match regardless of mood.
        limit: Maximum recommendations to return (default 3, max 10).
    """
    _load_flagship()

    limit = min(limit, 10)

    # Resolve use_case and mood through alias maps
    mapped_use = _USE_CASE_ALIASES.get(use_case.lower(), use_case.lower())
    search_terms = [mapped_use]
    if mood:
        mapped_mood = _MOOD_ALIASES.get(mood.lower(), mood.lower())
        search_terms.append(mapped_mood)

    query = " ".join(search_terms)
    hits = _keyword_search(query, _descriptions_cache)

    # If mood specified, filter to matching mood
    if mood:
        mood_l = mood.lower()
        hits = [h for h in hits if mood_l in h.get("description", "").lower()] or hits

    # Build recommendations
    recommendations = []
    for hit in hits[:limit]:
        slug = hit["slug"]
        if slug not in _flagship_cache:
            continue
        brand = _flagship_cache[slug]
        desc = hit.get("description", "")
        recommendations.append({
            "slug": slug,
            "name": brand.get("name", ""),
            "year": brand.get("year"),
            "recommendation_reason": f"Matches '{use_case}' use case" + (f" with '{mood}' mood" if mood else ""),
            "description": desc,
            "highlight": brand["roles"]["Highlight"]["hex"],
            "support": brand["roles"]["Support"]["hex"],
            "background": brand["roles"]["Background"]["hex"],
            "canvas": brand["roles"]["Canvas"]["hex"],
            "full_brand": brand
        })

    return json.dumps(recommendations, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

@mcp.resource("palettes://flagship/list")
def flagship_list() -> str:
    """List all 29 flagship themes with slugs, names, years, and descriptions."""
    _load_flagship()
    entries = []
    for slug, data in sorted(_flagship_cache.items()):
        desc = _get_description(slug) or ""
        entries.append({
            "slug": slug,
            "name": data.get("name", ""),
            "year": data.get("year"),
            "collection": "1939-flagship",
            "description": desc[:200] + "..." if len(desc) > 200 else desc,
            "highlight": data["roles"]["Highlight"]["hex"],
            "pv": data.get("perceptual_volume")
        })
    return json.dumps(entries, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def apply_palette_to_document(palette_name: str, document_type: str) -> str:
    """Generate instructions for applying a 1939 palette to a document.

    Args:
        palette_name: Name or slug of the palette (e.g., "Hugo's Mom")
        document_type: Type of document (powerpoint, word, website, google_slides)
    """
    _load_flagship()

    # Find the palette using 6-layer fuzzy matching
    slug = _match_palette(palette_name)
    if not slug or slug not in _flagship_cache:
        available = sorted([d.get("name", s) for s, d in _flagship_cache.items()])
        return f"Palette '{palette_name}' not found. Available: {', '.join(available)}"

    brand = _flagship_cache[slug]
    desc = _get_description(slug) or "No description available."

    doc_specific = ""
    if document_type.lower() in ("powerpoint", "ppt", "pptx", "google_slides", "slides"):
        mapping = brand.get("pptx_mapping", {})
        doc_specific = f"""## PowerPoint/Slides Mapping
- Title text: {brand['roles']['Highlight']['hex']} (Highlight)
- Body text: {brand['roles']['Text']['hex']} (Text)
- Slide background: {brand['roles']['Background']['hex']} (Background)
- Accent bar: {brand['roles']['Support']['hex']} (Support)
- Chart series 1: {brand['roles']['Chart1']['hex']} (Chart1)
- Chart series 2: {brand['roles']['Chart2']['hex']} (Chart2)

Use Background tint 700 for slide header bars: {brand['roles']['Background']['tints'][6]}
Use Canvas tint 100 for subtle card backgrounds: {brand['roles']['Canvas']['tints'][1]}"""
    elif document_type.lower() in ("word", "docx", "document", "report"):
        mapping = brand.get("docx_mapping", {})
        doc_specific = f"""## Word/Document Mapping
- Heading 1: {brand['roles']['Highlight']['hex']} (Highlight)
- Heading 2: {brand['roles']['Support']['hex']} (Support)
- Body text: {brand['roles']['Text']['hex']} (Text)
- Page background: {brand['roles']['Background']['hex']} (Background)
- Table header bg: {brand['roles']['Background']['tints'][6]} (Background tint 700)
- Table header text: {brand['roles']['Highlight']['tints'][0]} (Highlight tint 100)
- Links: {brand['roles']['Support']['hex']} (Support)"""
    elif document_type.lower() in ("website", "web", "html", "css"):
        doc_specific = f"""## CSS Custom Properties
Use the full loadTheme() pattern from references/applying-themes/web-css-custom-properties.md.

Key variables:
- --bg: {brand['roles']['Background']['hex']} (Background, or Canvas for light mode)
- --text: {brand['roles']['Text']['hex']} (Text)
- --surface: color-mix(in oklch, Background 88%, Text) (dark mode)
- --accent: {brand['roles']['Highlight']['hex']} (Highlight)
- --accent-dim: {brand['roles']['Support']['hex']} (Support)
- --chart1: {brand['roles']['Chart1']['hex']} (Chart1)
- --chart2: {brand['roles']['Chart2']['hex']} (Chart2)
- --border: color-mix(in oklch, Muted 80%, Background) (Muted blend)"""

    return f"""# Applying "{brand['name']}" to a {document_type}

{desc}

## All 8 Roles
| Role | Hex | Semantic Use |
|------|-----|-------------|
| Background | {brand['roles']['Background']['hex']} | Page bg, dark containers |
| Canvas | {brand['roles']['Canvas']['hex']} | Light surfaces, card bg |
| Text | {brand['roles']['Text']['hex']} | Body text, metadata |
| Highlight | {brand['roles']['Highlight']['hex']} | Headings, CTAs, emphasis |
| Support | {brand['roles']['Support']['hex']} | Secondary accent, links |
| Chart1 | {brand['roles']['Chart1']['hex']} | Primary data series |
| Chart2 | {brand['roles']['Chart2']['hex']} | Secondary data series |
| Muted | {brand['roles']['Muted']['hex']} | Borders, disabled states |

## Contrast Ratios
- text_on_background: {brand['contrast'].get('text_on_background', 'N/A')}
- highlight_on_background: {brand['contrast'].get('highlight_on_background', 'N/A')}
- canvas_on_background: {brand['contrast'].get('canvas_on_background', 'N/A')}

{doc_specific}

## Tint Scale (Highlight)
Each role has 10 perceptual tints. Highlight tints:
{', '.join(brand['roles']['Highlight']['tints'])}

Index 0=lightest (50-level), 4=center (500-level), 9=darkest (950-level).
"""


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_flagship()
    _load_memes()
    mcp.run()