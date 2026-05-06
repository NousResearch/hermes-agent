"""Deterministic research-intent classification.

This module is intentionally small and rule-based. It is shared by the
research orchestrator and CodeAct recipes so routing decisions are auditable
and not hidden inside a generated tool wrapper.
"""

from __future__ import annotations

import re
from typing import Any


_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    # Keep source-sensitive domains ahead of broad freshness/current-events
    # markers. A query such as "latest GLP-1 phase 3 drugs" should use the
    # medical/pharma source recipe, not the generic current-events profile.
    "medical_pharma": (
        "glp-1",
        "glp1",
        "gip",
        "incretin",
        "agonist",
        "drug",
        "pharma",
        "biotech",
        "clinical trial",
        "clinical trials",
        "phase 1",
        "phase 2",
        "phase 3",
        "fda",
        "ema",
        "nmpa",
        "pubmed",
        "diabetes",
        "obesity",
        "nash",
        "mash",
    ),
    "finance": (
        "finance",
        "financial",
        "stock",
        "stocks",
        "equity",
        "bond",
        "bonds",
        "yield",
        "earnings",
        "revenue",
        "guidance",
        "price target",
        "market cap",
        "analyst rating",
        "sec filing",
        "10-k",
        "10-q",
        "fed",
        "federal reserve",
        "inflation",
        "cpi",
        "gdp",
        "treasury",
        "crypto",
        "bitcoin",
        "ethereum",
    ),
    "geopolitics": (
        "geopolitics",
        "geopolitical",
        "foreign policy",
        "diplomacy",
        "sanctions",
        "nato",
        "united nations",
        "u.n.",
        "war",
        "conflict",
        "ceasefire",
        "treaty",
        "embassy",
        "ukraine",
        "russia",
        "china",
        "taiwan",
        "gaza",
        "israel",
    ),
    "sports": (
        "sports",
        "nba",
        "wnba",
        "nfl",
        "mlb",
        "nhl",
        "epl",
        "premier league",
        "champions league",
        "fifa",
        "ncaa",
        "lineup",
        "starting lineup",
        "starting five",
        "starting 5",
        "roster",
        "injury report",
        "box score",
        "depth chart",
        "standings",
    ),
    "engineering": (
        "engineering",
        "mechanical",
        "electrical",
        "civil engineering",
        "aerospace",
        "manufacturing",
        "materials",
        "standards",
        "iso",
        "ieee",
        "asme",
        "astm",
        "cad",
        "prototype",
        "failure analysis",
    ),
    "technical": (
        "api",
        "docs",
        "documentation",
        "github",
        "package",
        "library",
        "bug",
        "changelog",
    ),
    "technology": (
        "technology",
        "tech",
        "ai",
        "artificial intelligence",
        "semiconductor",
        "chip",
        "chips",
        "gpu",
        "smartphone",
        "laptop",
        "cloud",
        "cybersecurity",
        "startup",
        "product launch",
    ),
    "gaming": (
        "gaming",
        "video game",
        "videogame",
        "game patch",
        "patch notes",
        "dlc",
        "steam",
        "playstation",
        "xbox",
        "nintendo",
        "esports",
        "metacritic",
    ),
    "social_trends": (
        "social media",
        "trend",
        "trends",
        "trending",
        "viral",
        "tiktok",
        "instagram",
        "youtube",
        "twitter",
        "x.com",
        "reddit",
        "hashtag",
        "meme",
        "social sentiment",
    ),
    "music": (
        "music",
        "song",
        "album",
        "single",
        "artist",
        "band",
        "spotify",
        "apple music",
        "billboard",
        "chart",
        "charts",
        "tour",
        "concert",
        "label",
    ),
    "shopping": (
        "shopping",
        "shop",
        "buy",
        "deal",
        "deals",
        "discount",
        "retailer",
        "amazon",
        "walmart",
        "best buy",
        "availability",
    ),
    "domestic_news": (
        "domestic news",
        "national news",
        "u.s. news",
        "us news",
        "white house",
        "congress",
        "senate",
        "house of representatives",
        "governor",
        "mayor",
        "election",
        "state legislature",
    ),
    "current_events": (
        "today",
        "latest",
        "current",
        "currently",
        "breaking",
        "news",
        "as of",
        "as-of",
    ),
    "academic": (
        "paper",
        "arxiv",
        "study",
        "journal",
        "citation",
        "citations",
    ),
    "legal_regulatory": (
        "law",
        "regulation",
        "statute",
        "court",
        "legal",
    ),
    "company_market": (
        "price",
        "pricing",
        "company",
        "market",
        "filing",
    ),
    "product": (
        "best",
        "review",
        "buy",
        "recommend",
        "compare",
    ),
    "local": (
        "near me",
        "local",
        "restaurant",
        "weather",
    ),
    "obscure_lookup": (
        "obscure",
        "hard to find",
        "exact phrase",
    ),
}

_RESEARCH_ACTION_MARKERS: tuple[str, ...] = (
    "search",
    "research",
    "report",
    "find out",
    "find information",
    "information about",
    "investigate",
    "source",
    "sources",
    "citation",
    "citations",
    "evidence",
    "overview",
    "status",
    "track",
    "monitor",
)

_WEAK_ACTION_MARKERS: frozenset[str] = frozenset(
    {
        "search",
        "status",
    }
)

_LATEST_MARKERS: tuple[str, ...] = (
    "today",
    "latest",
    "current",
    "currently",
    "as of",
    "as-of",
    "up to date",
    "up-to-date",
    "real-time",
    "breaking",
    "trending",
    "viral",
)

_RECENT_MARKERS: tuple[str, ...] = (
    "recent",
    "new",
    "newest",
    "recently",
)

_DEVELOPMENT_FRESHNESS_MARKERS: tuple[str, ...] = (
    "development",
    "in development",
    "testing",
    "pipeline",
    "clinical trial",
    "clinical trials",
    "phase 1",
    "phase 2",
    "phase 3",
)

_SOURCE_INTENSIVE_TOPICS: frozenset[str] = frozenset(
    {
        "academic",
        "company_market",
        "current_events",
        "domestic_news",
        "engineering",
        "finance",
        "gaming",
        "geopolitics",
        "legal_regulatory",
        "medical_pharma",
        "music",
        "product",
        "shopping",
        "social_trends",
        "sports",
        "technology",
    }
)

_TARGETED_LOOKUP_MARKERS: tuple[str, ...] = (
    "site:",
    "filetype:",
    "inurl:",
    "intitle:",
)


def _normalize(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _contains(haystack: str, marker: str) -> bool:
    if not marker:
        return False
    if any(ch for ch in marker if not (ch.isalnum() or ch == "_")):
        return marker in haystack
    pattern = rf"(?<![a-z0-9_]){re.escape(marker)}(?![a-z0-9_])"
    return re.search(pattern, haystack) is not None


def _matched(haystack: str, markers: tuple[str, ...]) -> list[str]:
    return [marker for marker in markers if _contains(haystack, marker)]


def _classify_topic(q: str) -> tuple[str, list[str]]:
    for topic_type, markers in _TOPIC_MARKERS.items():
        matches = _matched(q, markers)
        if matches:
            return topic_type, matches
    return "general", []


def classify_research_intent(question: str) -> dict[str, Any]:
    """Classify whether text describes a source-grounded research task.

    Returns a plain dict so generated CodeAct namespace code can consume it
    without importing dataclass definitions.
    """
    q = _normalize(question)
    topic_type, topic_matches = _classify_topic(q)
    action_matches = _matched(q, _RESEARCH_ACTION_MARKERS)
    strong_action_matches = [
        marker for marker in action_matches if marker not in _WEAK_ACTION_MARKERS
    ]
    latest_matches = _matched(q, _LATEST_MARKERS)
    recent_matches = _matched(q, _RECENT_MARKERS)
    development_matches = _matched(q, _DEVELOPMENT_FRESHNESS_MARKERS)
    targeted_lookup_matches = _matched(q, _TARGETED_LOOKUP_MARKERS)

    freshness = "auto"
    if latest_matches or (topic_type == "medical_pharma" and development_matches):
        freshness = "latest"
    elif recent_matches:
        freshness = "recent"

    source_intensive = topic_type in _SOURCE_INTENSIVE_TOPICS
    is_research = bool(
        strong_action_matches
        or latest_matches
        or recent_matches
        or (source_intensive and topic_matches)
    )
    targeted_lookup = bool(targeted_lookup_matches)
    redirect_web_search = bool(
        is_research
        and not targeted_lookup
        and (
            strong_action_matches
            or latest_matches
            or recent_matches
            or (source_intensive and topic_matches)
        )
    )

    confidence = 0.0
    if strong_action_matches:
        confidence += 0.35
    if latest_matches or recent_matches:
        confidence += 0.30
    if source_intensive and topic_matches:
        confidence += 0.25
    if len(topic_matches) > 1:
        confidence += 0.10
    if targeted_lookup:
        confidence -= 0.20
    confidence = max(0.0, min(1.0, confidence))

    return {
        "is_research": is_research,
        "redirect_web_search": redirect_web_search,
        "topic_type": topic_type,
        "freshness": freshness,
        "confidence": confidence,
        "matched": {
            "topic": topic_matches,
            "action": action_matches,
            "latest": latest_matches,
            "recent": recent_matches,
            "development": development_matches,
            "targeted_lookup": targeted_lookup_matches,
        },
    }


def classify_topic_type(question: str) -> str:
    """Return only the topic profile for callers that do not need full intent."""
    return str(classify_research_intent(question).get("topic_type") or "general")
