"""Rule-based intent classification for the Memory Router.

Classification is DETERMINISTIC and KEYWORD-DRIVEN. It must never call an LLM
or read memory contents — it only inspects the query string (and an optional
explicit intent hint). This preserves cost, determinism, and auditability.

The keyword maps below are intentionally simple. The safe default for
low-confidence queries is :data:`DEFAULT_INTENT` (historical / broad index),
which guarantees recall rather than silent under-search.
"""

from __future__ import annotations

import re
from typing import Optional

from .intents import DEFAULT_INTENT, Intent

# Keyword -> intent. Evaluated in order; first match wins. Keep deterministic.
# Lower-cased query fragments.
_KEYWORD_MAP: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.IDENTITY, ("who am i", "whoami", "identity", "about joe", "about hermes", "what am i")),
    (Intent.DECISION, ("decision", "adr", "architectural decision", "why did we choose")),
    (Intent.PROJECT_STATE, ("project", "roadmap", "progress", "project state", "status of")),
    (Intent.RELATIONSHIP, ("related to", "connection", "graph", "relationship", "linked to")),
    (Intent.RECENT, ("recent", "last session", "what did we just", "latest")),
    # Historical is the catch-all and is handled by the default branch.
]

# Explicit intent-hint aliases (so callers can force an intent).
_HINT_MAP: dict[str, Intent] = {
    "identity": Intent.IDENTITY,
    "who": Intent.IDENTITY,
    "project": Intent.PROJECT_STATE,
    "project_state": Intent.PROJECT_STATE,
    "decision": Intent.DECISION,
    "adr": Intent.DECISION,
    "historical": Intent.HISTORICAL,
    "history": Intent.HISTORICAL,
    "search": Intent.HISTORICAL,
    "relationship": Intent.RELATIONSHIP,
    "recent": Intent.RECENT,
    "context": Intent.CONTEXT,
}

# Tokens that, in a free-text query, signal an explicit broad/historical ask.
_HISTORICAL_HINTS = ("search", "history", "what did we", "find", "remember when")

# Stopwords used only by some callers for tokenizing search queries; not required
# for classification. Kept here so the router owns its own token vocabulary.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "be", "by", "with", "at", "from", "as",
        "it", "this", "that", "these", "those", "i", "you", "we", "they",
    }
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def classify(query: str, intent_hint: Optional[str] = None) -> Intent:
    """Classify ``query`` into an :class:`Intent`.

    Args:
        query: the user's natural-language memory request.
        intent_hint: optional explicit override ("identity", "decision",
            "project", "historical", "relationship", "recent", "context").
            When valid, it is honored immediately (deterministic override).

    Returns:
        An :class:`Intent`. Low-confidence queries fall back to
        :data:`DEFAULT_INTENT` (historical / broad index).
    """
    if intent_hint is not None:
        hint = intent_hint.strip().lower()
        if hint in _HINT_MAP:
            return _HINT_MAP[hint]

    q = _normalize(query)
    if not q:
        return DEFAULT_INTENT

    for intent, keywords in _KEYWORD_MAP:
        for kw in keywords:
            if kw in q:
                return intent

    # Explicit broad/historical phrasing wins the default branch.
    if any(h in q for h in _HISTORICAL_HINTS):
        return Intent.HISTORICAL

    return DEFAULT_INTENT


def tokenize(query: str, drop_stopwords: bool = True) -> list[str]:
    """Split ``query`` into cleaned lowercase tokens (utility for search)."""
    toks = re.findall(r"[a-z0-9_]+", _normalize(query))
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks
