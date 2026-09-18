"""Gbrain semantic recall filter.

Score alone cannot separate signal from junk. Live hub measurement
(recall_fast, 2026-08-28 / PERS-117):

    telegram flood control ban   real      0.152
    zzqqxx-style junk            nonsense  scores in the same band

A floor of 0.10 keeps both. A floor above 0.152 drops real queries.
Keep the low floor for the clearly-irrelevant tail, and require at
least one content-word overlap with the prompt. Skip the overlap
check when the prompt has no content words, so an empty prompt
cannot wipe the tier.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence

GBRAIN_MIN_SCORE = float(os.getenv("HUB_RECALL_GBRAIN_MIN_SCORE", "0.10"))
GBRAIN_MIN_OVERLAP = int(os.getenv("HUB_RECALL_GBRAIN_MIN_OVERLAP", "1"))

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "as", "by", "from", "it", "its",
    "we", "you", "i", "he", "she", "they", "do", "does", "did", "what", "when",
    "where", "which", "who", "how", "why", "not", "no", "yes", "can", "will",
    "would", "should", "could", "about", "into", "over", "any", "all", "some",
    "de", "het", "een", "en", "van", "is", "op", "te", "dat", "die", "niet",
}


def keywords(text: str) -> set[str]:
    """Content words, lowercased. Short tokens and stopwords carry no signal.

    Keep 2-3 char tokens that were UPPERCASE in the source (BMG, APA, KPI).
    """
    raw = str(text or "")
    words = {
        w for w in re.findall(r"[a-z0-9]+", raw.lower())
        if len(w) > 3 and w not in _STOPWORDS
    }
    acronyms = {
        w.lower() for w in re.findall(r"\b[A-Z][A-Z0-9]{1,2}\b", raw)
        if w.lower() not in _STOPWORDS
    }
    return words | acronyms


def _score(hit: Mapping[str, Any]) -> float | None:
    raw = hit.get("score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def filter_gbrain_hits(
    prompt: str,
    hits: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    want = keywords(prompt)
    kept: list[dict[str, Any]] = []
    for hit in hits:
        score = _score(hit)
        if score is not None and score < GBRAIN_MIN_SCORE:
            continue
        if want:
            title = str(hit.get("title") or "")
            body = str(hit.get("text") or hit.get("content") or "")
            overlap = want & keywords(body + " " + title)
            if len(overlap) < GBRAIN_MIN_OVERLAP:
                continue
        kept.append(dict(hit))
    return kept
