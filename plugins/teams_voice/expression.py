"""Avatar emotion heuristic — Python port of openclaw ``expression.ts``.

A cheap, dependency-free lexical classifier that infers a coarse emotion from the
assistant's reply text, with no extra model call. The result is sent to the media
worker as an additive ``expression`` cue so the avatar shapes its mouth
(smile / frown / round "O"). Best-effort and purely cosmetic.

Priority order matters: surprise > sad > happy > neutral (a "!" plus a sad word
should read as surprise, matching the original).
"""

from __future__ import annotations

import re

# Coarse emotion labels understood by the worker's AvatarExpression mapping.
NEUTRAL = "neutral"
HAPPY = "happy"
SAD = "sad"
SURPRISED = "surprised"

_SURPRISED_PUNCT = re.compile(r"(!!|\?!|!\?)")
_SURPRISED_WORDS = frozenset(
    {"wow", "whoa", "woah", "omg", "unbelievable", "incredible", "what?!"}
)
_SAD_WORDS = frozenset(
    {
        "sorry",
        "unfortunately",
        "sadly",
        "failed",
        "failure",
        "error",
        "regret",
        "apolog",  # matches apolog-y / apolog-ize via substring check below
        "couldn't",
        "cannot",
        "unable",
    }
)
_HAPPY_WORDS = frozenset(
    {
        "great",
        "perfect",
        "awesome",
        "excellent",
        "wonderful",
        "fantastic",
        "glad",
        "happy",
        "congrats",
        "congratulations",
        "nice",
        "love",
    }
)

_WORD_RE = re.compile(r"[a-z']+")


def infer_emotion(text: str) -> str:
    """Return one of ``neutral|happy|sad|surprised`` for ``text``.

    Empty/whitespace input is ``neutral``. Matching is case-insensitive and
    word-oriented (with a couple of substring stems where the original allowed
    them, e.g. "apolog").
    """
    if not text or not text.strip():
        return NEUTRAL

    lowered = text.lower()
    if _SURPRISED_PUNCT.search(lowered):
        return SURPRISED

    words = set(_WORD_RE.findall(lowered))
    if words & _SURPRISED_WORDS:
        return SURPRISED
    if _matches(words, lowered, _SAD_WORDS):
        return SAD
    if _matches(words, lowered, _HAPPY_WORDS):
        return HAPPY
    return NEUTRAL


def _matches(words: set[str], lowered: str, vocab: frozenset[str]) -> bool:
    """Whole-word match, with substring fallback for stem entries (len <= 6)."""
    if words & vocab:
        return True
    # A few vocab entries are stems (e.g. "apolog"); allow substring for those.
    return any(term not in words and len(term) <= 6 and term in lowered for term in vocab)
