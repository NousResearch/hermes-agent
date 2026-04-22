"""Stop word list for term index extraction.

Uses the well-known NLTK English stop word list (179 words) as a baseline,
plus common JSON schema keys from tool output and pure-numeric filter.

This module is self-contained -- no external dependencies.
"""

import re

# Standard English stop words (NLTK list, public domain)
# Covers articles, conjunctions, prepositions, pronouns, auxiliary verbs,
# and common function words. Intentionally excludes short tech terms
# that overlap (e.g., "go", "it" as in IT/InfoTech handled by context).
_ENGLISH_STOP_WORDS = frozenset(
    w.lower() for w in [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
        "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
        "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
        "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
        "wasn", "weren", "won", "wouldn",
    ]
)

# JSON schema keys that appear constantly in tool output.
# These are field names from structured tool responses, not semantic content.
# Nobody searches for "exit_code" to find a past session.
_JSON_KEY_STOP_WORDS = frozenset([
    "output",
    "exit_code",
    "error",
    "null",
    "true",
    "false",
    "status",
    "content",
    "message",
    "cleared",
    "success",
])

# Combined stop word set
_STOP_WORDS = _ENGLISH_STOP_WORDS | _JSON_KEY_STOP_WORDS

# Pattern to detect pure numeric tokens (integers, floats, hex)
_NUMERIC_RE = re.compile(r"^[0-9]+$")


def is_stop_word(word: str) -> bool:
    """Check if a word is a stop word. Case-insensitive."""
    return word.lower() in _STOP_WORDS


def is_noise_term(word: str) -> bool:
    """Check if a term is noise that should be excluded from the index.

    This covers stop words AND pure numeric tokens, which provide zero
    search value. Nobody searches for '0', '1', or '42' to find a session.
    """
    lower = word.lower()
    return lower in _STOP_WORDS or _NUMERIC_RE.match(lower) is not None


def get_stop_words() -> frozenset:
    """Return the full stop word set (for inspection/bulk use)."""
    return _STOP_WORDS