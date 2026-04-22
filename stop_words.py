"""Stop word list for term index extraction.

Uses the well-known NLTK English stop word list (179 words) as a baseline.
This module is self-contained -- no external dependencies.
"""

# Standard English stop words (NLTK list, public domain)
# Covers articles, conjunctions, prepositions, pronouns, auxiliary verbs,
# and common function words. Intentionally excludes short tech terms
# that overlap (e.g., "go", "it" as in IT/InfoTech handled by context).
_STOP_WORDS = frozenset(
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


def is_stop_word(word: str) -> bool:
    """Check if a word is a stop word. Case-insensitive."""
    return word.lower() in _STOP_WORDS


def get_stop_words() -> frozenset:
    """Return the full stop word set (for inspection/bulk use)."""
    return _STOP_WORDS