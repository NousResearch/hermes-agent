"""Term index — inverted index extraction for session search fast path.

Extracts non-stop-word terms from message content for insertion into the
term_index table in SessionDB. Terms are lowercased, punctuation-stripped
(with preservation of path-like strings), and deduplicated per message.
"""

import re
from stop_words import is_stop_word

# Matches "words" including paths (foo/bar), filenames (file.py), and
# hyphenated terms (self-hosted). Filters out most punctuation but
# preserves dots in filenames and slashes in paths.
# Strategy: split on whitespace first, then strip leading/trailing punctuation.
_TERM_RE = re.compile(r"[a-zA-Z0-9][\w./\-]*[a-zA-Z0-9]|[a-zA-Z0-9]")


def extract_terms(content: str) -> list[str]:
    """Extract non-stop-word terms from message content.

    Returns a deduplicated, lowercased list of terms.
    Stops words, pure punctuation, and empty strings are excluded.
    """
    if not content:
        return []

    # Find candidate tokens
    raw_tokens = _TERM_RE.findall(content)

    seen = set()
    terms = []
    for token in raw_tokens:
        lower = token.lower()
        # Skip stop words
        if is_stop_word(lower):
            continue
        # Skip single characters except meaningful ones
        # (but these are already handled by stop words for 'a', 'I', etc.)
        # Deduplicate within this message
        if lower not in seen:
            seen.add(lower)
            terms.append(lower)

    return terms