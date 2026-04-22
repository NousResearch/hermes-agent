"""Term index — inverted index extraction for session search fast path.

Extracts non-stop-word terms from message content for insertion into the
term_index table in SessionDB. Terms are lowercased, punctuation-stripped
(with preservation of path-like strings), and deduplicated per message.

Noise filtering:
  - English stop words (NLTK list)
  - JSON schema keys from tool output (output, exit_code, error, etc.)
  - Pure numeric tokens (0, 1, 42, etc.)
"""

import re
from stop_words import is_noise_term

# Matches "words" including paths (foo/bar), filenames (file.py), and
# hyphenated terms (self-hosted). Filters out most punctuation but
# preserves dots in filenames and slashes in paths.
# Strategy: split on whitespace first, then strip leading/trailing punctuation.
_TERM_RE = re.compile(r"[a-zA-Z0-9][\w./\-]*[a-zA-Z0-9]|[a-zA-Z0-9]")


def extract_terms(content: str) -> list[str]:
    """Extract non-noise terms from message content.

    Returns a deduplicated, lowercased list of terms.
    Stop words, JSON keys, pure numerics, and empty strings are excluded.
    """
    if not content:
        return []

    # Find candidate tokens
    raw_tokens = _TERM_RE.findall(content)

    seen = set()
    terms = []
    for token in raw_tokens:
        lower = token.lower()
        # Skip noise: stop words, JSON keys, pure numerics
        if is_noise_term(lower):
            continue
        # Deduplicate within this message
        if lower not in seen:
            seen.add(lower)
            terms.append(lower)

    return terms