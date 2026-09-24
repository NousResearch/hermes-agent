"""Cheap content-sanity checks for the truncated-response continuation path.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import math
from collections import Counter

# Below this length the check doesn't run: short truncations trivially
# contain repeated tokens and are legitimately continued.
MIN_FRAGMENT_LENGTH = 400
# Exact-repeat window; far beyond ordinary phrasing reuse (citations, headings, similar code).
_REPEAT_WINDOW = 60
# A window repeating at least this often is a signal even for short fragments.
_MIN_REPEAT_COUNT = 5
# "Repetition-dominated" = repeated windows cover at least this fraction.
_DOMINANCE_RATIO = 0.5


def is_repetition_dominated(text: str) -> bool:
    """True when a single 60+ char substring recurs often enough to cover at least half
    of ``text`` — the signature of a repetition loop. Fail-open for non-string/short input.

    That shape is the signature of a model repetition loop (issue #86581), and continuing such a fragment is
    pointless — the continuation nudge would just stitch more repeated text into the final response.
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated enough to cover half the fragment (the common echo shape).
    if _line_repetition_dominated(text, n):
        return True

    # General path: fixed-size windows sliding one char at a time, catching loops that
    # don't align to line boundaries. A window must appear ``needed`` times to cover
    # >= _DOMINANCE_RATIO (and >= _MIN_REPEAT_COUNT).
    window = _REPEAT_WINDOW
    needed = max(_MIN_REPEAT_COUNT, math.ceil(n * _DOMINANCE_RATIO / window))
    counts: dict[str, int] = {}
    for i in range(n - window + 1):
        key = text[i : i + window]
        c = counts.get(key, 0) + 1
        if c >= needed:
            return True
        counts[key] = c
    return False


def _line_repetition_dominated(text: str, n: int) -> bool:
    """True when a single normalized line covers half the fragment via repeats."""
    counts = Counter(norm for norm in (line.strip() for line in text.splitlines()) if norm)
    return any(c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO for line, c in counts.items())
