"""Cheap content-sanity checks for the truncated-response continuation path.

Issue #86581: a model in a degenerate repetition loop can spend its ENTIRE
output budget echoing one fragment.  The ``finish_reason=length``
continuation path in ``conversation_loop.py`` would then retry with a
"continue, don't repeat" nudge — stitching a pathological fragment into the
final response with no content-sanity check.  In the incident behind #86581
a single turn produced a 60,698-char response delivered as 31 Discord
messages.

These helpers detect repetition-dominated fragments BEFORE the continuation
nudge is appended so the turn can abort with a clear user-facing error
(mirroring the existing ``_thinking_exhausted`` guard) instead of flooding.

The detection is deliberately conservative: only LONG verbatim repeats
(60+ chars) whose occurrences cover a majority of the fragment trip the
guard, so ordinary truncated responses (a sentence cut mid-word, a heading
repeated, code with similar-looking lines) are never blocked.
"""

from __future__ import annotations

import math
from typing import Any

from agent.message_content import flatten_message_text

# A fragment must be at least this long before the repetition check runs at
# all.  Short truncations (a sentence cut mid-word) can trivially contain
# repeated tokens and are legitimately continued.
MIN_FRAGMENT_LENGTH = 400

# Length of the exact-repeat window.  A verbatim repeat of this many chars
# is far beyond ordinary phrasing reuse (citations, headings, similar code).
_REPEAT_WINDOW = 60

# A window that repeats at least this many times is a repetition signal,
# even for short fragments.
_MIN_REPEAT_COUNT = 5

# A fragment is "repetition-dominated" when repeated windows account for at
# least this fraction of its characters.
_DOMINANCE_RATIO = 0.5


def is_repetition_dominated(text: Any) -> bool:
    """True when ``text`` is dominated by verbatim repeated fragments.

    A truncated response is "repetition-dominated" when a single 60+ char
    substring appears often enough that its occurrences cover at least half
    of the fragment.  That shape is the signature of a model repetition
    loop (issue #86581), and continuing such a fragment is pointless — the
    continuation nudge would just stitch more repeated text into the final
    response.

    Accepts plain string or multipart list content (which is flattened).
    Returns False for non-string / empty / short inputs (fail-open: never
    blocks a continuation the guard cannot confidently judge).
    """
    if isinstance(text, list):
        text = flatten_message_text(text)
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated often enough to cover half
    # the fragment (the most common echo shape — a repeated paragraph or
    # sentence on its own line).  Cheap, no big allocations.
    if _line_repetition_dominated(text, n):
        return True

    # General path: fixed-size exact-repeat windows, sliding one char at a
    # time.  Catches repetition loops that do not align to line boundaries.
    window = _REPEAT_WINDOW
    # A window must appear this many times for its occurrences to cover
    # >= DOMINANCE_RATIO of the fragment (and at least _MIN_REPEAT_COUNT).
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
    counts: dict[str, int] = {}
    for line in text.splitlines():
        norm = line.strip()
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
    for line, c in counts.items():
        if c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO:
            return True
    return False
