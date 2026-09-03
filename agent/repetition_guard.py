"""Cheap content-sanity checks for completed model output.

Issue #86581: a model in a degenerate repetition loop can spend its ENTIRE
output budget echoing one fragment.  The ``finish_reason=length``
continuation path in ``conversation_loop.py`` would then retry with a
"continue, don't repeat" nudge — stitching a pathological fragment into the
final response with no content-sanity check.  In the incident behind #86581
a single turn produced a 60,698-char response delivered as 31 Discord
messages.

These helpers detect repetition-dominated output before it is persisted or
delivered. Truncated responses are also checked before a continuation nudge is
appended.

The detection is deliberately conservative: only LONG verbatim repeats
(60+ chars) whose occurrences cover a majority of the fragment trip the
guard, so ordinary truncated responses (a sentence cut mid-word, a heading
repeated, code with similar-looking lines) are never blocked.
"""

from __future__ import annotations

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

# A fragment is "repetition-dominated" when one contiguous periodic run
# accounts for at least this fraction of its characters.
_DOMINANCE_RATIO = 0.5

# Sampling bounds keep the general path linear in output size with a small,
# fixed multiplier. A dominant contiguous run necessarily crosses many of
# these evenly spaced anchors.
_MAX_ANCHOR_SAMPLES = 32
_MAX_ANCHOR_MATCHES = 8


def is_repetition_dominated(text: str) -> bool:
    """True when ``text`` is dominated by verbatim repeated fragments.

    A response is "repetition-dominated" when a contiguous run containing at
    least five exact repetitions covers at least half of the output. That
    shape is the signature of a model repetition loop (issue #86581).

    Returns False for non-string / empty / short inputs (fail-open: never
    blocks a continuation the guard cannot confidently judge).
    """
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

    return _periodic_run_dominated(text, n)


def _periodic_run_dominated(text: str, n: int) -> bool:
    """Detect a dominant exact periodic run from evenly spaced anchors.

    Matching a 60-character anchor at a later position supplies a candidate
    period. Expanding the equality ``text[i] == text[i + period]`` in both
    directions recovers the full run, so coverage is measured using the true
    repeating unit rather than crediting every occurrence with only 60 chars.
    """
    window = _REPEAT_WINDOW
    max_start = n - window
    if max_start < 1:
        return False

    sample_step = max(
        1,
        (max_start + _MAX_ANCHOR_SAMPLES - 2) // (_MAX_ANCHOR_SAMPLES - 1),
    )
    sample_starts = list(range(0, max_start + 1, sample_step))
    if sample_starts[-1] != max_start:
        sample_starts.append(max_start)

    for start in sample_starts:
        anchor = text[start : start + window]
        search_from = start + 1
        for _ in range(_MAX_ANCHOR_MATCHES):
            match = text.find(anchor, search_from)
            if match < 0:
                break
            period = match - start
            if _candidate_run_dominated(text, n, start, period, window):
                return True
            search_from = match + 1
    return False


def _candidate_run_dominated(
    text: str,
    n: int,
    start: int,
    period: int,
    matched: int,
) -> bool:
    """Expand one known equal window and judge its exact run coverage."""
    left = start
    while left > 0 and text[left - 1] == text[left - 1 + period]:
        left -= 1

    right = start + matched
    while right + period < n and text[right] == text[right + period]:
        right += 1

    run_length = right + period - left
    return (
        run_length >= _MIN_REPEAT_COUNT * period
        and run_length >= n * _DOMINANCE_RATIO
    )


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
