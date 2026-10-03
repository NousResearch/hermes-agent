"""Cheap content-sanity checks for model output, completed or still streaming.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import math
import re
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

# What an interrupt checkpoint says INSTEAD of a repetition-dominated partial. Replaying the
# looped bytes (as the redirect's api_content or as the interrupted assistant row) re-seeds the
# loop on the next request and the corruption survives restarts (#112764); the model only needs
# to know the reply degenerated and was cut off.
REPETITION_LOOP_INTERRUPTED = "[the reply degenerated into a repetition loop and was interrupted]"

# Sampling bounds keep the general path linear in output size with a small,
# fixed multiplier. A dominant contiguous run necessarily crosses many of
# these evenly spaced anchors.
_MAX_ANCHOR_SAMPLES = 32
_MAX_ANCHOR_MATCHES = 8


# ``is_runaway_repetition``: a multi-line partial must be mostly copies of a few lines. Batch-style
# output (distinct INSERT rows, similar table rows) shares long prefixes and trips the window
# scan, but every line is distinct; a loop re-emits the same line(s).
_RUNAWAY_DISTINCT_LINE_RATIO = 0.5

# The finish_reason="stop" path discards a COMPLETED answer, so it only aborts at runaway scale:
# real stop-path loops (#100716) run 80k-350k chars, while asked-for repeats ("say X 50 times",
# identical table rows, templated YAML) stay in the low KB and must be delivered.
STOP_PATH_MIN_CHARS = 16_000

# A live stream is judged on at most this much of its latest text, so one check stays bounded
# however long the reply grows.
_STREAM_TAIL_CHARS = 4 * STOP_PATH_MIN_CHARS


def is_repetition_dominated(text: str) -> bool:
    """True when a contiguous run of at least five exact repetitions covers at least half
    of ``text`` — the signature of a model repetition loop (issue #86581). Coverage is measured
    with the true period, so long multi-line loop units count fully. Fail-open for short input.
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated enough to cover half the fragment (the common echo shape).
    if _line_repetition_dominated(text, n):
        return True

    # Window-count scan (#86581) catches loops whose repeats differ by a counter or noise token;
    # the periodic scan catches long exact units whose 60-char windows each recur too rarely.
    return _window_count_dominated(text, n) or _periodic_run_dominated(text, n)


def _window_count_dominated(text: str, n: int) -> bool:
    """True when one 60-char window recurs often enough to cover half of ``text``."""
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

    # Runs already expanded and rejected, as (left, right, period). A later anchor inside one of
    # them whose period is a multiple of that run's period would re-walk the same run.
    rejected: list[tuple[int, int, int]] = []
    for start in sample_starts:
        anchor = text[start : start + window]
        search_from = start + 1
        for _ in range(_MAX_ANCHOR_MATCHES):
            match = text.find(anchor, search_from)
            if match < 0:
                break
            period = match - start
            search_from = match + 1
            if any(lo <= start < hi and period % p == 0 for lo, hi, p in rejected):
                continue
            left, right = _expand_run(text, n, start, period, window)
            if right - left >= _MIN_REPEAT_COUNT * period and right - left >= n * _DOMINANCE_RATIO:
                return True
            rejected.append((left, right, period))
    return False


def _expand_run(text: str, n: int, start: int, period: int, matched: int) -> tuple[int, int]:
    """Expand one known equal window to the ``[left, right)`` bounds of its exact periodic run."""
    left = start
    while left > 0 and text[left - 1] == text[left - 1 + period]:
        left -= 1

    right = start + matched
    while right + period < n and text[right] == text[right + period]:
        right += 1
    return left, right + period


def is_runaway_repetition(text: str) -> bool:
    """Stricter than :func:`is_repetition_dominated`: also require the runaway shape.

    An interrupt checkpoint DROPS the partial when this fires, so a legitimately repetitive but
    correct reply (distinct batch rows) must not qualify: repeated windows have to dominate AND,
    when the text has line structure, at most half of its non-empty lines may be distinct.
    """
    if not is_repetition_dominated(text):
        return False
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < _MIN_REPEAT_COUNT:
        return True  # no line structure to judge by: a dominated single-line loop
    return len(set(lines)) <= len(lines) * _RUNAWAY_DISTINCT_LINE_RATIO


def _line_repetition_dominated(text: str, n: int) -> bool:
    """True when a single normalized line covers half the fragment via repeats."""
    counts = Counter(norm for norm in (line.strip() for line in text.splitlines()) if norm)
    return any(c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO for line, c in counts.items())


class RunawayStreamWatch:
    """The stop-path criterion applied WHILE one channel (visible text or reasoning) streams.

    A stream that keeps producing never reaches a completion check, and endpoints without an
    output cap (#127234) keep a looping model going for as long as the turn lives. The first
    check runs at ``STOP_PATH_MIN_CHARS`` (asked-for repeats stay below it, as on the stop path)
    and the gap doubles until it reaches one tail window, then stays there. A fixed stride keeps a
    loop that starts late in a long reply from streaming as long as the reply already was before
    a check sees it; each check reads only the tail, so the total work stays linear in the output.
    Each check also trims the held text to the tail, and checks are at most one tail window apart,
    so the watch never holds more than two tail windows plus the latest delta.
    """

    __slots__ = ("_parts", "_chars", "_next_check")

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._chars = 0
        self._next_check = STOP_PATH_MIN_CHARS

    def feed(self, text: str) -> bool:
        """Add one delta; True once the channel has become a runaway loop."""
        if not text:
            return False
        self._parts.append(text)
        self._chars += len(text)
        if self._chars < self._next_check:
            return False
        self._next_check = self._chars + min(self._chars, _STREAM_TAIL_CHARS)
        tail = "".join(self._parts)[-_STREAM_TAIL_CHARS:]
        self._parts = [tail]
        return is_runaway_repetition(tail)


# Semantic-cascade detection (#131098 defect 2). A model in free-association drift emits
# a long final reply whose tail walks an association chain (chemistry -> thermodynamics ->
# ... -> socks) with every n-gram unique, so ``is_runaway_repetition`` cannot see it.
# This catches that shape instead: the opening vocabulary is abandoned mid-reply and
# almost none of it resurfaces at the end.
#
# Thresholds (conservative: bias toward delivery; all three must agree to discard):
# - _CASCADE_HEAD_CHARS (1500): the committed-answer anchor. The incident prefix was only
#   368 chars, so the window also covers early drift; transitional scaffolding shared with
#   the tail is discounted via the middle (see below), not by shrinking the window.
# - _CASCADE_TAIL_CHARS (4000): the ending under test -- "did the reply end anywhere near
#   where it started?"
# - _CASCADE_RETURN_RATIO (0.10): of the head words used NOWHERE in the middle, at most
#   this fraction may resurface in the tail. The incident shape scores 0.00; a long
#   coherent reply scores 0.4+ or fails the floors below.
# - Floors (fail open): _CASCADE_MIN_ABANDONED_WORDS (30) and _CASCADE_MIN_NOVEL_WORDS (50).
#   Templated/batch tails reuse a tiny vocabulary (novel ~ 0) and coherent prose reuses its
#   head vocabulary throughout (abandoned ~ 0); both fail open here.
# - CASCADE_MIN_CHARS (8000): this gate's own length floor, deliberately lower than the
#   16k repetition floor. Total vocabulary abandonment is a stronger degeneration signal
#   than verbatim repetition, and the reported incident cascade was 12,657 chars -- a
#   shared 16k floor cannot catch the reported case.
#
# Known limitation: a legitimately chaptered 8k+ reply that abandons its opening topic
# fully and never returns reads cascade-shaped and trips this. Partial derailment after a
# long coherent body is out of scope (the middle still carries head vocabulary, so the
# reply fails open). No model calls, no embeddings, no new dependencies: three regex
# passes plus set differences, O(n) in reply length.
CASCADE_MIN_CHARS = 8_000
_CASCADE_HEAD_CHARS = 1500
_CASCADE_TAIL_CHARS = 4000
_CASCADE_RETURN_RATIO = 0.10
_CASCADE_MIN_ABANDONED_WORDS = 30
_CASCADE_MIN_NOVEL_WORDS = 50

_CASCADE_WORD_RE = re.compile(r"[a-z]+")
_CASCADE_STOPWORDS = frozenset(
    "a about above after again against all almost also always among amount another any "
    "anyone around because been before being both bottom cannot could doing done down "
    "during each either else enough even every few first from front full further had has "
    "have having here however into itself least less many more most never next none nothing "
    "often other over part same several should since some such than that then there these "
    "through under until upon were what when where which while with within without would "
    "your this they them those thus hence are was were for you his her its our their said "
    "will shall may might must can does did the and ours yours his hers theirs am is be "
    "been being".split()
)


def _cascade_content_words(text: str) -> set[str]:
    """Lowercase alpha tokens of length 3+ minus glue words; numbers, IDs and symbols ignored."""
    return {
        word
        for word in _CASCADE_WORD_RE.findall(text.lower())
        if len(word) >= 3 and word not in _CASCADE_STOPWORDS
    }


def is_semantic_cascade(text: str) -> bool:
    """True when a runaway-scale reply abandons its opening vocabulary and never returns.

    ``abandoned = head - middle`` (opening words used nowhere in the bulk) and
    ``novel = tail - middle`` (ending words introduced after the bulk); trips when at most
    ``_CASCADE_RETURN_RATIO`` of the abandoned words resurface among the novel ones.
    Middle-bulk discounting keeps recurring transitional scaffolding from counting as a
    topical return. Fail-open for short input, small vocabularies and non-strings.
    """
    if not isinstance(text, str):
        return False
    if len(text) < CASCADE_MIN_CHARS:
        return False
    head = _cascade_content_words(text[:_CASCADE_HEAD_CHARS])
    middle = _cascade_content_words(text[_CASCADE_HEAD_CHARS:-_CASCADE_TAIL_CHARS])
    tail = _cascade_content_words(text[-_CASCADE_TAIL_CHARS:])
    abandoned = head - middle
    novel = tail - middle
    if (
        len(abandoned) < _CASCADE_MIN_ABANDONED_WORDS
        or len(novel) < _CASCADE_MIN_NOVEL_WORDS
    ):
        return False
    return len(abandoned & novel) / len(abandoned) <= _CASCADE_RETURN_RATIO
