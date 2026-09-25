"""Cheap content-sanity checks for completed model output.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

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


# ---- thinking-channel loop guard ----------------------------------------------------------
# The checks above watch the VISIBLE reply on truncation/interrupt paths. A thinking channel
# can degenerate on its own while the visible reply stays fine: one char (usually a quoting
# bracket) grows run over run — 「「「「「「実行」」」」」」「「「「「「「「「「やる」」... — with no exact
# long-window repeat, so ``is_repetition_dominated`` misses it (17 of 21 messages in one real
# incident corpus). The looped bytes must be cut BEFORE storage: a DeepSeek-style
# ``reasoning_content`` echo replays them into the next request and re-seeds the loop
# (#112764 family). Thresholds calibrated against a real-world corpus of ~175k
# reasoning messages: 「」『』 runs >= 12 and runs of >= 160 identical non-formatting chars
# never fired on any message without a degenerate segment.
#
# A second shape escaped those run rules and is covered below too: QUOTE LITTER — openers
# inserted between tokens and mostly never closed (「の「diff「全体「像), net unmatched
# openers climbing steadily with no long run at all. It goes chronic in long-lived sessions
# once seeded (1143 rows across 4 sessions in the corpus scan; the litter rule fires on ~93%
# of them and on none of the healthy rows — analysis sessions that QUOTE degenerate excerpts
# peak at a 1.4% unmatched-opener rate and a 0.27 dense-bin fraction, under the thresholds).
THINKING_LOOP_TRUNCATED = "[thinking truncated: repetition loop detected]"

_BRACKET_RUN_CHARS = frozenset("「」『』")
_BRACKET_RUN_MIN = 12
_RUN_MIN = 160

# Quote-litter thresholds: the length floor gives the rate/bin statistics runway; a "dense
# bin" is a 200-char window holding >= _LITTER_BIN_QUOTES openers; the rule needs the net
# unmatched-opener rate to hold across a majority of bins.
_LITTER_MIN_CHARS = 600
_LITTER_MIN_NET = 20
_LITTER_RATE_PERMILLE = 15
_LITTER_BIN_CHARS = 200
_LITTER_BIN_QUOTES = 5
_LITTER_START_NET = 10
_RUN_EXCLUDED = frozenset("-=*_|+#~` \t\r\n")


class ReasoningLoopGuard:
    """Incremental degeneration detector for a streamed reasoning channel.

    Feed each reasoning delta in order (stop once ``tripped`` is True). ``trip_index`` is
    the offset — in the concatenation of everything fed — where the degenerate region starts;
    callers cut accumulators there so display, storage and reasoning echo all stop replaying
    the loop. O(chars), no rescans.
    """

    __slots__ = (
        "tripped", "trip_index", "_seen", "_run_char", "_run_len", "_run_start",
        "_opens", "_closes", "_litter_start", "_bins", "_dense_bins", "_bin_quotes",
    )

    def __init__(self) -> None:
        self.tripped = False
        self.trip_index = -1
        self._seen = 0
        self._run_char = ""
        self._run_len = 0
        self._run_start = 0
        self._opens = 0
        self._closes = 0
        self._litter_start = -1
        self._bins = 0
        self._dense_bins = 0
        self._bin_quotes = 0

    def feed(self, text: str) -> bool:
        if self.tripped or not isinstance(text, str) or not text:
            return self.tripped
        run_char, run_len, run_start = self._run_char, self._run_len, self._run_start
        i = self._seen
        for ch in text:
            if ch == run_char:
                run_len += 1
            else:
                run_char, run_len, run_start = ch, 1, i
            i += 1
            if ch in _BRACKET_RUN_CHARS and run_len >= _BRACKET_RUN_MIN:
                self._trip(run_start, ch, run_len)
                return True
            if run_len >= _RUN_MIN and ch not in _RUN_EXCLUDED and not ch.isspace():
                self._trip(run_start, ch, run_len)
                return True
            if ch == "「":
                self._opens += 1
                self._bin_quotes += 1
                if self._litter_start < 0 and self._opens - self._closes >= _LITTER_START_NET:
                    self._litter_start = i
            elif ch == "」":
                self._closes += 1
            if i % _LITTER_BIN_CHARS == 0:
                self._bins += 1
                if self._bin_quotes >= _LITTER_BIN_QUOTES:
                    self._dense_bins += 1
                self._bin_quotes = 0
        self._seen = i
        self._run_char, self._run_len, self._run_start = run_char, run_len, run_start
        if self._litter_trips(i):
            return True
        return self.tripped

    def _litter_trips(self, length: int) -> bool:
        """Shape 2: net-unclosed quote litter dense across a majority of 200-char bins."""
        net = self._opens - self._closes
        if length < _LITTER_MIN_CHARS or net < _LITTER_MIN_NET:
            return False
        if net * 1000 < length * _LITTER_RATE_PERMILLE:
            return False
        if self._bins == 0 or self._dense_bins * 2 < self._bins:
            return False
        self._trip(max(0, self._litter_start), "litter", net)
        return True

    def _trip(self, at: int, ch: str, length: int) -> None:
        self.tripped = True
        self.trip_index = at
        logger.debug("reasoning loop guard tripped: %r x%d at offset %d", ch, length, at)


def sanitize_degenerate_reasoning(text, *, marker: str = THINKING_LOOP_TRUNCATED):
    """Full-text pass for non-streaming intakes / storage boundaries.

    Returns ``text`` unchanged (same object) unless a degenerate shape (loop or litter) is
    found; then the
    degenerate tail is dropped and ``marker`` appended. Fail-open for non-strings.
    """
    if not isinstance(text, str) or not text:
        return text
    guard = ReasoningLoopGuard()
    if not guard.feed(text):
        return text
    prefix = text[: max(0, guard.trip_index)].rstrip()
    return f"{prefix}\n\n{marker}" if prefix else marker
