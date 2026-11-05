"""Detect degenerate repetition in model output.

A model that has lost the plot often emits the same short phrase over and
over -- ``"NOW FIN. NOW FIN. ..."``, ``"Executing the patrol."`` hundreds of
times -- instead of a report.  Such output must never reach a chat channel
verbatim: it can be hundreds of kilobytes, and chunking adapters
(``splits_long_messages = True``) will happily fan it into a chat one
message at a time.

This module has three separate responsibilities, kept deliberately apart so
the cron lane, the delivery lane, and the webhook lane can compose them
independently:

* :func:`is_degenerate` -- statistical detection of runaway repetition.
* :func:`collapse_degenerate` -- keep the readable text, drop the repetition.
* :func:`truncate_for_delivery` -- a hard size cap for the delivery path.

Detection is **statistical only**.  There is deliberately NO word blacklist:
a phrase like "FIN" or "GO" is perfectly legitimate prose, and an anchored
blacklist would both false-positive on real reports and fail to generalize
to the next incident's vocabulary.  Two independent features are measured,
because the two observed incident shapes look completely different:

* ``periodic_run`` -- the longest stretch where the token stream is exactly
  periodic with a short period.  A model that latches onto one phrase
  produces an enormous run (``"NOW FIN. NOW FIN. ..."`` has period 2 and a
  run of thousands); ordinary prose in any language repeats at most a handful
  of times in a row.
* ``window_dup_ratio`` -- the duplicate-gram ratio inside a sliding window.
  A model that does not repeat one identical phrase but cycles a small
  vocabulary (``"invoke ... result ... executing the patrol"``) still shaves
  the window's distinct-gram fraction to near zero.

Both features are language-agnostic: CJK ideographs are tokenized one
character at a time (CJK has no whitespace word boundaries) while Latin /
digit runs tokenize as words.

Calibration
-----------
Thresholds were fitted against the corpus of real cron outputs taken from
every profile on this machine (``~/.hermes/cron/output`` plus each
``~/.hermes/profiles/*/cron/output``).  The last measured sweep scored
**629 files**, of which 12 are the known-degenerate incidents (6 incidents,
each stored twice: a ``<job_id>_<timestamp>.txt`` copy and the markdown
copy under ``<job_id>/``).  The remaining 617 negatives include long patrol
reports, board tables, JSON/CSV dumps, generated code and
repetitive-but-informative logs.  The periodic feature is calibrated on *how
many whole copies of one short block* were emitted, not on the raw match
count; the two signals separate cleanly on that axis:

    population (scored files)      max periodic copies   max window_dup_ratio
    ------------------------------------------------------------------------
    known-degenerate (12)                   9870                  0.998
    real negatives (617)                    < 4                  0.461

Four of the 12 degenerate files score below the copy floor (they repeat a
small vocabulary rather than one block, so their periodic axis stays low
while their window ratio runs 0.92-0.93): the window feature is what
catches them, and the two features are independent by design.

``copies`` is ``(run + period) / period`` -- the number of back-to-back
repetitions of the same ``period``-token block.  The raw match count is the
wrong axis because it scales with block length: a report that quotes the same
five-token status line 40 times is 195 matching positions at period 5 (the
``_NORMAL_REPORT`` fixture below), yet carries real information, while a
two-token phrase repeated 1500 times is 3998 positions at period 2 and is
pure noise.  Counting complete copies collapses both to a comparable number.

The synthetic corpus in ``tests/gateway/test_degenerate_output_guard.py``
extends that with the shapes that broke earlier drafts of this detector: a
board-table patrol report and a status line repeated exactly 40 times (whose
naive match count -- 195, at period 5 -- tripped the old 25-match floor).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# --- Detection thresholds ---------------------------------------------------
#
# Calibrated against the real cron-output corpus described in the module
# docstring.  Measured values on that corpus:
#
#   population (scored files)      max periodic copies   max window_dup_ratio
#   ------------------------------------------------------------------------
#   known-degenerate (12)                   9870                  0.998
#   real negatives (617)                    < 4                  0.461
#
# The thresholds sit in the empty middle of that gap, so they are robust to
# content drift in either direction.

# Shortest periodic run (in matching token positions) that counts as
# runaway repetition.  This is the ABSOLUTE floor only: it keeps a single
# repeated token from tripping on a short list.  The real gate is
# MIN_PERIODIC_COPIES below -- see the note there for why the match count
# alone cannot be the threshold.
MIN_PERIODIC_RUN = 25

# Number of COMPLETE back-to-back copies of one short block that makes the
# block degenerate.  The raw match count is the wrong threshold because it
# scales with the block length: a legitimate report quoting the same
# five-token status line 40 times is 195 matching positions at period 5
# (the ``_NORMAL_REPORT`` fixture), while a two-token phrase repeated 1500
# times is 3998 positions at period 2.  Counting whole copies puts both on
# one axis: 40 copies for the report, 1500 for the noise.
#
# 48 sits in the empty gap measured on the real corpus: the largest honest
# repeat observed is 40 copies (the fixture below), the largest real corpus
# negative is 3 copies, and the smallest incident on this axis is 711 copies.
# A whole block copied 48 times back-to-back carries no new information per
# copy -- the model is stuck -- while a report that repeats a status line 40
# times is still a report.
#
# The floor is deliberately far above both populations: the corpus gap alone
# would justify a threshold anywhere from 4 to 711, and a wide margin keeps
# the axis from firing on a report shape the corpus never happened to
# contain.  Incidents that cycle a small vocabulary instead of repeating one
# block score below this floor and are caught by the window feature.
MIN_PERIODIC_COPIES = 48

# The periodic run must also span at least this many tokens in total
# (run + period).  Prevents a long-enough run of a single repeated token
# ("OK OK OK ...") from tripping on a short list.
MIN_PERIODIC_SPAN = 50

# Largest period examined.  Period 1 catches a single word repeated
# ("FIN FIN FIN"); period 2-4 catch short phrases ("NOW FIN", "GO NOW FIN").
# Beyond ~6 the match rate in real prose rises fast and the feature starts
# firing on legitimate text.
MAX_PERIOD = 6

# Duplicate-gram ratio (fraction of windowed k-grams that are repeats of an
# earlier gram in the same window) above which a window is flagged.
WINDOW_DUP_RATIO = 0.85

# Number of flagged windows required before the small-vocabulary-cycle
# feature fires.  Each window covers WINDOW_GRAMS 4-grams, so 25 flagged
# windows means at least ~32 KB of the payload sits inside a highly
# repetitive region -- a signal no legitimate report produced.
MIN_FLAGGED_WINDOWS = 25

# k for the windowed duplicate-ratio feature.
WINDOW_GRAM_SIZE = 4

# Sliding-window size, in k-grams, over which the duplicate ratio is measured.
# 512 grams is roughly a paragraph; small enough that a degenerate tail is
# detected even when it is only a modest fraction of the whole response, large
# enough that normal prose noise never reaches the threshold.
WINDOW_GRAMS = 512

# A sliding window is stepped rather than swept: a degenerate run is many
# thousands of grams long, so a coarse stride still lands inside it, and the
# stride keeps detection O(n) on a 200 KB payload.
WINDOW_STRIDE = 64

# Responses shorter than this cannot be meaningfully degenerate (and are
# cheap to deliver).  Below this many characters, detection is skipped.
MIN_CHARS_FOR_DETECTION = 120

# Minimum token count before detection runs at all.
MIN_TOKENS_FOR_DETECTION = 40

# --- Collapse constants -----------------------------------------------------

# Marker appended in place of the dropped degenerate span.
COLLAPSE_MARKER = (
    "\n\n[输出因模型退化重复已截断：中间的重复段已丢弃]"
)

# Total characters of readable text kept when collapsing.  Deliberately
# below the gateway's MAX_PLATFORM_OUTPUT (4000) so a collapsed message
# always fits one platform message and never needs the "full output saved
# to <path>" footer (which would require a successful audit save).
COLLAPSE_TOTAL_CHARS = 2800

# How the budget is split between the text before and after the degenerate
# span.  The prefix usually holds the model's preamble; the suffix usually
# holds the actual report the model did manage to write at the end.
COLLAPSE_PREFIX_CHARS = 1200
COLLAPSE_SUFFIX_CHARS = 1600

# Default hard cap for a single cron delivery, in characters.  Configurable
# via ``cron.max_delivery_chars`` in config.yaml.
DEFAULT_MAX_DELIVERY_CHARS = 20_000

# Footer appended when the delivery cap truncates content, pointing at the
# audit copy saved by the delivery layer.
TRUNCATION_FOOTER_TEMPLATE = (
    "\n\n... [truncated at {limit} chars; full output saved to {path}]"
)

# --- Tokenizer --------------------------------------------------------------

_CJK_RE = re.compile(
    r"["
    r"㐀-䶿"       # CJK Unified Ideographs Extension A
    r"一-鿿"       # CJK Unified Ideographs
    r"豈-﫿"       # CJK Compatibility Ideographs
    r"\U00020000-\U0002ffff"  # CJK Unified Ideographs Extension B+
    r"]"
)

_TOKEN_RE = re.compile(
    r"(?:"
    r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ffff]"
    r"|[A-Za-z0-9_]+"
    r")"
)


def tokenize(text: str) -> list[str]:
    """Split ``text`` into language-agnostic repetition units.

    CJK ideographs become one token each (no whitespace boundaries exist);
    Latin / digit runs are lowercased whole so ``Now`` and ``NOW`` collide.
    Punctuation and whitespace are dropped -- a looping model varies its
    punctuation far more than its words, and punctuation noise would dilute
    the k-gram features below.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group(0)
        if len(piece) == 1 and _CJK_RE.match(piece):
            tokens.append(piece)
        else:
            tokens.append(piece.lower())
    return tokens


# --- Features ---------------------------------------------------------------


def periodic_run(
    tokens: list[str],
    max_period: int = MAX_PERIOD,
) -> tuple[int, int, int]:
    """Longest exactly-periodic stretch of the token stream.

    A stretch is periodic with period ``p`` when ``tokens[i] == tokens[i+p]``
    holds for every ``i`` in the stretch -- i.e. the model is repeating a
    fixed block of ``p`` tokens over and over.

    Returns ``(run_length, period, token_start)`` where ``run_length`` counts
    matching positions and ``token_start`` is the token index where the run
    begins (``-1`` when no run was found).
    """
    n = len(tokens)
    best_len = 0
    best_period = 0
    best_start = -1
    for p in range(1, max_period + 1):
        if n - p <= 0:
            break
        cur = 0
        cur_at = 0
        for i in range(n - p):
            if tokens[i] == tokens[i + p]:
                if cur == 0:
                    cur_at = i
                cur += 1
                if cur > best_len:
                    best_len = cur
                    best_period = p
                    best_start = cur_at
            else:
                cur = 0
    return best_len, best_period, best_start


def periodic_copies(run_len: int, period: int) -> int:
    """Number of complete back-to-back copies of one ``period``-token block.

    ``periodic_run`` counts *matching positions*, so it grows with the block
    length: a five-token line repeated 40 times scores 195 matches, a
    two-token phrase repeated 1500 times scores 2998.  Comparing the two on
    the raw match count is meaningless; comparing them on whole copies is
    not -- 40 copies of a status line is still a report, 1500 copies of
    "NOW FIN" is noise.

    Returns 0 when no period was found.
    """
    if period <= 0 or run_len <= 0:
        return 0
    return (run_len + period) // period


def _periodic_run_is_degenerate(run_len: int, period: int) -> bool:
    """True when a periodic run repeats one short block too many times.

    Both gates must hold: the absolute match floor (a short run of one
    repeated token is not degeneracy) and the copy floor (the real
    discriminator -- see :data:`MIN_PERIODIC_COPIES`).
    """
    if period <= 0 or run_len < MIN_PERIODIC_RUN:
        return False
    if (run_len + period) < MIN_PERIODIC_SPAN:
        return False
    return periodic_copies(run_len, period) >= MIN_PERIODIC_COPIES


def window_dup_stats(
    tokens: list[str],
    gram_size: int = WINDOW_GRAM_SIZE,
    window: int = WINDOW_GRAMS,
    stride: int = WINDOW_STRIDE,
    threshold: float = WINDOW_DUP_RATIO,
) -> dict[str, float]:
    """Duplicate-gram statistics across sliding windows of ``window`` grams.

    Returns a dict with:

    * ``max_ratio`` -- the highest duplicate-gram ratio over all windows.
    * ``flagged_windows`` -- how many windows exceeded ``threshold``.
    * ``first_flagged`` -- token index where the first flagged window starts
      (``-1`` when none was flagged).
    * ``last_flagged_end`` -- token index just past the last flagged window
      (``-1`` when none was flagged).
    """
    grams = [
        tuple(tokens[i:i + gram_size])
        for i in range(len(tokens) - gram_size + 1)
    ]
    if not grams:
        return {
            "max_ratio": 0.0,
            "flagged_windows": 0,
            "first_flagged": -1,
            "last_flagged_end": -1,
        }
    if len(grams) < window:
        counter = Counter(grams)
        dup = sum(c - 1 for c in counter.values() if c > 1)
        ratio = dup / len(grams)
        flagged = 1 if ratio >= threshold else 0
        return {
            "max_ratio": ratio,
            "flagged_windows": flagged,
            "first_flagged": 0 if flagged else -1,
            "last_flagged_end": len(tokens) if flagged else -1,
        }

    max_ratio = 0.0
    flagged_windows = 0
    first_flagged = -1
    last_flagged_end = -1
    for start in range(0, len(grams) - window + 1, stride):
        counter = Counter(grams[start:start + window])
        dup = sum(c - 1 for c in counter.values() if c > 1)
        ratio = dup / window
        max_ratio = max(max_ratio, ratio)
        if ratio >= threshold:
            flagged_windows += 1
            if first_flagged < 0:
                first_flagged = start
            last_flagged_end = start + window
    return {
        "max_ratio": max_ratio,
        "flagged_windows": flagged_windows,
        "first_flagged": first_flagged,
        "last_flagged_end": last_flagged_end,
    }


# --- Detection --------------------------------------------------------------


def degeneracy_score(text: str) -> Optional[dict[str, float]]:
    """Return the measured repetition features for ``text``, or None if too short.

    Exposed for logging and tests -- callers wanting a yes/no answer should
    use :func:`is_degenerate`.
    """
    if not text or len(text) < MIN_CHARS_FOR_DETECTION:
        return None
    tokens = tokenize(text)
    if len(tokens) < MIN_TOKENS_FOR_DETECTION:
        return None
    run_len, period, run_at = periodic_run(tokens)
    window = window_dup_stats(tokens)
    return {
        "tokens": len(tokens),
        "chars": len(text),
        "periodic_run": run_len,
        "periodic_period": period,
        "periodic_copies": periodic_copies(run_len, period),
        "periodic_at": run_at,
        "window_dup_ratio": window["max_ratio"],
        "window_flagged": window["flagged_windows"],
        "window_first_flagged": window["first_flagged"],
        "window_last_flagged_end": window["last_flagged_end"],
    }


def is_degenerate(text: str) -> bool:
    """True when ``text`` shows runaway repetition of a phrase or small vocab.

    Short responses are never degenerate.  Either feature tripping is enough:
    a single-phrase loop (large ``periodic_run``) and a small-vocabulary
    cycle (many flagged windows) are both pathologies, and the real
    incidents each exhibited one of them.
    """
    if not text or len(text) < MIN_CHARS_FOR_DETECTION:
        return False
    tokens = tokenize(text)
    if len(tokens) < MIN_TOKENS_FOR_DETECTION:
        return False
    return _is_degenerate_tokens(tokens)


def _is_degenerate_tokens(tokens: list[str]) -> bool:
    """Detection on an already-tokenized stream."""
    run_len, period, _run_at = periodic_run(tokens)
    if _periodic_run_is_degenerate(run_len, period):
        return True
    window = window_dup_stats(tokens)
    if (
        window["max_ratio"] >= WINDOW_DUP_RATIO
        and window["flagged_windows"] >= MIN_FLAGGED_WINDOWS
    ):
        return True
    return False


def degenerate_span(text: str) -> Optional[tuple[int, int]]:
    """Token span ``[start, end)`` of the degenerate region, or None.

    The two features localize the region at very different resolutions:

    * ``periodic_run`` reports the **exact** token index where the exactly
      periodic stretch begins -- a real onset.
    * the window feature reports the start of the first *sliding window* that
      crossed the duplicate-ratio threshold -- a **coarse lower bound**, up to
      ``WINDOW_GRAMS`` (512) tokens early, because a window that merely
      *contains* the repetitive tail already trips the threshold.

    Taking the union of the two starts (the old behaviour) therefore always
    yielded the coarse window bound whenever both features fired, and the
    collapse swallowed up to 512 tokens of legitimate preamble: the readable
    summary a model wrote *before* it started looping was deleted along with
    the loop.

    So the start comes from the most precise feature available -- the exact
    periodic onset when the periodic feature fired, the coarse window bound
    otherwise.  The retained prefix is then checked against the detector
    itself: when the two features disagree and the text between the coarse
    bound and the exact onset is itself repetitive, the coarse bound is used
    instead.  That keeps the invariant that matters -- *a collapse never keeps
    degenerate content* -- while preserving the preamble whenever the
    preamble is genuinely clean.

    The end stays the union of the two ends.  The window envelope errs late
    (it can end a stride past the repetitive tail); dropping a little extra is
    safe, keeping garbage is not.
    """
    if not text or len(text) < MIN_CHARS_FOR_DETECTION:
        return None
    tokens = tokenize(text)
    if len(tokens) < MIN_TOKENS_FOR_DETECTION:
        return None

    ends: list[int] = []
    periodic_on: Optional[int] = None
    window_on: Optional[int] = None

    run_len, period, run_at = periodic_run(tokens)
    if _periodic_run_is_degenerate(run_len, period):
        periodic_on = run_at
        ends.append(run_at + run_len + period)

    window = window_dup_stats(tokens)
    if (
        window["max_ratio"] >= WINDOW_DUP_RATIO
        and window["flagged_windows"] >= MIN_FLAGGED_WINDOWS
    ):
        window_on = int(window["first_flagged"])
        ends.append(int(window["last_flagged_end"]))

    if periodic_on is None and window_on is None:
        return None

    if periodic_on is not None:
        start = periodic_on
    elif window_on is not None:
        start = window_on
    else:  # unreachable: guarded above
        return None
    if periodic_on is not None and window_on is not None and periodic_on > window_on:
        # The exact onset is later than the coarse window bound: the text
        # between them may be repetitive too.  Only keep the exact onset when
        # the prefix it would preserve is itself clean.
        if _prefix_is_degenerate(tokens, periodic_on):
            start = window_on
    return start, max(ends)


def _prefix_is_degenerate(tokens: list[str], start: int) -> bool:
    """True when the tokens before ``start`` are themselves degenerate.

    Uses the already-tokenized stream (no second tokenization pass) and the
    same feature thresholds as :func:`is_degenerate`.
    """
    prefix = tokens[:start]
    if len(prefix) < MIN_TOKENS_FOR_DETECTION:
        return False
    return _is_degenerate_tokens(prefix)


# --- Collapse ---------------------------------------------------------------


def collapse_degenerate(
    text: str,
    keep_prefix_chars: int = COLLAPSE_PREFIX_CHARS,
    keep_suffix_chars: int = COLLAPSE_SUFFIX_CHARS,
) -> str:
    """Return the readable parts of ``text`` with the degenerate span dropped.

    A degenerate response typically looks like::

        <preamble / thinking>  <thousands of repeated tokens>  <final report>

    so the readable content sits on *both* sides of the repetition.  This
    keeps a bounded prefix and a bounded suffix, drops everything in
    between, and inserts an explicit marker so a reader can tell a collapse
    from an abrupt ending.

    Returns the original text unchanged when it is not degenerate, and an
    empty string when nothing outside the degenerate span was salvageable -- the
    delivery lane treats an empty collapse as a suppression case.

    Deliberately NOT an excerpt of the original: when repetition covers the whole response
    there is no readable text to keep, and an excerpt of a pure loop ("NOW FIN. NOW FIN.
    ...") *is* the runaway repetition, only shorter.  Bounded-and-marked is still verbatim
    model pathology on the wire, which is the one thing this guard exists to prevent.  The
    caller decides what the reader gets instead (``gateway.delivery.guard_degenerate_output``
    sends a marked suppression notice; the cron lane preserves an honest ``no_agent`` log --
    see ``cron/scheduler_degenerate.py``).

    Idempotent: a payload that already carries :data:`COLLAPSE_MARKER` has been through this
    collapse once already, so it is returned unchanged.  The cron scheduler collapses a run's
    final response *before* composing its delivery, and both delivery lanes then re-run the
    guard on that already-collapsed text; re-collapsing it would mangle the result -- the
    marker is Chinese prose and a preserved excerpt still ends with a repetitive tail, so a
    second pass re-detected that tail and shrank a 2,826-char delivery to 125.
    """
    if not text:
        return ""
    if COLLAPSE_MARKER in text:
        # Already collapsed upstream: the text around the marker is the preserved report, so a
        # second collapse only destroys it.  Being marked is our own receipt, never a loop.
        return text
    span = degenerate_span(text)
    if span is None:
        return text

    start_token, end_token = span
    tokens = tokenize(text)
    start_char = _token_char_offset(text, start_token, tokens)
    end_char = _token_char_offset(text, min(end_token, len(tokens)), tokens)

    prefix = text[:start_char].rstrip()
    suffix = text[end_char:].lstrip()

    if len(prefix) > keep_prefix_chars:
        prefix = prefix[:keep_prefix_chars].rstrip()
    if len(suffix) > keep_suffix_chars:
        suffix = suffix[-keep_suffix_chars:].lstrip()

    if not prefix and not suffix:
        # Repetition covers the entire response: nothing readable survives outside the
        # degenerate span.  Return the empty collapse and let the caller choose the copy --
        # an excerpt of a pure loop would put the loop back on the wire (see docstring).
        return ""

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(COLLAPSE_MARKER)
    if suffix:
        parts.append(suffix)
    return "\n".join(parts)


def _token_char_offset(text: str, token_index: int, tokens: Optional[list[str]] = None) -> int:
    """Character offset of the ``token_index``-th token in ``text``.

    Walks the same regex the tokenizer uses so the two can never disagree on
    where a token begins.
    """
    if token_index <= 0:
        return 0
    if tokens is not None and token_index >= len(tokens):
        return len(text)
    for seen, match in enumerate(_TOKEN_RE.finditer(text)):
        if seen == token_index:
            return match.start()
    return len(text)


# --- Delivery truncation ----------------------------------------------------


def truncate_for_delivery(
    text: str,
    limit: int = DEFAULT_MAX_DELIVERY_CHARS,
    saved_path: Optional[str] = None,
) -> str:
    """Cap delivery size, keeping a pointer to the full audit copy.

    Independent of any adapter's native chunking: ``splits_long_messages``
    adapters previously received the entire payload and fanned a 213 KB
    degenerate response into a chat one message at a time.  This runs
    *before* that branch so no adapter ever receives an unbounded payload.

    ``saved_path`` (the on-disk audit copy written by the delivery layer) is
    embedded in the footer when provided; without it a bare truncation notice
    is appended so the reader knows the message was cut.
    """
    if text is None:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    if saved_path:
        footer = TRUNCATION_FOOTER_TEMPLATE.format(limit=limit, path=saved_path)
    else:
        footer = f"\n\n... [truncated at {limit} chars]"
    visible = max(0, limit - len(footer))
    return text[:visible].rstrip() + footer
