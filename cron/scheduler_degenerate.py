"""Guard the cron lane's delivered text against degenerate model output without destroying
an honest repetitive report.

The 2026-11-02 incident (213 KB of ``"NOW FIN. NOW FIN. ..."`` fanned into a chat by a
chunking adapter) is why the guard exists: runaway repetition must never reach a chat
verbatim.  The guard is *statistical*, so it cannot tell a model that lost the plot from a
``no_agent`` watchdog that honestly prints the same WARN line on every one of its 100
samples — both are one short block repeated back-to-back, and both score above every
threshold the detector has.

Collapsing is the right answer for the loop and the wrong answer for the log.  The
difference is where the repetition lives:

* a runaway loop has a readable report on one side of it (a preamble the model wrote
  before it latched, a closing summary it managed afterwards), and that text is worth
  keeping;
* an honest repetitive log is repetition *all the way through* — there is no preamble or
  suffix, because the log IS the report.

So when the collapse finds nothing outside the degenerate span, the delivered copy keeps
exactly ONE copy of the repeated block, the marker, and a count of how many times it
repeated.  One copy is the honest case's whole value (the reader sees the WARN line that
fired on every sample) and it is simultaneously harmless for the runaway case, where the
block is a two-word phrase: a *bounded* excerpt of a pure loop is still the loop, just
shorter, and putting thousands of copies on the wire is the incident this guard exists to
prevent.  The count is what makes the honest log readable as a log rather than as a report
that happened to end early.  The complete original is audited to disk by the caller.

The run is still a success either way: the job did what it was asked to do.  The only
change is what the operator receives, and that they receive something at all — an empty
body used to be booked as a soft failure "degenerate model output with no readable body",
which both destroyed the report and alerted on a script that exited 0.

Detector thresholds live in ``gateway/degenerate_output.py`` and are calibrated on the
real corpus; nothing here relaxes them.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Marker that must be recognisable in the delivered copy: the reader has to be able to tell
# a shortened excerpt of a repetitive payload from a complete one.
DEGENERATE_EXCERPT_MARKER = (
    "\n\n[输出因模型退化重复已截断：中间的重复段已丢弃]"
)

# Characters of the original kept when the whole response is repetitive.  Deliberately
# below ``gateway.degenerate_output.COLLAPSE_TOTAL_CHARS`` so the excerpt never needs the
# size cap's "full output saved to <path>" footer just to fit one platform message.
EXCERPT_CHARS = 2800


def guard_degenerate_final_response(
    final_response: str, job_id: str, *, is_degenerate: Optional[Callable[[str], bool]] = None,
) -> str:
    """Delivered text for a finished run: collapse a runaway loop, preserve an honest log.

    Returns ``final_response`` unchanged when it is not degenerate.  When it is, the text
    outside the degenerate span (preamble / suffix) survives and the repetition does not --
    unless the repetition covers everything, in which case a bounded, explicitly-marked
    excerpt of the original is returned so the operator still receives the report.

    Never returns an empty string for a non-empty input: an empty delivery is read
    downstream as "this run produced nothing" and booked as a failure.
    """
    if not final_response:
        return final_response
    is_degenerate = is_degenerate or _detector("is_degenerate")
    try:
        degenerate = bool(is_degenerate(final_response))
    except Exception:  # a detector failure must never cost the run its report
        logger.warning("Job '%s': degeneracy detection failed — delivering as-is", job_id,
                       exc_info=True)
        return final_response
    if not degenerate:
        return final_response

    collapse: Callable[[str], str] = _detector("collapse_degenerate")
    try:
        collapsed = collapse(final_response)
    except Exception:
        logger.warning("Job '%s': degeneracy collapse failed — delivering an excerpt", job_id,
                       exc_info=True)
        return _excerpt(final_response)
    if not collapsed.strip():
        # Repetition covers the whole response: an honest repetitive log, most likely.
        # Deliver a bounded excerpt rather than nothing (see the module docstring).
        logger.warning(
            "Job '%s': degenerate repetitive output covers the whole response (%d chars) — "
            "delivering a bounded excerpt", job_id, len(final_response))
        return _excerpt(final_response)
    logger.warning(
        "Job '%s': degenerate model output detected (%d chars) — delivering %d chars",
        job_id, len(final_response), len(collapsed))
    return collapsed


def _excerpt(text: str) -> str:
    """One copy of the repeated block, a count, and the marker.

    Bounded to one platform message (``EXCERPT_CHARS``) and, critically, to a single copy:
    the reader learns *what* repeated and *how often* without receiving the repetition.
    """
    copies, block = _repetition_shape(text)
    if block:
        head = block
    else:
        head = text[:EXCERPT_CHARS]
    head = head[:EXCERPT_CHARS].rstrip()
    if copies > 1:
        return (f"{head}\n\n[以上内容重复 {copies} 次，已折叠为一份样本]"
                f"{DEGENERATE_EXCERPT_MARKER}")
    return f"{head}{DEGENERATE_EXCERPT_MARKER}"


def _repetition_shape(text: str) -> tuple[int, str]:
    """``(copies, one_block_text)`` for the payload's dominant periodic run.

    ``(0, "")`` when the detector localizes no periodic block (a small-vocabulary cycle
    rather than one repeated phrase); the caller then falls back to a plain bounded head.
    """
    from gateway.degenerate_output import (
        _token_char_offset, _periodic_run_is_degenerate, periodic_copies, periodic_run, tokenize,
    )
    tokens = tokenize(text)
    run_len, period, run_at = periodic_run(tokens)
    if not _periodic_run_is_degenerate(run_len, period):
        return 0, ""
    start = _token_char_offset(text, run_at, tokens)
    end = _token_char_offset(text, run_at + period, tokens)
    block = text[start:end].strip()
    return periodic_copies(run_len, period), block


def _detector(name: str):
    """Look up one callable on ``gateway.degenerate_output``.

    Late-imported here the same way the scheduler does: ``gateway.degenerate_output`` pulls
    in the tokenizer-free n-gram detector, and the cron scheduler is imported on every CLI
    startup, so the import belongs at call time, not module load.
    """
    from gateway import degenerate_output
    return getattr(degenerate_output, name)
