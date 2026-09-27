"""The model-visible marker the context compressor leaves in pruned tool-call arguments.

Dependency-free leaf shared by the producer (``agent.context_compressor``) and the
dispatch-boundary detector (``agent.tool_dispatch_helpers``), so the matcher is derived
from the template instead of re-typing its wording.
"""

from __future__ import annotations

import re

# #83714 — this text lands inside the model's OWN replayed tool call, so it must not read like
# something the model would write itself: the bare "...[truncated]" it replaced was imitated into
# new calls and written to disk. Non-prose delimiters, an explicit "not original content"
# disclaimer, and per-instance counts keep a copied marker visibly wrong; the counts also make a
# verbatim copy stale, which is why the marker must never be re-applied (see ``_shrink``).
_COMPRESSION_MARKER_PREFIX = "⟪HERMES-CONTEXT-COMPRESSION:"
_COMPRESSION_MARKER_TEMPLATE = (
    _COMPRESSION_MARKER_PREFIX
    + " {omitted:,} of {total:,} chars omitted here by Hermes's context compressor. "
    "This is NOT part of the original tool call and must never be reproduced in new "
    "output — always write full, untruncated content.⟫"
)

# A minted marker (prefix + rendered counts through the first sentence). The prefix alone
# does not match, so source/docs that mention the constant can still be edited.
_COMPRESSION_MARKER_RE = re.compile(
    re.escape(_COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".")
    .replace(re.escape("{omitted:,}"), r"\d[\d,]*")
    .replace(re.escape("{total:,}"), r"\d[\d,]*")
)

# #121548 — every OTHER model-visible elision (turn text, summaries, skill bodies, diagnostics)
# mints the args marker's first sentence only (so the dispatch-boundary guard's
# _COMPRESSION_MARKER_RE catches a copy from any renderer) and fits small caps like clarify's 199.
_ELISION_MARKER_TEMPLATE = _COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".⟫"


def _elision_marker(omitted: int, total: int) -> str:
    """Render the non-imitable elision marker with per-instance byte counts."""
    return _ELISION_MARKER_TEMPLATE.format(omitted=omitted, total=total)


# Widest marker for any text under ~1 TB: callers with a small leftover budget skip the item
# when the budget cannot hold the marker plus content, instead of emitting a marker-only line.
ELISION_MARKER_MAX_LEN = len(_elision_marker(omitted=10**12 - 1, total=10**12 - 1))


def elide(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars, marking the elision with ``_elision_marker``.

    Every model-visible renderer must truncate through here (#121548). ``text`` is
    returned unchanged when it already fits. Otherwise the result is ``head + marker``
    with accurate omitted/total counts and never exceeds ``limit`` (sized against the
    widest rendering the counts can take), unless ``limit`` cannot hold the marker at
    all, in which case the marker alone is returned.
    """
    if len(text) <= limit:
        return text
    total = len(text)
    head_len = limit - len(_elision_marker(omitted=total, total=total))
    if head_len <= 0:
        return _elision_marker(omitted=total, total=total)
    kept = text[:head_len].rstrip()
    return kept + _elision_marker(omitted=total - len(kept), total=total)


def elide_middle(text: str, head: int, tail: int) -> str:
    """Keep ``head`` chars from the start and ``tail`` from the end, eliding the middle."""
    if len(text) <= head + tail:
        return text
    marker = _elision_marker(omitted=len(text) - head - tail, total=len(text))
    return text[:head] + marker + text[len(text) - tail:]
