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


# Every other model-visible elision uses the same counted marker, while omitting the
# tool-call-specific second sentence so it still fits small renderer caps.
_ELISION_MARKER_TEMPLATE = (
    _COMPRESSION_MARKER_PREFIX
    + " {omitted:,} of {total:,} chars omitted here by Hermes's context compressor.⟫"
)


def _elision_marker(omitted: int, total: int) -> str:
    return _ELISION_MARKER_TEMPLATE.format(omitted=omitted, total=total)


def elide(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars; return the marker alone if it cannot fit."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if len(text) <= limit:
        return text
    total = len(text)
    head_len = limit - len(_elision_marker(omitted=total, total=total))
    if head_len <= 0:
        return _elision_marker(omitted=total, total=total)
    kept = text[:head_len].rstrip()
    return kept + _elision_marker(omitted=total - len(kept), total=total)


def elide_middle(text: str, head: int, tail: int) -> str:
    """Keep the head and tail while marking the omitted middle as non-original."""
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    if len(text) <= head + tail:
        return text
    marker = _elision_marker(omitted=len(text) - head - tail, total=len(text))
    return text[:head] + marker + (text[-tail:] if tail else "")


# A minted marker (prefix + rendered counts through the first sentence). The prefix alone
# does not match, so source/docs that mention the constant can still be edited.
_COMPRESSION_MARKER_RE = re.compile(
    re.escape(_COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".")
    .replace(re.escape("{omitted:,}"), r"\d[\d,]*")
    .replace(re.escape("{total:,}"), r"\d[\d,]*")
)
