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


def elide_text(text: str, head_chars: int, tail_chars: int = 0) -> str:
    """Elide ``text`` to ``head_chars`` (+ optional ``tail_chars``) with the non-imitable marker.

    Central helper for #121572: every renderer that truncates model-visible text must call
    this instead of open-coding a bare truncation marker, which models imitate
    into new tool calls (#83714). Counts make each instance unique so copies go stale.
    """
    if _COMPRESSION_MARKER_PREFIX in text:
        return text
    total = len(text)
    head = text[:head_chars]
    omitted = total - head_chars - tail_chars
    if omitted <= 0:
        return text
    marker = _COMPRESSION_MARKER_TEMPLATE.format(omitted=omitted, total=total)
    if tail_chars > 0:
        return head + marker + text[total - tail_chars :]
    return head + marker
