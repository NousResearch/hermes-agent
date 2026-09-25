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

_COMPACT_MARKER_TEMPLATE = (
    _COMPRESSION_MARKER_PREFIX
    + " {omitted:,} of {total:,} chars omitted; NOT original content. Do not copy.⟫"
)


def elide_text(text: str, head_chars: int, tail_chars: int = 0, *, max_chars: int | None = None) -> str:
    """Retain head/tail around a counted, non-original marker within an optional cap."""
    if len(text) <= head_chars + tail_chars:
        return text
    template = _COMPACT_MARKER_TEMPLATE if max_chars is not None and max_chars < 250 else _COMPRESSION_MARKER_TEMPLATE
    if max_chars is not None:
        retained = min(head_chars + tail_chars, max_chars)
        while retained:
            marker = template.format(omitted=len(text) - retained, total=len(text))
            available = max_chars - len(marker)
            if retained <= available:
                break
            retained = max(0, available)
        head_chars = min(head_chars, retained)
        tail_chars = retained - head_chars
    marker = template.format(omitted=len(text) - head_chars - tail_chars, total=len(text))
    return text[:head_chars] + marker + (text[-tail_chars:] if tail_chars else "")

# A minted marker (prefix + rendered counts through the first sentence). The prefix alone
# does not match, so source/docs that mention the constant can still be edited.
_COMPRESSION_MARKER_RE = re.compile(
    re.escape(_COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".")
    .replace(re.escape("{omitted:,}"), r"\d[\d,]*")
    .replace(re.escape("{total:,}"), r"\d[\d,]*")
    + "|"
    + re.escape(_COMPACT_MARKER_TEMPLATE.split(";", 1)[0] + ";")
    .replace(re.escape("{omitted:,}"), r"\d[\d,]*")
    .replace(re.escape("{total:,}"), r"\d[\d,]*")
)
