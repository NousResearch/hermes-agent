"""Closing-fence marker separation for the Telegram adapter (``plugins/platforms/telegram/adapter.py``).

A marker appended to a closing fence line - a ``(N/M)`` chunk marker from ``truncate_message()``
or a ``(xN)`` tool-progress repeat counter (#138148) - invalidates the fence for anchored parsers,
so the block would render as escaped literal backticks. Moving the marker onto its own line keeps
the fence a fence.
"""

from __future__ import annotations

import re

_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r'(?m)^``` (?P<indicator>(?:\\)?\((?:\d+/\d+|×\d+)(?:\\)?\))$')


def _separate_chunk_indicator_from_fence(text: str) -> str:
    """Move a marker appended to a closing fence onto its own line."""
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r'```\n\g<indicator>', text)
