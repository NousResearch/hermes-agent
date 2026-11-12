"""Assistant reasoning stored once (#125273): the codec between a message's ``reasoning`` /
``reasoning_content`` pair and its two columns, shared by the row writer, every decoder and every check
that compares a stored row with a serialized message or with another row.

Reasoning-content providers, and every turn ``build_assistant_message`` promotes, hand back the same
text as both fields. The row keeps it in ``reasoning_content``, the replay carrier
(``_replayable_reasoning_text`` prefers it, the must-echo pad sends it verbatim), and stores
``reasoning`` as NULL. NULL rather than a marker string: an older build that opens the store reads
it as no ``reasoning``, so its displays and exports fall back to ``reasoning_content``, its replay
sends the same text, and the dict it rebuilds still equals the row, so it inserts nothing. A marker
would reach those displays, exports and ``<think>`` blocks as text, and SQLite's ``quote()``
(``.dump``, ``.recover``, ``iterdump``) cuts a NUL-led marker down to ``''``. The cost: a row that
only ever held a non-blank ``reasoning_content`` (an assistant merge carried it onto a turn without
reasoning) reads back with ``reasoning`` equal to it, the text that turn replays anyway. Blank pads
are never folded.
"""

from __future__ import annotations

from typing import Any


def _carries_text(role: Any, reasoning_content: Any) -> bool:
    return role == "assistant" and isinstance(reasoning_content, str) and bool(reasoning_content.strip())


def stored_reasoning(role: Any, reasoning: Any, reasoning_content: Any) -> Any:
    """The ``reasoning`` column value to write: NULL when it repeats a non-blank ``reasoning_content``."""
    return None if reasoning == reasoning_content and _carries_text(role, reasoning_content) else reasoning


def restored_reasoning(role: Any, reasoning: Any, reasoning_content: Any) -> Any:
    """The read-side inverse of :func:`stored_reasoning`: a NULL ``reasoning`` beside a non-blank
    ``reasoning_content`` is that text."""
    return reasoning_content if reasoning is None and _carries_text(role, reasoning_content) else reasoning


def row_as_written(row: Any) -> Any:
    """A stored ``messages`` row with its ``reasoning`` column in the shape this build writes. Rows written before
    the change hold the text in both columns, and this build's copies of them (a compaction carry, a rewrite) hold
    it once; every check that compares a stored row reads it through this, so the two are one version: a re-handed
    or re-flushed copy is neither inserted again nor rewritten, and an edit is never taken for a foreign write."""
    if row["reasoning"] is not None and stored_reasoning(row["role"], row["reasoning"], row["reasoning_content"]) is None:
        return {**dict(row), "reasoning": None}
    return row
