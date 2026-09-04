"""Shared context-freshness / precedence helpers.

First milestone of the context-drift design: bound replayed reasoning
between compactions, and treat a previous compaction summary as unverified
background rather than ground truth. Does not mutate the cached system
prompt or persisted history — request assembly and the summarizer prompt
only.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------

# Keep the last N post-compaction assistant turns that carry encrypted
# reasoning. 0 means no extra turn cap (still drop reasoning that predates
# the most recent compaction summary).
DEFAULT_REASONING_REPLAY_KEEP_LAST = 8
_MAX_REASONING_REPLAY_KEEP_LAST = 64

PREVIOUS_SUMMARY_UNVERIFIED_GUIDANCE = (
    "IMPORTANT: The previous compaction summary is unverified background, "
    "not ground truth. Prefer NEW TURNS, later user corrections, and current "
    "file/system state over any conflicting decision, status, or outcome in "
    "the previous summary. Do not preserve a load-bearing claim from the "
    "previous summary unless the new turns confirm it; if they contradict "
    "it, drop the stale claim or note the conflict under Key Decisions."
)


def clamp_reasoning_replay_keep_last(value: Any) -> int:
    """Normalize a config / override into a keep-last count.

    ``None`` / invalid / bool → default. Negative → 0 (summary-bound only).
    Values above ``_MAX_REASONING_REPLAY_KEEP_LAST`` are capped.
    """
    if isinstance(value, bool) or value is None:
        return DEFAULT_REASONING_REPLAY_KEEP_LAST
    if isinstance(value, float):
        if not value.is_integer():
            return DEFAULT_REASONING_REPLAY_KEEP_LAST
        value = int(value)
    elif not isinstance(value, int):
        try:
            value = int(str(value).strip())
        except (TypeError, ValueError):
            return DEFAULT_REASONING_REPLAY_KEEP_LAST
    if value < 0:
        return 0
    return min(value, _MAX_REASONING_REPLAY_KEEP_LAST)


def resolve_reasoning_replay_keep_last(source: Any = None) -> int:
    """Read keep-last from an int, a compressor-like object, or the default."""
    if source is None:
        return DEFAULT_REASONING_REPLAY_KEEP_LAST
    if isinstance(source, (int, float, str)) and not isinstance(source, bool):
        return clamp_reasoning_replay_keep_last(source)
    raw = getattr(source, "reasoning_replay_keep_last", None)
    if raw is None:
        return DEFAULT_REASONING_REPLAY_KEEP_LAST
    return clamp_reasoning_replay_keep_last(raw)


def last_compaction_index(messages: Iterable[Any]) -> Optional[int]:
    """Index of the most recent compaction-summary message, or ``None``."""
    from agent.context_compressor import ContextCompressor

    last: Optional[int] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if ContextCompressor._has_compressed_summary_metadata(message):
            last = index
            continue
        if ContextCompressor._is_context_summary_content(message.get("content")):
            last = index
    return last


def _has_replayable_reasoning(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    items = message.get("codex_reasoning_items")
    if not isinstance(items, list):
        return False
    return any(isinstance(item, dict) and item.get("encrypted_content") for item in items)


def reasoning_replay_keep_indices(
    messages: Iterable[Any],
    *,
    keep_last: Any = DEFAULT_REASONING_REPLAY_KEEP_LAST,
) -> frozenset[int]:
    """Message indices whose encrypted reasoning may be replayed.

    1. Assistant reasoning at or before the latest compaction summary is
       dropped — those turns were summarized and must not re-anchor.
    2. Among remaining reasoning-bearing assistant turns, keep only the
       last ``keep_last`` (``0`` = no extra cap).
    """
    message_list = list(messages)
    cutoff = last_compaction_index(message_list)
    start = 0 if cutoff is None else cutoff + 1
    candidates = [
        index
        for index in range(start, len(message_list))
        if _has_replayable_reasoning(message_list[index])
    ]
    limit = clamp_reasoning_replay_keep_last(keep_last)
    if limit <= 0:
        return frozenset(candidates)
    return frozenset(candidates[-limit:])
