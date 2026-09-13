"""Policies that decide which recent turns compaction keeps verbatim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Demotion and anchor relaxation use the same working-set boundary.
LEAN_TAIL_KEEP_TOOL_ROUNDS = 6


def has_long_inflight_tool_loop(
    messages: Sequence[Mapping[str, Any]],
    inflight_task: Mapping[str, Any] | None,
    *,
    keep_tool_rounds: int = LEAN_TAIL_KEEP_TOOL_ROUNDS,
) -> bool:
    """Return whether an unfinished real user turn exceeds the recent-tool window.

    The caller supplies the real in-flight task selected by the compressor's
    provenance-aware predicate. Identity lookup ensures a copied or stale row
    cannot relax anchors for an unrelated transcript.
    """
    if inflight_task is None:
        return False
    task_index = next(
        (index for index, message in enumerate(messages) if message is inflight_task),
        -1,
    )
    if task_index < 0:
        return False
    tool_rounds = sum(
        1
        for message in messages[task_index + 1 :]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    return tool_rounds > keep_tool_rounds
