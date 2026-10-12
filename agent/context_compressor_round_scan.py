"""Transcript scans for the pending tool round — the trailing tool results no assistant row has
answered yet. The compressor spares that round from demotion; the scans live here so the
compressor facade keeps shrinking."""

from __future__ import annotations

from typing import Any


def _last_index_with_role(messages: list[dict[str, Any]], role: str) -> int:
    """Index of the newest dict message with ``role``, or -1."""
    return max((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == role), default=-1)


def _pending_tool_round(messages: list[dict[str, Any]]) -> range:
    """Indices of the newest tool round no assistant row has answered yet; empty when the round was answered or
    the transcript ends elsewhere. Rows landing after the round do not answer it — a mid-turn /steer is delivered
    after the newest tool result before the next API call (two can land in one iteration), and a background-process
    notice or an interrupted-loop user message lands as a user row — so every trailing non-assistant row is
    skipped before the round is scanned."""
    end = len(messages)
    while end and messages[end - 1].get("role") not in ("assistant", "tool"):
        end -= 1
    start = end
    while start > 0 and messages[start - 1].get("role") == "tool":
        start -= 1
    return range(start, end)
