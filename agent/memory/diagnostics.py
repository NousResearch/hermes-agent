"""Diagnostics tools for the memory context relevance framework.

Provides ``memory_relevance_debug`` and ``memory_relevance_stats`` tools
for inspecting per-turn scoring and aggregate session statistics.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory ring buffer of per-turn relevance snapshots (max 50).
_TURN_LOG: List[Dict[str, Any]] = []
_MAX_LOG_SIZE = 50


def record_turn(
    turn_number: int,
    intent: Any,
    candidates_total: int,
    candidates_after_gating: int,
    injected_count: int,
    injected_chars: int,
    budget: int,
    top_scores: List[float],
    provider_counts: Dict[str, int],
) -> None:
    """Record a turn's relevance metrics for diagnostics."""
    global _TURN_LOG
    entry = {
        "turn": turn_number,
        "task_type": getattr(intent, "task_type", None),
        "task_confidence": round(getattr(intent, "task_confidence", 0.0), 2),
        "candidates": candidates_total,
        "after_gating": candidates_after_gating,
        "injected": injected_count,
        "injected_chars": injected_chars,
        "budget": budget,
        "top_scores": [round(s, 2) for s in top_scores[:5]],
        "providers": dict(provider_counts),
    }
    _TURN_LOG.append(entry)
    if len(_TURN_LOG) > _MAX_LOG_SIZE:
        _TURN_LOG.pop(0)


def clear_log() -> None:
    """Clear the diagnostics log."""
    _TURN_LOG.clear()


def get_turn_log() -> List[Dict[str, Any]]:
    """Return a copy of the turn log."""
    return list(_TURN_LOG)


def format_turn_debug(turns: int = 5) -> str:
    """Format a human-readable debug block for the last N turns."""
    recent = _TURN_LOG[-turns:] if _TURN_LOG else []
    if not recent:
        return "No relevance-scored turns yet."

    lines = []
    for entry in reversed(recent):
        t = entry["turn"]
        tt = entry.get("task_type", "?") or "?"
        tc = entry.get("task_confidence", 0.0)
        lines.append(f"")
        lines.append(f"Turn #{t}:")
        lines.append(f"  Intent: task_type={tt}, confidence={tc}")
        lines.append(
            f"  Candidates: {entry.get('candidates', 0)}  →  "
            f"After gating: {entry.get('after_gating', 0)}  →  "
            f"Injected: {entry.get('injected', 0)} ({entry.get('injected_chars', 0)} chars)"
        )
        if entry.get("top_scores"):
            lines.append(f"  Top scores: {entry['top_scores']}")
        if entry.get("providers"):
            prov = ", ".join(f"{k}: {v}" for k, v in entry.get("providers", {}).items())
            lines.append(f"  Providers: {prov}")
        lines.append(f"  Budget: {entry.get('budget', '?')} chars  Used: {entry.get('injected_chars', 0)} chars")

    return "\n".join(lines)


def format_session_stats() -> str:
    """Format aggregate session statistics."""
    if not _TURN_LOG:
        return "No relevance data for this session."

    total_turns = len(_TURN_LOG)
    total_candidates = sum(e.get("candidates", 0) for e in _TURN_LOG)
    total_injected = sum(e.get("injected", 0) for e in _TURN_LOG)
    total_chars = sum(e.get("injected_chars", 0) for e in _TURN_LOG)
    avg_score = sum(
        s for e in _TURN_LOG for s in e.get("top_scores", [])
    ) / max(sum(len(e.get("top_scores", [])) for e in _TURN_LOG), 1)

    # Aggregate provider counts
    provider_totals: Dict[str, int] = {}
    for e in _TURN_LOG:
        for p, c in e.get("providers", {}).items():
            provider_totals[p] = provider_totals.get(p, 0) + c

    # Score distribution
    all_scores = [s for e in _TURN_LOG for s in e.get("top_scores", [])]
    distribution: Dict[str, int] = {"0.8-1.0": 0, "0.6-0.8": 0, "0.4-0.6": 0, "0.15-0.4": 0}
    for s in all_scores:
        if s >= 0.8:
            distribution["0.8-1.0"] += 1
        elif s >= 0.6:
            distribution["0.6-0.8"] += 1
        elif s >= 0.4:
            distribution["0.4-0.6"] += 1
        else:
            distribution["0.15-0.4"] += 1

    lines = [
        "Memory Relevance — Session Statistics",
        "─" * 40,
        f"Total turns:         {total_turns}",
        f"Avg candidates/turn: {total_candidates / max(total_turns, 1):.1f}",
        f"Avg injected/turn:   {total_injected / max(total_turns, 1):.1f}",
        f"Avg score (injected): {avg_score:.2f}",
        f"Total chars injected: {total_chars:,} (avg {total_chars // max(total_turns, 1):,}/turn)",
        "",
        "Provider contributions:",
    ]
    for p, c in sorted(provider_totals.items(), key=lambda x: -x[1]):
        pct = c / max(total_injected, 1) * 100
        lines.append(f"  {p}: {c} entries ({pct:.0f}%)")

    lines.append("")
    lines.append("Score distribution (injected):")
    for band, count in sorted(distribution.items()):
        bar = "█" * count
        lines.append(f"  {band}: {count} {bar}")

    return "\n".join(lines)


# Tool handler functions (registered alongside memory tool)


def memory_relevance_debug(
    turns: int = 5,
    **kwargs,
) -> str:
    """Return a per-turn scoring breakdown for the last N turns.

    Args:
        turns: Number of recent turns to show (default 5).
    """
    return format_turn_debug(turns=turns)


def memory_relevance_stats(**kwargs) -> str:
    """Return aggregate relevance metrics across all turns in this session."""
    return format_session_stats()


# Tool schemas

RELEVANCE_DEBUG_SCHEMA = {
    "name": "memory_relevance_debug",
    "description": (
        "Return a per-turn scoring breakdown for the last N turns. Shows intent "
        "extraction results, candidate counts through each pipeline stage, score "
        "breakdowns, and budget usage. Call this to diagnose why a memory entry "
        "was or wasn't injected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "turns": {
                "type": "integer",
                "description": "Number of recent turns to show (default 5).",
                "default": 5,
            },
        },
    },
}

RELEVANCE_STATS_SCHEMA = {
    "name": "memory_relevance_stats",
    "description": (
        "Return aggregate memory relevance metrics across all turns in this "
        "session: average candidates per turn, average score, provider "
        "contributions, score distribution. Useful for understanding overall "
        "memory injection quality."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
