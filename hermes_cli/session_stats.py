"""Post-turn session statistics — a one-line summary after each agent response.

Prints a compact summary of turn duration and tool call breakdown.

Example output::

    12.3s · 5 tool calls — terminal(3) write_file(2)
"""

from __future__ import annotations

from typing import Dict


def format_turn_summary(
    duration_seconds: float,
    tool_counts: Dict[str, int],
) -> str:
    """Format a compact turn summary line.

    Returns a string like ``12.3s · 5 tool calls — terminal(3) write_file(2)``.
    """
    # Duration
    if duration_seconds < 60:
        dur = f"{duration_seconds:.1f}s"
    else:
        mins = int(duration_seconds // 60)
        secs = duration_seconds % 60
        dur = f"{mins}m{secs:.0f}s"

    total = sum(tool_counts.values())
    if total == 0:
        return f"  {dur} · no tool calls"

    sorted_tools = sorted(tool_counts.items(), key=lambda x: -x[1])
    breakdown = " ".join(f"{name}({count})" for name, count in sorted_tools)

    return f"  {dur} · {total} tool calls — {breakdown}"
