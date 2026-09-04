"""Lightweight conversation text helpers."""

from __future__ import annotations

from typing import List


def _join_truncated_parts(parts: List[str]) -> str:
    """Join continuation fragments, adding a newline where two would glue together (#78577)."""
    joined = ""
    for part in parts:
        if joined and not joined[-1].isspace() and part and not part[0].isspace():
            joined += "\n"
        joined += part
    return joined
