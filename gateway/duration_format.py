"""Tiny shared duration formatter for gateway progress surfaces.

Extracted from gateway/run.py so pure helper modules (e.g.
gateway/subagent_roster.py) can format elapsed times without importing the
18k-line run.py god-file. run.py re-exports this as ``_format_duration`` for
backward compatibility with its existing call sites.
"""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Render seconds as ``M:SS`` (or ``H:MM:SS`` past an hour). Clamps < 0."""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    if total < 0:
        total = 0
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
