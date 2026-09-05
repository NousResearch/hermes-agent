"""Small shared time-formatting helpers for CLI output."""

from __future__ import annotations

import time as _time
from datetime import datetime
import math


def relative_time(ts) -> str:
    """Format a timestamp as relative time (e.g., '2h ago', 'yesterday').

    Non-numeric values (corrupt TEXT rows under SQLite dynamic typing)
    render as "?" instead of raising TypeError (#102399); numeric
    strings coerce to float.
    """
    if not ts:
        return "?"
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(ts):
        return "?"
    delta = _time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    if delta < 604800:
        return f"{int(delta / 86400)}d ago"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return "?"
