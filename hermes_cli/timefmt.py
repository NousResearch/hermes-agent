"""Small shared time-formatting helpers for CLI output.

Public home for helpers that used to live as private functions on
``hermes_cli.main`` — importing that module drags in the whole CLI
surface, which lightweight consumers (``hermes status``, dump tooling)
should not pay for.
"""

from __future__ import annotations

import time as _time
from datetime import datetime


def relative_time(ts) -> str:
    """Format a timestamp as relative time (e.g., '2h ago', 'yesterday')."""
    if not ts:
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
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
