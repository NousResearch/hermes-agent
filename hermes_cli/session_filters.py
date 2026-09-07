"""Shared time/filter parsing for `hermes sessions prune` / `archive`."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*"
    r"(s|sec|secs|second|seconds|"
    r"m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|"
    r"d|day|days|"
    r"w|wk|wks|week|weeks)$"
)

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration_seconds(value: str) -> Optional[float]:
    """Parse ``5h`` / ``30m`` / ``2d`` / ``1w`` / ``90`` (bare = days, backward compatible with
    ``--older-than 90``) into seconds. Returns None when the value doesn't look like a duration."""
    s = str(value).strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s) * 86400
    m = _DURATION_RE.match(s)
    return None if not m else float(m.group(1)) * _UNIT_SECONDS[m.group(2)[0]]


def parse_point_in_time(value: str, flag: str) -> float:
    """Parse a CLI time value into an epoch timestamp.

    Durations mean "that long ago" (``5h`` = now minus 5 hours); ISO timestamps are taken as-is
    (naive = local time). Raises ``ValueError`` with a user-facing message on bad input.
    """
    s = str(value).strip()
    dur = parse_duration_seconds(s)
    if dur is not None:
        return time.time() - dur
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(
            f"Invalid value for {flag}: '{value}'. Use a duration like '5h', "
            f"'30m', '2d', '1w', a bare number of days, or an ISO timestamp "
            f"like '2026-07-05' or '2026-07-05 14:30'."
        ) from None
    return dt.timestamp() if dt.tzinfo is None else dt.astimezone(timezone.utc).timestamp()


def format_epoch(ts: Optional[float]) -> str:
    """Render an epoch timestamp as a short local-time string."""
    return "-" if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# (filter key, argparse attr, CLI flag, description template) for the four epoch bounds.
_TIME_BOUNDS = (
    ("last_active_before", "older_than", "--older-than", "last active before {v}"),
    ("last_active_after", "newer_than", "--newer-than", "last active after {v}"),
    ("started_before", "before", "--before", "started before {v}"),
    ("started_after", "after", "--after", "started after {v}"),
)
# (lower key, upper key, window label, lower flag, upper flag); checked in this order.
_WINDOWS = (
    ("started_after", "started_before", "start-time", "--after", "--before"),
    ("last_active_after", "last_active_before", "activity", "--newer-than", "--older-than"),
)
# (filter key, argparse attr, description template) for pass-through filters, in describe order.
# Numeric (min_/max_) filters are described when not None, text filters only when truthy.
_ARG_FILTERS = (
    ("source", "source", "source '{v}'"),
    ("title_like", "title", "title contains '{v}'"),
    ("end_reason", "end_reason", "end reason '{v}'"),
    ("cwd_prefix", "cwd", "cwd under '{v}'"),
    ("min_messages", "min_messages", ">= {v} messages"),
    ("max_messages", "max_messages", "<= {v} messages"),
    ("model_like", "model", "model contains '{v}'"),
    ("provider", "provider", "provider '{v}'"),
    ("user_id", "user", "user '{v}'"),
    ("chat_id", "chat_id", "chat '{v}'"),
    ("chat_type", "chat_type", "chat type '{v}'"),
    ("branch_like", "branch", "git branch contains '{v}'"),
    ("min_tokens", "min_tokens", ">= {v} tokens"),
    ("max_tokens", "max_tokens", "<= {v} tokens"),
    ("min_cost", "min_cost", ">= ${v}"),
    ("max_cost", "max_cost", "<= ${v}"),
    ("min_tool_calls", "min_tool_calls", ">= {v} tool calls"),
    ("max_tool_calls", "max_tool_calls", "<= {v} tool calls"),
)


def build_prune_filters(args: Any) -> Dict[str, Any]:
    """Translate argparse Namespace flags into SessionDB filter kwargs.

    ``--older-than`` / ``--newer-than`` bound last activity (latest message timestamp, falling back
    to ``started_at`` for empty sessions); ``--before`` / ``--after`` bound session start time.
    """
    bounds: Dict[str, Optional[float]] = {
        key: None if (raw := getattr(args, attr, None)) is None else parse_point_in_time(raw, flag)
        for key, attr, flag, _ in _TIME_BOUNDS
    }
    for lo, hi, label, lo_flag, hi_flag in _WINDOWS:
        if bounds[hi] is not None and bounds[lo] is not None and bounds[lo] >= bounds[hi]:
            raise ValueError(
                f"Empty {label} window: the {lo_flag} bound "
                f"({format_epoch(bounds[lo])}) is not earlier than the "
                f"{hi_flag} bound ({format_epoch(bounds[hi])})."
            )

    # older_than_days=None: the epoch bounds are the whole story; otherwise prune_sessions' default
    # 90-day cutoff would silently cap an --after/--newer-than-only window.
    return {"older_than_days": None, **bounds, **{key: getattr(args, attr, None) for key, attr, _ in _ARG_FILTERS}}


def describe_filters(filters: Dict[str, Any]) -> str:
    """Human-readable summary of active filters for confirmation prompts."""
    parts = [
        template.format(v=format_epoch(filters[key])) for key, _, _, template in _TIME_BOUNDS
        if filters.get(key) is not None
    ] + [
        template.format(v=filters[key]) for key, _, template in _ARG_FILTERS
        if ((filters.get(key) is not None) if key.startswith(("min_", "max_")) else bool(filters.get(key)))
    ]
    return ", ".join(parts) if parts else "no filters (all ended sessions)"
