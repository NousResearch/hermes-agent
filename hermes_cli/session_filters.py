"""Shared time/filter parsing for `hermes sessions prune` / `archive`.

Turns user-friendly CLI values into the epoch bounds and filter kwargs
consumed by ``SessionDB.prune_sessions`` / ``archive_sessions`` /
``list_prune_candidates``.

Two value shapes are accepted anywhere a point in time is expected:

* Durations (relative to now): ``5h``, ``30m``, ``2d``, ``1w`` — and, for
  backward compatibility with the original ``--older-than N`` flag, a bare
  integer which means **days**.
* Absolute timestamps: ``2026-07-05``, ``2026-07-05 14:30``,
  ``2026-07-05T14:30:00`` (any ISO-8601 form ``datetime.fromisoformat``
  understands; naive values are interpreted in local time).
"""

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
    """Parse ``5h`` / ``30m`` / ``2d`` / ``1w`` / ``90`` (bare = days) into
    seconds. Returns None when the value doesn't look like a duration."""
    s = str(value).strip().lower()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        # Bare number = days (backward compatible with --older-than 90)
        return float(s) * 86400
    m = _DURATION_RE.match(s)
    if not m:
        return None
    return float(m.group(1)) * _UNIT_SECONDS[m.group(2)[0]]


def parse_point_in_time(value: str, flag: str) -> float:
    """Parse a CLI time value into an epoch timestamp.

    Durations are interpreted as "that long ago" (``5h`` → now − 5 hours).
    Absolute ISO timestamps are returned as-is (naive = local time).
    Raises ``ValueError`` with a user-facing message on unparseable input.
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
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.astimezone(timezone.utc).timestamp()


def format_epoch(ts: Optional[float]) -> str:
    """Render an epoch timestamp as a short local-time string."""
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def build_prune_filters(args: Any) -> Dict[str, Any]:
    """Translate argparse Namespace flags into SessionDB filter kwargs.

    Understands: ``--older-than``, ``--newer-than``, ``--before``,
    ``--after``, ``--source``, ``--title``, ``--end-reason``, ``--cwd``,
    ``--min-messages``, ``--max-messages``, ``--archived``/``--no-archived``.

    ``--before``/``--older-than`` both set the upper bound (started_before);
    ``--after``/``--newer-than`` both set the lower bound (started_after).
    When both a duration flag and an absolute flag target the same bound,
    the tighter (more restrictive) bound wins.

    Raises ``ValueError`` on unparseable values or an empty/inverted window.
    """
    started_before: Optional[float] = None
    started_after: Optional[float] = None

    def _tighter(current: Optional[float], new: float, upper: bool) -> float:
        if current is None:
            return new
        return min(current, new) if upper else max(current, new)

    older_than = getattr(args, "older_than", None)
    if older_than is not None:
        started_before = _tighter(
            started_before, parse_point_in_time(older_than, "--older-than"), True
        )
    before = getattr(args, "before", None)
    if before is not None:
        started_before = _tighter(
            started_before, parse_point_in_time(before, "--before"), True
        )
    newer_than = getattr(args, "newer_than", None)
    if newer_than is not None:
        started_after = _tighter(
            started_after, parse_point_in_time(newer_than, "--newer-than"), False
        )
    after = getattr(args, "after", None)
    if after is not None:
        started_after = _tighter(
            started_after, parse_point_in_time(after, "--after"), False
        )

    if (
        started_before is not None
        and started_after is not None
        and started_after >= started_before
    ):
        raise ValueError(
            "Empty time window: the --after/--newer-than bound "
            f"({format_epoch(started_after)}) is not earlier than the "
            f"--before/--older-than bound ({format_epoch(started_before)})."
        )

    filters: Dict[str, Any] = {
        # older_than_days=None: the epoch bounds above are the whole story.
        # Without this, prune_sessions' default 90-day cutoff would silently
        # cap an --after/--newer-than-only window.
        "older_than_days": None,
        "started_before": started_before,
        "started_after": started_after,
        "source": getattr(args, "source", None),
        "title_like": getattr(args, "title", None),
        "end_reason": getattr(args, "end_reason", None),
        "cwd_prefix": getattr(args, "cwd", None),
        "min_messages": getattr(args, "min_messages", None),
        "max_messages": getattr(args, "max_messages", None),
        "model_like": getattr(args, "model", None),
        "provider": getattr(args, "provider", None),
        "user_id": getattr(args, "user", None),
        "chat_id": getattr(args, "chat_id", None),
        "chat_type": getattr(args, "chat_type", None),
        "branch_like": getattr(args, "branch", None),
        "min_tokens": getattr(args, "min_tokens", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "min_cost": getattr(args, "min_cost", None),
        "max_cost": getattr(args, "max_cost", None),
        "min_tool_calls": getattr(args, "min_tool_calls", None),
        "max_tool_calls": getattr(args, "max_tool_calls", None),
    }
    return filters


def describe_filters(filters: Dict[str, Any]) -> str:
    """Human-readable summary of active filters for confirmation prompts."""
    parts = []
    if filters.get("started_before") is not None:
        parts.append(f"started before {format_epoch(filters['started_before'])}")
    if filters.get("started_after") is not None:
        parts.append(f"started after {format_epoch(filters['started_after'])}")
    if filters.get("source"):
        parts.append(f"source '{filters['source']}'")
    if filters.get("title_like"):
        parts.append(f"title contains '{filters['title_like']}'")
    if filters.get("end_reason"):
        parts.append(f"end reason '{filters['end_reason']}'")
    if filters.get("cwd_prefix"):
        parts.append(f"cwd under '{filters['cwd_prefix']}'")
    if filters.get("min_messages") is not None:
        parts.append(f">= {filters['min_messages']} messages")
    if filters.get("max_messages") is not None:
        parts.append(f"<= {filters['max_messages']} messages")
    if filters.get("model_like"):
        parts.append(f"model contains '{filters['model_like']}'")
    if filters.get("provider"):
        parts.append(f"provider '{filters['provider']}'")
    if filters.get("user_id"):
        parts.append(f"user '{filters['user_id']}'")
    if filters.get("chat_id"):
        parts.append(f"chat '{filters['chat_id']}'")
    if filters.get("chat_type"):
        parts.append(f"chat type '{filters['chat_type']}'")
    if filters.get("branch_like"):
        parts.append(f"git branch contains '{filters['branch_like']}'")
    if filters.get("min_tokens") is not None:
        parts.append(f">= {filters['min_tokens']} tokens")
    if filters.get("max_tokens") is not None:
        parts.append(f"<= {filters['max_tokens']} tokens")
    if filters.get("min_cost") is not None:
        parts.append(f">= ${filters['min_cost']}")
    if filters.get("max_cost") is not None:
        parts.append(f"<= ${filters['max_cost']}")
    if filters.get("min_tool_calls") is not None:
        parts.append(f">= {filters['min_tool_calls']} tool calls")
    if filters.get("max_tool_calls") is not None:
        parts.append(f"<= {filters['max_tool_calls']} tool calls")
    return ", ".join(parts) if parts else "no filters (all ended sessions)"
