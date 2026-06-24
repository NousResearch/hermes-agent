"""
Read-only snapshot builder for /control-status.

Produces a Discord-friendly text snapshot of all active session-orchestration
tasks: state, age since created_at / last_output_ts, and highlights the oldest
WAITING_USER task so the user never misses an outstanding query.

Architecture note
-----------------
This module is **read-only** — it calls ``registry.list()`` and never writes
the registry.  The single-writer discipline is enforced by the cron watcher
(see registry.py); this module is a pure observer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Age helpers
# ---------------------------------------------------------------------------

def _age_seconds(row: Dict[str, Any], now: float) -> float:
    """Return the effective age in seconds for a row.

    Prefers ``last_output_ts`` (float epoch) when available and non-zero;
    falls back to ``created_at`` (ISO-8601 UTC string from SQLite).
    Returns 0 when neither is parseable.
    """
    last_ts = row.get("last_output_ts")
    if last_ts:
        try:
            ts = float(last_ts)
            if ts > 0:
                return max(0.0, now - ts)
        except (ValueError, TypeError):
            pass

    created = row.get("created_at")
    if created:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # Naive UTC string from SQLite — attach UTC explicitly.
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, now - dt.timestamp())
        except (ValueError, TypeError):
            pass

    return 0.0


def _format_age(seconds: float) -> str:
    """Return a human-readable age string e.g. '3m 12s', '1h 5m', '45s'."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(
    rows: List[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> str:
    """Build a Discord-friendly snapshot string from registry rows.

    Parameters
    ----------
    rows:
        List of row dicts as returned by ``SessionOrchestrationRegistry.list()``.
        May be empty (produces a "no active tasks" message).
    now:
        Epoch timestamp to use as "current time" for age calculations.
        Defaults to ``time.time()``.  Overridable in tests for determinism.

    Returns
    -------
    A plain-text, Discord-safe string.  No markdown that Discord doesn't render
    (e.g. no tables); uses bullet points and inline code spans for task IDs.
    """
    if now is None:
        now = time.time()

    if not rows:
        return "**Session Orchestration Status**\n\nNo active tasks."

    # Compute ages and find oldest WAITING_USER
    enriched: List[Dict[str, Any]] = []
    oldest_waiting: Optional[Dict[str, Any]] = None
    oldest_waiting_age: float = -1.0

    for row in rows:
        age = _age_seconds(row, now)
        entry = {**row, "_age_s": age}
        enriched.append(entry)

        state = str(row.get("state", "")).upper()
        if state == "WAITING_USER":
            if age > oldest_waiting_age:
                oldest_waiting_age = age
                oldest_waiting = entry

    # Sort: WAITING_USER first, then by age descending (longest-running at top)
    def _sort_key(e: Dict[str, Any]) -> tuple:
        state = str(e.get("state", "")).upper()
        waiting_first = 0 if state == "WAITING_USER" else 1
        return (waiting_first, -e["_age_s"])

    enriched.sort(key=_sort_key)

    lines: List[str] = [
        f"**Session Orchestration Status** — {len(enriched)} active task(s)",
        "",
    ]

    state_counts: Dict[str, int] = {}
    for entry in enriched:
        state = str(entry.get("state", "UNKNOWN")).upper()
        state_counts[state] = state_counts.get(state, 0) + 1

    # Summary counts line
    summary_parts = [f"{v} {k}" for k, v in sorted(state_counts.items())]
    lines.append("  " + " · ".join(summary_parts))
    lines.append("")

    # Per-task lines
    for entry in enriched:
        task_id = entry.get("task_id", "?")
        agent = entry.get("agent", "?")
        state = str(entry.get("state", "UNKNOWN")).upper()
        age_str = _format_age(entry["_age_s"])
        project = entry.get("project") or entry.get("workdir") or ""
        project_label = f" `{project}`" if project else ""

        # State emoji
        state_icon = {
            "RUNNING": "🟢",
            "WAITING_USER": "🔔",
            "PAUSED_HANDOFF": "⏸️",
            "STALLED": "⚠️",
            "DONE": "✅",
            "ERROR": "❌",
        }.get(state, "❓")

        line = f"{state_icon} `{task_id}` [{agent}]{project_label} — **{state}** · age {age_str}"
        lines.append(line)

    # Highlight oldest WAITING_USER
    if oldest_waiting is not None:
        ow_id = oldest_waiting.get("task_id", "?")
        ow_age = _format_age(oldest_waiting["_age_s"])
        lines.append("")
        lines.append(f"**Oldest WAITING_USER:** `{ow_id}` (waiting {ow_age})")

    return "\n".join(lines)
